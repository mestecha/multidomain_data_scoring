"""stage 2 llm judge evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from loguru import logger

from scripts.config import DOMAINS
from scripts.models import (
    BatchEntry,
    ContrastiveDirection,
    DomainName,
    PairLabel,
    Stage1Entry,
    Stage2Variant,
    VariantType,
    EvalResult,
)
from scripts.stage_2.prompts import build_eval_prompt


DEFAULT_MARGIN_THRESHOLD = 0.20


def build_eval_entries(
    variants: list[Stage2Variant],
    originals: dict[str, Stage1Entry],
    config_map: dict | None = None,
) -> tuple[list[BatchEntry], list[dict]]:
    """build batch entries that score variant continuations only.

    Stage 1 scores serve as ground truth for originals, so only
    variants need fresh scoring. All Stage 1 domain scores are
    stored in the manifest for richer downstream signal.
    """
    domains = config_map or DOMAINS
    batch_entries: list[BatchEntry] = []
    manifest_items: list[dict] = []
    skipped = 0

    for variant in variants:
        original = originals.get(variant.candidate_id)
        if original is None:
            logger.warning(
                "No original found for candidate_id={}", variant.candidate_id
            )
            skipped += 1
            continue

        config = domains[original.domain]

        # derive prefix from original using variant length
        turn_count = len(variant.variant_messages)
        all_msgs = original.messages

        if len(all_msgs) > turn_count:
            prefix_msgs = all_msgs[:-turn_count]
        else:
            prefix_msgs = all_msgs[:1]

        # extract all stage 1 domain scores for the original
        stage_1_scores = {
            d: original.scores[d]
            for d in config.prefixed_dim_names
            if d in original.scores and original.scores[d] is not None
        }

        prompt = build_eval_prompt(
            prefix_messages=prefix_msgs,
            variant_continuation=variant.variant_messages,
            config=config,
            domain_metadata=original.domain_metadata,
        )

        custom_id = f"s2v-{variant.variant_id}"
        entry = BatchEntry(custom_id=custom_id, prompt=prompt)
        batch_entries.append(entry)

        manifest_items.append(
            {
                "custom_id": custom_id,
                "variant_id": variant.variant_id,
                "candidate_id": variant.candidate_id,
                "domain": original.domain.value,
                "direction": variant.direction.value,
                "variant_type": variant.variant_type.value,
                "target_dimensions": list(variant.target_dimensions),
                "stage_1_scores": stage_1_scores,
            }
        )

    logger.info(
        "Built {} eval entries ({} skipped)",
        len(batch_entries),
        skipped,
    )
    return batch_entries, manifest_items


DEFAULT_STABILITY_THRESHOLD = 0.20

_GLOBAL_VARIANT_TYPES = {VariantType.GLOBAL_IMPROVE, VariantType.GLOBAL_DEGRADE}


def classify_variant(
    original_scores: dict[str, float],
    variant_scores: dict[str, float],
    direction: ContrastiveDirection,
    char_dims: list[str],
    margin_threshold: float,
    variant_type: VariantType,
    target_dims: list[str] | None = None,
    stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
) -> PairLabel:
    """classify a verified variant into one of 6 labels."""
    sign = 1 if direction == ContrastiveDirection.POSITIVE else -1
    is_global = variant_type in _GLOBAL_VARIANT_TYPES

    if is_global:
        orig_vals = [original_scores[d] for d in char_dims if d in original_scores]
        var_vals = [variant_scores[d] for d in char_dims if d in variant_scores]

        if not orig_vals or not var_vals:
            return PairLabel.REJECT

        # sign-normalize: positive = intended direction, negative = opposite
        delta = sign * (mean(var_vals) - mean(orig_vals))

        if delta > margin_threshold:
            return PairLabel.GLOBAL_PASS
        if delta < -margin_threshold:
            return PairLabel.GLOBAL_FLIP_PASS
        return PairLabel.REJECT

    # targeted variant
    if not target_dims:
        return PairLabel.REJECT

    t_orig = [original_scores[d] for d in target_dims if d in original_scores]
    t_var = [variant_scores[d] for d in target_dims if d in variant_scores]

    if not t_orig or not t_var:
        return PairLabel.REJECT

    t_delta = sign * (mean(t_var) - mean(t_orig))

    if t_delta < -margin_threshold:
        return PairLabel.TARGET_FLIP_PASS

    if t_delta <= margin_threshold:
        return PairLabel.REJECT

    # target moved correctly — classify non-target stability
    non_target = [d for d in char_dims if d not in target_dims]
    for d in non_target:
        if d in original_scores and d in variant_scores:
            if abs(variant_scores[d] - original_scores[d]) > stability_threshold:
                return PairLabel.TARGET_COARSE_PASS

    return PairLabel.TARGET_PASS


def parse_eval_results(
    batch_output_dir: Path,
    manifest_path: Path,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    config_map: dict | None = None,
) -> list[EvalResult]:
    """parse batch outputs, extract variant scores, classify against stage 1 scores."""
    domains = config_map or DOMAINS

    # load manifest
    manifest: dict[str, dict] = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            manifest[item["custom_id"]] = item

    results: list[EvalResult] = []
    parse_failures = 0
    label_counts: dict[str, int] = {label.value: 0 for label in PairLabel}

    for output_file in sorted(batch_output_dir.glob("*.jsonl")):
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                result = json.loads(line.strip())
                custom_id = result.get("custom_id", "")

                if custom_id not in manifest:
                    logger.warning(
                        "Unknown custom_id in eval output: {}", custom_id
                    )
                    parse_failures += 1
                    continue

                meta = manifest[custom_id]

                # extract variant scores from response
                try:
                    choices = result["response"]["body"]["choices"]
                    content = choices[0]["message"]["content"]
                    parsed = json.loads(content)
                    variant_scores = parsed["scores"]
                except (KeyError, json.JSONDecodeError, IndexError, TypeError) as e:
                    logger.warning(
                        "Failed to parse eval result for {}: {}", custom_id, e
                    )
                    parse_failures += 1
                    continue

                # stage 1 scores from manifest as ground truth for original
                original_scores = meta["stage_1_scores"]

                # look up domain config for characterizing dims
                domain = DomainName(meta["domain"])
                config = domains[domain]
                char_dims = config.characterizing_dims
                direction = ContrastiveDirection(meta["direction"])
                target_dims = meta.get("target_dimensions")
                variant_type = VariantType(meta["variant_type"])

                label = classify_variant(
                    original_scores=original_scores,
                    variant_scores=variant_scores,
                    direction=direction,
                    char_dims=char_dims,
                    margin_threshold=margin_threshold,
                    variant_type=variant_type,
                    target_dims=target_dims,
                )

                vr = EvalResult(
                    variant_id=meta["variant_id"],
                    original_scores=original_scores,
                    variant_scores=variant_scores,
                    label=label,
                )
                results.append(vr)
                label_counts[label.value] += 1

    total = len(results)
    usable = total - label_counts[PairLabel.REJECT.value]
    usable_rate = (usable / total * 100) if total > 0 else 0.0

    logger.info(
        "Eval: {}/{} usable ({:.1f}%), {} parse failures",
        usable,
        total,
        usable_rate,
        parse_failures,
    )
    for label_name, count in label_counts.items():
        logger.info("  {}: {}", label_name, count)

    return results
