"""stage 2 contrastive pair construction for dpo training."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from loguru import logger

from scripts.config import DOMAINS
from scripts.models import (
    ContrastiveDirection,
    Difficulty,
    PairEvaluation,
    PairLabel,
    PairMetadata,
    Split,
    Stage1Entry,
    Stage2Pair,
    Stage2Variant,
    EvalResult,
)


_FLIP_LABELS = {PairLabel.GLOBAL_FLIP_PASS, PairLabel.TARGET_FLIP_PASS}


# ── Difficulty thresholds ─────────────────────────────────────────────────

EASY_THRESHOLD = 0.30
MEDIUM_THRESHOLD = 0.15


def _compute_margin(
    original_scores: dict[str, float],
    variant_scores: dict[str, float],
    char_dims: list[str],
) -> float:
    orig_vals = [original_scores[d] for d in char_dims if d in original_scores]
    var_vals = [variant_scores[d] for d in char_dims if d in variant_scores]

    if not orig_vals or not var_vals:
        return 0.0

    return abs(mean(var_vals) - mean(orig_vals))


def _classify_difficulty(margin: float) -> Difficulty:
    if margin >= EASY_THRESHOLD:
        return Difficulty.EASY
    elif margin >= MEDIUM_THRESHOLD:
        return Difficulty.MEDIUM
    else:
        return Difficulty.HARD


def _validate_score_ordering(
    chosen_scores: dict[str, float],
    rejected_scores: dict[str, float],
    char_dims: list[str],
) -> bool:
    chosen_vals = [chosen_scores[d] for d in char_dims if d in chosen_scores]
    rejected_vals = [rejected_scores[d] for d in char_dims if d in rejected_scores]

    if not chosen_vals or not rejected_vals:
        return False

    return mean(chosen_vals) > mean(rejected_vals)


def build_pairs(
    variants: list[Stage2Variant],
    results: list[EvalResult],
    originals: dict[str, Stage1Entry],
    config_map: dict | None = None,
) -> list[Stage2Pair]:
    """assign chosen/rejected based on direction for verified variants."""
    domains = config_map or DOMAINS

    # index results by variant_id
    result_map: dict[str, EvalResult] = {r.variant_id: r for r in results}

    pairs: list[Stage2Pair] = []
    skipped_no_result = 0
    skipped_rejected = 0
    skipped_no_original = 0
    skipped_bad_ordering = 0
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    label_counts: dict[str, int] = {label.value: 0 for label in PairLabel if label != PairLabel.REJECT}

    for variant in variants:
        vr = result_map.get(variant.variant_id)
        if vr is None:
            skipped_no_result += 1
            continue

        if vr.label == PairLabel.REJECT:
            skipped_rejected += 1
            continue

        original = originals.get(variant.candidate_id)
        if original is None:
            skipped_no_original += 1
            continue

        config = domains[original.domain]
        char_dims = config.characterizing_dims
        effective_dims = (
            variant.target_dimensions if variant.target_dimensions else char_dims
        )

        # compute margin and difficulty
        margin = _compute_margin(vr.original_scores, vr.variant_scores, effective_dims)
        difficulty = _classify_difficulty(margin)

        # determine effective direction (flipped labels reverse intended direction)
        if vr.label in _FLIP_LABELS:
            if variant.direction == ContrastiveDirection.POSITIVE:
                effective_direction = ContrastiveDirection.NEGATIVE
            else:
                effective_direction = ContrastiveDirection.POSITIVE
        else:
            effective_direction = variant.direction

        # determine chosen/rejected based on effective direction
        turn_count = len(variant.variant_messages)
        all_msgs = original.messages

        if len(all_msgs) > turn_count:
            prefix_msgs = all_msgs[:-turn_count]
            original_continuation = all_msgs[-turn_count:]
        else:
            prefix_msgs = all_msgs[:1]
            original_continuation = all_msgs[1:] if len(all_msgs) > 1 else all_msgs

        if effective_direction == ContrastiveDirection.POSITIVE:
            chosen = variant.variant_messages
            rejected = original_continuation
            chosen_scores = vr.variant_scores
            rejected_scores = vr.original_scores
        else:
            chosen = original_continuation
            rejected = variant.variant_messages
            chosen_scores = vr.original_scores
            rejected_scores = vr.variant_scores

        # validate score ordering
        if not _validate_score_ordering(chosen_scores, rejected_scores, effective_dims):
            skipped_bad_ordering += 1
            continue

        pair_id = f"S2D-{len(pairs) + 1:06d}"

        metadata = PairMetadata(
            domain=original.domain,
            split=Split.TRAIN,
            contrastive_direction=effective_direction,
            target_dimensions=list(effective_dims),
            generation_model=variant.generation_model,
            turn_count=turn_count,
            variant_type=variant.variant_type,
            difficulty=difficulty,
            label=vr.label,
        )

        evaluation = PairEvaluation(
            method="lm_judge",
            stage_1_scores=vr.original_scores,
            variant_scores=vr.variant_scores,
        )

        pair = Stage2Pair(
            pair_id=pair_id,
            source_dialogue_id=original.dialogue_id,
            metadata=metadata,
            evaluation=evaluation,
            messages=prefix_msgs,
            chosen=chosen,
            rejected=rejected,
        )
        pairs.append(pair)
        difficulty_counts[difficulty.value] += 1
        label_counts[vr.label.value] += 1

    logger.info(
        "Built {} pairs (skipped: {} no result, {} rejected, {} no original, {} bad ordering)",
        len(pairs),
        skipped_no_result,
        skipped_rejected,
        skipped_no_original,
        skipped_bad_ordering,
    )
    logger.info(
        "Difficulty distribution: easy={}, medium={}, hard={}",
        difficulty_counts["easy"],
        difficulty_counts["medium"],
        difficulty_counts["hard"],
    )
    for label_name, count in label_counts.items():
        if count > 0:
            logger.info("  {}: {}", label_name, count)

    return pairs


def write_pairs(pairs: list[Stage2Pair], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(pair.model_dump_json() + "\n")
    logger.info("Wrote {} pairs to {}", len(pairs), output_path)
