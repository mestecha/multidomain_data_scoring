"""stage 2 variant generation via batch api."""

from __future__ import annotations

import json
import random
from pathlib import Path

from loguru import logger

from scripts.config import DOMAINS, TURN_DISTRIBUTION
from scripts.models import (
    BatchEntry,
    ContrastiveDirection,
    Message,
    Stage2Candidate,
    Stage2Variant,
    VariantType,
)
from scripts.stage_2.prompts import build_generation_prompt


_VARIANT_TYPE_TAGS: dict[VariantType, str] = {
    VariantType.GLOBAL_IMPROVE: "gimp",
    VariantType.GLOBAL_DEGRADE: "gdeg",
}


def _make_custom_id(candidate: Stage2Candidate) -> str:
    """build a unique custom_id encoding variant type and target dimension."""
    if candidate.variant_type == VariantType.DIMENSION_TARGETED:
        if not candidate.target_dimensions:
            raise ValueError(
                f"DIMENSION_TARGETED candidate {candidate.dialogue_id} "
                "has empty target_dimensions",
            )
        tag = f"dt-{candidate.target_dimensions[0]}"
    else:
        tag = _VARIANT_TYPE_TAGS.get(candidate.variant_type)
        if tag is None:
            raise ValueError(f"unhandled variant type: {candidate.variant_type}")
    return f"s2g-{candidate.dialogue_id}-{tag}"


def _sample_turn_count(distribution: dict[int, float] | None = None) -> int:
    dist = distribution or TURN_DISTRIBUTION
    counts = list(dist.keys())
    weights = list(dist.values())
    return random.choices(counts, weights=weights, k=1)[0]


def build_generation_entries(
    candidates: list[Stage2Candidate],
    config_map: dict | None = None,
) -> tuple[list[BatchEntry], list[dict]]:
    """build batch entries and manifest metadata for each candidate."""
    domains = config_map or DOMAINS
    batch_entries: list[BatchEntry] = []
    manifest_items: list[dict] = []

    skipped_short = 0

    for candidate in candidates:
        config = domains[candidate.domain]

        # need at least 2 messages (1 prefix + 1 continuation)
        max_continuation = len(candidate.messages) - 1
        if max_continuation < 1:
            skipped_short += 1
            continue

        turn_count = min(_sample_turn_count(), max_continuation)

        prompt = build_generation_prompt(
            candidate, turn_count, config,
            domain_metadata=candidate.domain_metadata,
        )

        custom_id = _make_custom_id(candidate)

        entry = BatchEntry(custom_id=custom_id, prompt=prompt)
        batch_entries.append(entry)

        manifest_entry: dict = {
            "custom_id": custom_id,
            "dialogue_id": candidate.dialogue_id,
            "domain": candidate.domain.value,
            "variant_type": candidate.variant_type.value,
            "direction": candidate.contrastive_direction.value,
            "turn_count": turn_count,
            "target_dimensions": candidate.target_dimensions,
        }
        if candidate.domain_metadata is not None:
            manifest_entry["domain_metadata"] = candidate.domain_metadata
        manifest_items.append(manifest_entry)

    if skipped_short:
        logger.warning("Skipped {} candidates with < 2 messages", skipped_short)
    logger.info("Built {} generation entries", len(batch_entries))
    return batch_entries, manifest_items


def write_shards(
    batch_entries: list[BatchEntry],
    shard_dir: Path,
    shard_size: int = 5000,
) -> list[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []

    for i in range(0, len(batch_entries), shard_size):
        shard_entries = batch_entries[i : i + shard_size]
        shard_num = i // shard_size
        shard_path = shard_dir / f"stage2_gen_shard_{shard_num:04d}.jsonl"

        with open(shard_path, "w", encoding="utf-8") as f:
            for entry in shard_entries:
                f.write(json.dumps(entry.to_api_dict(), ensure_ascii=False) + "\n")

        shard_paths.append(shard_path)
        logger.debug(
            "Wrote shard {} with {} entries", shard_path.name, len(shard_entries)
        )

    logger.info("Wrote {} shards to {}", len(shard_paths), shard_dir)
    return shard_paths


def write_manifest(manifest_items: list[dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in manifest_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(
        "Wrote manifest with {} items to {}", len(manifest_items), manifest_path
    )


def parse_generation_results(
    batch_output_dir: Path,
    manifest_path: Path,
) -> list[Stage2Variant]:
    """match batch outputs to manifest entries and extract variant messages."""
    # load manifest
    manifest: dict[str, dict] = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            manifest[item["custom_id"]] = item

    # parse batch outputs
    variants: list[Stage2Variant] = []
    failed = 0

    for output_file in sorted(batch_output_dir.glob("*.jsonl")):
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                result = json.loads(line.strip())
                custom_id = result.get("custom_id", "")

                if custom_id not in manifest:
                    logger.warning("Unknown custom_id in output: {}", custom_id)
                    failed += 1
                    continue

                meta = manifest[custom_id]

                # extract generated content from response
                try:
                    choices = result["response"]["body"]["choices"]
                    content = choices[0]["message"]["content"]
                    parsed = json.loads(content)
                    continuation = parsed.get("continuation", [])

                    variant_messages = []
                    for m in continuation:
                        msg_content = m["content"]
                        # coerce list content to string (malformed gpt response)
                        if isinstance(msg_content, list):
                            msg_content = " ".join(str(s) for s in msg_content)
                        # normalize role to lowercase
                        role = m["role"].lower()
                        if role not in ("user", "assistant"):
                            raise ValueError(f"invalid role: {role}")
                        variant_messages.append(
                            Message(role=role, content=msg_content)
                        )
                except (
                    KeyError, json.JSONDecodeError, IndexError,
                    TypeError, ValueError, Exception,
                ) as e:
                    logger.warning("Failed to parse output for {}: {}", custom_id, e)
                    failed += 1
                    continue

                if not variant_messages:
                    logger.warning("Empty continuation for {}", custom_id)
                    failed += 1
                    continue

                variant = Stage2Variant(
                    candidate_id=meta["dialogue_id"],
                    variant_id=f"var-{custom_id}",
                    variant_type=VariantType(meta["variant_type"]),
                    direction=ContrastiveDirection(meta["direction"]),
                    prefix_messages=[],  # populated during eval
                    variant_messages=variant_messages,
                    generation_model="gpt-5.1-batch",
                    target_dimensions=meta.get("target_dimensions", []),
                )
                variants.append(variant)

    logger.info(
        "Parsed {} variants from batch outputs ({} failed)",
        len(variants),
        failed,
    )
    return variants
