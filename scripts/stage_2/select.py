"""stage 2 candidate selection from stage 1 train data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from scripts.config import DOMAINS, get_domain
from scripts.models import (
    ContrastiveDirection,
    DomainName,
    Stage1Entry,
    Stage2Candidate,
    VariantType,
)


# ── Tier boundaries ───────────────────────────────────────────────────────

LOW_UPPER = 0.4
MEDIUM_UPPER = 0.7
MEDIUM_DIRECTION_THRESHOLD = 0.55


def load_stage1_entries(input_path: Path) -> list[Stage1Entry]:
    """load stage 1 entries from a jsonl file, handling legacy field names."""
    entries: list[Stage1Entry] = []
    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                # handle legacy field names
                if "source" in raw and "domain" not in raw:
                    raw["domain"] = raw.pop("source")
                if "stage_1_dialogue_id" in raw and "dialogue_id" not in raw:
                    raw["dialogue_id"] = raw.pop("stage_1_dialogue_id")
                # extract scores from flat dict if needed
                if "scores" not in raw:
                    domain_cfg = get_domain(raw["domain"])
                    scores = {}
                    for dim_name in domain_cfg.prefixed_dim_names:
                        if dim_name in raw:
                            scores[dim_name] = raw[dim_name]
                    raw["scores"] = scores
                entries.append(Stage1Entry(**raw))
            except Exception as e:
                logger.warning("Skipping line {}: {}", line_num, e)
    return entries


def _classify_candidates(
    entry: Stage1Entry,
    char_dims: list[str],
) -> list[Stage2Candidate]:
    """classify a single entry and build multiple candidates (1 global + N targeted)."""
    avg = entry.characterizing_avg(char_dims)
    if avg is None:
        return []

    char_scores = {
        d: entry.scores[d]
        for d in char_dims
        if d in entry.scores and entry.scores[d] is not None
    }

    if avg < LOW_UPPER:
        direction = ContrastiveDirection.POSITIVE
        global_variant = VariantType.GLOBAL_IMPROVE
    elif avg < MEDIUM_UPPER:
        if avg < MEDIUM_DIRECTION_THRESHOLD:
            direction = ContrastiveDirection.POSITIVE
        else:
            direction = ContrastiveDirection.NEGATIVE
        global_variant = (
            VariantType.GLOBAL_IMPROVE
            if direction == ContrastiveDirection.POSITIVE
            else VariantType.GLOBAL_DEGRADE
        )
    else:
        direction = ContrastiveDirection.NEGATIVE
        global_variant = VariantType.GLOBAL_DEGRADE

    base = dict(
        dialogue_id=entry.dialogue_id,
        domain=entry.domain,
        messages=entry.messages,
        characterizing_scores=char_scores,
        contrastive_direction=direction,
        domain_metadata=entry.domain_metadata,
    )

    candidates = [
        Stage2Candidate(
            **base,
            target_dimensions=char_dims,
            variant_type=global_variant,
        ),
    ]
    for dim in char_dims:
        candidates.append(
            Stage2Candidate(
                **base,
                target_dimensions=[dim],
                variant_type=VariantType.DIMENSION_TARGETED,
            ),
        )

    return candidates


def _classify_commonsense_candidates(
    entry: Stage1Entry,
    char_dims: list[str],
) -> list[Stage2Candidate]:
    """classify a commonsense entry and build 1 DIMENSION_TARGETED per char dim."""
    avg = entry.characterizing_avg(char_dims)
    if avg is None:
        return []

    char_scores = {
        d: entry.scores[d]
        for d in char_dims
        if d in entry.scores and entry.scores[d] is not None
    }

    if avg < LOW_UPPER:
        direction = ContrastiveDirection.POSITIVE
    elif avg < MEDIUM_UPPER:
        direction = (
            ContrastiveDirection.POSITIVE
            if avg < MEDIUM_DIRECTION_THRESHOLD
            else ContrastiveDirection.NEGATIVE
        )
    else:
        direction = ContrastiveDirection.NEGATIVE

    base = dict(
        dialogue_id=entry.dialogue_id,
        domain=entry.domain,
        messages=entry.messages,
        characterizing_scores=char_scores,
        contrastive_direction=direction,
        domain_metadata=entry.domain_metadata,
    )

    return [
        Stage2Candidate(
            **base,
            target_dimensions=[dim],
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        for dim in char_dims
    ]


def select_candidates(
    entries: list[Stage1Entry],
    domain: DomainName | None = None,
    max_samples: int | None = None,
) -> list[Stage2Candidate]:
    """stratify train entries into quality tiers and assign contrastive direction."""
    # filter to train split only
    train_entries = [e for e in entries if e.split is None or e.split.value == "train"]

    if domain is not None:
        train_entries = [e for e in train_entries if e.domain == domain]

    logger.info("Selecting from {} train entries", len(train_entries))

    candidates: list[Stage2Candidate] = []
    tier_counts = {"low": 0, "medium": 0, "high": 0}

    for entry in train_entries:
        domain_cfg = DOMAINS[entry.domain]
        char_dims = domain_cfg.characterizing_dims

        if entry.domain == DomainName.COMMONSENSE:
            new_candidates = _classify_commonsense_candidates(entry, char_dims)
        else:
            new_candidates = _classify_candidates(entry, char_dims)

        if not new_candidates:
            continue

        candidates.extend(new_candidates)

        # count tiers by dialogue, not by candidate
        avg = entry.characterizing_avg(char_dims)
        if avg < LOW_UPPER:
            tier_counts["low"] += 1
        elif avg < MEDIUM_UPPER:
            tier_counts["medium"] += 1
        else:
            tier_counts["high"] += 1

    logger.info(
        "Selected {} candidates from {} dialogues: low={}, medium={}, high={}",
        len(candidates),
        sum(tier_counts.values()),
        tier_counts["low"],
        tier_counts["medium"],
        tier_counts["high"],
    )

    if max_samples is not None and len(candidates) > max_samples:
        candidates = candidates[:max_samples]
        logger.info("Capped to {} candidates", max_samples)

    return candidates


def write_candidates(candidates: list[Stage2Candidate], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(c.model_dump_json() + "\n")
    logger.info("Wrote {} candidates to {}", len(candidates), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: Select candidates from Stage 1 train data",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/stage_1.jsonl",
        help="Path to Stage 1 JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/stage_2/candidates.jsonl",
        help="Path for output candidates JSONL",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=[d.value for d in DomainName],
        help="Filter to a single domain",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of candidates to select",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: {}", input_path)
        sys.exit(1)

    domain_filter = DomainName(args.domain) if args.domain else None

    entries = load_stage1_entries(input_path)
    logger.info("Loaded {} Stage 1 entries", len(entries))

    candidates = select_candidates(
        entries,
        domain=domain_filter,
        max_samples=args.max_samples,
    )

    write_candidates(candidates, Path(args.output))


if __name__ == "__main__":
    main()
