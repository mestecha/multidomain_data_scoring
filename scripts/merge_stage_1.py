#!/usr/bin/env python3
"""merge domain outputs into unified stage 1 jsonl format."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from scripts.config import CS_RELATION_TO_DIM, DOMAINS, all_prefixed_dimensions
from scripts.models import DomainName

SPEAKER_PATTERNS = [
    r"^Human [AB]:\s*",
    r"^Speaker [12]:\s*",
    r"^Person [AB]:\s*",
    r"^User:\s*",
    r"^Assistant:\s*",
    r"^[AB]:\s*",
    r"^SPK\d{2}-[A-Z]{3}\d{3}:\s*",
    r"^Target:\s*",
]


def parse_dialogue_to_messages(content: str) -> list[dict[str, str]]:
    """strip speaker prefixes and assign alternating user/assistant roles."""
    # Handle escaped newlines (e.g. multicultural data stores literal \n)
    content = content.replace("\\n", "\n")
    lines = content.strip().split("\n")
    messages = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        text = line
        for pattern in SPEAKER_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        if text:
            messages.append(text)

    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": text}
        for i, text in enumerate(messages)
    ]


def normalize_score(score: Any) -> float | None:
    if score is None:
        return None

    try:
        if isinstance(score, dict):
            score = score.get("score")

        if score is None:
            return None

        score_float = float(score)

        if 0.0 <= score_float <= 1.0:
            return round(score_float, 2)

        # Convert from 1-5 scale to 0-1
        normalized = (score_float - 1) / 4.0
        return round(max(0.0, min(1.0, normalized)), 2)
    except (ValueError, TypeError):
        return None


def load_domain_data(
    output_dir: Path,
    domain_name: str,
) -> list[dict[str, Any]]:
    config = DOMAINS[DomainName(domain_name)]
    items: list[dict[str, Any]] = []
    domain_dir = output_dir / domain_name

    if not domain_dir.exists():
        logger.warning("Domain directory not found: {}", domain_dir)
        return items

    pattern = f"{config.prefix}_*.jsonl"
    jsonl_files = list(domain_dir.glob(pattern))

    if not jsonl_files:
        logger.warning("No files found for {} with pattern {}", domain_name, pattern)
        return items

    logger.info("Processing {}: found {} file(s)", domain_name, len(jsonl_files))

    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    item["_domain"] = domain_name
                    items.append(item)
                except json.JSONDecodeError as e:
                    logger.error(
                        "Parse error at {}:{}: {}", jsonl_file.name, line_num, e
                    )

    logger.info("  Loaded {} items from {}", len(items), domain_name)
    return items


_MULTICULTURAL_METADATA_FIELDS = (
    "country_1",
    "country_2",
    "demographics_1",
    "demographics_2",
    "statement_original",
    "statement_cultural",
    "situation",
    "cultural_reasoning_1",
    "cultural_reasoning_2",
    "arousal_reasoning",
    "arousal_score",
    "social_norms_1",
    "social_norms_2",
    "prejudices_1",
    "prejudices_2",
)


def load_multicultural_metadata(
    data_dir: Path = Path("data/input/multicultural/countries"),
) -> dict[str, dict[str, str]]:
    """Build uid → metadata lookup from raw multicultural CSVs."""
    lookup: dict[str, dict[str, str]] = {}
    if not data_dir.exists():
        logger.warning("Multicultural data dir not found: {}", data_dir)
        return lookup

    for csv_file in sorted(data_dir.glob("*.csv")):
        with open(csv_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("uid", "")
                if not uid:
                    continue
                metadata = {
                    field: row.get(field, "") for field in _MULTICULTURAL_METADATA_FIELDS
                }
                lookup[uid] = metadata

    logger.info("Loaded multicultural metadata for {} dialogues", len(lookup))
    return lookup


def load_commonsense_gold(
    gold_path: Path = Path("data/input/commonsense/instruction/train_multitask_gold_12k.json"),
) -> dict[str, dict[str, str]]:
    """Build pid → gold relation metadata from multitask annotation file.

    Each entry has a 'gold' field (the primary commonsense relation) and a 'pid'
    that joins to the TSV's PID column. When a PID has multiple annotations,
    the first occurrence is kept.

    Returns:
        Mapping from PID to {"gold_relation": str, "target_dimension": str}.
    """
    lookup: dict[str, dict[str, str]] = {}
    if not gold_path.exists():
        logger.warning("Commonsense gold file not found: {}", gold_path)
        return lookup

    with open(gold_path, encoding="utf-8") as f:
        entries = json.load(f)

    for entry in entries:
        pid = entry.get("pid", "")
        gold = entry.get("gold", "")
        if not pid or not gold:
            continue
        # First occurrence wins (some PIDs have multiple annotations)
        if pid in lookup:
            continue
        target_dim = CS_RELATION_TO_DIM.get(gold, "")
        lookup[pid] = {"gold_relation": gold, "target_dimension": target_dim}

    logger.info(
        "Loaded commonsense gold for {} dialogues ({} raw entries)",
        len(lookup),
        len(entries),
    )
    return lookup


def create_stage1_entry(
    item: dict[str, Any],
    dialogue_id: int,
    domain_name: str,
    multicultural_lookup: dict[str, dict[str, str]] | None = None,
    commonsense_gold: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """all 23 dimensions present; non-domain dims are None."""
    config = DOMAINS[DomainName(domain_name)]
    prefix = config.prefix

    content = item.get("content", "")
    messages = parse_dialogue_to_messages(content)

    # Collect domain-specific scores
    domain_scores: dict[str, float] = {}
    for dim_name in config.dim_names:
        prefixed_dim = f"{prefix}_{dim_name}"
        score_value = item.get(prefixed_dim)
        if score_value is None:
            score_value = item.get(dim_name)
        normalized = normalize_score(score_value)
        if normalized is not None:
            domain_scores[prefixed_dim] = normalized

    # All 23 dimensions with None defaults, domain scores overwrite
    all_dims = all_prefixed_dimensions()
    full_scores: dict[str, float | None] = {dim: None for dim in all_dims}
    full_scores.update(domain_scores)

    # Domain-specific metadata via backtracking
    domain_metadata: dict[str, str] | None = None
    if domain_name == DomainName.MULTICULTURAL.value and multicultural_lookup:
        uid = item.get("uid", "")
        if uid and uid in multicultural_lookup:
            domain_metadata = multicultural_lookup[uid]
    elif domain_name == DomainName.COMMONSENSE.value and commonsense_gold:
        pid = item.get("pid", "")
        if pid and pid in commonsense_gold:
            domain_metadata = commonsense_gold[pid]

    entry: dict[str, Any] = {
        "dialogue_id": f"S1D-{dialogue_id:06d}",
        "source_id": item.get("_id", ""),
        "domain": domain_name,
        "messages": messages,
        "split": None,
        "scores": full_scores,
        "domain_metadata": domain_metadata,
    }

    return entry


def merge_domains(
    output_dir: Path,
    stage1_output: Path,
    dry_run: bool = False,
) -> int:
    logger.info("=" * 60)
    logger.info("Stage 1 Data Merge")
    logger.info("=" * 60)

    all_items: list[tuple[dict[str, Any], str]] = []
    for domain_name in DOMAINS:
        items = load_domain_data(output_dir, domain_name.value)
        for item in items:
            all_items.append((item, domain_name.value))

    if not all_items:
        logger.error("No items loaded from any domain!")
        return 0

    logger.info("Total items loaded: {}", len(all_items))

    # Load domain-specific metadata lookups once for backtracking
    multicultural_lookup = load_multicultural_metadata()
    commonsense_gold = load_commonsense_gold()

    stage1_entries = [
        create_stage1_entry(
            item, idx, domain_name, multicultural_lookup, commonsense_gold,
        )
        for idx, (item, domain_name) in enumerate(all_items, start=1)
    ]

    if dry_run:
        logger.info(
            "[dry run] Would write {} entries to {}", len(stage1_entries), stage1_output
        )
    else:
        stage1_output.parent.mkdir(parents=True, exist_ok=True)
        with open(stage1_output, "w", encoding="utf-8") as f:
            for entry in stage1_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Wrote {} entries to {}", len(stage1_entries), stage1_output)

    # Summary
    sources_count: dict[str, int] = {}
    for entry in stage1_entries:
        source = entry["domain"]
        sources_count[source] = sources_count.get(source, 0) + 1

    logger.info("-" * 40)
    for source, count in sorted(sources_count.items()):
        logger.info("{:20s}: {:6d}", source, count)
    logger.info("{:20s}: {:6d}", "total", len(stage1_entries))

    return len(stage1_entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Stage 1 domain outputs")

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage_1.jsonl"),
        help="Output file path (default: data/stage_1.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/stage_1"),
        help="Directory containing domain output folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without writing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=level, format="<level>{message}</level>")

    count = merge_domains(
        output_dir=args.output_dir,
        stage1_output=args.output,
        dry_run=args.dry_run,
    )

    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
