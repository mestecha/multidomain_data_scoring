"""train/test split assignment for stage 1 data, stratified by domain."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from scripts.config import MULTICULTURAL_TEST_DIR, SPLIT_RATIOS


def _load_forced_test_ids(test_dir: Path) -> set[str]:
    """read multicultural test CSVs and return source_ids (mu-{uid})."""
    forced: set[str] = set()
    for csv_path in sorted(test_dir.glob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                forced.add(f"mu-{row['uid']}")
    return forced


def assign_splits(
    entries: list[dict[str, Any]],
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    forced_test_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """assign train/test splits stratified by domain, mutates entries in place.

    forced_test_ids are placed in test and count toward their domain's test
    quota.  remaining entries are split per-domain so every domain hits
    ~ratios["test"] independently.
    """
    ratios = ratios or SPLIT_RATIOS
    forced_test_ids = forced_test_ids or set()
    test_ratio = ratios["test"]
    rng = np.random.RandomState(seed)

    # group indices by domain, separating forced from rest
    forced_by_domain: dict[str, list[int]] = defaultdict(list)
    rest_by_domain: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        domain = e["domain"]
        if e.get("source_id") in forced_test_ids:
            forced_by_domain[domain].append(i)
        else:
            rest_by_domain[domain].append(i)

    # mark forced entries as test
    for indices in forced_by_domain.values():
        for idx in indices:
            entries[idx]["split"] = "test"

    # per-domain split: target_test - n_forced = extra needed from rest
    all_domains = sorted(set(list(forced_by_domain) + list(rest_by_domain)))
    for domain in all_domains:
        forced = forced_by_domain[domain]
        rest = rest_by_domain[domain]
        domain_total = len(forced) + len(rest)
        target_test_domain = round(domain_total * test_ratio)
        n_extra = max(0, target_test_domain - len(forced))

        if n_extra > 0 and rest:
            rest_arr = np.array(rest)
            rng.shuffle(rest_arr)
            for idx in rest_arr[:n_extra]:
                entries[idx]["split"] = "test"
            for idx in rest_arr[n_extra:]:
                entries[idx]["split"] = "train"
        else:
            for idx in rest:
                entries[idx]["split"] = "train"

    # log distribution
    split_counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        domain = entry["domain"]
        split = entry["split"]
        if domain not in split_counts:
            split_counts[domain] = {}
        split_counts[domain][split] = split_counts[domain].get(split, 0) + 1

    for domain, counts in sorted(split_counts.items()):
        total_d = sum(counts.values())
        parts = ", ".join(
            f"{s}: {c} ({c / total_d:.1%})" for s, c in sorted(counts.items())
        )
        logger.info("  {}: {}", domain, parts)

    return entries


def relabel_stage2_pairs(
    stage1_path: Path,
    pairs_path: Path,
    output_path: Path | None = None,
) -> int:
    """update metadata.split in pairs.jsonl to match stage_1.jsonl splits.

    Reads the dialogue_id → split mapping from stage1_path and rewrites
    every pair's metadata.split to match its source dialogue.  Pairs whose
    source_dialogue_id is not found in stage 1 are left unchanged.

    Returns:
        Number of pairs whose split field was changed.
    """
    id_to_split: dict[str, str] = {}
    with open(stage1_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            dialogue_id = entry.get("dialogue_id") or entry.get("stage_1_dialogue_id")
            split = entry.get("split")
            if dialogue_id and split:
                id_to_split[dialogue_id] = split

    logger.info("Loaded {} dialogue→split mappings from {}", len(id_to_split), stage1_path)

    out = output_path or pairs_path
    updated_pairs: list[dict] = []
    changed = 0

    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            dialogue_id = pair.get("source_dialogue_id")
            if dialogue_id and dialogue_id in id_to_split:
                new_split = id_to_split[dialogue_id]
                old_split = pair.get("metadata", {}).get("split")
                if old_split != new_split:
                    pair["metadata"]["split"] = new_split
                    changed += 1
            updated_pairs.append(pair)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for pair in updated_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info("Relabeled {}/{} pairs → {}", changed, len(updated_pairs), out)
    return changed


def split_stage1_file(
    input_path: Path,
    output_path: Path | None = None,
    seed: int = 42,
) -> int:
    output_path = output_path or input_path

    entries = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    # load forced multicultural test ids
    forced_ids: set[str] = set()
    if MULTICULTURAL_TEST_DIR.exists():
        forced_ids = _load_forced_test_ids(MULTICULTURAL_TEST_DIR)
        logger.info("Loaded {} forced test IDs from {}", len(forced_ids), MULTICULTURAL_TEST_DIR)

    logger.info("Loaded {} entries for splitting", len(entries))
    entries = assign_splits(entries, seed=seed, forced_test_ids=forced_ids)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.success("Wrote {} entries with splits to {}", len(entries), output_path)
    return len(entries)
