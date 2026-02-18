"""train/val/test split assignment for stage 1 data, stratified by domain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from sklearn.model_selection import StratifiedShuffleSplit

from scripts.config import SPLIT_RATIOS


def assign_splits(
    entries: list[dict[str, Any]],
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """assign train/val/test splits stratified by domain, mutates entries in place."""
    ratios = ratios or SPLIT_RATIOS

    # Extract domain labels for stratification
    domains = [e["domain"] for e in entries]

    # First split: separate test set
    test_ratio = ratios["test"]
    train_val_ratio = 1.0 - test_ratio

    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    train_val_idx, test_idx = next(sss_test.split(entries, domains))

    # Mark test entries
    for idx in test_idx:
        entries[idx]["split"] = "test"

    # Second split: separate val from train within train_val set
    train_val_entries = [entries[i] for i in train_val_idx]
    train_val_domains = [domains[i] for i in train_val_idx]

    # val_ratio relative to train_val set
    val_relative = ratios["val"] / train_val_ratio

    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_relative, random_state=seed
    )
    train_idx_local, val_idx_local = next(
        sss_val.split(train_val_entries, train_val_domains)
    )

    # Map back to original indices
    for local_idx in val_idx_local:
        original_idx = train_val_idx[local_idx]
        entries[original_idx]["split"] = "val"

    for local_idx in train_idx_local:
        original_idx = train_val_idx[local_idx]
        entries[original_idx]["split"] = "train"

    # Log distribution
    split_counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        domain = entry["domain"]
        split = entry["split"]
        if domain not in split_counts:
            split_counts[domain] = {}
        split_counts[domain][split] = split_counts[domain].get(split, 0) + 1

    for domain, counts in sorted(split_counts.items()):
        total = sum(counts.values())
        parts = ", ".join(
            f"{s}: {c} ({c / total:.1%})" for s, c in sorted(counts.items())
        )
        logger.info("  {}: {}", domain, parts)

    return entries


def split_stage1_file(
    input_path: Path,
    output_path: Path | None = None,
    seed: int = 42,
) -> int:
    output_path = output_path or input_path

    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    logger.info("Loaded {} entries for splitting", len(entries))
    entries = assign_splits(entries, seed=seed)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.success("Wrote {} entries with splits to {}", len(entries), output_path)
    return len(entries)
