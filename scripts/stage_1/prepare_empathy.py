"""empathy domain processor for stage 1 evaluation."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from scripts.config import EMPATHY, TURN_BUCKETS
from scripts.models import ManifestItem
from scripts.stage_1.base import DomainProcessor
from scripts.stage_1.prompts import build_empathy_prompt

# Target turn-count distribution
_TURN_BUCKET_WEIGHTS = {
    (2, 6): 0.25,
    (7, 10): 0.30,
    (11, 16): 0.30,
    (17, 999): 0.15,
}


def _count_turns(content: str) -> int:
    lines = [ln.strip() for ln in content.strip().split("\n") if ln.strip()]
    return max(len(lines), 1)


def _assign_bucket(turn_count: int) -> tuple[int, int]:
    for lo, hi in TURN_BUCKETS:
        if lo <= turn_count <= hi:
            return (lo, hi)
    return TURN_BUCKETS[-1]


def _stratified_sample(
    items: list[dict[str, Any]],
    target: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """stratified sample by turn-count bucket, with shortfall redistribution."""
    rng = random.Random(seed)

    # Deduplicate
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        fp = DomainProcessor.fingerprint(item.get("content", ""))
        if fp not in seen:
            seen.add(fp)
            unique.append(item)

    # Group by bucket
    by_bucket: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for item in unique:
        turns = _count_turns(item.get("content", ""))
        bucket = _assign_bucket(turns)
        by_bucket[bucket].append(item)

    # Calculate targets per bucket
    bucket_targets: dict[tuple[int, int], int] = {}
    for bucket, weight in _TURN_BUCKET_WEIGHTS.items():
        bucket_targets[bucket] = int(target * weight)

    # Adjust rounding
    allocated = sum(bucket_targets.values())
    if allocated < target:
        largest = max(bucket_targets, key=lambda b: len(by_bucket.get(b, [])))
        bucket_targets[largest] += target - allocated

    # Sample each bucket, tracking shortfall
    sampled: list[dict[str, Any]] = []
    shortfall = 0
    for bucket in TURN_BUCKETS:
        bucket_key = tuple(bucket)
        available = by_bucket.get(bucket_key, [])
        needed = bucket_targets.get(bucket_key, 0)

        rng.shuffle(available)
        take = min(needed, len(available))
        sampled.extend(available[:take])
        shortfall += needed - take

    # Redistribute shortfall
    if shortfall > 0:
        already = {id(item) for item in sampled}
        remaining = [
            item
            for bucket_list in by_bucket.values()
            for item in bucket_list
            if id(item) not in already
        ]
        rng.shuffle(remaining)
        sampled.extend(remaining[:shortfall])

    rng.shuffle(sampled)
    return sampled


class EmpathyProcessor(DomainProcessor):
    def __init__(
        self,
        base_path: Path | None = None,
        max_samples: int | None = None,
        max_count: int = 12800,
        seed: int = 42,
    ):
        super().__init__(EMPATHY, base_path, max_samples=max_samples)
        self.max_count = max_count
        self.seed = seed
        self.data_path = self.input_dir / "raw_train_empathy_data.json"

    def load_data(self) -> list[dict[str, Any]]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Handle different possible formats
        if isinstance(raw, list):
            dialogues = raw
        elif isinstance(raw, dict):
            dialogues = raw.get("dialogue", raw.get("dialogues", []))
        else:
            dialogues = []

        # Convert to standard format
        items = []
        for d in dialogues:
            if isinstance(d, str):
                items.append({"content": d})
            elif isinstance(d, dict):
                content = d.get("content", d.get("dialogue", d.get("input", "")))
                items.append({"content": content, **d})

        logger.debug("Raw items: {}", len(items))

        selected = _stratified_sample(
            items,
            target=self.max_count,
            seed=self.seed,
        )

        # Log distribution
        bucket_counts: dict[tuple[int, int], int] = defaultdict(int)
        for item in selected:
            bucket_counts[_assign_bucket(_count_turns(item.get("content", "")))] += 1
        logger.debug("Bucket distribution: {}", dict(bucket_counts))

        return selected

    def build_prompt(self, item: dict[str, Any]) -> str:
        content = item.get("content", "")
        if not content:
            return ""
        return build_empathy_prompt(content)

    def create_custom_id(self, item: dict[str, Any]) -> str:
        content = item.get("content", "")
        fp = self.fingerprint(content)[:16]
        return f"{self.prefix}-{fp}"

    def get_fingerprint_text(self, item: dict[str, Any]) -> str | None:
        return item.get("content")

    def create_manifest_item(
        self,
        item: dict[str, Any],
        custom_id: str,
    ) -> ManifestItem:
        return ManifestItem(
            custom_id=custom_id,
            content=item.get("content", ""),
        )
