"""coherence domain processor for stage 1 evaluation."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from scripts.config import COHERENCE, COHERENCE_LABEL_RATIO, TURN_BUCKETS
from scripts.models import ManifestItem
from scripts.stage_1.base import DomainProcessor
from scripts.stage_1.prompts import build_coherence_prompt

# Target turn-count distribution (within each label group)
_TURN_BUCKET_WEIGHTS = {
    (2, 6): 0.20,
    (7, 10): 0.25,
    (11, 16): 0.30,
    (17, 999): 0.25,
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
    label_ratio: dict[str, float],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """stratified sample by label and turn-count bucket, with shortfall redistribution."""
    rng = random.Random(seed)

    # Deduplicate
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        fp = DomainProcessor.fingerprint(item.get("content", ""))
        if fp not in seen:
            seen.add(fp)
            unique.append(item)

    # Group by label → bucket
    groups: dict[str, dict[tuple[int, int], list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in unique:
        label = item.get("label", "Yes")
        turns = _count_turns(item.get("content", ""))
        bucket = _assign_bucket(turns)
        groups[label][bucket].append(item)

    # Sample per label
    sampled: list[dict[str, Any]] = []
    for label, ratio in label_ratio.items():
        label_target = int(target * ratio)
        label_items = groups.get(label, {})

        # Allocate per bucket
        bucket_targets: dict[tuple[int, int], int] = {}
        for bucket, weight in _TURN_BUCKET_WEIGHTS.items():
            bucket_targets[bucket] = int(label_target * weight)

        # Adjust for rounding
        allocated = sum(bucket_targets.values())
        if allocated < label_target:
            # Give remainder to largest bucket
            largest = max(bucket_targets, key=lambda b: len(label_items.get(b, [])))
            bucket_targets[largest] += label_target - allocated

        # Sample each bucket
        shortfall = 0
        for bucket in TURN_BUCKETS:
            bucket_key = tuple(bucket)
            available = label_items.get(bucket_key, [])
            needed = bucket_targets.get(bucket_key, 0)

            rng.shuffle(available)
            take = min(needed, len(available))
            sampled.extend(available[:take])
            shortfall += needed - take

        # Redistribute shortfall from any remaining items
        if shortfall > 0:
            already = {id(item) for item in sampled}
            remaining = [
                item
                for bucket_list in label_items.values()
                for item in bucket_list
                if id(item) not in already
            ]
            rng.shuffle(remaining)
            sampled.extend(remaining[:shortfall])

    rng.shuffle(sampled)
    return sampled


class CoherenceProcessor(DomainProcessor):
    def __init__(
        self,
        base_path: Path | None = None,
        max_samples: int | None = None,
        max_count: int = 12800,
        seed: int = 42,
    ):
        super().__init__(COHERENCE, base_path, max_samples=max_samples)
        self.max_count = max_count
        self.seed = seed
        self.data_path = self.input_dir / "raw_train_coherence_data.json"

    def load_data(self) -> list[dict[str, Any]]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        dialogues = raw.get("dialogue", [])
        labels = raw.get("labels", [])

        if len(dialogues) != len(labels):
            raise ValueError(
                f"Mismatch: {len(dialogues)} dialogues but {len(labels)} labels"
            )

        items = []
        for dialogue, label in zip(dialogues, labels):
            if isinstance(dialogue, str):
                items.append({"content": dialogue, "label": label})

        logger.debug("Raw items: {}", len(items))

        selected = _stratified_sample(
            items,
            target=self.max_count,
            label_ratio=COHERENCE_LABEL_RATIO,
            seed=self.seed,
        )

        # Log distribution
        label_counts: dict[str, int] = defaultdict(int)
        bucket_counts: dict[tuple[int, int], int] = defaultdict(int)
        for item in selected:
            label_counts[item.get("label", "?")] += 1
            bucket_counts[_assign_bucket(_count_turns(item.get("content", "")))] += 1

        logger.debug("Label distribution: {}", dict(label_counts))
        logger.debug("Bucket distribution: {}", dict(bucket_counts))

        return selected

    def build_prompt(self, item: dict[str, Any]) -> str:
        content = item.get("content", "")
        if not content:
            return ""
        return build_coherence_prompt(content)

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
            label=item.get("label", ""),
        )
