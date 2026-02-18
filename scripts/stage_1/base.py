"""abstract interface for stage 1 domain processors."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger

from scripts.models import BatchEntry, DomainConfig, ManifestItem


class DomainProcessor(ABC):
    def __init__(
        self,
        config: DomainConfig,
        base_path: Path | None = None,
        max_samples: int | None = None,
        model: str = "gpt-5.1-batch",
        shard_size: int = 5000,
    ):
        self.config = config
        self.base_path = base_path or Path(".")
        self.max_samples = max_samples
        self.model = model
        self.shard_size = shard_size
        self.input_dir = self.base_path / "data" / "input" / config.name.value
        self.shard_dir = self.input_dir / "shards"
        self.manifest_path = self.input_dir / "manifest.jsonl"

    @property
    def domain(self) -> str:
        return self.config.name.value

    @property
    def prefix(self) -> str:
        return self.config.prefix

    @abstractmethod
    def load_data(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def build_prompt(self, item: dict[str, Any]) -> str: ...

    @abstractmethod
    def create_custom_id(self, item: dict[str, Any]) -> str: ...

    @staticmethod
    def fingerprint(text: str) -> str:
        return hashlib.md5(text.strip().encode()).hexdigest()

    def create_batch_entry(self, item: dict[str, Any]) -> BatchEntry | None:
        prompt = self.build_prompt(item)
        if not prompt:
            return None

        custom_id = self.create_custom_id(item)
        return BatchEntry(custom_id=custom_id, prompt=prompt, model=self.model)

    def write_shards(
        self,
        entries: list[BatchEntry],
        manifest_items: list[ManifestItem],
    ) -> list[Path]:
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        shard_paths = []
        for i in range(0, len(entries), self.shard_size):
            shard_entries = entries[i : i + self.shard_size]
            shard_num = i // self.shard_size + 1
            shard_path = self.shard_dir / f"{self.prefix}_shard_{shard_num:04d}.jsonl"

            with open(shard_path, "w", encoding="utf-8") as f:
                for entry in shard_entries:
                    f.write(json.dumps(entry.to_api_dict(), ensure_ascii=False) + "\n")

            shard_paths.append(shard_path)
            logger.debug("Wrote {}: {} entries", shard_path.name, len(shard_entries))

        # Write manifest
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            for item in manifest_items:
                f.write(
                    json.dumps(item.model_dump(exclude_none=True), ensure_ascii=False)
                    + "\n"
                )
        logger.debug("Wrote manifest: {}", self.manifest_path.name)

        return shard_paths

    def prepare(self) -> dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Preparing {} domain (prefix: {}_)", self.domain, self.prefix)
        logger.info("=" * 60)

        # Load data
        logger.info("Loading data...")
        data = self.load_data()
        logger.info("Loaded {} items", len(data))

        # Apply max_samples limit if set
        if self.max_samples and len(data) > self.max_samples:
            data = data[: self.max_samples]
            logger.info("Limited to {} samples", self.max_samples)

        # Process items
        logger.info("Processing items...")
        entries: list[BatchEntry] = []
        manifest_items: list[ManifestItem] = []
        seen_fingerprints: set[str] = set()
        skipped = {"duplicate": 0, "invalid": 0}

        for item in data:
            # Deduplication
            fp = self.get_fingerprint_text(item)
            if fp:
                fp_hash = self.fingerprint(fp)
                if fp_hash in seen_fingerprints:
                    skipped["duplicate"] += 1
                    continue
                seen_fingerprints.add(fp_hash)

            # Create batch entry
            entry = self.create_batch_entry(item)
            if entry is None:
                skipped["invalid"] += 1
                continue

            entries.append(entry)
            manifest_items.append(self.create_manifest_item(item, entry.custom_id))

        logger.info("Valid entries: {}", len(entries))
        logger.debug("Skipped (duplicate): {}", skipped["duplicate"])
        logger.debug("Skipped (invalid): {}", skipped["invalid"])

        # Write shards
        logger.info("Writing shards...")
        shard_paths = self.write_shards(entries, manifest_items)

        # Summary
        summary = {
            "domain": self.domain,
            "prefix": self.prefix,
            "total_items": len(data),
            "valid_entries": len(entries),
            "skipped_duplicate": skipped["duplicate"],
            "skipped_invalid": skipped["invalid"],
            "num_shards": len(shard_paths),
            "shard_paths": [str(p) for p in shard_paths],
        }

        logger.info("=" * 60)
        logger.success("{} entries in {} shard(s)", len(entries), len(shard_paths))
        logger.info("=" * 60)

        return summary

    def get_fingerprint_text(self, item: dict[str, Any]) -> str | None:
        """override in subclass; return None to skip deduplication."""
        return None

    def create_manifest_item(
        self,
        item: dict[str, Any],
        custom_id: str,
    ) -> ManifestItem:
        """override in subclass for domain-specific metadata."""
        return ManifestItem(custom_id=custom_id)
