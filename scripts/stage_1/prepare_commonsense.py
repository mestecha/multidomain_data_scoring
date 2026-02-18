"""commonsense domain processor for stage 1 evaluation (dual-prompt strategy)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from loguru import logger

from scripts.config import COMMONSENSE
from scripts.models import BatchEntry, ManifestItem
from scripts.stage_1.base import DomainProcessor
from scripts.stage_1.prompts import (
    build_commonsense_dim_prompt,
    build_commonsense_dlg_prompt,
)

# Relation response column names in the TSV
RELATION_RESPONSE_COLS = [
    "HinderedBy_response",
    "IsAfter_response",
    "xAttr_response",
    "xReact_response",
    "oReact_response",
    "xWant_response",
    "oWant_response",
]


class CommonsenseProcessor(DomainProcessor):
    """builds two batch entries per item: {id}-dim and {id}-dlg."""

    def __init__(
        self,
        base_path: Path | None = None,
        max_samples: int | None = None,
    ):
        super().__init__(COMMONSENSE, base_path, max_samples=max_samples)
        self.train_path = self.input_dir / "fabricated" / "train_gpt35.tsv"
        self.test_path = self.input_dir / "fabricated" / "test_gpt35.tsv"

    def load_data(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []

        for tsv_path in [self.train_path, self.test_path]:
            if not tsv_path.exists():
                logger.warning("File not found: {}", tsv_path)
                continue

            loaded = self._load_tsv(tsv_path)
            logger.debug("Loaded {} items from {}", len(loaded), tsv_path.name)
            items.extend(loaded)

        logger.info("Total commonsense items: {}", len(items))
        return items

    def _load_tsv(self, path: Path) -> list[dict[str, Any]]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                items.append(dict(row))
        return items

    def _parse_context(self, ctx: str) -> str:
        parts = ctx.split("[UTT]")
        lines = []
        for part in parts:
            part = part.strip()
            if part:
                part = part.replace("user1:", "Speaker 1:")
                part = part.replace("user2:", "Speaker 2:")
                lines.append(part)
        return "\n".join(lines)

    def _get_relation_responses(self, item: dict[str, Any]) -> dict[str, str]:
        responses = {}
        for col in RELATION_RESPONSE_COLS:
            # Column is like "HinderedBy_response" → key is "HinderedBy"
            relation = col.replace("_response", "")
            responses[relation] = item.get(col, "")
        return responses

    def build_prompt(self, item: dict[str, Any]) -> str:
        # satisfies abstract interface; actual prompts built in prepare()
        return self._build_dlg_prompt(item)

    def _build_dim_prompt(self, item: dict[str, Any]) -> str:
        context = item.get("CTX", "")
        target = item.get("SEG", "")
        if not context or not target:
            return ""

        parsed_context = self._parse_context(context)
        relation_responses = self._get_relation_responses(item)

        return build_commonsense_dim_prompt(
            context=parsed_context,
            target_utterance=target,
            relation_responses=relation_responses,
        )

    def _build_dlg_prompt(self, item: dict[str, Any]) -> str:
        context = item.get("CTX", "")
        target = item.get("SEG", "")
        if not context or not target:
            return ""

        parsed_context = self._parse_context(context)
        return build_commonsense_dlg_prompt(
            context=parsed_context,
            target_utterance=target,
        )

    def create_custom_id(self, item: dict[str, Any]) -> str:
        pid = item.get("PID", "")
        if pid:
            return f"{self.prefix}-{pid}"
        ctx = item.get("CTX", "") + item.get("SEG", "")
        fp = self.fingerprint(ctx)[:16]
        return f"{self.prefix}-{fp}"

    def get_fingerprint_text(self, item: dict[str, Any]) -> str | None:
        ctx = item.get("CTX", "")
        seg = item.get("SEG", "")
        return ctx + seg if ctx or seg else None

    def create_manifest_item(
        self,
        item: dict[str, Any],
        custom_id: str,
    ) -> ManifestItem:
        ctx = item.get("CTX", "")
        seg = item.get("SEG", "")
        parsed_ctx = self._parse_context(ctx) if ctx else ""
        content = f"{parsed_ctx}\nTarget: {seg}" if seg else parsed_ctx

        return ManifestItem(
            custom_id=custom_id,
            content=content,
            pid=item.get("PID", ""),
            uid=item.get("UID", ""),
        )

    def prepare(self) -> dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Preparing {} domain (prefix: {}_)", self.domain, self.prefix)
        logger.info("Dual-prompt strategy: 2 calls per item")
        logger.info("=" * 60)

        # Load data
        logger.info("Loading data...")
        data = self.load_data()
        logger.info("Loaded {} items", len(data))

        # Apply max_samples limit
        if self.max_samples and len(data) > self.max_samples:
            data = data[: self.max_samples]
            logger.info("Limited to {} samples", self.max_samples)

        # Process items — build TWO entries per item
        logger.info("Processing items (dual-prompt)...")
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

            base_id = self.create_custom_id(item)

            # Build dim-specific prompt (Call 1)
            dim_prompt = self._build_dim_prompt(item)
            # Build dlg-level prompt (Call 2)
            dlg_prompt = self._build_dlg_prompt(item)

            if not dim_prompt or not dlg_prompt:
                skipped["invalid"] += 1
                continue

            dim_entry = BatchEntry(
                custom_id=f"{base_id}-dim",
                prompt=dim_prompt,
                model=self.model,
            )
            dlg_entry = BatchEntry(
                custom_id=f"{base_id}-dlg",
                prompt=dlg_prompt,
                model=self.model,
            )

            entries.append(dim_entry)
            entries.append(dlg_entry)

            # One manifest item per dialogue (not per prompt)
            manifest_items.append(self.create_manifest_item(item, base_id))

        logger.info("Valid dialogues: {}", len(manifest_items))
        logger.info("Total batch entries: {} (2 per dialogue)", len(entries))
        logger.debug("Skipped (duplicate): {}", skipped["duplicate"])
        logger.debug("Skipped (invalid): {}", skipped["invalid"])

        # Write shards
        logger.info("Writing shards...")
        shard_paths = self.write_shards(entries, manifest_items)

        summary = {
            "domain": self.domain,
            "prefix": self.prefix,
            "total_items": len(data),
            "valid_dialogues": len(manifest_items),
            "valid_entries": len(entries),
            "skipped_duplicate": skipped["duplicate"],
            "skipped_invalid": skipped["invalid"],
            "num_shards": len(shard_paths),
            "shard_paths": [str(p) for p in shard_paths],
        }

        logger.info("=" * 60)
        logger.success(
            "{} dialogues → {} entries in {} shard(s)",
            len(manifest_items),
            len(entries),
            len(shard_paths),
        )
        logger.info("=" * 60)

        return summary
