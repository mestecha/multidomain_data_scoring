"""multicultural domain processor for stage 1 evaluation."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from loguru import logger

from scripts.config import MULTICULTURAL
from scripts.models import ManifestItem
from scripts.stage_1.base import DomainProcessor
from scripts.stage_1.prompts import build_multicultural_prompt


class MulticulturalProcessor(DomainProcessor):
    def __init__(
        self,
        base_path: Path | None = None,
        max_samples: int | None = None,
    ):
        super().__init__(MULTICULTURAL, base_path, max_samples=max_samples)
        self.countries_dir = self.input_dir / "countries"

    def load_data(self) -> list[dict[str, Any]]:
        csv_files = sorted(self.countries_dir.glob("*.csv"))
        logger.debug("Found {} country files", len(csv_files))

        items = []
        for csv_path in csv_files:
            country_code = csv_path.stem
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["_source_file"] = country_code
                    items.append(row)
            logger.debug("  {}: loaded", country_code)

        return items

    def build_prompt(self, item: dict[str, Any]) -> str:
        conversation = item.get("conversation", "")
        statement = item.get("statement_original", "")
        country_1 = item.get("country_1", "")
        country_2 = item.get("country_2", "")

        if not conversation or not statement:
            return ""

        turns_text = self._parse_conversation(conversation)
        if not turns_text:
            return ""

        return build_multicultural_prompt(
            turns_text=turns_text,
            statement=statement,
            country_1=country_1,
            country_2=country_2,
        )

    def _parse_conversation(self, conversation: str) -> str:
        """convert SPK01-XXX001 format to numbered turn lines."""
        lines = conversation.strip().split("\n")
        speaker_pattern = re.compile(r"^(SPK\d{2}-[A-Z]{3}\d{3}):\s*(.*)$")

        turns = []
        turn_id = 1

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = speaker_pattern.match(line)
            if match:
                speaker_id = match.group(1)
                content = match.group(2).strip()
                speaker_num = 1 if speaker_id.startswith("SPK01") else 2
                turns.append(f"Turn {turn_id} (Speaker {speaker_num}): {content}")
                turn_id += 1
            elif turns:
                turns[-1] += " " + line

        return "\n".join(turns)

    def create_custom_id(self, item: dict[str, Any]) -> str:
        uid = item.get("uid", "")
        if uid:
            return f"{self.prefix}-{uid}"
        conversation = item.get("conversation", "")
        fp = self.fingerprint(conversation)[:16]
        return f"{self.prefix}-{fp}"

    def get_fingerprint_text(self, item: dict[str, Any]) -> str | None:
        return item.get("conversation")

    def create_manifest_item(
        self,
        item: dict[str, Any],
        custom_id: str,
    ) -> ManifestItem:
        return ManifestItem(
            custom_id=custom_id,
            content=item.get("conversation", ""),
            uid=item.get("uid", ""),
            country_1=item.get("country_1", ""),
            country_2=item.get("country_2", ""),
            source_file=item.get("_source_file", ""),
        )
