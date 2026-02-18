"""parse gpt batch api results into domain-specific output files."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from scripts.config import get_domain
from scripts.models import DomainConfig, DomainName, ScoreScale


def strip_code_fences(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```$", "", s).strip()
    left, right = s.find("{"), s.rfind("}")
    if left != -1 and right != -1 and right > left:
        s = s[left : right + 1]
    return s


def fix_json(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def parse_message_to_dict(s: str) -> dict[str, Any] | None:
    txt = fix_json(strip_code_fences(s))
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(txt)
            return obj if isinstance(obj, dict) else None
        except (ValueError, SyntaxError):
            return None


def coerce_score(x: Any) -> int | None:
    """clamp to integer in [1, 5]."""
    try:
        v = int(round(float(x)))
        return min(5, max(1, v))
    except (ValueError, TypeError):
        return None


def normalize_1_5_score(score: int | None) -> float | None:
    """map 1-5 integer to 0.0-1.0 float."""
    if score is None:
        return None
    return round((score - 1) / 4.0, 2)


def coerce_float(x: Any) -> float | None:
    """clamp to float in [0.0, 1.0]."""
    try:
        v = float(x)
        return round(min(1.0, max(0.0, v)), 2)
    except (ValueError, TypeError):
        return None


class ResultParser:
    def __init__(self, config: DomainConfig):
        self.config = config
        self.domain = config.name.value
        self.prefix = config.prefix
        self.dimensions = config.dim_names
        self.score_scale = config.score_scale

    def parse_batch_output(
        self,
        batch_output_dir: Path,
        manifest_path: Path,
        output_path: Path,
    ) -> dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Parsing {} results", self.domain)
        logger.info("=" * 60)

        batch_df = self._load_batch_outputs(batch_output_dir)
        logger.info("Loaded {} batch results", len(batch_df))

        manifest = self._load_manifest(manifest_path)
        logger.info("Loaded {} manifest entries", len(manifest))

        # For commonsense, merge dim + dlg results before joining manifest
        if self.config.name == DomainName.COMMONSENSE:
            parsed_items, failed_items = self._parse_commonsense(batch_df, manifest)
        else:
            parsed_items, failed_items = self._parse_standard(batch_df, manifest)

        logger.info("Parsed {} items, failed {}", len(parsed_items), len(failed_items))
        self._log_failures(failed_items)

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in parsed_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.success("Wrote {}", output_path)

        return {
            "domain": self.domain,
            "total_batch": len(batch_df),
            "parsed": len(parsed_items),
            "failed": len(failed_items),
            "failed_items": failed_items,
            "output_path": str(output_path),
        }

    def _parse_standard(
        self,
        batch_df: pd.DataFrame,
        manifest: pd.DataFrame,
    ) -> tuple[list[dict], list[dict]]:
        batch_ids = set(batch_df["custom_id"].tolist())
        manifest_ids = set(manifest["custom_id"].tolist())
        missing_ids = manifest_ids - batch_ids

        merged = batch_df.merge(manifest, how="left", on="custom_id")

        parsed_items = []
        failed_items: list[dict[str, Any]] = []

        for cid in missing_ids:
            failed_items.append(
                {
                    "custom_id": cid,
                    "reason": "missing_from_batch",
                    "detail": "Request sent but no response received",
                }
            )

        for row in merged.itertuples(index=False):
            custom_id = getattr(row, "custom_id", "unknown")

            if getattr(row, "status", None) != "Success":
                error_msg = getattr(row, "error", None)
                failed_items.append(
                    {
                        "custom_id": custom_id,
                        "reason": "api_error",
                        "detail": error_msg
                        or f"status_code={getattr(row, 'status_code', 'unknown')}",
                    }
                )
                continue

            message = getattr(row, "message", "")
            parsed = parse_message_to_dict(message)
            if parsed is None:
                failed_items.append(
                    {
                        "custom_id": custom_id,
                        "reason": "json_parse_error",
                        "detail": message[:100] if message else "empty message",
                    }
                )
                continue

            normalized = self._normalize_scores(parsed)
            if normalized is None:
                failed_items.append(
                    {
                        "custom_id": custom_id,
                        "reason": "score_normalization_error",
                        "detail": str(parsed)[:100],
                    }
                )
                continue

            item = {
                "_id": row.custom_id,
                "_model": getattr(row, "model", ""),
                "_file": getattr(row, "file", ""),
            }
            for col in manifest.columns:
                if col != "custom_id" and hasattr(row, col):
                    item[col] = getattr(row, col)
            item.update(normalized)
            parsed_items.append(item)

        return parsed_items, failed_items

    def _parse_commonsense(
        self,
        batch_df: pd.DataFrame,
        manifest: pd.DataFrame,
    ) -> tuple[list[dict], list[dict]]:
        """merge -dim and -dlg responses by base custom_id."""
        parsed_items = []
        failed_items: list[dict[str, Any]] = []

        # Build lookup from batch results: base_id → {suffix: row}
        batch_by_base: dict[str, dict[str, Any]] = defaultdict(dict)
        for row in batch_df.itertuples(index=False):
            cid = getattr(row, "custom_id", "")
            if cid.endswith("-dim"):
                base_id = cid[:-4]
                batch_by_base[base_id]["dim"] = row
            elif cid.endswith("-dlg"):
                base_id = cid[:-4]
                batch_by_base[base_id]["dlg"] = row
            else:
                # Legacy single-prompt entries (backwards compat)
                batch_by_base[cid]["single"] = row

        # Build manifest lookup
        manifest_dict = {}
        for row in manifest.itertuples(index=False):
            manifest_dict[getattr(row, "custom_id", "")] = row

        # Process each manifest entry
        for m_row in manifest.itertuples(index=False):
            base_id = getattr(m_row, "custom_id", "")
            batch_entries = batch_by_base.get(base_id, {})

            # Handle legacy single-prompt format
            if "single" in batch_entries and "dim" not in batch_entries:
                row = batch_entries["single"]
                if getattr(row, "status", None) != "Success":
                    failed_items.append(
                        {
                            "custom_id": base_id,
                            "reason": "api_error",
                            "detail": str(getattr(row, "error", "")),
                        }
                    )
                    continue
                message = getattr(row, "message", "")
                parsed = parse_message_to_dict(message)
                if parsed is None:
                    failed_items.append(
                        {
                            "custom_id": base_id,
                            "reason": "json_parse_error",
                            "detail": message[:100],
                        }
                    )
                    continue
                normalized = self._normalize_scores(parsed)
                if normalized is None:
                    failed_items.append(
                        {
                            "custom_id": base_id,
                            "reason": "score_normalization_error",
                            "detail": str(parsed)[:100],
                        }
                    )
                    continue
                item = self._build_output_item(base_id, m_row, manifest)
                item.update(normalized)
                parsed_items.append(item)
                continue

            # Dual-prompt: need both dim and dlg
            merged_scores: dict[str, Any] = {}

            for suffix, expected_dims in [
                ("dim", ["causality", "consistency", "reaction", "desire"]),
                ("dlg", ["coherence", "empathy"]),
            ]:
                row = batch_entries.get(suffix)
                if row is None:
                    failed_items.append(
                        {
                            "custom_id": f"{base_id}-{suffix}",
                            "reason": "missing_from_batch",
                            "detail": f"No {suffix} response for {base_id}",
                        }
                    )
                    merged_scores = {}
                    break

                if getattr(row, "status", None) != "Success":
                    failed_items.append(
                        {
                            "custom_id": f"{base_id}-{suffix}",
                            "reason": "api_error",
                            "detail": str(getattr(row, "error", "")),
                        }
                    )
                    merged_scores = {}
                    break

                message = getattr(row, "message", "")
                parsed = parse_message_to_dict(message)
                if parsed is None:
                    failed_items.append(
                        {
                            "custom_id": f"{base_id}-{suffix}",
                            "reason": "json_parse_error",
                            "detail": message[:100] if message else "empty",
                        }
                    )
                    merged_scores = {}
                    break

                # Normalize only the expected dimensions from this call
                for dim in expected_dims:
                    prefixed = f"{self.prefix}_{dim}"
                    value = parsed.get(prefixed)
                    if value is None:
                        value = parsed.get(dim)
                    normalized_val = self._normalize_single_value(value)
                    if normalized_val is not None:
                        merged_scores[prefixed] = normalized_val

            if not merged_scores:
                continue

            # Check we got all 6 dimensions
            expected_all = {f"{self.prefix}_{d}" for d in self.dimensions}
            if not expected_all.issubset(merged_scores.keys()):
                missing = expected_all - merged_scores.keys()
                failed_items.append(
                    {
                        "custom_id": base_id,
                        "reason": "incomplete_merge",
                        "detail": f"Missing dims: {missing}",
                    }
                )
                continue

            item = self._build_output_item(base_id, m_row, manifest)
            item.update(merged_scores)
            parsed_items.append(item)

        return parsed_items, failed_items

    def _build_output_item(
        self,
        base_id: str,
        m_row: Any,
        manifest: pd.DataFrame,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {"_id": base_id}
        for col in manifest.columns:
            if col != "custom_id" and hasattr(m_row, col):
                item[col] = getattr(m_row, col)
        return item

    def _normalize_scores(self, parsed: dict[str, Any]) -> dict[str, Any] | None:
        result = {}
        for dim in self.dimensions:
            prefixed = f"{self.prefix}_{dim}"
            value = parsed.get(prefixed)
            if value is None:
                value = parsed.get(dim)

            if isinstance(value, dict):
                score = value.get("score")
                reasoning = value.get("reasoning")
                result[prefixed] = {
                    "reasoning": reasoning,
                    "score": self._normalize_single_value(score),
                }
            else:
                result[prefixed] = self._normalize_single_value(value)

        return result

    def _normalize_single_value(self, value: Any) -> float | None:
        if self.score_scale == ScoreScale.SCALE_1_5:
            score = coerce_score(value)
            return normalize_1_5_score(score)
        else:
            return coerce_float(value)

    def _load_batch_outputs(self, output_dir: Path) -> pd.DataFrame:
        output_files = list(output_dir.glob("batch_*.jsonl"))
        if not output_files:
            output_files = list(output_dir.glob("*.jsonl"))

        rows = []
        for file_path in output_files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        custom_id = data.get("custom_id", "")
                        response = data.get("response", {})
                        body = response.get("body", {})

                        choices = body.get("choices", [])
                        message = ""
                        if choices:
                            message = choices[0].get("message", {}).get("content", "")

                        rows.append(
                            {
                                "file": file_path.name,
                                "custom_id": custom_id,
                                "model": body.get("model", ""),
                                "status_code": response.get("status_code", 0),
                                "message": message,
                                "error": data.get("error"),
                                "status": "Success"
                                if response.get("status_code") == 200
                                else "Failed",
                            }
                        )
                    except json.JSONDecodeError:
                        continue

        return pd.DataFrame(rows)

    def _load_manifest(self, manifest_path: Path) -> pd.DataFrame:
        rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame(rows)

    def _log_failures(self, failed_items: list[dict]) -> None:
        if not failed_items:
            return

        by_reason: dict[str, list[str]] = {}
        for item in failed_items:
            reason = item["reason"]
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(item["custom_id"])

        logger.warning("Failed items breakdown:")
        for reason, ids in by_reason.items():
            logger.warning("  {}: {} items", reason, len(ids))
            for cid in ids[:5]:
                detail = next(
                    (f["detail"] for f in failed_items if f["custom_id"] == cid),
                    "",
                )
                logger.warning("    - {} | {}", cid, detail[:80])
            if len(ids) > 5:
                logger.warning("    ... and {} more", len(ids) - 5)


def parse_domain_results(
    domain_name: str,
    batch_output_dir: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    config = get_domain(domain_name)
    parser = ResultParser(config)
    return parser.parse_batch_output(batch_output_dir, manifest_path, output_path)
