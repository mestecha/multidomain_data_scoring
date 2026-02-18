"""retry failed batch entries by extracting, re-sharding, and resubmitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

MAX_ENTRIES_PER_SHARD = 50_000
MAX_SHARD_SIZE_MB = 200
HOLDOUT_THRESHOLD = 100


def extract_failed_ids(parse_result: dict[str, Any]) -> set[str]:
    """get custom_ids from parse failures that are retryable."""
    retryable_reasons = {
        "missing_from_batch",
        "json_parse_error",
        "api_error",
        "score_normalization_error",
        "incomplete_merge",
    }
    failed = parse_result.get("failed_items", [])
    ids: set[str] = set()
    for item in failed:
        if item.get("reason") in retryable_reasons:
            ids.add(item["custom_id"])
    return ids


def _load_shard_entries(shard_dir: Path) -> dict[str, dict]:
    """index all shard entries by custom_id."""
    entries: dict[str, dict] = {}
    for shard_file in sorted(shard_dir.glob("*.jsonl")):
        if shard_file.parent.name == "output":
            continue
        with open(shard_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    cid = entry.get("custom_id", "")
                    if cid:
                        entries[cid] = entry
                except json.JSONDecodeError:
                    continue
    return entries


def build_retry_shards(
    failed_ids: set[str],
    domain: str,
    shard_dir: Path,
    prefix: str = "retry",
) -> list[Path]:
    """create retry shard files from failed custom_ids."""
    all_entries = _load_shard_entries(shard_dir)

    # For commonsense dual-prompt, a base_id failure means we need
    # both the -dim and -dlg entries
    retry_entries: list[dict] = []
    missing_from_shards: list[str] = []

    for cid in sorted(failed_ids):
        # Direct match (standard domains or suffixed commonsense ids)
        if cid in all_entries:
            retry_entries.append(all_entries[cid])
            continue

        # Commonsense incomplete_merge: base_id failed, need both -dim and -dlg
        dim_id = f"{cid}-dim"
        dlg_id = f"{cid}-dlg"
        if dim_id in all_entries and dlg_id in all_entries:
            retry_entries.append(all_entries[dim_id])
            retry_entries.append(all_entries[dlg_id])
            continue

        # Suffixed id like "cs-xxx-dim" — look it up directly
        if cid.endswith(("-dim", "-dlg")) and cid in all_entries:
            retry_entries.append(all_entries[cid])
            continue

        missing_from_shards.append(cid)

    if missing_from_shards:
        logger.warning(
            "{} failed ids not found in original shards: {}",
            len(missing_from_shards),
            missing_from_shards[:10],
        )

    if not retry_entries:
        logger.info("No retry entries to write for {}", domain)
        return []

    # Write retry shards respecting size limits
    retry_dir = shard_dir / "retry"
    retry_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous retry shards
    for old_file in retry_dir.glob(f"{prefix}_*.jsonl"):
        old_file.unlink()

    shard_paths: list[Path] = []
    shard_num = 1
    current_entries: list[str] = []
    current_size = 0

    for entry in retry_entries:
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        line_size = len(line.encode("utf-8"))

        if (
            current_entries
            and (
                len(current_entries) >= MAX_ENTRIES_PER_SHARD
                or current_size + line_size > MAX_SHARD_SIZE_MB * 1024 * 1024
            )
        ):
            path = _write_shard(retry_dir, prefix, shard_num, current_entries)
            shard_paths.append(path)
            shard_num += 1
            current_entries = []
            current_size = 0

        current_entries.append(line)
        current_size += line_size

    if current_entries:
        path = _write_shard(retry_dir, prefix, shard_num, current_entries)
        shard_paths.append(path)

    logger.info(
        "Created {} retry shard(s) with {} entries for {}",
        len(shard_paths),
        len(retry_entries),
        domain,
    )
    return shard_paths


def _write_shard(
    retry_dir: Path, prefix: str, shard_num: int, lines: list[str]
) -> Path:
    path = retry_dir / f"{prefix}_shard_{shard_num:04d}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return path


def merge_retry_results(
    domain: str,
    retry_output_dir: Path,
    main_results_path: Path,
    manifest_path: Path,
    config_name: str,
) -> dict[str, Any]:
    """re-parse retry outputs and merge into main results file."""
    from scripts.stage_1.parse_results import parse_domain_results

    retry_results_path = retry_output_dir / f"{domain}_retry_results.jsonl"

    result = parse_domain_results(
        config_name,
        retry_output_dir,
        manifest_path,
        retry_results_path,
    )

    # Append successfully retried items to main results
    if retry_results_path.exists():
        retried_items: list[str] = []
        with open(retry_results_path, "r", encoding="utf-8") as f:
            retried_items = f.readlines()

        if retried_items:
            with open(main_results_path, "a", encoding="utf-8") as f:
                f.writelines(retried_items)
            logger.success(
                "Appended {} retried items to {}", len(retried_items), main_results_path
            )

        # Clean up temporary retry results
        retry_results_path.unlink(missing_ok=True)

    return result


def check_holdout_threshold(
    all_failures: dict[str, list[dict]],
) -> tuple[bool, int]:
    """check if total persistent failures are under the holdout threshold."""
    total = sum(len(items) for items in all_failures.values())
    return total <= HOLDOUT_THRESHOLD, total
