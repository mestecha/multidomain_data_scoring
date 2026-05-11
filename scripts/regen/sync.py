#!/usr/bin/env python3
"""sync gpt-5.1 generation for malformed-pair regeneration.

Replaces the Azure batch step in the regen pipeline when the batch deployment
is degraded. Reads pre-rendered batch shards, calls chat.completions sync in
parallel, and writes results in batch-output format so the existing
`parse_generation_results` parser consumes them without changes.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from loguru import logger

SHARD_DIR = Path("data/stage_2/regen/shards_gen")
OUT_DIR = Path("data/stage_2/regen/shards_gen/output")
DEFAULT_DEPLOYMENT = "gpt-5.1"


def _client():
    from dotenv import load_dotenv
    from openai import AzureOpenAI

    load_dotenv()
    return AzureOpenAI(
        api_key=os.environ["AZURE_BATCH_GPT_API_KEY"],
        api_version=os.environ["AZURE_BATCH_GPT_API_VERSION"],
        azure_endpoint=os.environ["AZURE_BATCH_GPT_ENDPOINT"],
    )


def _existing_custom_ids(out_dir: Path) -> set[str]:
    """custom_ids already written across all output files (for resume)."""
    seen: set[str] = set()
    for p in out_dir.glob("*.jsonl"):
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["custom_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return seen


@click.command()
@click.option("--shard-dir", default=str(SHARD_DIR), type=click.Path())
@click.option("--out-dir", default=str(OUT_DIR), type=click.Path())
@click.option("--deployment", default=DEFAULT_DEPLOYMENT)
@click.option("--max-workers", default=8)
@click.option("--limit", default=0, help="cap requests (0 = all)")
@click.option("--temperature", default=None, type=float, help="override body.temperature (default: keep shard's value)")
def main(shard_dir: str, out_dir: str, deployment: str, max_workers: int, limit: int, temperature: float | None) -> None:
    """run sync generation on every shard line, emit one output file per shard."""
    s_dir = Path(shard_dir)
    o_dir = Path(out_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(s_dir.glob("*.jsonl"))
    if not shards:
        logger.error("no shards in {}", s_dir)
        return

    done = _existing_custom_ids(o_dir)
    logger.info("found {} already-written custom_ids; will skip those", len(done))

    client = _client()

    def run_one(row: dict) -> dict:
        custom_id = row["custom_id"]
        body = dict(row["body"])
        body["model"] = deployment
        if temperature is not None:
            body["temperature"] = temperature
        try:
            resp = client.chat.completions.create(**body)
            return {
                "custom_id": custom_id,
                "response": {"status_code": 200, "body": resp.model_dump()},
                "error": None,
            }
        except Exception as e:
            logger.warning("{}: {}", custom_id, e)
            return {
                "custom_id": custom_id,
                "response": None,
                "error": {"message": str(e), "type": type(e).__name__},
            }

    total_rows = 0
    for shard in shards:
        rows: list[dict] = []
        with open(shard, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["custom_id"] in done:
                    continue
                rows.append(r)
                if limit and len(rows) >= limit:
                    break
        if not rows:
            logger.info("{}: all rows already done", shard.name)
            continue

        out_path = o_dir / shard.name
        logger.info("processing {} ({} rows pending → {})", shard.name, len(rows), out_path.name)

        n_ok = 0
        n_err = 0
        with open(out_path, "a", encoding="utf-8") as out_f, ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i, result in enumerate(pool.map(run_one, rows), start=1):
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                if result["error"] is None:
                    n_ok += 1
                else:
                    n_err += 1
                if i % 50 == 0:
                    logger.info("  {}: {}/{} (ok={} err={})", shard.name, i, len(rows), n_ok, n_err)
        logger.success("{}: done — ok={} err={}", shard.name, n_ok, n_err)
        total_rows += len(rows)
        if limit:
            break

    logger.success("regen sync complete — processed {} rows total", total_rows)


if __name__ == "__main__":
    main()
