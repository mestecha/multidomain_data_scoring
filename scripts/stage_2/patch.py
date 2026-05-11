#!/usr/bin/env python3
"""drop prejudicial pair_ids and append new pairs to data/stage_2.jsonl.

  1. Reads data/stage_2.jsonl.
  2. Drops any pair whose pair_id is listed in --drop-ids.
  3. Appends pairs from --append (renumbered to continue the max id).
  4. Archives the prior data/stage_2.jsonl to data/archive/stage_2/<stamp>.jsonl.
  5. Writes the result back to data/stage_2.jsonl.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import click
from loguru import logger

STAGE2_PATH = Path("data/stage_2.jsonl")
ARCHIVE_DIR = Path("data/archive/stage_2")


def _max_pair_id(rows: list[dict]) -> int:
    return max((int(r["pair_id"].rsplit("-", 1)[-1]) for r in rows), default=0)


@click.command()
@click.option("--drop-ids", required=True, type=click.Path(exists=True), help="file with pair_ids to drop, one per line")
@click.option("--append", default=None, type=click.Path(exists=True), help="JSONL with new pairs to append (renumbered)")
@click.option("--stage2", default=str(STAGE2_PATH), type=click.Path(exists=True))
@click.option("--archive-dir", default=str(ARCHIVE_DIR), type=click.Path())
def main(drop_ids: str, append: str | None, stage2: str, archive_dir: str) -> None:
    """drop + append + archive prior."""
    drop = {l.strip() for l in open(drop_ids) if l.strip()}
    logger.info("will drop {} pair_ids", len(drop))

    stage2_p = Path(stage2)
    keep_rows = []
    n_total = n_dropped = 0
    with open(stage2_p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            n_total += 1
            if d["pair_id"] in drop:
                n_dropped += 1
            else:
                keep_rows.append(d)
    logger.info("kept {}/{} pairs ({} dropped)", len(keep_rows), n_total, n_dropped)

    new_rows = []
    if append:
        max_n = _max_pair_id(keep_rows)
        with open(append, encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                d = json.loads(line)
                d["pair_id"] = f"S2D-{max_n + i:06d}"
                new_rows.append(d)
        logger.info("renumbered {} new pairs starting at S2D-{:06d}", len(new_rows), max_n + 1)

    arch = Path(archive_dir)
    arch.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M")
    arch_path = arch / f"{stamp}.jsonl"
    shutil.move(str(stage2_p), str(arch_path))
    logger.info("archived previous → {}", arch_path)

    with open(stage2_p, "w", encoding="utf-8") as f:
        for r in keep_rows + new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.success("wrote {} ({} kept + {} appended = {} total)", stage2_p, len(keep_rows), len(new_rows), len(keep_rows) + len(new_rows))


if __name__ == "__main__":
    main()
