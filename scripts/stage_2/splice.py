#!/usr/bin/env python3
"""splice main + regen pairs into the new canonical stage_2.jsonl.

  1. Reads data/stage_2/main.jsonl   (the surviving pre-regen pairs).
  2. Reads data/stage_2/regen/main.jsonl (regen pairs; pair_ids start at 1).
  3. Renumbers regen pair_ids to follow the max id in main (no collision).
  4. Archives the prior data/stage_2.jsonl to data/archive/stage_2/<stamp>.jsonl.
  5. Writes the combined stream to data/stage_2.jsonl.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import click
from loguru import logger

MAIN_PATH = Path("data/stage_2/main.jsonl")
REGEN_PAIRS_PATH = Path("data/stage_2/regen/main.jsonl")
STAGE2_PATH = Path("data/stage_2.jsonl")
ARCHIVE_DIR = Path("data/archive/stage_2")


def _max_pair_id(path: Path) -> int:
    max_n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            n = int(json.loads(line)["pair_id"].rsplit("-", 1)[-1])
            if n > max_n:
                max_n = n
    return max_n


@click.command()
@click.option("--main-path", default=str(MAIN_PATH), type=click.Path(exists=True))
@click.option("--regen", default=str(REGEN_PAIRS_PATH), type=click.Path(exists=True))
@click.option("--out", default=str(STAGE2_PATH), type=click.Path())
@click.option("--archive-dir", default=str(ARCHIVE_DIR), type=click.Path())
def main(main_path: str, regen: str, out: str, archive_dir: str) -> None:
    """combine + renumber → new stage_2.jsonl, archive old."""
    out_p = Path(out)
    arch = Path(archive_dir)
    arch.mkdir(parents=True, exist_ok=True)

    if out_p.exists():
        stamp = datetime.now().strftime("%Y%m%dT%H%M")
        arch_path = arch / f"{stamp}.jsonl"
        shutil.move(str(out_p), str(arch_path))
        logger.info("archived previous {} → {}", out_p, arch_path)

    max_main = _max_pair_id(Path(main_path))
    logger.info("max pair_id in main: S2D-{:06d}", max_main)

    n_main = 0
    n_regen = 0
    with open(out_p, "w", encoding="utf-8") as fout:
        with open(main_path, encoding="utf-8") as fin:
            for line in fin:
                fout.write(line)
                n_main += 1
        with open(regen, encoding="utf-8") as fin:
            for line in fin:
                d = json.loads(line)
                d["pair_id"] = f"S2D-{max_main + n_regen + 1:06d}"
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_regen += 1

    logger.success(
        "wrote {} ({} main + {} regen = {} total)",
        out_p, n_main, n_regen, n_main + n_regen,
    )


if __name__ == "__main__":
    main()
