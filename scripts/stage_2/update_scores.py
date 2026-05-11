#!/usr/bin/env python3
"""drop malformed pairs from stage_2.jsonl and refresh mu_* stage_1_scores.

Two things at once:
  1. Filter out the 8,214 pairs listed in data/audits/qa/malformed_pairs.jsonl
     (they will be replaced by the regen output downstream).
  2. For every remaining multicultural pair, replace evaluation.stage_1_scores
     with the current values from data/stage_1.jsonl (which reflect the v2
     multicultural rescore). Non-multicultural pairs pass through unchanged.

Output: data/stage_2/clean.jsonl (one row per kept pair).
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from loguru import logger

STAGE2_PATH = Path("data/stage_2.jsonl")
STAGE1_PATH = Path("data/stage_1.jsonl")
MALFORMED_PATH = Path("data/audits/qa/malformed_pairs.jsonl")
OUT_PATH = Path("data/stage_2/clean.jsonl")
MU_DIMS = ("mu_cultural_value", "mu_cultural_specificity", "mu_naturalness", "mu_coherence", "mu_empathy")


def _load_malformed_pair_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            ids.add(json.loads(line)["pair_id"])
    return ids


def _load_mu_scores(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("domain") != "multicultural":
                continue
            scores = d.get("scores") or {}
            out[d["dialogue_id"]] = {k: scores.get(k) for k in MU_DIMS}
    return out


@click.command()
@click.option("--stage2", default=str(STAGE2_PATH), type=click.Path(exists=True))
@click.option("--stage1", default=str(STAGE1_PATH), type=click.Path(exists=True))
@click.option("--malformed", default=str(MALFORMED_PATH), type=click.Path(exists=True))
@click.option("--out", default=str(OUT_PATH), type=click.Path())
def main(stage2: str, stage1: str, malformed: str, out: str) -> None:
    """filter malformed + refresh mu scores → clean.jsonl."""
    malformed_ids = _load_malformed_pair_ids(Path(malformed))
    logger.info("loaded {} malformed pair_ids", len(malformed_ids))

    mu_scores = _load_mu_scores(Path(stage1))
    logger.info("loaded mu_scores for {} stage_1 dialogues", len(mu_scores))

    n_total = n_dropped = n_mu_changed = n_mu_unchanged = n_mu_missing = n_kept = 0
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stage2, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            n_total += 1
            d = json.loads(line)
            if d["pair_id"] in malformed_ids:
                n_dropped += 1
                continue

            if d["metadata"].get("domain") == "multicultural":
                src = d["source_dialogue_id"]
                fresh = mu_scores.get(src)
                if fresh is None:
                    n_mu_missing += 1
                else:
                    old = d["evaluation"].get("stage_1_scores") or {}
                    if all(old.get(k) == fresh.get(k) for k in MU_DIMS):
                        n_mu_unchanged += 1
                    else:
                        d["evaluation"]["stage_1_scores"] = fresh
                        n_mu_changed += 1

            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_kept += 1

    logger.success(
        "wrote {} ({} kept of {} total; dropped {} malformed; mu changed {}, unchanged {}, missing-src {})",
        out_path, n_kept, n_total, n_dropped, n_mu_changed, n_mu_unchanged, n_mu_missing,
    )


if __name__ == "__main__":
    main()
