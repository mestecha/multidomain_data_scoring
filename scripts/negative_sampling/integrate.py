#!/usr/bin/env python3
"""integrate the 1,400 self-taught negatives into stage_1.jsonl.

Reads:
  - data/input/multicultural/negative_sampling/main.jsonl              (seeds + Wang outputs)
  - data/input/multicultural/negative_sampling/scoring/outputs/main_scored.jsonl  (gpt-5.1 scores)
  - data/input/multicultural/countries/train/<COUNTRY>.csv              (enriched metadata)
  - data/stage_1.jsonl                                                  (for max existing id)

Writes:
  - data/stage_1_with_negatives.jsonl  (current stage_1 + N appended negatives)

Schema decisions:
  - dialogue_id: continues from current max (S1D-051265+), sequential, deterministic order
  - source_id:   mu-NS-<COUNTRY>-<NNNNNN>, per-country sequential counter
  - domain:      multicultural
  - split:       train
  - scores:      5 mu_* from main_scored, other 18 dims set to None
  - domain_metadata: 15 fields from train CSV row + nested `negative_sampling` block
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import click
from loguru import logger

MAIN_PATH = Path("data/input/multicultural/negative_sampling/main.jsonl")
SCORED_PATH = Path("data/input/multicultural/negative_sampling/scoring/outputs/main_scored.jsonl")
COUNTRIES_DIR = Path("data/input/multicultural/countries/train")
STAGE1_PATH = Path("data/stage_1.jsonl")
OUT_PATH = Path("data/stage_1_with_negatives.jsonl")

ALL_DIMS = (
    "co_topic_coherence", "co_discourse_structure", "co_logical_consistency",
    "co_temporal_causal_coherence", "co_mutual_grounding", "co_overall_coherence_score",
    "em_emotional_awareness", "em_emotional_validation", "em_perspective_taking",
    "em_supportive_engagement", "em_helpful_response", "em_overall_empathy_score",
    "cs_causality", "cs_consistency", "cs_reaction", "cs_desire", "cs_coherence", "cs_empathy",
    "mu_cultural_value", "mu_cultural_specificity", "mu_naturalness", "mu_coherence", "mu_empathy",
)
MU_DIMS = ("mu_cultural_value", "mu_cultural_specificity", "mu_naturalness", "mu_coherence", "mu_empathy")
METADATA_FIELDS = (
    "country_1", "country_2", "demographics_1", "demographics_2",
    "statement_original", "statement_cultural", "situation",
    "cultural_reasoning_1", "cultural_reasoning_2",
    "arousal_reasoning", "arousal_score",
    "social_norms_1", "social_norms_2",
    "prejudices_1", "prejudices_2",
)
TARGET_DIM_FULL = {
    "cv": "mu_cultural_value",
    "cs": "mu_cultural_specificity",
    "nat": "mu_naturalness",
    "coh": "mu_coherence",
}


def _load_scores(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("scores") is None:
                continue
            out[(d["source_uid"], d["target_dim"])] = d["scores"]
    return out


def _load_negatives(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _load_country_rows(countries_dir: Path) -> dict[str, dict[str, dict]]:
    """returns {country_code: {uid: row}}."""
    out: dict[str, dict[str, dict]] = {}
    for csv_path in sorted(countries_dir.glob("*.csv")):
        code = csv_path.stem
        rows: dict[str, dict] = {}
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["uid"]] = row
        out[code] = rows
    return out


def _max_dialogue_id(path: Path) -> int:
    max_n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            n = int(d["dialogue_id"].rsplit("-", 1)[-1])
            if n > max_n:
                max_n = n
    return max_n


def _build_row(
    neg: dict,
    scores: dict[str, float],
    csv_row: dict,
    dialogue_id: str,
    source_id: str,
) -> dict:
    metadata = {k: csv_row.get(k, "") for k in METADATA_FIELDS}
    metadata["negative_sampling"] = {
        "method": "wang-self-taught-evaluator-2024",
        "generator_model": "claude-sonnet-4-7",
        "target_dim": TARGET_DIM_FULL[neg["target_dim"]],
        "modified_instruction": neg["modified_instruction"],
        "source_uid": neg["source_uid"],
    }
    return {
        "dialogue_id": dialogue_id,
        "source_id": source_id,
        "domain": "multicultural",
        "messages": neg["messages"],
        "split": "train",
        "scores": {dim: scores.get(dim) if dim in MU_DIMS else None for dim in ALL_DIMS},
        "domain_metadata": metadata,
    }


@click.command()
@click.option("--main-path", default=str(MAIN_PATH), type=click.Path(exists=True))
@click.option("--scored-path", default=str(SCORED_PATH), type=click.Path(exists=True))
@click.option("--countries-dir", default=str(COUNTRIES_DIR), type=click.Path(exists=True))
@click.option("--stage1", default=str(STAGE1_PATH), type=click.Path(exists=True))
@click.option("--out", default=str(OUT_PATH), type=click.Path())
def main(main_path: str, scored_path: str, countries_dir: str, stage1: str, out: str) -> None:
    """append scored negatives to stage_1; emit data/stage_1_with_negatives.jsonl."""
    negatives = _load_negatives(Path(main_path))
    logger.info("loaded {} negatives", len(negatives))

    scores = _load_scores(Path(scored_path))
    logger.info("loaded {} scored negatives", len(scores))

    country_rows = _load_country_rows(Path(countries_dir))
    logger.info("loaded country rows for {} countries", len(country_rows))

    start_id = _max_dialogue_id(Path(stage1)) + 1
    logger.info("next dialogue_id starts at S1D-{:06d}", start_id)

    negatives.sort(key=lambda n: (n["country_code"], n["target_dim"], n["source_uid"]))

    new_rows: list[dict] = []
    per_country_counter: dict[str, int] = {}
    n_missing_scores = 0
    n_missing_csv = 0

    for neg in negatives:
        sc = scores.get((neg["source_uid"], neg["target_dim"]))
        if sc is None:
            n_missing_scores += 1
            continue
        country = neg["country_code"]
        csv_row = country_rows.get(country, {}).get(neg["source_uid"])
        if csv_row is None:
            n_missing_csv += 1
            continue
        per_country_counter[country] = per_country_counter.get(country, 0) + 1
        dialogue_id = f"S1D-{start_id + len(new_rows):06d}"
        source_id = f"mu-NS-{country}-{per_country_counter[country]:06d}"
        new_rows.append(_build_row(neg, sc, csv_row, dialogue_id, source_id))

    logger.info(
        "built {} new rows (skipped {} no-score, {} no-csv)",
        len(new_rows), n_missing_scores, n_missing_csv,
    )

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    n_existing = 0
    with open(stage1, encoding="utf-8") as fin, open(out_p, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line)
            n_existing += 1
        for row in new_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.success(
        "wrote {} ({} existing + {} negatives)",
        out_p, n_existing, len(new_rows),
    )


if __name__ == "__main__":
    main()
