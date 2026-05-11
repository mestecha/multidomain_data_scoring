#!/usr/bin/env python3
"""score the multicultural negative-sampling dialogues with gpt-5.1 sync."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from loguru import logger

from scripts.stage_1.prompts import build_multicultural_prompt

MAIN_PATH = Path("data/input/multicultural/negative_sampling/main.jsonl")
OUT_PATH = Path("data/input/multicultural/negative_sampling/scoring/outputs/main_scored.jsonl")
DIMS = ["mu_cultural_value", "mu_cultural_specificity", "mu_naturalness", "mu_coherence", "mu_empathy"]


def _turns_text(messages: list[dict]) -> str:
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
    )


def _client():
    from dotenv import load_dotenv
    from openai import AzureOpenAI

    load_dotenv()
    return AzureOpenAI(
        api_key=os.environ["AZURE_BATCH_GPT_API_KEY"],
        api_version=os.environ["AZURE_BATCH_GPT_API_VERSION"],
        azure_endpoint=os.environ["AZURE_BATCH_GPT_ENDPOINT"],
    )


def _parse_scores(content: str) -> dict[str, float] | None:
    c = content.strip()
    if c.startswith("```"):
        c = c.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        d = json.loads(c)
    except json.JSONDecodeError:
        return None
    out: dict[str, float] = {}
    for dim in DIMS:
        v = d.get(dim)
        if isinstance(v, dict):
            v = v.get("score")
        if v is None:
            return None
        try:
            out[dim] = round(float(v), 2)
        except (TypeError, ValueError):
            return None
    return out


@click.command()
@click.option("--main-path", default=str(MAIN_PATH), type=click.Path())
@click.option("--out-path", default=str(OUT_PATH), type=click.Path())
@click.option("--deployment", default="gpt-5")
@click.option("--max-workers", default=8)
def main(main_path: str, out_path: str, deployment: str, max_workers: int) -> None:
    """score all negatives with gpt-5/gpt-5.1 sync, parallel."""
    rows = [json.loads(l) for l in open(main_path)]
    logger.info("scoring {} negatives via {}", len(rows), deployment)

    client = _client()
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    def score_one(row: dict) -> dict:
        prompt = build_multicultural_prompt(
            turns_text=_turns_text(row["messages"]),
            statement=row["statement_original"],
            country_1=row["country_1"],
            country_2=row["country_2"],
            enrich_qid=True,
        )
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
            )
            scores = _parse_scores(resp.choices[0].message.content or "")
        except Exception as e:
            logger.warning("{}: {}", row["source_uid"], e)
            scores = None
        return {
            "source_uid": row["source_uid"],
            "country_code": row["country_code"],
            "target_dim": row["target_dim"],
            "scores": scores,
        }

    results: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for r in pool.map(score_one, rows):
            results.append(r)
            done += 1
            if done % 50 == 0:
                logger.info("progress: {}/{}", done, len(rows))

    with open(out_p, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    failed = sum(1 for r in results if r["scores"] is None)
    logger.success("wrote {} ({} ok, {} failed)", out_p, len(results) - failed, failed)


if __name__ == "__main__":
    main()
