"""render wang-style prompts per country/dim into chunk files for subagents."""

from __future__ import annotations

import json
from pathlib import Path

import click
from loguru import logger

from scripts.negative_sampling import seeds
from scripts.negative_sampling.wang_prompts import build as wang_build

INPUTS_DIR = Path("data/input/multicultural/negative_sampling/inputs")
DEFAULT_PER_DIM = 50
DIMS = ("cv", "cs", "nat", "coh")


@click.command()
@click.option("--country", default="all", help="country code (AUS, CHL, ...) or 'all'")
@click.option("--per-dim", default=DEFAULT_PER_DIM, help="dialogues per target dim")
@click.option("--seed", default=42)
def main(country: str, per_dim: int, seed: int) -> None:
    """render wang prompts → INPUTS_DIR; one chunk per (country, dim)."""
    countries = [country] if country != "all" else list(seeds.COUNTRY_FILES)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for code in countries:
        for dim in DIMS:
            sample_rows = seeds.sample(code, per_dim, seed=seed + hash(dim) % 1000)
            path = INPUTS_DIR / f"{code}_{dim}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for row in sample_rows:
                    prompt = wang_build(dim, row, row.get("conversation", ""))
                    f.write(json.dumps({
                        "source_uid": row.get("uid"),
                        "country_code": code,
                        "country_full": seeds.COUNTRY_FILES[code],
                        "target_dim": dim,
                        "prompt": prompt,
                        "country_1": row.get("country_1"),
                        "country_2": row.get("country_2"),
                        "statement_original": row.get("statement_original"),
                    }, ensure_ascii=False) + "\n")
            total += 1

    logger.success("wrote {} chunks → {}", total, INPUTS_DIR)


if __name__ == "__main__":
    main()
