#!/usr/bin/env python3
"""post-regen orchestrator (no API calls).

Two subcommands closing the regen pipeline:

  parse-gen
    Reads sync regen outputs (data/stage_2/regen/shards_gen/output/*.jsonl),
    parses them into Stage2Variants, writes variants.jsonl, and renders the
    eval batch shards + manifest at data/stage_2/regen/shards_eval/.

  parse-eval-and-build
    Reads sync eval outputs (data/stage_2/regen/shards_eval/output/*.jsonl),
    parses EvalResults, runs build_pairs against variants + originals, and
    writes data/stage_2/regen/pairs.jsonl (one Stage2Pair per line).

Run order:
  1. uv run python -m scripts.finalize_regen parse-gen
  2. uv run python -m scripts.regen_malformed_sync \\
        --shard-dir data/stage_2/regen/shards_eval \\
        --out-dir   data/stage_2/regen/shards_eval/output
  3. uv run python -m scripts.finalize_regen parse-eval-and-build
  4. uv run python -m scripts.splice_stage2

Pair IDs assigned here start at S2D-000001 inside the regen file; the splice
step renumbers them to live alongside the clean.jsonl pairs.
"""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger

from scripts.stage_2.eval import build_eval_entries, parse_eval_results
from scripts.stage_2.generate import (
    parse_generation_results,
    write_manifest,
    write_shards,
)
from scripts.stage_2.pairs import build_pairs

REGEN_DIR = Path("data/stage_2/regen")
GEN_OUT_DIR = REGEN_DIR / "shards_gen" / "output"
GEN_MANIFEST = REGEN_DIR / "manifest_gen.jsonl"
VARIANTS_PATH = REGEN_DIR / "variants.jsonl"
EVAL_SHARD_DIR = REGEN_DIR / "shards_eval"
EVAL_MANIFEST = REGEN_DIR / "manifest_eval.jsonl"
EVAL_OUT_DIR = EVAL_SHARD_DIR / "output"
PAIRS_PATH = REGEN_DIR / "main.jsonl"
STAGE1_PATH = Path("data/stage_1.jsonl")


def _load_originals(path: Path) -> dict:
    from scripts.stage_2.select import load_stage1_entries

    entries = load_stage1_entries(path)
    return {e.dialogue_id: e for e in entries}


@click.group()
def cli() -> None:
    """post-regen finalize commands."""


@cli.command("parse-gen")
@click.option("--gen-out", default=str(GEN_OUT_DIR), type=click.Path(exists=True))
@click.option("--gen-manifest", default=str(GEN_MANIFEST), type=click.Path(exists=True))
@click.option("--variants", default=str(VARIANTS_PATH), type=click.Path())
@click.option("--stage1", default=str(STAGE1_PATH), type=click.Path(exists=True))
@click.option("--eval-shards", default=str(EVAL_SHARD_DIR), type=click.Path())
@click.option("--eval-manifest", default=str(EVAL_MANIFEST), type=click.Path())
def parse_gen(
    gen_out: str, gen_manifest: str, variants: str,
    stage1: str, eval_shards: str, eval_manifest: str,
) -> None:
    """parse regen outputs → variants.jsonl + eval shards."""
    parsed = parse_generation_results(Path(gen_out), Path(gen_manifest))
    logger.info("parsed {} variants from regen outputs", len(parsed))

    v_path = Path(variants)
    v_path.parent.mkdir(parents=True, exist_ok=True)
    with open(v_path, "w", encoding="utf-8") as f:
        for v in parsed:
            f.write(v.model_dump_json() + "\n")
    logger.success("wrote {}", v_path)

    originals = _load_originals(Path(stage1))
    logger.info("loaded {} originals from {}", len(originals), stage1)

    batch_entries, manifest_items = build_eval_entries(parsed, originals)
    write_shards(batch_entries, Path(eval_shards))
    write_manifest(manifest_items, Path(eval_manifest))


@cli.command("parse-eval-and-build")
@click.option("--variants", default=str(VARIANTS_PATH), type=click.Path(exists=True))
@click.option("--eval-out", default=str(EVAL_OUT_DIR), type=click.Path(exists=True))
@click.option("--eval-manifest", default=str(EVAL_MANIFEST), type=click.Path(exists=True))
@click.option("--stage1", default=str(STAGE1_PATH), type=click.Path(exists=True))
@click.option("--pairs", default=str(PAIRS_PATH), type=click.Path())
def parse_eval_and_build(
    variants: str, eval_out: str, eval_manifest: str, stage1: str, pairs: str,
) -> None:
    """parse eval outputs → EvalResults; run build_pairs → pairs.jsonl."""
    from scripts.models import Stage2Variant

    variant_list: list[Stage2Variant] = []
    with open(variants, encoding="utf-8") as f:
        for line in f:
            variant_list.append(Stage2Variant.model_validate_json(line.strip()))
    logger.info("loaded {} variants from {}", len(variant_list), variants)

    results = parse_eval_results(Path(eval_out), Path(eval_manifest))
    logger.info("parsed {} eval results", len(results))

    originals = _load_originals(Path(stage1))
    logger.info("loaded {} originals", len(originals))

    built = build_pairs(variant_list, results, originals)
    logger.info("built {} regen pairs", len(built))

    p_path = Path(pairs)
    p_path.parent.mkdir(parents=True, exist_ok=True)
    with open(p_path, "w", encoding="utf-8") as f:
        for p in built:
            f.write(p.model_dump_json() + "\n")
    logger.success("wrote {} ({} pairs)", p_path, len(built))


if __name__ == "__main__":
    cli()
