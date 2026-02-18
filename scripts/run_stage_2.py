#!/usr/bin/env python3
"""stage 2 pipeline orchestrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from scripts.models import DomainName, Stage1Entry

DOMAIN_NAMES = [d.value for d in DomainName]
STEPS = ["select", "generate", "eval", "pairs", "all"]

# ── Default paths ─────────────────────────────────────────────────────────

STAGE_1_PATH = Path("data/stage_1.jsonl")
STAGE_2_DIR = Path("data/stage_2")
CANDIDATES_PATH = STAGE_2_DIR / "candidates.jsonl"
GEN_SHARD_DIR = STAGE_2_DIR / "shards"
MANIFEST_GEN_PATH = STAGE_2_DIR / "manifest_gen.jsonl"
GEN_OUTPUT_DIR = STAGE_2_DIR / "shards" / "output"
EVAL_SHARD_DIR = STAGE_2_DIR / "shards_eval"
MANIFEST_EVAL_PATH = STAGE_2_DIR / "manifest_eval.jsonl"
EVAL_OUTPUT_DIR = STAGE_2_DIR / "shards_eval" / "output"
PAIRS_PATH = STAGE_2_DIR / "pairs.jsonl"


def _load_originals(path: Path) -> dict[str, Stage1Entry]:
    from scripts.stage_2.select import load_stage1_entries

    entries = load_stage1_entries(path)
    return {e.dialogue_id: e for e in entries}


# ── Step implementations ──────────────────────────────────────────────────


def step_select(
    domain: str | None,
    max_samples: int | None,
    input_path: Path = STAGE_1_PATH,
    output_path: Path = CANDIDATES_PATH,
) -> None:
    logger.info("=" * 60)
    logger.info("STEP: SELECT - Choosing candidates from Stage 1 train data")
    logger.info("=" * 60)

    from scripts.stage_2.select import (
        load_stage1_entries,
        select_candidates,
        write_candidates,
    )

    entries = load_stage1_entries(input_path)
    logger.info("Loaded {} Stage 1 entries", len(entries))

    domain_filter = DomainName(domain) if domain else None
    candidates = select_candidates(
        entries, domain=domain_filter, max_samples=max_samples
    )
    write_candidates(candidates, output_path)


def step_generate(
    output_shard_dir: Path = GEN_SHARD_DIR,
    manifest_path: Path = MANIFEST_GEN_PATH,
    candidates_path: Path = CANDIDATES_PATH,
) -> None:
    logger.info("=" * 60)
    logger.info("STEP: GENERATE - Building generation batch shards")
    logger.info("=" * 60)

    from scripts.stage_2.generate import (
        build_generation_entries,
        write_manifest,
        write_shards,
    )
    from scripts.models import Stage2Candidate

    # load candidates
    candidates: list[Stage2Candidate] = []
    with open(candidates_path, encoding="utf-8") as f:
        for line in f:
            candidates.append(Stage2Candidate.model_validate_json(line.strip()))

    logger.info("Loaded {} candidates", len(candidates))

    batch_entries, manifest_items = build_generation_entries(candidates)
    write_shards(batch_entries, output_shard_dir)
    write_manifest(manifest_items, manifest_path)

    logger.info(
        "Generation shards ready at {}. Run batch_runner to submit.",
        output_shard_dir,
    )


def step_eval(
    gen_output_dir: Path = GEN_OUTPUT_DIR,
    gen_manifest_path: Path = MANIFEST_GEN_PATH,
    eval_shard_dir: Path = EVAL_SHARD_DIR,
    eval_manifest_path: Path = MANIFEST_EVAL_PATH,
    stage1_path: Path = STAGE_1_PATH,
) -> None:
    logger.info("=" * 60)
    logger.info("STEP: EVAL - Building evaluation batch shards")
    logger.info("=" * 60)

    from scripts.stage_2.eval import build_eval_entries
    from scripts.stage_2.generate import parse_generation_results, write_shards

    # parse generation outputs
    variants = parse_generation_results(gen_output_dir, gen_manifest_path)
    logger.info("Parsed {} variants from generation", len(variants))

    # load originals
    originals = _load_originals(stage1_path)

    # build eval entries
    batch_entries, manifest_items = build_eval_entries(variants, originals)

    # write eval shards
    from scripts.stage_2.generate import write_manifest

    write_shards(batch_entries, eval_shard_dir)
    write_manifest(manifest_items, eval_manifest_path)

    logger.info(
        "Eval shards ready at {}. Run batch_runner to submit.",
        eval_shard_dir,
    )


def step_pairs(
    margin_threshold: float = 0.05,
    gen_output_dir: Path = GEN_OUTPUT_DIR,
    gen_manifest_path: Path = MANIFEST_GEN_PATH,
    eval_output_dir: Path = EVAL_OUTPUT_DIR,
    eval_manifest_path: Path = MANIFEST_EVAL_PATH,
    stage1_path: Path = STAGE_1_PATH,
    output_path: Path = PAIRS_PATH,
) -> None:
    logger.info("=" * 60)
    logger.info("STEP: PAIRS - Building contrastive pairs")
    logger.info("=" * 60)

    from scripts.stage_2.eval import parse_eval_results
    from scripts.stage_2.generate import parse_generation_results
    from scripts.stage_2.pairs import build_pairs, write_pairs

    # parse generation outputs
    variants = parse_generation_results(gen_output_dir, gen_manifest_path)

    # parse eval results
    results = parse_eval_results(
        eval_output_dir, eval_manifest_path, margin_threshold=margin_threshold
    )

    # load originals
    originals = _load_originals(stage1_path)

    # build pairs
    pairs = build_pairs(variants, results, originals)
    write_pairs(pairs, output_path)


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 Contrastive Pair Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--step",
        choices=STEPS,
        required=True,
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--domain",
        choices=["all"] + DOMAIN_NAMES,
        default="all",
        help="Domain(s) to process (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max candidates to select (select step only)",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.05,
        help="Minimum eval margin for pass (default: 0.05)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # configure logging
    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    domain = None if args.domain == "all" else args.domain

    logger.info("=" * 60)
    logger.info("STAGE 2 CONTRASTIVE PAIR GENERATION PIPELINE")
    logger.info("=" * 60)
    logger.info("Step: {}", args.step)
    logger.info("Domain: {}", args.domain)
    if args.max_samples:
        logger.info("Max samples: {}", args.max_samples)

    if args.step == "select":
        step_select(domain=domain, max_samples=args.max_samples)

    elif args.step == "generate":
        step_generate()

    elif args.step == "eval":
        step_eval()

    elif args.step == "pairs":
        step_pairs(margin_threshold=args.margin_threshold)

    elif args.step == "all":
        step_select(domain=domain, max_samples=args.max_samples)
        step_generate()
        # note: generate and eval steps produce shards for the
        # batch_runner. In a real run, you would submit shards between steps.
        logger.info(
            "Shards written. Submit generation shards via batch_runner, "
            "then re-run with --step eval, then --step pairs."
        )

    logger.info("=" * 60)
    logger.success("STEP '{}' COMPLETE", args.step)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
