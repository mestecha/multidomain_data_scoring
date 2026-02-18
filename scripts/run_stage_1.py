#!/usr/bin/env python3
"""stage 1 pipeline orchestrator: prepare, run, parse, merge, split."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from scripts.config import DOMAINS
from scripts.models import DomainName

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DOMAIN_NAMES = [d.value for d in DomainName]
STEPS = ["prepare", "run", "parse", "retry", "merge", "split", "all"]


def get_processor(domain: str, max_samples: int | None = None) -> Any:
    match domain:
        case "coherence":
            from scripts.stage_1.prepare_coherence import CoherenceProcessor

            return CoherenceProcessor(max_samples=max_samples)
        case "empathy":
            from scripts.stage_1.prepare_empathy import EmpathyProcessor

            return EmpathyProcessor(max_samples=max_samples)
        case "multicultural":
            from scripts.stage_1.prepare_multicultural import MulticulturalProcessor

            return MulticulturalProcessor(max_samples=max_samples)
        case "commonsense":
            from scripts.stage_1.prepare_commonsense import CommonsenseProcessor

            return CommonsenseProcessor(max_samples=max_samples)
        case _:
            raise ValueError(f"Unknown domain: {domain}")


def step_prepare(domains: list[str], max_samples: int | None = None) -> None:
    logger.info("=" * 60)
    logger.info("STEP: PREPARE - Generating batch API shards")
    if max_samples:
        logger.info("  (limited to {} samples per domain)", max_samples)
    logger.info("=" * 60)

    for domain in domains:
        try:
            logger.debug("Loading {} data...", domain)
            processor = get_processor(domain, max_samples=max_samples)
            processor.prepare()
        except Exception as e:
            logger.error("ERROR preparing {}: {}", domain, e)
            raise


def step_run(
    domains: list[str],
    max_concurrent: int = 5,
    max_file_size_mb: int = 200,
) -> None:
    logger.info("=" * 60)
    logger.info("STEP: RUN - Executing batch API calls")
    logger.info("=" * 60)

    try:
        from scripts.batch_runner import BatchPool, BatchPoolConfig, ShardJob
    except ImportError:
        logger.error("Could not import batch_runner")
        logger.error("Make sure you're running from the project root.")
        return

    all_jobs: list[ShardJob] = []

    for domain in domains:
        input_path = Path("data") / "input" / domain / "shards"

        if not input_path.exists():
            logger.warning("SKIP: No shards found at {}", input_path)
            continue

        output_path = input_path / "output"
        output_path.mkdir(parents=True, exist_ok=True)

        for item in input_path.iterdir():
            if item.is_file() and item.suffix == ".jsonl":
                job = ShardJob(file_path=item, domain=domain)
                all_jobs.append(job)
                logger.debug("Found shard: {} (domain: {})", item.name, domain)

    if not all_jobs:
        logger.warning("No shards found across any domain. Exiting.")
        return

    logger.info(
        "Found {} total shards across {} domain(s)", len(all_jobs), len(domains)
    )

    config = BatchPoolConfig(
        max_concurrent_jobs=max_concurrent,
        max_file_size_mb=max_file_size_mb,
    )
    pool = BatchPool(config=config)

    async def run_all() -> None:
        await pool.run(all_jobs)
        for domain in domains:
            output_path = Path("data") / "input" / domain / "shards" / "output"
            domain_jobs = [j for j in all_jobs if j.domain == domain]
            await pool.save_results(domain_jobs, output_path)

    try:
        asyncio.run(run_all())

        for domain in domains:
            domain_jobs = [j for j in all_jobs if j.domain == domain]
            completed = sum(1 for j in domain_jobs if j.status.value == "completed")
            failed = sum(
                1 for j in domain_jobs if j.status.value in ("failed", "file_failed")
            )
            logger.info("{}: {} completed, {} failed", domain, completed, failed)

        logger.success("All batch jobs processed")

    except Exception as e:
        logger.error("ERROR: {}", e)
        raise


def step_parse(domains: list[str]) -> None:
    logger.info("=" * 60)
    logger.info("STEP: PARSE - Converting batch outputs to domain files")
    logger.info("=" * 60)

    from scripts.stage_1.parse_results import parse_domain_results

    for domain in domains:
        logger.info("--- Parsing {} results ---", domain)

        config = DOMAINS[DomainName(domain)]
        prefix = config.prefix
        batch_output_dir = Path("data") / "input" / domain / "shards" / "output"
        manifest_path = Path("data") / "input" / domain / "manifest.jsonl"
        output_path = Path("data") / "output" / domain / f"{prefix}_results.jsonl"

        if not batch_output_dir.exists():
            logger.warning("SKIP: No batch outputs at {}", batch_output_dir)
            continue

        if not manifest_path.exists():
            logger.warning("SKIP: No manifest at {}", manifest_path)
            continue

        try:
            result = parse_domain_results(
                domain, batch_output_dir, manifest_path, output_path
            )
            logger.success(
                "Parsed {} items, failed {}", result["parsed"], result["failed"]
            )
        except Exception as e:
            logger.error("ERROR: {}", e)
            raise


def step_retry(
    domains: list[str],
    max_concurrent: int = 5,
    max_retries: int = 3,
) -> None:
    logger.info("=" * 60)
    logger.info("STEP: RETRY - Reprocessing failed parse entries")
    logger.info("=" * 60)

    from scripts.stage_1.parse_results import parse_domain_results
    from scripts.stage_1.retry import (
        build_retry_shards,
        check_holdout_threshold,
        extract_failed_ids,
        merge_retry_results,
    )

    for attempt in range(1, max_retries + 1):
        logger.info("--- Retry attempt {}/{} ---", attempt, max_retries)

        # Step 1: parse all domains to find current failures
        all_failures: dict[str, list[dict]] = {}
        domains_to_retry: list[str] = []

        for domain in domains:
            config = DOMAINS[DomainName(domain)]
            prefix = config.prefix
            batch_output_dir = Path("data") / "input" / domain / "shards" / "output"
            manifest_path = Path("data") / "input" / domain / "manifest.jsonl"
            output_path = Path("data") / "output" / domain / f"{prefix}_results.jsonl"

            if not batch_output_dir.exists() or not manifest_path.exists():
                continue

            result = parse_domain_results(
                domain, batch_output_dir, manifest_path, output_path
            )
            failed_ids = extract_failed_ids(result)
            if failed_ids:
                all_failures[domain] = result.get("failed_items", [])
                domains_to_retry.append(domain)
                logger.info("{}: {} entries to retry", domain, len(failed_ids))

        if not domains_to_retry:
            logger.success("No failures to retry across any domain")
            return

        under_threshold, total = check_holdout_threshold(all_failures)
        if under_threshold:
            logger.success(
                "Total failures ({}) under holdout threshold. Holding out.",
                total,
            )
            _write_holdout_log(all_failures)
            return

        # Step 2: build retry shards
        try:
            from scripts.batch_runner import BatchPool, BatchPoolConfig, ShardJob
        except ImportError:
            logger.error("Could not import batch_runner")
            return

        all_jobs: list[ShardJob] = []
        for domain in domains_to_retry:
            config = DOMAINS[DomainName(domain)]
            shard_dir = Path("data") / "input" / domain / "shards"
            result = parse_domain_results(
                domain,
                shard_dir / "output",
                Path("data") / "input" / domain / "manifest.jsonl",
                Path("data") / "output" / domain / f"{config.prefix}_results.jsonl",
            )
            failed_ids = extract_failed_ids(result)
            retry_paths = build_retry_shards(
                failed_ids, domain, shard_dir, prefix=f"retry_{attempt}"
            )
            for path in retry_paths:
                all_jobs.append(ShardJob(file_path=path, domain=domain))

        if not all_jobs:
            logger.info("No retry shards created")
            return

        # Step 3: submit retry shards
        pool_config = BatchPoolConfig(max_concurrent_jobs=max_concurrent)
        pool = BatchPool(config=pool_config)

        async def run_retries() -> None:
            await pool.run(all_jobs)
            for domain in domains_to_retry:
                retry_output = (
                    Path("data") / "input" / domain / "shards" / "retry" / "output"
                )
                retry_output.mkdir(parents=True, exist_ok=True)
                domain_jobs = [j for j in all_jobs if j.domain == domain]
                await pool.save_results(domain_jobs, retry_output)

        asyncio.run(run_retries())

        # Step 4: merge retry results back into main output
        for domain in domains_to_retry:
            config = DOMAINS[DomainName(domain)]
            retry_output = (
                Path("data") / "input" / domain / "shards" / "retry" / "output"
            )
            main_results = (
                Path("data") / "output" / domain / f"{config.prefix}_results.jsonl"
            )
            manifest = Path("data") / "input" / domain / "manifest.jsonl"

            if retry_output.exists() and any(retry_output.glob("*.jsonl")):
                merge_retry_results(
                    domain, retry_output, main_results, manifest, domain
                )

        logger.info("Retry attempt {} complete", attempt)

    # Final check after all retries
    logger.info("Checking final failure counts...")
    final_failures: dict[str, list[dict]] = {}
    for domain in domains:
        config = DOMAINS[DomainName(domain)]
        result = parse_domain_results(
            domain,
            Path("data") / "input" / domain / "shards" / "output",
            Path("data") / "input" / domain / "manifest.jsonl",
            Path("data") / "output" / domain / f"{config.prefix}_results.jsonl",
        )
        if result["failed"] > 0:
            final_failures[domain] = result.get("failed_items", [])

    under_threshold, total = check_holdout_threshold(final_failures)
    if under_threshold:
        logger.success(
            "Final failures ({}) within holdout threshold (<100). Proceeding.", total
        )
    else:
        logger.warning(
            "Final failures ({}) exceed holdout threshold. Review needed.", total
        )
    _write_holdout_log(final_failures)


def _write_holdout_log(failures: dict[str, list[dict]]) -> None:
    holdout_path = Path("data") / "output" / "holdout_failures.jsonl"
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    with open(holdout_path, "w", encoding="utf-8") as f:
        for domain, items in failures.items():
            for item in items:
                item["domain"] = domain
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    total = sum(len(items) for items in failures.values())
    if total > 0:
        logger.info("Wrote {} holdout failures to {}", total, holdout_path)


def step_merge() -> None:
    logger.info("=" * 60)
    logger.info("STEP: MERGE - Consolidating domain outputs")
    logger.info("=" * 60)

    from scripts.merge_stage_1 import merge_domains

    count = merge_domains(
        output_dir=Path("data/stage_1"),
        stage1_output=Path("data/stage_1.jsonl"),
    )
    if count == 0:
        logger.error("Merge produced no entries!")


def step_split() -> None:
    logger.info("=" * 60)
    logger.info("STEP: SPLIT - Assigning train/val/test splits")
    logger.info("=" * 60)

    from scripts.split import split_stage1_file

    stage1_path = Path("data/stage_1.jsonl")
    if not stage1_path.exists():
        logger.error("stage_1.jsonl not found. Run merge step first.")
        return

    split_stage1_file(stage1_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 Evaluation Pipeline Orchestrator",
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
        help="Max samples per domain for testing (default: all)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose debug logging",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent batch jobs (default: 5)",
    )

    parser.add_argument(
        "--max-file-size-mb",
        type=int,
        default=200,
        help="Max file size in MB per shard (default: 200)",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    # Resolve domains
    if args.domain == "all":
        domains = DOMAIN_NAMES
    else:
        domains = [args.domain]

    logger.info("=" * 60)
    logger.info("STAGE 1 EVALUATION PIPELINE")
    logger.info("=" * 60)
    logger.info("Step: {}", args.step)
    logger.info("Domains: {}", ", ".join(domains))
    if args.max_samples:
        logger.info("Max samples: {}", args.max_samples)

    # Execute steps
    if args.step == "prepare":
        step_prepare(domains, max_samples=args.max_samples)
    elif args.step == "run":
        step_run(
            domains,
            max_concurrent=args.max_concurrent,
            max_file_size_mb=args.max_file_size_mb,
        )
    elif args.step == "parse":
        step_parse(domains)
    elif args.step == "retry":
        step_retry(domains, max_concurrent=args.max_concurrent)
    elif args.step == "merge":
        step_merge()
    elif args.step == "split":
        step_split()
    elif args.step == "all":
        step_prepare(domains, max_samples=args.max_samples)
        step_run(
            domains,
            max_concurrent=args.max_concurrent,
            max_file_size_mb=args.max_file_size_mb,
        )
        step_parse(domains)
        step_retry(domains, max_concurrent=args.max_concurrent)
        step_merge()
        step_split()

    logger.info("=" * 60)
    logger.success("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
