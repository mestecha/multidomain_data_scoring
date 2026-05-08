#!/usr/bin/env python3
"""distribution checks and side-by-side prompt rendering for data-quality audits."""

from __future__ import annotations

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import click
from loguru import logger

from scripts.config import DOMAINS
from scripts.models import (
    ContrastiveDirection,
    DomainName,
    Message,
    Stage2Candidate,
    VariantType,
)
from scripts.stage_2.prompts import build_generation_prompt


@click.group()
def cli() -> None:
    """data-quality audit tools."""


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--domain", default=None, help="filter to a single domain")
def dist(path: Path, domain: str | None) -> None:
    """per-dim distribution stats for a stage_1.jsonl file."""
    scores: dict[str, list[float]] = defaultdict(list)
    domain_counts: Counter[str] = Counter()

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if domain and d.get("domain") != domain:
                continue
            domain_counts[d.get("domain", "?")] += 1
            for k, v in (d.get("scores") or {}).items():
                if v is not None:
                    scores[k].append(v)

    logger.info("domains: {}", dict(domain_counts))
    logger.info(
        "{:<30} {:>6} {:>6} {:>6} {:>6} {:>6} {:>9} {:>8}",
        "dim", "n", "min", "med", "max", "std", "frac>=.5", "distinct",
    )
    for dim in sorted(scores):
        vs = sorted(scores[dim])
        n = len(vs)
        if not n:
            continue
        med = vs[n // 2]
        std = st.pstdev(vs)
        frac = sum(1 for v in vs if v >= 0.5) / n
        distinct = len({round(v, 3) for v in vs})
        logger.info(
            "{:<30} {:>6} {:>6.2f} {:>6.2f} {:>6.2f} {:>6.2f} {:>9.3f} {:>8}",
            dim, n, min(vs), med, max(vs), std, frac, distinct,
        )


@cli.command()
@click.option("--n", default=3, help="how many dialogues to render")
@click.option("--domain", default="multicultural", help="domain to render")
@click.option("--stage1-path", default="data/stage_1.jsonl", type=click.Path())
def render(n: int, domain: str, stage1_path: str) -> None:
    """render generation prompts under each toggle combination, side by side."""
    candidates: list[Stage2Candidate] = []
    with open(stage1_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("domain") != domain:
                continue
            entry_msgs = [Message(**m) for m in d.get("messages", [])]
            if len(entry_msgs) < 2:
                continue
            candidates.append(
                Stage2Candidate(
                    dialogue_id=d["dialogue_id"],
                    domain=DomainName(domain),
                    messages=entry_msgs,
                    target_dimensions=[f"{DOMAINS[DomainName(domain)].prefix}_cultural_value"]
                    if domain == "multicultural"
                    else [],
                    variant_type=VariantType.DIMENSION_TARGETED,
                    contrastive_direction=ContrastiveDirection.NEGATIVE,
                    domain_metadata=d.get("domain_metadata"),
                    characterizing_scores={},
                )
            )
            if len(candidates) >= n:
                break

    config = DOMAINS[DomainName(domain)]
    for i, c in enumerate(candidates, 1):
        click.echo("\n" + "=" * 100)
        click.echo(f"DIALOGUE {i}/{len(candidates)} — id={c.dialogue_id}  direction=NEGATIVE")
        click.echo("=" * 100)

        for label, enrich in [("BASELINE (no drivers)", False), ("ENRICHED (drivers on)", True)]:
            click.echo(f"\n----- {label} -----")
            prompt = build_generation_prompt(
                c, turn_count=3, config=config,
                domain_metadata=c.domain_metadata,
                enrich_drivers=enrich,
            )
            click.echo(prompt)


if __name__ == "__main__":
    cli()
