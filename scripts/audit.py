#!/usr/bin/env python3
"""distribution checks, prompt rendering, and lm comparison for data audits."""

from __future__ import annotations

import json
import os
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime
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
from scripts.stage_1.prompts import (
    QID_INFO,
    STATEMENT_TO_QID,
    build_multicultural_prompt,
)
from scripts.stage_2.prompts import build_generation_prompt

AUDITS_DIR = Path("data/audits")


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


def _azure_client():
    from dotenv import load_dotenv
    from openai import AzureOpenAI

    load_dotenv()
    return AzureOpenAI(
        api_key=os.environ["AZURE_BATCH_GPT_API_KEY"],
        api_version=os.environ["AZURE_BATCH_GPT_API_VERSION"],
        azure_endpoint=os.environ["AZURE_BATCH_GPT_ENDPOINT"],
    )


def _call(client, deployment: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def _parse_score(content: str, dim: str) -> float | None:
    try:
        # strip any code fences
        c = content.strip()
        if c.startswith("```"):
            c = c.split("\n", 1)[1].rsplit("```", 1)[0]
        d = json.loads(c)
        v = d.get(dim)
        if isinstance(v, dict):
            v = v.get("score")
        return float(v) if v is not None else None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


@cli.group()
def compare() -> None:
    """side-by-side LM comparisons of toggle conditions."""


@compare.command("rubric")
@click.option("--n", default=15, help="dialogues to score")
@click.option("--deployment", default="gpt-5", help="azure sync deployment name")
@click.option("--stage1-path", default="data/stage_1.jsonl", type=click.Path())
def compare_rubric(n: int, deployment: str, stage1_path: str) -> None:
    """score N multicultural dialogues with qid_addendum OFF vs ON."""
    out_dir = AUDITS_DIR / "rubric"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    with open(stage1_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("domain") != "multicultural":
                continue
            md = d.get("domain_metadata") or {}
            stmt = md.get("statement_original", "")
            if STATEMENT_TO_QID.get(stmt) not in QID_INFO:
                continue
            candidates.append(d)
            if len(candidates) >= n:
                break

    logger.info("loaded {} dialogues with qid coverage", len(candidates))
    if not candidates:
        logger.warning("no qid-covered dialogues found; cannot compare")
        return

    client = _azure_client()
    results: list[dict] = []

    for i, d in enumerate(candidates, 1):
        md = d["domain_metadata"]
        turns_text = "\n".join(
            f"User: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content']}"
            for m in d["messages"]
        )
        kwargs = dict(
            turns_text=turns_text,
            statement=md["statement_original"],
            country_1=md["country_1"],
            country_2=md["country_2"],
        )
        prompt_off = build_multicultural_prompt(**kwargs, enrich_qid=False)
        prompt_on = build_multicultural_prompt(**kwargs, enrich_qid=True)

        logger.info("[{}/{}] {} ({})", i, len(candidates), d["dialogue_id"], STATEMENT_TO_QID[md["statement_original"]])
        try:
            resp_off = _call(client, deployment, prompt_off)
            resp_on = _call(client, deployment, prompt_on)
        except Exception as e:
            logger.warning("call failed: {}", e)
            continue

        results.append({
            "dialogue_id": d["dialogue_id"],
            "qid": STATEMENT_TO_QID.get(md["statement_original"]),
            "country_1": md["country_1"],
            "country_2": md["country_2"],
            "ground_truth": d["scores"]["mu_cultural_value"],
            "score_off": _parse_score(resp_off, "mu_cultural_value"),
            "score_on": _parse_score(resp_on, "mu_cultural_value"),
            "raw_off": resp_off[:800],
            "raw_on": resp_on[:800],
        })

    out_path = out_dir / f"out_{datetime.now():%Y%m%dT%H%M}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.success("wrote {}", out_path)

    off_scores = [r["score_off"] for r in results if r["score_off"] is not None]
    on_scores = [r["score_on"] for r in results if r["score_on"] is not None]
    gt_scores = [r["ground_truth"] for r in results if r["ground_truth"] is not None]

    summary = {
        "n": len(results),
        "off_mean": st.mean(off_scores) if off_scores else None,
        "off_std": st.pstdev(off_scores) if off_scores else None,
        "on_mean": st.mean(on_scores) if on_scores else None,
        "on_std": st.pstdev(on_scores) if on_scores else None,
        "ground_truth_mean": st.mean(gt_scores) if gt_scores else None,
        "delta_mean": (st.mean(on_scores) - st.mean(off_scores)) if (on_scores and off_scores) else None,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("summary: {}", json.dumps(summary, indent=2))


@compare.command("drivers")
@click.option("--n", default=15, help="dialogues to generate variants for")
@click.option("--deployment", default="gpt-5", help="azure sync deployment name")
@click.option("--stage1-path", default="data/stage_1.jsonl", type=click.Path())
def compare_drivers(n: int, deployment: str, stage1_path: str) -> None:
    """generate stage-2 variants with drivers OFF vs ON, save side-by-side."""
    out_dir = AUDITS_DIR / "drivers"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = DOMAINS[DomainName.MULTICULTURAL]
    candidates: list[tuple[Stage2Candidate, dict]] = []

    with open(stage1_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("domain") != "multicultural":
                continue
            msgs = [Message(**m) for m in d.get("messages", [])]
            if len(msgs) < 4:
                continue
            md = d.get("domain_metadata") or {}
            cand = Stage2Candidate(
                dialogue_id=d["dialogue_id"],
                domain=DomainName.MULTICULTURAL,
                messages=msgs,
                target_dimensions=["mu_cultural_value"],
                variant_type=VariantType.DIMENSION_TARGETED,
                contrastive_direction=ContrastiveDirection.NEGATIVE,
                domain_metadata=md,
                characterizing_scores={
                    k: v for k, v in d.get("scores", {}).items()
                    if k.startswith("mu_") and v is not None
                },
            )
            candidates.append((cand, d))
            if len(candidates) >= n:
                break

    logger.info("loaded {} multicultural candidates", len(candidates))
    client = _azure_client()
    results: list[dict] = []

    for i, (cand, d) in enumerate(candidates, 1):
        prompt_off = build_generation_prompt(
            cand, turn_count=3, config=config,
            domain_metadata=cand.domain_metadata, enrich_drivers=False,
        )
        prompt_on = build_generation_prompt(
            cand, turn_count=3, config=config,
            domain_metadata=cand.domain_metadata, enrich_drivers=True,
        )

        logger.info("[{}/{}] {} {} x {}", i, len(candidates), cand.dialogue_id,
                    cand.domain_metadata.get("country_1"), cand.domain_metadata.get("country_2"))
        try:
            resp_off = _call(client, deployment, prompt_off)
            resp_on = _call(client, deployment, prompt_on)
        except Exception as e:
            logger.warning("call failed: {}", e)
            continue

        results.append({
            "dialogue_id": cand.dialogue_id,
            "country_1": cand.domain_metadata.get("country_1"),
            "country_2": cand.domain_metadata.get("country_2"),
            "variant_off": resp_off,
            "variant_on": resp_on,
        })

    out_path = out_dir / f"out_{datetime.now():%Y%m%dT%H%M}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.success("wrote {}", out_path)
    logger.info("n={} variants generated under each toggle", len(results))


if __name__ == "__main__":
    cli()
