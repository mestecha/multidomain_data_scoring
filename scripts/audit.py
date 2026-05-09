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
    summary["deployment"] = deployment
    summary_path = out_dir / f"summary_{deployment}_{datetime.now():%Y%m%dT%H%M}.json"
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


@compare.command("score-variants")
@click.argument("drivers_output", type=click.Path(exists=True, path_type=Path))
@click.option("--deployment", default="gpt-5.1")
@click.option("--stage1-path", default="data/stage_1.jsonl", type=click.Path())
def score_variants(drivers_output: Path, deployment: str, stage1_path: str) -> None:
    """score variant_off and variant_on on mu_cultural_value to measure gap."""
    out_dir = AUDITS_DIR / "score_variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    # index source dialogues by id
    sources: dict[str, dict] = {}
    with open(stage1_path) as f:
        for line in f:
            d = json.loads(line)
            sources[d["dialogue_id"]] = d

    with open(drivers_output) as f:
        rows = [json.loads(l) for l in f]
    logger.info("scoring {} variants from {}", len(rows), drivers_output)

    client = _azure_client()
    results = []
    for i, r in enumerate(rows, 1):
        src = sources.get(r["dialogue_id"])
        if not src:
            continue
        md = src.get("domain_metadata") or {}

        def score_one(variant_text: str) -> float | None:
            try:
                turns = json.loads(variant_text).get("continuation", [])
                turns_text = "\n".join(
                    f"User: {t['content']}" if t["role"] == "user" else f"Assistant: {t['content']}"
                    for t in turns
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                return None
            prompt = build_multicultural_prompt(
                turns_text=turns_text,
                statement=md["statement_original"],
                country_1=md["country_1"], country_2=md["country_2"],
                enrich_qid=True,
            )
            try:
                resp = _call(client, deployment, prompt)
                return _parse_score(resp, "mu_cultural_value")
            except Exception as e:
                logger.warning("score call failed: {}", e)
                return None

        logger.info("[{}/{}] {}", i, len(rows), r["dialogue_id"])
        s_off = score_one(r["variant_off"])
        s_on = score_one(r["variant_on"])
        results.append({
            "dialogue_id": r["dialogue_id"],
            "score_variant_off": s_off,
            "score_variant_on": s_on,
            "gap": (s_off - s_on) if (s_off is not None and s_on is not None) else None,
        })

    out = out_dir / f"out_{deployment}_{datetime.now():%Y%m%dT%H%M}.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.success("wrote {}", out)

    valid = [r for r in results if r["gap"] is not None]
    if valid:
        gaps = [r["gap"] for r in valid]
        offs = [r["score_variant_off"] for r in valid]
        ons = [r["score_variant_on"] for r in valid]
        logger.info(
            "n={}  off_mean={:.3f}  on_mean={:.3f}  gap_mean={:.3f}  gap_std={:.3f}  inversions={}/{}",
            len(valid), st.mean(offs), st.mean(ons), st.mean(gaps), st.pstdev(gaps),
            sum(1 for g in gaps if g <= 0), len(valid),
        )


@cli.command("qid-gen")
@click.option("--deployment", default="gpt-5.1")
@click.option("--max-workers", default=8)
@click.option("--limit", default=0, help="cap how many qids to process (0=all)")
def qid_gen(deployment: str, max_workers: int, limit: int) -> None:
    """generate qid_meaning entries (means/not_this/shows_up_as/dont_say) for missing qids."""
    import csv as _csv
    from concurrent.futures import ThreadPoolExecutor

    base = Path("data/input/multicultural")
    qid_path = base / "qid_meaning.csv"
    statements_path = base / "statements.csv"

    existing: set[str] = set()
    with open(qid_path) as f:
        for row in _csv.DictReader(f):
            existing.add(row["qid"])

    statements_by_qid: dict[str, list[tuple[str, str]]] = {}
    with open(statements_path) as f:
        for row in _csv.DictReader(f):
            statements_by_qid.setdefault(row["qid"], []).append((row["country"], row["statement"]))

    missing = [q for q in statements_by_qid if q not in existing]
    if limit:
        missing = missing[:limit]
    logger.info("missing qids: {} / {}", len(missing), len(statements_by_qid))

    template = """You are operationalizing a WVS-style cultural value question. Given the question id and example statements, write 4 fields that help an LM judge whether a dialogue reflects this value.

QID: {qid}
EXAMPLE STATEMENTS:
{examples}

Return strict JSON with these exact keys:
{{
  "means": "<one phrase: what the value really measures (semantic core)>",
  "not_this": "<one phrase: things commonly confused with this value but distinct>",
  "shows_up_as": "<3 short concrete behavioral markers separated by '; '>",
  "dont_say": "<3 short phrases people say that miss the point, separated by '; '>"
}}

Return only valid JSON, no commentary."""

    client = _azure_client()

    def gen_one(qid: str) -> dict | None:
        examples = "\n".join(f"- ({c}) {s}" for c, s in statements_by_qid[qid][:5])
        prompt = template.format(qid=qid, examples=examples)
        try:
            resp = _call(client, deployment, prompt)
            c = resp.strip()
            if c.startswith("```"):
                c = c.split("\n", 1)[1].rsplit("```", 1)[0]
            d = json.loads(c)
            return {"qid": qid, **{k: d.get(k, "") for k in ["means", "not_this", "shows_up_as", "dont_say"]}}
        except Exception as e:
            logger.warning("{}: {}", qid, e)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        new_rows = list(filter(None, pool.map(gen_one, missing)))
    logger.success("generated {}/{} new qid entries", len(new_rows), len(missing))

    backup = qid_path.with_suffix(".csv.bak")
    qid_path.replace(backup)
    logger.info("backed up original to {}", backup)

    with open(qid_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=["qid", "means", "not_this", "shows_up_as", "dont_say"])
        writer.writeheader()
        # original entries
        with open(backup) as orig:
            for row in _csv.DictReader(orig):
                writer.writerow(row)
        # new
        for row in new_rows:
            writer.writerow(row)
    logger.success("wrote {} (now {} entries)", qid_path, len(existing) + len(new_rows))


@cli.command("rescore-mc")
@click.option("--deployment", default="gpt-5.1")
@click.option("--max-workers", default=8)
@click.option("--limit", default=0, help="cap dialogues (0=all 12,816)")
@click.option("--stage1-path", default="data/stage_1.jsonl", type=click.Path())
@click.option("--output", default="data/stage_1_v2.jsonl", type=click.Path())
def rescore_mc(deployment: str, max_workers: int, limit: int, stage1_path: str, output: str) -> None:
    """re-score multicultural stage-1 with the new rubric + qid_addendum.

    preserves non-multicultural rows; replaces only mu_* score blocks.
    writes to a separate file; original stage_1.jsonl untouched.
    """
    from concurrent.futures import ThreadPoolExecutor

    rows: list[dict] = []
    with open(stage1_path) as f:
        for line in f:
            rows.append(json.loads(line))
    mc_idx = [i for i, r in enumerate(rows) if r.get("domain") == "multicultural"]
    if limit:
        mc_idx = mc_idx[:limit]
    logger.info("re-scoring {} multicultural dialogues / {} total rows", len(mc_idx), len(rows))

    client = _azure_client()

    def rescore_one(i: int) -> tuple[int, dict | None]:
        d = rows[i]
        md = d.get("domain_metadata") or {}
        turns_text = "\n".join(
            f"User: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content']}"
            for m in d.get("messages", [])
        )
        prompt = build_multicultural_prompt(
            turns_text=turns_text,
            statement=md.get("statement_original", ""),
            country_1=md.get("country_1", ""),
            country_2=md.get("country_2", ""),
            enrich_qid=True,
        )
        try:
            resp = _call(client, deployment, prompt)
            c = resp.strip()
            if c.startswith("```"):
                c = c.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(c)
            new_scores = {}
            for k in ["mu_cultural_value", "mu_cultural_specificity", "mu_naturalness", "mu_coherence", "mu_empathy"]:
                v = parsed.get(k)
                if isinstance(v, dict):
                    v = v.get("score")
                if v is not None:
                    new_scores[k] = round(float(v), 2)
            return i, new_scores
        except Exception as e:
            logger.warning("dialogue {} failed: {}", d.get("dialogue_id"), e)
            return i, None

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, new_scores in pool.map(rescore_one, mc_idx):
            done += 1
            if new_scores:
                rows[i]["scores"] = {**rows[i].get("scores", {}), **new_scores}
            if done % 50 == 0:
                logger.info("progress: {}/{}", done, len(mc_idx))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.success("wrote {} ({} rows total, {} multicultural rescored)", output, len(rows), len(mc_idx))


if __name__ == "__main__":
    cli()
