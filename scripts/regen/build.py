#!/usr/bin/env python3
"""regenerate stage 2 candidates from malformed pairs using stage_1_v2_opus labels."""

from __future__ import annotations

import json
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
from scripts.stage_2.generate import (
    build_generation_entries,
    write_manifest,
    write_shards,
)

MALFORMED_PATH = Path("data/audits/qa/malformed_pairs.jsonl")
SOURCE_PATH = Path("data/stage_1.jsonl")
REGEN_DIR = Path("data/stage_2/regen")
CANDIDATES_PATH = REGEN_DIR / "candidates.jsonl"
SHARD_DIR = REGEN_DIR / "shards_gen"
MANIFEST_PATH = REGEN_DIR / "manifest_gen.jsonl"


def _variant_type(scope: str, direction: str) -> VariantType:
    if scope == "global":
        return VariantType.GLOBAL_IMPROVE if direction == "positive" else VariantType.GLOBAL_DEGRADE
    return VariantType.DIMENSION_TARGETED


@click.command()
@click.option("--malformed-path", default=str(MALFORMED_PATH), type=click.Path())
@click.option("--source-path", default=str(SOURCE_PATH), type=click.Path())
@click.option("--regen-dir", default=str(REGEN_DIR), type=click.Path())
def main(malformed_path: str, source_path: str, regen_dir: str) -> None:
    """build stage_2 candidates + batch shards for the malformed-pair regen."""
    regen = Path(regen_dir)
    regen.mkdir(parents=True, exist_ok=True)
    cand_path = regen / "candidates.jsonl"
    shard_dir = regen / "shards_gen"
    manifest_path = regen / "manifest_gen.jsonl"

    malformed = [json.loads(l) for l in open(malformed_path)]
    logger.info("loaded {} malformed entries", len(malformed))

    sources = {}
    with open(source_path) as f:
        for line in f:
            d = json.loads(line)
            sources[d["dialogue_id"]] = d
    logger.info("loaded {} source dialogues from {}", len(sources), source_path)

    candidates: list[Stage2Candidate] = []
    missing = 0
    for m in malformed:
        sid = m["source_dialogue_id"]
        s = sources.get(sid)
        if s is None:
            missing += 1
            continue
        domain = DomainName(m["domain"])
        char_dims = DOMAINS[domain].characterizing_dims
        char_scores = {k: v for k, v in (s.get("scores") or {}).items() if v is not None and k in char_dims}
        cand = Stage2Candidate(
            dialogue_id=sid,
            domain=domain,
            messages=[Message(**msg) for msg in s["messages"]],
            characterizing_scores=char_scores,
            target_dimensions=m["target_dimensions"],
            contrastive_direction=ContrastiveDirection(m["contrastive_direction"]),
            variant_type=_variant_type(m["scope"], m["contrastive_direction"]),
            domain_metadata=s.get("domain_metadata"),
        )
        candidates.append(cand)
    logger.info("built {} candidates, missing {} (source not found in v2_opus)", len(candidates), missing)

    with open(cand_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(c.model_dump_json() + "\n")
    logger.success("wrote {} ({} candidates)", cand_path, len(candidates))

    batch_entries, manifest_items = build_generation_entries(candidates, enrich_drivers=True)
    write_shards(batch_entries, shard_dir)
    write_manifest(manifest_items, manifest_path)
    logger.success("ready to submit: {} batch entries in {}", len(batch_entries), shard_dir)


if __name__ == "__main__":
    main()
