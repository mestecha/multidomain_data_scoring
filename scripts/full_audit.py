#!/usr/bin/env python3
"""exhaustive systematic audit of data/stage_1.jsonl and data/stage_2.jsonl.

Runs every structural, schema, referential, and content invariant we know.
Prints per-check counts plus up to 3 sample failures.
Exit code 0 only if every check returns 0 violations.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

STAGE_1 = Path("data/stage_1.jsonl")
STAGE_2 = Path("data/stage_2.jsonl")

DOMAINS = {"coherence", "empathy", "commonsense", "multicultural"}
SPLITS = {"train", "test"}
ROLES = {"user", "assistant"}
DIRECTIONS = {"positive", "negative"}
SCOPES = {"global", "target"}
LABELS = {"global_pass", "target_pass", "target_coarse_pass",
          "global_flip_pass", "target_flip_pass", "reject"}
METHODS = {"lm_judge", "human", "rules"}

DIMS_23 = {
    "co_topic_coherence", "co_logical_consistency", "co_temporal_causal_coherence",
    "co_discourse_structure", "co_mutual_grounding", "co_overall_coherence_score",
    "em_emotional_awareness", "em_emotional_validation", "em_perspective_taking",
    "em_supportive_engagement", "em_helpful_response", "em_overall_empathy_score",
    "cs_causality", "cs_consistency", "cs_reaction", "cs_desire", "cs_coherence", "cs_empathy",
    "mu_cultural_value", "mu_cultural_specificity", "mu_naturalness", "mu_coherence", "mu_empathy",
}
DOMAIN_PREFIX = {"coherence": "co_", "empathy": "em_", "commonsense": "cs_", "multicultural": "mu_"}

PLACEHOLDER_RE = re.compile(r"[xX]{4,}|\[REDACTED\]|\[TODO\]|\bFIXME\b|lorem ipsum", re.I)

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")

CHAR_DIMS_BY_DOMAIN = {
    "coherence": ["co_topic_coherence", "co_logical_consistency"],
    "empathy": ["em_emotional_awareness", "em_perspective_taking"],
    "commonsense": ["cs_causality", "cs_consistency", "cs_reaction", "cs_desire"],
    "multicultural": ["mu_cultural_value", "mu_cultural_specificity"],
}

S1_KEYS = {"dialogue_id", "source_id", "domain", "messages", "split", "scores", "domain_metadata"}
S2_KEYS = {"pair_id", "source_dialogue_id", "metadata", "evaluation", "messages", "chosen", "rejected"}
S2_META = {"domain", "split", "generation_model", "turn_count", "difficulty",
           "contrast", "dimensions"}
S2_CONTRAST = {"scope", "direction", "decision", "intent_followed"}
S2_DIMENSIONS = {"domain", "target", "decision"}
S2_EVAL = {"method", "stage_1_scores", "stage_2_scores",
           "chosen_score_source", "rejected_score_source"}


class Report:
    def __init__(self):
        self.findings: dict[str, list] = {}

    def add(self, check: str, item):
        self.findings.setdefault(check, []).append(item)

    def total(self) -> int:
        return sum(len(v) for v in self.findings.values())

    def print(self, header: str) -> None:
        print(f"\n{'=' * 70}\n  {header}\n{'=' * 70}")
        if not self.findings:
            print("  ALL CHECKS PASSED")
            return
        for check in sorted(self.findings):
            items = self.findings[check]
            print(f"  ✗ {check}: {len(items)}")
            for sample in items[:3]:
                print(f"      {sample}")
            if len(items) > 3:
                print(f"      ... +{len(items)-3} more")


def _alt_ok(side) -> bool:
    if not side:
        return False
    for i in range(1, len(side)):
        if side[i]["role"] == side[i-1]["role"]:
            return False
    return True


def _has_placeholder(text: str) -> bool:
    return bool(PLACEHOLDER_RE.search(text))


def audit_stage_1() -> tuple[Report, dict[str, str]]:
    """audit stage_1; also return dialogue_id → domain map for stage_2 referential check."""
    r = Report()
    seen_dialogue_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    dialogue_to_domain: dict[str, str] = {}

    with open(STAGE_1, encoding="utf-8") as f:
        for line_n, line in enumerate(f, 1):
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                r.add("json_parse_fail", f"line {line_n}: {e}")
                continue

            # schema keys
            if set(d.keys()) != S1_KEYS:
                r.add("schema_mismatch", f"line {line_n} {d.get('dialogue_id')}: {set(d.keys()) ^ S1_KEYS}")
                continue

            did = d["dialogue_id"]
            sid = d["source_id"]
            dom = d["domain"]
            spl = d["split"]

            # dialogue_id format
            if not re.fullmatch(r"S1D-\d{6}", str(did)):
                r.add("bad_dialogue_id_format", f"line {line_n}: {did!r}")
            if did in seen_dialogue_ids:
                r.add("duplicate_dialogue_id", f"line {line_n}: {did}")
            seen_dialogue_ids.add(did)
            dialogue_to_domain[did] = dom

            # source_id uniqueness (string)
            if sid in seen_source_ids:
                r.add("duplicate_source_id", f"line {line_n}: {sid}")
            seen_source_ids.add(sid)

            # enums
            if dom not in DOMAINS:
                r.add("bad_domain", f"line {line_n} {did}: {dom!r}")
            if spl not in SPLITS:
                r.add("bad_split", f"line {line_n} {did}: {spl!r}")

            # messages
            msgs = d["messages"]
            if not msgs:
                r.add("empty_messages", f"line {line_n} {did}")
                continue
            if msgs[0]["role"] != "user":
                r.add("first_role_not_user", f"line {line_n} {did}: {msgs[0]['role']}")
            if not _alt_ok(msgs):
                r.add("role_alternation_violation", f"line {line_n} {did}")
            for i, m in enumerate(msgs):
                if not isinstance(m, dict) or set(m.keys()) != {"role", "content"}:
                    r.add("malformed_message_object", f"line {line_n} {did}[{i}]")
                    continue
                if m["role"] not in ROLES:
                    r.add("invalid_role", f"line {line_n} {did}[{i}]: {m['role']!r}")
                if not isinstance(m["content"], str):
                    r.add("non_string_content", f"line {line_n} {did}[{i}]: {type(m['content']).__name__}")
                    continue
                if not m["content"].strip():
                    r.add("empty_content_turn", f"line {line_n} {did}[{i}]")
                if _has_placeholder(m["content"]):
                    r.add("placeholder_in_content", f"line {line_n} {did}[{i}]: {m['content'][:60]!r}")

            # scores
            scores = d.get("scores") or {}
            if set(scores.keys()) != DIMS_23:
                r.add("scores_keys_mismatch", f"line {line_n} {did}: {set(scores.keys()) ^ DIMS_23}")
            else:
                # values: domain dims must be float in [0,1], others null
                prefix = DOMAIN_PREFIX.get(dom)
                for k, v in scores.items():
                    if k.startswith(prefix):
                        if v is None:
                            r.add("missing_domain_score", f"line {line_n} {did}: {k}")
                        elif not isinstance(v, (int, float)):
                            r.add("non_numeric_score", f"line {line_n} {did}: {k}={v!r}")
                        else:
                            if v != v:  # NaN
                                r.add("nan_score", f"line {line_n} {did}: {k}")
                            elif v < 0.0 or v > 1.0:
                                r.add("score_out_of_range", f"line {line_n} {did}: {k}={v}")
                    else:
                        if v is not None:
                            r.add("non_domain_score_not_null", f"line {line_n} {did}: {k}={v}")

    return r, dialogue_to_domain


def audit_stage_2(dialogue_to_domain: dict[str, str]) -> Report:
    r = Report()
    seen_pair_ids: set[str] = set()

    with open(STAGE_2, encoding="utf-8") as f:
        for line_n, line in enumerate(f, 1):
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                r.add("json_parse_fail", f"line {line_n}: {e}")
                continue

            if set(d.keys()) != S2_KEYS:
                r.add("schema_mismatch", f"line {line_n} {d.get('pair_id')}: {set(d.keys()) ^ S2_KEYS}")
                continue

            pid = d["pair_id"]
            sid = d["source_dialogue_id"]

            if not re.fullmatch(r"S2D-\d{6}", str(pid)):
                r.add("bad_pair_id_format", f"line {line_n}: {pid!r}")
            if pid in seen_pair_ids:
                r.add("duplicate_pair_id", f"line {line_n}: {pid}")
            seen_pair_ids.add(pid)

            # referential integrity
            src_domain = dialogue_to_domain.get(sid)
            if src_domain is None:
                r.add("source_dialogue_id_missing", f"line {line_n} {pid}: {sid}")
            elif src_domain != d["metadata"]["domain"]:
                r.add("domain_mismatch_with_source", f"line {line_n} {pid}: meta={d['metadata']['domain']} src={src_domain}")

            # metadata structure
            meta = d["metadata"]
            if set(meta.keys()) != S2_META:
                r.add("metadata_keys_mismatch", f"line {line_n} {pid}: {set(meta.keys()) ^ S2_META}")
            else:
                if meta["domain"] not in DOMAINS:
                    r.add("bad_meta_domain", f"line {line_n} {pid}: {meta['domain']!r}")
                if meta["split"] not in SPLITS:
                    r.add("bad_meta_split", f"line {line_n} {pid}: {meta['split']!r}")
                if not isinstance(meta["turn_count"], int) or meta["turn_count"] <= 0:
                    r.add("bad_turn_count", f"line {line_n} {pid}: {meta['turn_count']}")
                if meta["difficulty"] not in {"easy", "medium", "hard"}:
                    r.add("bad_difficulty", f"line {line_n} {pid}: {meta['difficulty']!r}")
                if set(meta["contrast"].keys()) != S2_CONTRAST:
                    r.add("contrast_keys_mismatch", f"line {line_n} {pid}")
                else:
                    c = meta["contrast"]
                    if c["scope"] not in SCOPES:
                        r.add("bad_scope", f"line {line_n} {pid}: {c['scope']!r}")
                    if c["direction"] not in DIRECTIONS:
                        r.add("bad_direction", f"line {line_n} {pid}: {c['direction']!r}")
                    if c["decision"] not in LABELS:
                        r.add("bad_decision_label", f"line {line_n} {pid}: {c['decision']!r}")
                    if not isinstance(c["intent_followed"], bool):
                        r.add("bad_intent_followed_type", f"line {line_n} {pid}")
                if set(meta["dimensions"].keys()) != S2_DIMENSIONS:
                    r.add("dimensions_keys_mismatch", f"line {line_n} {pid}")
                else:
                    dims = meta["dimensions"]
                    dom_dims = set(dims["domain"])
                    if not set(dims["target"]).issubset(dom_dims):
                        r.add("target_dims_outside_domain", f"line {line_n} {pid}")
                    if not set(dims["decision"]).issubset(dom_dims):
                        r.add("decision_dims_outside_domain", f"line {line_n} {pid}")

            # evaluation
            ev = d["evaluation"]
            if set(ev.keys()) != S2_EVAL:
                r.add("evaluation_keys_mismatch", f"line {line_n} {pid}: {set(ev.keys()) ^ S2_EVAL}")
            else:
                if ev["method"] not in METHODS:
                    r.add("bad_eval_method", f"line {line_n} {pid}: {ev['method']!r}")
                if ev["chosen_score_source"] not in {"stage_1", "stage_2"}:
                    r.add("bad_chosen_score_source", f"line {line_n} {pid}")
                if ev["rejected_score_source"] not in {"stage_1", "stage_2"}:
                    r.add("bad_rejected_score_source", f"line {line_n} {pid}")
                # score keys should match per-domain
                for label, sc in (("stage_1_scores", ev["stage_1_scores"]),
                                  ("stage_2_scores", ev["stage_2_scores"])):
                    if not sc:
                        r.add(f"empty_{label}", f"line {line_n} {pid}")
                        continue
                    for k, v in sc.items():
                        if k not in DIMS_23:
                            r.add(f"{label}_unknown_dim", f"line {line_n} {pid}: {k}")
                        elif v is not None and (v != v or not isinstance(v, (int, float)) or v < 0 or v > 1):
                            r.add(f"{label}_value_invalid", f"line {line_n} {pid}: {k}={v!r}")

            # messages/chosen/rejected non-empty
            for side_name in ("messages", "chosen", "rejected"):
                side = d[side_name]
                if not side:
                    r.add(f"empty_{side_name}", f"line {line_n} {pid}")
                    continue
                for i, m in enumerate(side):
                    if not isinstance(m, dict) or set(m.keys()) != {"role", "content"}:
                        r.add(f"malformed_{side_name}_msg", f"line {line_n} {pid}[{i}]")
                        continue
                    if m["role"] not in ROLES:
                        r.add(f"invalid_role_in_{side_name}", f"line {line_n} {pid}[{i}]: {m['role']!r}")
                    if not isinstance(m["content"], str):
                        r.add(f"non_string_content_{side_name}", f"line {line_n} {pid}[{i}]")
                        continue
                    if not m["content"].strip():
                        r.add(f"empty_content_in_{side_name}", f"line {line_n} {pid}[{i}]")
                    if _has_placeholder(m["content"]):
                        r.add(f"placeholder_in_{side_name}", f"line {line_n} {pid}[{i}]: {m['content'][:60]!r}")

            # alternation per side
            if d["messages"] and not _alt_ok(d["messages"]):
                r.add("alt_violation_in_messages", f"line {line_n} {pid}")
            if d["chosen"] and not _alt_ok(d["chosen"]):
                r.add("alt_violation_in_chosen", f"line {line_n} {pid}")
            if d["rejected"] and not _alt_ok(d["rejected"]):
                r.add("alt_violation_in_rejected", f"line {line_n} {pid}")

            # role boundary
            if d["messages"]:
                tail = d["messages"][-1]["role"]
                expected_first = "assistant" if tail == "user" else "user"
                if d["chosen"] and d["chosen"][0]["role"] != expected_first:
                    r.add("role_boundary_chosen", f"line {line_n} {pid}: expected {expected_first} got {d['chosen'][0]['role']}")
                if d["rejected"] and d["rejected"][0]["role"] != expected_first:
                    r.add("role_boundary_rejected", f"line {line_n} {pid}: expected {expected_first} got {d['rejected'][0]['role']}")
                # full concat alternation
                full = d["messages"] + d["chosen"]
                if not _alt_ok(full):
                    r.add("concat_alt_violation_chosen", f"line {line_n} {pid}")
                full = d["messages"] + d["rejected"]
                if not _alt_ok(full):
                    r.add("concat_alt_violation_rejected", f"line {line_n} {pid}")

            # score-ordering invariant: mean(chosen on effective dims) > mean(rejected on effective dims)
            try:
                meta = d["metadata"]
                ev = d["evaluation"]
                dom = meta["domain"]
                scope = meta["contrast"]["scope"]
                target = meta["dimensions"].get("target") or []
                effective = CHAR_DIMS_BY_DOMAIN.get(dom, []) if scope == "global" else target
                direction = meta["contrast"]["direction"]
                intent = meta["contrast"]["intent_followed"]
                # chosen/rejected score source per pair
                if (direction == "positive") == intent:
                    chosen_sc, rejected_sc = ev["stage_2_scores"], ev["stage_1_scores"]
                else:
                    chosen_sc, rejected_sc = ev["stage_1_scores"], ev["stage_2_scores"]
                if effective:
                    cv = [chosen_sc.get(k) for k in effective if chosen_sc.get(k) is not None]
                    rv = [rejected_sc.get(k) for k in effective if rejected_sc.get(k) is not None]
                    if cv and rv and (sum(cv) / len(cv)) <= (sum(rv) / len(rv)):
                        r.add("score_ordering_violation", f"line {line_n} {pid}: chosen_mean={sum(cv)/len(cv):.3f} <= rejected_mean={sum(rv)/len(rv):.3f} on {effective}")
            except (KeyError, TypeError):
                pass  # already counted via other schema checks

            # control chars in any side
            for side_name in ("messages", "chosen", "rejected"):
                for i, m in enumerate(d[side_name]):
                    if isinstance(m.get("content"), str) and CONTROL_RE.search(m["content"]):
                        r.add(f"control_chars_in_{side_name}", f"line {line_n} {pid}[{i}]")

            # content equality
            if d["chosen"] and d["rejected"]:
                ch_str = json.dumps(d["chosen"], sort_keys=True, ensure_ascii=False)
                rj_str = json.dumps(d["rejected"], sort_keys=True, ensure_ascii=False)
                if ch_str == rj_str:
                    r.add("chosen_equals_rejected", f"line {line_n} {pid}")
                # NEAR-identical: same length string, >99% chars match
                if len(ch_str) == len(rj_str) and len(ch_str) > 40:
                    common = sum(1 for a, b in zip(ch_str, rj_str) if a == b)
                    if common / len(ch_str) > 0.99 and ch_str != rj_str:
                        r.add("chosen_near_rejected", f"line {line_n} {pid}: {100*common/len(ch_str):.1f}% match")

    return r


def main() -> int:
    if not STAGE_1.exists() or not STAGE_2.exists():
        print("missing data/stage_1.jsonl or data/stage_2.jsonl")
        return 2

    r1, did2dom = audit_stage_1()
    r1.print(f"STAGE 1 — {len(did2dom)} dialogues")
    r2 = audit_stage_2(did2dom)
    r2.print(f"STAGE 2")

    total = r1.total() + r2.total()
    print(f"\n{'=' * 70}")
    print(f"  TOTAL VIOLATIONS: {total}")
    print(f"{'=' * 70}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
