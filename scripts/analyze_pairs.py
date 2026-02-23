#!/usr/bin/env python3
"""Analyze data/stage_2/pairs.jsonl and print summary statistics."""

import json
from collections import defaultdict
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stage_2" / "pairs.jsonl"

# Characterizing dimensions per domain
CHAR_DIMS = {
    "coherence":    ["co_topic_coherence", "co_logical_consistency"],
    "empathy":      ["em_emotional_awareness", "em_perspective_taking"],
    "commonsense":  ["cs_causality", "cs_consistency", "cs_reaction", "cs_desire"],
    "multicultural": ["mu_cultural_value", "mu_cultural_specificity"],
}

MARGIN_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]

# ── accumulators ──────────────────────────────────────────────────────────────
# 1) Score averages: domain -> dim -> {"original": [vals], "variant": [vals]}
score_acc = defaultdict(lambda: defaultdict(lambda: {"original": [], "variant": []}))

# 2) Contrastive direction distribution: domain -> {"variant_chosen": 0, "original_chosen": 0}
direction_acc = defaultdict(lambda: {"variant_chosen": 0, "original_chosen": 0})

# 3) Variant type breakdown: domain -> {"global_improve": 0, "global_degrade": 0, "dimension_targeted": 0}
variant_acc = defaultdict(lambda: {"global_improve": 0, "global_degrade": 0, "dimension_targeted": 0})

# 4) Margins per domain: domain -> [margin_values]
margin_acc = defaultdict(list)

# ── single pass ───────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    for line in f:
        pair = json.loads(line)
        meta = pair["metadata"]
        domain = meta["domain"]
        contrast = meta["contrast"]
        evaluation = pair["evaluation"]
        decision_dims = meta["dimensions"]["decision"]

        scope = contrast["scope"]
        direction = contrast["direction"]
        intent_followed = contrast["intent_followed"]

        original = evaluation["stage_1_scores"]  # stage_1 = original
        variant = evaluation["stage_2_scores"]    # stage_2 = variant

        chosen_src = evaluation["chosen_score_source"]
        rejected_src = evaluation["rejected_score_source"]
        chosen = original if chosen_src == "stage_1" else variant
        rejected = original if rejected_src == "stage_1" else variant

        # 1) Score averages for characterizing dimensions
        if domain in CHAR_DIMS:
            for dim in CHAR_DIMS[domain]:
                if dim in original and dim in variant:
                    score_acc[domain][dim]["original"].append(original[dim])
                    score_acc[domain][dim]["variant"].append(variant[dim])

        # 2) Contrastive direction
        #    Variant is chosen when: (positive + intent_followed) OR (negative + NOT intent_followed)
        #    Original is chosen when: (negative + intent_followed) OR (positive + NOT intent_followed)
        if (direction == "positive" and intent_followed) or (direction == "negative" and not intent_followed):
            direction_acc[domain]["variant_chosen"] += 1
        else:
            direction_acc[domain]["original_chosen"] += 1

        # 3) Variant type
        if scope == "global" and direction == "positive":
            variant_acc[domain]["global_improve"] += 1
        elif scope == "global" and direction == "negative":
            variant_acc[domain]["global_degrade"] += 1
        else:  # scope == "target"
            variant_acc[domain]["dimension_targeted"] += 1

        # 4) Margin = mean(chosen[d] - rejected[d]) for decision dims
        if decision_dims:
            diffs = [chosen[d] - rejected[d] for d in decision_dims if d in chosen and d in rejected]
            if diffs:
                margin = sum(diffs) / len(diffs)
                margin_acc[domain].append(margin)


# ── helpers ───────────────────────────────────────────────────────────────────
def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


DOMAIN_ORDER = ["coherence", "empathy", "commonsense", "multicultural"]


# ── 1) Score averages table ──────────────────────────────────────────────────
print("## 1. Average Original vs Variant Scores (Characterizing Dimensions)\n")
for domain in DOMAIN_ORDER:
    dims = CHAR_DIMS[domain]
    print(f"### {domain.capitalize()}\n")
    print(f"| {'Dimension':<30s} | {'Avg Original':>12s} | {'Avg Variant':>12s} | {'Shift':>8s} |")
    print(f"|{'-'*32}|{'-'*14}|{'-'*14}|{'-'*10}|")
    for dim in dims:
        o = avg(score_acc[domain][dim]["original"])
        v = avg(score_acc[domain][dim]["variant"])
        s = v - o
        print(f"| {dim:<30s} | {o:>12.3f} | {v:>12.3f} | {s:>+8.3f} |")
    print()


# ── 2) Contrastive direction distribution ────────────────────────────────────
print("## 2. Contrastive Direction Distribution\n")
print(f"| {'Domain':<14s} | {'Variant Chosen':>15s} | {'Original Chosen':>16s} | {'Total':>7s} | {'% Variant':>10s} |")
print(f"|{'-'*16}|{'-'*17}|{'-'*18}|{'-'*9}|{'-'*12}|")
for domain in DOMAIN_ORDER:
    vc = direction_acc[domain]["variant_chosen"]
    oc = direction_acc[domain]["original_chosen"]
    total = vc + oc
    pct = 100.0 * vc / total if total else 0.0
    print(f"| {domain:<14s} | {vc:>15,d} | {oc:>16,d} | {total:>7,d} | {pct:>9.1f}% |")
print()


# ── 3) Variant type breakdown ────────────────────────────────────────────────
print("## 3. Variant Type Breakdown (per Domain)\n")
print(f"| {'Domain':<14s} | {'global_improve':>15s} | {'global_degrade':>15s} | {'dimension_targeted':>19s} | {'Total':>7s} |")
print(f"|{'-'*16}|{'-'*17}|{'-'*17}|{'-'*21}|{'-'*9}|")
for domain in DOMAIN_ORDER:
    gi = variant_acc[domain]["global_improve"]
    gd = variant_acc[domain]["global_degrade"]
    dt = variant_acc[domain]["dimension_targeted"]
    total = gi + gd + dt
    print(f"| {domain:<14s} | {gi:>15,d} | {gd:>15,d} | {dt:>19,d} | {total:>7,d} |")
print()


# ── 4) Margin threshold sensitivity ──────────────────────────────────────────
print("## 4. Margin Threshold Sensitivity (% Pairs Lost if Margin <= Threshold)\n")

# Header
header = f"| {'Domain':<14s} |"
for t in MARGIN_THRESHOLDS:
    header += f" {'<=' + f'{t:.2f}':>7s} |"
print(header)

sep = f"|{'-'*16}|"
for _ in MARGIN_THRESHOLDS:
    sep += f"{'-'*9}|"
print(sep)

for domain in DOMAIN_ORDER:
    margins = margin_acc[domain]
    total = len(margins)
    row = f"| {domain:<14s} |"
    for t in MARGIN_THRESHOLDS:
        lost = sum(1 for m in margins if m <= t)
        pct = 100.0 * lost / total if total else 0.0
        row += f" {pct:>6.1f}% |"
    print(row)
print()
