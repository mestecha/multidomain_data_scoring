# Multi-Domain Data Scoring 

Last updated: 2026-05-11


## Purpose

This pipeline builds preference pairs for DPO training. It takes raw dialogues from four quality domains — coherence, empathy, commonsense, and multicultural — scores them on domain-specific dimensions, then generates contrastive rewrites where one version is clearly better than the other. The final output is a set of (chosen, rejected) dialogue continuation pairs that teach a model to distinguish quality along specific dimensions.


## Pipeline Overview

```
Raw dialogues (4 corpora, ~113k total)
    ↓ sample + deduplicate + self-taught negatives (multicultural)
52,584 dialogues
    ↓ Stage 1: GPT judge scores each on its domain's dimensions
52,584 scored entries (stage_1.jsonl, all 23 dims per entry)
    ↓ stratified split (seed=42)
47,462 train  ·  5,122 test
    ↓                ↓
Stage 2          Stage 2 (test pairs = ground truths)
    ↓                ↓
~167k candidates (3-4 variants per dialogue)
    ↓ generate contrastive rewrites (Azure batch + sync gpt-5.1 fallback)
    ↓ evaluate both versions scored independently
    ↓ 6-label classification (margin > 0.05)
    ↓ flip-pass recovery for opposite-direction variants
    ↓ regen pass for malformed pairs + Opus 4.7 subagent recovery
160,858 preference pairs (stage_2.jsonl)
    ↓
144,767 train pairs  ·  16,091 test pairs (10.0%)
```

The current canonical reflects a v2 rebuild (May 2026) that fixed a role-boundary bug, rescored multicultural, integrated self-taught negatives, and ran three regen passes plus a placeholder repair. See [v2 Rebuild](#v2-rebuild-may-2026) below for details.

Stage 2 processes all splits. Test pairs carry `split: "test"` in metadata and serve as ground truths for model evaluation.


## Stage 1 — Scoring

Each dialogue is evaluated by a GPT judge that sees the full conversation and a rubric describing the domain's quality dimensions. The judge returns a score per dimension.

### Data Sources

Coherence starts from 50,048 dialogues and samples 12,800 through 70/30 coherent/incoherent label stratification across 4 turn-count buckets (2-6, 7-10, 11-16, 17+). Empathy starts from 37,402 and samples 12,800 across the same buckets with shortfall redistribution for the 17+ bucket. Commonsense uses all 12,901 available dialogues minus 39 duplicates, producing 12,862, each requiring two batch calls (one for dimensional scoring, one for dialogue-level scoring). Multicultural uses all 12,816 rows directly — each row is one cross-cultural conversation with rich metadata.

### The 23 Scoring Dimensions

Every entry in stage_1.jsonl carries all 23 dimension keys regardless of domain. The domain's own scores are filled as floats in 0.0–1.0, the other 17–18 are null. This uniform schema simplifies downstream processing.

Some dimensions are marked as "characterizing" — these are the ones that define the domain's core quality and drive all Stage 2 decisions. The distinction matters because Stage 2 only targets characterizing dimensions when generating contrastive variants.

**Coherence** has 6 dimensions scored on a 1-5 integer scale, normalized to 0-1. Two are characterizing:

| Dimension | Mean | Std | Characterizing |
|-----------|------|-----|:-:|
| co_topic_coherence | 0.761 | 0.280 | yes |
| co_logical_consistency | 0.715 | 0.288 | yes |
| co_temporal_causal_coherence | 0.762 | 0.269 | |
| co_discourse_structure | 0.666 | 0.282 | |
| co_mutual_grounding | 0.663 | 0.299 | |
| co_overall_coherence_score | 0.712 | 0.281 | |

The characterizing average across the 11,517 train entries is 0.738 (std 0.267). Most coherence dialogues score high, so Stage 2 predominantly degrades them to create contrastive pairs.

**Empathy** has 6 dimensions, also 1-5 → 0-1. Two are characterizing:

| Dimension | Mean | Std | Characterizing |
|-----------|------|-----|:-:|
| em_emotional_awareness | 0.307 | 0.257 | yes |
| em_perspective_taking | 0.265 | 0.234 | yes |
| em_emotional_validation | 0.208 | 0.255 | |
| em_supportive_engagement | 0.228 | 0.258 | |
| em_helpful_response | 0.204 | 0.236 | |
| em_overall_empathy_score | 0.249 | 0.245 | |

The characterizing average is 0.286 (std 0.238). Empathy dialogues score low across the board — the source corpus contains many dismissive or unhelpful responses, making it ideal for generating improved variants.

**Commonsense** has 6 dimensions on a native 0-1 scale. Four are characterizing — each maps to specific ATOMIC commonsense relations:

| Dimension | Mean | Std | Characterizing | ATOMIC Relations |
|-----------|------|-----|:-:|----------------|
| cs_causality | 0.138 | 0.211 | yes | HinderedBy, IsAfter |
| cs_consistency | 0.319 | 0.221 | yes | xAttr |
| cs_reaction | 0.338 | 0.260 | yes | xReact, oReact |
| cs_desire | 0.160 | 0.234 | yes | xWant, oWant |
| cs_coherence | 0.917 | 0.140 | | |
| cs_empathy | 0.608 | 0.213 | | |

The characterizing average is 0.238 (std 0.142). Causality scores extremely low (mean 0.138), while consistency and reaction score roughly double that. The non-characterizing dimensions (coherence at 0.917, empathy at 0.608) are high — these dialogues are coherent and somewhat empathetic, they just lack commonsense grounding.

The 4-characterizing-dim distribution across the 11,576 train entries:

```
avg score    count    pct
0.0 – 0.1    1,799   15.5%  ███████████████
0.1 – 0.2    2,768   23.9%  ████████████████████████
0.2 – 0.3    3,140   27.1%  ███████████████████████████
0.3 – 0.4    2,320   20.0%  ████████████████████
0.4 – 0.5      898    7.8%  ████████
0.5 – 0.6      481    4.2%  ████
0.6 – 0.7      143    1.2%  █
0.7 – 0.8       24    0.2%
0.8 – 0.9        3    0.0%
```

86.6% of entries fall below the 0.4 low-tier threshold. This heavy left skew means nearly all commonsense variants will be improvements — the originals are weak on commonsense and need strengthening.

**Multicultural** has 5 dimensions on a native 0-1 scale. Two are characterizing. The figures below are post-v2-rescore (5-anchor scale + contrast clauses) over the current 14,212 multicultural entries (12,816 original cross-cultural dialogues + 1,396 self-taught negatives):

| Dimension | Mean | Std | Characterizing |
|-----------|------|-----|:-:|
| mu_cultural_value | 0.607 | 0.276 | yes |
| mu_cultural_specificity | 0.770 | 0.240 | yes |
| mu_naturalness | 0.841 | 0.080 | |
| mu_coherence | 0.899 | 0.067 | |
| mu_empathy | 0.341 | 0.284 | |

The characterizing average is 0.688 (std 0.184). The original cross-cultural dialogues were generated with explicit cultural prompts so most score high, but the rescore corrected the prior inflation (`mu_cultural_value` was 0.698→0.607 after dropping the 0.0/1.0-only anchors) and the negatives pull the lower tail down. Stage 2 predominantly degrades the high-scoring originals.

### Stage 1 Output

52,584 scored entries total after the v2 rebuild (initial 51,264 — 14 batch failures held out — minus 76 placeholder-corrupted dialogues, plus 1,396 retained multicultural self-taught negatives).

| Domain | Total | Train | Test |
|--------|------:|------:|-----:|
| Coherence | 12,796 | 11,516 | 1,280 |
| Empathy | 12,714 | 11,440 | 1,274 |
| Commonsense | 12,862 | 11,576 | 1,286 |
| Multicultural | 14,212 | 12,930 | 1,282 |
| **Total** | **52,584** | **47,462** | **5,122** |

Split is 90/10, stratified by domain, seed=42. The 1,396 negatives are train-only.


## Stage 2 — Contrastive Variant Generation

Stage 2 processes all 51,264 entries across both splits. Each dialogue produces multiple variant candidates — 3 for non-commonsense domains (1 global + 2 dimension-targeted) and 4 for commonsense (1 per characterizing dim) — yielding ~166,654 total candidates. Each candidate is a contrastive rewrite that is either better or worse than the original on specific dimensions. After independent evaluation by a second LLM judge, the evaluated pairs become DPO training data. Each pair inherits the split from its source dialogue — test pairs serve as ground truths for model evaluation.

### Tier Classification

The average of characterizing dimension scores places each entry into a quality tier that determines the generation strategy:

| Tier | Char. avg range | Generation strategy | Direction |
|------|:-:|---|---|
| Low | < 0.40 | Global improve + per-dim targeted | positive |
| Medium-low | 0.40 – 0.55 | Global improve + per-dim targeted | positive |
| Medium-high | 0.55 – 0.70 | Global degrade + per-dim targeted | negative |
| High | >= 0.70 | Global degrade + per-dim targeted | negative |

Every non-commonsense dialogue produces 3 candidates regardless of tier: 1 global (improve or degrade based on direction) + 1 dimension-targeted per characterizing dim. Every commonsense dialogue produces 4 dimension-targeted candidates (one per characterizing dim, no global). This produces more diverse generation requests and richer contrastive signal per dialogue.

The resulting tier distribution reflects each domain's score profile:

| Domain | Low | Medium | High | Char. avg |
|--------|----:|-------:|-----:|:-:|
| Coherence | 2,421 (18.9%) | 2,773 (21.7%) | 7,603 (59.4%) | 0.738 |
| Empathy | 9,452 (73.9%) | 2,403 (18.8%) | 934 (7.3%) | 0.286 |
| Commonsense | 11,139 (86.6%) | 1,691 (13.1%) | 32 (0.2%) | 0.238 |
| Multicultural | 851 (6.0%) | 5,409 (38.1%) | 7,952 (56.0%) | 0.688 |

Coherence and multicultural are high-quality corpora — their pairs come mainly from degrading good dialogues. Empathy and commonsense are low-quality — their pairs come from improving weak dialogues. This is the natural consequence of the source data, not a design choice. The multicultural row reflects the post-rescore distribution over 14,212 entries — the rescore plus the lower-scoring negatives shifted more entries into the Low/Medium tiers than the original 12,816-row table showed.

The direction split confirms this pattern (multicultural is post-rescore over 14,212):

| Domain | Positive (improve) | Negative (degrade) |
|--------|---:|---:|
| Coherence | 3,809 (29.8%) | 8,988 (70.2%) |
| Empathy | 11,047 (86.4%) | 1,742 (13.6%) |
| Commonsense | 12,529 (97.4%) | 333 (2.6%) |
| Multicultural | 3,154 (22.2%) | 11,058 (77.8%) |

### Multi-Variant Candidate Volume

Each dialogue fans out to multiple candidates. The variant type determines how the generation prompt is structured.

| Domain | Dialogues | Variants/dlg | Total candidates | global | dimension_targeted |
|--------|----------:|:---:|---:|---:|---:|
| Coherence | 12,797 | 3 | 38,391 | 12,797 | 25,594 |
| Empathy | 12,789 | 3 | 38,367 | 12,789 | 25,578 |
| Multicultural | 12,816 | 3 | 38,448 | 12,816 | 25,632 |
| Commonsense | 12,862 | 4 | 51,448 | — | 51,448 |
| **Total** | **51,264** | | **~166,654** | **38,402** | **128,252** |

For non-commonsense domains, each dialogue's global candidate uses GLOBAL_IMPROVE or GLOBAL_DEGRADE based on the tier, and the 2 dimension-targeted candidates each focus on a single characterizing dim. Commonsense produces only dimension-targeted candidates — one per dim, no global — because isolating a single commonsense reasoning skill per pair produces cleaner DPO signal. Of the 12,862 commonsense entries, ~25% have gold ATOMIC relation labels preserved in domain_metadata as reference, though targeting now covers all 4 dimensions regardless.

### Generation Prompt Structure

Each generation prompt contains the dialogue split into a shared prefix (unchanged context) and an original continuation (the part to rewrite). The continuation length is sampled from a weighted distribution: 1 turn (15%), 3 turns (40%), 5 turns (30%), 7 turns (15%).

The prompt also includes a dimension rubric from config.py that tells the model what each target dimension means. For global variants the rubric covers all characterizing dimensions; for dimension-targeted variants it covers only the targets.

Generation prompts include several quality controls. A direction-aware CURRENT SCORES block provides magnitude guidance that accounts for floor/ceiling effects — a degrade prompt targeting an already-low score produces "make a minimal, subtle change" rather than gibberish. Targeted prompts include a NON-TARGET DIMENSIONS block listing dimensions to preserve, and global degrade prompts similarly protect non-characterizing dimensions. All degrade prompts carry subtlety constraints to prevent obviously broken text, and the output format uses explicit JSON role objects rather than pipe notation.

All four domains include a GENERATION GUIDANCE block with concrete writing strategies per dimension and direction (e.g., improving cs_causality → "ensure clear cause-effect relationships"; degrading cs_reaction → "reactions slightly off — a bit too muted or intense"). Multicultural prompts additionally include a CULTURAL CONTEXT block with 15 fields from the raw data: both countries, demographics, cultural perspectives, value statements, social norms, cross-cultural prejudices, and emotional dynamics. The direction qualifier reads "more culturally grounded" for improve and "less culturally grounded" for degrade.

Real example prompts for all four domains are saved in `data/stage_2_prompts_preview.txt`.

The ~167k candidates are written to shards of ~5,000 entries each for batch API submission.

### Generation Results

34 shards submitted to Azure OpenAI batch API (gpt-5.1-batch) in two runs (25 original + 9 for previously uncovered dialogues), 5 concurrent jobs each, all completed successfully.

| Metric | Value |
|--------|-------|
| Manifest entries | 166,657 |
| Shards submitted | 34 (5,000 entries each) |
| **Forwarded to eval** | **166,365** |

The parser handles malformed responses (null content, list-type content, capitalized roles). V1 outputs are archived under `data/archive/stage_2/20260312_pipeline/backup/`.

### Evaluation

A second LLM judge (gpt-5.1 via Azure batch API) scores the variant continuation on all domain dimensions (0.0–1.0). The judge sees the shared prefix and variant only — Stage 1 scores serve as ground truth for the original. Classification uses characterizing dimensions; all scores are stored.

The eval prompt includes SCORING ANCHORS (0.0–0.2 "severely deficient" through 0.9–1.0 "excellent") and prefix-continuation reasoning instructions. Multicultural entries receive the full cultural context block matching the generation prompt.

### 6-Label Classification System

Each label encodes variant origin (global vs targeted) and signal quality (clean pass, coarse pass, or flipped). Margin threshold is configurable (0.05 in v2).

| Label | Applies to | Rule | Pair formation |
|---|---|---|---|
| **global_pass** | Global variants | Avg of all char dims moved in intended direction by > margin | Standard: intended direction |
| **target_pass** | Targeted variants | Target dim moved correctly by > margin, non-targets within ±stability | Standard: intended direction |
| **target_coarse_pass** | Targeted variants | Target dim moved correctly by > margin, non-targets drifted beyond ±stability | Standard: intended direction |
| **global_flip_pass** | Global variants | Avg moved OPPOSITE to intended by > margin | Swap chosen/rejected, intent_followed=false |
| **target_flip_pass** | Targeted variants | Target dim moved OPPOSITE to intended by > margin | Swap chosen/rejected, intent_followed=false |
| **reject** | Both | Target/avg moved ≤ margin in either direction, or missing scores | Discarded — no contrastive signal |

The ±0.20 stability threshold in `target_pass` vs `target_coarse_pass` classifies non-target co-movement quality — it no longer causes rejection. The two thresholds (margin 0.05, stability 0.20) serve different purposes and are set independently.

### Evaluation Results

34 eval shards (166,365 entries) in two runs, 5 concurrent jobs each, all completed successfully. The label breakdown below is from the original run; the post-rebuild canonical (`data/stage_2.jsonl`) has 160,858 pairs after dropping 8,214 role-malformed pairs (7,630 replaced via three regen passes, 92.9%) and 314 more in the placeholder-repair patch (313 placeholder-corrupted + 1 score-inversion, 66 replaced). Net change from the original 161,690: −832 pairs.

**Original run: 161,690 usable pairs**

| Domain | global_pass | target_pass | target_coarse_pass | global_flip_pass | target_flip_pass | **Usable (orig)** | **Final (v2)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Empathy | 12,724 | 1,377 | 23,729 | 4 | 199 | 38,033 | **37,750** |
| Coherence | 12,243 | 13,723 | 9,553 | 241 | 1,670 | 37,430 | **37,197** |
| Multicultural | 11,954 | 8,820 | 11,648 | 300 | 2,470 | 35,192 | **35,009** |
| Commonsense | 0 | 73 | 50,149 | 0 | 813 | 51,035 | **50,902** |
| **Total** | **36,921** | **23,993** | **95,079** | **545** | **5,152** | **161,690** | **160,858** |

5,697 usable pairs (3.5%) have `intent_followed=false` (flip_pass pairs with swapped chosen/rejected).

The variant type distribution across domains:

| Domain | Global improve | Global degrade | Dimension targeted | Total |
|--------|---:|---:|---:|---:|
| Empathy | 10,991 | 1,737 | 25,305 | 38,033 |
| Coherence | 3,803 | 8,681 | 24,946 | 37,430 |
| Multicultural | 1,214 | 11,040 | 22,938 | 35,192 |
| Commonsense | — | — | 51,035 | 51,035 |
| **Total** | **16,008** | **21,458** | **124,224** | **161,690** |

Within dimension-targeted pairs, the per-dimension breakdown:

| Dimension | target_pass | target_coarse_pass | target_flip_pass | Total |
|-----------|---:|---:|---:|---:|
| co_logical_consistency | 7,492 | 4,610 | 533 | 12,635 |
| co_topic_coherence | 6,231 | 4,943 | 1,137 | 12,311 |
| em_emotional_awareness | 1,033 | 11,632 | 30 | 12,695 |
| em_perspective_taking | 344 | 12,097 | 169 | 12,610 |
| mu_cultural_specificity | 7,185 | 4,804 | 146 | 12,135 |
| mu_cultural_value | 1,635 | 6,844 | 2,324 | 10,803 |
| cs_causality | 31 | 12,674 | 105 | 12,810 |
| cs_consistency | 16 | 12,595 | 168 | 12,779 |
| cs_desire | 15 | 12,473 | 289 | 12,777 |
| cs_reaction | 11 | 12,407 | 251 | 12,669 |
| **Total** | **23,993** | **95,079** | **5,152** | **124,224** |

Pair difficulty is based on the margin between chosen and rejected scores:

| Domain | Easy (≥ 0.30) | Medium (0.15–0.30) | Hard (< 0.15) | Total |
|--------|---:|---:|---:|---:|
| Empathy | 35,991 (94.6%) | 1,432 (3.8%) | 610 (1.6%) | 38,033 |
| Coherence | 25,299 (67.6%) | 6,378 (17.0%) | 5,753 (15.4%) | 37,430 |
| Commonsense | 46,897 (91.9%) | 2,336 (4.6%) | 1,802 (3.5%) | 51,035 |
| Multicultural | 17,642 (50.1%) | 10,900 (31.0%) | 6,650 (18.9%) | 35,192 |
| **Total** | **125,829 (77.8%)** | **21,046 (13.0%)** | **14,815 (9.2%)** | **161,690** |

Multicultural is the hardest domain (50% easy, 19% hard).

The contrastive direction shows which side of each pair was chosen:

| Domain | Positive (variant is chosen) | Negative (original is chosen) | Total |
|--------|---:|---:|---:|
| Empathy | 32,946 (86.6%) | 5,087 (13.4%) | 38,033 |
| Coherence | 11,390 (30.4%) | 26,040 (69.6%) | 37,430 |
| Commonsense | 49,863 (97.7%) | 1,172 (2.3%) | 51,035 |
| Multicultural | 3,154 (9.0%) | 32,038 (91.0%) | 35,192 |
| **Total** | **97,353 (60.2%)** | **64,337 (39.8%)** | **161,690** |

Empathy and commonsense originals score low → variants improve → variant is chosen. Coherence and multicultural originals score high → variants degrade → original is chosen.

Average characterizing dimension scores across all pairs:

| Domain | Dimension | Original avg | Variant avg | Shift |
|--------|-----------|:---:|:---:|:---:|
| Coherence | co_topic_coherence | 0.759 | 0.800 | +0.041 |
| Coherence | co_logical_consistency | 0.715 | 0.646 | -0.069 |
| Empathy | em_emotional_awareness | 0.305 | 0.856 | +0.551 |
| Empathy | em_perspective_taking | 0.264 | 0.835 | +0.571 |
| Commonsense | cs_causality | 0.137 | 0.892 | +0.756 |
| Commonsense | cs_consistency | 0.317 | 0.904 | +0.587 |
| Commonsense | cs_reaction | 0.335 | 0.877 | +0.542 |
| Commonsense | cs_desire | 0.158 | 0.891 | +0.733 |
| Multicultural | mu_cultural_value | 0.619 | 0.706 | +0.088 |
| Multicultural | mu_cultural_specificity | 0.791 | 0.515 | -0.276 |

Empathy and commonsense show large positive shifts. Coherence shows mixed shifts — subtler prompts produce nuanced changes rather than uniformly crushing all dimensions. The multicultural rows are post-v2-rescore: `mu_cultural_specificity` shifts clearly (-0.276) and `mu_cultural_value` now shows a small positive shift (+0.088) rather than the flat ~0 of the original run — the rescored evaluator picks up value changes the original rubric missed.

### Pair Construction

Standard labels: positive-direction → variant is chosen; negative-direction → original is chosen. Flip labels: chosen/rejected are swapped; `contrast.intent_followed` is set to false.

Each pair carries a difficulty label (easy ≥ 0.30, medium 0.15–0.30, hard < 0.15) and an `S2D-NNNNNN` ID. The original run numbered pairs `S2D-000001` through `S2D-161690`; the v2 rebuild dropped the malformed and placeholder pairs (leaving gaps in that range) and appended regenerated pairs from `S2D-153477` onward, so the current canonical runs up to `S2D-169386` with 160,858 pairs. Score ordering is validated on the effective dims before emission via `mean(chosen) > mean(rejected)` — see [Known Risks](#known-risks) for the per-dim implication.


## Changes

The pipeline went through several data-level fixes before reaching its current state. A falsy-value bug where `x or y` dropped 0.0 caused 10,560 commonsense batch failures, fixed by switching to `x is None`. Multicultural dialogues had escaped newlines — the raw CSV stored literal `\n` strings, collapsing all 12,816 dialogues to single messages — fixed in parse_dialogue_to_messages. The score template was unified so every entry carries all 23 dimension keys regardless of domain (domain scores filled, others null).

On the content side, generation and eval prompts now receive 15 cultural metadata fields for multicultural entries, backtracked from the raw CSV via each dialogue's uid. Gold ATOMIC relation labels for ~25% of commonsense dialogues are loaded at merge time into domain_metadata. cs_reaction and cs_desire were promoted to characterizing dimensions, enabling 4-dim full-coverage targeting for commonsense.

Structurally, each dialogue now produces 3–4 candidates (a 3.25x increase to ~167k) with custom IDs refactored to `s2g-{id}-{gimp|gdeg|dt-{dim}}` to prevent collisions. "verify/verification" was renamed to "eval/evaluation" across the codebase, `data/output/` became `data/stage_1/`, and compound pair IDs were simplified to sequential `S2D-{n:06d}`.

The v2 prompt quality pass fixed 17 issues across 3 review rounds with 28 new tests, followed by a full stage 2 re-run. Generation prompts gained direction-aware score context with floor/ceiling magnitude guidance, non-target dimension constraints, subtlety constraints on all degrade variants, per-domain writing strategies, and subtler commonsense degrade approaches. Multicultural prompts received explicit direction qualifiers ("more/less culturally grounded") and cultural context positioned before instructions. Eval prompts gained five-level score anchoring (0.0–1.0) and prefix-continuation reasoning. Across both, pipe notation was replaced with explicit JSON role objects.


## v2 Rebuild (May 2026)

A pre-training audit on `data/stage_2.jsonl` surfaced four issues that propagated silently from the original pipeline: a role-boundary bug that silently dropped 5.1% of pairs at training time, an inflated multicultural rubric (mean `mu_cultural_value` 0.70 with no contrast clauses, Spearman ~0.29 in Figure 3), absence of self-taught negative examples, and 313 pairs containing degenerate `xxxx`/`XXXX` placeholder turns inherited from the original LM generator. The rebuild addressed each in sequence; the canonical numbers above reflect the result.

### Phase 1 — role-boundary guardrails

Stage 2 pairs require `chosen[0].role` and `rejected[0].role` to equal the opposite of `messages[-1].role` so the concatenation keeps user/assistant alternation; otherwise the reward-model trainer's `apply_chat_template` silently rejects the pair. 8,214 pairs (5.1%) violated this. Three minimal edits in `scripts/stage_2/{prompts,generate,pairs}.py` enforce the invariant: the generation prompt now embeds an `expected_first_role` derived from the prefix tail, the generation parser drops continuations that disobey, and `build_pairs` skips with a `skipped_bad_role` counter. The 8,214 already-malformed pairs were tagged for regen.

### Phase 2 — multicultural rescore

The original `MULTICULTURAL_PROMPT` had only 0.0/1.0 anchors and no contrast clauses, inflating `mu_cultural_value` to mean 0.70 with 80% in the upper quintile. Rewrite added a 5-anchor scale (0.0/0.25/0.5/0.75/1.0), forced-contrast clauses, CoT-before-score, reflection, an adversarial-dialogue clause (so genuine pro-value embodiments aren't flagged as rejection), and an ATTRIBUTION CHECK to prevent cross-dim contamination. Three rescore iterations later (gpt-5.1 → gpt-5.1 patched → gpt-5.1 with `--only-dim` flag) the final canonical was lifted by 856 tier-1+2+3 dialogues re-rescored by Opus 4.7 subagents (correcting the rejection-stance bias gpt-5.1 exhibited on adversarial cases). Sonnet QA verdict: APPROVE_WITH_CAVEATS 31/33 = 94%. Distribution: mean 0.61, std 0.27, distinct 23, quintiles 8/12/20/25/34%.

### Phase 3 — self-taught negatives

Following Wang et al. 2024 (Self-Taught Evaluators), 1,400 negative dialogues were generated for the multicultural domain: an LLM invented a "modified instruction" that shifts one specific axis of the original cross-cultural setup (a different cultural value, a deculturated setting, a non-conversational format, or a non-linear structure) and produced a high-quality response to that modified instruction. The output is a good response for the modified task but a bad response for the original — making it a structurally weak example on the target dim without being gameable by surface patterns. Generation: Claude Sonnet via Agent-tool subagents (cross-model against the gpt-5.1 stage-1 judge), 7 countries × 4 target dims × 50 dialogues. Scoring: gpt-5.1 sync, 8 workers. 1,399 of 1,400 scored successfully; 1,396 were retained in the canonical (3 — `S1D-051436/051458/051462` — were dropped later because the "formal hearing" modified-instruction format produced multi-party transcripts that break the binary user/assistant alternation). Integrated into stage_1 with dialogue IDs `S1D-051265..052663`, source IDs `mu-NS-<COUNTRY>-NNNNNN`, and a nested `negative_sampling` metadata block recording method, generator model, target dim, modified instruction, and source uid.

### Phase 4 — regen passes (8,214 → 7,630 recovered, 92.9%)

The Azure batch deployment was degraded the day of the regen (2.8h stuck in validating). The pipeline pivoted to synchronous `gpt-5.1` (`scripts/regen/sync.py`), running through generation and eval as drop-in replacements. Three passes:

- **Pass 1** — gpt-5.1 sync at T=0.0. 8,214 candidates → 8,170 variants → 6,552 pairs (79.8% recovery).
- **Pass 2** — gpt-5.1 sync at T=0.5 on the 1,662 not recovered in pass 1. → 749 pairs (45.1% additional).
- **Pass 3** — Opus 4.7 subagents (Agent tool, not API; 10 waves × 4 parallel × 25 items/chunk) on the 913 still missing. → 329 pairs (36% additional). 0 role-compliance failures across all 913 — Opus's structural fidelity on the schema was perfect.

Sum across the three passes: 7,630 of 8,214 (92.9%). The remaining 584 are dominated by Azure content-filter rejections on `cs_reaction` items and dimension-direction pairs where the requested shift is structurally incompatible with the source dialogue.

### Phase 5 — placeholder repair

313 stage_2 pairs and 105 stage_1 dialogues contained `xxxx`/`XXXX` placeholders from degenerate LM outputs in the original generator. 75 dialogues had placeholders in user turns (no anchor) and were dropped; 1 coherence dialogue (`S1D-000128`) had an `XXXXXXXXXX` variant the empathy-focused repair subagents preserved as-is, so it was added to the dropped set (76 dropped total). 29 dialogues had user turns intact and were repaired by Opus subagents that rewrote each `xxxx` assistant turn as a plausible empathetic response calibrated to the dialogue's existing scores. The ~90 stage_2 pairs depending on repaired sources were rebuilt via `scripts/regen/{build,sync,finalize}.py`, yielding 66 pairs after eval filtering. `scripts/stage_2/patch.py` then dropped 314 pairs from the canonical (313 placeholder-corrupted + 1 score-inversion pair flagged by QA, see below) and appended the 66 new pairs (IDs `S2D-169321..169386`). Final: 0 placeholders corpus-wide (regex `[xX]{4,}`); net pair loss from this patch is 248.

### Cross-model QA

A Sonnet subagent audited a stratified 40-pair sample (5 clean + 5 regen per domain). Structural: 40/40 clean. Content: 38/40 usable, 1 hard fail (multicultural `S2D-075642` with score inversion across 3 dims — dropped from the canonical), 15 minor caveats. Regen pairs hold parity with clean pairs. The audit's top-3 flagged patterns were the `cs_coherence` scorer over-rewarding verbatim echo fragments, the `mu_empathy` scorer under-scoring culturally-compressed authentic dialogue, and residual `xxxx` placeholder tokens (now removed). The first two map to the scorer-bias limitation in [`data/audits/qa/known_limitations.md`](data/audits/qa/known_limitations.md), which also documents the mean-validator's per-dim inversions (12% corpus-wide) and an estimated ~2.5% cross-model judgment-disagreement noise floor.


## Known Risks

See [`data/audits/qa/known_limitations.md`](data/audits/qa/known_limitations.md) for the three v2 limitations affecting training (mean-validator per-dim inversions, scorer bias on `cs_coherence` / `mu_empathy`, LM-judge noise).

76.6% of commonsense dialogues have exactly 5 messages. When continuation length exceeds the original, the fallback uses a 1-message prefix, so generated variants may not meaningfully correspond to the original. Separately, Stage 2 eval uses a single-continuation format with score anchoring while Stage 1 uses domain-specific rubrics — this asymmetry is by design (independent validation) but could affect pass rates.

The label distribution is dominated by target_coarse_pass at 95,079 pairs (59%), which carry valid target signal but noisy non-target behavior. DPO training may want to weight by label. Another 5,697 pairs (3.5%) are flip_pass with reversed contrastive direction, and mu_cultural_value has the highest flip rate at 22%.

Margin threshold sensitivity varies by domain. The Coherence/Empathy/Commonsense columns are from the original run; the Multicultural column is recomputed on the post-v2-rescore canonical (35,009 pairs):

| Threshold | Coherence (orig) | Empathy (orig) | Commonsense (orig) | Multicultural (v2) |
|:-:|---:|---:|---:|---:|
| **0.05** | **6.4%** | **0.7%** | **1.1%** | **2.1%** |
| 0.10 | 15.0% | 1.2% | 1.3% | 7.5% |
| 0.15 | 23.7% | 1.9% | 1.6% | 15.9% |
| 0.20 | 32.7% | 3.3% | 1.9% | 27.7% |
| 0.25 | 39.4% | 5.0% | 2.3% | 41.3% |
| 0.30 | 43.0% | 6.3% | 2.7% | 54.9% |
| 0.40 | 60.9% | 13.3% | 5.3% | 80.3% |
| 0.50 | 80.5% | 26.1% | 14.5% | 92.3% |
| 0.60 | 91.3% | 35.9% | 31.6% | 97.6% |

Coherence is sensitive (32.7% lost at 0.20) because subtler degradation prompts produce smaller margins. Multicultural is robust at the operating threshold (2.1% lost at 0.05) but fragile above 0.25 — half its pairs disappear by 0.30. Empathy and commonsense are robust across the board.


## File Structure

```
scripts/
  config.py           Domain configs, 23 dimensions, characterizing flags, ATOMIC relation mapping
  models.py           Pydantic models (Stage1Entry, Stage2Candidate, Stage2Variant, Stage2Pair)
  merge_stage_1.py    Merge domain outputs, backtrack multicultural metadata and commonsense gold
  split.py            Train/test stratified split assignment, stage 2 pair relabeling
  batch_runner.py     Async Azure batch API runner
  run_stage_1.py      Stage 1 orchestrator (prepare/run/parse/retry/merge/split)
  run_stage_2.py      Stage 2 orchestrator (select/generate/eval/pairs)
  render_prompts.py   Render all prompt variants for visual review
  analyze_pairs.py    Score statistics, direction distributions, margin threshold sensitivity analysis
  audit.py            Distribution / rescore / qa subcommands used during v2 rebuild
  build_human_eval_multicultural.py  Build data/eval/human_scores_multicultural.jsonl from v1/v2 repos
  stage_1/
    base.py           Shared batch-prep helpers
    prepare_*.py       Per-domain data preparation and sampling
    parse_results.py  Batch result parsing and score normalization
    retry.py          Re-submit failed batch items
    prompts.py        Stage 1 scoring prompts (v2 5-anchor multicultural prompt)
  stage_2/
    select.py         Multi-variant candidate selection, tier classification, per-dim targeting
    prompts.py        Generation and eval templates with role-boundary guardrail
    generate.py       Variant generation entry builder + parse_generation_results
    eval.py           6-label classification + parse_eval_results
    pairs.py          Preference pair construction with flip logic and role-boundary check
    update_scores.py  Filter malformed + refresh multicultural stage_1_scores in stage_2
    splice.py         Combine main.jsonl + regen main.jsonl into canonical stage_2.jsonl
    patch.py          Drop prejudicial pair_ids + append new pairs
  regen/
    build.py          Build Stage2Candidates + gen shards from a malformed_pairs file
    sync.py           Synchronous Azure OpenAI runner with --temperature override (gpt-5.1 fallback when batch is degraded)
    finalize.py       parse-gen + parse-eval-and-build subcommands
  negative_sampling/
    seeds.py          Sample multicultural train rows for negative generation
    wang_prompts.py   Wang et al. 2024 Self-Taught Evaluator templates per target dim
    run.py            Render Wang prompts into chunk files for subagent dispatch
    score.py          Sync gpt-5.1 scoring of generated negatives
    integrate.py      Merge scored negatives into stage_1.jsonl with new IDs and metadata

data/
  stage_1.jsonl                       52,584 dialogues — canonical, all 23 dims, includes 1,396 self-taught negatives
  stage_2.jsonl                       160,858 preference pairs — canonical, post-v2-rebuild
  stage_1_template.json               Human-readable Stage 1 format reference
  stage_2_template.json               Human-readable Stage 2 pair format reference
  stage_2_prompts_preview.txt         One real generation prompt per domain
  stage_1/                            Per-domain batch outputs + repair/ (Opus chunks for placeholder fix)
  stage_2/main.jsonl                  Pre-regen surviving pairs (intermediate of the rebuild)
  stage_2/regen/                      First regen pass — {candidates, manifest_gen, manifest_eval,
                                      shards_gen, shards_eval, variants, main}.jsonl at this level
  stage_2/regen/{recovery,opus,repair}/   Passes 2, 3, and the placeholder-repair pass, each with
                                          the same set of files scoped to its subset
  input/multicultural/
    countries/{train,test}/<CODE>.csv   Per-country source rows
    negative_sampling/                  Wang-style dataset + scoring outputs + README.md
    resources/                          qid_meaning.csv (+ .bak), statements.csv, norms.csv,
                                        prejudices.csv, cultural_items.json, drivers/prompts_reference.py
  audits/qa/
    known_limitations.md              v2 rebuild's surviving limitations for the RM team
    malformed_pairs.jsonl             8,214 role-boundary-violating pair_ids the rebuild fixed
    stage_2_audit.md                  Sonnet phase-gate audit report
    stage_2_sample.jsonl              Stratified 40-pair sample used for audit
    readme_numbers_audit.md           Fact-check report for this README's numbers
    readme_prose_audit.md             Fact-check report for this README's prose/structure
  archive/                            Per-folder .changelog + date-named snapshots
    stage_1/{<date>T<HHMM>.jsonl or <date>_<descr>.jsonl, ...}
    stage_2/{<date>T<HHMM>.jsonl, 20260312_pipeline/, ...}
    multicultural/{.changelog, 20260217/{countries,manifest.jsonl,shards}/}
```

### Data Split Summary

| Split | Stage 1 entries | Stage 2 pairs | Purpose |
|-------|----------------:|--------------:|---------|
| Train | 47,462 | 144,767 | DPO training pairs |
| Test | 5,122 | 16,091 | Ground truth pairs for model evaluation |
| **Total** | **52,584** | **160,858** | |

Both files carry a `split` field. Stage 2 processes all splits — each pair inherits the split from its source dialogue. The self-taught negatives are train-only.
