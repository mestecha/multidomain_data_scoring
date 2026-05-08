# Multi-Domain Data Scoring 

Last updated: 2026-03-12


## Purpose

This pipeline builds preference pairs for DPO training. It takes raw dialogues from four quality domains — coherence, empathy, commonsense, and multicultural — scores them on domain-specific dimensions, then generates contrastive rewrites where one version is clearly better than the other. The final output is a set of (chosen, rejected) dialogue continuation pairs that teach a model to distinguish quality along specific dimensions.


## Pipeline Overview

```
Raw dialogues (4 corpora, ~113k total)
    ↓ sample + deduplicate
51,264 dialogues
    ↓ Stage 1: GPT judge scores each on its domain's dimensions
51,264 scored entries (stage_1.jsonl, all 23 dims per entry)
    ↓ stratified split (seed=42)
46,137 train  ·  5,127 test
    ↓                ↓
Stage 2          Stage 2 (test pairs = ground truths)
    ↓                ↓
~167k candidates (3-4 variants per dialogue)
    ↓ GPT batch API: generate contrastive rewrites
    ↓ GPT batch API: evaluate both versions scored independently
    ↓ 6-label classification (margin > 0.05)
    ↓ flip-pass recovery for opposite-direction variants
161,690 preference pairs (stage_2.jsonl)
    ↓
145,522 train pairs  ·  16,168 test pairs (10.0%)
```

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

**Multicultural** has 5 dimensions on a native 0-1 scale. Two are characterizing:

| Dimension | Mean | Std | Characterizing |
|-----------|------|-----|:-:|
| mu_cultural_value | 0.698 | 0.271 | yes |
| mu_cultural_specificity | 0.857 | 0.142 | yes |
| mu_naturalness | 0.897 | 0.049 | |
| mu_coherence | 0.899 | 0.057 | |
| mu_empathy | 0.299 | 0.250 | |

The characterizing average is 0.778 (std 0.154). These dialogues are already culturally grounded (they were generated with explicit cultural prompts), so Stage 2 predominantly degrades them.

### Stage 1 Output

51,264 scored entries total. 14 batch failures were held out (99.97% success rate).

| Domain | Total | Train | Test |
|--------|------:|------:|-----:|
| Coherence | 12,797 | 11,517 | 1,280 |
| Empathy | 12,789 | 11,510 | 1,279 |
| Commonsense | 12,862 | 11,576 | 1,286 |
| Multicultural | 12,816 | 11,534 | 1,282 |
| **Total** | **51,264** | **46,137** | **5,127** |

Split is 90/10, stratified by domain, seed=42.


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
| Multicultural | 219 (1.7%) | 2,946 (23.0%) | 9,651 (75.3%) | 0.777 |

Coherence and multicultural are high-quality corpora — their pairs come mainly from degrading good dialogues. Empathy and commonsense are low-quality — their pairs come from improving weak dialogues. This is the natural consequence of the source data, not a design choice.

The direction split confirms this pattern:

| Domain | Positive (improve) | Negative (degrade) |
|--------|---:|---:|
| Coherence | 3,809 (29.8%) | 8,988 (70.2%) |
| Empathy | 11,047 (86.4%) | 1,742 (13.6%) |
| Commonsense | 12,529 (97.4%) | 333 (2.6%) |
| Multicultural | 1,216 (9.5%) | 11,600 (90.5%) |

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

Real example prompts for all four domains are saved in `data/stage_2_example_prompts.txt`.

The ~167k candidates are written to shards of ~5,000 entries each for batch API submission.

### Generation Results

34 shards submitted to Azure OpenAI batch API (gpt-5.1-batch) in two runs (25 original + 9 for previously uncovered dialogues), 5 concurrent jobs each, all completed successfully.

| Metric | Value |
|--------|-------|
| Manifest entries | 166,657 |
| Shards submitted | 34 (5,000 entries each) |
| **Forwarded to eval** | **166,365** |

The parser handles malformed responses (null content, list-type content, capitalized roles). V1 outputs in `data/stage_2/backup/`.

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

34 eval shards (166,365 entries) in two runs, 5 concurrent jobs each, all completed successfully.

**Overall: 161,690 usable pairs**

| Domain | global_pass | target_pass | target_coarse_pass | global_flip_pass | target_flip_pass | **Usable** |
|---|---:|---:|---:|---:|---:|---:|
| Empathy | 12,724 | 1,377 | 23,729 | 4 | 199 | **38,033** |
| Coherence | 12,243 | 13,723 | 9,553 | 241 | 1,670 | **37,430** |
| Multicultural | 11,954 | 8,820 | 11,648 | 300 | 2,470 | **35,192** |
| Commonsense | 0 | 73 | 50,149 | 0 | 813 | **51,035** |
| **Total** | **36,921** | **23,993** | **95,079** | **545** | **5,152** | **161,690** |

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
| Multicultural | mu_cultural_value | 0.708 | 0.706 | -0.002 |
| Multicultural | mu_cultural_specificity | 0.857 | 0.514 | -0.343 |

Empathy and commonsense show large positive shifts. Coherence shows mixed shifts — subtler prompts produce nuanced changes rather than uniformly crushing all dimensions. Multicultural mu_cultural_specificity shifts clearly (-0.340) while mu_cultural_value stays flat — the evaluator detects specificity changes but not value changes.

### Pair Construction

Standard labels: positive-direction → variant is chosen; negative-direction → original is chosen. Flip labels: chosen/rejected are swapped; `contrast.intent_followed` is set to false.

Each pair carries a difficulty label (easy ≥ 0.30, medium 0.15–0.30, hard < 0.15) and a sequential ID (`S2D-000001` through `S2D-161690`). Score ordering is validated before emission — 0 pairs skipped for bad ordering.


## Changes

The pipeline went through several data-level fixes before reaching its current state. A falsy-value bug where `x or y` dropped 0.0 caused 10,560 commonsense batch failures, fixed by switching to `x is None`. Multicultural dialogues had escaped newlines — the raw CSV stored literal `\n` strings, collapsing all 12,816 dialogues to single messages — fixed in parse_dialogue_to_messages. The score template was unified so every entry carries all 23 dimension keys regardless of domain (domain scores filled, others null).

On the content side, generation and eval prompts now receive 15 cultural metadata fields for multicultural entries, backtracked from the raw CSV via each dialogue's uid. Gold ATOMIC relation labels for ~25% of commonsense dialogues are loaded at merge time into domain_metadata. cs_reaction and cs_desire were promoted to characterizing dimensions, enabling 4-dim full-coverage targeting for commonsense.

Structurally, each dialogue now produces 3–4 candidates (a 3.25x increase to ~167k) with custom IDs refactored to `s2g-{id}-{gimp|gdeg|dt-{dim}}` to prevent collisions. "verify/verification" was renamed to "eval/evaluation" across the codebase, `data/output/` became `data/stage_1/`, and compound pair IDs were simplified to sequential `S2D-{n:06d}`.

The v2 prompt quality pass fixed 17 issues across 3 review rounds with 28 new tests, followed by a full stage 2 re-run. Generation prompts gained direction-aware score context with floor/ceiling magnitude guidance, non-target dimension constraints, subtlety constraints on all degrade variants, per-domain writing strategies, and subtler commonsense degrade approaches. Multicultural prompts received explicit direction qualifiers ("more/less culturally grounded") and cultural context positioned before instructions. Eval prompts gained five-level score anchoring (0.0–1.0) and prefix-continuation reasoning. Across both, pipe notation was replaced with explicit JSON role objects.


## Known Risks

76.6% of commonsense dialogues have exactly 5 messages. When continuation length exceeds the original, the fallback uses a 1-message prefix, so generated variants may not meaningfully correspond to the original. Separately, Stage 2 eval uses a single-continuation format with score anchoring while Stage 1 uses domain-specific rubrics — this asymmetry is by design (independent validation) but could affect pass rates.

The label distribution is dominated by target_coarse_pass at 95,079 pairs (59%), which carry valid target signal but noisy non-target behavior. DPO training may want to weight by label. Another 5,697 pairs (3.5%) are flip_pass with reversed contrastive direction, and mu_cultural_value has the highest flip rate at 22%.

Margin threshold sensitivity varies by domain:

| Threshold | Coherence | Empathy | Commonsense | Multicultural |
|:-:|---:|---:|---:|---:|
| **0.05** | **6.4%** | **0.7%** | **1.1%** | **10.0%** |
| 0.10 | 15.0% | 1.2% | 1.3% | 15.8% |
| 0.15 | 23.7% | 1.9% | 1.6% | 22.9% |
| 0.20 | 32.7% | 3.3% | 1.9% | 37.7% |
| 0.25 | 39.4% | 5.0% | 2.3% | 50.6% |
| 0.30 | 43.0% | 6.3% | 2.7% | 60.8% |
| 0.40 | 60.9% | 13.3% | 5.3% | 84.1% |
| 0.50 | 80.5% | 26.1% | 14.5% | 95.2% |
| 0.60 | 91.3% | 35.9% | 31.6% | 98.9% |

Coherence is sensitive (32.7% lost at 0.20) because subtler degradation prompts produce smaller margins. Multicultural is fragile at higher thresholds. Empathy and commonsense are robust.


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
  render_prompts.py   Render all 32 prompt variants to data/all_prompts_review.txt for visual review
  analyze_pairs.py    Score statistics, direction distributions, margin threshold sensitivity analysis
  stage_1/
    prepare_*.py      Per-domain data preparation and sampling
    parse_results.py  Batch result parsing and score normalization
  stage_2/
    select.py         Multi-variant candidate selection, tier classification, per-dim targeting
    prompts.py        Generation and eval templates, cultural context, domain guidance, score anchoring
    generate.py       Variant generation batch entry builder, collision-safe custom_id scheme
    eval.py           6-label classification (classify_variant), eval result parsing
    pairs.py          Preference pair construction with flip logic, effective direction, label propagation

data/
  stage_1/                              Per-domain batch output files (coherence/, empathy/, commonsense/, multicultural/, holdout_failures.jsonl)
  stage_1.jsonl                         51,264 scored entries (188 MB) — all 23 dims, S1D-* IDs, split: train/test
  stage_2.jsonl                         161,690 preference pairs — S2D-* IDs, 6-label classification (shareable copy of stage_2/pairs.jsonl)
  stage_1_template.json                 Human-readable Stage 1 format reference (categorical fields show all possible values)
  stage_2_template.json                 Human-readable Stage 2 pair format reference (categorical fields show all possible values)
  stage_2_example_prompts.txt           One real generation prompt per domain
  all_prompts_review.txt                All 32 rendered prompt variants for visual review
  stage_2/candidates.jsonl              ~166,654 selected candidates (3-4 per dialogue) with tier, direction, and target dims
  stage_2/shards/                       25 generation request shards (original run)
  stage_2/shards/output/                25 batch output files (original run)
  stage_2/shards_missing/               9 generation request shards (gap-fill run)
  stage_2/shards_missing/output/        9 batch output files (gap-fill run)
  stage_2/manifest_gen.jsonl            124,967 manifest entries (original run)
  stage_2/manifest_gen_missing.jsonl    41,690 manifest entries (gap-fill run)
  stage_2/shards_eval/                  25 eval request shards (original run)
  stage_2/shards_eval/output/           25 batch output files with variant scores (original run)
  stage_2/shards_eval_missing/          9 eval request shards (gap-fill run)
  stage_2/shards_eval_missing/output/   9 batch output files with variant scores (gap-fill run)
  stage_2/manifest_eval.jsonl           124,772 manifest entries (original run)
  stage_2/manifest_eval_missing.jsonl   41,593 manifest entries (gap-fill run)
  stage_2/pairs.jsonl                   161,690 preference pairs — S2D-* IDs, 6-label classification, split: train/test
  stage_2/backup/                       v1 outputs: shards/output_v1/, shards_eval/output_v1_biased/, stage_2.jsonl
```

### Data Split Summary

| Split | Stage 1 entries | Stage 2 pairs | Purpose |
|-------|----------------:|--------------:|---------|
| Train | 46,137 | 145,522 | DPO training pairs |
| Test | 5,127 | 16,168 | Ground truth pairs for model evaluation |
| **Total** | **51,264** | **161,690** | |

Both `stage_1.jsonl` and `stage_2/pairs.jsonl` carry a `split` field. Stage 2 processes all splits — each pair inherits the split from its source dialogue.

348 tests passing across 13 test files.
