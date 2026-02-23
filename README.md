# Multi-Domain Data Scoring 

Last updated: 2026-02-23


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
38,447 train  ·  5,127 val  ·  7,690 test
    ↓                ↓              ↓
Stage 2          HELD OUT       HELD OUT
    ↓
~125k candidates (3-4 variants per dialogue)
    ↓ GPT batch API: generate contrastive rewrites
    ↓ GPT batch API: evaluate both versions scored independently
    ↓ 6-label classification (margin > 0.05)
    ↓ flip-pass recovery for opposite-direction variants
121,326 preference pairs (stage_2.jsonl)
```

Val and test splits are never touched by Stage 2. They exist solely for downstream evaluation of the trained model.


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

The characterizing average across the 9,597 train entries is 0.739 (std 0.268). Most coherence dialogues score high, so Stage 2 predominantly degrades them to create contrastive pairs.

**Empathy** has 6 dimensions, also 1-5 → 0-1. Two are characterizing:

| Dimension | Mean | Std | Characterizing |
|-----------|------|-----|:-:|
| em_emotional_awareness | 0.307 | 0.257 | yes |
| em_perspective_taking | 0.265 | 0.234 | yes |
| em_emotional_validation | 0.208 | 0.255 | |
| em_supportive_engagement | 0.228 | 0.258 | |
| em_helpful_response | 0.204 | 0.236 | |
| em_overall_empathy_score | 0.249 | 0.245 | |

The characterizing average is 0.284 (std 0.237). Empathy dialogues score low across the board — the source corpus contains many dismissive or unhelpful responses, which makes it ideal for generating improved variants.

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

The 4-characterizing-dim distribution across the 9,647 train entries:

```
avg score    count    pct
0.0 – 0.1    1,488   15.4%  ███████████████
0.1 – 0.2    2,340   24.3%  ████████████████████████
0.2 – 0.3    2,606   27.0%  ███████████████████████████
0.3 – 0.4    1,925   20.0%  ████████████████████
0.4 – 0.5      738    7.7%  ████████
0.5 – 0.6      407    4.2%  ████
0.6 – 0.7      118    1.2%  █
0.7 – 0.8       23    0.2%
0.8 – 0.9        2    0.0%
```

86.6% of entries fall below the 0.4 low-tier threshold. This heavy left skew means nearly all commonsense variants will be improvements, which is exactly what you want — the originals are weak on commonsense and need strengthening.

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

| Domain | Total | Train | Val | Test |
|--------|------:|------:|----:|-----:|
| Coherence | 12,797 | 9,597 | 1,280 | 1,920 |
| Empathy | 12,789 | 9,592 | 1,279 | 1,918 |
| Commonsense | 12,862 | 9,647 | 1,286 | 1,929 |
| Multicultural | 12,816 | 9,611 | 1,282 | 1,923 |
| **Total** | **51,264** | **38,447** | **5,127** | **7,690** |

Split is 75/10/15, stratified by domain, seed=42.


## Stage 2 — Contrastive Variant Generation

Stage 2 operates only on the 38,447 train entries. Each dialogue produces multiple variant candidates — 3 for non-commonsense domains (1 global + 2 dimension-targeted) and 4 for commonsense (1 per characterizing dim) — yielding ~124,988 total candidates. Each candidate is a contrastive rewrite that is either better or worse than the original on specific dimensions. After independent evaluation by a second LLM judge, the evaluated pairs become DPO training data.

### Tier Classification

The average of characterizing dimension scores places each entry into a quality tier that determines the generation strategy:

| Tier | Char. avg range | Generation strategy | Direction |
|------|:-:|---|---|
| Low | < 0.40 | Global improve + per-dim targeted | positive |
| Medium-low | 0.40 – 0.55 | Global improve + per-dim targeted | positive |
| Medium-high | 0.55 – 0.70 | Global degrade + per-dim targeted | negative |
| High | >= 0.70 | Global degrade + per-dim targeted | negative |

Every non-commonsense dialogue produces 3 candidates regardless of tier: 1 global (improve or degrade based on direction) + 1 dimension-targeted per characterizing dim. Every commonsense dialogue produces 4 dimension-targeted candidates (one per characterizing dim, no global). This multi-variant strategy gives the batch API more diverse generation requests and the DPO training set richer contrastive signal per dialogue.

The resulting tier distribution reflects each domain's score profile:

| Domain | Low | Medium | High | Char. avg |
|--------|----:|-------:|-----:|:-:|
| Coherence | 1,822 (19.0%) | 2,036 (21.2%) | 5,739 (59.8%) | 0.739 |
| Empathy | 7,133 (74.4%) | 1,772 (18.5%) | 687 (7.2%) | 0.284 |
| Commonsense | 8,359 (86.6%) | 1,263 (13.1%) | 25 (0.3%) | 0.238 |
| Multicultural | 167 (1.7%) | 2,188 (22.8%) | 7,256 (75.5%) | 0.778 |

Coherence and multicultural are high-quality corpora — their pairs come mainly from degrading good dialogues. Empathy and commonsense are low-quality — their pairs come from improving weak dialogues. This is the natural consequence of the source data, not a design choice.

The direction split confirms this pattern:

| Domain | Positive (improve) | Negative (degrade) |
|--------|---:|---:|
| Coherence | 2,860 (29.8%) | 6,737 (70.2%) |
| Empathy | 8,299 (86.5%) | 1,293 (13.5%) |
| Commonsense | 9,401 (97.4%) | 246 (2.6%) |
| Multicultural | 916 (9.5%) | 8,695 (90.5%) |

### Multi-Variant Candidate Volume

Each dialogue fans out to multiple candidates. The variant type determines how the generation prompt is structured.

| Domain | Dialogues | Variants/dlg | Total candidates | global | dimension_targeted |
|--------|----------:|:---:|---:|---:|---:|
| Coherence | 9,597 | 3 | 28,791 | 9,597 | 19,194 |
| Empathy | 9,592 | 3 | 28,776 | 9,592 | 19,184 |
| Multicultural | 9,611 | 3 | 28,833 | 9,611 | 19,222 |
| Commonsense | 9,647 | 4 | 38,588 | — | 38,588 |
| **Total** | **38,447** | | **~124,988** | **28,800** | **96,188** |

For non-commonsense domains, each dialogue's global candidate uses GLOBAL_IMPROVE (positive direction) or GLOBAL_DEGRADE (negative direction) based on the tier. The 2 dimension-targeted candidates each focus on a single characterizing dim. Commonsense produces only dimension-targeted candidates — one per dim — because single-dimension targeting produces cleaner DPO signal for commonsense reasoning.

### Commonsense Full-Coverage Targeting

Commonsense follows a fundamentally different approach from the other domains. Instead of producing a global variant plus per-dim targeted variants, every commonsense dialogue produces exactly 4 dimension-targeted candidates — one per characterizing dim (cs_causality, cs_consistency, cs_reaction, cs_desire). No global variant is generated.

This design decision exists for two reasons. First, the four commonsense characterizing dimensions correspond directly to the seven ATOMIC commonsense relations that the source corpus was built from. Targeting each dimension independently produces more meaningful contrasts than asking the model to simultaneously change causality, consistency, reaction, and desire. Second, single-dimension targeting produces cleaner training signal for DPO. When the model learns from a pair where only cs_causality differs between chosen and rejected, it learns a precise association between causal reasoning quality and preference.

The result is perfectly balanced coverage: every commonsense dialogue contributes exactly 1 candidate per dimension, yielding 9,647 candidates per dim across 38,588 total.

**Gold annotations.** The source corpus includes a gold annotation file with 2,395 matched train entries (24.8%) that have a gold_relation in their domain_metadata. Gold annotations are preserved as reference metadata but no longer affect dimension targeting — every dialogue covers all 4 dimensions regardless of gold relation. The ATOMIC relation-to-dimension mapping in config.py (CS_RELATION_TO_DIM) is retained for use during Stage 1 merge.

### Generation Prompt Structure

Each generation prompt contains the dialogue split into a shared prefix (unchanged context) and an original continuation (the part to rewrite). The continuation length is sampled from a weighted distribution: 1 turn (15%), 3 turns (40%), 5 turns (30%), 7 turns (15%).

The prompt also includes a rubric describing the target dimensions, pulled from the dimension descriptions in config.py. The rubric tells the model what each dimension means and how to evaluate it. For `global_improve` and `global_degrade` variants, the rubric covers all characterizing dimensions. For `dimension_targeted` variants, it covers only the target dimensions.

**Prompt quality controls (v2):**

All generation prompts include direction-aware score context. A CURRENT SCORES block shows the current value of each target dimension with magnitude guidance that accounts for floor/ceiling effects: a low score with improve direction gets "make a substantial change", but a low score with degrade direction gets "make a minimal, subtle change" (since there's little room to degrade further). This prevents the generator from producing gibberish when asked to degrade an already-weak dimension.

Targeted prompts include a NON-TARGET DIMENSIONS block explicitly listing characterizing dimensions that should NOT change, with their descriptions. This anchors the model's understanding of what to preserve. Global degrade prompts include a similar block for non-characterizing dimensions (e.g., cs_coherence, cs_empathy for commonsense) to prevent the model from breaking baseline quality while degrading characterizing dimensions.

All degrade prompts (global and targeted) include subtlety constraints: "the degradation should be subtle and natural-sounding", "avoid abrupt non-sequiturs, obviously rude responses, or incoherent text". Without these, the model tends to produce obviously broken text that would be trivially distinguishable by any reader.

The JSON output format in all templates uses explicit role objects (`{"role": "user", "content": "..."}`) rather than the ambiguous pipe notation (`"user"|"assistant"`).

**Domain-specific prompt additions:**

Commonsense, coherence, and empathy prompts include a GENERATION GUIDANCE block after the main prompt. This block gives concrete writing strategies for the specific target dimension and direction. For example, when improving cs_causality, the guidance says to ensure clear cause-effect relationships, use temporal markers, and make prerequisites explicit. When degrading cs_reaction, it says reactions should be "slightly off — a bit too muted or intense for the situation" rather than obviously inappropriate. Coherence guidance covers topic coherence and logical consistency; empathy guidance covers emotional awareness and perspective taking. This guidance exists because "improve causality" is ambiguous — the model needs concrete instructions about what quality looks like in dialogue.

Multicultural prompts include a CULTURAL CONTEXT block with 15 fields backtracked from the raw data via each dialogue's uid: both countries, both speakers' demographics and cultural perspectives, the value statement and its culturally adapted form, the situation, social norms for each culture, cross-cultural prejudices, and emotional dynamics. The direction qualifier is explicit: "more culturally grounded" for improve, "less culturally grounded" for degrade. Without this context, the model would have no way to meaningfully modify cultural quality — it wouldn't know which countries, values, or social dynamics are at play. This is the most information-dense prompt in the pipeline, often exceeding 2,000 words.

Real example prompts for all four domains are saved in `data/stage_2_example_prompts.txt`.

The ~125k candidates are written to 25 shards of ~5,000 entries each for batch API submission.

### Generation Results

The current results are from the v2 re-run with improved prompts (see Prompt Quality Fixes below). V1 outputs are backed up at `data/stage_2/shards/output_v1/`.

The 25 shards were submitted to the Azure OpenAI batch API (gpt-5.1-batch) with 5 concurrent jobs. All 25 completed successfully with 0 shard failures.

| Metric | Value |
|--------|-------|
| Manifest entries | 124,967 |
| Shards submitted | 25 (5,000 entries each, last shard 4,967) |
| Successful generations | 124,792 (99.86%) |
| Null content (failed) | 169 (0.14%) |
| Skipped (<2 messages) | 21 |

**Parse resilience.** During eval input preparation, `parse_generation_results` encountered malformed GPT responses: null content, list-type content instead of string, and capitalized role names ("User" instead of "user"). The parser coerces list content to string, normalizes roles to lowercase, and catches all exception types. Final parse: 124,772 variants from 124,967 entries (189 parse failures = 0.15%).

**V1 batch processing notes (historical).** The initial v1 run required three rounds due to Azure quota limits and 12 content-filtered entries from 5 source dialogues (all sexual:high). Those 12 were regenerated locally using Claude Opus 4.6 and stored in `opus_recovered_12.jsonl`. The v2 re-run encountered no quota issues and completed in a single round.

### Evaluation

A second LLM judge (gpt-5.1 via Azure batch API) scores the generated variant continuation on all domain dimensions (0.0 to 1.0 each). The judge sees the shared prefix and the variant continuation only. Stage 1 scores serve as ground truth for the original, so only the variant needs fresh scoring. Classification uses only characterizing dimensions, but all scores are stored for richer downstream signal.

The eval prompt includes SCORING ANCHORS (0.0–0.2 "severely deficient" through 0.9–1.0 "excellent") to prevent central tendency bias and explicit instructions to consider how the continuation builds upon the shared prefix before scoring.

For multicultural entries, the eval prompt receives a full cultural context block matching the generation prompt: both countries, demographics, cultural reasoning, social norms, cross-cultural prejudices, and value statements. This ensures the judge evaluates cultural dimensions with the same cultural background available to the generator.

### 6-Label Classification System

The old binary pass/fail with a ±0.20 stability constraint rejected 67.6% of variants — almost all because correlated dimensions naturally co-move when one is changed. This is expected behavior, not a defect.

The new system replaces the binary gate with a 6-label classifier. Every label encodes two pieces of information: the variant origin (global vs targeted) and the signal quality (clean pass, coarse pass, or flipped). The margin threshold (configurable, set to 0.05 in v2) ensures minimal contrastive signal while maximizing yield.

| Label | Applies to | Rule | Pair formation |
|---|---|---|---|
| **global_pass** | Global variants | Avg of all char dims moved in intended direction by > margin | Standard: intended direction |
| **target_pass** | Targeted variants | Target dim moved correctly by > margin, non-targets within ±stability | Standard: intended direction |
| **target_coarse_pass** | Targeted variants | Target dim moved correctly by > margin, non-targets drifted beyond ±stability | Standard: intended direction |
| **global_flip_pass** | Global variants | Avg moved OPPOSITE to intended by > margin | Swap chosen/rejected, intent_followed=false |
| **target_flip_pass** | Targeted variants | Target dim moved OPPOSITE to intended by > margin | Swap chosen/rejected, intent_followed=false |
| **reject** | Both | Target/avg moved ≤ margin in either direction, or missing scores | Discarded — no contrastive signal |

The ±0.20 stability threshold in `target_pass` vs `target_coarse_pass` classifies non-target co-movement quality — it no longer causes rejection. The margin threshold and stability threshold serve different purposes and have different values (0.05 and 0.20 respectively).

### Evaluation Results

The current results are from the v2 re-run with improved prompts. V1 eval outputs are backed up at `data/stage_2/shards_eval/output_v1_biased/`.

The 25 eval shards (124,772 entries) were submitted to the Azure batch API with 5 concurrent jobs. All 25 completed successfully with 0 failures.

**Overall: 121,326 usable / 124,768 classified (97.2%), 3,442 rejected**

| Domain | global_pass | target_pass | target_coarse_pass | global_flip_pass | target_flip_pass | reject | **Usable** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Empathy | 9,542 | 1,012 | 17,818 | 4 | 141 | 0 | **28,517** |
| Coherence | 9,193 | 10,297 | 7,189 | 182 | 1,219 | 0 | **28,080** |
| Multicultural | 8,953 | 6,621 | 8,767 | 241 | 1,866 | 0 | **26,448** |
| Commonsense | 0 | 52 | 37,636 | 0 | 593 | 0 | **38,281** |
| **Total** | **27,688** | **17,982** | **71,410** | **427** | **3,819** | **3,442** | **121,326** |

Note: the 3,442 rejects were filtered during pair construction (margin ≤ 0.05) and do not appear in the final pairs.jsonl. The label counts above reflect the 121,326 emitted pairs. 4,246 pairs (3.5%) have `intent_followed=false` — these are flip_pass pairs where chosen/rejected were swapped.

**V2 vs v1 comparison.** The v2 re-run with improved prompts produced notable shifts: target_pass increased from 13,516 to 17,982 (+33%), indicating better non-target preservation from the explicit NON-TARGET DIMENSIONS constraints. Total flip_pass decreased from 8,358 to 4,246 (−49%), indicating the generator more reliably moves scores in the intended direction. Multicultural target_flip_pass dropped from 6,786 to 1,866 (−73%) — the full cultural context in eval prompts and direction-aware score guidance reduced direction confusion.

**By variant type:**

| Domain | Global improve | Global degrade | Dimension targeted | Total |
|--------|---:|---:|---:|---:|
| Empathy | 8,256 | 1,290 | 18,971 | 28,517 |
| Coherence | 2,854 | 6,521 | 18,705 | 28,080 |
| Multicultural | 916 | 8,278 | 17,254 | 26,448 |
| Commonsense | — | — | 38,281 | 38,281 |
| **Total** | **12,026** | **16,089** | **93,211** | **121,326** |

**By target dimension (dimension_targeted pairs only):**

| Dimension | target_pass | target_coarse_pass | target_flip_pass | Total |
|-----------|---:|---:|---:|---:|
| co_logical_consistency | 5,605 | 3,483 | 385 | 9,473 |
| co_topic_coherence | 4,692 | 3,706 | 834 | 9,232 |
| em_emotional_awareness | 760 | 8,743 | 19 | 9,522 |
| em_perspective_taking | 252 | 9,075 | 122 | 9,449 |
| mu_cultural_specificity | 5,410 | 3,600 | 108 | 9,118 |
| mu_cultural_value | 1,211 | 5,167 | 1,758 | 8,136 |
| cs_causality | 23 | 9,509 | 78 | 9,610 |
| cs_consistency | 10 | 9,448 | 125 | 9,583 |
| cs_desire | 11 | 9,366 | 208 | 9,585 |
| cs_reaction | 8 | 9,313 | 182 | 9,503 |
| **Total** | **17,982** | **71,410** | **3,819** | **93,211** |

Coherence shows the biggest target_pass improvement in v2: co_logical_consistency went from 4,520 to 5,605, co_topic_coherence from 1,378 to 4,692 — non-target drift decreased substantially with the explicit non-target constraints. Multicultural mu_cultural_value target_flip_pass dropped from 5,222 (57%) to 1,758 (22%) — still the highest flip rate but greatly improved by the direction-aware prompts.

**Difficulty distribution:**

| Domain | Easy (≥ 0.30) | Medium (0.15–0.30) | Hard (< 0.15) | Total |
|--------|---:|---:|---:|---:|
| Empathy | 27,022 (94.8%) | 1,046 (3.7%) | 449 (1.6%) | 28,517 |
| Coherence | 19,036 (67.8%) | 4,748 (16.9%) | 4,296 (15.3%) | 28,080 |
| Commonsense | 35,183 (91.9%) | 1,749 (4.6%) | 1,349 (3.5%) | 38,281 |
| Multicultural | 13,162 (49.8%) | 8,270 (31.3%) | 5,016 (19.0%) | 26,448 |
| **Total** | **94,403 (77.8%)** | **15,813 (13.0%)** | **11,110 (9.2%)** | **121,326** |

The dataset skews easy (78%) because the LLM reliably produces large score shifts. Multicultural is the hardest domain — only 50% easy, with 19% hard — because the evaluator perceives smaller cultural quality differences than the generator intended. Coherence shifted from 88% easy (v1) to 68% easy (v2) — the improved prompts produce subtler, more realistic degradations with smaller margins instead of heavy-handed rewrites. Empathy remains the easiest at 95%.

**Contrastive direction distribution:**

| Domain | Positive (variant is chosen) | Negative (original is chosen) | Total |
|--------|---:|---:|---:|
| Empathy | 24,869 (87.2%) | 3,648 (12.8%) | 28,517 |
| Coherence | 9,915 (35.3%) | 18,165 (64.7%) | 28,080 |
| Commonsense | 37,628 (98.3%) | 653 (1.7%) | 38,281 |
| Multicultural | 4,362 (16.5%) | 22,086 (83.5%) | 26,448 |
| **Total** | **76,774 (63.3%)** | **44,552 (36.7%)** | **121,326** |

The direction split reflects source data quality. Empathy and commonsense originals score low → variants improve → positive direction (chosen = variant). Coherence and multicultural originals score high → variants degrade → negative direction (chosen = original). Multicultural's negative share increased from 60% (v1) to 84% (v2) because the reduced flip_pass count means fewer pairs switch from their intended degradation direction.

**Score distributions (variant vs. original):**

| Domain | Dimension | Original avg | Variant avg | Shift |
|--------|-----------|:---:|:---:|:---:|
| Coherence | co_topic_coherence | 0.761 | 0.801 | +0.040 |
| Coherence | co_logical_consistency | 0.717 | 0.646 | -0.070 |
| Empathy | em_emotional_awareness | 0.304 | 0.857 | +0.553 |
| Empathy | em_perspective_taking | 0.262 | 0.835 | +0.573 |
| Commonsense | cs_causality | 0.137 | 0.892 | +0.756 |
| Commonsense | cs_consistency | 0.317 | 0.905 | +0.587 |
| Commonsense | cs_reaction | 0.333 | 0.877 | +0.544 |
| Commonsense | cs_desire | 0.158 | 0.891 | +0.733 |
| Multicultural | mu_cultural_value | 0.708 | 0.708 | -0.000 |
| Multicultural | mu_cultural_specificity | 0.858 | 0.518 | -0.340 |

Empathy and commonsense show massive positive shifts (variants dramatically improve on weak originals). Coherence shows a mixed pattern in v2: co_topic_coherence slightly increased while co_logical_consistency decreased — the subtler degradation prompts produce more nuanced changes rather than uniformly crushing all dimensions. Multicultural mu_cultural_specificity shows a clear -0.340 shift while mu_cultural_value stays flat — the evaluator consistently detects specificity changes but is insensitive to value changes.

### Pair Construction

Each non-rejected variant becomes a preference pair. The 6-label system determines how the pair is formed:

For standard labels (global_pass, target_pass, target_coarse_pass): positive-direction variants use the generated version as chosen and original as rejected. Negative-direction variants reverse this.

For flip labels (global_flip_pass, target_flip_pass): the variant moved in the opposite direction from intended. The pair swaps chosen/rejected accordingly; `contrast.direction` preserves the original intent and `contrast.intent_followed` is set to false. A positive-intended variant that degraded quality becomes a pair where the original is chosen.

Each pair carries both a difficulty label and a pair label. Difficulty is based on the margin between chosen and rejected scores on target dimensions: easy (≥ 0.30), medium (0.15–0.30), hard (< 0.15). The pair label identifies provenance — how the pair was formed and what kind of signal it carries.

Each pair receives a sequential ID in the format `S2D-{n:06d}` (S2D-000001 through S2D-121326), matching Stage 1's `S1D-*` convention. The ID is assigned by insertion order during pair construction — it carries no semantic meaning beyond uniqueness.

Score ordering is validated before pair emission: the chosen version must actually score higher than the rejected version on the target dimensions. This catches edge cases where the margin calculation and the ordering disagree due to rounding or multi-dimensional effects. In the final run, 0 pairs were skipped for bad ordering — the 6-label flip logic produces correctly ordered pairs by construction.


## Bugs Fixed and Changes Made

**Falsy-value bug.** The Python pattern `x or y` drops 0.0 as falsy, which caused 10,560 commonsense batch failures and missing dimension scores in parse_results.py and merge_stage_1.py. Fixed with explicit `x is None` checks.

**Multicultural escaped newlines.** The raw CSV data stored literal `\n` strings instead of real newline characters. All 12,816 multicultural dialogues collapsed into single-message conversations when parsed. Fixed in parse_dialogue_to_messages to handle both escaped and real newlines.

**Score template uniformity.** Entries previously carried only their own domain's scores. Now every entry has all 23 dimension keys — domain scores filled, others null. This uniform schema eliminates special-casing in downstream processing.

**Multicultural cultural context.** Generation and eval prompts previously received no cultural metadata. The model saw only the dialogue and a generic rubric, with no way to know which countries, demographics, or cultural values were involved. Fixed by backtracking from each dialogue's uid to the raw CSV at merge time, carrying 15 metadata fields through to prompt construction.

**Commonsense gold relation metadata.** The commonsense annotation file provides gold ATOMIC relation labels for ~25% of dialogues. These are now loaded at merge time and stored in domain_metadata as gold_relation, enabling precise per-dialogue dimension targeting in Stage 2.

**Commonsense 4-dimension targeting.** cs_reaction and cs_desire were promoted from non-characterizing to characterizing, bringing commonsense to 4 characterizing dimensions (matching the 4 ATOMIC dimension categories). All commonsense variants now use single-dimension targeting with gold-based or rotation-based dimension selection. The eval pipeline was extended with a stability check that ensures non-target dimensions stay within ±0.20. The pairs pipeline was updated to compute margin and validate ordering on actual target dimensions rather than all characterizing dimensions.

**Multi-variant generation.** Each dialogue now produces multiple candidates instead of one. Non-commonsense domains produce 3 (1 global + 2 dimension-targeted), commonsense produces 4 (1 per characterizing dim). This 3.25x increase from 38,447 to ~125k candidates gives the batch API more diverse generation requests and the DPO training set richer contrastive signal. Output is fully deterministic. Commonsense gold-annotation and rotation-based dimension selection were replaced with full-coverage targeting (every dialogue covers all 4 dims). Custom IDs in generate.py were refactored from `s2g-{id}-{type[:3]}` to `s2g-{id}-{gimp|gdeg|dt-{dim}}` to prevent collisions when multiple candidates share a dialogue_id.

**Naming consistency.** The Stage 2 evaluation step was renamed from "verify/verification" to "eval/evaluation" across the codebase and data files (`verify.py` → `eval.py`, `shards_verify/` → `shards_eval/`, `VerificationResult` → `EvalResult`, `gen_manifest`/`ver_manifest` → `manifest_gen`/`manifest_eval`). The rename reflects that the LLM judge scores both versions — that is evaluation, not binary verification. Stage 1 batch output directory was renamed from `data/output/` to `data/stage_1/` for consistency with the `data/stage_2/` convention.

**Pair ID simplification.** Stage 2 pair IDs were simplified from compound strings (`pair-var-s2g-S1D-016667-dt-em_emotional_awareness`) to clean sequential IDs (`S2D-000001` through `S2D-121326`), matching Stage 1's `S1D-*` convention. The old format leaked internal pipeline identifiers (batch custom_id, variant_id prefix) into the output. The new format decouples pair identity from pipeline internals.

**Prompt quality fixes (v2).** A systematic review of all generation and eval prompts identified 17 issues across 3 rounds, backed by CLAIR, ConvoSense, CRoW, RATE, and SynPO research. All fixes applied to `scripts/stage_2/prompts.py` with 28 new tests added. The full stage 2 pipeline was re-run after fixes. Key changes:

- *Direction-aware score context*: generation prompts now include CURRENT SCORES blocks with magnitude guidance that accounts for floor/ceiling effects. A degrade direction + low score gets "make a minimal, subtle change" instead of "make a substantial change" which would produce gibberish.
- *Non-target dimension constraints*: targeted prompts explicitly list non-target characterizing dimensions with "do NOT change" header. Global degrade prompts list non-characterizing dimensions to preserve (e.g., cs_coherence, cs_empathy).
- *Subtlety constraints for degradation*: all degrade prompts include "the degradation should be subtle and natural-sounding" and "avoid abrupt non-sequiturs" to prevent obviously broken text.
- *Score anchoring in eval*: eval prompts include SCORING ANCHORS (0.0–0.2 "severely deficient" through 0.9–1.0 "excellent") to reduce central tendency bias.
- *Prefix-continuation reasoning*: eval instructions explicitly require considering how the continuation builds upon the shared prefix before scoring.
- *Full multicultural eval context*: eval prompts now receive the same full cultural profile as generation prompts (both countries, demographics, cultural reasoning, social norms, cross-cultural prejudices) instead of the lighter summary used in v1.
- *Multicultural direction qualifier*: replaced ambiguous "more (or less)" with explicit "more culturally grounded" or "less culturally grounded" based on direction.
- *Commonsense degrade guidance*: rewrote degrade strategies from obvious ("violate temporal ordering") to subtle ("use causal language but connect wrong events").
- *Coherence and empathy generation guidance*: added GENERATION GUIDANCE for all domains, not just commonsense.
- *Cultural context positioning*: cultural context blocks now appear BEFORE instructions in both generation and eval prompts, ensuring the model has cultural background before formulating responses.
- *JSON output format*: replaced pipe notation (`"user"|"assistant"`) with explicit role objects in all templates.


## Known Risks

Commonsense dialogues are short — 76.6% have exactly 5 messages and 23.4% have exactly 4. When the sampled continuation length is 5 or 7 turns, the continuation exceeds the original dialogue length. The fallback logic splits the dialogue into a 1-message prefix and uses the rest as the continuation, but the generated variant may not meaningfully correspond to the original at these lengths.

Evaluation uses a different prompt structure than Stage 1 scoring. Stage 1 uses domain-specific, carefully structured prompts with relation-level rubrics (for commonsense) and full cultural context (for multicultural). Stage 2 evaluation uses score anchoring and prefix-continuation reasoning but a single-continuation format. This is partially by design — independent validation should not replicate the exact scoring methodology — but could affect pass rates if the evaluator disagrees systematically with Stage 1 scores.

The ±0.20 stability threshold serves as a classification boundary (target_pass vs target_coarse_pass), not a rejection gate. Target_coarse_pass pairs (71,410 — 59% of the dataset) have valid contrastive signal on the target dimension but noisier non-target behavior. Downstream DPO training may want to weight these differently or filter by label — the label field on every pair enables this without re-running the pipeline. The 4,246 flip_pass pairs (427 global + 3,819 targeted) carry reversed contrastive direction; multicultural mu_cultural_value accounts for the highest flip rate at 22%. These should be monitored for training stability.

**Margin threshold sensitivity.** The margin threshold (set to 0.05 in v2) affects domains unevenly. The percentage of accepted pairs that would be lost at each threshold (margin ≤ threshold → rejected):

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

Coherence is notably more sensitive in v2 than v1 (32.7% lost at 0.20 vs 7.5%) because the subtler degradation prompts produce smaller margins — the model no longer crushes coherence dimensions uniformly. Multicultural remains the most fragile domain at higher thresholds. Empathy and commonsense remain robust because variants produce large score shifts. The 0.05 threshold was chosen to maximize yield while filtering only trivially small margins.


## File Structure

```
scripts/
  config.py           Domain configs, 23 dimensions, characterizing flags, ATOMIC relation mapping
  models.py           Pydantic models (Stage1Entry, Stage2Candidate, Stage2Variant, Stage2Pair)
  merge_stage_1.py    Merge domain outputs, backtrack multicultural metadata and commonsense gold
  split.py            Train/val/test stratified split assignment
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
  stage_1.jsonl                         51,264 scored entries (188 MB) — all 23 dims, S1D-* IDs, split: train/val/test
  stage_2.jsonl                         121,326 preference pairs — S2D-* IDs, 6-label classification (shareable copy of stage_2/pairs.jsonl)
  stage_1_template.json                 Human-readable Stage 1 format reference (categorical fields show all possible values)
  stage_2_template.json                 Human-readable Stage 2 pair format reference (categorical fields show all possible values)
  stage_2_example_prompts.txt           One real generation prompt per domain
  all_prompts_review.txt                All 32 rendered prompt variants for visual review
  stage_2/candidates.jsonl              124,988 selected candidates (3-4 per dialogue) with tier, direction, and target dims
  stage_2/shards/                       25 generation request shards (5,000 entries each)
  stage_2/shards/output/                25 batch output files (v2 re-run)
  stage_2/shards/output_v1/             v1 generation outputs backup (26 files, 243 MB)
  stage_2/manifest_gen.jsonl            124,967 manifest entries mapping custom_id to dialogue metadata
  stage_2/shards_eval/                  25 eval request shards (124,772 entries)
  stage_2/shards_eval/output/           25 batch output files with variant scores (v2 re-run)
  stage_2/shards_eval/output_v1_biased/ v1 eval outputs backup (25 files, 176 MB) — biased by pre-fix prompts
  stage_2/manifest_eval.jsonl           124,772 manifest entries mapping eval custom_id to variant metadata
  stage_2/pairs.jsonl                   121,326 preference pairs — S2D-* IDs, 6-label classification
```

### Data Split Summary

| Split | Entries | Purpose | File |
|-------|--------:|---------|------|
| Train | 38,447 | Stage 2 input → 121,326 DPO pairs | `data/stage_1.jsonl` (filter `split == "train"`) |
| Val | 5,127 | Held out for downstream evaluation | `data/stage_1.jsonl` (filter `split == "val"`) |
| Test | 7,690 | Held out for downstream evaluation | `data/stage_1.jsonl` (filter `split == "test"`) |

Val and test entries are in `data/stage_1.jsonl` alongside train entries, distinguished by the `split` field on each JSON line. Stage 2 (`select.py`) filters to `split == "train"` before candidate selection — val/test entries are never touched, generated from, or used during pair construction. They exist solely for evaluating the trained model downstream (e.g., scoring model outputs on the same dimensions to measure alignment quality).

335 tests passing across 13 test files.
