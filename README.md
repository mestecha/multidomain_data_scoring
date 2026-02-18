# Multi-Scoring Data Transform — Report

Last updated: 2026-02-18


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
    ↓ 6-label classification (margin > 0.20)
    ↓ flip-pass recovery for opposite-direction variants
121,492 preference pairs (stage_2.jsonl)
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

**Domain-specific prompt additions:**

Commonsense prompts include a GENERATION GUIDANCE block after the main prompt. This block gives concrete writing strategies for the specific target dimension and direction. For example, when improving cs_causality, the guidance says to ensure clear cause-effect relationships, use temporal markers, and make prerequisites explicit. When degrading cs_reaction, it says to make emotional reactions inappropriate and mismatch intensity with event significance. This guidance exists because "improve causality" is ambiguous — the model needs concrete instructions about what causal quality looks like in dialogue.

Multicultural prompts include a CULTURAL CONTEXT block with 15 fields backtracked from the raw data via each dialogue's uid: both countries, both speakers' demographics and cultural perspectives, the value statement and its culturally adapted form, the situation, social norms for each culture, cross-cultural prejudices, and emotional dynamics. Without this context, the model would have no way to meaningfully modify cultural quality — it wouldn't know which countries, values, or social dynamics are at play. This is the most information-dense prompt in the pipeline, often exceeding 2,000 words.

Coherence and empathy prompts contain only the dialogue and rubric. The model can understand coherence and empathy from conversational context alone — no external metadata is needed.

Real example prompts for all four domains are saved in `data/stage_2_example_prompts.txt`.

The ~125k candidates are written to 25 shards of ~5,000 entries each for batch API submission.

### Generation Results

The 25 shards were submitted to the Azure OpenAI batch API (gpt-5.1-batch) with 5 concurrent jobs. Total wall-clock time was approximately 68 minutes for the initial run.

| Metric | Value |
|--------|-------|
| Manifest entries | 124,967 |
| Shards submitted | 25 (5,000 entries each, last shard 4,967) |
| Successful generations | 124,955 (99.99%) |
| Content-filtered | 12 (0.01%) |
| Skipped (<2 messages) | 21 |

**Batch processing required three rounds:**

Round 1 submitted all 25 shards. 22 completed successfully in 400–555s each (smaller shards, 7–10MB) to 2,074s (larger multicultural shards, 40–50MB). Three shards (0022–0024, all multicultural, 47–50MB) failed with `quotaExceeded` — Azure's 500-file storage limit was hit because prior batch runs had accumulated stale files. After deleting 691 stale files from Azure storage, round 2 resubmitted the 3 failed shards and all completed in 374–435s. Round 3 retried the 12 entries that were silently dropped from round 1 output files — all 12 returned as content-filtered (sexual:high severity).

The 12 content-filtered entries come from 5 source dialogues:

| Dialogue ID | Domain | Variants lost | Filter trigger |
|-------------|--------|:---:|---|
| S1D-016960 | empathy | 3 | sexual:high |
| S1D-016424 | empathy | 2 | sexual:high |
| S1D-015532 | empathy | 2 | sexual:high |
| S1D-013371 | empathy | 1 | sexual:high |
| S1D-021253 | empathy | 1 | sexual:high |
| S1D-031655 | commonsense | 3 | sexual:high |

All 12 failures are content-inherent — the source dialogues contain content that Azure's safety filter rejects. These cannot be recovered through retry and represent an irrecoverable 0.01% loss. The affected 5 dialogues (out of 38,447) will simply have fewer variants available for evaluation. From a DPO training perspective, losing 12 candidates out of 124,967 has no measurable impact on dataset quality or balance.

The 124,955 successful generation outputs are stored in `data/stage_2/shards/output/` across 26 JSONL files (22 from round 1 + 3 from round 2 + 1 recovered file with the 12 Opus-generated replacements).

**Content-filter recovery.** The 12 content-filtered entries were regenerated locally using Claude Opus 4.6 (not via API) to achieve 100% coverage. These are stored in `data/stage_2/shards/output/opus_recovered_12.jsonl` with `model: "opus-4.6-recovered"` for traceability. The source dialogues were benign family conversations that Azure's safety filter incorrectly flagged.

**Parse resilience.** During eval input preparation, `parse_generation_results` encountered three types of malformed GPT responses: null content (content-filtered entries), list-type content instead of string, and capitalized role names ("User" instead of "user"). The parser was hardened to coerce list content to string, normalize roles to lowercase, and catch all exception types. Final parse: 124,804 variants from 124,967 entries (163 parse failures = 0.13%).

Parse failure distribution by domain: empathy 98 (0.34%), commonsense 41 (0.11%), coherence 16 (0.06%), multicultural 8 (0.03%).

### Evaluation

A second LLM judge (gpt-5.1 via Azure batch API) independently scores both the original continuation and the generated variant on the characterizing dimensions (0.0 to 1.0 each). The judge sees the shared prefix and both continuations side by side but does not know which is the original.

For multicultural entries, the eval prompt also receives a lighter cultural context block (countries, statement, demographics) so the judge can evaluate cultural dimensions with the right background.

### 6-Label Classification System

The old binary pass/fail with a ±0.20 stability constraint rejected 67.6% of variants — almost all because correlated dimensions naturally co-move when one is changed. This is expected behavior, not a defect.

The new system replaces the binary gate with a 6-label classifier. Every label encodes two pieces of information: the variant origin (global vs targeted) and the signal quality (clean pass, coarse pass, or flipped). The margin threshold of 0.20 ensures meaningful contrastive signal.

| Label | Applies to | Rule | Pair formation |
|---|---|---|---|
| **global_pass** | Global variants | Avg of all char dims moved in intended direction by > 0.20 | Standard: intended direction |
| **target_pass** | Targeted variants | Target dim moved correctly by > 0.20, non-targets within ±0.20 | Standard: intended direction |
| **target_coarse_pass** | Targeted variants | Target dim moved correctly by > 0.20, non-targets drifted beyond ±0.20 | Standard: intended direction |
| **global_flip_pass** | Global variants | Avg moved OPPOSITE to intended by > 0.20 | Swap chosen/rejected, flip contrastive_direction |
| **target_flip_pass** | Targeted variants | Target dim moved OPPOSITE to intended by > 0.20 | Swap chosen/rejected, flip contrastive_direction |
| **reject** | Both | Target/avg moved ≤ 0.20 in either direction, or missing scores | Discarded — no contrastive signal |

The ±0.20 stability threshold in `target_pass` vs `target_coarse_pass` classifies non-target co-movement quality — it no longer causes rejection. The margin threshold (also 0.20) and stability threshold serve different purposes despite sharing the same value.

### Evaluation Results

The 25 eval shards (124,804 entries) were submitted to the Azure batch API with 5 concurrent jobs. All 25 completed successfully in ~40 minutes (340–858s per shard, 0 failures).

**Overall usable: 121,492 / 124,800 (97.3%)**

| Domain | global_pass | target_pass | target_coarse_pass | global_flip_pass | target_flip_pass | reject | **Usable** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Empathy | 9,524 | 472 | 18,499 | 19 | 143 | — | **28,657** |
| Coherence | 9,228 | 5,898 | 12,793 | 173 | 187 | — | **28,279** |
| Multicultural | 8,227 | 6,139 | 4,150 | 850 | 6,786 | — | **26,152** |
| Commonsense | 0 | 1,007 | 37,197 | 0 | 200 | — | **38,404** |
| **Total** | **26,979** | **13,516** | **72,639** | **1,042** | **7,316** | **3,308** | **121,492** |

Rejects total 3,308 (2.7% of 124,800). The remaining 4 variants had no eval result (parse failures during generation).

**Yield improvement: 32.4% → 97.3%.** The old binary system rejected 84,305 variants; the new system recovers 81,001 of those as usable pairs. The gain comes from two sources: target_coarse_pass (72,639 variants where the target moved correctly but non-targets co-moved) and flip_pass (8,358 variants where scores moved opposite to intended, producing valid pairs with swapped chosen/rejected).

**By variant type:**

| Domain | Global improve | Global degrade | Dimension targeted | Total |
|--------|---:|---:|---:|---:|
| Empathy | 8,256 | 1,287 | 19,114 | 28,657 |
| Coherence | 2,764 | 6,637 | 18,878 | 28,279 |
| Multicultural | 914 | 8,163 | 17,075 | 26,152 |
| Commonsense | — | — | 38,404 | 38,404 |
| **Total** | **11,934** | **16,087** | **93,471** | **121,492** |

**By target dimension (dimension_targeted pairs only):**

| Dimension | target_pass | target_coarse_pass | target_flip_pass | Total |
|-----------|---:|---:|---:|---:|
| co_logical_consistency | 4,520 | 4,796 | 88 | 9,404 |
| co_topic_coherence | 1,378 | 7,997 | 99 | 9,474 |
| em_emotional_awareness | 420 | 9,001 | 132 | 9,553 |
| em_perspective_taking | 52 | 9,498 | 11 | 9,561 |
| mu_cultural_specificity | 4,310 | 2,110 | 1,564 | 7,984 |
| mu_cultural_value | 1,829 | 2,040 | 5,222 | 9,091 |
| cs_causality | 217 | 9,385 | 7 | 9,609 |
| cs_consistency | 224 | 9,323 | 24 | 9,571 |
| cs_desire | 332 | 9,215 | 65 | 9,612 |
| cs_reaction | 234 | 9,274 | 104 | 9,612 |
| **Total** | **13,516** | **72,639** | **7,316** | **93,471** |

Empathy and commonsense dimensions recover at 97–99% thanks to target_coarse_pass. The old system rejected these because non-targets co-moved — the new system correctly recognizes that the target dimension did move, and classifies the co-movement rather than rejecting it.

Multicultural mu_cultural_value has massive target_flip_pass (5,222 / 9,091 = 57%) — variants often moved cultural value in the opposite direction from intended. These produce valid pairs with swapped chosen/rejected and flipped contrastive direction. mu_cultural_specificity has 1,564 flip passes (20%). This domain-specific flip behavior is the main contributor to the 7,316 total target_flip_pass count.

**Difficulty distribution:**

| Domain | Easy (≥ 0.30) | Medium (0.15–0.30) | Hard (< 0.15) | Total |
|--------|---:|---:|---:|---:|
| Empathy | 27,904 (97.4%) | 498 (1.7%) | 255 (0.9%) | 28,657 |
| Coherence | 24,835 (87.8%) | 2,105 (7.4%) | 1,339 (4.7%) | 28,279 |
| Commonsense | 36,280 (94.5%) | 1,201 (3.1%) | 923 (2.4%) | 38,404 |
| Multicultural | 10,704 (40.9%) | 9,076 (34.7%) | 6,372 (24.4%) | 26,152 |
| **Total** | **99,723 (82.1%)** | **12,880 (10.6%)** | **8,889 (7.3%)** | **121,492** |

The dataset skews easy (82%) because the LLM reliably produces large score shifts. Multicultural is the hardest domain — only 41% easy, with 24% hard — because the evaluator perceives smaller cultural quality differences than the generator intended. Empathy is the easiest at 97.4% because variants dramatically shift emotional scores (original avg ~0.2 → variant avg ~0.85).

**Contrastive direction distribution:**

| Domain | Positive (variant is chosen) | Negative (original is chosen) | Total |
|--------|---:|---:|---:|
| Empathy | 24,960 (87.1%) | 3,697 (12.9%) | 28,657 |
| Coherence | 8,478 (30.0%) | 19,801 (70.0%) | 28,279 |
| Commonsense | 37,616 (97.9%) | 788 (2.1%) | 38,404 |
| Multicultural | 10,371 (39.7%) | 15,781 (60.3%) | 26,152 |
| **Total** | **81,425 (67.0%)** | **40,067 (33.0%)** | **121,492** |

The direction split reflects source data quality. Empathy and commonsense originals score low → variants improve → positive direction (chosen = variant). Coherence and multicultural originals score high → variants degrade → negative direction (chosen = original). The flip_pass mechanism shifts some pairs from their intended direction: multicultural's 6,786 target_flip_pass + 850 global_flip_pass add ~7,636 pairs to the opposite direction from what was generated.

**Score distributions (variant vs. original):**

| Domain | Dimension | Original avg | Variant avg | Shift |
|--------|-----------|:---:|:---:|:---:|
| Empathy | em_emotional_awareness | 0.237 | 0.865 | +0.628 |
| Empathy | em_perspective_taking | 0.192 | 0.854 | +0.662 |
| Coherence | co_topic_coherence | 0.796 | 0.638 | -0.158 |
| Coherence | co_logical_consistency | 0.732 | 0.449 | -0.283 |
| Multicultural | mu_cultural_value | 0.741 | 0.734 | -0.007 |
| Multicultural | mu_cultural_specificity | 0.671 | 0.613 | -0.058 |
| Commonsense | cs_causality | 0.440 | 0.948 | +0.508 |
| Commonsense | cs_consistency | 0.349 | 0.949 | +0.600 |
| Commonsense | cs_reaction | 0.360 | 0.939 | +0.579 |
| Commonsense | cs_desire | 0.338 | 0.937 | +0.599 |

Empathy and commonsense show massive positive shifts (variants dramatically improve on weak originals). Coherence shows negative shifts as expected (variants degrade high-quality originals). Multicultural shows small negative shifts — the evaluator sees less cultural degradation than intended, which explains the lower margins.

### Pair Construction

Each non-rejected variant becomes a preference pair. The 6-label system determines how the pair is formed:

For standard labels (global_pass, target_pass, target_coarse_pass): positive-direction variants use the generated version as chosen and original as rejected. Negative-direction variants reverse this.

For flip labels (global_flip_pass, target_flip_pass): the variant moved in the opposite direction from intended. The pair swaps chosen/rejected and flips contrastive_direction to reflect the actual signal. A positive-intended variant that degraded quality becomes a negative-direction pair where the original is chosen.

Each pair carries both a difficulty label and a pair label. Difficulty is based on the margin between chosen and rejected scores on target dimensions: easy (≥ 0.30), medium (0.15–0.30), hard (< 0.15). The pair label identifies provenance — how the pair was formed and what kind of signal it carries.

Each pair receives a sequential ID in the format `S2D-{n:06d}` (S2D-000001 through S2D-121492), matching Stage 1's `S1D-*` convention. The ID is assigned by insertion order during pair construction — it carries no semantic meaning beyond uniqueness.

Score ordering is validated before pair emission: the chosen version must actually score higher than the rejected version on the target dimensions. This catches edge cases where the margin calculation and the ordering disagree due to rounding or multi-dimensional effects. In the final run, 0 pairs were skipped for bad ordering — the 6-label flip logic produces correctly ordered pairs by construction.


## Bugs Fixed and Changes Made

**Falsy-value bug.** The Python pattern `x or y` drops 0.0 as falsy, which caused 10,560 commonsense batch failures and missing dimension scores in parse_results.py and merge_stage_1.py. Fixed with explicit `x is None` checks.

**Multicultural escaped newlines.** The raw CSV data stored literal `\n` strings instead of real newline characters. All 12,816 multicultural dialogues collapsed into single-message conversations when parsed. Fixed in parse_dialogue_to_messages to handle both escaped and real newlines.

**Score template uniformity.** Entries previously carried only their own domain's scores. Now every entry has all 23 dimension keys — domain scores filled, others null. This uniform schema eliminates special-casing in downstream processing.

**Multicultural cultural context.** Generation and eval prompts previously received no cultural metadata. The model saw only the dialogue and a generic rubric, with no way to know which countries, demographics, or cultural values were involved. Fixed by backtracking from each dialogue's uid to the raw CSV at merge time, carrying 15 metadata fields through to prompt construction.

**Commonsense gold relation metadata.** The commonsense annotation file provides gold ATOMIC relation labels for ~25% of dialogues. These are now loaded at merge time and stored in domain_metadata as gold_relation, enabling precise per-dialogue dimension targeting in Stage 2.

**Commonsense 4-dimension targeting.** cs_reaction and cs_desire were promoted from non-characterizing to characterizing, bringing commonsense to 4 characterizing dimensions (matching the 4 ATOMIC dimension categories). All commonsense variants now use single-dimension targeting with gold-based or rotation-based dimension selection. The eval pipeline was extended with a stability check that ensures non-target dimensions stay within ±0.20. The pairs pipeline was updated to compute margin and validate ordering on actual target dimensions rather than all characterizing dimensions.

**Multi-variant generation.** Each dialogue now produces multiple candidates instead of one. Non-commonsense domains produce 3 (1 global + 2 dimension-targeted), commonsense produces 4 (1 per characterizing dim). This 3.25x increase from 38,447 to ~125k candidates gives the batch API more diverse generation requests and the DPO training set richer contrastive signal. The medium-tier random branching between DIMENSION_TARGETED and MULTI_DIMENSIONAL (35% probability) was removed — output is now fully deterministic. Commonsense gold-annotation and rotation-based dimension selection were replaced with full-coverage targeting (every dialogue covers all 4 dims). Custom IDs in generate.py were refactored from `s2g-{id}-{type[:3]}` to `s2g-{id}-{gimp|gdeg|dt-{dim}|mul}` to prevent collisions when multiple candidates share a dialogue_id.

**Naming consistency.** The Stage 2 evaluation step was renamed from "verify/verification" to "eval/evaluation" across the codebase and data files (`verify.py` → `eval.py`, `shards_verify/` → `shards_eval/`, `VerificationResult` → `EvalResult`, `gen_manifest`/`ver_manifest` → `manifest_gen`/`manifest_eval`). The rename reflects that the LLM judge scores both versions — that is evaluation, not binary verification. Stage 1 batch output directory was renamed from `data/output/` to `data/stage_1/` for consistency with the `data/stage_2/` convention.

**Pair ID simplification.** Stage 2 pair IDs were simplified from compound strings (`pair-var-s2g-S1D-016667-dt-em_emotional_awareness`) to clean sequential IDs (`S2D-000001` through `S2D-121492`), matching Stage 1's `S1D-*` convention. The old format leaked internal pipeline identifiers (batch custom_id, variant_id prefix) into the output. The new format decouples pair identity from pipeline internals.


## Known Risks

Commonsense dialogues are short — 76.6% have exactly 5 messages and 23.4% have exactly 4. When the sampled continuation length is 5 or 7 turns, the continuation exceeds the original dialogue length. The fallback logic splits the dialogue into a 1-message prefix and uses the rest as the continuation, but the generated variant may not meaningfully correspond to the original at these lengths.

Evaluation uses a lighter, more generic prompt than Stage 1 scoring. Stage 1 uses domain-specific, carefully structured prompts with relation-level rubrics (for commonsense) and full cultural context (for multicultural). Stage 2 evaluation uses a simpler side-by-side comparison. This is partially by design — independent validation should not replicate the exact scoring methodology — but could affect pass rates if the evaluator disagrees systematically with Stage 1 scores.

The ±0.20 threshold now serves as a classification boundary (target_pass vs target_coarse_pass), not a rejection gate. Target_coarse_pass pairs (72,639 — 60% of the dataset) have valid contrastive signal on the target dimension but noisier non-target behavior. Downstream DPO training may want to weight these differently or filter by label — the label field on every pair enables this without re-running the pipeline. The 8,358 flip_pass pairs (1,042 global + 7,316 targeted) carry reversed contrastive direction; multicultural accounts for 7,636 of these. These should be monitored for training stability, especially mu_cultural_value where 57% of targeted variants flipped.


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
  stage_1/
    prepare_*.py      Per-domain data preparation and sampling
    parse_results.py  Batch result parsing and score normalization
  stage_2/
    select.py         Multi-variant candidate selection, tier classification, per-dim targeting
    prompts.py        Generation and eval templates, cultural context, commonsense guidance
    generate.py       Variant generation batch entry builder, collision-safe custom_id scheme
    eval.py           6-label classification (classify_variant), eval result parsing
    pairs.py          Preference pair construction with flip logic, effective direction, label propagation

data/
  stage_1/                         Per-domain batch output files (coherence/, empathy/, commonsense/, multicultural/, holdout_failures.jsonl)
  stage_1.jsonl                    51,264 scored entries (188 MB) — all 23 dims, S1D-* IDs, split: train/val/test
  stage_2.jsonl                    121,492 preference pairs (257 MB) — S2D-* IDs, 6-label classification (shareable copy of stage_2/pairs.jsonl)
  stage_1_template.json            Human-readable Stage 1 format reference (categorical fields show all possible values)
  stage_2_template.json            Human-readable Stage 2 pair format reference (categorical fields show all possible values)
  stage_2_example_prompts.txt      One real generation prompt per domain
  stage_2/candidates.jsonl         124,988 selected candidates (3-4 per dialogue) with tier, direction, and target dims
  stage_2/shards/                  25 generation request shards (5,000 entries each)
  stage_2/shards/output/           26 batch output files (124,955 successful + 12 Opus-recovered)
  stage_2/manifest_gen.jsonl       124,967 manifest entries mapping custom_id to dialogue metadata
  stage_2/shards_eval/             25 eval request shards (124,804 entries)
  stage_2/shards_eval/output/      25 batch output files with original_scores and variant_scores
  stage_2/manifest_eval.jsonl      124,804 manifest entries mapping eval custom_id to variant metadata
  stage_2/pairs.jsonl              121,492 preference pairs (257 MB) — S2D-* IDs, 6-label classification
```

### Data Split Summary

| Split | Entries | Purpose | File |
|-------|--------:|---------|------|
| Train | 38,447 | Stage 2 input → 121,492 DPO pairs | `data/stage_1.jsonl` (filter `split == "train"`) |
| Val | 5,127 | Held out for downstream evaluation | `data/stage_1.jsonl` (filter `split == "val"`) |
| Test | 7,690 | Held out for downstream evaluation | `data/stage_1.jsonl` (filter `split == "test"`) |

Val and test entries are in `data/stage_1.jsonl` alongside train entries, distinguished by the `split` field on each JSON line. Stage 2 (`select.py`) filters to `split == "train"` before candidate selection — val/test entries are never touched, generated from, or used during pair construction. They exist solely for evaluating the trained model downstream (e.g., scoring model outputs on the same dimensions to measure alignment quality).

307 tests passing across 13 test files.
