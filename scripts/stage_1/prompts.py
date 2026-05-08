"""evaluation prompt templates for all stage 1 domains."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.config import COHERENCE, EMPATHY

# ── Multicultural qid lookups (eager-loaded at import) ────────────────────

MULTICULTURAL_DATA_DIR = Path("data/input/multicultural")


def _read_statement_to_qid() -> dict[str, str]:
    out: dict[str, str] = {}
    path = MULTICULTURAL_DATA_DIR / "statements.csv"
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["statement"]] = row["qid"]
    return out


def _read_qid_info() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    path = MULTICULTURAL_DATA_DIR / "qid_meaning.csv"
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["qid"]] = row
    return out


STATEMENT_TO_QID = _read_statement_to_qid()
QID_INFO = _read_qid_info()


def _qid_addendum(statement: str) -> str:
    qid = STATEMENT_TO_QID.get(statement)
    if not qid:
        return ""
    info = QID_INFO.get(qid)
    if not info:
        return ""
    parts = [
        f"\n   FOR THIS SPECIFIC VALUE ({qid}):",
        f"   - Means: {info.get('means', '').strip()}",
        f"   - Not confused with: {info.get('not_this', '').strip()}",
        f"   - Shows up as: {info.get('shows_up_as', '').strip()}",
        f"   - Does NOT show as: {info.get('dont_say', '').strip()}",
    ]
    return "\n".join(parts) + "\n"

# =============================================================================
# COHERENCE DOMAIN (co_)
# =============================================================================

COHERENCE_SCHEMA = """{
"topic_coherence": {"reasoning": <str>, "score": <int>},
"discourse_structure": {"reasoning": <str>, "score": <int>},
"logical_consistency": {"reasoning": <str>, "score": <int>},
"temporal_causal_coherence": {"reasoning": <str>, "score": <int>},
"mutual_grounding": {"reasoning": <str>, "score": <int>},
"overall_coherence_score": <int>
}"""

COHERENCE_PROMPT = """Human: The following is dialogue content to analyze:

Content:
{content}

A: I understand.

H: Your job is to assess dialogue coherence by answering FIVE questions FIRST
(each with a short reason), THEN assigning 1-5 scores per dimension and an overall 1-5.

Guidelines:
- Answer each question in less than 50 words.
- After all answers, give only the scores (no justifications in this part):
  1-5 for each dimension and overall.
- Score with integers 1-5: 1=very poor, 2=poor, 3=fair, 4=good, 5=excellent.

Questions:
- {q_topic_coherence}
- {q_discourse_structure}
- {q_logical_consistency}
- {q_temporal_causal_coherence}
- {q_mutual_grounding}

Return ONLY the following JSON structure, with no extra commentary:
{schema}

A: Sure, here are my reasonings and scores:
"""


def build_coherence_prompt(content: str) -> str:
    dims = {d.name: d.description for d in COHERENCE.dimensions}
    return COHERENCE_PROMPT.format(
        content=(content or "").strip(),
        q_topic_coherence=dims["topic_coherence"],
        q_discourse_structure=dims["discourse_structure"],
        q_logical_consistency=dims["logical_consistency"],
        q_temporal_causal_coherence=dims["temporal_causal_coherence"],
        q_mutual_grounding=dims["mutual_grounding"],
        schema=COHERENCE_SCHEMA,
    )


# =============================================================================
# EMPATHY DOMAIN (em_)
# =============================================================================

EMPATHY_SCHEMA = """{
"emotional_awareness": {"reasoning": <str>, "score": <int>},
"emotional_validation": {"reasoning": <str>, "score": <int>},
"perspective_taking": {"reasoning": <str>, "score": <int>},
"supportive_engagement": {"reasoning": <str>, "score": <int>},
"helpful_response": {"reasoning": <str>, "score": <int>},
"overall_empathy_score": <int>
}"""

EMPATHY_PROMPT = """Human: The following is dialogue content to analyze:

Content:
{content}

A: I understand.

H: Your job is to assess dialogue empathy by answering FIVE questions FIRST
(each with a short reason), THEN assigning 1-5 scores per dimension and an overall 1-5.

Guidelines:
- Answer each question in less than 50 words.
- After all answers, give only the scores (no justifications in this part):
  1-5 for each dimension and overall.
- Score with integers 1-5: 1=very poor, 2=poor, 3=fair, 4=good, 5=excellent.

Questions:
- {q_emotional_awareness}
- {q_emotional_validation}
- {q_perspective_taking}
- {q_supportive_engagement}
- {q_helpful_response}

Return ONLY the following JSON structure, with no extra commentary:
{schema}

A: Sure, here are my reasonings and scores:
"""


def build_empathy_prompt(content: str) -> str:
    dims = {d.name: d.description for d in EMPATHY.dimensions}
    return EMPATHY_PROMPT.format(
        content=(content or "").strip(),
        q_emotional_awareness=dims["emotional_awareness"],
        q_emotional_validation=dims["emotional_validation"],
        q_perspective_taking=dims["perspective_taking"],
        q_supportive_engagement=dims["supportive_engagement"],
        q_helpful_response=dims["helpful_response"],
        schema=EMPATHY_SCHEMA,
    )


# =============================================================================
# MULTICULTURAL DOMAIN (mu_)
# =============================================================================

MULTICULTURAL_PROMPT = """You are evaluating a cross-cultural dialogue between speakers from {country_1} and {country_2}.

DIALOGUE:
{turns_text}

CULTURAL VALUE STATEMENT:
{statement}

CALIBRATION:
Across a representative set of dialogues, scores should spread across the full 0.0-1.0 range. If you find yourself rating most dialogues above 0.75 on any dimension, recheck the rubric — most dialogues are mid-quality. Use the full range. Anchors below define what each level means; pick the closest one.

TASK:
Rate each aspect on a 0.0-1.0 scale using the anchored rubric below. Three dimensions require an explicit "reasoning" field BEFORE the score; this forces evidence-based scoring.

1. CULTURAL VALUE (mu_cultural_value): How strongly is the cultural value from the statement reflected across the dialogue?
   ANCHORS:
   - 0.0 = dialogue contradicts the value or shows the opposite stance
   - 0.25 = value is absent; dialogue is neutral or off-topic from it
   - 0.5 = value is alluded to weakly or only by one speaker; not central
   - 0.75 = value clearly motivates at least one speaker's stance, with concrete textual evidence
   - 1.0 = value drives the dialogue's content; both speakers act consistently with it; multiple turns reinforce
   FORCED CONTRAST (lower the score even if the dialogue otherwise reads well):
   - Score ≤ 0.25 if the dialogue could plausibly support the OPPOSITE value
   - Score ≤ 0.5 if speakers state the value abstractly but their behavior contradicts it
   - Score ≤ 0.5 if the value is mentioned in only one turn without follow-through
   REFLECTION (required before scoring ≥ 0.75): cite the specific turn(s) and quote a phrase that demonstrates the value. If you cannot cite a specific turn, score ≤ 0.5.
{qid_addendum}
2. CULTURAL SPECIFICITY (mu_cultural_specificity): Does the dialogue clearly reflect the distinct cultural backgrounds of both speakers — in their values, norms, or ways of expressing themselves — rather than sounding generic?
   - 0.0 = generic, culture-neutral, could be from anywhere
   - 0.5 = some cultural elements present, but could be more specific
   - 1.0 = clearly reflects both cultural backgrounds, culturally specific and distinctive

3. NATURALNESS (mu_naturalness): How natural does the dialogue sound?
   ANCHORS:
   - 0.0 = stilted, robotic, obviously generated; no human would speak this way
   - 0.25 = wooden phrasing, awkward formality; comprehensible but mechanical
   - 0.5 = mostly natural with occasional awkward phrasings or unnatural transitions
   - 0.75 = sounds natural; minor stiffness in 1-2 places at most
   - 1.0 = indistinguishable from a real recorded conversation in this register
   FORCED CONTRAST:
   - Score ≤ 0.5 if any turn uses obviously unnatural phrasing (overly formal, repetitive sentence structure, AI-style enumerations)
   - Score ≤ 0.5 if greetings, sign-offs, or transitions feel templated
   - Score ≤ 0.75 if turn lengths are unrealistically uniform
   REFLECTION (required before scoring ≥ 0.75): identify which turn sounds most natural and explain why. If every turn feels equally polished, that itself is a sign of unnaturalness — score ≤ 0.75.

4. COHERENCE (mu_coherence): How clear and followable is the dialogue, considering the cultural context?
   ANCHORS:
   - 0.0 = incoherent; turns don't connect, topics shift incomprehensibly
   - 0.25 = some turns seem disconnected; reader needs guesswork to follow
   - 0.5 = mostly followable; one or two abrupt shifts or unclear references
   - 0.75 = clear progression; topic threads are easy to track
   - 1.0 = exceptionally clear; every turn builds on prior context with explicit references
   FORCED CONTRAST:
   - Score ≤ 0.5 if any pronoun reference is ambiguous
   - Score ≤ 0.5 if a topic appears without setup or disappears without resolution
   - Score ≤ 0.75 if speakers don't acknowledge each other's points
   REFLECTION (required before scoring ≥ 0.75): describe the topic flow turn by turn. If you skip turns or paraphrase loosely, score ≤ 0.75.

5. EMPATHY (mu_empathy): How much empathy and care do speakers show toward each other's cultural perspectives?
   - 0.0 = low empathy, dismissive, or culturally insensitive
   - 1.0 = high empathy and cultural sensitivity

OUTPUT FORMAT (strict JSON):
{{
  "mu_cultural_value": {{"reasoning": "<cite turn N and quote phrase, or explain why score is low>", "score": <float between 0.0 and 1.0>}},
  "mu_cultural_specificity": <float between 0.0 and 1.0>,
  "mu_naturalness": {{"reasoning": "<which turn is most/least natural and why>", "score": <float between 0.0 and 1.0>}},
  "mu_coherence": {{"reasoning": "<describe topic flow turn by turn>", "score": <float between 0.0 and 1.0>}},
  "mu_empathy": <float between 0.0 and 1.0>
}}

Return only valid JSON, no additional text."""


def build_multicultural_prompt(
    *,
    turns_text: str,
    statement: str,
    country_1: str,
    country_2: str,
    enrich_qid: bool = False,
) -> str:
    return MULTICULTURAL_PROMPT.format(
        turns_text=turns_text,
        statement=statement,
        country_1=country_1,
        country_2=country_2,
        qid_addendum=_qid_addendum(statement) if enrich_qid else "",
    )


# =============================================================================
# COMMONSENSE DOMAIN (cs_) — DUAL-PROMPT STRATEGY
# =============================================================================

# Call 1: Dimension-specific with GPT-3.5 relation responses as reference
COMMONSENSE_DIM_PROMPT = """You are evaluating a dialogue for commonsense reasoning quality.

CONTEXT:
{context}

ORIGINAL TARGET UTTERANCE:
{target_utterance}

COMMONSENSE-AUGMENTED ALTERNATIVES:
Below are modified versions of the target utterance generated by GPT-3.5.
Each highlights a specific commonsense aspect the utterance could express.

[CAUSALITY references]
{causality_refs}

[CONSISTENCY reference]
{consistency_refs}

[REACTION references]
{reaction_refs}

[DESIRE references]
{desire_refs}

TASK:
For each dimension, evaluate how well the ORIGINAL target utterance
captures that commonsense aspect, using the augmented alternatives as
reference points for what could be expressed.

Score 0.0-1.0 where:
- 0.0 = original completely misses this aspect; augmented version reveals major gap
- 0.5 = original partially captures this aspect
- 1.0 = original already fully captures this; augmented adds nothing new

OUTPUT FORMAT (strict JSON):
{{"cs_causality": <float>, "cs_consistency": <float>, "cs_reaction": <float>, "cs_desire": <float>}}

Return only valid JSON, no additional text."""

# Call 2: Dialogue-level (coherence + empathy) — no relation responses needed
COMMONSENSE_DLG_PROMPT = """You are evaluating a dialogue for commonsense reasoning quality.

CONTEXT:
{context}

TARGET UTTERANCE:
{target_utterance}

TASK:
Rate each aspect of dialogue quality on a scale from 0.0 to 1.0:

1. COHERENCE (cs_coherence): How logically coherent is the dialogue overall?
   - 0.0 = dialogue is confusing, contradictory, or hard to follow
   - 1.0 = dialogue flows logically and is easy to follow

2. EMPATHY (cs_empathy): How much understanding and care do speakers show for each other?
   - 0.0 = speakers are dismissive or uncaring
   - 1.0 = speakers demonstrate understanding and care

OUTPUT FORMAT (strict JSON):
{{"cs_coherence": <float between 0.0 and 1.0>, "cs_empathy": <float between 0.0 and 1.0>}}

Return only valid JSON, no additional text."""


def build_commonsense_dim_prompt(
    *,
    context: str,
    target_utterance: str,
    relation_responses: dict[str, str],
) -> str:
    def _ref(key: str, label: str) -> str:
        val = relation_responses.get(key, "").strip()
        return f"- {label}: {val}" if val else f"- {label}: (not available)"

    causality = "\n".join(
        [
            _ref("HinderedBy", "What could hinder this"),
            _ref("IsAfter", "What happened before"),
        ]
    )
    consistency = _ref("xAttr", "Speaker's character traits")
    reaction = "\n".join(
        [
            _ref("xReact", "Speaker's emotional reaction"),
            _ref("oReact", "Listener's emotional reaction"),
        ]
    )
    desire = "\n".join(
        [
            _ref("xWant", "What speaker wants next"),
            _ref("oWant", "What listener wants next"),
        ]
    )

    return COMMONSENSE_DIM_PROMPT.format(
        context=context,
        target_utterance=target_utterance,
        causality_refs=causality,
        consistency_refs=consistency,
        reaction_refs=reaction,
        desire_refs=desire,
    )


def build_commonsense_dlg_prompt(
    *,
    context: str,
    target_utterance: str,
) -> str:
    return COMMONSENSE_DLG_PROMPT.format(
        context=context,
        target_utterance=target_utterance,
    )
