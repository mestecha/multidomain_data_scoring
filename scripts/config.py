"""single source of truth for all pipeline configuration."""

from __future__ import annotations

from pathlib import Path

from scripts.models import (
    DimensionDef,
    DomainConfig,
    DomainName,
    ScoreScale,
)

# ── Domain Configurations ──────────────────────────────────────────────────

COHERENCE = DomainConfig(
    name=DomainName.COHERENCE,
    prefix="co",
    dimensions=(
        DimensionDef(
            name="topic_coherence",
            description=(
                "Does the conversation stay focused on related topics without "
                "random jumps, and when topics change, do the changes make sense?"
            ),
            is_characterizing=True,
        ),
        DimensionDef(
            name="discourse_structure",
            description=(
                "Do the turns of conversation follow recognizable patterns, like "
                "questions being answered, requests being responded to, and "
                "greetings at the start?"
            ),
        ),
        DimensionDef(
            name="logical_consistency",
            description=(
                "Are the facts and claims in the conversation free from "
                "contradictions, and do they make sense given the dialogue's context?"
            ),
            is_characterizing=True,
        ),
        DimensionDef(
            name="temporal_causal_coherence",
            description=(
                "Do the events and actions mentioned in the conversation happen in a "
                "believable time order, and do they explain each other in a logical way?"
            ),
        ),
        DimensionDef(
            name="mutual_grounding",
            description=(
                "Do the people in the conversation show they understand and build on "
                "what the other said?"
            ),
        ),
        DimensionDef(
            name="overall_coherence_score",
            description="Overall coherence quality of the dialogue.",
        ),
    ),
    score_scale=ScoreScale.SCALE_1_5,
    overall_dim_name="co_overall_coherence_score",
)

EMPATHY = DomainConfig(
    name=DomainName.EMPATHY,
    prefix="em",
    dimensions=(
        DimensionDef(
            name="emotional_awareness",
            description=(
                "Does the responder recognize and acknowledge the speaker's "
                "emotional state?"
            ),
            is_characterizing=True,
        ),
        DimensionDef(
            name="emotional_validation",
            description=(
                "Does the responder validate the speaker's feelings without "
                "dismissing or minimizing them?"
            ),
        ),
        DimensionDef(
            name="perspective_taking",
            description=(
                "Does the responder show understanding of the speaker's perspective "
                "and situation?"
            ),
            is_characterizing=True,
        ),
        DimensionDef(
            name="supportive_engagement",
            description=(
                "Does the responder engage supportively, showing genuine care "
                "and concern?"
            ),
        ),
        DimensionDef(
            name="helpful_response",
            description=(
                "Does the responder offer helpful suggestions or comfort appropriate "
                "to the situation?"
            ),
        ),
        DimensionDef(
            name="overall_empathy_score",
            description="Overall empathy quality of the dialogue.",
        ),
    ),
    score_scale=ScoreScale.SCALE_1_5,
    overall_dim_name="em_overall_empathy_score",
)

COMMONSENSE = DomainConfig(
    name=DomainName.COMMONSENSE,
    prefix="cs",
    dimensions=(
        DimensionDef(
            name="causality",
            description=(
                "Are causal and temporal relationships valid and plausible? "
                "Consider: What could hinder the action? What happens before/after?"
            ),
            is_characterizing=True,
        ),
        DimensionDef(
            name="consistency",
            description=(
                "Do participants act consistently with inferable character traits? "
                "Consider: Are behaviors consistent with apparent personality, "
                "knowledge, social role?"
            ),
            is_characterizing=True,
        ),
        DimensionDef(
            name="reaction",
            description=(
                "How plausible are the emotional reactions shown? "
                "Consider: Do speakers react emotionally in ways that make sense "
                "given the context?"
            ),
            is_characterizing=True,
        ),
        DimensionDef(
            name="desire",
            description=(
                "How coherent are the expressed desires and intentions? "
                "Consider: Do the wants and goals expressed make sense given the "
                "situation?"
            ),
            is_characterizing=True,
        ),
        DimensionDef(
            name="coherence",
            description="How logically coherent is the dialogue overall?",
        ),
        DimensionDef(
            name="empathy",
            description=(
                "How much understanding and care do speakers show for each other?"
            ),
        ),
    ),
    score_scale=ScoreScale.SCALE_0_1,
    overall_dim_name=None,
)

MULTICULTURAL = DomainConfig(
    name=DomainName.MULTICULTURAL,
    prefix="mu",
    dimensions=(
        DimensionDef(
            name="cultural_value",
            description=(
                "How strongly is the cultural value from the statement reflected "
                "across the dialogue?"
            ),
            is_characterizing=True,
        ),
        DimensionDef(
            name="cultural_specificity",
            description=(
                "Does the dialogue clearly reflect the distinct cultural backgrounds "
                "of both speakers rather than sounding generic?"
            ),
            is_characterizing=True,
        ),
        DimensionDef(
            name="naturalness",
            description="How natural does the dialogue sound?",
        ),
        DimensionDef(
            name="coherence",
            description=(
                "How clear and followable is the dialogue, considering the "
                "cultural context?"
            ),
        ),
        DimensionDef(
            name="empathy",
            description=(
                "How much empathy and care do speakers show toward each other's "
                "cultural perspectives?"
            ),
        ),
    ),
    score_scale=ScoreScale.SCALE_0_1,
    overall_dim_name=None,
)

# ── Registry ───────────────────────────────────────────────────────────────

DOMAINS: dict[DomainName, DomainConfig] = {
    DomainName.COHERENCE: COHERENCE,
    DomainName.EMPATHY: EMPATHY,
    DomainName.COMMONSENSE: COMMONSENSE,
    DomainName.MULTICULTURAL: MULTICULTURAL,
}

# ── Commonsense Relations ──────────────────────────────────────────────────

CS_RELATION_TO_DIM: dict[str, str] = {
    "HinderedBy": "cs_causality",
    "IsAfter": "cs_causality",
    "xAttr": "cs_consistency",
    "xReact": "cs_reaction",
    "oReact": "cs_reaction",
    "xWant": "cs_desire",
    "oWant": "cs_desire",
}

# ── Pipeline Constants ─────────────────────────────────────────────────────

# Stage 2 continuation length distribution
TURN_DISTRIBUTION: dict[int, float] = {1: 0.15, 3: 0.40, 5: 0.30, 7: 0.15}

# Train/test split ratios
SPLIT_RATIOS: dict[str, float] = {"train": 0.90, "test": 0.10}

MULTICULTURAL_TEST_DIR: Path = Path("data/input/multicultural/countries/test")

# Sampling turn count buckets (inclusive bounds)
TURN_BUCKETS: list[tuple[int, int]] = [(2, 6), (7, 10), (11, 16), (17, 999)]

# Coherence label distribution for stratified sampling
COHERENCE_LABEL_RATIO: dict[str, float] = {"Yes": 0.70, "No": 0.30}

# ── Helpers ────────────────────────────────────────────────────────────────


def get_domain(name: str) -> DomainConfig:
    return DOMAINS[DomainName(name)]


def all_prefixed_dimensions() -> list[str]:
    dims: list[str] = []
    for config in DOMAINS.values():
        dims.extend(config.prefixed_dim_names)
    return dims
