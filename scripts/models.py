"""pydantic models for all pipeline stages."""

from __future__ import annotations

from enum import Enum
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, field_validator


# ── Enums ──────────────────────────────────────────────────────────────────


class DomainName(str, Enum):
    COHERENCE = "coherence"
    EMPATHY = "empathy"
    COMMONSENSE = "commonsense"
    MULTICULTURAL = "multicultural"


class ScoreScale(str, Enum):
    SCALE_1_5 = "1-5"
    SCALE_0_1 = "0-1"


class Split(str, Enum):
    TRAIN = "train"
    TEST = "test"


class ContrastiveDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class VariantType(str, Enum):
    GLOBAL_IMPROVE = "global_improve"
    GLOBAL_DEGRADE = "global_degrade"
    DIMENSION_TARGETED = "dimension_targeted"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PairLabel(str, Enum):
    GLOBAL_PASS = "global_pass"
    TARGET_PASS = "target_pass"
    TARGET_COARSE_PASS = "target_coarse_pass"
    GLOBAL_FLIP_PASS = "global_flip_pass"
    TARGET_FLIP_PASS = "target_flip_pass"
    REJECT = "reject"


# ── Core ───────────────────────────────────────────────────────────────────


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class DimensionDef(BaseModel):
    name: str
    description: str
    is_characterizing: bool = False


class DomainConfig(BaseModel, frozen=True):
    name: DomainName
    prefix: str
    dimensions: tuple[DimensionDef, ...]
    score_scale: ScoreScale
    overall_dim_name: str | None = None

    @property
    def has_overall(self) -> bool:
        return self.overall_dim_name is not None

    @property
    def dim_names(self) -> list[str]:
        return [d.name for d in self.dimensions]

    @property
    def prefixed_dim_names(self) -> list[str]:
        return [f"{self.prefix}_{d.name}" for d in self.dimensions]

    @property
    def characterizing_dims(self) -> list[str]:
        return [
            f"{self.prefix}_{d.name}" for d in self.dimensions if d.is_characterizing
        ]


# ── Stage 1 ────────────────────────────────────────────────────────────────


class BatchEntry(BaseModel):
    custom_id: str
    prompt: str
    model: str = "gpt-5.1-batch"

    def to_api_dict(self) -> dict[str, Any]:
        """convert to azure openai batch api format."""
        return {
            "custom_id": self.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [{"role": "user", "content": self.prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
        }


class ManifestItem(BaseModel):
    """tracks batch items back to source data."""

    custom_id: str
    content: str = ""
    label: str | None = None
    pid: str | None = None
    uid: str | None = None
    country_1: str | None = None
    country_2: str | None = None
    source_file: str | None = None


class Stage1Entry(BaseModel):
    """scored dialogue; all 23 dimensions present, non-domain dims are None."""

    dialogue_id: str
    source_id: str
    domain: DomainName
    messages: list[Message]
    split: Split | None = None
    scores: dict[str, float | None]
    domain_metadata: dict[str, str] | None = None

    @field_validator("scores")
    @classmethod
    def scores_not_empty(
        cls, v: dict[str, float | None],
    ) -> dict[str, float | None]:
        if not any(val is not None for val in v.values()):
            raise ValueError("scores must have at least one non-None value")
        return v

    def characterizing_avg(self, char_dims: list[str]) -> float | None:
        values = [
            self.scores[d]
            for d in char_dims
            if d in self.scores and self.scores[d] is not None
        ]
        return mean(values) if values else None


# ── Stage 2 ────────────────────────────────────────────────────────────────


class Stage2Candidate(BaseModel):
    """internal only, not serialized to output."""

    dialogue_id: str
    domain: DomainName
    messages: list[Message]
    characterizing_scores: dict[str, float]
    target_dimensions: list[str]
    contrastive_direction: ContrastiveDirection
    variant_type: VariantType
    domain_metadata: dict[str, str] | None = None


class Stage2Variant(BaseModel):
    """internal only, not serialized to output."""

    candidate_id: str
    variant_id: str
    variant_type: VariantType
    direction: ContrastiveDirection
    prefix_messages: list[Message]
    variant_messages: list[Message]
    generation_model: str
    target_dimensions: list[str] = []


class EvalResult(BaseModel):
    """internal only, not serialized to output."""

    variant_id: str
    original_scores: dict[str, float]
    variant_scores: dict[str, float]
    label: PairLabel


class PairContrast(BaseModel, frozen=True):
    scope: Literal["global", "target"]
    direction: ContrastiveDirection
    decision: PairLabel
    intent_followed: bool


class PairDimensions(BaseModel, frozen=True):
    domain: list[str]
    target: list[str]
    decision: list[str]


class PairMetadata(BaseModel, frozen=True):
    domain: DomainName
    split: Split
    generation_model: str
    turn_count: int
    difficulty: Difficulty
    contrast: PairContrast
    dimensions: PairDimensions


class PairEvaluation(BaseModel):
    method: Literal["lm_judge", "human", "rules"] = "lm_judge"
    stage_1_scores: dict[str, float]
    stage_2_scores: dict[str, float]
    chosen_score_source: Literal["stage_1", "stage_2"]
    rejected_score_source: Literal["stage_1", "stage_2"]


class Stage2Pair(BaseModel):
    """final output: one preference pair for dpo training."""

    pair_id: str
    source_dialogue_id: str
    metadata: PairMetadata
    evaluation: PairEvaluation
    messages: list[Message]
    chosen: list[Message]
    rejected: list[Message]
