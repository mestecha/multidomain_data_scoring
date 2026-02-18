"""Tests for pydantic models in scripts/models.py."""

from __future__ import annotations


import pytest
from pydantic import ValidationError

from scripts.models import (
    BatchEntry,
    ContrastiveDirection,
    Difficulty,
    DimensionDef,
    DomainConfig,
    DomainName,
    ManifestItem,
    Message,
    PairEvaluation,
    PairLabel,
    PairMetadata,
    ScoreScale,
    Split,
    Stage1Entry,
    Stage2Pair,
    VariantType,
)


# ── DomainConfig ──────────────────────────────────────────────────────────


class TestDomainConfig:
    """Tests for DomainConfig model."""

    @pytest.fixture()
    def config_with_overall(self) -> DomainConfig:
        return DomainConfig(
            name=DomainName.COHERENCE,
            prefix="co",
            dimensions=(
                DimensionDef(
                    name="dim_a",
                    description="First dimension",
                    is_characterizing=True,
                ),
                DimensionDef(
                    name="dim_b",
                    description="Second dimension",
                    is_characterizing=False,
                ),
                DimensionDef(
                    name="overall_score",
                    description="Overall",
                    is_characterizing=False,
                ),
            ),
            score_scale=ScoreScale.SCALE_1_5,
            overall_dim_name="co_overall_score",
        )

    @pytest.fixture()
    def config_without_overall(self) -> DomainConfig:
        return DomainConfig(
            name=DomainName.COMMONSENSE,
            prefix="cs",
            dimensions=(
                DimensionDef(
                    name="causality",
                    description="Causality",
                    is_characterizing=True,
                ),
                DimensionDef(
                    name="consistency",
                    description="Consistency",
                    is_characterizing=True,
                ),
            ),
            score_scale=ScoreScale.SCALE_0_1,
            overall_dim_name=None,
        )

    def test_has_overall_true(self, config_with_overall: DomainConfig) -> None:
        assert config_with_overall.has_overall is True

    def test_has_overall_false(self, config_without_overall: DomainConfig) -> None:
        assert config_without_overall.has_overall is False

    def test_dim_names(self, config_with_overall: DomainConfig) -> None:
        assert config_with_overall.dim_names == [
            "dim_a",
            "dim_b",
            "overall_score",
        ]

    def test_prefixed_dim_names(self, config_with_overall: DomainConfig) -> None:
        assert config_with_overall.prefixed_dim_names == [
            "co_dim_a",
            "co_dim_b",
            "co_overall_score",
        ]

    def test_characterizing_dims(self, config_with_overall: DomainConfig) -> None:
        assert config_with_overall.characterizing_dims == ["co_dim_a"]

    def test_characterizing_dims_multiple(
        self, config_without_overall: DomainConfig
    ) -> None:
        assert config_without_overall.characterizing_dims == [
            "cs_causality",
            "cs_consistency",
        ]

    def test_frozen_immutability(self, config_with_overall: DomainConfig) -> None:
        with pytest.raises(ValidationError):
            config_with_overall.prefix = "xx"


# ── BatchEntry ────────────────────────────────────────────────────────────


class TestBatchEntry:
    """Tests for BatchEntry model."""

    def test_to_api_dict_format(self) -> None:
        entry = BatchEntry(
            custom_id="test-001",
            prompt="Evaluate this dialogue.",
            model="gpt-5.1-batch",
        )
        result = entry.to_api_dict()

        assert result["custom_id"] == "test-001"
        assert result["method"] == "POST"
        assert result["url"] == "/v1/chat/completions"
        assert result["body"]["model"] == "gpt-5.1-batch"
        assert result["body"]["messages"] == [
            {"role": "user", "content": "Evaluate this dialogue."}
        ]
        assert result["body"]["temperature"] == 0.0
        assert result["body"]["response_format"] == {"type": "json_object"}

    def test_to_api_dict_custom_model(self) -> None:
        entry = BatchEntry(
            custom_id="test-002",
            prompt="Hello",
            model="gpt-4o",
        )
        result = entry.to_api_dict()
        assert result["body"]["model"] == "gpt-4o"

    def test_default_model(self) -> None:
        entry = BatchEntry(custom_id="test-003", prompt="Hello")
        assert entry.model == "gpt-5.1-batch"


# ── ManifestItem ──────────────────────────────────────────────────────────


class TestManifestItem:
    """Tests for ManifestItem model."""

    def test_exclude_none_serialization(self) -> None:
        item = ManifestItem(
            custom_id="test-001",
            content="Some dialogue",
            label="Yes",
        )
        dumped = item.model_dump(exclude_none=True)
        assert "custom_id" in dumped
        assert "content" in dumped
        assert "label" in dumped
        # None fields should not be present
        assert "pid" not in dumped
        assert "uid" not in dumped
        assert "country_1" not in dumped
        assert "country_2" not in dumped
        assert "source_file" not in dumped

    def test_all_fields_present(self) -> None:
        item = ManifestItem(
            custom_id="test-002",
            content="content",
            label="No",
            pid="P001",
            uid="U001",
            country_1="US",
            country_2="JP",
            source_file="data.csv",
        )
        dumped = item.model_dump(exclude_none=True)
        assert len(dumped) == 8

    def test_empty_content_default(self) -> None:
        item = ManifestItem(custom_id="test-003")
        assert item.content == ""


# ── Stage1Entry ───────────────────────────────────────────────────────────


class TestStage1Entry:
    """Tests for Stage1Entry model."""

    def test_scores_not_empty_validation(self) -> None:
        with pytest.raises(ValidationError, match="at least one non-None value"):
            Stage1Entry(
                dialogue_id="S1D-000001",
                source_id="co-abc",
                domain=DomainName.COHERENCE,
                messages=[Message(role="user", content="Hi")],
                scores={},
            )

    def test_all_none_scores_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one non-None value"):
            Stage1Entry(
                dialogue_id="S1D-000001",
                source_id="co-abc",
                domain=DomainName.COHERENCE,
                messages=[Message(role="user", content="Hi")],
                scores={"co_topic_coherence": None, "em_emotional_awareness": None},
            )

    def test_valid_entry(self) -> None:
        entry = Stage1Entry(
            dialogue_id="S1D-000001",
            source_id="co-abc",
            domain=DomainName.COHERENCE,
            messages=[Message(role="user", content="Hi")],
            scores={"co_topic_coherence": 0.75, "em_emotional_awareness": None},
        )
        assert entry.dialogue_id == "S1D-000001"
        assert entry.domain == DomainName.COHERENCE

    def test_null_scores_accepted(self) -> None:
        scores = {"co_topic_coherence": 0.75, "em_emotional_awareness": None}
        entry = Stage1Entry(
            dialogue_id="S1D-000001",
            source_id="co-abc",
            domain=DomainName.COHERENCE,
            messages=[Message(role="user", content="Hi")],
            scores=scores,
        )
        assert entry.scores["co_topic_coherence"] == 0.75
        assert entry.scores["em_emotional_awareness"] is None

    def test_characterizing_avg(self) -> None:
        entry = Stage1Entry(
            dialogue_id="S1D-000001",
            source_id="co-abc",
            domain=DomainName.COHERENCE,
            messages=[Message(role="user", content="Hi")],
            scores={
                "co_topic_coherence": 0.80,
                "co_logical_consistency": 0.60,
                "co_discourse_structure": 0.50,
            },
        )
        avg = entry.characterizing_avg(["co_topic_coherence", "co_logical_consistency"])
        assert avg == pytest.approx(0.70)

    def test_characterizing_avg_skips_none(self) -> None:
        entry = Stage1Entry(
            dialogue_id="S1D-000001",
            source_id="co-abc",
            domain=DomainName.COHERENCE,
            messages=[Message(role="user", content="Hi")],
            scores={
                "co_topic_coherence": 0.80,
                "co_logical_consistency": None,
            },
        )
        avg = entry.characterizing_avg(
            ["co_topic_coherence", "co_logical_consistency"],
        )
        # Only co_topic_coherence is non-None, so avg == 0.80
        assert avg == pytest.approx(0.80)

    def test_characterizing_avg_missing_dims(self) -> None:
        entry = Stage1Entry(
            dialogue_id="S1D-000001",
            source_id="co-abc",
            domain=DomainName.COHERENCE,
            messages=[Message(role="user", content="Hi")],
            scores={"co_topic_coherence": 0.80},
        )
        avg = entry.characterizing_avg(["co_nonexistent_dim"])
        assert avg is None

    def test_characterizing_avg_partial_match(self) -> None:
        entry = Stage1Entry(
            dialogue_id="S1D-000001",
            source_id="co-abc",
            domain=DomainName.COHERENCE,
            messages=[Message(role="user", content="Hi")],
            scores={"co_topic_coherence": 0.80},
        )
        avg = entry.characterizing_avg(["co_topic_coherence", "co_nonexistent"])
        assert avg == pytest.approx(0.80)

    def test_domain_metadata_default_none(self) -> None:
        entry = Stage1Entry(
            dialogue_id="S1D-000001",
            source_id="co-abc",
            domain=DomainName.COHERENCE,
            messages=[Message(role="user", content="Hi")],
            scores={"co_topic_coherence": 0.75},
        )
        assert entry.domain_metadata is None

    def test_domain_metadata_round_trip(self) -> None:
        metadata = {"country_1": "Japan", "country_2": "USA"}
        entry = Stage1Entry(
            dialogue_id="S1D-000001",
            source_id="mu-abc",
            domain=DomainName.MULTICULTURAL,
            messages=[Message(role="user", content="Hi")],
            scores={"mu_cultural_value": 0.75},
            domain_metadata=metadata,
        )
        json_str = entry.model_dump_json()
        restored = Stage1Entry.model_validate_json(json_str)
        assert restored.domain_metadata == metadata

    def test_json_round_trip(self) -> None:
        entry = Stage1Entry(
            dialogue_id="S1D-000001",
            source_id="co-abc",
            domain=DomainName.COHERENCE,
            messages=[
                Message(role="user", content="Hi"),
                Message(role="assistant", content="Hello!"),
            ],
            split=Split.TRAIN,
            scores={"co_topic_coherence": 0.75, "co_logical_consistency": 0.80},
        )
        json_str = entry.model_dump_json()
        restored = Stage1Entry.model_validate_json(json_str)

        assert restored.dialogue_id == entry.dialogue_id
        assert restored.source_id == entry.source_id
        assert restored.domain == entry.domain
        assert restored.split == entry.split
        assert restored.scores == entry.scores
        assert len(restored.messages) == len(entry.messages)
        assert restored.messages[0].role == "user"
        assert restored.messages[1].role == "assistant"

    def test_split_default_none(self) -> None:
        entry = Stage1Entry(
            dialogue_id="S1D-000001",
            source_id="co-abc",
            domain=DomainName.COHERENCE,
            messages=[Message(role="user", content="Hi")],
            scores={"co_topic_coherence": 0.75},
        )
        assert entry.split is None


# ── Message ───────────────────────────────────────────────────────────────


class TestMessage:
    """Tests for Message model."""

    def test_valid_user_role(self) -> None:
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"

    def test_valid_assistant_role(self) -> None:
        msg = Message(role="assistant", content="Hi there")
        assert msg.role == "assistant"

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="system", content="Not allowed")

    def test_invalid_role_arbitrary_string(self) -> None:
        with pytest.raises(ValidationError):
            Message(role="human", content="Not allowed")


# ── Stage 2 Models ────────────────────────────────────────────────────────


class TestPairMetadata:
    """Tests for PairMetadata model."""

    def test_creation(self) -> None:
        meta = PairMetadata(
            domain=DomainName.COHERENCE,
            split=Split.TRAIN,
            contrastive_direction=ContrastiveDirection.POSITIVE,
            target_dimensions=["co_topic_coherence"],
            generation_model="gpt-5.1",
            turn_count=4,
            variant_type=VariantType.GLOBAL_IMPROVE,
            difficulty=Difficulty.MEDIUM,
            label=PairLabel.GLOBAL_PASS,
        )
        assert meta.domain == DomainName.COHERENCE
        assert meta.turn_count == 4
        assert meta.label == PairLabel.GLOBAL_PASS


class TestPairEvaluation:
    """Tests for PairEvaluation model."""

    def test_default_method(self) -> None:
        evaluation = PairEvaluation(
            stage_1_scores={"co_topic_coherence": 0.75},
            variant_scores={"co_topic_coherence": 0.90},
        )
        assert evaluation.method == "lm_judge"

    def test_custom_method(self) -> None:
        evaluation = PairEvaluation(
            method="human",
            stage_1_scores={"co_topic_coherence": 0.75},
            variant_scores={"co_topic_coherence": 0.90},
        )
        assert evaluation.method == "human"


class TestStage2Pair:
    """Tests for Stage2Pair serialization."""

    def test_serialization_round_trip(self) -> None:
        pair = Stage2Pair(
            pair_id="PAIR-000001",
            source_dialogue_id="S1D-000001",
            metadata=PairMetadata(
                domain=DomainName.COHERENCE,
                split=Split.TRAIN,
                contrastive_direction=ContrastiveDirection.POSITIVE,
                target_dimensions=["co_topic_coherence"],
                generation_model="gpt-5.1",
                turn_count=4,
                variant_type=VariantType.GLOBAL_IMPROVE,
                difficulty=Difficulty.MEDIUM,
                label=PairLabel.GLOBAL_PASS,
            ),
            evaluation=PairEvaluation(
                stage_1_scores={"co_topic_coherence": 0.75},
                variant_scores={"co_topic_coherence": 0.90},
            ),
            messages=[Message(role="user", content="Hi")],
            chosen=[Message(role="assistant", content="Good response")],
            rejected=[Message(role="assistant", content="Bad response")],
        )

        json_str = pair.model_dump_json()
        restored = Stage2Pair.model_validate_json(json_str)

        assert restored.pair_id == pair.pair_id
        assert restored.source_dialogue_id == pair.source_dialogue_id
        assert restored.metadata.domain == DomainName.COHERENCE
        assert restored.evaluation.method == "lm_judge"
        assert len(restored.chosen) == 1
        assert len(restored.rejected) == 1
