"""Tests for Stage 2 candidate selection logic."""

from __future__ import annotations


from scripts.models import (
    ContrastiveDirection,
    DomainName,
    Message,
    Split,
    Stage1Entry,
    VariantType,
)
from scripts.stage_2.select import (
    LOW_UPPER,
    MEDIUM_DIRECTION_THRESHOLD,
    MEDIUM_UPPER,
    _classify_candidates,
    _classify_commonsense_candidates,
    select_candidates,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_entry(
    scores: dict[str, float | None],
    domain: DomainName = DomainName.COMMONSENSE,
    split: Split | None = Split.TRAIN,
    dialogue_id: str = "dlg-001",
    domain_metadata: dict[str, str] | None = None,
) -> Stage1Entry:
    """Helper to build a minimal Stage1Entry."""
    return Stage1Entry(
        dialogue_id=dialogue_id,
        source_id="src-001",
        domain=domain,
        split=split,
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I am well."),
        ],
        scores=scores,
        domain_metadata=domain_metadata,
    )


# ── Low tier -> GLOBAL_IMPROVE (positive) + DIMENSION_TARGETED ───────────


class TestLowTier:
    """Low-scoring entries should get positive direction with 1 global + N targeted."""

    def test_low_score_produces_3_candidates(self) -> None:
        entry = _make_entry(
            {"co_topic_coherence": 0.2, "co_logical_consistency": 0.1},
            domain=DomainName.COHERENCE,
        )
        char_dims = ["co_topic_coherence", "co_logical_consistency"]
        candidates = _classify_candidates(entry, char_dims)

        assert len(candidates) == 3
        # First is global
        assert candidates[0].variant_type == VariantType.GLOBAL_IMPROVE
        assert candidates[0].target_dimensions == char_dims
        # Remaining are dimension-targeted
        for c in candidates[1:]:
            assert c.variant_type == VariantType.DIMENSION_TARGETED
            assert len(c.target_dimensions) == 1
        assert all(
            c.contrastive_direction == ContrastiveDirection.POSITIVE
            for c in candidates
        )

    def test_low_boundary(self) -> None:
        """Score exactly at LOW_UPPER boundary falls into medium tier."""
        entry = _make_entry(
            {"co_topic_coherence": LOW_UPPER, "co_logical_consistency": LOW_UPPER},
            domain=DomainName.COHERENCE,
        )
        char_dims = ["co_topic_coherence", "co_logical_consistency"]
        candidates = _classify_candidates(entry, char_dims)

        assert len(candidates) == 3
        # Avg == 0.4 is NOT < 0.4, so falls into medium tier with positive direction
        assert candidates[0].variant_type in (
            VariantType.GLOBAL_IMPROVE,
            VariantType.GLOBAL_DEGRADE,
        )

    def test_low_zero_scores(self) -> None:
        entry = _make_entry(
            {"co_topic_coherence": 0.0, "co_logical_consistency": 0.0},
            domain=DomainName.COHERENCE,
        )
        char_dims = ["co_topic_coherence", "co_logical_consistency"]
        candidates = _classify_candidates(entry, char_dims)

        assert len(candidates) == 3
        assert all(
            c.contrastive_direction == ContrastiveDirection.POSITIVE
            for c in candidates
        )
        assert candidates[0].variant_type == VariantType.GLOBAL_IMPROVE


# ── High tier -> GLOBAL_DEGRADE (negative) + DIMENSION_TARGETED ──────────


class TestHighTier:
    """High-scoring entries should get negative direction with 1 global + N targeted."""

    def test_high_score_produces_3_candidates(self) -> None:
        entry = _make_entry(
            {"co_topic_coherence": 0.9, "co_logical_consistency": 0.85},
            domain=DomainName.COHERENCE,
        )
        char_dims = ["co_topic_coherence", "co_logical_consistency"]
        candidates = _classify_candidates(entry, char_dims)

        assert len(candidates) == 3
        assert candidates[0].variant_type == VariantType.GLOBAL_DEGRADE
        assert all(
            c.contrastive_direction == ContrastiveDirection.NEGATIVE
            for c in candidates
        )

    def test_high_boundary(self) -> None:
        """Score exactly at MEDIUM_UPPER boundary falls into high tier."""
        entry = _make_entry(
            {"co_topic_coherence": MEDIUM_UPPER, "co_logical_consistency": MEDIUM_UPPER},
            domain=DomainName.COHERENCE,
        )
        char_dims = ["co_topic_coherence", "co_logical_consistency"]
        candidates = _classify_candidates(entry, char_dims)

        assert len(candidates) == 3
        assert candidates[0].variant_type == VariantType.GLOBAL_DEGRADE

    def test_high_perfect_scores(self) -> None:
        entry = _make_entry(
            {"co_topic_coherence": 1.0, "co_logical_consistency": 1.0},
            domain=DomainName.COHERENCE,
        )
        char_dims = ["co_topic_coherence", "co_logical_consistency"]
        candidates = _classify_candidates(entry, char_dims)

        assert len(candidates) == 3
        assert candidates[0].variant_type == VariantType.GLOBAL_DEGRADE


# ── Medium tier -> GLOBAL + DIMENSION_TARGETED ────────────────────────────


class TestMediumTier:
    """Medium-scoring entries get 1 global + N targeted, direction based on threshold."""

    def test_medium_low_gets_positive_direction(self) -> None:
        """avg < 0.55 -> positive direction."""
        entry = _make_entry(
            {"co_topic_coherence": 0.45, "co_logical_consistency": 0.50},
            domain=DomainName.COHERENCE,
        )
        char_dims = ["co_topic_coherence", "co_logical_consistency"]
        candidates = _classify_candidates(entry, char_dims)

        assert len(candidates) == 3
        assert candidates[0].variant_type == VariantType.GLOBAL_IMPROVE
        assert all(
            c.contrastive_direction == ContrastiveDirection.POSITIVE
            for c in candidates
        )

    def test_medium_high_gets_negative_direction(self) -> None:
        """avg >= 0.55 -> negative direction."""
        entry = _make_entry(
            {"co_topic_coherence": 0.60, "co_logical_consistency": 0.65},
            domain=DomainName.COHERENCE,
        )
        char_dims = ["co_topic_coherence", "co_logical_consistency"]
        candidates = _classify_candidates(entry, char_dims)

        assert len(candidates) == 3
        assert candidates[0].variant_type == VariantType.GLOBAL_DEGRADE
        assert all(
            c.contrastive_direction == ContrastiveDirection.NEGATIVE
            for c in candidates
        )

    def test_medium_direction_boundary(self) -> None:
        """avg exactly at 0.55 -> negative direction."""
        entry = _make_entry(
            {
                "co_topic_coherence": MEDIUM_DIRECTION_THRESHOLD,
                "co_logical_consistency": MEDIUM_DIRECTION_THRESHOLD,
            },
            domain=DomainName.COHERENCE,
        )
        char_dims = ["co_topic_coherence", "co_logical_consistency"]
        candidates = _classify_candidates(entry, char_dims)

        assert len(candidates) == 3
        assert all(
            c.contrastive_direction == ContrastiveDirection.NEGATIVE
            for c in candidates
        )

    def test_medium_always_deterministic(self) -> None:
        """No random branching — always 3 candidates with same structure."""
        entry = _make_entry(
            {"co_topic_coherence": 0.50, "co_logical_consistency": 0.50},
            domain=DomainName.COHERENCE,
        )
        char_dims = ["co_topic_coherence", "co_logical_consistency"]

        for _ in range(20):
            candidates = _classify_candidates(entry, char_dims)
            assert len(candidates) == 3
            assert candidates[0].variant_type == VariantType.GLOBAL_IMPROVE
            assert candidates[1].variant_type == VariantType.DIMENSION_TARGETED
            assert candidates[2].variant_type == VariantType.DIMENSION_TARGETED


# ── Missing scores ────────────────────────────────────────────────────────


class TestMissingScores:
    """Entries with missing characterizing scores should be excluded."""

    def test_no_characterizing_scores_returns_empty(self) -> None:
        entry = _make_entry({"cs_coherence": 0.5, "cs_empathy": 0.6})
        char_dims = ["cs_causality", "cs_consistency"]
        candidates = _classify_candidates(entry, char_dims)

        assert candidates == []

    def test_partial_scores_still_classified(self) -> None:
        """If at least one characterizing dim is present, we get candidates."""
        entry = _make_entry(
            {"co_topic_coherence": 0.3},
            domain=DomainName.COHERENCE,
        )
        char_dims = ["co_topic_coherence", "co_logical_consistency"]
        candidates = _classify_candidates(entry, char_dims)

        assert len(candidates) == 3


# ── select_candidates integration ─────────────────────────────────────────


class TestSelectCandidates:
    """Integration tests for the full select_candidates function."""

    def test_filters_to_train_split(self) -> None:
        train_entry = _make_entry(
            {"co_topic_coherence": 0.2, "co_logical_consistency": 0.2},
            domain=DomainName.COHERENCE,
            split=Split.TRAIN,
            dialogue_id="train-1",
        )
        val_entry = _make_entry(
            {"co_topic_coherence": 0.2, "co_logical_consistency": 0.2},
            domain=DomainName.COHERENCE,
            split=Split.VAL,
            dialogue_id="val-1",
        )
        test_entry = _make_entry(
            {"co_topic_coherence": 0.2, "co_logical_consistency": 0.2},
            domain=DomainName.COHERENCE,
            split=Split.TEST,
            dialogue_id="test-1",
        )

        candidates = select_candidates([train_entry, val_entry, test_entry])
        # 1 dialogue × 3 variants
        assert len(candidates) == 3
        assert all(c.dialogue_id == "train-1" for c in candidates)

    def test_none_split_treated_as_train(self) -> None:
        entry = _make_entry(
            {"co_topic_coherence": 0.2, "co_logical_consistency": 0.2},
            domain=DomainName.COHERENCE,
            split=None,
        )
        candidates = select_candidates([entry])
        assert len(candidates) == 3

    def test_max_samples_caps_output(self) -> None:
        entries = [
            _make_entry(
                {"co_topic_coherence": 0.2, "co_logical_consistency": 0.2},
                domain=DomainName.COHERENCE,
                dialogue_id=f"dlg-{i:03d}",
            )
            for i in range(10)
        ]
        # 10 dialogues × 3 = 30 candidates, cap to 5
        candidates = select_candidates(entries, max_samples=5)
        assert len(candidates) == 5

    def test_domain_filter(self) -> None:
        cs_entry = _make_entry(
            {"cs_causality": 0.2, "cs_consistency": 0.2, "cs_reaction": 0.2, "cs_desire": 0.2},
            domain=DomainName.COMMONSENSE,
            dialogue_id="cs-1",
        )
        co_entry = _make_entry(
            {"co_topic_coherence": 0.2, "co_logical_consistency": 0.2},
            domain=DomainName.COHERENCE,
            dialogue_id="co-1",
        )
        candidates = select_candidates(
            [cs_entry, co_entry], domain=DomainName.COMMONSENSE,
        )
        # 1 CS dialogue × 4 variants
        assert len(candidates) == 4
        assert all(c.dialogue_id == "cs-1" for c in candidates)

    def test_characterizing_scores_populated(self) -> None:
        entry = _make_entry(
            {"cs_causality": 0.2, "cs_consistency": 0.3, "cs_reaction": 0.25, "cs_desire": 0.15},
        )
        candidates = select_candidates([entry])
        # All 4 candidates share the same characterizing_scores
        for c in candidates:
            assert c.characterizing_scores == {
                "cs_causality": 0.2,
                "cs_consistency": 0.3,
                "cs_reaction": 0.25,
                "cs_desire": 0.15,
            }

    def test_domain_metadata_threaded_to_candidate(self) -> None:
        metadata = {"country_1": "Japan", "country_2": "USA"}
        entry = _make_entry(
            {"mu_cultural_value": 0.2, "mu_cultural_specificity": 0.2},
            domain=DomainName.MULTICULTURAL,
            domain_metadata=metadata,
        )
        candidates = select_candidates([entry])
        # 3 candidates, all carry metadata
        assert len(candidates) == 3
        assert all(c.domain_metadata == metadata for c in candidates)

    def test_domain_metadata_none_for_non_multicultural(self) -> None:
        entry = _make_entry(
            {"cs_causality": 0.2, "cs_consistency": 0.2, "cs_reaction": 0.2, "cs_desire": 0.2},
        )
        candidates = select_candidates([entry])
        assert all(c.domain_metadata is None for c in candidates)

    def test_coherence_produces_3_per_dialogue(self) -> None:
        entry = _make_entry(
            {
                "co_topic_coherence": 0.8,
                "co_logical_consistency": 0.9,
                "co_discourse_structure": 0.7,
                "co_temporal_causal_coherence": 0.6,
                "co_mutual_grounding": 0.5,
                "co_overall_coherence_score": 0.7,
            },
            domain=DomainName.COHERENCE,
        )
        candidates = select_candidates([entry])
        assert len(candidates) == 3
        # 1 global + 2 dimension-targeted
        global_c = [c for c in candidates if c.variant_type != VariantType.DIMENSION_TARGETED]
        targeted_c = [c for c in candidates if c.variant_type == VariantType.DIMENSION_TARGETED]
        assert len(global_c) == 1
        assert len(targeted_c) == 2
        assert "co_topic_coherence" in candidates[0].characterizing_scores
        assert "co_logical_consistency" in candidates[0].characterizing_scores

    def test_commonsense_produces_4_per_dialogue(self) -> None:
        entry = _make_entry(
            {"cs_causality": 0.1, "cs_consistency": 0.1, "cs_reaction": 0.1, "cs_desire": 0.1},
        )
        candidates = select_candidates([entry])
        assert len(candidates) == 4
        assert all(c.variant_type == VariantType.DIMENSION_TARGETED for c in candidates)

    def test_commonsense_covers_all_4_dims(self) -> None:
        entry = _make_entry(
            {"cs_causality": 0.2, "cs_consistency": 0.2, "cs_reaction": 0.2, "cs_desire": 0.2},
        )
        candidates = select_candidates([entry])
        dims_covered = {c.target_dimensions[0] for c in candidates}
        assert dims_covered == {"cs_causality", "cs_consistency", "cs_reaction", "cs_desire"}

    def test_commonsense_high_score_negative_direction(self) -> None:
        entry = _make_entry(
            {"cs_causality": 0.9, "cs_consistency": 0.9, "cs_reaction": 0.9, "cs_desire": 0.9},
        )
        candidates = select_candidates([entry])
        assert len(candidates) == 4
        assert all(c.variant_type == VariantType.DIMENSION_TARGETED for c in candidates)
        assert all(
            c.contrastive_direction == ContrastiveDirection.NEGATIVE
            for c in candidates
        )

    def test_non_cs_variant_structure(self) -> None:
        """Non-commonsense: 1 global + 2 dimension-targeted, each with single dim."""
        entry = _make_entry(
            {"co_topic_coherence": 0.2, "co_logical_consistency": 0.2},
            domain=DomainName.COHERENCE,
        )
        candidates = select_candidates([entry])
        assert len(candidates) == 3
        assert candidates[0].variant_type == VariantType.GLOBAL_IMPROVE
        assert candidates[0].target_dimensions == ["co_topic_coherence", "co_logical_consistency"]
        assert candidates[1].target_dimensions == ["co_topic_coherence"]
        assert candidates[2].target_dimensions == ["co_logical_consistency"]


# ── Commonsense multi-variant ────────────────────────────────────────────


class TestCommonsenseMultiVariant:
    """Commonsense entries produce 4 DIMENSION_TARGETED candidates, one per dim."""

    CS_SCORES = {"cs_causality": 0.2, "cs_consistency": 0.2, "cs_reaction": 0.2, "cs_desire": 0.2}
    CHAR_DIMS = ["cs_causality", "cs_consistency", "cs_reaction", "cs_desire"]

    def test_all_4_dims_covered(self) -> None:
        entry = _make_entry(self.CS_SCORES)
        candidates = _classify_commonsense_candidates(entry, self.CHAR_DIMS)
        assert len(candidates) == 4
        dims = [c.target_dimensions[0] for c in candidates]
        assert set(dims) == set(self.CHAR_DIMS)

    def test_each_candidate_single_dim(self) -> None:
        entry = _make_entry(self.CS_SCORES)
        candidates = _classify_commonsense_candidates(entry, self.CHAR_DIMS)
        for c in candidates:
            assert len(c.target_dimensions) == 1
            assert c.variant_type == VariantType.DIMENSION_TARGETED

    def test_low_tier_positive_direction(self) -> None:
        entry = _make_entry(self.CS_SCORES)
        candidates = _classify_commonsense_candidates(entry, self.CHAR_DIMS)
        assert all(
            c.contrastive_direction == ContrastiveDirection.POSITIVE
            for c in candidates
        )

    def test_high_tier_negative_direction(self) -> None:
        high_scores = {d: 0.9 for d in self.CS_SCORES}
        entry = _make_entry(high_scores)
        candidates = _classify_commonsense_candidates(entry, self.CHAR_DIMS)
        assert all(
            c.contrastive_direction == ContrastiveDirection.NEGATIVE
            for c in candidates
        )

    def test_missing_scores_returns_empty(self) -> None:
        entry = _make_entry({"cs_coherence": 0.5, "cs_empathy": 0.6})
        candidates = _classify_commonsense_candidates(entry, self.CHAR_DIMS)
        assert candidates == []

    def test_gold_relation_preserved_in_metadata(self) -> None:
        """Gold relation stays in domain_metadata as reference, no longer affects targeting."""
        entry = _make_entry(
            self.CS_SCORES,
            domain_metadata={"gold_relation": "xAttr"},
        )
        candidates = _classify_commonsense_candidates(entry, self.CHAR_DIMS)
        # Still 4 candidates covering all dims
        assert len(candidates) == 4
        dims = {c.target_dimensions[0] for c in candidates}
        assert dims == set(self.CHAR_DIMS)
        # Metadata preserved
        assert all(c.domain_metadata == {"gold_relation": "xAttr"} for c in candidates)

    def test_multiple_dialogues_correct_count(self) -> None:
        entries = [
            _make_entry(self.CS_SCORES, dialogue_id=f"dlg-{i:03d}")
            for i in range(10)
        ]
        candidates = select_candidates(entries)
        # 10 dialogues × 4 variants each = 40
        assert len(candidates) == 40
