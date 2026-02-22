"""Tests for Stage 2 pair construction logic."""

from __future__ import annotations

import re

import pytest

from scripts.models import (
    ContrastiveDirection,
    Difficulty,
    DomainName,
    Message,
    PairLabel,
    Split,
    Stage1Entry,
    Stage2Variant,
    VariantType,
    EvalResult,
)
from scripts.stage_2.pairs import (
    EASY_THRESHOLD,
    MEDIUM_THRESHOLD,
    _classify_difficulty,
    _compute_margin,
    _validate_score_ordering,
    build_pairs,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


CHAR_DIMS = ["cs_causality", "cs_consistency"]


def _make_original(dialogue_id: str = "dlg-001") -> Stage1Entry:
    return Stage1Entry(
        dialogue_id=dialogue_id,
        source_id="src-001",
        domain=DomainName.COMMONSENSE,
        split=Split.TRAIN,
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I am well."),
        ],
        scores={"cs_causality": 0.3, "cs_consistency": 0.3},
    )


def _make_variant(
    candidate_id: str = "dlg-001",
    variant_id: str = "var-001",
    direction: ContrastiveDirection = ContrastiveDirection.POSITIVE,
    variant_type: VariantType = VariantType.GLOBAL_IMPROVE,
) -> Stage2Variant:
    return Stage2Variant(
        candidate_id=candidate_id,
        variant_id=variant_id,
        variant_type=variant_type,
        direction=direction,
        prefix_messages=[],
        variant_messages=[
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I am doing great, thanks!"),
        ],
        generation_model="gpt-5.1-batch",
    )


def _make_result(
    variant_id: str = "var-001",
    original_scores: dict[str, float] | None = None,
    variant_scores: dict[str, float] | None = None,
    label: PairLabel = PairLabel.GLOBAL_PASS,
) -> EvalResult:
    return EvalResult(
        variant_id=variant_id,
        original_scores=original_scores or {"cs_causality": 0.3, "cs_consistency": 0.3},
        variant_scores=variant_scores or {"cs_causality": 0.7, "cs_consistency": 0.7},
        label=label,
    )


# ── Difficulty classification ─────────────────────────────────────────────


class TestDifficultyClassification:
    """Test difficulty bins: easy >= 0.30, medium >= 0.15, hard < 0.15."""

    def test_easy(self) -> None:
        assert _classify_difficulty(0.35) == Difficulty.EASY
        assert _classify_difficulty(0.50) == Difficulty.EASY

    def test_easy_boundary(self) -> None:
        assert _classify_difficulty(EASY_THRESHOLD) == Difficulty.EASY

    def test_medium(self) -> None:
        assert _classify_difficulty(0.20) == Difficulty.MEDIUM
        assert _classify_difficulty(0.25) == Difficulty.MEDIUM

    def test_medium_boundary(self) -> None:
        assert _classify_difficulty(MEDIUM_THRESHOLD) == Difficulty.MEDIUM

    def test_hard(self) -> None:
        assert _classify_difficulty(0.05) == Difficulty.HARD
        assert _classify_difficulty(0.10) == Difficulty.HARD

    def test_hard_boundary(self) -> None:
        assert _classify_difficulty(MEDIUM_THRESHOLD - 0.001) == Difficulty.HARD

    def test_zero_margin(self) -> None:
        assert _classify_difficulty(0.0) == Difficulty.HARD


# ── Margin computation ────────────────────────────────────────────────────


class TestComputeMargin:
    """Test margin = abs(mean(variant) - mean(original)) on char dims."""

    def test_basic_margin(self) -> None:
        original = {"cs_causality": 0.3, "cs_consistency": 0.3}
        variant = {"cs_causality": 0.7, "cs_consistency": 0.7}
        margin = _compute_margin(original, variant, CHAR_DIMS)
        assert margin == pytest.approx(0.4)

    def test_negative_direction_same_margin(self) -> None:
        """Margin is absolute, same regardless of direction."""
        original = {"cs_causality": 0.7, "cs_consistency": 0.7}
        variant = {"cs_causality": 0.3, "cs_consistency": 0.3}
        margin = _compute_margin(original, variant, CHAR_DIMS)
        assert margin == pytest.approx(0.4)

    def test_zero_margin(self) -> None:
        original = {"cs_causality": 0.5, "cs_consistency": 0.5}
        variant = {"cs_causality": 0.5, "cs_consistency": 0.5}
        margin = _compute_margin(original, variant, CHAR_DIMS)
        assert margin == pytest.approx(0.0)

    def test_missing_dims_zero_margin(self) -> None:
        original = {"other": 0.5}
        variant = {"other": 0.8}
        margin = _compute_margin(original, variant, CHAR_DIMS)
        assert margin == pytest.approx(0.0)


# ── Score ordering validation ─────────────────────────────────────────────


class TestValidateScoreOrdering:
    """Chosen scores must be higher than rejected on char dims."""

    def test_valid_ordering(self) -> None:
        chosen = {"cs_causality": 0.7, "cs_consistency": 0.7}
        rejected = {"cs_causality": 0.3, "cs_consistency": 0.3}
        assert _validate_score_ordering(chosen, rejected, CHAR_DIMS) is True

    def test_invalid_ordering(self) -> None:
        chosen = {"cs_causality": 0.3, "cs_consistency": 0.3}
        rejected = {"cs_causality": 0.7, "cs_consistency": 0.7}
        assert _validate_score_ordering(chosen, rejected, CHAR_DIMS) is False

    def test_equal_ordering_fails(self) -> None:
        chosen = {"cs_causality": 0.5, "cs_consistency": 0.5}
        rejected = {"cs_causality": 0.5, "cs_consistency": 0.5}
        assert _validate_score_ordering(chosen, rejected, CHAR_DIMS) is False


# ── build_pairs integration ──────────────────────────────────────────────


class TestBuildPairs:
    """Integration tests for the full build_pairs function."""

    def test_only_non_rejected_variants_included(self) -> None:
        """All 5 pass labels produce pairs, REJECT does not."""
        original = _make_original()
        originals = {original.dialogue_id: original}

        pass_labels = [
            PairLabel.GLOBAL_PASS,
            PairLabel.TARGET_PASS,
            PairLabel.TARGET_COARSE_PASS,
            PairLabel.GLOBAL_FLIP_PASS,
            PairLabel.TARGET_FLIP_PASS,
        ]
        variants = []
        results = []
        for i, lbl in enumerate(pass_labels):
            vid = f"var-{lbl.value}"
            # flip labels need reversed scores for ordering validation
            if lbl in {PairLabel.GLOBAL_FLIP_PASS, PairLabel.TARGET_FLIP_PASS}:
                variants.append(_make_variant(
                    variant_id=vid, direction=ContrastiveDirection.POSITIVE,
                ))
                results.append(_make_result(
                    variant_id=vid, label=lbl,
                    original_scores={"cs_causality": 0.7, "cs_consistency": 0.7},
                    variant_scores={"cs_causality": 0.3, "cs_consistency": 0.3},
                ))
            else:
                variants.append(_make_variant(variant_id=vid))
                results.append(_make_result(variant_id=vid, label=lbl))

        # add a reject
        variants.append(_make_variant(variant_id="var-reject"))
        results.append(_make_result(variant_id="var-reject", label=PairLabel.REJECT))

        pairs = build_pairs(variants, results, originals)
        pair_ids = {p.pair_id for p in pairs}
        assert len(pairs) == 5
        assert all(re.match(r"^S2D-\d{6}$", pid) for pid in pair_ids)

    def test_positive_direction_variant_is_chosen(self) -> None:
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant(direction=ContrastiveDirection.POSITIVE)
        result = _make_result(
            original_scores={"cs_causality": 0.3, "cs_consistency": 0.3},
            variant_scores={"cs_causality": 0.7, "cs_consistency": 0.7},
        )

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1

        pair = pairs[0]
        assert pair.chosen == variant.variant_messages
        assert pair.metadata.contrast.direction == ContrastiveDirection.POSITIVE
        assert pair.evaluation.chosen_score_source == "stage_2"
        assert pair.evaluation.rejected_score_source == "stage_1"

    def test_negative_direction_original_is_chosen(self) -> None:
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant(
            direction=ContrastiveDirection.NEGATIVE,
            variant_type=VariantType.GLOBAL_DEGRADE,
        )
        result = _make_result(
            original_scores={"cs_causality": 0.7, "cs_consistency": 0.7},
            variant_scores={"cs_causality": 0.3, "cs_consistency": 0.3},
            label=PairLabel.GLOBAL_PASS,
        )

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1

        pair = pairs[0]
        assert pair.rejected == variant.variant_messages
        assert pair.metadata.contrast.direction == ContrastiveDirection.NEGATIVE
        assert pair.evaluation.chosen_score_source == "stage_1"
        assert pair.evaluation.rejected_score_source == "stage_2"

    def test_difficulty_assigned(self) -> None:
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant()
        result = _make_result(
            original_scores={"cs_causality": 0.3, "cs_consistency": 0.3},
            variant_scores={"cs_causality": 0.7, "cs_consistency": 0.7},
        )

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1
        # margin = 0.4, which is >= 0.30 -> easy
        assert pairs[0].metadata.difficulty == Difficulty.EASY

    def test_hard_difficulty(self) -> None:
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant()
        result = _make_result(
            original_scores={"cs_causality": 0.45, "cs_consistency": 0.45},
            variant_scores={"cs_causality": 0.55, "cs_consistency": 0.55},
        )

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1
        # margin = 0.1, which is < 0.15 -> hard
        assert pairs[0].metadata.difficulty == Difficulty.HARD

    def test_output_schema_has_required_fields(self) -> None:
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant()
        result = _make_result()

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1

        pair = pairs[0]
        assert pair.pair_id is not None
        assert pair.source_dialogue_id == "dlg-001"
        assert pair.metadata.domain == DomainName.COMMONSENSE
        assert pair.metadata.split == Split.TRAIN
        assert pair.evaluation.method == "lm_judge"
        assert isinstance(pair.messages, list)
        assert isinstance(pair.chosen, list)
        assert isinstance(pair.rejected, list)

    def test_evaluation_carries_scores(self) -> None:
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant()
        orig_scores = {"cs_causality": 0.3, "cs_consistency": 0.3}
        var_scores = {"cs_causality": 0.7, "cs_consistency": 0.7}
        result = _make_result(
            original_scores=orig_scores,
            variant_scores=var_scores,
        )

        pairs = build_pairs([variant], [result], originals)
        pair = pairs[0]
        assert pair.evaluation.stage_1_scores == orig_scores
        assert pair.evaluation.stage_2_scores == var_scores
        # positive direction: variant is chosen (stage_2), original is rejected (stage_1)
        assert pair.evaluation.chosen_score_source == "stage_2"
        assert pair.evaluation.rejected_score_source == "stage_1"

    def test_no_variants_produces_empty(self) -> None:
        pairs = build_pairs([], [], {})
        assert pairs == []

    def test_missing_original_skipped(self) -> None:
        variant = _make_variant(candidate_id="nonexistent")
        result = _make_result(variant_id="var-001", label=PairLabel.GLOBAL_PASS)
        pairs = build_pairs([variant], [result], {})
        assert pairs == []

    def test_bad_ordering_skipped(self) -> None:
        """If chosen scores < rejected scores, pair is skipped."""
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant(direction=ContrastiveDirection.POSITIVE)
        # Variant scores LOWER than original -> bad ordering for positive direction
        result = _make_result(
            original_scores={"cs_causality": 0.7, "cs_consistency": 0.7},
            variant_scores={"cs_causality": 0.3, "cs_consistency": 0.3},
            label=PairLabel.GLOBAL_PASS,
        )

        pairs = build_pairs([variant], [result], originals)
        assert pairs == []

    def test_metadata_turn_count(self) -> None:
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant()
        result = _make_result()

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1
        assert pairs[0].metadata.turn_count == 2

    def test_metadata_contrast_scope(self) -> None:
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant(variant_type=VariantType.GLOBAL_IMPROVE)
        result = _make_result()

        pairs = build_pairs([variant], [result], originals)
        assert pairs[0].metadata.contrast.scope == "global"

    def test_target_dimensions_used_for_margin(self) -> None:
        """When variant has target_dimensions, margin uses those dims only."""
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant(
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        variant.target_dimensions = ["cs_causality"]

        # cs_causality: 0.3 -> 0.8 = 0.5 margin on target (easy)
        # cs_consistency: 0.3 -> 0.3 = 0 margin (would dilute to 0.25 if global)
        result = _make_result(
            original_scores={"cs_causality": 0.3, "cs_consistency": 0.3},
            variant_scores={"cs_causality": 0.8, "cs_consistency": 0.3},
            label=PairLabel.TARGET_PASS,
        )

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1
        # Margin should be 0.5 (easy) not 0.25 (medium)
        assert pairs[0].metadata.difficulty == Difficulty.EASY
        assert pairs[0].metadata.dimensions.target == ["cs_causality"]

    def test_global_variant_has_empty_target_dims(self) -> None:
        """Global variant has target=[] in dimensions, decision=char_dims."""
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant()
        assert variant.target_dimensions == []

        result = _make_result(
            original_scores={"cs_causality": 0.3, "cs_consistency": 0.3},
            variant_scores={"cs_causality": 0.7, "cs_consistency": 0.7},
        )

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1
        assert pairs[0].metadata.dimensions.target == []
        assert set(pairs[0].metadata.dimensions.decision) == {
            "cs_causality", "cs_consistency", "cs_reaction", "cs_desire",
        }

    def test_score_ordering_uses_target_dims(self) -> None:
        """Score ordering validation uses target_dims, not all char_dims."""
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant(
            direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        variant.target_dimensions = ["cs_causality"]

        result = _make_result(
            original_scores={"cs_causality": 0.3, "cs_consistency": 0.3},
            variant_scores={"cs_causality": 0.7, "cs_consistency": 0.2},
            label=PairLabel.TARGET_PASS,
        )

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1

    # ── Flip-pass tests ──────────────────────────────────────────────────

    def test_global_flip_pass_swaps_chosen_rejected(self) -> None:
        """GLOBAL_FLIP_PASS with POSITIVE direction → original is chosen."""
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant(direction=ContrastiveDirection.POSITIVE)
        result = _make_result(
            label=PairLabel.GLOBAL_FLIP_PASS,
            original_scores={"cs_causality": 0.7, "cs_consistency": 0.7},
            variant_scores={"cs_causality": 0.3, "cs_consistency": 0.3},
        )

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1
        pair = pairs[0]
        # Flipped: original is chosen (higher scores), variant is rejected
        assert pair.rejected == variant.variant_messages

    def test_target_flip_pass_swaps_chosen_rejected(self) -> None:
        """TARGET_FLIP_PASS with POSITIVE direction → original is chosen."""
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant(
            direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        variant.target_dimensions = ["cs_causality"]
        result = _make_result(
            label=PairLabel.TARGET_FLIP_PASS,
            original_scores={"cs_causality": 0.7, "cs_consistency": 0.5},
            variant_scores={"cs_causality": 0.3, "cs_consistency": 0.5},
        )

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.rejected == variant.variant_messages

    def test_flip_pass_stores_original_direction(self) -> None:
        """Flipped pair's contrast.direction is the ORIGINAL intent, not effective."""
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant(direction=ContrastiveDirection.POSITIVE)
        result = _make_result(
            label=PairLabel.GLOBAL_FLIP_PASS,
            original_scores={"cs_causality": 0.7, "cs_consistency": 0.7},
            variant_scores={"cs_causality": 0.3, "cs_consistency": 0.3},
        )

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1
        pair = pairs[0]
        # direction is original intent, NOT effective
        assert pair.metadata.contrast.direction == ContrastiveDirection.POSITIVE
        assert pair.metadata.contrast.intent_followed is False
        assert pair.metadata.contrast.decision == PairLabel.GLOBAL_FLIP_PASS

    def test_intent_followed_true_for_normal_passes(self) -> None:
        """Non-flip labels have intent_followed=True."""
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant()
        result = _make_result(label=PairLabel.GLOBAL_PASS)

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1
        assert pairs[0].metadata.contrast.intent_followed is True
        assert pairs[0].metadata.contrast.decision == PairLabel.GLOBAL_PASS

    def test_intent_followed_false_for_target_flip(self) -> None:
        """TARGET_FLIP_PASS has intent_followed=False."""
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant(
            direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        variant.target_dimensions = ["cs_causality"]
        result = _make_result(
            label=PairLabel.TARGET_FLIP_PASS,
            original_scores={"cs_causality": 0.7, "cs_consistency": 0.5},
            variant_scores={"cs_causality": 0.3, "cs_consistency": 0.5},
        )

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1
        assert pairs[0].metadata.contrast.intent_followed is False
        assert pairs[0].metadata.contrast.scope == "target"

    def test_dimensions_domain_has_all_dims(self) -> None:
        """dimensions.domain lists all domain dims, not just characterizing."""
        original = _make_original()
        originals = {original.dialogue_id: original}

        variant = _make_variant()
        result = _make_result()

        pairs = build_pairs([variant], [result], originals)
        assert len(pairs) == 1
        # commonsense has 6 dims total
        assert len(pairs[0].metadata.dimensions.domain) == 6
        assert "cs_coherence" in pairs[0].metadata.dimensions.domain
        assert "cs_empathy" in pairs[0].metadata.dimensions.domain
