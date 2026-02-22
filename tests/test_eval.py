"""tests for stage 2 evaluation logic."""

from __future__ import annotations


from scripts.config import DOMAINS
from scripts.models import (
    ContrastiveDirection,
    DomainName,
    Message,
    PairLabel,
    VariantType,
)
from scripts.stage_2.prompts import build_eval_prompt
from scripts.stage_2.eval import (
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_STABILITY_THRESHOLD,
    classify_variant,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

CHAR_DIMS = ["cs_causality", "cs_consistency"]
CS_ALL_DIMS = ["cs_causality", "cs_consistency", "cs_reaction", "cs_desire"]


# ── Global variant classification ────────────────────────────────────────


class TestClassifyGlobalVariants:
    """Global variants: classify by avg delta across all char dims."""

    def test_positive_intended_moved_up_global_pass(self) -> None:
        original = {"cs_causality": 0.3, "cs_consistency": 0.3}
        variant = {"cs_causality": 0.7, "cs_consistency": 0.7}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CHAR_DIMS, DEFAULT_MARGIN_THRESHOLD, VariantType.GLOBAL_IMPROVE,
        )
        assert label == PairLabel.GLOBAL_PASS

    def test_negative_intended_moved_down_global_pass(self) -> None:
        original = {"cs_causality": 0.8, "cs_consistency": 0.8}
        variant = {"cs_causality": 0.4, "cs_consistency": 0.4}

        label = classify_variant(
            original, variant, ContrastiveDirection.NEGATIVE,
            CHAR_DIMS, DEFAULT_MARGIN_THRESHOLD, VariantType.GLOBAL_DEGRADE,
        )
        assert label == PairLabel.GLOBAL_PASS

    def test_positive_intended_moved_down_global_flip_pass(self) -> None:
        """Asked for improvement but got degradation — flip pass."""
        original = {"cs_causality": 0.7, "cs_consistency": 0.7}
        variant = {"cs_causality": 0.3, "cs_consistency": 0.3}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CHAR_DIMS, DEFAULT_MARGIN_THRESHOLD, VariantType.GLOBAL_IMPROVE,
        )
        assert label == PairLabel.GLOBAL_FLIP_PASS

    def test_negative_intended_moved_up_global_flip_pass(self) -> None:
        """Asked for degradation but got improvement — flip pass."""
        original = {"cs_causality": 0.3, "cs_consistency": 0.3}
        variant = {"cs_causality": 0.7, "cs_consistency": 0.7}

        label = classify_variant(
            original, variant, ContrastiveDirection.NEGATIVE,
            CHAR_DIMS, DEFAULT_MARGIN_THRESHOLD, VariantType.GLOBAL_DEGRADE,
        )
        assert label == PairLabel.GLOBAL_FLIP_PASS

    def test_barely_moved_reject(self) -> None:
        """Delta within ±margin_threshold → reject."""
        original = {"cs_causality": 0.50, "cs_consistency": 0.50}
        variant = {"cs_causality": 0.55, "cs_consistency": 0.55}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CHAR_DIMS, DEFAULT_MARGIN_THRESHOLD, VariantType.GLOBAL_IMPROVE,
        )
        assert label == PairLabel.REJECT

    def test_missing_scores_reject(self) -> None:
        original = {"other_dim": 0.5}
        variant = {"other_dim": 0.8}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CHAR_DIMS, DEFAULT_MARGIN_THRESHOLD, VariantType.GLOBAL_IMPROVE,
        )
        assert label == PairLabel.REJECT

    def test_margin_exactly_at_threshold_reject(self) -> None:
        """Delta == threshold → reject (strictly > required)."""
        original = {"cs_causality": 0.30, "cs_consistency": 0.30}
        variant = {"cs_causality": 0.50, "cs_consistency": 0.50}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CHAR_DIMS, margin_threshold=0.2,
            variant_type=VariantType.GLOBAL_IMPROVE,
        )
        assert label == PairLabel.REJECT

    def test_partial_dims_still_evaluated(self) -> None:
        """If only some characterizing dims present, use what we have."""
        original = {"cs_causality": 0.3}
        variant = {"cs_causality": 0.8}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CHAR_DIMS, DEFAULT_MARGIN_THRESHOLD, VariantType.GLOBAL_IMPROVE,
        )
        assert label == PairLabel.GLOBAL_PASS


# ── Targeted variant classification ──────────────────────────────────────


class TestClassifyTargetedVariants:
    """Targeted variants: classify by target delta + non-target stability."""

    def test_target_moved_nontargets_stable_target_pass(self) -> None:
        original = {"cs_causality": 0.3, "cs_consistency": 0.5, "cs_reaction": 0.5, "cs_desire": 0.5}
        variant = {"cs_causality": 0.7, "cs_consistency": 0.5, "cs_reaction": 0.5, "cs_desire": 0.5}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CS_ALL_DIMS, DEFAULT_MARGIN_THRESHOLD,
            VariantType.DIMENSION_TARGETED, target_dims=["cs_causality"],
        )
        assert label == PairLabel.TARGET_PASS

    def test_target_moved_nontargets_drifted_target_coarse_pass(self) -> None:
        """Non-target drift > stability_threshold → coarse pass (not reject)."""
        original = {"cs_causality": 0.3, "cs_consistency": 0.5, "cs_reaction": 0.5, "cs_desire": 0.5}
        variant = {"cs_causality": 0.7, "cs_consistency": 0.1, "cs_reaction": 0.5, "cs_desire": 0.5}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CS_ALL_DIMS, DEFAULT_MARGIN_THRESHOLD,
            VariantType.DIMENSION_TARGETED, target_dims=["cs_causality"],
        )
        assert label == PairLabel.TARGET_COARSE_PASS

    def test_target_moved_opposite_target_flip_pass(self) -> None:
        """Target moved opposite to intended → flip pass."""
        original = {"cs_causality": 0.7, "cs_consistency": 0.5, "cs_reaction": 0.5, "cs_desire": 0.5}
        variant = {"cs_causality": 0.3, "cs_consistency": 0.5, "cs_reaction": 0.5, "cs_desire": 0.5}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CS_ALL_DIMS, DEFAULT_MARGIN_THRESHOLD,
            VariantType.DIMENSION_TARGETED, target_dims=["cs_causality"],
        )
        assert label == PairLabel.TARGET_FLIP_PASS

    def test_target_barely_moved_reject(self) -> None:
        original = {"cs_causality": 0.3, "cs_consistency": 0.5, "cs_reaction": 0.5, "cs_desire": 0.5}
        variant = {"cs_causality": 0.32, "cs_consistency": 0.5, "cs_reaction": 0.5, "cs_desire": 0.5}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CS_ALL_DIMS, DEFAULT_MARGIN_THRESHOLD,
            VariantType.DIMENSION_TARGETED, target_dims=["cs_causality"],
        )
        assert label == PairLabel.REJECT

    def test_missing_target_scores_reject(self) -> None:
        original = {"cs_consistency": 0.5}
        variant = {"cs_consistency": 0.5}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CS_ALL_DIMS, DEFAULT_MARGIN_THRESHOLD,
            VariantType.DIMENSION_TARGETED, target_dims=["cs_causality"],
        )
        assert label == PairLabel.REJECT

    def test_negative_direction_target_degrades_target_pass(self) -> None:
        original = {"cs_causality": 0.7, "cs_consistency": 0.5, "cs_reaction": 0.5, "cs_desire": 0.5}
        variant = {"cs_causality": 0.3, "cs_consistency": 0.5, "cs_reaction": 0.5, "cs_desire": 0.5}

        label = classify_variant(
            original, variant, ContrastiveDirection.NEGATIVE,
            CS_ALL_DIMS, DEFAULT_MARGIN_THRESHOLD,
            VariantType.DIMENSION_TARGETED, target_dims=["cs_causality"],
        )
        assert label == PairLabel.TARGET_PASS

    def test_stability_boundary_exactly_at_threshold_is_target_pass(self) -> None:
        """Non-target drift exactly at stability_threshold → TARGET_PASS (uses > not >=)."""
        original = {"cs_causality": 0.3, "cs_consistency": 0.5, "cs_reaction": 0.5, "cs_desire": 0.5}
        variant = {"cs_causality": 0.7, "cs_consistency": 0.7, "cs_reaction": 0.5, "cs_desire": 0.5}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CS_ALL_DIMS, DEFAULT_MARGIN_THRESHOLD,
            VariantType.DIMENSION_TARGETED, target_dims=["cs_causality"],
            stability_threshold=0.20,
        )
        assert label == PairLabel.TARGET_PASS

    def test_stability_exceeded_is_target_coarse_pass(self) -> None:
        """Non-target drift > stability_threshold → TARGET_COARSE_PASS."""
        original = {"cs_causality": 0.3, "cs_consistency": 0.5, "cs_reaction": 0.5, "cs_desire": 0.5}
        variant = {"cs_causality": 0.7, "cs_consistency": 0.71, "cs_reaction": 0.5, "cs_desire": 0.5}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CS_ALL_DIMS, DEFAULT_MARGIN_THRESHOLD,
            VariantType.DIMENSION_TARGETED, target_dims=["cs_causality"],
            stability_threshold=0.20,
        )
        assert label == PairLabel.TARGET_COARSE_PASS

    def test_no_target_dims_rejects(self) -> None:
        """Targeted variant with empty target_dims → reject."""
        original = {"cs_causality": 0.3, "cs_consistency": 0.3}
        variant = {"cs_causality": 0.7, "cs_consistency": 0.7}

        label = classify_variant(
            original, variant, ContrastiveDirection.POSITIVE,
            CHAR_DIMS, DEFAULT_MARGIN_THRESHOLD,
            VariantType.DIMENSION_TARGETED, target_dims=[],
        )
        assert label == PairLabel.REJECT


# ── Verification prompt cultural context ─────────────────────────────────


class TestEvalPromptCulturalContext:
    """test that multicultural eval prompts contain cultural context."""

    def test_multicultural_eval_has_cultural_context(self) -> None:
        metadata = {
            "country_1": "Japan",
            "country_2": "United States",
            "demographics_1": "Age 30, male",
            "demographics_2": "Age 25, female",
            "statement_original": "Respect for elders",
            "statement_cultural": "Cultural respect",
            "situation": "Business meeting",
            "cultural_reasoning_1": "Japanese reasoning",
            "cultural_reasoning_2": "American reasoning",
            "arousal_reasoning": "Moderate tension",
            "arousal_score": "3",
            "social_norms_1": "Bow when greeting",
            "social_norms_2": "Handshake",
            "prejudices_1": "Stereotypes about US",
            "prejudices_2": "Stereotypes about Japan",
        }
        config = DOMAINS[DomainName.MULTICULTURAL]
        prompt = build_eval_prompt(
            prefix_messages=[Message(role="user", content="Hello")],
            variant_continuation=[Message(role="assistant", content="Hey")],
            config=config,
            domain_metadata=metadata,
        )
        assert "CULTURAL CONTEXT" in prompt
        assert "Japan" in prompt
        assert "United States" in prompt
        assert "Respect for elders" in prompt

    def test_non_multicultural_eval_no_cultural_context(self) -> None:
        config = DOMAINS[DomainName.COMMONSENSE]
        prompt = build_eval_prompt(
            prefix_messages=[Message(role="user", content="Hello")],
            variant_continuation=[Message(role="assistant", content="Hey")],
            config=config,
        )
        assert "CULTURAL CONTEXT" not in prompt

    def test_eval_prompt_scores_variant_only(self) -> None:
        """eval prompt should score a single continuation, not two."""
        config = DOMAINS[DomainName.COMMONSENSE]
        prompt = build_eval_prompt(
            prefix_messages=[Message(role="user", content="Hello")],
            variant_continuation=[Message(role="assistant", content="Test")],
            config=config,
        )
        assert "CONTINUATION:" in prompt
        assert "CONTINUATION A" not in prompt
        assert "CONTINUATION B" not in prompt
        assert '"scores":' in prompt
        assert '"scores_a"' not in prompt
