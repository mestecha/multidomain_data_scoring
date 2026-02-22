"""Tests for Stage 2 generation logic."""

from __future__ import annotations


from scripts.config import DOMAINS
from scripts.models import (
    ContrastiveDirection,
    DomainName,
    Message,
    Stage2Candidate,
    VariantType,
)
from scripts.stage_2.generate import build_generation_entries
from scripts.stage_2.prompts import build_generation_prompt


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_candidate(
    variant_type: VariantType = VariantType.GLOBAL_IMPROVE,
    direction: ContrastiveDirection = ContrastiveDirection.POSITIVE,
    domain: DomainName = DomainName.COMMONSENSE,
    dialogue_id: str = "dlg-001",
    domain_metadata: dict[str, str] | None = None,
) -> Stage2Candidate:
    """Helper to build a minimal Stage2Candidate."""
    config = DOMAINS[domain]
    char_scores = (
        {"cs_causality": 0.3, "cs_consistency": 0.2}
        if domain == DomainName.COMMONSENSE
        else {dim: 0.3 for dim in config.characterizing_dims}
    )
    return Stage2Candidate(
        dialogue_id=dialogue_id,
        domain=domain,
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I am doing well."),
            Message(role="user", content="Good to hear."),
        ],
        characterizing_scores=char_scores,
        target_dimensions=config.characterizing_dims,
        contrastive_direction=direction,
        variant_type=variant_type,
        domain_metadata=domain_metadata,
    )


# ── Prompt content tests ─────────────────────────────────────────────────


class TestGenerationPrompt:
    """Test that generation prompts include expected content."""

    def test_prompt_includes_turn_count(self) -> None:
        candidate = _make_candidate()
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(candidate, turn_count=3, config=config)

        assert "3" in prompt
        assert "turn" in prompt.lower()

    def test_prompt_includes_domain(self) -> None:
        candidate = _make_candidate()
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "commonsense" in prompt.lower()

    def test_prompt_adapts_to_global_improve(self) -> None:
        candidate = _make_candidate(variant_type=VariantType.GLOBAL_IMPROVE)
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(candidate, turn_count=3, config=config)

        assert "stronger" in prompt.lower()
        assert "improve" in prompt.lower()

    def test_prompt_adapts_to_global_degrade(self) -> None:
        candidate = _make_candidate(
            variant_type=VariantType.GLOBAL_DEGRADE,
            direction=ContrastiveDirection.NEGATIVE,
        )
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(candidate, turn_count=3, config=config)

        assert "weaker" in prompt.lower()
        assert "degrade" in prompt.lower()

    def test_prompt_adapts_to_dimension_targeted(self) -> None:
        candidate = _make_candidate(variant_type=VariantType.DIMENSION_TARGETED)
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(candidate, turn_count=3, config=config)

        assert "specifically" in prompt.lower()
        # Target dims should be named
        assert "cs_causality" in prompt
        assert "cs_consistency" in prompt

    def test_prompt_includes_rubric_dimensions(self) -> None:
        candidate = _make_candidate()
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        # Rubric should contain the characterizing dim descriptions
        assert "cs_causality" in prompt
        assert "cs_consistency" in prompt

    def test_prompt_includes_dialogue_content(self) -> None:
        candidate = _make_candidate()
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "Hello" in prompt
        assert "Hi there" in prompt

    def test_multicultural_prompt_contains_cultural_context(self) -> None:
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
        candidate = _make_candidate(
            domain=DomainName.MULTICULTURAL,
            domain_metadata=metadata,
        )
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(
            candidate, turn_count=3, config=config,
            domain_metadata=metadata,
        )

        assert "CULTURAL CONTEXT" in prompt
        assert "Japan" in prompt
        assert "United States" in prompt
        assert "Respect for elders" in prompt
        assert "Business meeting" in prompt
        assert "Bow when greeting" in prompt

    def test_coherence_prompt_has_no_cultural_context(self) -> None:
        candidate = _make_candidate(domain=DomainName.COMMONSENSE)
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(candidate, turn_count=3, config=config)

        assert "CULTURAL CONTEXT" not in prompt

    def test_commonsense_prompt_has_generation_guidance(self) -> None:
        candidate = Stage2Candidate(
            dialogue_id="dlg-cs",
            domain=DomainName.COMMONSENSE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
                Message(role="user", content="What happened?"),
            ],
            characterizing_scores={"cs_causality": 0.2},
            target_dimensions=["cs_causality"],
            contrastive_direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        config = DOMAINS[DomainName.COMMONSENSE]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "GENERATION GUIDANCE" in prompt
        assert "cs_causality" in prompt
        assert "cause-effect" in prompt

    def test_commonsense_degrade_guidance(self) -> None:
        candidate = Stage2Candidate(
            dialogue_id="dlg-cs-neg",
            domain=DomainName.COMMONSENSE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
            characterizing_scores={"cs_reaction": 0.8},
            target_dimensions=["cs_reaction"],
            contrastive_direction=ContrastiveDirection.NEGATIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        config = DOMAINS[DomainName.COMMONSENSE]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "GENERATION GUIDANCE" in prompt
        assert "slightly off" in prompt.lower()

    def test_commonsense_global_no_generation_guidance(self) -> None:
        candidate = _make_candidate(
            domain=DomainName.COMMONSENSE,
            variant_type=VariantType.GLOBAL_IMPROVE,
        )
        config = DOMAINS[DomainName.COMMONSENSE]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "GENERATION GUIDANCE" not in prompt

    def test_non_commonsense_no_generation_guidance(self) -> None:
        candidate = _make_candidate(domain=DomainName.EMPATHY)
        config = DOMAINS[DomainName.EMPATHY]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "GENERATION GUIDANCE" not in prompt

    def test_targeted_prompt_has_non_target_constraint(self) -> None:
        """Targeted prompts list non-target dims with 'do NOT change'."""
        candidate = Stage2Candidate(
            dialogue_id="dlg-nt",
            domain=DomainName.COMMONSENSE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
                Message(role="user", content="What happened?"),
            ],
            characterizing_scores={"cs_causality": 0.3, "cs_consistency": 0.5},
            target_dimensions=["cs_causality"],
            contrastive_direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        config = DOMAINS[DomainName.COMMONSENSE]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "do NOT change" in prompt
        assert "cs_consistency" in prompt
        assert "cs_reaction" in prompt
        assert "cs_desire" in prompt

    def test_global_prompt_has_no_non_target_constraint(self) -> None:
        candidate = _make_candidate(variant_type=VariantType.GLOBAL_IMPROVE)
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "do NOT change" not in prompt

    def test_degrade_targeted_has_subtlety_constraint(self) -> None:
        candidate = Stage2Candidate(
            dialogue_id="dlg-sub",
            domain=DomainName.COMMONSENSE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
            characterizing_scores={"cs_causality": 0.5},
            target_dimensions=["cs_causality"],
            contrastive_direction=ContrastiveDirection.NEGATIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        config = DOMAINS[DomainName.COMMONSENSE]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "subtle" in prompt.lower()
        assert "non-sequiturs" in prompt.lower()

    def test_improve_targeted_has_no_subtlety_constraint(self) -> None:
        candidate = Stage2Candidate(
            dialogue_id="dlg-nosub",
            domain=DomainName.COMMONSENSE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
            characterizing_scores={"cs_causality": 0.3},
            target_dimensions=["cs_causality"],
            contrastive_direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        config = DOMAINS[DomainName.COMMONSENSE]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "non-sequiturs" not in prompt.lower()

    def test_global_degrade_has_subtlety_lines(self) -> None:
        candidate = _make_candidate(
            variant_type=VariantType.GLOBAL_DEGRADE,
            direction=ContrastiveDirection.NEGATIVE,
        )
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "subtle and natural-sounding" in prompt
        assert "non-sequiturs" in prompt.lower()

    def test_targeted_prompt_has_score_context(self) -> None:
        candidate = Stage2Candidate(
            dialogue_id="dlg-sc",
            domain=DomainName.COMMONSENSE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
            characterizing_scores={"cs_causality": 0.25},
            target_dimensions=["cs_causality"],
            contrastive_direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        config = DOMAINS[DomainName.COMMONSENSE]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "current score: 0.25" in prompt
        assert "substantial" in prompt.lower()

    def test_targeted_high_score_gets_subtle_magnitude(self) -> None:
        candidate = Stage2Candidate(
            dialogue_id="dlg-hi",
            domain=DomainName.COMMONSENSE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
            characterizing_scores={"cs_causality": 0.85},
            target_dimensions=["cs_causality"],
            contrastive_direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        config = DOMAINS[DomainName.COMMONSENSE]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "current score: 0.85" in prompt
        assert "refined, subtle improvement" in prompt.lower()

    def test_json_output_format_no_pipe_notation(self) -> None:
        """No pipe notation in any generation template."""
        for vt, direction in [
            (VariantType.GLOBAL_IMPROVE, ContrastiveDirection.POSITIVE),
            (VariantType.GLOBAL_DEGRADE, ContrastiveDirection.NEGATIVE),
            (VariantType.DIMENSION_TARGETED, ContrastiveDirection.POSITIVE),
        ]:
            candidate = _make_candidate(variant_type=vt, direction=direction)
            config = DOMAINS[candidate.domain]
            prompt = build_generation_prompt(candidate, turn_count=1, config=config)

            assert '"user"|"assistant"' not in prompt
            assert '"role": "user"' in prompt
            assert '"role": "assistant"' in prompt

    def test_multicultural_degrade_uses_less_qualifier(self) -> None:
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
        candidate = _make_candidate(
            domain=DomainName.MULTICULTURAL,
            direction=ContrastiveDirection.NEGATIVE,
            variant_type=VariantType.GLOBAL_DEGRADE,
            domain_metadata=metadata,
        )
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(
            candidate, turn_count=3, config=config, domain_metadata=metadata,
        )

        assert "less culturally grounded" in prompt
        assert "more (or less)" not in prompt

    def test_multicultural_improve_uses_more_qualifier(self) -> None:
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
        candidate = _make_candidate(
            domain=DomainName.MULTICULTURAL,
            direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.GLOBAL_IMPROVE,
            domain_metadata=metadata,
        )
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(
            candidate, turn_count=3, config=config, domain_metadata=metadata,
        )

        assert "more culturally grounded" in prompt
        assert "more (or less)" not in prompt

    def test_degrade_low_score_gets_minimal_magnitude(self) -> None:
        """degrade + low score = floor effect → minimal change guidance."""
        candidate = Stage2Candidate(
            dialogue_id="dlg-floor",
            domain=DomainName.COMMONSENSE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
            characterizing_scores={"cs_causality": 0.2},
            target_dimensions=["cs_causality"],
            contrastive_direction=ContrastiveDirection.NEGATIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        config = DOMAINS[DomainName.COMMONSENSE]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "already low" in prompt.lower()
        assert "minimal, subtle change" in prompt.lower()

    def test_degrade_high_score_gets_substantial_magnitude(self) -> None:
        """degrade + high score = lots of headroom → substantial change."""
        candidate = Stage2Candidate(
            dialogue_id="dlg-ceil",
            domain=DomainName.COMMONSENSE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
            characterizing_scores={"cs_causality": 0.85},
            target_dimensions=["cs_causality"],
            contrastive_direction=ContrastiveDirection.NEGATIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        config = DOMAINS[DomainName.COMMONSENSE]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "current score is high" in prompt.lower()
        assert "substantial change" in prompt.lower()

    def test_global_improve_has_score_context(self) -> None:
        candidate = _make_candidate(variant_type=VariantType.GLOBAL_IMPROVE)
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "CURRENT SCORES" in prompt

    def test_global_degrade_has_score_context(self) -> None:
        candidate = _make_candidate(
            variant_type=VariantType.GLOBAL_DEGRADE,
            direction=ContrastiveDirection.NEGATIVE,
        )
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "CURRENT SCORES" in prompt

    def test_global_degrade_has_non_char_constraint(self) -> None:
        """global degrade lists non-characterizing dims to preserve."""
        candidate = _make_candidate(
            domain=DomainName.COHERENCE,
            variant_type=VariantType.GLOBAL_DEGRADE,
            direction=ContrastiveDirection.NEGATIVE,
        )
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "DIMENSIONS NOT LISTED ABOVE" in prompt
        assert "co_discourse_structure" in prompt

    def test_multicultural_cultural_context_before_instructions(self) -> None:
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
        candidate = _make_candidate(
            domain=DomainName.MULTICULTURAL,
            domain_metadata=metadata,
        )
        config = DOMAINS[candidate.domain]
        prompt = build_generation_prompt(
            candidate, turn_count=3, config=config, domain_metadata=metadata,
        )

        cultural_idx = prompt.find("CULTURAL CONTEXT")
        instructions_idx = prompt.find("INSTRUCTIONS:")
        assert cultural_idx < instructions_idx

    def test_coherence_targeted_has_generation_guidance(self) -> None:
        candidate = Stage2Candidate(
            dialogue_id="dlg-co-guid",
            domain=DomainName.COHERENCE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
                Message(role="user", content="What?"),
            ],
            characterizing_scores={"co_topic_coherence": 0.3},
            target_dimensions=["co_topic_coherence"],
            contrastive_direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        config = DOMAINS[DomainName.COHERENCE]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "GENERATION GUIDANCE" in prompt
        assert "co_topic_coherence" in prompt

    def test_empathy_targeted_has_generation_guidance(self) -> None:
        candidate = Stage2Candidate(
            dialogue_id="dlg-em-guid",
            domain=DomainName.EMPATHY,
            messages=[
                Message(role="user", content="I feel sad"),
                Message(role="assistant", content="I understand"),
            ],
            characterizing_scores={"em_emotional_awareness": 0.3},
            target_dimensions=["em_emotional_awareness"],
            contrastive_direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        config = DOMAINS[DomainName.EMPATHY]
        prompt = build_generation_prompt(candidate, turn_count=1, config=config)

        assert "GENERATION GUIDANCE" in prompt
        assert "em_emotional_awareness" in prompt

    def test_different_domains_produce_different_prompts(self) -> None:
        cs_candidate = _make_candidate(domain=DomainName.COMMONSENSE)
        em_candidate = Stage2Candidate(
            dialogue_id="dlg-002",
            domain=DomainName.EMPATHY,
            messages=[
                Message(role="user", content="I feel sad"),
                Message(role="assistant", content="I understand"),
            ],
            characterizing_scores={
                "em_emotional_awareness": 0.3,
                "em_perspective_taking": 0.2,
            },
            target_dimensions=["em_emotional_awareness", "em_perspective_taking"],
            contrastive_direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.GLOBAL_IMPROVE,
        )

        cs_prompt = build_generation_prompt(
            cs_candidate, 1, DOMAINS[DomainName.COMMONSENSE]
        )
        em_prompt = build_generation_prompt(
            em_candidate, 1, DOMAINS[DomainName.EMPATHY]
        )

        assert "commonsense" in cs_prompt.lower()
        assert "empathy" in em_prompt.lower()
        assert cs_prompt != em_prompt


# ── Batch entry construction ──────────────────────────────────────────────


class TestBuildGenerationEntries:
    """Test batch entry and manifest construction."""

    def test_produces_correct_count(self) -> None:
        candidates = [_make_candidate(dialogue_id=f"dlg-{i:03d}") for i in range(5)]
        entries, manifest = build_generation_entries(candidates)

        assert len(entries) == 5
        assert len(manifest) == 5

    def test_custom_ids_are_unique(self) -> None:
        candidates = [_make_candidate(dialogue_id=f"dlg-{i:03d}") for i in range(10)]
        entries, _ = build_generation_entries(candidates)
        ids = [e.custom_id for e in entries]

        assert len(ids) == len(set(ids))

    def test_custom_id_format_global_improve(self) -> None:
        candidate = _make_candidate(
            variant_type=VariantType.GLOBAL_IMPROVE,
            dialogue_id="dlg-abc",
        )
        entries, _ = build_generation_entries([candidate])

        assert entries[0].custom_id == "s2g-dlg-abc-gimp"

    def test_custom_id_format_global_degrade(self) -> None:
        candidate = _make_candidate(
            variant_type=VariantType.GLOBAL_DEGRADE,
            direction=ContrastiveDirection.NEGATIVE,
            dialogue_id="dlg-xyz",
        )
        entries, _ = build_generation_entries([candidate])

        assert entries[0].custom_id == "s2g-dlg-xyz-gdeg"

    def test_custom_id_for_dimension_targeted(self) -> None:
        candidate = Stage2Candidate(
            dialogue_id="dlg-dim",
            domain=DomainName.COMMONSENSE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
            ],
            characterizing_scores={"cs_causality": 0.3},
            target_dimensions=["cs_causality"],
            contrastive_direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        entries, _ = build_generation_entries([candidate])

        assert entries[0].custom_id == "s2g-dlg-dim-dt-cs_causality"

    def test_same_dialogue_multiple_variants_unique_ids(self) -> None:
        """Multiple candidates from the same dialogue must produce unique custom_ids."""
        base_kwargs = dict(
            dialogue_id="dlg-shared",
            domain=DomainName.COMMONSENSE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
            ],
            characterizing_scores={"cs_causality": 0.3, "cs_consistency": 0.2},
            contrastive_direction=ContrastiveDirection.POSITIVE,
        )
        candidates = [
            Stage2Candidate(
                **base_kwargs,
                target_dimensions=["cs_causality", "cs_consistency"],
                variant_type=VariantType.GLOBAL_IMPROVE,
            ),
            Stage2Candidate(
                **base_kwargs,
                target_dimensions=["cs_causality"],
                variant_type=VariantType.DIMENSION_TARGETED,
            ),
            Stage2Candidate(
                **base_kwargs,
                target_dimensions=["cs_consistency"],
                variant_type=VariantType.DIMENSION_TARGETED,
            ),
        ]
        entries, _ = build_generation_entries(candidates)
        ids = [e.custom_id for e in entries]

        assert len(ids) == len(set(ids)), f"Duplicate custom_ids: {ids}"

    def test_manifest_contains_metadata(self) -> None:
        candidate = _make_candidate()
        _, manifest = build_generation_entries([candidate])

        item = manifest[0]
        assert item["dialogue_id"] == "dlg-001"
        assert item["domain"] == "commonsense"
        assert item["variant_type"] == "global_improve"
        assert item["direction"] == "positive"
        assert "turn_count" in item
        # Candidate has 5 messages, so max continuation = 4; turn_count capped
        assert item["turn_count"] in {1, 3, 4}
        assert "target_dimensions" in item

    def test_manifest_carries_target_dimensions(self) -> None:
        candidate = Stage2Candidate(
            dialogue_id="dlg-td",
            domain=DomainName.COMMONSENSE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
            characterizing_scores={"cs_causality": 0.2},
            target_dimensions=["cs_causality"],
            contrastive_direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        _, manifest = build_generation_entries([candidate])
        assert manifest[0]["target_dimensions"] == ["cs_causality"]

    def test_empty_input_produces_empty_output(self) -> None:
        entries, manifest = build_generation_entries([])
        assert entries == []
        assert manifest == []

    def test_turn_count_capped_to_max_continuation(self) -> None:
        """turn_count cannot exceed len(messages) - 1."""
        candidate = Stage2Candidate(
            dialogue_id="dlg-short",
            domain=DomainName.COMMONSENSE,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
                Message(role="user", content="Bye"),
            ],
            characterizing_scores={"cs_causality": 0.3},
            target_dimensions=["cs_causality"],
            contrastive_direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.DIMENSION_TARGETED,
        )
        # 3 messages → max continuation = 2
        _, manifest = build_generation_entries([candidate])
        assert manifest[0]["turn_count"] <= 2

    def test_single_message_candidate_skipped(self) -> None:
        """Candidates with < 2 messages are skipped."""
        candidate = Stage2Candidate(
            dialogue_id="dlg-tiny",
            domain=DomainName.COHERENCE,
            messages=[Message(role="user", content="Hello")],
            characterizing_scores={"co_topical": 0.3},
            target_dimensions=["co_topical"],
            contrastive_direction=ContrastiveDirection.POSITIVE,
            variant_type=VariantType.GLOBAL_IMPROVE,
        )
        entries, manifest = build_generation_entries([candidate])
        assert entries == []
        assert manifest == []
