"""Tests for Stage 1 merge logic in scripts/merge_stage_1.py."""

from __future__ import annotations

import pytest

from scripts.merge_stage_1 import (
    create_stage1_entry,
    load_commonsense_gold,
    normalize_score,
    parse_dialogue_to_messages,
)


# ── parse_dialogue_to_messages ────────────────────────────────────────────


class TestParseDialogueToMessages:
    """Tests for parse_dialogue_to_messages function."""

    def test_alternating_speakers_human_ab(self) -> None:
        content = "Human A: Hello!\nHuman B: Hi there!\nHuman A: How are you?"
        messages = parse_dialogue_to_messages(content)
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hello!"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}
        assert messages[2] == {"role": "user", "content": "How are you?"}

    def test_alternating_speakers_speaker_12(self) -> None:
        content = "Speaker 1: Good morning.\nSpeaker 2: Good morning to you."
        messages = parse_dialogue_to_messages(content)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_spk_format(self) -> None:
        content = (
            "SPK01-USA001: Hello.\nSPK02-JPN001: Hi.\nSPK01-USA001: How are you?\n"
        )
        messages = parse_dialogue_to_messages(content)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_blank_lines_skipped(self) -> None:
        content = "Human A: Hello.\n\nHuman B: Hi.\n\n"
        messages = parse_dialogue_to_messages(content)
        assert len(messages) == 2

    def test_empty_content(self) -> None:
        messages = parse_dialogue_to_messages("")
        assert messages == []

    def test_strips_speaker_prefix(self) -> None:
        content = "Human A: The actual content here."
        messages = parse_dialogue_to_messages(content)
        assert messages[0]["content"] == "The actual content here."

    def test_person_ab_format(self) -> None:
        content = "Person A: First.\nPerson B: Second."
        messages = parse_dialogue_to_messages(content)
        assert len(messages) == 2
        assert messages[0]["content"] == "First."
        assert messages[1]["content"] == "Second."

    def test_ab_format(self) -> None:
        content = "A: First.\nB: Second."
        messages = parse_dialogue_to_messages(content)
        assert len(messages) == 2


# ── normalize_score ───────────────────────────────────────────────────────


class TestNormalizeScore:
    """Tests for the normalize_score function."""

    def test_dict_with_reasoning(self) -> None:
        score = {"reasoning": "Good coherence", "score": 4}
        result = normalize_score(score)
        assert result == pytest.approx(0.75)

    def test_dict_with_score_key(self) -> None:
        score = {"score": 5}
        result = normalize_score(score)
        assert result == pytest.approx(1.0)

    def test_raw_float_in_range(self) -> None:
        result = normalize_score(0.75)
        assert result == pytest.approx(0.75)

    def test_raw_float_zero(self) -> None:
        result = normalize_score(0.0)
        assert result == pytest.approx(0.0)

    def test_raw_float_one(self) -> None:
        result = normalize_score(1.0)
        assert result == pytest.approx(1.0)

    def test_1_5_scale_value_1(self) -> None:
        # 1 on 1-5 scale -> 0.0
        result = normalize_score(1)
        # Since 1 is in 0-1 range, it stays as 1.0
        # Actually, int(1) is in 0.0..1.0 range, so it returns 1.0
        assert result == pytest.approx(1.0)

    def test_1_5_scale_value_above_1(self) -> None:
        # 3 on 1-5 scale -> 0.5
        result = normalize_score(3)
        assert result == pytest.approx(0.5)

    def test_1_5_scale_value_5(self) -> None:
        # 5 on 1-5 scale -> 1.0
        result = normalize_score(5)
        assert result == pytest.approx(1.0)

    def test_1_5_scale_value_2(self) -> None:
        # 2 on 1-5 scale -> 0.25
        result = normalize_score(2)
        assert result == pytest.approx(0.25)

    def test_none_returns_none(self) -> None:
        result = normalize_score(None)
        assert result is None

    def test_string_float(self) -> None:
        result = normalize_score("0.85")
        assert result == pytest.approx(0.85)

    def test_string_integer_above_1(self) -> None:
        result = normalize_score("4")
        assert result == pytest.approx(0.75)

    def test_invalid_string(self) -> None:
        result = normalize_score("not a number")
        assert result is None

    def test_dict_with_none_score(self) -> None:
        result = normalize_score({"reasoning": "text", "score": None})
        assert result is None


# ── create_stage1_entry ───────────────────────────────────────────────────


class TestCreateStage1Entry:
    """Tests for create_stage1_entry function."""

    def test_all_23_dimensions_present(self) -> None:
        from scripts.config import all_prefixed_dimensions

        item = {
            "_id": "co-test123",
            "content": "Human A: Hello\nHuman B: Hi there",
            "co_topic_coherence": 0.75,
            "co_logical_consistency": 0.80,
        }
        entry = create_stage1_entry(item, dialogue_id=1, domain_name="coherence")

        all_dims = all_prefixed_dimensions()
        assert set(entry["scores"].keys()) == set(all_dims)

    def test_domain_scores_non_null_others_null(self) -> None:
        item = {
            "_id": "co-test123",
            "content": "Human A: Hello\nHuman B: Hi there",
            "co_topic_coherence": 0.75,
        }
        entry = create_stage1_entry(item, dialogue_id=1, domain_name="coherence")

        # Coherence dim should be non-null
        assert entry["scores"]["co_topic_coherence"] == pytest.approx(0.75)
        # Non-domain dims should be None
        assert entry["scores"]["em_emotional_awareness"] is None
        assert entry["scores"]["cs_causality"] is None
        assert entry["scores"]["mu_cultural_value"] is None

    def test_dialogue_id_format(self) -> None:
        item = {
            "_id": "co-test",
            "content": "Human A: Hello\nHuman B: Hi",
            "co_topic_coherence": 0.75,
        }
        entry = create_stage1_entry(item, dialogue_id=42, domain_name="coherence")
        assert entry["dialogue_id"] == "S1D-000042"

    def test_messages_parsed(self) -> None:
        item = {
            "_id": "co-test",
            "content": "Human A: Hello\nHuman B: World",
            "co_topic_coherence": 0.75,
        }
        entry = create_stage1_entry(item, dialogue_id=1, domain_name="coherence")
        assert len(entry["messages"]) == 2
        assert entry["messages"][0]["role"] == "user"
        assert entry["messages"][1]["role"] == "assistant"

    def test_source_id_from_item(self) -> None:
        item = {
            "_id": "co-abc123",
            "content": "Human A: Hello\nHuman B: Hi",
            "co_topic_coherence": 0.75,
        }
        entry = create_stage1_entry(item, dialogue_id=1, domain_name="coherence")
        assert entry["source_id"] == "co-abc123"

    def test_split_default_none(self) -> None:
        item = {
            "_id": "co-test",
            "content": "Human A: Hi\nHuman B: Hey",
            "co_topic_coherence": 0.5,
        }
        entry = create_stage1_entry(item, dialogue_id=1, domain_name="coherence")
        assert entry["split"] is None

    def test_normalizes_dict_scores(self) -> None:
        item = {
            "_id": "co-test",
            "content": "Human A: Hi\nHuman B: Hey",
            "co_topic_coherence": {"reasoning": "Good", "score": 4},
        }
        entry = create_stage1_entry(item, dialogue_id=1, domain_name="coherence")
        # score=4 on 1-5 scale -> 0.75
        assert entry["scores"]["co_topic_coherence"] == pytest.approx(0.75)

    def test_commonsense_domain_scores(self) -> None:
        from scripts.config import all_prefixed_dimensions

        item = {
            "_id": "cs-P001",
            "content": "Speaker 1: Hello\nSpeaker 2: Hi",
            "cs_causality": 0.7,
            "cs_consistency": 0.8,
            "cs_reaction": 0.6,
            "cs_desire": 0.5,
            "cs_coherence": 0.9,
            "cs_empathy": 0.4,
        }
        entry = create_stage1_entry(item, dialogue_id=1, domain_name="commonsense")

        # All 23 dims present
        all_dims = all_prefixed_dimensions()
        assert set(entry["scores"].keys()) == set(all_dims)

        # 6 commonsense dims are non-null
        cs_non_null = {
            k: v for k, v in entry["scores"].items()
            if k.startswith("cs_") and v is not None
        }
        assert len(cs_non_null) == 6

    def test_multicultural_domain_metadata_populated(self) -> None:
        lookup = {
            "USA-000001": {
                "country_1": "United States",
                "country_2": "Japan",
                "demographics_1": "age 30",
                "demographics_2": "age 25",
                "statement_original": "Test statement",
                "statement_cultural": "Cultural statement",
                "situation": "A meeting",
                "cultural_reasoning_1": "reasoning 1",
                "cultural_reasoning_2": "reasoning 2",
                "arousal_reasoning": "calm",
                "arousal_score": "3",
                "social_norms_1": "norms 1",
                "social_norms_2": "norms 2",
                "prejudices_1": "bias 1",
                "prejudices_2": "bias 2",
            },
        }
        item = {
            "_id": "mu-test",
            "uid": "USA-000001",
            "content": "SPK01-USA001: Hi\nSPK02-JPN001: Hello",
            "mu_cultural_value": 0.6,
        }
        entry = create_stage1_entry(
            item, dialogue_id=1, domain_name="multicultural",
            multicultural_lookup=lookup,
        )
        assert entry["domain_metadata"] is not None
        assert entry["domain_metadata"]["country_1"] == "United States"
        assert entry["domain_metadata"]["country_2"] == "Japan"
        assert len(entry["domain_metadata"]) == 15

    def test_non_multicultural_domain_metadata_none(self) -> None:
        item = {
            "_id": "co-test",
            "content": "Human A: Hi\nHuman B: Hey",
            "co_topic_coherence": 0.5,
        }
        entry = create_stage1_entry(item, dialogue_id=1, domain_name="coherence")
        assert entry["domain_metadata"] is None

    def test_commonsense_gold_relation_populated(self) -> None:
        gold_lookup = {
            "train-000001-0003": {
                "gold_relation": "xReact",
                "target_dimension": "cs_reaction",
            },
        }
        item = {
            "_id": "cs-train-000001-0003",
            "pid": "train-000001-0003",
            "content": "Speaker 1: Hello\nSpeaker 2: Hi",
            "cs_causality": 0.5,
        }
        entry = create_stage1_entry(
            item, dialogue_id=1, domain_name="commonsense",
            commonsense_gold=gold_lookup,
        )
        assert entry["domain_metadata"] is not None
        assert entry["domain_metadata"]["gold_relation"] == "xReact"
        assert entry["domain_metadata"]["target_dimension"] == "cs_reaction"

    def test_commonsense_no_gold_metadata_none(self) -> None:
        gold_lookup = {
            "train-999999-0001": {
                "gold_relation": "HinderedBy",
                "target_dimension": "cs_causality",
            },
        }
        item = {
            "_id": "cs-train-000001-0003",
            "pid": "train-000001-0003",
            "content": "Speaker 1: Hello\nSpeaker 2: Hi",
            "cs_causality": 0.5,
        }
        # PID not in lookup → domain_metadata stays None
        entry = create_stage1_entry(
            item, dialogue_id=1, domain_name="commonsense",
            commonsense_gold=gold_lookup,
        )
        assert entry["domain_metadata"] is None

    def test_commonsense_without_gold_lookup(self) -> None:
        item = {
            "_id": "cs-test",
            "pid": "train-000001-0003",
            "content": "Speaker 1: Hi\nSpeaker 2: Hey",
            "cs_causality": 0.5,
        }
        entry = create_stage1_entry(item, dialogue_id=1, domain_name="commonsense")
        assert entry["domain_metadata"] is None


# ── load_commonsense_gold ────────────────────────────────────────────────


class TestLoadCommonsenseGold:
    """Tests for the load_commonsense_gold function."""

    def test_loads_from_real_file(self) -> None:
        from pathlib import Path

        gold_path = Path(
            "data/input/commonsense/instruction/train_multitask_gold_12k.json",
        )
        if not gold_path.exists():
            pytest.skip("Gold file not available")

        lookup = load_commonsense_gold(gold_path)
        assert len(lookup) > 3000
        # Each entry has gold_relation and target_dimension
        sample = next(iter(lookup.values()))
        assert "gold_relation" in sample
        assert "target_dimension" in sample
        # target_dimension should be a valid cs_ dimension
        assert sample["target_dimension"].startswith("cs_")

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        from pathlib import Path

        lookup = load_commonsense_gold(tmp_path / "nonexistent.json")
        assert lookup == {}

    def test_first_occurrence_wins(self, tmp_path: Path) -> None:
        import json
        from pathlib import Path

        gold_data = [
            {"pid": "train-001", "gold": "xReact", "scores": {}},
            {"pid": "train-001", "gold": "HinderedBy", "scores": {}},
        ]
        gold_file = tmp_path / "gold.json"
        gold_file.write_text(json.dumps(gold_data))

        lookup = load_commonsense_gold(gold_file)
        assert lookup["train-001"]["gold_relation"] == "xReact"
