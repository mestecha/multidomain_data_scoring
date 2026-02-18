"""Tests for commonsense domain processor in scripts/stage_1/prepare_commonsense.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.stage_1.prepare_commonsense import (
    RELATION_RESPONSE_COLS,
    CommonsenseProcessor,
)


class TestCommonsenseProcessor:
    """Tests for CommonsenseProcessor helper methods.

    The __init__ only sets paths, so we can instantiate without calling
    load_data (which requires actual TSV files).
    """

    @pytest.fixture()
    def processor(self, tmp_data_dir: Path) -> CommonsenseProcessor:
        return CommonsenseProcessor(base_path=tmp_data_dir)

    # ── _parse_context ────────────────────────────────────────────────

    def test_parse_context_basic(self, processor: CommonsenseProcessor) -> None:
        ctx = "user1: Hello there [UTT] user2: Hi, how are you?"
        result = processor._parse_context(ctx)
        assert "Speaker 1: Hello there" in result
        assert "Speaker 2: Hi, how are you?" in result

    def test_parse_context_multiple_utts(self, processor: CommonsenseProcessor) -> None:
        ctx = "user1: First turn [UTT] user2: Second turn [UTT] user1: Third turn"
        result = processor._parse_context(ctx)
        lines = result.strip().split("\n")
        assert len(lines) == 3

    def test_parse_context_empty(self, processor: CommonsenseProcessor) -> None:
        result = processor._parse_context("")
        assert result == ""

    def test_parse_context_strips_whitespace(
        self, processor: CommonsenseProcessor
    ) -> None:
        ctx = "user1: Hello   [UTT]   user2: World  "
        result = processor._parse_context(ctx)
        assert "Speaker 1: Hello" in result
        assert "Speaker 2: World" in result

    # ── _get_relation_responses ───────────────────────────────────────

    def test_get_relation_responses_all_7(
        self,
        processor: CommonsenseProcessor,
        sample_commonsense_row: dict,
    ) -> None:
        responses = processor._get_relation_responses(sample_commonsense_row)
        assert len(responses) == 7
        expected_keys = {
            "HinderedBy",
            "IsAfter",
            "xAttr",
            "xReact",
            "oReact",
            "xWant",
            "oWant",
        }
        assert set(responses.keys()) == expected_keys

    def test_get_relation_responses_values(
        self,
        processor: CommonsenseProcessor,
        sample_commonsense_row: dict,
    ) -> None:
        responses = processor._get_relation_responses(sample_commonsense_row)
        assert responses["HinderedBy"] == "The store could have been closed."
        assert responses["xWant"] == "The speaker wants to cook a nice meal."

    def test_get_relation_responses_missing_cols(
        self, processor: CommonsenseProcessor
    ) -> None:
        item = {"PID": "P001", "CTX": "some context"}
        responses = processor._get_relation_responses(item)
        assert len(responses) == 7
        # All should be empty strings when cols are missing
        for val in responses.values():
            assert val == ""

    # ── _build_dim_prompt ─────────────────────────────────────────────

    def test_build_dim_prompt_includes_relation_responses(
        self,
        processor: CommonsenseProcessor,
        sample_commonsense_row: dict,
    ) -> None:
        prompt = processor._build_dim_prompt(sample_commonsense_row)
        assert len(prompt) > 0
        # Should contain content from context and target
        assert "Speaker 1" in prompt or "Speaker 2" in prompt
        # Should reference relation responses
        assert "store could have been closed" in prompt
        assert "cook a nice meal" in prompt

    def test_build_dim_prompt_empty_context_returns_empty(
        self, processor: CommonsenseProcessor
    ) -> None:
        item = {"CTX": "", "SEG": "some target"}
        prompt = processor._build_dim_prompt(item)
        assert prompt == ""

    def test_build_dim_prompt_empty_target_returns_empty(
        self, processor: CommonsenseProcessor
    ) -> None:
        item = {"CTX": "user1: hello [UTT] user2: hi", "SEG": ""}
        prompt = processor._build_dim_prompt(item)
        assert prompt == ""

    # ── _build_dlg_prompt ─────────────────────────────────────────────

    def test_build_dlg_prompt_basic(
        self,
        processor: CommonsenseProcessor,
        sample_commonsense_row: dict,
    ) -> None:
        prompt = processor._build_dlg_prompt(sample_commonsense_row)
        assert len(prompt) > 0

    def test_build_dlg_prompt_shorter_than_dim(
        self,
        processor: CommonsenseProcessor,
        sample_commonsense_row: dict,
    ) -> None:
        dim_prompt = processor._build_dim_prompt(sample_commonsense_row)
        dlg_prompt = processor._build_dlg_prompt(sample_commonsense_row)
        # The dialogue-level prompt should be shorter because it doesn't
        # include relation responses
        assert len(dlg_prompt) < len(dim_prompt)

    def test_build_dlg_prompt_empty_context_returns_empty(
        self, processor: CommonsenseProcessor
    ) -> None:
        item = {"CTX": "", "SEG": "some target"}
        prompt = processor._build_dlg_prompt(item)
        assert prompt == ""

    # ── create_custom_id ──────────────────────────────────────────────

    def test_create_custom_id_uses_pid(
        self,
        processor: CommonsenseProcessor,
        sample_commonsense_row: dict,
    ) -> None:
        custom_id = processor.create_custom_id(sample_commonsense_row)
        assert custom_id == "cs-P001"

    def test_create_custom_id_falls_back_to_fingerprint(
        self, processor: CommonsenseProcessor
    ) -> None:
        item = {"CTX": "some context", "SEG": "some segment"}
        custom_id = processor.create_custom_id(item)
        assert custom_id.startswith("cs-")
        suffix = custom_id[3:]
        assert len(suffix) == 16


class TestRelationResponseCols:
    """Tests for the RELATION_RESPONSE_COLS constant."""

    def test_has_7_columns(self) -> None:
        assert len(RELATION_RESPONSE_COLS) == 7

    def test_all_end_with_response(self) -> None:
        for col in RELATION_RESPONSE_COLS:
            assert col.endswith("_response"), (
                f"Column '{col}' does not end with '_response'"
            )
