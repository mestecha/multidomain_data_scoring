"""Tests for multicultural domain processor in scripts/stage_1/prepare_multicultural.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.stage_1.prepare_multicultural import MulticulturalProcessor


class TestMulticulturalProcessor:
    """Tests for MulticulturalProcessor helper methods."""

    @pytest.fixture()
    def processor(self, tmp_data_dir: Path) -> MulticulturalProcessor:
        return MulticulturalProcessor(base_path=tmp_data_dir)

    # ── _parse_conversation ───────────────────────────────────────────

    def test_parse_conversation_basic(self, processor: MulticulturalProcessor) -> None:
        conversation = (
            "SPK01-USA001: Hello, nice to meet you.\n"
            "SPK02-JPN001: Nice to meet you too.\n"
        )
        result = processor._parse_conversation(conversation)
        assert "Turn 1 (Speaker 1): Hello, nice to meet you." in result
        assert "Turn 2 (Speaker 2): Nice to meet you too." in result

    def test_parse_conversation_multiple_turns(
        self, processor: MulticulturalProcessor
    ) -> None:
        conversation = (
            "SPK01-USA001: First turn.\n"
            "SPK02-JPN001: Second turn.\n"
            "SPK01-USA001: Third turn.\n"
            "SPK02-JPN001: Fourth turn.\n"
        )
        result = processor._parse_conversation(conversation)
        lines = [ln for ln in result.split("\n") if ln.strip()]
        assert len(lines) == 4
        assert "Turn 1 (Speaker 1)" in lines[0]
        assert "Turn 2 (Speaker 2)" in lines[1]
        assert "Turn 3 (Speaker 1)" in lines[2]
        assert "Turn 4 (Speaker 2)" in lines[3]

    def test_parse_conversation_empty(self, processor: MulticulturalProcessor) -> None:
        result = processor._parse_conversation("")
        assert result == ""

    def test_parse_conversation_blank_lines_skipped(
        self, processor: MulticulturalProcessor
    ) -> None:
        conversation = "SPK01-USA001: Hello.\n\nSPK02-JPN001: Hi.\n"
        result = processor._parse_conversation(conversation)
        lines = [ln for ln in result.split("\n") if ln.strip()]
        assert len(lines) == 2

    def test_parse_conversation_continuation_lines(
        self, processor: MulticulturalProcessor
    ) -> None:
        """Lines without SPK prefix should be appended to previous turn."""
        conversation = (
            "SPK01-USA001: Hello, I wanted to say\n"
            "something important to you.\n"
            "SPK02-JPN001: Sure, go ahead.\n"
        )
        result = processor._parse_conversation(conversation)
        lines = [ln for ln in result.split("\n") if ln.strip()]
        assert len(lines) == 2
        # First turn should contain the continuation
        assert "something important" in lines[0]

    # ── build_prompt ──────────────────────────────────────────────────

    def test_build_prompt_includes_countries(
        self,
        processor: MulticulturalProcessor,
        sample_multicultural_row: dict,
    ) -> None:
        prompt = processor.build_prompt(sample_multicultural_row)
        assert "US" in prompt
        assert "JP" in prompt

    def test_build_prompt_includes_statement(
        self,
        processor: MulticulturalProcessor,
        sample_multicultural_row: dict,
    ) -> None:
        prompt = processor.build_prompt(sample_multicultural_row)
        assert "Respect for elders" in prompt

    def test_build_prompt_nonempty(
        self,
        processor: MulticulturalProcessor,
        sample_multicultural_row: dict,
    ) -> None:
        prompt = processor.build_prompt(sample_multicultural_row)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_prompt_empty_conversation_returns_empty(
        self, processor: MulticulturalProcessor
    ) -> None:
        item = {
            "conversation": "",
            "statement_original": "A statement",
            "country_1": "US",
            "country_2": "JP",
        }
        prompt = processor.build_prompt(item)
        assert prompt == ""

    def test_build_prompt_empty_statement_returns_empty(
        self, processor: MulticulturalProcessor
    ) -> None:
        item = {
            "conversation": "SPK01-USA001: Hello\nSPK02-JPN001: Hi",
            "statement_original": "",
            "country_1": "US",
            "country_2": "JP",
        }
        prompt = processor.build_prompt(item)
        assert prompt == ""

    # ── create_custom_id ──────────────────────────────────────────────

    def test_create_custom_id_from_uid(
        self,
        processor: MulticulturalProcessor,
        sample_multicultural_row: dict,
    ) -> None:
        custom_id = processor.create_custom_id(sample_multicultural_row)
        assert custom_id == "mu-MC-12345"

    def test_create_custom_id_falls_back_to_fingerprint(
        self, processor: MulticulturalProcessor
    ) -> None:
        item = {"conversation": "SPK01-USA001: Hello"}
        custom_id = processor.create_custom_id(item)
        assert custom_id.startswith("mu-")
        suffix = custom_id[3:]
        assert len(suffix) == 16

    def test_create_custom_id_deterministic(
        self,
        processor: MulticulturalProcessor,
        sample_multicultural_row: dict,
    ) -> None:
        id_a = processor.create_custom_id(sample_multicultural_row)
        id_b = processor.create_custom_id(sample_multicultural_row)
        assert id_a == id_b
