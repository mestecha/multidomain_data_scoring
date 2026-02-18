"""Tests for Stage 1 result parsing in scripts/stage_1/parse_results.py."""

from __future__ import annotations

import pytest

from scripts.stage_1.parse_results import (
    coerce_float,
    coerce_score,
    normalize_1_5_score,
    parse_message_to_dict,
    strip_code_fences,
)


# ── strip_code_fences ─────────────────────────────────────────────────────


class TestStripCodeFences:
    """Tests for strip_code_fences function."""

    def test_plain_json_unchanged(self) -> None:
        result = strip_code_fences('{"score": 5}')
        assert result == '{"score": 5}'

    def test_backtick_wrapping(self) -> None:
        result = strip_code_fences('```\n{"score": 5}\n```')
        assert result == '{"score": 5}'

    def test_json_fence(self) -> None:
        result = strip_code_fences('```json\n{"score": 5}\n```')
        assert result == '{"score": 5}'

    def test_json_fence_case_insensitive(self) -> None:
        result = strip_code_fences('```JSON\n{"score": 5}\n```')
        assert result == '{"score": 5}'

    def test_handles_none(self) -> None:
        result = strip_code_fences(None)
        assert result == ""

    def test_handles_empty_string(self) -> None:
        result = strip_code_fences("")
        assert result == ""

    def test_extracts_json_from_surrounding_text(self) -> None:
        result = strip_code_fences('Here is the result: {"score": 5} done')
        assert result == '{"score": 5}'

    def test_nested_braces(self) -> None:
        input_str = '{"a": {"b": 1}, "c": 2}'
        result = strip_code_fences(input_str)
        assert result == '{"a": {"b": 1}, "c": 2}'


# ── coerce_score ──────────────────────────────────────────────────────────


class TestCoerceScore:
    """Tests for coerce_score function (1-5 clamping)."""

    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            (3, 3),
            (3.0, 3),
            ("3", 3),
            (3.7, 4),
            (3.2, 3),
        ],
    )
    def test_normal_values(self, input_val, expected: int) -> None:
        assert coerce_score(input_val) == expected

    def test_clamps_to_1(self) -> None:
        assert coerce_score(0) == 1
        assert coerce_score(-5) == 1

    def test_clamps_to_5(self) -> None:
        assert coerce_score(10) == 5
        assert coerce_score(100) == 5

    def test_returns_none_for_invalid(self) -> None:
        assert coerce_score("not a number") is None
        assert coerce_score(None) is None
        assert coerce_score("") is None


# ── normalize_1_5_score ───────────────────────────────────────────────────


class TestNormalize15Score:
    """Tests for normalize_1_5_score function."""

    def test_score_1_maps_to_0(self) -> None:
        assert normalize_1_5_score(1) == 0.0

    def test_score_3_maps_to_0_5(self) -> None:
        assert normalize_1_5_score(3) == 0.5

    def test_score_5_maps_to_1_0(self) -> None:
        assert normalize_1_5_score(5) == 1.0

    def test_score_2_maps_correctly(self) -> None:
        assert normalize_1_5_score(2) == 0.25

    def test_score_4_maps_correctly(self) -> None:
        assert normalize_1_5_score(4) == 0.75

    def test_none_returns_none(self) -> None:
        assert normalize_1_5_score(None) is None


# ── coerce_float ──────────────────────────────────────────────────────────


class TestCoerceFloat:
    """Tests for coerce_float function (0.0-1.0 clamping)."""

    @pytest.mark.parametrize(
        ("input_val", "expected"),
        [
            (0.5, 0.5),
            ("0.75", 0.75),
            (0, 0.0),
            (1, 1.0),
            (0.333, 0.33),
        ],
    )
    def test_normal_values(self, input_val, expected: float) -> None:
        assert coerce_float(input_val) == pytest.approx(expected)

    def test_clamps_below_zero(self) -> None:
        assert coerce_float(-0.5) == 0.0
        assert coerce_float(-100) == 0.0

    def test_clamps_above_one(self) -> None:
        assert coerce_float(1.5) == 1.0
        assert coerce_float(99.0) == 1.0

    def test_returns_none_for_invalid(self) -> None:
        assert coerce_float("not a number") is None
        assert coerce_float(None) is None


# ── parse_message_to_dict ─────────────────────────────────────────────────


class TestParseMessageToDict:
    """Tests for parse_message_to_dict function."""

    def test_valid_json(self) -> None:
        result = parse_message_to_dict('{"score": 5, "reasoning": "Good"}')
        assert result == {"score": 5, "reasoning": "Good"}

    def test_trailing_comma(self) -> None:
        result = parse_message_to_dict('{"score": 5, "reasoning": "Good",}')
        assert result is not None
        assert result["score"] == 5

    def test_code_fences(self) -> None:
        input_str = '```json\n{"score": 5}\n```'
        result = parse_message_to_dict(input_str)
        assert result is not None
        assert result["score"] == 5

    def test_returns_none_for_unparseable(self) -> None:
        result = parse_message_to_dict("this is not json at all")
        assert result is None

    def test_empty_string(self) -> None:
        result = parse_message_to_dict("")
        assert result is None

    def test_nested_json(self) -> None:
        input_str = '{"dim_a": {"reasoning": "text", "score": 4}, "dim_b": 0.5}'
        result = parse_message_to_dict(input_str)
        assert result is not None
        assert result["dim_a"]["score"] == 4
        assert result["dim_b"] == 0.5
