"""Tests for coherence domain processor in scripts/stage_1/prepare_coherence.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.config import COHERENCE_LABEL_RATIO
from scripts.stage_1.prepare_coherence import (
    CoherenceProcessor,
    _assign_bucket,
    _count_turns,
    _stratified_sample,
)


# ── _count_turns ──────────────────────────────────────────────────────────


class TestCountTurns:
    """Tests for the _count_turns helper."""

    def test_single_line(self) -> None:
        assert _count_turns("Hello") == 1

    def test_multiple_lines(self) -> None:
        content = "Line 1\nLine 2\nLine 3"
        assert _count_turns(content) == 3

    def test_blank_lines_ignored(self) -> None:
        content = "Line 1\n\nLine 2\n\n\nLine 3"
        assert _count_turns(content) == 3

    def test_empty_string_returns_1(self) -> None:
        assert _count_turns("") == 1

    def test_whitespace_only_returns_1(self) -> None:
        assert _count_turns("   \n  \n  ") == 1


# ── _assign_bucket ────────────────────────────────────────────────────────


class TestAssignBucket:
    """Tests for the _assign_bucket helper."""

    @pytest.mark.parametrize(
        ("turn_count", "expected"),
        [
            (2, (2, 6)),
            (6, (2, 6)),
            (7, (7, 10)),
            (10, (7, 10)),
            (11, (11, 16)),
            (16, (11, 16)),
            (17, (17, 999)),
            (50, (17, 999)),
            (999, (17, 999)),
        ],
    )
    def test_bucket_assignment(
        self, turn_count: int, expected: tuple[int, int]
    ) -> None:
        assert _assign_bucket(turn_count) == expected

    def test_value_below_lowest_bucket(self) -> None:
        # turn_count=1 is below the (2, 6) range, should fall to last bucket
        result = _assign_bucket(1)
        assert result == (17, 999)  # falls through to last bucket


# ── _stratified_sample ────────────────────────────────────────────────────


class TestStratifiedSample:
    """Tests for the _stratified_sample function."""

    @pytest.fixture()
    def sample_items(self) -> list[dict]:
        """Create 200 items with mixed labels and turn counts."""
        items = []
        for i in range(140):
            # 3 lines = bucket (2,6)
            items.append(
                {
                    "content": f"Turn A {i}\nTurn B {i}\nTurn C {i}",
                    "label": "Yes",
                }
            )
        for i in range(60):
            # 8 lines = bucket (7,10)
            lines = "\n".join([f"Turn {j} item {i}" for j in range(8)])
            items.append(
                {
                    "content": lines,
                    "label": "No",
                }
            )
        return items

    def test_produces_items_near_target(self, sample_items: list[dict]) -> None:
        target = 100
        result = _stratified_sample(
            sample_items,
            target=target,
            label_ratio=COHERENCE_LABEL_RATIO,
            seed=42,
        )
        # Should be close to target; may be slightly less if buckets are
        # underpopulated
        assert len(result) >= target * 0.8
        assert len(result) <= target * 1.1

    def test_respects_label_ratio_approximately(self, sample_items: list[dict]) -> None:
        target = 100
        result = _stratified_sample(
            sample_items,
            target=target,
            label_ratio=COHERENCE_LABEL_RATIO,
            seed=42,
        )
        yes_count = sum(1 for item in result if item["label"] == "Yes")
        no_count = sum(1 for item in result if item["label"] == "No")
        total = yes_count + no_count

        if total > 0:
            yes_ratio = yes_count / total
            # Should be roughly 70% Yes / 30% No (within tolerance)
            assert yes_ratio == pytest.approx(0.70, abs=0.15), (
                f"Yes ratio {yes_ratio:.2f} not approximately 0.70"
            )

    def test_deduplicates(self) -> None:
        items = [
            {"content": "Same content", "label": "Yes"},
            {"content": "Same content", "label": "Yes"},
            {"content": "Different content", "label": "No"},
        ]
        result = _stratified_sample(
            items, target=10, label_ratio=COHERENCE_LABEL_RATIO, seed=42
        )
        contents = [item["content"] for item in result]
        assert contents.count("Same content") <= 1

    def test_deterministic(self, sample_items: list[dict]) -> None:
        result_a = _stratified_sample(
            sample_items,
            target=50,
            label_ratio=COHERENCE_LABEL_RATIO,
            seed=42,
        )
        result_b = _stratified_sample(
            sample_items,
            target=50,
            label_ratio=COHERENCE_LABEL_RATIO,
            seed=42,
        )
        contents_a = [item["content"] for item in result_a]
        contents_b = [item["content"] for item in result_b]
        assert contents_a == contents_b


# ── CoherenceProcessor ────────────────────────────────────────────────────


class TestCoherenceProcessor:
    """Tests for CoherenceProcessor methods (no I/O)."""

    @pytest.fixture()
    def processor(self, tmp_data_dir: Path) -> CoherenceProcessor:
        return CoherenceProcessor(base_path=tmp_data_dir)

    def test_build_prompt_nonempty(
        self,
        processor: CoherenceProcessor,
        sample_coherence_dialogue: dict,
    ) -> None:
        prompt = processor.build_prompt(sample_coherence_dialogue)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_prompt_contains_content(
        self,
        processor: CoherenceProcessor,
        sample_coherence_dialogue: dict,
    ) -> None:
        prompt = processor.build_prompt(sample_coherence_dialogue)
        # The prompt should embed the dialogue content
        assert "how are you doing today" in prompt

    def test_build_prompt_empty_content_returns_empty(
        self,
        processor: CoherenceProcessor,
    ) -> None:
        item = {"content": "", "label": "Yes"}
        prompt = processor.build_prompt(item)
        assert prompt == ""

    def test_create_custom_id_format(
        self,
        processor: CoherenceProcessor,
        sample_coherence_dialogue: dict,
    ) -> None:
        custom_id = processor.create_custom_id(sample_coherence_dialogue)
        assert custom_id.startswith("co-")
        # 16 hex chars after prefix
        suffix = custom_id[3:]
        assert len(suffix) == 16
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_create_custom_id_deterministic(
        self,
        processor: CoherenceProcessor,
        sample_coherence_dialogue: dict,
    ) -> None:
        id_a = processor.create_custom_id(sample_coherence_dialogue)
        id_b = processor.create_custom_id(sample_coherence_dialogue)
        assert id_a == id_b
