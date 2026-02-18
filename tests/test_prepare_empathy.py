"""Tests for empathy domain processor in scripts/stage_1/prepare_empathy.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.stage_1.prepare_empathy import (
    EmpathyProcessor,
    _assign_bucket,
    _count_turns,
    _stratified_sample,
)


# ── _stratified_sample (turn-count-only) ──────────────────────────────────


class TestEmpathyStratifiedSample:
    """Tests for turn-count-only stratified sampling."""

    @pytest.fixture()
    def sample_items(self) -> list[dict]:
        """Create items across turn count buckets."""
        items = []
        # Short dialogues (2-6 turns)
        for i in range(50):
            lines = "\n".join([f"Line {j}" for j in range(3)])
            items.append({"content": f"{lines} unique_{i}"})
        # Medium dialogues (7-10 turns)
        for i in range(50):
            lines = "\n".join([f"Line {j}" for j in range(8)])
            items.append({"content": f"{lines} unique_{i}"})
        # Long dialogues (11-16 turns)
        for i in range(50):
            lines = "\n".join([f"Line {j}" for j in range(12)])
            items.append({"content": f"{lines} unique_{i}"})
        # Very long dialogues (17+)
        for i in range(30):
            lines = "\n".join([f"Line {j}" for j in range(20)])
            items.append({"content": f"{lines} unique_{i}"})
        return items

    def test_produces_items(self, sample_items: list[dict]) -> None:
        result = _stratified_sample(sample_items, target=80, seed=42)
        assert len(result) > 0

    def test_near_target_count(self, sample_items: list[dict]) -> None:
        target = 80
        result = _stratified_sample(sample_items, target=target, seed=42)
        assert len(result) >= target * 0.7
        assert len(result) <= target * 1.1

    def test_deduplicates(self) -> None:
        items = [
            {"content": "same content"},
            {"content": "same content"},
            {"content": "different content"},
        ]
        result = _stratified_sample(items, target=10, seed=42)
        contents = [item["content"] for item in result]
        assert contents.count("same content") <= 1

    def test_deterministic(self, sample_items: list[dict]) -> None:
        result_a = _stratified_sample(sample_items, target=50, seed=42)
        result_b = _stratified_sample(sample_items, target=50, seed=42)
        contents_a = [item["content"] for item in result_a]
        contents_b = [item["content"] for item in result_b]
        assert contents_a == contents_b

    def test_samples_from_multiple_buckets(self, sample_items: list[dict]) -> None:
        result = _stratified_sample(sample_items, target=100, seed=42)
        buckets_seen = set()
        for item in result:
            turns = _count_turns(item["content"])
            bucket = _assign_bucket(turns)
            buckets_seen.add(bucket)
        # Should sample from at least 2 different buckets
        assert len(buckets_seen) >= 2


# ── EmpathyProcessor ──────────────────────────────────────────────────────


class TestEmpathyProcessor:
    """Tests for EmpathyProcessor methods (no I/O)."""

    @pytest.fixture()
    def processor(self, tmp_data_dir: Path) -> EmpathyProcessor:
        return EmpathyProcessor(base_path=tmp_data_dir)

    def test_build_prompt_nonempty(self, processor: EmpathyProcessor) -> None:
        item = {
            "content": "Speaker 1: I'm feeling really sad today.\nSpeaker 2: I'm so sorry to hear that."
        }
        prompt = processor.build_prompt(item)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_prompt_contains_content(self, processor: EmpathyProcessor) -> None:
        item = {"content": "I'm feeling really sad today."}
        prompt = processor.build_prompt(item)
        assert "feeling really sad" in prompt

    def test_build_prompt_empty_content_returns_empty(
        self, processor: EmpathyProcessor
    ) -> None:
        item = {"content": ""}
        prompt = processor.build_prompt(item)
        assert prompt == ""

    def test_create_custom_id_format(self, processor: EmpathyProcessor) -> None:
        item = {"content": "Some dialogue content"}
        custom_id = processor.create_custom_id(item)
        assert custom_id.startswith("em-")
        suffix = custom_id[3:]
        assert len(suffix) == 16

    def test_create_custom_id_deterministic(self, processor: EmpathyProcessor) -> None:
        item = {"content": "Some dialogue content"}
        id_a = processor.create_custom_id(item)
        id_b = processor.create_custom_id(item)
        assert id_a == id_b
