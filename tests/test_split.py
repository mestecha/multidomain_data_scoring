"""Tests for train/val/test split assignment in scripts/split.py."""

from __future__ import annotations

from collections import Counter

import pytest

from scripts.split import assign_splits


def _make_entries(
    per_domain: int = 50,
    domains: list[str] | None = None,
) -> list[dict]:
    """Create synthetic entries for split testing.

    Each entry needs at minimum a 'domain' field.
    We create enough entries to make stratified splitting reliable.
    """
    domains = domains or ["coherence", "empathy", "commonsense", "multicultural"]
    entries = []
    for domain in domains:
        for i in range(per_domain):
            entries.append(
                {
                    "domain": domain,
                    "dialogue_id": f"S1D-{domain}-{i:04d}",
                    "scores": {f"dim_{i}": 0.5},
                }
            )
    return entries


class TestAssignSplits:
    """Tests for the assign_splits function."""

    def test_every_entry_gets_a_split(self) -> None:
        entries = _make_entries(per_domain=50)
        result = assign_splits(entries, seed=42)
        for entry in result:
            assert "split" in entry, f"Entry missing split: {entry}"
            assert entry["split"] in {"train", "val", "test"}

    def test_ratios_within_tolerance(self) -> None:
        """Split ratios should be within 2% of 75/10/15."""
        entries = _make_entries(per_domain=100)
        result = assign_splits(entries, seed=42)
        total = len(result)

        split_counts = Counter(e["split"] for e in result)
        train_ratio = split_counts["train"] / total
        val_ratio = split_counts["val"] / total
        test_ratio = split_counts["test"] / total

        assert train_ratio == pytest.approx(0.75, abs=0.02), (
            f"Train ratio {train_ratio:.3f} not within 2% of 0.75"
        )
        assert val_ratio == pytest.approx(0.10, abs=0.02), (
            f"Val ratio {val_ratio:.3f} not within 2% of 0.10"
        )
        assert test_ratio == pytest.approx(0.15, abs=0.02), (
            f"Test ratio {test_ratio:.3f} not within 2% of 0.15"
        )

    def test_each_domain_proportional_in_each_split(self) -> None:
        """Each domain should appear proportionally in every split."""
        entries = _make_entries(per_domain=100)
        result = assign_splits(entries, seed=42)

        # Group by domain and split
        domain_split_counts: dict[str, dict[str, int]] = {}
        for entry in result:
            domain = entry["domain"]
            split = entry["split"]
            if domain not in domain_split_counts:
                domain_split_counts[domain] = Counter()
            domain_split_counts[domain][split] += 1

        for domain, counts in domain_split_counts.items():
            domain_total = sum(counts.values())
            for split_name, expected_ratio in [
                ("train", 0.75),
                ("val", 0.10),
                ("test", 0.15),
            ]:
                actual_ratio = counts.get(split_name, 0) / domain_total
                assert actual_ratio == pytest.approx(expected_ratio, abs=0.05), (
                    f"Domain '{domain}' split '{split_name}': "
                    f"ratio {actual_ratio:.3f} not within 5% of {expected_ratio}"
                )

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed should produce identical splits."""
        entries_a = _make_entries(per_domain=50)
        entries_b = _make_entries(per_domain=50)

        result_a = assign_splits(entries_a, seed=123)
        result_b = assign_splits(entries_b, seed=123)

        splits_a = [e["split"] for e in result_a]
        splits_b = [e["split"] for e in result_b]
        assert splits_a == splits_b

    def test_different_seed_produces_different_splits(self) -> None:
        """Different seeds should usually produce different assignments."""
        entries_a = _make_entries(per_domain=50)
        entries_b = _make_entries(per_domain=50)

        result_a = assign_splits(entries_a, seed=42)
        result_b = assign_splits(entries_b, seed=99)

        splits_a = [e["split"] for e in result_a]
        splits_b = [e["split"] for e in result_b]
        # It is theoretically possible for two seeds to match, but
        # with 200 entries, extremely unlikely.
        assert splits_a != splits_b

    def test_preserves_existing_fields(self) -> None:
        """assign_splits should not remove existing fields."""
        entries = _make_entries(per_domain=20)
        result = assign_splits(entries, seed=42)
        for entry in result:
            assert "domain" in entry
            assert "dialogue_id" in entry
            assert "scores" in entry
