"""Tests for pipeline configuration in scripts/config.py."""

from __future__ import annotations

import pytest

from scripts.config import (
    COHERENCE,
    COMMONSENSE,
    CS_RELATION_TO_DIM,
    DOMAINS,
    EMPATHY,
    MULTICULTURAL,
    SPLIT_RATIOS,
    all_prefixed_dimensions,
    get_domain,
)
from scripts.models import DomainName


class TestDomainConfigurations:
    """Tests for domain configuration definitions."""

    @pytest.mark.parametrize(
        "config",
        [COHERENCE, EMPATHY, COMMONSENSE, MULTICULTURAL],
        ids=["coherence", "empathy", "commonsense", "multicultural"],
    )
    def test_every_domain_has_at_least_2_characterizing_dims(self, config) -> None:
        char_dims = config.characterizing_dims
        assert len(char_dims) >= 2, (
            f"{config.name.value} has {len(char_dims)} characterizing dims, "
            f"expected at least 2: {char_dims}"
        )

    def test_all_prefixes_unique(self) -> None:
        prefixes = [config.prefix for config in DOMAINS.values()]
        assert len(prefixes) == len(set(prefixes)), (
            f"Duplicate prefixes found: {prefixes}"
        )

    def test_all_prefixes_are_2_chars(self) -> None:
        for domain_name, config in DOMAINS.items():
            assert len(config.prefix) == 2, (
                f"{domain_name.value} prefix '{config.prefix}' is not 2 chars"
            )

    def test_all_prefixed_dimensions_returns_23(self) -> None:
        dims = all_prefixed_dimensions()
        assert len(dims) == 23, (
            f"Expected 23 total prefixed dimensions, got {len(dims)}: {dims}"
        )

    def test_all_prefixed_dimensions_are_unique(self) -> None:
        dims = all_prefixed_dimensions()
        assert len(dims) == len(set(dims)), "Duplicate prefixed dimensions found"

    def test_all_prefixed_dimensions_have_correct_prefix(self) -> None:
        for config in DOMAINS.values():
            for prefixed in config.prefixed_dim_names:
                assert prefixed.startswith(f"{config.prefix}_"), (
                    f"Dimension '{prefixed}' does not start with '{config.prefix}_'"
                )


class TestCSRelationToDim:
    """Tests for commonsense relation-to-dimension mapping."""

    def test_covers_7_relations(self) -> None:
        assert len(CS_RELATION_TO_DIM) == 7

    def test_expected_relations(self) -> None:
        expected = {
            "HinderedBy",
            "IsAfter",
            "xAttr",
            "xReact",
            "oReact",
            "xWant",
            "oWant",
        }
        assert set(CS_RELATION_TO_DIM.keys()) == expected

    def test_all_values_are_prefixed_cs_dims(self) -> None:
        for relation, dim in CS_RELATION_TO_DIM.items():
            assert dim.startswith("cs_"), (
                f"Relation '{relation}' maps to '{dim}' which is not cs-prefixed"
            )


class TestSplitRatios:
    """Tests for split ratio constants."""

    def test_ratios_sum_to_one(self) -> None:
        total = sum(SPLIT_RATIOS.values())
        assert total == pytest.approx(1.0), f"Split ratios sum to {total}, expected 1.0"

    def test_has_train_val_test(self) -> None:
        assert "train" in SPLIT_RATIOS
        assert "val" in SPLIT_RATIOS
        assert "test" in SPLIT_RATIOS

    def test_train_is_largest(self) -> None:
        assert SPLIT_RATIOS["train"] > SPLIT_RATIOS["val"]
        assert SPLIT_RATIOS["train"] > SPLIT_RATIOS["test"]


class TestGetDomain:
    """Tests for the get_domain() helper."""

    @pytest.mark.parametrize(
        "name",
        ["coherence", "empathy", "commonsense", "multicultural"],
    )
    def test_get_domain_all_four(self, name: str) -> None:
        config = get_domain(name)
        assert config.name == DomainName(name)

    def test_get_domain_unknown_raises_key_error(self) -> None:
        with pytest.raises((KeyError, ValueError)):
            get_domain("nonexistent_domain")

    def test_get_domain_returns_correct_prefix(self) -> None:
        assert get_domain("coherence").prefix == "co"
        assert get_domain("empathy").prefix == "em"
        assert get_domain("commonsense").prefix == "cs"
        assert get_domain("multicultural").prefix == "mu"
