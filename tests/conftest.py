"""Shared fixtures for Stage 1 test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def sample_coherence_dialogue() -> dict[str, str]:
    """A coherence dialogue item with multi-line content and a label."""
    return {
        "content": (
            "Human A: Hi, how are you doing today?\n"
            "Human B: I'm doing great, thanks for asking.\n"
            "Human A: That's wonderful. Did you see the news this morning?\n"
            "Human B: Yes, I saw the headlines. Pretty interesting stuff.\n"
            "Human A: I thought so too. The article about climate change was eye-opening.\n"
            "Human B: Absolutely, we should all be more conscious about it.\n"
        ),
        "label": "Yes",
    }


@pytest.fixture()
def sample_commonsense_row() -> dict[str, str]:
    """A dict simulating a TSV row for the commonsense domain."""
    return {
        "PID": "P001",
        "UID": "U001",
        "SID": "S001",
        "CTX": "user1: I went to the store [UTT] user2: Did you buy anything?",
        "SEG": "I bought some groceries for dinner tonight.",
        "HinderedBy_response": "The store could have been closed.",
        "IsAfter_response": "The speaker decided they needed food.",
        "xAttr_response": "The speaker is responsible and organized.",
        "xReact_response": "The speaker feels satisfied with the purchase.",
        "oReact_response": "The listener is curious about the groceries.",
        "xWant_response": "The speaker wants to cook a nice meal.",
        "oWant_response": "The listener wants to join for dinner.",
    }


@pytest.fixture()
def sample_multicultural_row() -> dict[str, str]:
    """A dict simulating a CSV row for the multicultural domain."""
    return {
        "conversation": (
            "SPK01-USA001: Hello, it's nice to meet you.\n"
            "SPK02-JPN001: Nice to meet you too. I appreciate you taking the time.\n"
            "SPK01-USA001: Of course! I wanted to learn more about your culture.\n"
            "SPK02-JPN001: I'd be happy to share. What would you like to know?\n"
        ),
        "statement_original": "Respect for elders is a core cultural value.",
        "country_1": "US",
        "country_2": "JP",
        "uid": "MC-12345",
    }


@pytest.fixture()
def sample_stage1_entry() -> dict:
    """A dict matching the Stage1Entry format."""
    return {
        "dialogue_id": "S1D-000001",
        "source_id": "co-abc123",
        "domain": "coherence",
        "messages": [
            {"role": "user", "content": "Hi, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"},
        ],
        "split": "train",
        "scores": {
            "co_topic_coherence": 0.75,
            "co_discourse_structure": 0.50,
            "co_logical_consistency": 0.80,
            "co_temporal_causal_coherence": 0.60,
            "co_mutual_grounding": 0.70,
            "co_overall_coherence_score": 0.65,
        },
    }


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a sample data directory structure under tmp_path."""
    input_dir = tmp_path / "data" / "input"
    for domain in ("coherence", "empathy", "commonsense", "multicultural"):
        domain_dir = input_dir / domain
        domain_dir.mkdir(parents=True)
        shards_dir = domain_dir / "shards"
        shards_dir.mkdir()

    # Create commonsense fabricated subdir
    (input_dir / "commonsense" / "fabricated").mkdir(parents=True, exist_ok=True)

    # Create multicultural countries subdir
    (input_dir / "multicultural" / "countries").mkdir(parents=True, exist_ok=True)

    return tmp_path
