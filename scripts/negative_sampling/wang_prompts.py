"""wang et al. self-taught evaluator prompt adapted to multicultural domain.

We DON'T tell the LLM to "make it bad". We ask it to invent a SLIGHTLY
DIFFERENT task and produce a high-quality response to that. The output is
high-quality for the modified task but BAD as a response to the original
(low on the target dim).
"""

from __future__ import annotations

# Per-dim modification direction (constrains what dimension the modification
# should push lower). Without these, the LLM might modify in any direction.

MOD_DIRECTIONS = {
    "cv": (
        "explores a DIFFERENT but related cultural value or theme "
        "(e.g., if the original is about religiosity, the modified instruction "
        "could be about generational respect; the modified value should be "
        "thematically adjacent but not the same value)."
    ),
    "cs": (
        "specifies the dialogue should take place in a CULTURALLY GENERIC "
        "setting (e.g., generic corporate meeting, academic conference, "
        "online customer support, no country-specific markers expected). "
        "The cultural value can still be discussed but without country-specific "
        "anchoring."
    ),
    "nat": (
        "requests a NON-CONVERSATIONAL format (e.g., a structured analytical "
        "exchange, a formal debate transcript, a written legal deposition, "
        "a corporate Q&A). The format itself prevents natural conversational "
        "flow."
    ),
    "coh": (
        "requests a NON-LINEAR FORMAT WHERE COHERENCE IS NOT EXPECTED "
        "(e.g., parallel monologues that don't address each other, fragmented "
        "exchange via asynchronous messages with topic interruptions, "
        "stream-of-consciousness alternation). The format itself prevents "
        "coherent flow."
    ),
}


WANG_TEMPLATE = """\
Below is a multicultural dialogue generation task between two cross-cultural speakers.

ORIGINAL INSTRUCTION (do not respond to this directly):
Generate a 6-10 turn cross-cultural dialogue between two speakers.

- Speaker 1 ({country_1}): {demographics_1}
  Cultural worldview: {cultural_reasoning_1}
  Relevant social norms: {social_norms_1}
  Assumptions about {country_2}: {prejudices_1}

- Speaker 2 ({country_2}): {demographics_2}
  Cultural worldview: {cultural_reasoning_2}
  Relevant social norms: {social_norms_2}
  Assumptions about {country_1}: {prejudices_2}

Cultural value being explored: "{statement_cultural}"
Situation: {situation}
Emotional dynamics: {arousal_reasoning} (arousal level {arousal_score}/5)

ORIGINAL HIGH-QUALITY RESPONSE (this is a good response to the ORIGINAL instruction):
{baseline_dialogue}

YOUR TASK:
First, generate a MODIFIED instruction that is highly relevant to the original but {mod_direction}

Then, produce a high-quality dialogue response that satisfies your MODIFIED instruction.

Critical:
- Your output dialogue should be of high quality FOR THE MODIFIED INSTRUCTION you wrote.
- It will NOT serve as a good response to the ORIGINAL instruction; that is the intended outcome.
- DO NOT explicitly write a "bad" or "low-quality" response. Write your honest best response to the modified instruction.
- The modified instruction must be specific and actionable; not just a vague rephrase.

Output format (strict JSON, no commentary):

{{
  "modified_instruction": "<1-3 sentences describing your modified instruction>",
  "modified_dialogue": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    ...
  ]
}}

The dialogue must be 6-10 turns alternating user/assistant starting with user.
"""


def build(target_dim: str, row: dict, baseline_dialogue: str) -> str:
    """render the wang-et-al style prompt for one row + target dim."""
    return WANG_TEMPLATE.format(
        mod_direction=MOD_DIRECTIONS[target_dim],
        baseline_dialogue=baseline_dialogue,
        **row,
    )
