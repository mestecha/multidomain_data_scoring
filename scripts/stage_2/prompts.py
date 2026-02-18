"""stage 2 generation and evaluation prompt templates."""

from __future__ import annotations

from scripts.models import (
    ContrastiveDirection,
    DomainConfig,
    DomainName,
    Message,
    Stage2Candidate,
    VariantType,
)


# ── Generation templates ──────────────────────────────────────────────────

GLOBAL_IMPROVE_TEMPLATE = """\
You are an expert dialogue writer specializing in {domain} quality.

Below is a multi-turn dialogue. Your task is to rewrite the continuation \
to be a stronger example of {domain} quality across all dimensions.

SHARED PREFIX (do not modify):
{prefix_text}

ORIGINAL CONTINUATION (to improve):
{continuation_text}

SCORING RUBRIC - improve ALL of these dimensions:
{rubric}

INSTRUCTIONS:
- Write exactly {turn_count} turn(s) as the improved continuation.
- Keep the same speaker roles and general topic.
- Make the continuation clearly better on every dimension above.
- Output ONLY the rewritten continuation turns as JSON:

{{"continuation": [{{"role": "user"|"assistant", "content": "..."}}]}}
"""

GLOBAL_DEGRADE_TEMPLATE = """\
You are an expert dialogue writer specializing in {domain} quality.

Below is a multi-turn dialogue. Your task is to rewrite the continuation \
to be weaker on {domain} quality across all dimensions.

SHARED PREFIX (do not modify):
{prefix_text}

ORIGINAL CONTINUATION (to degrade):
{continuation_text}

SCORING RUBRIC - weaken ALL of these dimensions:
{rubric}

INSTRUCTIONS:
- Write exactly {turn_count} turn(s) as the degraded continuation.
- Keep the same speaker roles and general topic.
- Make the continuation clearly worse on every dimension above.
- The result should still be a plausible dialogue, not gibberish.
- Output ONLY the rewritten continuation turns as JSON:

{{"continuation": [{{"role": "user"|"assistant", "content": "..."}}]}}
"""

DIMENSION_TARGETED_TEMPLATE = """\
You are an expert dialogue writer specializing in {domain} quality.

Below is a multi-turn dialogue. Your task is to specifically \
{direction_verb} the following dimensions: {target_dims_str}, \
while keeping other aspects similar.

SHARED PREFIX (do not modify):
{prefix_text}

ORIGINAL CONTINUATION (to modify):
{continuation_text}

TARGET DIMENSIONS to {direction_verb}:
{rubric}

INSTRUCTIONS:
- Write exactly {turn_count} turn(s) as the modified continuation.
- Keep the same speaker roles and general topic.
- Focus changes on the target dimensions listed above.
- Output ONLY the rewritten continuation turns as JSON:

{{"continuation": [{{"role": "user"|"assistant", "content": "..."}}]}}
"""

MULTI_DIMENSIONAL_TEMPLATE = """\
You are an expert dialogue writer specializing in {domain} quality.

Below is a multi-turn dialogue. Your task is to {direction_verb} \
multiple dimensions: {target_dims_str}.

SHARED PREFIX (do not modify):
{prefix_text}

ORIGINAL CONTINUATION (to modify):
{continuation_text}

DIMENSIONS to {direction_verb}:
{rubric}

INSTRUCTIONS:
- Write exactly {turn_count} turn(s) as the modified continuation.
- Keep the same speaker roles and general topic.
- {direction_verb_cap} the specified dimensions simultaneously.
- Output ONLY the rewritten continuation turns as JSON:

{{"continuation": [{{"role": "user"|"assistant", "content": "..."}}]}}
"""


# ── Verification template ─────────────────────────────────────────────────

VERIFICATION_TEMPLATE = """\
You are an expert dialogue evaluator for {domain} quality.

Below is a shared dialogue prefix followed by two continuations. \
Score BOTH continuations on the characterizing dimensions listed below.

SHARED PREFIX:
{prefix_text}

CONTINUATION A (original):
{original_text}

CONTINUATION B (variant):
{variant_text}

DIMENSIONS TO SCORE (0.0 to 1.0 each):
{rubric}

INSTRUCTIONS:
- Score each continuation independently on each dimension.
- Use the full 0.0-1.0 range.
- Output ONLY valid JSON:

{{"original_scores": {{{score_keys}}}, "variant_scores": {{{score_keys}}}}}
"""


# ── Commonsense generation guidance ──────────────────────────────────────

COMMONSENSE_GENERATION_GUIDANCE: dict[str, dict[str, str]] = {
    "cs_causality": {
        "improve": (
            "- Ensure clear cause-effect relationships between events\n"
            "- Use temporal markers (before, after, because, therefore)\n"
            "- Make prerequisites and consequences explicit and logical"
        ),
        "degrade": (
            "- Violate temporal ordering (effect before cause)\n"
            "- Introduce impossible prerequisites\n"
            "- Break causal chains so consequences don't follow from actions"
        ),
    },
    "cs_consistency": {
        "improve": (
            "- Maintain character attributes and knowledge across turns\n"
            "- Keep world state coherent with established facts\n"
            "- Ensure actions align with the speaker's apparent personality and role"
        ),
        "degrade": (
            "- Introduce character inconsistencies (contradict earlier statements)\n"
            "- Violate established facts about the situation\n"
            "- Have speakers act against their established personality or knowledge"
        ),
    },
    "cs_reaction": {
        "improve": (
            "- Make emotional reactions appropriate to the events described\n"
            "- Include realistic affective responses from both speakers\n"
            "- Match reaction intensity to the significance of the event"
        ),
        "degrade": (
            "- Make emotional reactions inappropriate (laugh at tragedy, ignore good news)\n"
            "- Omit expected emotional responses to significant events\n"
            "- Mismatch reaction intensity with event significance"
        ),
    },
    "cs_desire": {
        "improve": (
            "- Make motivations and goals clear and realistic given the context\n"
            "- Align expressed desires with the speaker's situation and character\n"
            "- Ensure actions logically follow from stated intentions"
        ),
        "degrade": (
            "- Introduce illogical or contradictory motivations\n"
            "- Have goals misaligned with what the character would realistically want\n"
            "- Make actions that don't follow from any stated or implied intention"
        ),
    },
}


# ── Multicultural context templates ───────────────────────────────────────

MULTICULTURAL_GENERATION_CONTEXT = """\

CULTURAL CONTEXT:
This dialogue takes place between speakers from {country_1} and {country_2}.

Cultural value being explored: "{statement_original}"
Culturally adapted perspective: "{statement_cultural}"

Situation: {situation}

SPEAKER 1 ({country_1}):
{demographics_1}
Cultural perspective: {cultural_reasoning_1}
Relevant social norms: {social_norms_1}

SPEAKER 2 ({country_2}):
{demographics_2}
Cultural perspective: {cultural_reasoning_2}
Relevant social norms: {social_norms_2}

CROSS-CULTURAL DYNAMICS:
How {country_1} may perceive {country_2}: {prejudices_1}
How {country_2} may perceive {country_1}: {prejudices_2}
Emotional dynamics: {arousal_reasoning}

When {direction_verb_ing} the continuation, use these cultural elements to make your \
rewrite more (or less) culturally grounded, specific, and authentic.
"""

MULTICULTURAL_VERIFICATION_CONTEXT = """\

CULTURAL CONTEXT:
This is a cross-cultural dialogue between speakers from {country_1} and {country_2}.
Cultural value statement: "{statement_original}"

Speaker 1: {demographics_1}
Speaker 2: {demographics_2}

Score cultural dimensions with these specific cultural backgrounds in mind.
"""


# ── Helpers ───────────────────────────────────────────────────────────────


def _format_messages(messages: list[Message]) -> str:
    lines = []
    for msg in messages:
        role_label = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role_label}: {msg.content}")
    return "\n".join(lines)


def _build_rubric(config: DomainConfig, target_dims: list[str]) -> str:
    lines = []
    for dim_def in config.dimensions:
        prefixed = f"{config.prefix}_{dim_def.name}"
        if prefixed in target_dims:
            lines.append(f"- {prefixed}: {dim_def.description}")
    return "\n".join(lines) if lines else "(no specific dimensions)"


def _build_score_keys(config: DomainConfig) -> str:
    keys = []
    for dim_def in config.dimensions:
        if dim_def.is_characterizing:
            prefixed = f"{config.prefix}_{dim_def.name}"
            keys.append(f'"{prefixed}": <float>')
    return ", ".join(keys)


# ── Public API ────────────────────────────────────────────────────────────


def build_generation_prompt(
    candidate: Stage2Candidate,
    turn_count: int,
    config: DomainConfig,
    domain_metadata: dict[str, str] | None = None,
) -> str:
    """select template by variant type and fill in prefix, continuation, and rubric."""
    # split into prefix + continuation (last `turn_count` messages)
    all_msgs = candidate.messages
    if len(all_msgs) > turn_count:
        prefix_msgs = all_msgs[:-turn_count]
        continuation_msgs = all_msgs[-turn_count:]
    else:
        prefix_msgs = all_msgs[:1]
        continuation_msgs = all_msgs[1:] if len(all_msgs) > 1 else all_msgs

    prefix_text = _format_messages(prefix_msgs)
    continuation_text = _format_messages(continuation_msgs)
    rubric = _build_rubric(config, candidate.target_dimensions)

    direction_verb = (
        "improve"
        if candidate.contrastive_direction == ContrastiveDirection.POSITIVE
        else "degrade"
    )
    direction_verb_cap = direction_verb.capitalize()
    direction_verb_ing = "improving" if direction_verb == "improve" else "degrading"
    target_dims_str = ", ".join(candidate.target_dimensions)

    template_map = {
        VariantType.GLOBAL_IMPROVE: GLOBAL_IMPROVE_TEMPLATE,
        VariantType.GLOBAL_DEGRADE: GLOBAL_DEGRADE_TEMPLATE,
        VariantType.DIMENSION_TARGETED: DIMENSION_TARGETED_TEMPLATE,
        VariantType.MULTI_DIMENSIONAL: MULTI_DIMENSIONAL_TEMPLATE,
    }

    template = template_map[candidate.variant_type]

    prompt = template.format(
        domain=config.name.value,
        prefix_text=prefix_text,
        continuation_text=continuation_text,
        rubric=rubric,
        turn_count=turn_count,
        direction_verb=direction_verb,
        direction_verb_cap=direction_verb_cap,
        target_dims_str=target_dims_str,
    )

    # inject commonsense generation guidance for single-dim targeting
    if (
        config.name == DomainName.COMMONSENSE
        and candidate.target_dimensions
    ):
        target_dim = candidate.target_dimensions[0]
        guidance_entry = COMMONSENSE_GENERATION_GUIDANCE.get(target_dim)
        if guidance_entry:
            guidance_text = guidance_entry.get(direction_verb, "")
            if guidance_text:
                prompt += (
                    f"\n\nGENERATION GUIDANCE for {target_dim}:\n{guidance_text}\n"
                )

    # inject multicultural generation context
    if (
        config.name == DomainName.MULTICULTURAL
        and domain_metadata
    ):
        cultural_context = MULTICULTURAL_GENERATION_CONTEXT.format(
            direction_verb_ing=direction_verb_ing,
            **domain_metadata,
        )
        prompt = prompt + cultural_context

    return prompt


def build_eval_prompt(
    original_messages: list[Message],
    original_continuation: list[Message],
    variant_continuation: list[Message],
    config: DomainConfig,
    domain_metadata: dict[str, str] | None = None,
) -> str:
    prefix_text = _format_messages(original_messages)
    original_text = _format_messages(original_continuation)
    variant_text = _format_messages(variant_continuation)
    rubric = _build_rubric(config, config.characterizing_dims)
    score_keys = _build_score_keys(config)

    prompt = VERIFICATION_TEMPLATE.format(
        domain=config.name.value,
        prefix_text=prefix_text,
        original_text=original_text,
        variant_text=variant_text,
        rubric=rubric,
        score_keys=score_keys,
    )

    # inject multicultural eval context
    if (
        config.name == DomainName.MULTICULTURAL
        and domain_metadata
    ):
        cultural_context = MULTICULTURAL_VERIFICATION_CONTEXT.format(**domain_metadata)
        prompt = prompt + cultural_context

    return prompt
