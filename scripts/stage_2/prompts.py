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
{score_context}
INSTRUCTIONS:
- Write exactly {turn_count} turn(s) as the improved continuation.
- Keep the same speaker roles and general topic.
- Make the continuation clearly better on every dimension above.
- Output ONLY the rewritten continuation turns as JSON:

{{"continuation": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
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
{score_context}{non_char_constraint}
INSTRUCTIONS:
- Write exactly {turn_count} turn(s) as the degraded continuation.
- Keep the same speaker roles and general topic.
- Make the continuation clearly worse on every dimension above.
- The result should still be a plausible dialogue, not gibberish.
- The degradation should be subtle and natural-sounding, not obvious. \
A reasonable reader should need to think carefully to identify why the continuation is worse.
- Avoid abrupt non-sequiturs, obviously rude responses, or incoherent text.
- Output ONLY the rewritten continuation turns as JSON:

{{"continuation": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
"""

DIMENSION_TARGETED_TEMPLATE = """\
You are an expert dialogue writer specializing in {domain} quality.

Below is a multi-turn dialogue. Your task is to specifically \
{direction_verb} the following dimensions: {target_dims_str}, \
while keeping other aspects similar.
{non_target_constraint}
SHARED PREFIX (do not modify):
{prefix_text}

ORIGINAL CONTINUATION (to modify):
{continuation_text}

TARGET DIMENSIONS to {direction_verb}:
{rubric}
{score_context}
INSTRUCTIONS:
- Write exactly {turn_count} turn(s) as the modified continuation.
- Keep the same speaker roles and general topic.
- Focus changes on the target dimensions listed above.
{subtlety_constraint}\
- Output ONLY the rewritten continuation turns as JSON:

{{"continuation": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
"""


# ── Eval template ─────────────────────────────────────────────────────────

EVAL_TEMPLATE = """\
You are an expert dialogue evaluator for {domain} quality.

Below is a shared dialogue prefix followed by a continuation. \
Score the continuation on all dimensions listed below.

SHARED PREFIX:
{prefix_text}

CONTINUATION:
{continuation_text}

DIMENSIONS TO SCORE (0.0 to 1.0 each):
{rubric}

SCORING ANCHORS:
0.0-0.2: Severely deficient — criterion almost entirely absent
0.3-0.4: Weak — criterion present but poorly executed
0.5-0.6: Adequate — criterion meets basic expectations
0.7-0.8: Good — criterion clearly and consistently satisfied
0.9-1.0: Excellent — criterion exemplary, difficult to improve

INSTRUCTIONS:
- Before scoring, consider how the continuation responds to and builds upon \
the shared prefix. Score each dimension considering both the prefix and the \
continuation together.
- Use the full 0.0-1.0 range.
- Output ONLY valid JSON:

{{"scores": {{{score_keys}}}}}
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
            "- Use causal language but connect wrong events\n"
            "- Introduce slightly implausible prerequisites\n"
            "- Make causal chains subtly off so consequences feel questionable"
        ),
    },
    "cs_consistency": {
        "improve": (
            "- Maintain character attributes and knowledge across turns\n"
            "- Keep world state coherent with established facts\n"
            "- Ensure actions align with the speaker's apparent personality and role"
        ),
        "degrade": (
            "- Introduce subtle inconsistencies that require careful reading to notice\n"
            "- Slightly bend established facts about the situation\n"
            "- Have speakers act in ways that feel slightly off for their character"
        ),
    },
    "cs_reaction": {
        "improve": (
            "- Make emotional reactions appropriate to the events described\n"
            "- Include realistic affective responses from both speakers\n"
            "- Match reaction intensity to the significance of the event"
        ),
        "degrade": (
            "- Make emotional reactions slightly off — a bit too muted or intense\n"
            "- Underplay or slightly miss expected emotional responses\n"
            "- Have reaction intensity subtly mismatched with event significance"
        ),
    },
    "cs_desire": {
        "improve": (
            "- Make motivations and goals clear and realistic given the context\n"
            "- Align expressed desires with the speaker's situation and character\n"
            "- Ensure actions logically follow from stated intentions"
        ),
        "degrade": (
            "- Make motivations slightly misaligned with the situation\n"
            "- Have goals that are plausible but not quite right for the character\n"
            "- Make actions that loosely follow intentions but feel slightly off"
        ),
    },
}


# ── Coherence generation guidance ────────────────────────────────────────

COHERENCE_GENERATION_GUIDANCE: dict[str, dict[str, str]] = {
    "co_topic_coherence": {
        "improve": (
            "- Keep all turns on the same topic thread without introducing new subjects\n"
            "- Use referential links back to earlier turns (e.g., 'As you mentioned...')\n"
            "- Transition between sub-topics with explicit connective language"
        ),
        "degrade": (
            "- Introduce slight topic drift that feels natural but loses the thread\n"
            "- Reduce explicit references to earlier turns\n"
            "- Let sub-topic transitions happen without clear connective language"
        ),
    },
    "co_logical_consistency": {
        "improve": (
            "- Ensure all claims are compatible with previously stated facts\n"
            "- Make reasoning steps explicit and logically sound\n"
            "- Resolve any apparent contradictions with clarifying context"
        ),
        "degrade": (
            "- Introduce subtle factual tensions with earlier statements\n"
            "- Leave reasoning gaps that require inference to notice\n"
            "- Allow minor contradictions that a careful reader would catch"
        ),
    },
}


# ── Empathy generation guidance ─────────────────────────────────────────

EMPATHY_GENERATION_GUIDANCE: dict[str, dict[str, str]] = {
    "em_emotional_awareness": {
        "improve": (
            "- Name the speaker's emotion explicitly in the response\n"
            "- Acknowledge both the event and its emotional impact\n"
            "- Avoid immediately pivoting to advice before acknowledging the feeling"
        ),
        "degrade": (
            "- Acknowledge the situation but miss the underlying emotion slightly\n"
            "- Move to practical advice a bit too quickly\n"
            "- Use slightly generic emotional language instead of specific recognition"
        ),
    },
    "em_perspective_taking": {
        "improve": (
            "- Show understanding of why the speaker feels the way they do\n"
            "- Reference the speaker's specific circumstances, not generic advice\n"
            "- Demonstrate that you can see the situation from their point of view"
        ),
        "degrade": (
            "- Respond with slightly generic understanding that could apply to anyone\n"
            "- Miss subtle aspects of the speaker's specific situation\n"
            "- Offer perspective that is reasonable but not quite tuned to their experience"
        ),
    },
}


DOMAIN_GENERATION_GUIDANCE: dict[DomainName, dict[str, dict[str, str]]] = {
    DomainName.COMMONSENSE: COMMONSENSE_GENERATION_GUIDANCE,
    DomainName.COHERENCE: COHERENCE_GENERATION_GUIDANCE,
    DomainName.EMPATHY: EMPATHY_GENERATION_GUIDANCE,
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
rewrite {direction_qualifier} culturally grounded, specific, and authentic.
"""

MULTICULTURAL_VERIFICATION_CONTEXT = """\

CULTURAL CONTEXT:
This is a cross-cultural dialogue between speakers from {country_1} and {country_2}.

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


def _build_score_context(
    dims: list[str],
    scores: dict[str, float],
    direction: ContrastiveDirection,
) -> str:
    """Build CURRENT SCORES block with direction-aware magnitude guidance."""
    is_improve = direction == ContrastiveDirection.POSITIVE
    lines = []
    for dim in dims:
        if dim not in scores:
            continue
        score = scores[dim]
        if is_improve:
            if score <= 0.3:
                mag = "The current score is low. Make a substantial change."
            elif score <= 0.6:
                mag = "The current score is moderate. Make a clear improvement."
            else:
                mag = "The current score is already high. Make a refined, subtle improvement."
        else:
            if score >= 0.7:
                mag = "The current score is high. Make a substantial change."
            elif score >= 0.4:
                mag = "The current score is moderate. Make a clear degradation."
            else:
                mag = "The current score is already low. Make a minimal, subtle change."
        lines.append(f"  {dim} [current score: {score:.2f}] — {mag}")
    if not lines:
        return ""
    return "\nCURRENT SCORES:\n" + "\n".join(lines)


def _build_score_keys(config: DomainConfig) -> str:
    keys = []
    for dim_def in config.dimensions:
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

    # ── Non-characterizing dim constraint for global degrade ──
    non_char_constraint = ""
    if candidate.variant_type == VariantType.GLOBAL_DEGRADE:
        non_char_dims = [
            f"{config.prefix}_{d.name}"
            for d in config.dimensions
            if not d.is_characterizing
            and f"{config.prefix}_{d.name}" not in candidate.target_dimensions
        ]
        if non_char_dims:
            non_char_constraint = (
                "\nDIMENSIONS NOT LISTED ABOVE — keep these at their current level:\n- "
                + ", ".join(non_char_dims)
            )

    # ── Score context (all variant types) ──
    score_context = _build_score_context(
        candidate.target_dimensions,
        candidate.characterizing_scores,
        candidate.contrastive_direction,
    )

    # ── Dimension-targeted enrichments ──
    non_target_constraint = ""
    subtlety_constraint = ""

    if candidate.variant_type == VariantType.DIMENSION_TARGETED:
        # non-target dims with descriptions
        non_target_dims = [
            d for d in config.characterizing_dims
            if d not in candidate.target_dimensions
        ]
        if non_target_dims:
            lines = []
            for dim_def in config.dimensions:
                prefixed = f"{config.prefix}_{dim_def.name}"
                if prefixed in non_target_dims:
                    lines.append(f"  - {prefixed}: {dim_def.description}")
            non_target_constraint = (
                "\nNON-TARGET DIMENSIONS — do NOT change these:\n"
                + "\n".join(lines)
            )

        # subtlety for degrade
        if candidate.contrastive_direction == ContrastiveDirection.NEGATIVE:
            subtlety_constraint = (
                "- The degradation should be subtle and natural-sounding, not obvious. "
                "A reasonable reader should need to think carefully to identify why "
                "the continuation is worse.\n"
                "- Avoid abrupt non-sequiturs, obviously rude responses, "
                "or incoherent text.\n"
            )

    template_map = {
        VariantType.GLOBAL_IMPROVE: GLOBAL_IMPROVE_TEMPLATE,
        VariantType.GLOBAL_DEGRADE: GLOBAL_DEGRADE_TEMPLATE,
        VariantType.DIMENSION_TARGETED: DIMENSION_TARGETED_TEMPLATE,
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
        non_target_constraint=non_target_constraint,
        subtlety_constraint=subtlety_constraint,
        score_context=score_context,
        non_char_constraint=non_char_constraint,
    )

    # inject generation guidance for dimension-targeted variants
    domain_guidance = DOMAIN_GENERATION_GUIDANCE.get(config.name)
    if (
        domain_guidance
        and candidate.variant_type == VariantType.DIMENSION_TARGETED
        and candidate.target_dimensions
    ):
        target_dim = candidate.target_dimensions[0]
        guidance_entry = domain_guidance.get(target_dim)
        if guidance_entry:
            guidance_text = guidance_entry.get(direction_verb, "")
            if guidance_text:
                prompt += (
                    f"\n\nGENERATION GUIDANCE for {target_dim}:\n{guidance_text}\n"
                )

    # inject multicultural generation context (before instructions)
    if (
        config.name == DomainName.MULTICULTURAL
        and domain_metadata
    ):
        direction_qualifier = "more" if direction_verb == "improve" else "less"
        cultural_context = MULTICULTURAL_GENERATION_CONTEXT.format(
            direction_verb_ing=direction_verb_ing,
            direction_qualifier=direction_qualifier,
            **domain_metadata,
        )
        # Insert before INSTRUCTIONS block
        instructions_marker = "\nINSTRUCTIONS:"
        idx = prompt.find(instructions_marker)
        if idx != -1:
            prompt = prompt[:idx] + cultural_context + prompt[idx:]
        else:
            prompt = prompt + cultural_context

    return prompt


def build_eval_prompt(
    prefix_messages: list[Message],
    variant_continuation: list[Message],
    config: DomainConfig,
    domain_metadata: dict[str, str] | None = None,
) -> str:
    """build eval prompt that scores the variant continuation only.

    Stage 1 scores serve as ground truth for the original, so only
    the variant needs fresh scoring.
    """
    prefix_text = _format_messages(prefix_messages)
    continuation_text = _format_messages(variant_continuation)
    rubric = _build_rubric(config, config.prefixed_dim_names)
    score_keys = _build_score_keys(config)

    prompt = EVAL_TEMPLATE.format(
        domain=config.name.value,
        prefix_text=prefix_text,
        continuation_text=continuation_text,
        rubric=rubric,
        score_keys=score_keys,
    )

    # inject multicultural eval context (before instructions)
    if (
        config.name == DomainName.MULTICULTURAL
        and domain_metadata
    ):
        cultural_context = MULTICULTURAL_VERIFICATION_CONTEXT.format(**domain_metadata)
        instructions_marker = "\nINSTRUCTIONS:"
        idx = prompt.find(instructions_marker)
        if idx != -1:
            prompt = prompt[:idx] + cultural_context + prompt[idx:]
        else:
            prompt = prompt + cultural_context

    return prompt
