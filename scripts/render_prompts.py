"""Render all prompt variants for visual review."""

from __future__ import annotations

from pathlib import Path

from scripts.config import DOMAINS
from scripts.models import (
    ContrastiveDirection,
    DomainName,
    Message,
    Stage2Candidate,
    VariantType,
)
from scripts.stage_2.prompts import build_eval_prompt, build_generation_prompt

OUTPUT = Path("data/all_prompts_review.txt")

SAMPLE_MESSAGES = [
    Message(role="user", content="I've been thinking about changing careers."),
    Message(role="assistant", content="That's a big decision. What's motivating the change?"),
    Message(role="user", content="I feel stuck and want something more meaningful."),
    Message(role="assistant", content="I understand that feeling. Have you explored what fields interest you?"),
    Message(role="user", content="I've been looking into teaching, actually."),
]

MULTICULTURAL_META = {
    "country_1": "Japan",
    "country_2": "Brazil",
    "demographics_1": "Age 45, female, Tokyo",
    "demographics_2": "Age 30, male, São Paulo",
    "statement_original": "Respect for hierarchy in professional settings",
    "statement_cultural": "Age-based deference in workplace interactions",
    "situation": "Planning a cross-cultural team project",
    "cultural_reasoning_1": "Seniority and formal address are valued",
    "cultural_reasoning_2": "Warmth and informality build trust",
    "arousal_reasoning": "Potential tension between formality expectations",
    "arousal_score": "4",
    "social_norms_1": "Use honorifics and bow",
    "social_norms_2": "First-name basis and physical warmth",
    "prejudices_1": "Brazilians are too casual",
    "prejudices_2": "Japanese are too rigid",
}


def main() -> None:
    lines: list[str] = []
    prompt_num = 0

    for domain_name, config in DOMAINS.items():
        meta = MULTICULTURAL_META if domain_name == DomainName.MULTICULTURAL else None
        char_dims = config.characterizing_dims

        # Determine variant combos
        combos: list[tuple[VariantType, ContrastiveDirection, list[str]]] = []

        # Global improve (all char dims)
        combos.append((VariantType.GLOBAL_IMPROVE, ContrastiveDirection.POSITIVE, char_dims))
        # Global degrade (all char dims)
        combos.append((VariantType.GLOBAL_DEGRADE, ContrastiveDirection.NEGATIVE, char_dims))
        # Targeted improve per dim
        for dim in char_dims:
            combos.append((VariantType.DIMENSION_TARGETED, ContrastiveDirection.POSITIVE, [dim]))
        # Targeted degrade per dim
        for dim in char_dims:
            combos.append((VariantType.DIMENSION_TARGETED, ContrastiveDirection.NEGATIVE, [dim]))

        for vt, direction, target_dims in combos:
            prompt_num += 1
            scores = {}
            for d in char_dims:
                scores[d] = 0.25 if d == target_dims[0] else 0.55
            if len(target_dims) > 1:
                scores = {d: 0.45 for d in char_dims}

            candidate = Stage2Candidate(
                dialogue_id=f"render-{prompt_num:03d}",
                domain=domain_name,
                messages=SAMPLE_MESSAGES,
                characterizing_scores=scores,
                target_dimensions=target_dims,
                contrastive_direction=direction,
                variant_type=vt,
                domain_metadata=meta,
            )

            prompt = build_generation_prompt(
                candidate, turn_count=3, config=config, domain_metadata=meta,
            )

            header = (
                f"{'=' * 80}\n"
                f"PROMPT #{prompt_num}: {domain_name.value} | {vt.value} | "
                f"{direction.value} | targets={target_dims}\n"
                f"{'=' * 80}"
            )
            lines.append(header)
            lines.append(prompt)
            lines.append("")

        # One eval prompt per domain
        prompt_num += 1
        eval_prompt = build_eval_prompt(
            prefix_messages=SAMPLE_MESSAGES[:2],
            variant_continuation=SAMPLE_MESSAGES[2:],
            config=config,
            domain_metadata=meta,
        )
        header = (
            f"{'=' * 80}\n"
            f"PROMPT #{prompt_num}: {domain_name.value} | EVAL\n"
            f"{'=' * 80}"
        )
        lines.append(header)
        lines.append(eval_prompt)
        lines.append("")

    OUTPUT.write_text("\n".join(lines))
    print(f"Wrote {prompt_num} prompts to {OUTPUT}")


if __name__ == "__main__":
    main()
