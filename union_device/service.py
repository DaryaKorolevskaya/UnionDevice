"""Оркестрация: валидация → LLM → проверки → (повтор с замечаниями) → карточка."""

from __future__ import annotations

from typing import Any, List

from .checks import find_problems, verify_description_facts
from .config import Settings, settings as default_settings
from .llm import ClaudeGenerator, GeminiGenerator, Generator, MockGenerator, OpenAICompatGenerator
from .prompts import build_repair_prompt, build_user_prompt
from .render import to_markdown
from .schemas import NO_DATA, Characteristic, DescriptionFact, GenerateResponse, LlmDraft, ProductCard
from .validation import EXPECTED_SPECS, ValidatedInput, find_spec, validate_product


def make_generator(s: Settings = default_settings) -> Generator:
    if s.provider in ("openrouter", "ollama"):
        return OpenAICompatGenerator(s)
    if s.provider == "gemini":
        return GeminiGenerator(s)
    if s.provider == "anthropic":
        return ClaudeGenerator(s)
    return MockGenerator()


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def covered_labels(facts: List[DescriptionFact]) -> set:
    """Какие ожидаемые группы характеристик закрыты фактами из описания."""
    names = {f.name: f.value for f in facts}
    return {label for label in EXPECTED_SPECS if find_spec(names, label)}


def build_characteristics(v: ValidatedInput, facts: List[DescriptionFact]) -> List[Characteristic]:
    """Список характеристик собирается кодом, а не моделью: значения переносятся из входа 1:1,
    факты из описания — только с дословно найденной цитатой, остальное помечается «данных нет»."""
    items = [Characteristic(name=_cap(k), value=val, source="input") for k, val in v.product.specs.items()]
    items += [Characteristic(name=_cap(f.name), value=f.value, source="description") for f in facts]
    # характеристики, переданные во входе без значения
    items += [Characteristic(name=_cap(k), value=NO_DATA, source="missing") for k in v.empty_specs]
    # ожидаемые для КИП характеристики, которых нет ни в specs, ни в описании
    covered = covered_labels(facts)
    items += [
        Characteristic(name=m, value=NO_DATA, source="missing")
        for m in v.missing_specs
        if m in EXPECTED_SPECS and m not in covered
    ]
    return items


def dedupe_clarifications(clarifications: List[str], missing: List[str]) -> List[str]:
    """Убирает из вопросов модели то, что уже есть в списке отсутствующих полей."""
    seen = [m.lower().split("→")[-1].strip() for m in missing]
    out: List[str] = []
    for c in clarifications:
        low = c.lower()
        if any(m and (m in low or low.strip(" .") in m) for m in seen) or c in out:
            continue
        out.append(c)
    return out


def generate_card(data: Any, generator: Generator | None = None, max_attempts: int | None = None) -> GenerateResponse:
    v = validate_product(data)  # InputError → 422
    gen = generator or make_generator()
    attempts_limit = max_attempts or default_settings.max_attempts
    if isinstance(gen, MockGenerator):
        gen.product = v.product

    messages: List[dict] = [{"role": "user", "content": build_user_prompt(v)}]
    draft: LlmDraft | None = None
    problems: List[str] = []
    attempt = 0
    for attempt in range(1, attempts_limit + 1):
        draft = gen.generate(messages)  # LlmError → 4xx/5xx
        problems = find_problems(draft, v.product)
        if not problems:
            break
        messages += [
            {"role": "assistant", "content": draft.model_dump_json()},
            {"role": "user", "content": build_repair_prompt(problems)},
        ]

    assert draft is not None
    warnings = list(v.warnings)
    if problems:
        warnings += [f"Не исправлено автоматически: {p}" for p in problems]
    facts, rejected = verify_description_facts(draft.description_facts, v.product)
    warnings += [f"Факт из описания отброшен: {r}" for r in rejected]
    covered = covered_labels(facts)
    missing = v.missing_fields + [
        f"specs → {m}" for m in v.missing_specs if m not in covered
    ]

    card = ProductCard(
        product_name=v.product.name,
        manufacturer=v.product.manufacturer or NO_DATA,
        seo_title=draft.seo_title,
        seo_description=draft.seo_description,
        characteristics=build_characteristics(v, facts),
        search_queries=draft.search_queries,
        commercial=draft.commercial,
        missing_fields=missing,
        clarifications=dedupe_clarifications(draft.clarifications, missing),
        warnings=warnings,
        generator=gen.name,
        attempts=attempt,
    )
    return GenerateResponse(card=card, markdown=to_markdown(card))
