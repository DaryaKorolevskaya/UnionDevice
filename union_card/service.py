"""Оркестрация: валидация → LLM → проверки → (повтор с замечаниями) → карточка."""

from __future__ import annotations

from typing import Any, List

from .checks import find_problems
from .config import Settings, settings as default_settings
from .llm import ClaudeGenerator, GeminiGenerator, Generator, MockGenerator, OpenAICompatGenerator
from .prompts import build_repair_prompt, build_user_prompt
from .render import to_markdown
from .schemas import NO_DATA, Characteristic, GenerateResponse, LlmDraft, ProductCard
from .validation import EXPECTED_SPECS, ValidatedInput, validate_product


def make_generator(s: Settings = default_settings) -> Generator:
    if s.provider in ("openrouter", "ollama"):
        return OpenAICompatGenerator(s)
    if s.provider == "gemini":
        return GeminiGenerator(s)
    if s.provider == "anthropic":
        return ClaudeGenerator(s)
    return MockGenerator()


def build_characteristics(v: ValidatedInput) -> List[Characteristic]:
    """Список характеристик собирается кодом, а не моделью: значения переносятся из входа 1:1,
    недостающие ожидаемые характеристики помечаются «данных нет». Выдумать тут нечего."""
    cap = lambda s: s[:1].upper() + s[1:]
    items = [Characteristic(name=cap(k), value=val, source="input") for k, val in v.product.specs.items()]
    # характеристики, переданные во входе без значения
    items += [Characteristic(name=cap(k), value=NO_DATA, source="missing") for k in v.empty_specs]
    # ожидаемые для КИП характеристики, которых во входе нет вовсе
    items += [Characteristic(name=m, value=NO_DATA, source="missing") for m in v.missing_specs if m in EXPECTED_SPECS]
    return items


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

    card = ProductCard(
        product_name=v.product.name,
        manufacturer=v.product.manufacturer or NO_DATA,
        seo_title=draft.seo_title,
        seo_description=draft.seo_description,
        characteristics=build_characteristics(v),
        search_queries=draft.search_queries,
        commercial=draft.commercial,
        missing_fields=v.missing_fields + [f"specs → {m}" for m in v.missing_specs],
        clarifications=draft.clarifications,
        warnings=warnings,
        generator=gen.name,
        attempts=attempt,
    )
    return GenerateResponse(card=card, markdown=to_markdown(card))
