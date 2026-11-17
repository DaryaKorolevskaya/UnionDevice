"""Проверки ответа модели: длины полей и «выдуманные» числа."""

from __future__ import annotations

import re
from typing import List, Set, Tuple

from .schemas import DescriptionFact, LlmDraft, ProductInput

TITLE_MAX = 70
DESC_MIN, DESC_MAX = 800, 1200
QUERIES_MIN, QUERIES_MAX = 3, 5

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Числа, которые допустимы в тексте и без входных данных (порядковые слова, «24/7» и т.п. тут не нужны).
_ALWAYS_OK = {"1", "2", "3", "4", "5"}


# Оценки и факты, которые модель любит «добавить от себя». Ошибка, только если их нет во входных данных.
SUBJECTIVE_PATTERNS = {
    "высокая точность": r"высок\w* точност",
    "точный отсчёт/измерение": r"\bточн\w* (отсч[её]т|измерени|показани|контрол)",
    "надёжный": r"над[её]жн",
    "долговечный": r"долговечн",
    "лучший": r"\bлучш",
    "идеальный": r"идеальн",
    "гарантия": r"гарант",
    "сертификат": r"сертифи",
    "поверка": r"\bповер(к|ен|оч)",
    "взрывозащита": r"взрывозащ",
    "нержавеющая сталь": r"нержаве",
    "срок службы": r"срок\w* служб",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def _numbers(text: str) -> Set[str]:
    return {m.replace(",", ".").lstrip("0") or "0" for m in _NUM_RE.findall(text)}


def source_numbers(product: ProductInput) -> Set[str]:
    return _numbers(_source_text(product))


def _source_text(product: ProductInput) -> str:
    return " ".join([product.name, product.manufacturer, product.description, *product.specs.keys(), *product.specs.values()])


def find_problems(draft: LlmDraft, product: ProductInput) -> List[str]:
    problems: List[str] = []

    if len(draft.seo_title) > TITLE_MAX:
        problems.append(f"seo_title длиной {len(draft.seo_title)} символов, нужно не больше {TITLE_MAX}.")

    n = len(draft.seo_description)
    if not DESC_MIN <= n <= DESC_MAX:
        problems.append(f"seo_description длиной {n} символов, нужно {DESC_MIN}–{DESC_MAX}.")

    q = len(draft.search_queries)
    if not QUERIES_MIN <= q <= QUERIES_MAX:
        problems.append(f"search_queries содержит {q} запросов, нужно {QUERIES_MIN}–{QUERIES_MAX}.")

    allowed = source_numbers(product) | _ALWAYS_OK
    generated = " ".join(
        [
            draft.seo_title,
            draft.seo_description,
            *draft.search_queries,
            *draft.commercial.target_audience,
            *draft.commercial.tasks,
            draft.commercial.pitch,
        ]
    )
    source = _norm(_source_text(product))
    claims = [
        label
        for label, pattern in SUBJECTIVE_PATTERNS.items()
        if re.search(pattern, _norm(generated)) and not re.search(pattern, source)
    ]
    if claims:
        problems.append(
            "В тексте есть оценки или факты, которых нет во входных данных: " + ", ".join(claims)
            + ". Удали эти формулировки."
        )

    invented = sorted(_numbers(generated) - allowed)
    if invented:
        problems.append(
            "В тексте есть числа, которых нет во входных данных: " + ", ".join(invented) + "."
        )
    return problems


def verify_description_facts(
    facts: List[DescriptionFact], product: ProductInput
) -> Tuple[List[DescriptionFact], List[str]]:
    """Принимает только факты, чья цитата дословно есть в описании, а числа значения — в цитате."""
    desc = _norm(product.description)
    accepted: List[DescriptionFact] = []
    rejected: List[str] = []
    for f in facts:
        quote = _norm(f.quote).strip(" .,;«»\"")
        if not f.value.strip() or len(quote) < 5 or quote not in desc:
            rejected.append(f"«{f.name}»: цитата не найдена в описании")
        elif not _numbers(f.value) <= _numbers(f.quote):
            rejected.append(f"«{f.name}»: значение не совпадает с цитатой")
        else:
            accepted.append(f)
    return accepted, rejected
