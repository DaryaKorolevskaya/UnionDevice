"""Проверки ответа модели: длины полей и «выдуманные» числа."""

from __future__ import annotations

import re
from typing import List, Set

from .schemas import LlmDraft, ProductInput

TITLE_MAX = 70
DESC_MIN, DESC_MAX = 800, 1200
QUERIES_MIN, QUERIES_MAX = 3, 5

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Числа, которые допустимы в тексте и без входных данных (порядковые слова, «24/7» и т.п. тут не нужны).
_ALWAYS_OK = {"1", "2", "3", "4", "5"}


def _numbers(text: str) -> Set[str]:
    return {m.replace(",", ".").lstrip("0") or "0" for m in _NUM_RE.findall(text)}


def source_numbers(product: ProductInput) -> Set[str]:
    blob = " ".join([product.name, product.manufacturer, product.description, *product.specs.keys(), *product.specs.values()])
    return _numbers(blob)


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
    invented = sorted(_numbers(generated) - allowed)
    if invented:
        problems.append(
            "В тексте есть числа, которых нет во входных данных: " + ", ".join(invented) + "."
        )
    return problems
