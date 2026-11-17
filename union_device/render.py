"""Markdown-представление карточки — удобно копировать в CMS или отправлять на согласование."""

from __future__ import annotations

from .schemas import ProductCard


def to_markdown(c: ProductCard) -> str:
    lines = [
        f"# {c.product_name}",
        "",
        f"**SEO-заголовок** ({len(c.seo_title)} симв.): {c.seo_title}",
        "",
        f"## SEO-описание ({len(c.seo_description)} симв.)",
        "",
        c.seo_description,
        "",
        "## Ключевые характеристики",
        "",
        "| Характеристика | Значение |",
        "|---|---|",
        f"| Производитель | {c.manufacturer} |",
        *[
            f"| {ch.name} | {ch.value}{' _(из описания, проверьте)_' if ch.source == 'description' else ''} |"
            for ch in c.characteristics
        ],
        "",
        "## Поисковые запросы",
        "",
        *[f"- {q}" for q in c.search_queries],
        "",
        "## Для менеджера",
        "",
        "**Кому подходит:**",
        *[f"- {a}" for a in c.commercial.target_audience],
        "",
        "**Какие задачи решает:**",
        *[f"- {t}" for t in c.commercial.tasks],
        "",
        f"**Как предложить:** {c.commercial.pitch}",
    ]
    if c.missing_fields or c.clarifications:
        lines += ["", "## Требует уточнения", ""]
        lines += [f"- Нет данных: {m}" for m in c.missing_fields]
        lines += [f"- {q}" for q in c.clarifications]
    if c.warnings:
        lines += ["", "## Предупреждения", ""] + [f"- ⚠ {w}" for w in c.warnings]
    lines += ["", f"_Сгенерировано: {c.generator}, попыток: {c.attempts}. Черновик — проверьте перед публикацией._", ""]
    return "\n".join(lines)
