"""CLI: python -m union_device.cli data/products.json -o examples/

Принимает файл с одним товаром или списком товаров, сохраняет JSON и Markdown на каждый товар.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .config import settings
from .llm import LlmError
from .service import generate_card
from .validation import InputError, parse_json


_TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
               ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t",
                "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))


def _slug(s: str) -> str:
    s = "".join(_TR.get(ch, ch) for ch in s.lower())
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:60] or "product"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Генерация черновиков карточек товаров КИП")
    ap.add_argument("input", help="JSON-файл: объект товара или массив товаров")
    ap.add_argument("-o", "--out", default="examples", help="папка для результатов")
    args = ap.parse_args(argv)

    try:
        data = parse_json(Path(args.input).read_bytes())
    except (OSError, InputError) as e:
        print(f"Ошибка чтения входа: {e} {getattr(e, 'details', '')}", file=sys.stderr)
        return 2

    items = data if isinstance(data, list) else [data]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Режим: {'mock (без LLM)' if settings.mock else settings.model}")

    failed = 0
    for i, item in enumerate(items, 1):
        label = item.get("name") if isinstance(item, dict) else None
        try:
            res = generate_card(item)
        except InputError as e:
            failed += 1
            print(f"[{i}] {label or '—'}: входные данные отклонены — {e} {e.details}", file=sys.stderr)
            continue
        except LlmError as e:
            failed += 1
            print(f"[{i}] {label or '—'}: ошибка LLM — {e}", file=sys.stderr)
            continue
        stem = out / f"{i:02d}_{_slug(res.card.product_name)}"
        stem.with_suffix(".json").write_text(
            json.dumps(res.card.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        stem.with_suffix(".md").write_text(res.markdown, encoding="utf-8")
        c = res.card
        print(
            f"[{i}] {c.product_name}: title {len(c.seo_title)} симв., описание {len(c.seo_description)} симв., "
            f"попыток {c.attempts}, предупреждений {len(c.warnings)} → {stem}.md"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
