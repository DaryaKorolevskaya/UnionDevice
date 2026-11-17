"""Список бесплатных текстовых моделей OpenRouter: python -m union_device.models

Список у OpenRouter меняется, поэтому он берётся из публичного API (ключ не нужен).
Выбранную модель укажите в .env: UNION_MODEL=<id> (можно несколько через запятую).
"""

from __future__ import annotations

import json
import sys
import urllib.request

from .config import is_free_openrouter_model

URL = "https://openrouter.ai/api/v1/models"


def free_models() -> list[dict]:
    with urllib.request.urlopen(URL, timeout=30) as r:
        data = json.load(r)["data"]
    out = []
    for m in data:
        p = m.get("pricing") or {}
        arch = m.get("architecture") or {}
        if p.get("prompt") != "0" or p.get("completion") != "0":
            continue
        if not is_free_openrouter_model(m["id"]):
            continue  # временно бесплатные превью без «:free» сервис не использует
        if "text" not in (arch.get("output_modalities") or ["text"]):
            continue  # генерация картинок/музыки не подходит
        params = m.get("supported_parameters") or []
        out.append(
            {
                "id": m["id"],
                "context": m.get("context_length") or 0,
                "json": "response_format" in params,
            }
        )
    return sorted(out, key=lambda x: (not x["json"], x["id"]))


def main() -> int:
    try:
        models = free_models()
    except Exception as e:  # сеть, формат ответа
        print(f"Не удалось получить список моделей: {e}", file=sys.stderr)
        return 1
    print(f"Бесплатные модели OpenRouter: {len(models)}\n")
    print(f"{'id':55} {'контекст':>10}  JSON-режим")
    for m in models:
        print(f"{m['id']:55} {m['context']:>10,}  {'да' if m['json'] else 'нет'}")
    print("\nСервису нужен JSON-режим (response_format). Модель задаётся в .env: UNION_MODEL=<id>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
