"""Список бесплатных текстовых моделей OpenRouter (GET /api/models).

Список у OpenRouter меняется, поэтому он берётся из публичного API (ключ не нужен).
Выбранные модели укажите в .env: UNION_MODELS=<id>,<id> (порядок = приоритет).
"""

from __future__ import annotations

import json
import urllib.request

from ..config import is_free_openrouter_model

URL = "https://openrouter.ai/api/v1/models"


def free_models() -> list[dict]:
    with urllib.request.urlopen(URL, timeout=30) as r:
        data = json.load(r)["data"]
    out = []
    for m in data:
        p = m.get("pricing") or {}
        if p.get("prompt") != "0" or p.get("completion") != "0" or not is_free_openrouter_model(m["id"]):
            continue
        if "text" not in ((m.get("architecture") or {}).get("output_modalities") or ["text"]):
            continue  # генерация картинок/музыки не подходит
        out.append(
            {
                "id": m["id"],
                "context": m.get("context_length") or 0,
                # сервису нужен JSON-режим ответа
                "json_mode": "response_format" in (m.get("supported_parameters") or []),
            }
        )
    return sorted(out, key=lambda x: (not x["json_mode"], x["id"]))
