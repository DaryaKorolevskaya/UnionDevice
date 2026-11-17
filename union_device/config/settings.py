"""Настройки сервиса. Ключ API читается только из окружения или .env (в .gitignore)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1"

# Бесплатные модели OpenRouter в порядке приоритета. Если модель перегружена или недоступна,
# запрос уходит следующей. Актуальный список: GET /api/models
DEFAULT_MODELS = [
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free",  # роутер OpenRouter: сам выбирает свободную бесплатную модель
]


def is_free_openrouter_model(model: str) -> bool:
    """На OpenRouter бесплатные модели имеют суффикс «:free»; openrouter/free — роутер по бесплатным."""
    return model.endswith(":free") or model == "openrouter/free"


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    models: List[str]
    max_attempts: int

    @property
    def is_openrouter(self) -> bool:
        return "openrouter.ai" in self.base_url

    @classmethod
    def from_env(cls) -> "Settings":
        models_env = os.getenv("UNION_MODELS", "")
        models = [m.strip() for m in models_env.split(",") if m.strip()] or DEFAULT_MODELS
        s = cls(
            api_key=os.getenv("OPENROUTER_API_KEY") or None,
            base_url=os.getenv("UNION_BASE_URL") or OPENROUTER_URL,
            models=models,
            max_attempts=int(os.getenv("UNION_MAX_ATTEMPTS") or "3"),
        )
        paid = [m for m in s.models if not is_free_openrouter_model(m)]
        if s.is_openrouter and paid and not _flag("UNION_ALLOW_PAID"):
            raise ValueError(
                f"Модель «{paid[0]}» на OpenRouter платная. Разрешены только бесплатные модели "
                "(id оканчивается на «:free» или openrouter/free). Чтобы всё же использовать платную, "
                "задайте UNION_ALLOW_PAID=1."
            )
        return s


settings = Settings.from_env()
