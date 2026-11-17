"""Настройки сервиса. Ключ API читается только из окружения или .env (в .gitignore)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str
    effort: str
    max_attempts: int
    mock: bool

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("ANTHROPIC_API_KEY") or None
        force_mock = os.getenv("UNION_MOCK", "").strip().lower() in {"1", "true", "yes"}
        return cls(
            api_key=api_key,
            model=os.getenv("UNION_MODEL", "claude-opus-5-5"),
            effort=os.getenv("UNION_EFFORT", "medium"),
            max_attempts=int(os.getenv("UNION_MAX_ATTEMPTS", "3")),
            # Без ключа сервис всё равно запускается — в демо-режиме без LLM.
            mock=force_mock or api_key is None,
        )


settings = Settings.from_env()
