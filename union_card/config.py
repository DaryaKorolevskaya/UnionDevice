"""Настройки сервиса. Ключи API читаются только из окружения или .env (в .gitignore)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    "gemini": "gemini-flash-latest",  # есть бесплатный тариф в Google AI Studio
    "anthropic": "claude-opus-5-5",
    "mock": "mock",
}


@dataclass(frozen=True)
class Settings:
    provider: str  # gemini | anthropic | mock
    api_key: str | None
    model: str
    effort: str
    max_attempts: int

    @property
    def mock(self) -> bool:
        return self.provider == "mock"

    @classmethod
    def from_env(cls) -> "Settings":
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None
        anthropic_key = os.getenv("ANTHROPIC_API_KEY") or None
        provider = os.getenv("UNION_PROVIDER", "auto").strip().lower()
        if os.getenv("UNION_MOCK", "").strip().lower() in {"1", "true", "yes"}:
            provider = "mock"
        if provider == "auto":
            # Без ключей сервис всё равно запускается — в демо-режиме без LLM.
            provider = "gemini" if gemini_key else "anthropic" if anthropic_key else "mock"
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"UNION_PROVIDER должен быть gemini, anthropic, mock или auto, а не «{provider}»")
        key = {"gemini": gemini_key, "anthropic": anthropic_key}.get(provider)
        if provider != "mock" and not key:
            raise ValueError(f"Для UNION_PROVIDER={provider} не задан API-ключ в .env")
        return cls(
            provider=provider,
            api_key=key,
            model=os.getenv("UNION_MODEL") or DEFAULT_MODELS[provider],
            effort=os.getenv("UNION_EFFORT", "medium"),
            max_attempts=int(os.getenv("UNION_MAX_ATTEMPTS", "3")),
        )


settings = Settings.from_env()
