"""Настройки сервиса. Ключи API читаются только из окружения или .env (в .gitignore)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

PROVIDERS = {
    # провайдер: (модель по умолчанию, base_url для OpenAI-совместимых API)
    "openrouter": ("google/gemma-4-31b-it:free", "https://openrouter.ai/api/v1"),  # открытая модель, бесплатно
    "ollama": ("qwen2.5:7b-instruct", "http://localhost:11434/v1"),  # локально, без ключа
    "gemini": ("gemini-3.8-flash", None),
    "anthropic": ("claude-opus-5-5", None),
    "mock": ("mock", None),
}


@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str | None
    model: str
    base_url: str | None
    effort: str
    max_attempts: int

    @property
    def mock(self) -> bool:
        return self.provider == "mock"

    @classmethod
    def from_env(cls) -> "Settings":
        keys = {
            "openrouter": os.getenv("OPENROUTER_API_KEY") or None,
            "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None,
            "anthropic": os.getenv("ANTHROPIC_API_KEY") or None,
        }
        provider = (os.getenv("UNION_PROVIDER") or "auto").strip().lower()
        if os.getenv("UNION_MOCK", "").strip().lower() in {"1", "true", "yes"}:
            provider = "mock"
        if provider == "auto":
            # первый провайдер, для которого задан ключ; без ключей — демо-режим без LLM
            provider = next((p for p in ("openrouter", "gemini", "anthropic") if keys[p]), "mock")
        if provider not in PROVIDERS:
            raise ValueError(f"UNION_PROVIDER должен быть одним из {', '.join(PROVIDERS)} или auto, а не «{provider}»")
        if provider in keys and not keys[provider]:
            raise ValueError(f"Для UNION_PROVIDER={provider} не задан API-ключ в .env")
        default_model, default_url = PROVIDERS[provider]
        return cls(
            provider=provider,
            api_key=keys.get(provider),
            model=os.getenv("UNION_MODEL") or default_model,
            base_url=os.getenv("UNION_BASE_URL") or default_url,
            effort=os.getenv("UNION_EFFORT") or "medium",
            max_attempts=int(os.getenv("UNION_MAX_ATTEMPTS") or "3"),
        )


settings = Settings.from_env()
