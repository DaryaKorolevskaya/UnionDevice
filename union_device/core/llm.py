"""LLMClient — обращение к модели через OpenAI-совместимый API (по умолчанию OpenRouter).

Модели перебираются по списку: если модель перегружена, недоступна или не уложилась в лимит,
запрос уходит следующей. Модель, ответившая последней, становится первой для следующего запроса.
"""

from __future__ import annotations

import json
from typing import List

import openai
from pydantic import ValidationError

from ..config import Settings, settings as default_settings
from .prompts import SYSTEM_PROMPT
from .schemas import LlmDraft


class LlmError(RuntimeError):
    """Ошибка обращения к LLM, которую можно показать пользователю."""

    def __init__(self, message: str, status: int = 502, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable  # имеет смысл попробовать другую модель


class LLMClient:
    def __init__(self, s: Settings = default_settings):
        if s.is_openrouter and not s.api_key:
            raise LlmError("Не задан OPENROUTER_API_KEY в .env (бесплатный ключ: https://openrouter.ai/settings/keys).", 503)
        self.models = list(s.models)
        self._idx = 0
        # просим модели с рассуждением думать недолго, иначе они съедают лимит токенов
        self._extra = {"reasoning": {"effort": "low"}} if s.is_openrouter else None
        self._client = openai.OpenAI(
            api_key=s.api_key or "not-needed",  # локальной Ollama ключ не нужен
            base_url=s.base_url,
            max_retries=1,  # при 429/5xx быстрее перейти на следующую модель, чем долго ждать
            timeout=180,
            default_headers={"X-Title": "UnionDevice"},
        )
        # Не все бесплатные модели поддерживают строгую JSON-схему, поэтому просим JSON-объект,
        # передаём схему в промпте и проверяем ответ Pydantic'ом.
        schema = json.dumps(LlmDraft.model_json_schema(), ensure_ascii=False)
        self._system = (
            f"{SYSTEM_PROMPT}\n## Формат ответа\nВерни ТОЛЬКО JSON-объект без markdown и пояснений, "
            f"строго по этой JSON-схеме:\n{schema}\n"
        )

    @property
    def name(self) -> str:
        """Модель, которая ответила последней."""
        return self.models[self._idx]

    def generate(self, messages: List[dict]) -> LlmDraft:
        text = self._complete(messages)
        try:
            return LlmDraft.model_validate_json(strip_fences(text))
        except ValidationError as e:
            # одна повторная попытка с сообщением об ошибке формата
            retry = [
                *messages,
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"Ответ не соответствует схеме: {e.errors()[:3]}. Верни исправленный JSON."},
            ]
            try:
                return LlmDraft.model_validate_json(strip_fences(self._complete(retry)))
            except ValidationError as e2:
                raise LlmError("Модель дважды вернула ответ не по схеме, попробуйте ещё раз.", 502) from e2

    def _complete(self, messages: List[dict]) -> str:
        """Перебирает модели по кругу, начиная с последней успешной."""
        last: LlmError | None = None
        for model in self.models[self._idx :] + self.models[: self._idx]:
            try:
                text = self._call_model(model, messages)
                self._idx = self.models.index(model)
                return text
            except LlmError as e:
                if not e.retryable:  # ключ, оплата, некорректный запрос — другая модель не поможет
                    raise
                last = e
        assert last is not None
        raise last

    def _call_model(self, model: str, messages: List[dict]) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": self._system}, *messages],
                response_format={"type": "json_object"},
                temperature=0.3,  # меньше «творчества» — меньше выдумок
                max_tokens=16000,  # запас для моделей с рассуждением
                extra_body=self._extra,
            )
        except openai.AuthenticationError as e:
            raise LlmError("LLM API: неверный API-ключ.", 401) from e
        except openai.PermissionDeniedError as e:
            raise LlmError("LLM API: нет доступа к модели.", 403) from e
        except openai.NotFoundError as e:
            raise LlmError(f"Модель «{model}» не найдена.", 404, retryable=True) from e
        except openai.RateLimitError as e:
            raise LlmError("Бесплатные модели перегружены или исчерпан дневной лимит, попробуйте позже.", 429, retryable=True) from e
        except openai.BadRequestError as e:
            raise LlmError(f"LLM API отклонил запрос: {e.message}", 502) from e
        except openai.APIStatusError as e:
            if e.status_code == 402:
                raise LlmError("LLM API требует оплату — модель не бесплатная.", 402) from e
            raise LlmError(f"LLM API вернул ошибку {e.status_code}, попробуйте позже.", 503, retryable=True) from e
        except openai.APITimeoutError as e:
            raise LlmError("Модель не ответила вовремя, попробуйте ещё раз.", 504, retryable=True) from e
        except openai.APIConnectionError as e:
            raise LlmError("Нет связи с LLM API (сеть; для Ollama — запущена ли она?).", 503) from e

        if not resp.choices:
            raise LlmError("LLM API вернул пустой ответ.", 502, retryable=True)
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise LlmError("Модель вернула неполный ответ.", 502, retryable=True)
        return choice.message.content or ""


def strip_fences(text: str) -> str:
    """Некоторые модели оборачивают JSON в ```json ... ``` или добавляют текст вокруг."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    start, end = t.find("{"), t.rfind("}")
    return t[start : end + 1] if start != -1 and end != -1 else t
