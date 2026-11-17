"""Клиенты генерации: OpenAI-совместимые API (OpenRouter, Ollama), Gemini, Claude и офлайн-заглушка."""

from __future__ import annotations

import json
from typing import List, Protocol

import anthropic
from pydantic import ValidationError

from .config import Settings
from .prompts import SYSTEM_PROMPT
from .schemas import Commercial, LlmDraft


class LlmError(RuntimeError):
    """Ошибка обращения к LLM, которую можно показать пользователю."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


class Generator(Protocol):
    name: str

    def generate(self, messages: List[dict]) -> LlmDraft: ...


class ClaudeGenerator:
    def __init__(self, settings: Settings):
        self.name = settings.model
        self._effort = settings.effort
        # SDK сам повторяет 408/409/429/5xx и сетевые ошибки.
        self._client = anthropic.Anthropic(api_key=settings.api_key, max_retries=3, timeout=120)

    def generate(self, messages: List[dict]) -> LlmDraft:
        try:
            response = self._client.messages.parse(
                model=self.name,
                max_tokens=16000,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                output_config={"effort": self._effort},
                output_format=LlmDraft,
            )
        except anthropic.AuthenticationError as e:
            raise LlmError("LLM API: неверный или отсутствующий ANTHROPIC_API_KEY.", 401) from e
        except anthropic.PermissionDeniedError as e:
            raise LlmError("LLM API: у ключа нет доступа к выбранной модели.", 403) from e
        except anthropic.NotFoundError as e:
            raise LlmError(f"LLM API: модель «{self.name}» не найдена (проверьте UNION_MODEL).", 502) from e
        except anthropic.RateLimitError as e:
            raise LlmError("LLM API: превышен лимит запросов, попробуйте позже.", 429) from e
        except anthropic.BadRequestError as e:
            raise LlmError(f"LLM API отклонил запрос: {e.message}", 502) from e
        except anthropic.APIStatusError as e:
            raise LlmError(f"LLM API вернул ошибку {e.status_code}, попробуйте позже.", 503) from e
        except anthropic.APIConnectionError as e:
            raise LlmError("Нет связи с LLM API (сеть/таймаут).", 503) from e

        if response.stop_reason == "refusal":
            raise LlmError("Модель отказалась обрабатывать запрос.", 422)
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise LlmError("Модель вернула неполный ответ, попробуйте ещё раз.", 502)
        return response.parsed_output


class OpenAICompatGenerator:
    """Любой OpenAI-совместимый API: OpenRouter (открытые модели, есть бесплатные) или локальная Ollama.

    Не все открытые модели поддерживают строгую JSON-схему, поэтому просим JSON-объект,
    кладём схему в системный промпт и валидируем ответ Pydantic'ом сами.
    """

    def __init__(self, settings: Settings):
        import openai

        self.name = settings.model
        self._openai = openai
        self._client = openai.OpenAI(
            api_key=settings.api_key or "ollama",  # Ollama ключ не проверяет
            base_url=settings.base_url,
            max_retries=4,  # 429/5xx бесплатных моделей — частое явление
            timeout=180,
            default_headers={"X-Title": "Union Card"},
        )
        schema = json.dumps(LlmDraft.model_json_schema(), ensure_ascii=False)
        self._system = (
            f"{SYSTEM_PROMPT}\n## Формат ответа\nВерни ТОЛЬКО JSON-объект без markdown и пояснений, "
            f"строго по этой JSON-схеме:\n{schema}\n"
        )

    def _call(self, messages: List[dict]) -> str:
        oa = self._openai
        try:
            resp = self._client.chat.completions.create(
                model=self.name,
                messages=[{"role": "system", "content": self._system}, *messages],
                response_format={"type": "json_object"},
                temperature=0.3,  # меньше «творчества» — меньше выдумок
                max_tokens=4000,
            )
        except oa.AuthenticationError as e:
            raise LlmError("LLM API: неверный API-ключ.", 401) from e
        except oa.PermissionDeniedError as e:
            raise LlmError("LLM API: нет доступа к модели.", 403) from e
        except oa.NotFoundError as e:
            raise LlmError(f"LLM API: модель «{self.name}» не найдена (проверьте UNION_MODEL).", 502) from e
        except oa.RateLimitError as e:
            raise LlmError("LLM API: исчерпан лимит бесплатных запросов, попробуйте позже.", 429) from e
        except oa.BadRequestError as e:
            raise LlmError(f"LLM API отклонил запрос: {e.message}", 502) from e
        except oa.APIStatusError as e:
            raise LlmError(f"LLM API вернул ошибку {e.status_code}, попробуйте позже.", 503) from e
        except oa.APIConnectionError as e:
            raise LlmError("Нет связи с LLM API (сеть/таймаут, для Ollama — запущена ли она?).", 503) from e
        if not resp.choices:
            raise LlmError("LLM API вернул пустой ответ, попробуйте ещё раз.", 502)
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise LlmError("Модель вернула неполный ответ, попробуйте ещё раз.", 502)
        return choice.message.content or ""

    def generate(self, messages: List[dict]) -> LlmDraft:
        text = self._call(messages)
        try:
            return LlmDraft.model_validate_json(_strip_fences(text))
        except ValidationError as e:
            # одна повторная попытка с сообщением об ошибке формата
            retry = [*messages, {"role": "assistant", "content": text},
                     {"role": "user", "content": f"Ответ не соответствует схеме: {e.errors()[:3]}. Верни исправленный JSON."}]
            try:
                return LlmDraft.model_validate_json(_strip_fences(self._call(retry)))
            except ValidationError as e2:
                raise LlmError("Модель дважды вернула ответ не по схеме, попробуйте другую модель.", 502) from e2


def _strip_fences(text: str) -> str:
    """Некоторые модели оборачивают JSON в ```json ... ``` даже при json_object."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    start, end = t.find("{"), t.rfind("}")
    return t[start : end + 1] if start != -1 and end != -1 else t


class GeminiGenerator:
    """Google Gemini (google-genai SDK). Бесплатный ключ: https://aistudio.google.com/apikey"""

    def __init__(self, settings: Settings):
        from google import genai
        from google.genai import types

        self.name = settings.model
        self._types = types
        self._client = genai.Client(
            api_key=settings.api_key,
            http_options=types.HttpOptions(
                timeout=120_000,  # мс
                # бесплатный тариф часто отвечает 429/503 — SDK повторит с экспоненциальной паузой
                retry_options=types.HttpRetryOptions(attempts=4, http_status_codes=[429, 500, 502, 503, 504]),
            ),
        )

    def generate(self, messages: List[dict]) -> LlmDraft:
        from google.genai import errors

        t = self._types
        contents = [
            t.Content(role="model" if m["role"] == "assistant" else "user", parts=[t.Part(text=m["content"])])
            for m in messages
        ]
        try:
            response = self._client.models.generate_content(
                model=self.name,
                contents=contents,
                config=t.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=LlmDraft,
                    temperature=0.3,  # меньше «творчества» — меньше выдумок
                ),
            )
        except errors.ClientError as e:
            if e.code in (401, 403) or "API_KEY" in str(e):
                raise LlmError("Gemini API: неверный GEMINI_API_KEY или нет доступа.", 401) from e
            if "location is not supported" in (e.message or ""):
                raise LlmError(
                    "Gemini API недоступен из вашего региона (User location is not supported). "
                    "Используйте другой провайдер в UNION_PROVIDER.", 451
                ) from e
            if e.code == 404:
                raise LlmError(f"Gemini API: модель «{self.name}» не найдена (проверьте UNION_MODEL).", 502) from e
            if e.code == 429:
                raise LlmError("Gemini API: исчерпана квота бесплатного тарифа, попробуйте позже.", 429) from e
            raise LlmError(f"Gemini API отклонил запрос ({e.code}): {e.message}", 502) from e
        except errors.ServerError as e:
            raise LlmError(f"Gemini API временно недоступен ({e.code}), попробуйте позже.", 503) from e
        except Exception as e:  # сеть, таймаут — SDK не даёт отдельного класса
            raise LlmError(f"Нет связи с Gemini API: {type(e).__name__}", 503) from e

        candidate = response.candidates[0] if response.candidates else None
        reason = candidate.finish_reason.name if candidate and candidate.finish_reason else None
        if reason in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}:
            raise LlmError("Модель отказалась обрабатывать запрос (фильтр безопасности).", 422)
        if reason == "MAX_TOKENS":
            raise LlmError("Модель вернула неполный ответ, попробуйте ещё раз.", 502)
        if isinstance(response.parsed, LlmDraft):
            return response.parsed
        try:  # на случай, если SDK не распарсил ответ сам
            return LlmDraft.model_validate_json(response.text or "")
        except Exception as e:
            raise LlmError("Модель вернула ответ не по схеме, попробуйте ещё раз.", 502) from e


class MockGenerator:
    """Детерминированный шаблон без LLM — чтобы сервис запускался и без API-ключа.

    Использует только входные данные, поэтому тоже не «выдумывает» характеристики.
    """

    name = "mock (без LLM)"

    def __init__(self) -> None:
        self.product = None  # выставляется сервисом перед вызовом

    def generate(self, messages: List[dict]) -> LlmDraft:
        p = self.product
        specs = list(p.specs.items())
        brand = f" {p.manufacturer}" if p.manufacturer else ""
        first_val = f" {specs[0][1]}" if specs else ""
        title = f"{p.name}{first_val}{brand}"
        if len(title) > 70:
            title = p.name[:70]

        spec_text = "; ".join(f"{k} — {v}" for k, v in specs) or "характеристики уточняются"
        parts = [
            f"{p.name}{(' производства ' + p.manufacturer) if p.manufacturer else ''} — прибор для промышленного "
            "контроля и учёта технологических параметров.",
            p.description,
            f"Основные характеристики: {spec_text}.",
            "Прибор подходит для оснащения технологических линий, котельных, насосных станций и систем "
            "водоснабжения, а также для замены вышедших из строя приборов того же типа. Перед заказом "
            "сверьте параметры с требованиями проекта; при необходимости менеджер поможет подобрать "
            "исполнение, комплект монтажных частей и оформит документы для закупки.",
            "Карточка сформирована в демо-режиме без LLM: текст собран по шаблону из входных данных. "
            "Для полноценного SEO-текста укажите OPENROUTER_API_KEY в .env и повторите генерацию.",
        ]
        desc = "\n\n".join(x for x in parts if x)
        while len(desc) < 800:
            desc += " Уточнить наличие, сроки поставки и условия для оптовых заказов можно у менеджера."
        if len(desc) > 1200:
            desc = desc[:1197].rsplit(" ", 1)[0] + "…"

        base = p.name.lower()
        queries = [f"{base} купить", f"{base} цена"]
        if p.manufacturer:
            queries.append(f"{base} {p.manufacturer.lower()}")
        if specs:
            queries.append(f"{base} {specs[0][1].lower()}")
        queries = (queries + [f"{base} в наличии"])[:5]

        return LlmDraft(
            seo_title=title,
            seo_description=desc,
            search_queries=queries,
            commercial=Commercial(
                target_audience=["Службы КИПиА промышленных предприятий", "Отделы снабжения", "Монтажные организации"],
                tasks=["Контроль параметров технологического процесса", "Замена и дооснащение приборного парка"],
                pitch="Демо-режим: commercial-блок формируется LLM после указания API-ключа.",
            ),
            clarifications=[],
        )
