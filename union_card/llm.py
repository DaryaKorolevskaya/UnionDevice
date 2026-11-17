"""Клиенты генерации: Claude через Anthropic API и офлайн-заглушка для демо без ключа."""

from __future__ import annotations

from typing import List, Protocol

import anthropic

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
            "Для полноценного SEO-текста укажите ANTHROPIC_API_KEY и повторите генерацию.",
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
