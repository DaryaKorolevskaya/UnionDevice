"""Модели входных данных, ответа LLM и итоговой карточки."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

NO_DATA = "данных нет"


class ProductInput(BaseModel):
    name: str = ""
    manufacturer: str = ""
    specs: Dict[str, str] = Field(default_factory=dict)
    description: str = ""


# --- То, что возвращает LLM (structured output) ---------------------------------

class Commercial(BaseModel):
    target_audience: List[str] = Field(description="Кому подходит прибор: отрасли, роли, типы предприятий")
    tasks: List[str] = Field(description="Какие задачи решает прибор")
    pitch: str = Field(description="1–2 предложения для менеджера: как предложить прибор клиенту")


class LlmDraft(BaseModel):
    seo_title: str = Field(description="SEO-заголовок, не длиннее 70 символов")
    seo_description: str = Field(description="SEO-описание 800–1200 символов, без выдуманных значений")
    search_queries: List[str] = Field(description="3–5 поисковых запросов")
    commercial: Commercial
    clarifications: List[str] = Field(
        description="Что стоит уточнить у производителя/технолога, чтобы карточка была полной"
    )


# --- Итоговая карточка ----------------------------------------------------------

class Characteristic(BaseModel):
    name: str
    value: str
    source: Literal["input", "missing"]


class ProductCard(BaseModel):
    product_name: str
    manufacturer: str
    seo_title: str
    seo_description: str
    characteristics: List[Characteristic]
    search_queries: List[str]
    commercial: Commercial
    missing_fields: List[str]
    clarifications: List[str]
    warnings: List[str]
    generator: str  # модель или "mock"
    attempts: int


class GenerateResponse(BaseModel):
    card: ProductCard
    markdown: str


class ErrorResponse(BaseModel):
    error: str
    details: Optional[List[str]] = None
