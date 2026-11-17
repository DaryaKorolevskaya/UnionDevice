import json

import pytest
from fastapi.testclient import TestClient

from union_card.api import app
from union_card.checks import find_problems
from union_card.schemas import NO_DATA, Commercial, LlmDraft
from union_card.service import generate_card
from union_card.validation import InputError, parse_json, validate_product

MANOMETER = {
    "name": "Манометр показывающий DM 02",
    "manufacturer": "Манотомь",
    "specs": {"диапазон измерений": "0–1,6 МПа", "класс точности": "1,5", "присоединение": "G1/2", "диаметр корпуса": "100 мм"},
    "description": "Прибор предназначен для измерения избыточного давления жидкостей и газов.",
}


def draft(**kw):
    base = dict(
        seo_title="Манометр DM 02 0–1,6 МПа G1/2 Манотомь",
        seo_description="Манометр DM 02 с диапазоном 0–1,6 МПа и классом точности 1,5. " * 15,
        search_queries=["манометр dm 02", "манометр 0-1,6 мпа", "манометр g1/2 100 мм"],
        commercial=Commercial(target_audience=["КИПиА"], tasks=["контроль давления"], pitch="Надёжный манометр."),
        clarifications=[],
    )
    base.update(kw)
    return LlmDraft(**base)


class FakeGen:
    name = "fake"

    def __init__(self, drafts):
        self.drafts, self.calls = list(drafts), []

    def generate(self, messages):
        self.calls.append(messages)
        return self.drafts.pop(0)


# --- валидация входа ---

def test_invalid_json():
    with pytest.raises(InputError) as e:
        parse_json('{"name": "x",}')
    assert "Строка 1" in e.value.details[0]


@pytest.mark.parametrize("data", [
    {"manufacturer": "X", "specs": {"a": "1"}},  # нет названия
    {"name": "Термометр", "specs": {}, "description": ""},  # не на что опереться
    "строка",
    {"name": "Уровнемер", "specs": "100 мм"},
])
def test_rejected_inputs(data):
    with pytest.raises(InputError):
        validate_product(data)


def test_missing_specs_detected():
    v = validate_product({**MANOMETER, "specs": {**MANOMETER["specs"], "температура среды": ""}})
    assert "Питание" in v.missing_specs
    assert "температура среды" in v.missing_specs
    assert "Диапазон измерений" not in v.missing_specs


# --- проверки ответа ---

def test_invented_number_flagged():
    product = validate_product(MANOMETER).product
    problems = find_problems(draft(seo_title="Манометр DM 02 IP65 Манотомь"), product)
    assert any("65" in p for p in problems)


def test_title_too_long_flagged():
    product = validate_product(MANOMETER).product
    assert any("seo_title" in p for p in find_problems(draft(seo_title="М" * 71), product))


# --- сервис ---

def test_repair_loop_then_success():
    gen = FakeGen([draft(seo_title="Манометр DM 02 на 2,5 МПа"), draft()])
    res = generate_card(MANOMETER, generator=gen, max_attempts=3)
    assert res.card.attempts == 2
    assert "2.5" in gen.calls[1][-1]["content"]  # замечание отправлено модели
    assert not any("Не исправлено" in w for w in res.card.warnings)


def test_characteristics_from_input_only():
    res = generate_card(MANOMETER, generator=FakeGen([draft()]))
    by_name = {c.name: c for c in res.card.characteristics}
    assert by_name["Диапазон измерений"].value == "0–1,6 МПа"
    assert by_name["Питание"].value == NO_DATA
    assert "| Питание | данных нет |" in res.markdown


def test_unfixed_problems_become_warnings():
    bad = draft(seo_description="коротко")
    res = generate_card(MANOMETER, generator=FakeGen([bad, bad]), max_attempts=2)
    assert any("seo_description" in w for w in res.card.warnings)


# --- HTTP API (демо-режим) ---

def test_api_ok_and_errors(monkeypatch):
    client = TestClient(app)
    r = client.post("/api/generate", content=json.dumps(MANOMETER))
    assert r.status_code == 200
    assert r.json()["card"]["seo_title"]
    r = client.post("/api/generate", content="{oops")
    assert r.status_code == 422 and "JSON" in r.json()["error"]


def test_provider_auto_selection(monkeypatch):
    from union_card.config import Settings

    for k in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "UNION_MOCK", "UNION_MODEL", "UNION_PROVIDER", "UNION_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    assert Settings.from_env().provider == "mock"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert Settings.from_env().provider == "anthropic"
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    s = Settings.from_env()
    assert (s.provider, s.model) == ("gemini", "gemini-3.8-flash")


def test_strip_fences_and_openrouter_default(monkeypatch):
    from union_card.config import Settings
    from union_card.llm import _strip_fences

    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fences('Вот ответ: {"a": 1} готово') == '{"a": 1}'
    for k in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "UNION_MOCK", "UNION_MODEL", "UNION_PROVIDER", "UNION_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    s = Settings.from_env()
    assert s.provider == "openrouter" and s.model.endswith(":free") and "openrouter.ai" in s.base_url
