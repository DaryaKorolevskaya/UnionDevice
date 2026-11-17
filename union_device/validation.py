"""Разбор и проверка входного JSON до обращения к LLM.

Принцип: всё, что можно проверить детерминированно, проверяется кодом, а не моделью.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .schemas import ProductInput


class InputError(ValueError):
    """Входные данные нельзя обработать. details — список понятных пользователю причин."""

    def __init__(self, message: str, details: List[str] | None = None):
        super().__init__(message)
        self.details = details or []


# Характеристики, которые контент-менеджер ожидает увидеть в карточке КИП.
# Каждая группа — регулярные выражения-синонимы, по которым ищем ключ во входных specs.
EXPECTED_SPECS: Dict[str, List[str]] = {
    "Диапазон измерений": ["диапазон", "предел"],
    "Класс точности / погрешность": ["класс точности", "погрешност", "точност"],
    "Измеряемая среда": [r"^(измеряемая |рабочая )?среда$", r"^тип среды"],
    "Присоединение к процессу": ["присоедин", "резьб", "штуцер"],
    "Выходной сигнал / интерфейс": ["выход", "интерфейс", "сигнал", "rs-485", "modbus", "hart"],
    "Питание": ["питани", "напряжени"],
    "Корпус / степень защиты": ["корпус", r"\bip\d*\b", "защит", "исполнени"],
    "Рабочая температура": ["температур"],
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


@dataclass
class ValidatedInput:
    product: ProductInput
    missing_fields: List[str] = field(default_factory=list)  # пустые/отсутствующие поля верхнего уровня
    missing_specs: List[str] = field(default_factory=list)  # ожидаемые характеристики, которых нет
    empty_specs: List[str] = field(default_factory=list)  # ключи из входа без значения
    warnings: List[str] = field(default_factory=list)


def parse_json(raw: str | bytes) -> Any:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not raw or not raw.strip():
        raise InputError("Пустой запрос: передайте JSON с данными о товаре.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise InputError(
            "Некорректный JSON.",
            [f"Строка {e.lineno}, столбец {e.colno}: {e.msg}"],
        ) from e


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(_as_str(v) for v in value if _as_str(v))
    return json.dumps(value, ensure_ascii=False)


def validate_product(data: Any) -> ValidatedInput:
    if not isinstance(data, dict):
        raise InputError("Ожидается JSON-объект с полями name, manufacturer, specs, description.")

    warnings: List[str] = []
    known = {"name", "manufacturer", "specs", "description"}
    extra = sorted(set(data) - known)
    if extra:
        warnings.append(f"Неизвестные поля проигнорированы: {', '.join(extra)}")

    for key in ("name", "manufacturer", "description"):
        if key in data and data[key] is not None and not isinstance(data[key], (str, int, float)):
            raise InputError(f"Поле «{key}» должно быть строкой.")

    raw_specs = data.get("specs") or {}
    if isinstance(raw_specs, list):
        # допускаем [{"name": ..., "value": ...}]
        try:
            raw_specs = {_as_str(i["name"]): i.get("value") for i in raw_specs}
        except (TypeError, KeyError):
            raise InputError("Поле «specs» должно быть объектом {\"характеристика\": \"значение\"}.")
    if not isinstance(raw_specs, dict):
        raise InputError("Поле «specs» должно быть объектом {\"характеристика\": \"значение\"}.")

    specs: Dict[str, str] = {}
    empty_specs: List[str] = []
    for k, v in raw_specs.items():
        k_s, v_s = _as_str(k), _as_str(v)
        if not k_s:
            continue
        if not v_s or _norm(v_s) in {"-", "—", "n/a", "нет данных", "данных нет", "?"}:
            empty_specs.append(k_s)
            continue
        specs[k_s] = v_s

    product = ProductInput(
        name=_as_str(data.get("name")),
        manufacturer=_as_str(data.get("manufacturer")),
        specs=specs,
        description=_as_str(data.get("description")),
    )

    missing_fields = [
        label
        for label, value in (
            ("name (название)", product.name),
            ("manufacturer (производитель)", product.manufacturer),
            ("specs (характеристики)", product.specs),
            ("description (описание)", product.description),
        )
        if not value
    ]

    if not product.name:
        raise InputError("Нельзя сформировать карточку без названия товара.", missing_fields)
    if not product.specs and not product.description:
        raise InputError(
            "Недостаточно данных: нет ни характеристик, ни описания — модели не на что опереться.",
            missing_fields,
        )

    if empty_specs:
        warnings.append(f"Характеристики без значения (будут помечены «данных нет»): {', '.join(empty_specs)}")
    if product.description and len(product.description) < 80:
        warnings.append("Описание очень короткое — SEO-текст будет опираться в основном на характеристики.")

    # Пустой ключ «температура среды» закрывает группу «Рабочая температура» — не дублируем её.
    all_keys = {**product.specs, **{k: "" for k in empty_specs}}
    missing_specs = [label for label in EXPECTED_SPECS if not find_spec(all_keys, label)] + empty_specs
    return ValidatedInput(product, missing_fields, missing_specs, empty_specs, warnings)


def find_spec(specs: Dict[str, str], label: str) -> str | None:
    """Ищет во входных характеристиках ключ, соответствующий ожидаемой группе."""
    for key in specs:
        nk = _norm(key)
        if any(re.search(syn, nk) for syn in EXPECTED_SPECS.get(label, [])):
            return key
    return None
