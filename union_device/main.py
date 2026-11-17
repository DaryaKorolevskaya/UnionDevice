"""Точка входа: веб-сервер (форма + HTTP API). Запуск: uvicorn union_device.main:app"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import settings
from .core.llm import LlmError, LLMClient
from .core.pipeline import generate_card
from .utils.free_models import free_models
from .utils.validation import InputError, parse_json, validate_product

WEB = Path(__file__).parent / "web"

app = FastAPI(title="UnionDevice — AI-помощник для карточек КИП", version="0.1.0")
_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Один клиент на процесс: он помнит, какая модель ответила последней."""
    global _client
    if _client is None:
        _client = LLMClient()  # LlmError, если не задан ключ
    return _client


@app.exception_handler(InputError)
async def _input_error(_: Request, e: InputError):
    return JSONResponse(status_code=422, content={"error": str(e), "details": e.details})


@app.exception_handler(LlmError)
async def _llm_error(_: Request, e: LlmError):
    return JSONResponse(status_code=e.status, content={"error": str(e)})


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    ready = bool(settings.api_key) or not settings.is_openrouter
    # модель, которая будет использована следующей (последняя ответившая или первая в списке)
    current = _client.name if _client else settings.models[0]
    return {"status": "ok", "llm_ready": ready, "model": current, "models": settings.models}


@app.get("/api/models")
def models():
    """Бесплатные модели OpenRouter, которые можно указать в UNION_MODELS."""
    try:
        return {"current": settings.models, "free": free_models()}
    except Exception as e:  # сеть, формат ответа OpenRouter
        return JSONResponse(status_code=503, content={"error": f"Не удалось получить список моделей: {e}"})


@app.post("/api/generate")
async def generate(request: Request):
    # Тело читаем вручную, чтобы самим вернуть понятную ошибку на кривой JSON.
    data = parse_json(await request.body())
    validate_product(data)  # ошибки входа (422) — раньше, чем проверка ключа
    # генерация синхронная (вызов API) — выполняем в пуле потоков, не блокируя event loop
    result = await run_in_threadpool(generate_card, data, get_client())
    return result.model_dump()
