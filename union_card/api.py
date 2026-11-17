"""HTTP API + простая веб-форма."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import settings
from .llm import LlmError
from .service import generate_card
from .validation import InputError, parse_json

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Union Card — AI-помощник для карточек КИП", version="0.1.0")


@app.exception_handler(InputError)
async def _input_error(_: Request, e: InputError):
    return JSONResponse(status_code=422, content={"error": str(e), "details": e.details})


@app.exception_handler(LlmError)
async def _llm_error(_: Request, e: LlmError):
    return JSONResponse(status_code=e.status, content={"error": str(e)})


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "mock" if settings.mock else "llm", "model": settings.model}


@app.post("/api/generate")
async def generate(request: Request):
    # Тело читаем вручную, чтобы самим вернуть понятную ошибку на кривой JSON.
    data = parse_json(await request.body())
    # generate_card синхронный (SDK-вызов) — выполняем в пуле потоков, не блокируя event loop.
    result = await run_in_threadpool(generate_card, data)
    return result.model_dump()
