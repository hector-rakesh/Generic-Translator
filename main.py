"""
DocParser API
─────────────
Upload a JSON Schema + a document (PDF / DOCX / XLSX) and receive a
structured JSON payload extracted by an LLM.

Run:
    uvicorn main:app --reload
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from core.config import get_settings
from routers.parse import router as parse_router

settings = get_settings()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_collection()
    yield

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocParser API",
    description=(
        "Extract structured JSON from unstructured documents "
        "(PDF, DOCX, XLSX) using an LLM guided by a JSON Schema."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(parse_router)


# ── Global exception handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "detail": str(exc)},
    )


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "DocParser API",
        "version": "1.0.0",
        "docs": "/docs",
        "provider": settings.llm_provider,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
