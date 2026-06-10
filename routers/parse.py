"""
/parse  router.

Endpoints:
  POST /parse/        — upload schema + document, get extracted JSON back
  GET  /parse/health  — liveness check
"""

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from core.config import get_settings
from models.schemas import ErrorResponse, ParseResponse
from services.file_parser import parse_file
from services.llm_service import run_extraction
from services.validator import validate_output

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/parse", tags=["Parse"])

MAX_BYTES = settings.max_file_size_mb * 1024 * 1024


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health", summary="Liveness check")
async def health():

    return {"status": "ok"}    
    # from openai import OpenAI
    # import os
    # client = OpenAI(
    #     api_key=os.environ.get("GROQ_API_KEY"),
    #     base_url="https://api.groq.com/openai/v1",
    # )

    # response = client.responses.create(
    #     input="Explain the importance of fast language models",
    #     model="llama-3.3-70b-versatile",
    # )
    # print(response.output_text)




# ── Main parse endpoint ────────────────────────────────────────────────────────

@router.post(
    "/",
    summary="Extract structured JSON from a document using a JSON Schema",
    response_model=ParseResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Strict schema validation failed"},
        400: {"model": ErrorResponse, "description": "Bad input (unsupported file type, empty file, invalid schema)"},
        500: {"model": ErrorResponse, "description": "LLM or internal error"},
    },
)
async def parse_document(
    schema_file: UploadFile = File(
        ...,
        description="JSON Schema file (.json). "
                    "Add `x-strict: true/false` to each property to control validation strictness.",
    ),
    input_file: UploadFile = File(
        ...,
        description="Document to parse: PDF, DOCX, DOC, XLSX, or XLS.",
    ),
    debug: bool = Query(
        False,
        description="When true, include the raw LLM output in the response for debugging.",
    ),
):
    # ── 1. Read & validate schema ──────────────────────────────────────────────
    schema_bytes = await schema_file.read()
    if not schema_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Schema file is empty.")

    try:
        schema = json.loads(schema_bytes)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON in schema file: {exc}",
        )

    if not isinstance(schema, dict):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Schema must be a JSON object.",
        )

    # ── 2. Read & size-check the input file ────────────────────────────────────
    input_bytes = await input_file.read()
    if not input_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Input file is empty.")

    if len(input_bytes) > MAX_BYTES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Input file is too large ({len(input_bytes) / 1024 / 1024:.1f} MB). "
                f"Maximum allowed: {settings.max_file_size_mb} MB."
            ),
        )

    # ── 3. Parse file → text ───────────────────────────────────────────────────
    try:
        document_text = parse_file(input_file.filename or "upload", input_bytes)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during file parsing")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File parsing failed: {exc}",
        )

    # ── 4. LLM extraction ──────────────────────────────────────────────────────
    try:
        extracted, raw_llm = run_extraction(document_text, schema)
    except ValueError as exc:
        # JSON parsing of LLM output failed
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("LLM call failed")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM extraction failed: {exc}",
        )

    # ── 5. Schema validation (per-field strictness) ────────────────────────────
    try:
        cleaned, warnings = validate_output(extracted, schema)
    except ValueError as exc:
        # At least one strict field failed — return 422
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                success=False,
                error="Schema validation failed",
                detail=str(exc),
                output=cleaned,
            ).model_dump(),
        )

    # ── 6. Build response ──────────────────────────────────────────────────────
    return ParseResponse(
        success=True,
        data=cleaned,
        warnings=warnings,
        raw_llm_output=raw_llm if debug else None,
    )
