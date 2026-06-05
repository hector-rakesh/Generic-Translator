from pydantic import BaseModel
from typing import Any


class ParseResponse(BaseModel):
    """Successful parse response."""
    success: bool = True
    data: dict[str, Any]
    warnings: list[str] = []          # populated for lenient fields that had issues
    raw_llm_output: str | None = None  # included only in debug mode


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    detail: str | None = None


class ValidationDetail(BaseModel):
    """Per-field validation result used internally."""
    field: str
    strict: bool
    valid: bool
    message: str | None = None
