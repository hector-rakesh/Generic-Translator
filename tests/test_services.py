"""
Unit tests for DocParser services.
Run: pytest tests/ -v
"""

import io
import json
import pytest


# ── Validator tests ────────────────────────────────────────────────────────────

from services.validator import validate_output


SCHEMA = {
    "type": "object",
    "properties": {
        "name":   {"type": "string",  "x-strict": True},
        "amount": {"type": "number",  "x-strict": True},
        "notes":  {"type": "string",  "x-strict": False},
        "email":  {"type": "string",  "format": "email", "x-strict": False},
    },
    "required": ["name", "amount"],
}


def test_valid_data_passes():
    data = {"name": "Acme Corp", "amount": 500.0}
    cleaned, warnings = validate_output(data, SCHEMA)
    assert cleaned["name"] == "Acme Corp"
    assert warnings == []


def test_strict_violation_raises():
    data = {"name": 123, "amount": 500.0}   # name should be string
    with pytest.raises(ValueError, match="strict"):
        validate_output(data, SCHEMA)


def test_lenient_violation_produces_warning():
    data = {"name": "Acme", "amount": 100.0, "notes": 42}  # notes should be string
    cleaned, warnings = validate_output(data, SCHEMA)
    assert len(warnings) == 1
    assert "notes" in warnings[0]


def test_multiple_strict_violations_listed():
    data = {"name": None, "amount": "not-a-number"}
    with pytest.raises(ValueError) as exc_info:
        validate_output(data, SCHEMA)
    msg = str(exc_info.value)
    assert "name" in msg or "amount" in msg


def test_unknown_field_passes():
    # JSON Schema does not reject additional properties by default
    data = {"name": "X", "amount": 1.0, "extra_field": "ignored"}
    cleaned, warnings = validate_output(data, SCHEMA)
    assert cleaned["extra_field"] == "ignored"


# ── LLM JSON extraction tests ──────────────────────────────────────────────────

from services.llm_service import (
    _deep_merge,
    _split_document_text,
    extract_json_from_response,
)


def test_clean_json_parses():
    raw = '{"invoice_number": "INV-001", "total": 250.0}'
    result = extract_json_from_response(raw)
    assert result["invoice_number"] == "INV-001"


def test_fenced_json_parses():
    raw = '```json\n{"key": "value"}\n```'
    result = extract_json_from_response(raw)
    assert result["key"] == "value"


def test_json_with_prose_before():
    raw = 'Here is the extracted data:\n{"key": "value"}'
    result = extract_json_from_response(raw)
    assert result["key"] == "value"


def test_invalid_json_raises():
    raw = "I could not extract anything useful."
    with pytest.raises(ValueError, match="valid JSON"):
        extract_json_from_response(raw)


def test_split_document_text_returns_single_chunk_for_short_text():
    text = "short document"
    assert _split_document_text(text, max_chars=100, overlap=10) == [text]


def test_split_document_text_splits_long_text():
    text = "alpha\n\n" + ("word " * 200) + "\n\nbeta"
    chunks = _split_document_text(text, max_chars=120, overlap=20)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert "alpha" in chunks[0]
    assert "beta" in chunks[-1]


def test_deep_merge_combines_nested_arrays_by_id():
    left = {
        "credentialing_standards": [
            {"standard_id": "CR 1", "elements": [{"element_id": "A", "intent": "left"}]}
        ]
    }
    right = {
        "credentialing_standards": [
            {"standard_id": "CR 1", "elements": [{"element_id": "B", "intent": "right"}]},
            {"standard_id": "CR 2", "standard_title": "Policies"},
        ]
    }
    merged = _deep_merge(left, right)
    assert len(merged["credentialing_standards"]) == 2
    assert len(merged["credentialing_standards"][0]["elements"]) == 2


# ── File parser tests ──────────────────────────────────────────────────────────

from services.file_parser import parse_file


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_file("document.txt", b"some text")


def test_pdf_parsing(tmp_path):
    """Create a minimal valid PDF and check extraction."""
    import fpdf  # install fpdf2 for this test
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, txt="Invoice Number: INV-2024-001")
    pdf_bytes = pdf.output()
    text = parse_file("test.pdf", bytes(pdf_bytes))
    assert "INV-2024-001" in text


def test_xlsx_parsing():
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Name", "Amount"])
    ws.append(["Acme Corp", 500])
    buf = io.BytesIO()
    wb.save(buf)
    text = parse_file("data.xlsx", buf.getvalue())
    assert "Acme Corp" in text
    assert "500" in text
