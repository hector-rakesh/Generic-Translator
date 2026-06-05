"""
File parsing service.

Converts uploaded files into a clean text representation
that can be fed directly into the LLM prompt.

Supported:
  - PDF  → all pages concatenated
  - DOCX → paragraphs + tables
  - XLSX → first sheet, all rows as markdown table
"""

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── PDF ────────────────────────────────────────────────────────────────────────

def _parse_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"--- Page {i} ---\n{text.strip()}")

    if not pages:
        raise ValueError(
            "Could not extract text from PDF. "
            "The file may be scanned/image-only — OCR support is not yet included."
        )

    return "\n\n".join(pages)


# ── DOCX ───────────────────────────────────────────────────────────────────────

def _parse_docx(data: bytes) -> str:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(data))
    parts: list[str] = []

    for block in doc.element.body:
        tag = block.tag.split("}")[-1]  # strip namespace

        if tag == "p":
            para = Paragraph(block, doc)
            text = para.text.strip()
            if text:
                parts.append(text)

        elif tag == "tbl":
            table = Table(block, doc)
            rows: list[str] = []
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                rows.append(" | ".join(cells))
            if rows:
                # simple markdown table
                header = rows[0]
                separator = " | ".join(["---"] * len(rows[0].split(" | ")))
                parts.append("\n".join([header, separator] + rows[1:]))

    if not parts:
        raise ValueError("Could not extract any text from the DOCX file.")

    return "\n\n".join(parts)


# ── XLSX ───────────────────────────────────────────────────────────────────────

def _parse_xlsx(data: bytes) -> str:
    import pandas as pd

    # Try openpyxl first (xlsx), fall back to xlrd (xls)
    try:
        df = pd.read_excel(io.BytesIO(data), sheet_name=0, engine="openpyxl")
    except Exception:
        df = pd.read_excel(io.BytesIO(data), sheet_name=0, engine="xlrd")

    if df.empty:
        raise ValueError("The Excel sheet is empty.")

    # Drop completely empty rows/columns
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    return df.to_markdown(index=False)


# ── Public entry point ─────────────────────────────────────────────────────────

PARSERS = {
    ".pdf":  _parse_pdf,
    ".docx": _parse_docx,
    ".doc":  _parse_docx,   # python-docx handles both
    ".xlsx": _parse_xlsx,
    ".xls":  _parse_xlsx,
}


def parse_file(filename: str, data: bytes) -> str:
    """
    Extract text from *data* based on *filename* extension.

    Returns a plain-text / markdown string ready for LLM consumption.
    Raises ValueError for unsupported types or extraction failures.
    """
    suffix = Path(filename).suffix.lower()

    parser = PARSERS.get(suffix)
    if parser is None:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported types: {', '.join(PARSERS)}"
        )

    logger.info("Parsing file '%s' (type=%s, size=%d bytes)", filename, suffix, len(data))
    text = parser(data)
    logger.info("Extracted %d characters from '%s'", len(text), filename)
    return text
