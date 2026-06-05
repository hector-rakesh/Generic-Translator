# DocParser API

Extract structured JSON from unstructured documents (PDF, DOCX, XLSX) using
an LLM guided by a JSON Schema you supply at runtime.

---

## Architecture

```
POST /parse/
│
├── 1. File Parser      (services/file_parser.py)
│     PDF  → all pages concatenated as text
│     DOCX → paragraphs + tables as markdown
│     XLSX → first sheet as markdown table
│
├── 2. LLM Extraction   (services/llm_service.py)
│     Builds a prompt from document text + schema
│     Calls the configured LLM provider
│     Parses raw response → dict
│
└── 3. Validator        (services/validator.py)
      Runs jsonschema.validate()
      Per-field x-strict → hard error or warning
      Returns cleaned data + warnings list
```

---

## Quick Start

### 1. Install dependencies

```bash
cd docparser
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set HF_API_TOKEN at minimum
```

### 3. Run

```bash
uvicorn main:app --reload
```

API docs: http://localhost:8000/docs

---

## Using the API

### `POST /parse/`

Multipart form upload with two files:

| Field         | Type   | Description                                |
|---------------|--------|--------------------------------------------|
| `schema_file` | file   | JSON Schema file (`.json`)                 |
| `input_file`  | file   | Document to parse (PDF / DOCX / XLSX)      |
| `debug`       | query  | `true` to include raw LLM output           |

#### cURL example

```bash
curl -X POST http://localhost:8000/parse/ \
  -F "schema_file=@examples/invoice_schema.json" \
  -F "input_file=@my_invoice.pdf"
```

#### Success response (200)

```json
{
  "success": true,
  "data": {
    "invoice_number": "INV-2024-001",
    "invoice_date":   "2024-03-15",
    "vendor_name":    "Acme Corp",
    "total_amount":   1250.00,
    "currency":       "USD"
  },
  "warnings": [
    "'vendor_email': 'notanemail' is not a 'email'"
  ]
}
```

#### Validation error response (422)

Returned when any `x-strict: true` field fails:

```json
{
  "success": false,
  "error": "Schema validation failed",
  "detail": "Schema validation failed on 1 strict field(s):\n  • 'total_amount': …"
}
```

---

## JSON Schema: the `x-strict` extension

Add `"x-strict": true | false` to any property definition:

```json
{
  "properties": {
    "invoice_number": {
      "type": "string",
      "x-strict": true     ← missing or wrong type → HTTP 422
    },
    "notes": {
      "type": "string",
      "x-strict": false    ← missing or wrong type → warning in response
    }
  }
}
```

**Default behaviour (no `x-strict` key): treated as `true`** — this is
the safe default so you always know about violations unless you
explicitly opt a field into lenient mode.

---

## Switching LLM providers

### Current: HuggingFace (free tier)

```ini
# .env
LLM_PROVIDER=huggingface
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
```

Any HuggingFace model that supports the **text-generation** task works.
Recommended instruction-following models:

- `mistralai/Mistral-7B-Instruct-v0.3` (default)
- `HuggingFaceH4/zephyr-7b-beta`
- `microsoft/Phi-3-mini-4k-instruct`

### Future: LLaMA local (Ollama / llama.cpp / vLLM)

```ini
# .env
LLM_PROVIDER=llama_local
LLAMA_BASE_URL=http://localhost:11434/v1
LLAMA_MODEL_ID=llama3
```

No code changes needed — just update `.env`.

---

## Running tests

```bash
pip install pytest fpdf2
pytest tests/ -v
```

---

## Project layout

```
docparser/
├── main.py                   FastAPI app + middleware
├── requirements.txt
├── .env.example
│
├── core/
│   └── config.py             Pydantic settings (reads .env)
│
├── routers/
│   └── parse.py              POST /parse/ endpoint
│
├── services/
│   ├── file_parser.py        PDF / DOCX / XLSX → text
│   ├── llm_service.py        LLM backends + prompt builder
│   └── validator.py          JSON Schema validation (x-strict aware)
│
├── models/
│   └── schemas.py            Pydantic request/response models
│
├── examples/
│   └── invoice_schema.json   Example schema with x-strict usage
│
└── tests/
    └── test_services.py      Unit tests
```
