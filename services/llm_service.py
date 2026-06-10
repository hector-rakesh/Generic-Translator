"""
LLM service — provider-agnostic.

Current backends:
  • groq         — Groq Inference API
  • huggingface  — HuggingFace Inference API (free tier)
  • llama_local  — OpenAI-compatible endpoint (Ollama / llama.cpp / vLLM)

Switching providers: change LLM_PROVIDER in .env.
No application code needs to change.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from huggingface_hub import InferenceClient
from groq import Groq, APIConnectionError, AuthenticationError, APIError

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Groq free-tier per-request token caps (leave headroom below documented TPM).
GROQ_MODEL_TOKEN_LIMITS: dict[str, int] = {
    "llama-3.3-70b-versatile": 11_000,
    "llama-3.1-8b-instant": 5_500,
    "meta-llama/llama-4-scout-17b-16e-instruct": 28_000,
    "openai/gpt-oss-20b": 7_500,
    "openai/gpt-oss-120b": 7_500,
    "groq/compound": 65_000,
    "groq/compound-mini": 65_000,
}
CHARS_PER_TOKEN_ESTIMATE = 4.5


class PayloadTooLargeError(RuntimeError):
    """Raised when an LLM provider rejects a request for exceeding size limits."""


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(
    document_text: str,
    schema: dict[str, Any],
    *,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
) -> str:
    """
    Build the extraction prompt.

    The model is instructed to:
    1. Read the document.
    2. Extract values described in the JSON schema.
    3. Honour any `description` / `enum` / `x-*` hints in the schema.
    4. Return ONLY a valid JSON object — no markdown, no explanation.
    """
    # Compact JSON keeps the schema in fewer tokens (important for Groq TPM limits).
    schema_str = json.dumps(schema, separators=(",", ":"))

    chunk_note = ""
    if chunk_index is not None and total_chunks is not None and total_chunks > 1:
        chunk_note = f"""
## Document section
This is section {chunk_index + 1} of {total_chunks} from the full document.
Extract only information present in THIS section.
Use `null` for schema fields not found in this section.
Do not invent values that are not present in this section.
"""

    return f"""You are a precise data-extraction assistant.

## Task
Extract information from the DOCUMENT below and return a JSON object that
conforms exactly to the provided JSON SCHEMA.
{chunk_note}
## Rules
- Return ONLY the JSON object. No markdown fences, no explanation.
- For every property in the schema, try to find the value in the document.
- If a value cannot be found, use `null` (unless the schema specifies a default).
- Follow `type`, `enum`, `format`, and `description` hints in the schema.
- For boolean fields, map words like "yes/no", "true/false", "enabled/disabled"
  to the correct boolean value.
- For number/integer fields, strip currency symbols and thousand separators.
- Do NOT invent values that are not present in the document.

## JSON Schema
{schema_str}

## Document
{document_text}

## Output (JSON only)
"""


def _prompt_overhead_chars(schema: dict[str, Any]) -> int:
    """Chars used by schema + instructions when the document body is empty."""
    return len(build_prompt("", schema))


def _groq_max_tokens_per_request() -> int:
    if settings.groq_max_tokens_per_request > 0:
        return settings.groq_max_tokens_per_request

    model = settings.groq_model_id.lower()
    if model in GROQ_MODEL_TOKEN_LIMITS:
        return GROQ_MODEL_TOKEN_LIMITS[model]

    for model_id, limit in GROQ_MODEL_TOKEN_LIMITS.items():
        if model_id in model or model.endswith(model_id.split("/")[-1]):
            return limit

    return 10_000


def _max_prompt_chars_for_provider() -> int:
    if settings.llm_provider.lower() == "groq":
        return int(_groq_max_tokens_per_request() * CHARS_PER_TOKEN_ESTIMATE)
    return settings.llm_max_prompt_chars


def _max_document_chars_per_request(schema: dict[str, Any]) -> int:
    """Keep each LLM request under the configured prompt size budget."""
    overhead = _prompt_overhead_chars(schema)
    budget = max(2_000, _max_prompt_chars_for_provider() - overhead)
    return budget


def _split_document_text(
    text: str,
    max_chars: int,
    overlap: int,
) -> list[str]:
    """Split long documents on paragraph/line/word boundaries with overlap."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary <= start:
                boundary = text.rfind("\n", start, end)
            if boundary <= start:
                boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks or [text]


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


_LIST_KEY_CANDIDATES = (
    "standard_id",
    "element_id",
    "factor_number",
    "id",
    "survey_type",
    "name",
)


def _item_key(item: dict[str, Any]) -> str | None:
    for key in _LIST_KEY_CANDIDATES:
        if key in item and item[key] is not None:
            return f"{key}:{item[key]}"
    return None


def _merge_lists(left: list[Any], right: list[Any]) -> list[Any]:
    if not left:
        return list(right)
    if not right:
        return list(left)

    if not all(isinstance(item, dict) for item in left + right):
        merged: list[Any] = []
        seen: set[Any] = set()
        for item in left + right:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        return merged

    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in left + right:
        key = _item_key(item)
        if key is None:
            key = f"__anon_{len(order)}"
        if key not in by_key:
            by_key[key] = dict(item)
            order.append(key)
        else:
            by_key[key] = _deep_merge(by_key[key], item)
    return [by_key[key] for key in order]


def _deep_merge(left: Any, right: Any) -> Any:
    if _is_empty(left):
        return right
    if _is_empty(right):
        return left

    if isinstance(left, dict) and isinstance(right, dict):
        merged = dict(left)
        for key, value in right.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    if isinstance(left, list) and isinstance(right, list):
        return _merge_lists(left, right)

    if isinstance(left, str) and isinstance(right, str):
        return right if len(right) > len(left) else left

    return right


def _is_payload_too_large(exc: Exception) -> bool:
    message = str(exc).lower()
    return "413" in message or "too large" in message or "payload too large" in message


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send *prompt* and return the raw text response."""


# ── Groq backend ────────────────────────────────────────────────────────

class GroqClient(BaseLLMClient):
    """
    Calls the Groq Inference API.
    """

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Get one at https://console.groq.com and add it to your .env file."
            )
        self.model_id = settings.groq_model_id
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_not_exception_type(PayloadTooLargeError),
    )
    def complete(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,       # near-deterministic for extraction
                max_tokens=2048,
            )
        except APIConnectionError as exc:
            raise RuntimeError(
                "Network error contacting Groq API. "
                "Check your internet connectivity."
            ) from exc
        except AuthenticationError as exc:
            raise RuntimeError(
                "Invalid GROQ_API_KEY. Verify it at https://console.groq.com."
            ) from exc
        except APIError as exc:
            if _is_payload_too_large(exc):
                raise PayloadTooLargeError(
                    f"Groq request too large for model '{self.model_id}'. "
                    f"Prompt was {len(prompt)} chars."
                ) from exc
            raise RuntimeError(f"Groq API error: {exc}") from exc

        return completion.choices[0].message.content.strip()


# ── HuggingFace backend ────────────────────────────────────────────────────────

class HuggingFaceClient(BaseLLMClient):
    """
    Uses the HuggingFace Inference API (free tier).
    Model must support the text-generation task.
    """

    BASE_URL = "https://api-inference.huggingface.co/models"

    def __init__(self):
        if not settings.HF_API_TOKEN:
            raise RuntimeError(
                "HF_API_TOKEN is not set. "
                "Add it to your .env file or environment variables."
            )
        self.model_id = settings.hf_model_id
        # self.headers = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}
        print("Using HuggingFace token:", settings.HF_API_TOKEN)
        self.client = InferenceClient(token=settings.HF_API_TOKEN)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def complete(self, prompt: str) -> str:

        # Using HuggingFace URL
        #------------------------------------------
        # url = f"{self.BASE_URL}/{self.model_id}"

        # payload = {
        #     "inputs": prompt,
        #     "parameters": {
        #         "max_new_tokens": 2048,
        #         "temperature": 0.1,       # near-deterministic for extraction
        #         "return_full_text": False, # return only generated tokens
        #         "do_sample": False,
        #     },
        # }

        # logger.debug("Calling HuggingFace API: model=%s", self.model_id)
        # response = requests.post(url, headers=self.headers, json=payload, timeout=120)

        # if response.status_code == 503:
        #     # Model is loading — tenacity will retry
        #     raise RuntimeError(f"Model '{self.model_id}' is loading. Retrying…")

        # response.raise_for_status()

        # result = response.json()
        #------------------------------------------

        # Using HuggingFace API Client
        #------------------------------------------
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            print("ERROR::: Error at chat completion create function.....")
            print(exc)
            raise RuntimeError(
                "Network error contacting HuggingFace Inference API. "
                "Check internet/DNS connectivity, your network/proxy settings, "
                "or switch to a local provider with LLM_PROVIDER=llama_local."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "HuggingFace API request failed. Verify your HF_API_TOKEN and model ID."
            ) from exc

        if hasattr(completion, "choices") and completion.choices:
            return completion.choices[0].message.content.strip()

        if isinstance(completion, list) and completion:
            return completion[0].get("generated_text", "").strip()

        raise RuntimeError(f"Unexpected HuggingFace response format: {completion}")


# ── LLaMA local backend ────────────────────────────────────────────────────────

class LlamaLocalClient(BaseLLMClient):
    """
    Calls an OpenAI-compatible /v1/chat/completions endpoint.
    Works with Ollama, llama.cpp server, vLLM, LM Studio, etc.
    """

    def __init__(self):
        self.base_url = settings.llama_base_url.rstrip("/")
        self.model_id = settings.llama_model_id

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    def complete(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        logger.debug("Calling local LLaMA endpoint: %s", url)
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]


# ── Factory ────────────────────────────────────────────────────────────────────

def get_llm_client() -> BaseLLMClient:
    provider = settings.llm_provider.lower()
    if provider == "huggingface":
        return HuggingFaceClient()
    if provider == "llama_local":
        return LlamaLocalClient()
    if provider == "groq":
        return GroqClient()
    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        "Valid options: 'huggingface', 'llama_local', 'groq'."
    )


# ── JSON extraction helper ─────────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_from_response(raw: str) -> dict[str, Any]:
    """
    Parse JSON from the LLM response, tolerating common formatting noise
    (markdown fences, leading/trailing prose, BOM, etc.).
    """
    text = raw.strip().lstrip("\ufeff")

    # Try direct parse first (model returned clean JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Find the first { … } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "LLM did not return valid JSON.\n"
        f"Raw response (first 500 chars):\n{raw[:500]}"
    )


def _sleep_for_groq(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("Sleeping for 5 seconds")
        time.sleep(5)
        result = func(*args, **kwargs)
        logger.info("Done sleeping")
        return result
    return wrapper


# ── Main public function ───────────────────────────────────────────────────────
@_sleep_for_groq
def _extract_single_chunk(
    client: BaseLLMClient,
    document_text: str,
    schema: dict[str, Any],
    *,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
) -> tuple[dict[str, Any], str]:
    prompt = build_prompt(
        document_text,
        schema,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )
    logger.info("Sending extraction prompt (%d chars) to LLM", len(prompt))
    raw = client.complete(prompt)
    logger.info("LLM responded (%d chars)", len(raw))
    logger.debug("Raw LLM output:\n%s", raw)
    return extract_json_from_response(raw), raw


def run_extraction(
    document_text: str,
    schema: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """
    Run LLM extraction.

    Large documents are split into chunks that fit within the provider's
    per-request size limits, then partial results are merged.

    Returns:
        (parsed_dict, raw_llm_output)
    """
    client = get_llm_client()
    max_doc_chars = _max_document_chars_per_request(schema)
    overlap = settings.llm_chunk_overlap_chars
    chunk_size = max_doc_chars

    if settings.llm_provider.lower() == "groq":
        logger.info(
            "Groq model=%s token budget=%d (~%d prompt chars)",
            settings.groq_model_id,
            _groq_max_tokens_per_request(),
            _max_prompt_chars_for_provider(),
        )

    while True:
        chunks = _split_document_text(document_text, chunk_size, overlap)
        try:
            if len(chunks) == 1:
                return _extract_single_chunk(client, chunks[0], schema)

            logger.info(
                "Document split into %d chunks (max %d chars each, overlap %d)",
                len(chunks),
                chunk_size,
                overlap,
            )

            merged: dict[str, Any] | None = None
            raw_parts: list[str] = []
            for index, chunk in enumerate(chunks):
                logger.info("Processing chunk %d of %d", index, len(chunks))
                # if index > 0 and settings.llm_provider.lower() == "groq":
                #     # logger.info("Sleeping for 5 seconds")
                #     # Spread requests to stay under Groq's per-minute token cap.
                #     time.sleep(2)

                parsed, raw = _extract_single_chunk(
                    client,
                    chunk,
                    schema,
                    chunk_index=index,
                    total_chunks=len(chunks),
                )
                merged = parsed if merged is None else _deep_merge(merged, parsed)
                raw_parts.append(raw)

            return merged or {}, "\n---\n".join(raw_parts)

        except (PayloadTooLargeError, RetryError) as exc:
            cause = exc if isinstance(exc, PayloadTooLargeError) else exc.last_attempt.exception()
            if cause is None or not _is_payload_too_large(cause) or chunk_size <= 4_000:
                raise
            chunk_size = max(4_000, chunk_size // 2)
            logger.warning(
                "LLM payload too large; retrying with smaller chunks (%d chars)",
                chunk_size,
            )
