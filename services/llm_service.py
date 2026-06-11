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
# These are INPUT token budgets — set conservatively to avoid 429s.
GROQ_MODEL_TOKEN_LIMITS: dict[str, int] = {
    "llama-3.3-70b-versatile": 11_000,
    "llama-3.1-8b-instant": 5_500,
    "meta-llama/llama-4-scout-17b-16e-instruct": 28_000,
    "openai/gpt-oss-20b": 7_500,
    "openai/gpt-oss-120b": 7_500,
    "groq/compound": 65_000,
    "groq/compound-mini": 65_000,
}

# Output token budget for chat completions (Groq + HuggingFace).
GROQ_MAX_COMPLETION_TOKENS = 4_096
HF_MAX_COMPLETION_TOKENS = 4_096

CHARS_PER_TOKEN_ESTIMATE = 4.5

# Inter-chunk delay (seconds) to stay under Groq's per-minute token cap.
# Only applied between chunks (not before the first one).
GROQ_INTER_CHUNK_DELAY_SECS = 5


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

    Design principles:
    - System-style framing first so the model locks into extraction mode.
    - Schema is compacted to save tokens.
    - Explicit rules about null handling, key completeness, and output format
      reduce validation failures caused by the model omitting keys or
      wrapping output in markdown.
    """
    schema_str = json.dumps(schema, separators=(",", ":"))

    chunk_note = ""
    if chunk_index is not None and total_chunks is not None and total_chunks > 1:
        chunk_note = f"""\
## Document section
This is section {chunk_index + 1} of {total_chunks} of the full document.
- Extract ONLY information present in THIS section.
- For schema fields not found in this section, output `null` — do NOT omit the key.
- Do not carry over values from memory or other sections.
"""

    return f"""\
You are a precise JSON data-extraction engine. Your only output is a single \
valid JSON object — never markdown, never explanation, never code fences.

## Task
Read the DOCUMENT below and fill every field described in the JSON SCHEMA.

## Critical rules
1. OUTPUT FORMAT: Return ONLY the raw JSON object. No ```json fences, \
no preamble, no trailing text. The very first character of your response \
must be `{{` and the last must be `}}`.
2. KEY COMPLETENESS: Every key present in the schema MUST appear in your output. \
Never omit a key — use `null` when the value is not in the document.
3. NULL HANDLING: Use JSON `null` (not the string "null", not "N/A", not "") \
for missing values.
4. TYPES: Strictly follow each field's `type`. \
Strip currency symbols and thousand separators from numbers. \
Map "yes/no", "true/false", "enabled/disabled" to boolean `true`/`false`.
5. ENUMS: Only use values listed in the field's `enum` array. If the document \
value doesn't match any enum, use `null`.
6. NO INVENTION: Do not infer, assume, or hallucinate values absent from the document.
7. DESCRIPTIONS: Use the `description` hint on each field to resolve ambiguity.
{chunk_note}
## JSON Schema
{schema_str}

## Document
{document_text}

## Output (raw JSON only — starts with `{{`)
"""


def _prompt_overhead_chars(schema: dict[str, Any]) -> int:
    """Chars used by schema + instructions when the document body is empty."""
    return len(build_prompt("", schema))


def _groq_max_tokens_per_request(model_id: str | None = None) -> int:
    if settings.groq_max_tokens_per_request > 0 and model_id is None:
        return settings.groq_max_tokens_per_request

    model = (model_id or settings.groq_model_id).lower()
    if model in GROQ_MODEL_TOKEN_LIMITS:
        return GROQ_MODEL_TOKEN_LIMITS[model]

    for known_id, limit in GROQ_MODEL_TOKEN_LIMITS.items():
        if known_id in model or model.endswith(known_id.split("/")[-1]):
            return limit

    return 10_000


def _hf_inference_backend(model_id: str) -> str | None:
    """Return the inference backend suffix, e.g. 'groq' from 'model-id:groq'."""
    if ":" not in model_id:
        return None
    return model_id.rsplit(":", 1)[-1].lower()


def _max_prompt_chars_for_provider() -> int:
    llm_provider = settings.llm_provider.lower()

    if llm_provider == "groq":
        return int(_groq_max_tokens_per_request() * CHARS_PER_TOKEN_ESTIMATE)

    if llm_provider == "huggingface":
        backend = _hf_inference_backend(settings.hf_model_id)
        if backend == "groq":
            # HF routes to Groq — apply the same per-request token budget.
            return int(_groq_max_tokens_per_request(settings.hf_model_id) * CHARS_PER_TOKEN_ESTIMATE)

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


def _extract_chat_completion_text(completion: Any, *, provider: str) -> str:
    """
    Parse an OpenAI-style chat completion response.

    Both Groq and HuggingFace InferenceClient return:
        completion.choices[0].message.content
    """
    if not hasattr(completion, "choices") or not completion.choices:
        raise RuntimeError(f"Unexpected {provider} response format: {completion!r}")

    choice = completion.choices[0]
    message = choice.message
    content = message.content if hasattr(message, "content") else None

    if not content or not str(content).strip():
        raise RuntimeError(f"Empty response from {provider}.")

    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        logger.warning(
            "%s response was truncated (finish_reason=length). "
            "Extracted JSON may be incomplete.",
            provider,
        )

    return str(content).strip()


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    """
    Provider-agnostic interface for LLM text completion.

    Every backend (Groq, HuggingFace, local LLaMA) implements `complete(prompt)`.
    The rest of the app — build_prompt → complete → extract_json_from_response —
    never imports provider-specific SDKs.
    """

    @abstractmethod
    def complete(self, prompt: str, document_text: str) -> str:
        """Send *prompt* and return the raw text response."""


# ── Groq backend ────────────────────────────────────────────────────────

class GroqClient(BaseLLMClient):
    """
    Calls the Groq Inference API.

    Recommended model: llama-3.3-70b-versatile
      - Best instruction-following + JSON fidelity on free tier
      - 6k tokens/request input, 8k output ceiling
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
    def complete(self, prompt: str, document_text: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,           # fully deterministic for extraction
                max_tokens=GROQ_MAX_COMPLETION_TOKENS,
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

        return _extract_chat_completion_text(completion, provider="Groq")


# ── HuggingFace backend ────────────────────────────────────────────────────────

class HuggingFaceClient(BaseLLMClient):
    """
    Uses the HuggingFace Inference API via InferenceClient.

    Follows the official HF pattern:
        client = InferenceClient(api_key=HF_TOKEN)
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct:groq",
            messages=[{"role": "user", "content": prompt}],
        )
        text = completion.choices[0].message.content

  Model IDs may include a backend suffix (e.g. `:groq`) to route through
  HuggingFace's serverless inference partners.
    """

    def __init__(self):
        if not settings.HF_API_TOKEN:
            raise RuntimeError(
                "HF_API_TOKEN is not set. "
                "Get one at https://huggingface.co/settings/tokens and add it to .env."
            )
        self.model_id = settings.hf_model_id
        self.client = InferenceClient(api_key=settings.HF_API_TOKEN)
        logger.info("HuggingFace InferenceClient ready (model=%s)", self.model_id)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_not_exception_type(PayloadTooLargeError),
    )
    def complete(self, prompt: str, document_text: str) -> str:
        try:
            logger.info("Into complete method....HuggingFace InferenceClient ready (model=%s)", self.model_id)
            
            # if self.model_id == "Qwen/Qwen2.5-7B-Instruct":
            #     completion = self.client.chat.completions.create(
            #         model=self.model_id,
            #         messages=[{"role": "system", "content": prompt},
            #         {"role": "user", "content": document_text}],
            #         temperature=0.0,
            #         max_tokens=HF_MAX_COMPLETION_TOKENS,
            #     )
            # else:
            completion = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=HF_MAX_COMPLETION_TOKENS,
                    )
        except Exception as exc:
            if _is_payload_too_large(exc):
                raise PayloadTooLargeError(
                    f"HuggingFace request too large for model '{self.model_id}'. "
                    f"Prompt was {len(prompt)} chars."
                ) from exc
            raise RuntimeError(
                "HuggingFace Inference API call failed. "
                "Verify HF_API_TOKEN and hf_model_id in .env."
            ) from exc

        return _extract_chat_completion_text(completion, provider="HuggingFace")


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
    def complete(self, prompt: str, document_text: str) -> str:
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 4096,
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


# ── Main public function ───────────────────────────────────────────────────────

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
    raw = client.complete(prompt, document_text)
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
    llm_provider = settings.llm_provider.lower()
    is_groq = llm_provider == "groq"
    is_hf_groq = (
        llm_provider == "huggingface"
        and _hf_inference_backend(settings.hf_model_id) == "groq"
    )
    needs_chunk_delay = is_groq or is_hf_groq

    if is_groq:
        logger.info(
            "Groq model=%s token budget=%d (~%d prompt chars)",
            settings.groq_model_id,
            _groq_max_tokens_per_request(),
            _max_prompt_chars_for_provider(),
        )
    elif is_hf_groq:
        logger.info(
            "HuggingFace model=%s (groq backend) token budget=%d (~%d prompt chars)",
            settings.hf_model_id,
            _groq_max_tokens_per_request(settings.hf_model_id),
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
                # Inter-chunk delay on Groq to stay under per-minute token cap.
                # Applied BETWEEN chunks only (not before the first one).
                if index > 0 and needs_chunk_delay:
                    logger.info(
                        "Chunk %d/%d — waiting %ds before next request",
                        index,
                        len(chunks),
                        GROQ_INTER_CHUNK_DELAY_SECS,
                    )
                    time.sleep(GROQ_INTER_CHUNK_DELAY_SECS)

                logger.info("Processing chunk %d of %d", index + 1, len(chunks))
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