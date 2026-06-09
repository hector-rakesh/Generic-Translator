"""
LLM service — provider-agnostic.

Current backends:
  • huggingface  — HuggingFace Inference API (free tier)
  • llama_local  — OpenAI-compatible endpoint (Ollama / llama.cpp / vLLM)

Switching providers: change LLM_PROVIDER in .env.
No application code needs to change.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from huggingface_hub import InferenceClient
from groq import Groq, APIConnectionError, AuthenticationError, APIError

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(document_text: str, schema: dict[str, Any]) -> str:
    """
    Build the extraction prompt.

    The model is instructed to:
    1. Read the document.
    2. Extract values described in the JSON schema.
    3. Honour any `description` / `enum` / `x-*` hints in the schema.
    4. Return ONLY a valid JSON object — no markdown, no explanation.
    """
    schema_str = json.dumps(schema, indent=2)

    return f"""You are a precise data-extraction assistant.

## Task
Extract information from the DOCUMENT below and return a JSON object that
conforms exactly to the provided JSON SCHEMA.

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
        "Valid options: 'huggingface', 'llama_local'."
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

def run_extraction(
    document_text: str,
    schema: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """
    Run LLM extraction.

    Returns:
        (parsed_dict, raw_llm_output)
    """
    client = get_llm_client()
    prompt = build_prompt(document_text, schema)

    logger.info("Sending extraction prompt (%d chars) to LLM", len(prompt))
    raw = client.complete(prompt)
    logger.info("LLM responded (%d chars)", len(raw))
    logger.debug("Raw LLM output:\n%s", raw)

    parsed = extract_json_from_response(raw)
    return parsed, raw
