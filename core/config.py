from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # LLM provider: "huggingface" | "llama_local | "groq"
    llm_provider: str = "groq"

    # HuggingFace
    HF_API_TOKEN: str = ""
    # hf_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_model_id: str = "meta-llama/Llama-3.3-70B-Instruct"
    # hf_model_id: str = "microsoft/Phi-3-mini-4k-instruct"

    # GROQ — Llama 4 Scout allows 30k TPM vs 12k for llama-3.3-70b-versatile (free tier)
    groq_model_id: str = "llama-3.3-70b-versatile"
    GROQ_API_KEY: str = ""
    # 0 = auto from groq_model_id; set explicitly to override token budget per request
    groq_max_tokens_per_request: int = 0

    # LLaMA local
    llama_base_url: str = "http://localhost:11434/v1"
    llama_model_id: str = "llama3"

    # LLM chunking (non-Groq providers; Groq uses groq_max_tokens_per_request)
    llm_max_prompt_chars: int = 90_000
    llm_chunk_overlap_chars: int = 1_000

    # API
    max_file_size_mb: int = 20
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
