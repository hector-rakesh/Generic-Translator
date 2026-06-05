from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # LLM provider: "huggingface" | "llama_local"
    llm_provider: str = "huggingface"

    # HuggingFace
    hf_api_token: str = ""
    # hf_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_model_id: str = "meta-llama/Llama-3.3-70B-Instruct"

    # LLaMA local
    llama_base_url: str = "http://localhost:11434/v1"
    llama_model_id: str = "llama3"

    # API
    max_file_size_mb: int = 20
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
