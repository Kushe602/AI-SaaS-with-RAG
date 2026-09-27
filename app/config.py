"""Application configuration, loaded from environment variables / .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "DocuChat"
    secret_key: str = "change-me-please"
    access_token_expire_minutes: int = 60 * 24 * 7  # one week

    # Database
    database_url: str = "sqlite+aiosqlite:///./docuchat.db"

    # LLM (OpenAI-compatible: works with any provider/gateway that speaks the
    # OpenAI API — OpenAI, Groq, Together, OpenRouter, a local server, etc.)
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.justwoker.icu/v1"
    llm_model: str = "gpt-4o-mini"
    max_context_chunks: int = 6
    max_answer_tokens: int = 1024
    # Keyless demo answers: assemble a cited answer from the retrieved passages
    # instead of calling the model (used by the public Render demo).
    use_fake_llm: bool = False

    # Embeddings
    use_fake_embeddings: bool = False
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Uploads
    max_upload_mb: int = 10

    # Plan limits (usage metering)
    free_max_documents: int = 5
    free_daily_questions: int = 25
    pro_max_documents: int = 100
    pro_daily_questions: int = 1000


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
