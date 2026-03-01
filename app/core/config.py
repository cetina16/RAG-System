from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    mistral_api_key: str = ""
    mistral_model: str = "mistral-large-latest"
    mistral_api_base: str = "https://api.mistral.ai/v1"

    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_batch_size: int = 32

    # Vector store
    faiss_index_path: str = "data/faiss_index"

    # Retrieval
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60

    # Cache
    cache_max_size: int = 256
    cache_ttl_seconds: int = 300

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
