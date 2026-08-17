from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_timeout: int = 120
    ollama_keep_alive: str = "10m"

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # Anthropic
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Router
    default_task_type: str = "general"
    default_model_provider: str = "ollama"

    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000

    # Guardrails
    enable_input_guardrails: bool = True
    enable_output_guardrails: bool = True

    # Embeddings
    embedding_provider: str = "ollama"  # or openai, etc.
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # RAG
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    rag_top_k: int = 5
    vector_store_type: str = "faiss"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()