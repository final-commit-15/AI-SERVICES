from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List
from pydantic import Field


class Settings(BaseSettings):
    # Application
    app_name: str = "agentforge-ai-services"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    api_prefix: str = "/v1"
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "agentforge"
    postgres_user: str = "agentforge"
    postgres_password: str = "changeme"
    database_url: str = "postgresql+asyncpg://agentforge:changeme@localhost:5432/agentforge"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: Optional[str] = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_prefix: str = "agentforge"

    # LLM Providers
    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_vision_model: str = "llava:7b"
    ollama_timeout: int = 120
    ollama_keep_alive: str = "10m"
    ollama_num_gpu: int = 1
    ollama_num_thread: int = 4

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_org_id: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_max_retries: int = 3
    openai_timeout: int = 60

    # Anthropic
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    anthropic_max_retries: int = 3
    anthropic_timeout: int = 60

    # Google Gemini
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-1.5-flash"
    gemini_embedding_model: str = "text-embedding-004"
    gemini_max_retries: int = 3
    gemini_timeout: int = 60

    # Groq
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-70b-versatile"
    groq_max_retries: int = 3
    groq_timeout: int = 30

    # Together AI
    together_api_key: Optional[str] = None
    together_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    together_embedding_model: str = "togethercomputer/m2-bert-80M-8k-retrieval"
    together_max_retries: int = 3
    together_timeout: int = 60

    # OpenRouter
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "anthropic/claude-3.5-sonnet"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_max_retries: int = 3
    openrouter_timeout: int = 60

    # Model Router
    default_task_type: str = "general"
    default_model_provider: str = "ollama"
    router_enable_fallback: bool = True
    router_fallback_chain: str = "ollama,openai,anthropic,groq,together,openrouter"
    router_cost_optimization: bool = True
    router_latency_optimization: bool = False

    # Task-specific models
    router_chat_model: str = "ollama:qwen2.5:7b"
    router_coding_model: str = "openai:gpt-4o"
    router_reasoning_model: str = "anthropic:claude-3.5-sonnet-20241022"
    router_summarization_model: str = "ollama:qwen2.5:7b"
    router_vision_model: str = "ollama:llava:7b"
    router_embedding_model: str = "ollama:nomic-embed-text"
    router_speech_model: str = "openai:whisper-1"
    router_image_gen_model: str = "openai:dall-e-3"

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 10000
    cache_redis_enabled: bool = True

    # Guardrails
    enable_input_guardrails: bool = True
    enable_output_guardrails: bool = True
    guardrails_max_input_length: int = 10000
    guardrails_max_output_length: int = 50000
    guardrails_block_injection: bool = True
    guardrails_block_pii: bool = True
    guardrails_block_toxicity: bool = True

    # Embeddings
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    embedding_batch_size: int = 100
    embedding_cache_enabled: bool = True

    # RAG
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    rag_top_k: int = 5
    rag_rerank_enabled: bool = True
    rag_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_hybrid_search: bool = True
    rag_vector_store: str = "qdrant"
    rag_collection_name: str = "documents"

    # Conversation Memory
    memory_enabled: bool = True
    memory_backend: str = "redis"
    memory_short_term_ttl: int = 3600
    memory_long_term_ttl: int = 2592000
    memory_max_messages: int = 50
    memory_summarization_enabled: bool = True
    memory_summarization_model: str = "ollama:qwen2.5:7b"

    # Tools
    tools_enabled: bool = True
    tools_timeout: int = 30
    tools_max_iterations: int = 10

    # Speech
    speech_provider: str = "openai"
    speech_stt_model: str = "whisper-1"
    speech_tts_model: str = "tts-1"
    speech_tts_voice: str = "alloy"
    speech_max_file_size: str = "25MB"

    # Image Generation
    image_gen_provider: str = "openai"
    image_gen_model: str = "dall-e-3"
    image_gen_size: str = "1024x1024"
    image_gen_quality: str = "standard"
    image_gen_style: str = "vivid"

    # Health Monitoring
    health_check_enabled: bool = True
    health_check_interval: int = 30
    health_check_timeout: int = 10
    health_check_failure_threshold: int = 3

    # Analytics
    analytics_enabled: bool = True
    analytics_retention_days: int = 90
    cost_tracking_enabled: bool = True
    cost_alert_threshold_usd: float = 100.0

    # Authentication
    auth_enabled: bool = True
    jwt_secret_key: str = "your-super-secret-jwt-key-change-in-production-min-32-chars"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    api_key_enabled: bool = True
    api_key_header: str = "X-API-Key"

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_requests_per_hour: int = 1000
    rate_limit_burst: int = 10

    # Admin
    admin_api_enabled: bool = True
    admin_api_key: str = "admin-secret-key-change-in-production"
    admin_rate_limit: int = 1000

    # WebSocket
    ws_enabled: bool = True
    ws_heartbeat_interval: int = 30
    ws_max_connections: int = 1000

    # Streaming
    streaming_enabled: bool = True
    streaming_chunk_size: int = 1024
    streaming_timeout: int = 300

    # File Upload
    upload_max_size: str = "50MB"
    upload_allowed_types: str = "pdf,txt,md,docx,html,json,csv"
    upload_dir: str = "/tmp/uploads"

    # External
    sentry_dsn: Optional[str] = None
    datadog_api_key: Optional[str] = None
    datadog_app_key: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()