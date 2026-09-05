from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


class ProviderName(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROQ = "groq"
    TOGETHER = "together"
    OPENROUTER = "openrouter"


class ProviderConfig(BaseModel):
    name: ProviderName
    enabled: bool = True
    priority: int = 100
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: str
    models: List[str] = []
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    supports_embeddings: bool = False
    supports_audio: bool = False
    supports_image_gen: bool = False
    max_tokens: Optional[int] = None
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    timeout: int = 60
    max_retries: int = 3


class ProviderHealth(BaseModel):
    provider: ProviderName
    status: Literal["healthy", "degraded", "unhealthy", "unknown"]
    latency_ms: Optional[float] = None
    error_rate: float = 0.0
    last_check: Optional[int] = None
    last_error: Optional[str] = None
    models_available: List[str] = []


class ModelInfo(BaseModel):
    id: str
    provider: ProviderName
    name: str
    description: Optional[str] = None
    context_window: int
    max_output_tokens: int
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    supports_embeddings: bool = False
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    capabilities: List[str] = []
    tags: List[str] = []


class ProviderListResponse(BaseModel):
    providers: List[ProviderConfig]
    default: Optional[ProviderName] = None