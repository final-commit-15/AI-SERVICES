from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator, Union
from dataclasses import dataclass

from libs.schemas_common.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ToolDefinition,
    ToolCall,
)
from libs.schemas_common.embeddings import EmbeddingRequest, EmbeddingResponse
from libs.schemas_common.providers import ProviderName, ModelInfo


@dataclass
class ProviderCapabilities:
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    supports_embeddings: bool = False
    supports_audio: bool = False
    supports_image_gen: bool = False
    supports_responses_api: bool = False
    max_context_window: int = 4096
    max_output_tokens: int = 4096


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "",
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = None
        self._capabilities = ProviderCapabilities()

    @property
    @abstractmethod
    def provider_name(self) -> ProviderName:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities."""
        pass

    @property
    @abstractmethod
    def models(self) -> List[ModelInfo]:
        """Return list of available models."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (create clients, verify connectivity)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        pass

    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """List available models."""
        pass

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat completion request."""
        pass

    @abstractmethod
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream chat completion."""
        pass

    @abstractmethod
    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings."""
        pass

    async def responses(self, request: "ResponsesRequest") -> "ResponsesResponse":
        """OpenAI-compatible Responses API (optional)."""
        raise NotImplementedError("Responses API not supported by this provider")

    async def responses_stream(self, request: "ResponsesRequest") -> AsyncIterator["ResponsesStreamEvent"]:
        """Stream Responses API (optional)."""
        raise NotImplementedError("Responses API streaming not supported by this provider")

    async def transcribe_audio(self, file: bytes, model: str, **kwargs) -> str:
        """Speech-to-text transcription (optional)."""
        raise NotImplementedError("Audio transcription not supported by this provider")

    async def synthesize_speech(self, text: str, model: str, voice: str, **kwargs) -> bytes:
        """Text-to-speech synthesis (optional)."""
        raise NotImplementedError("Speech synthesis not supported by this provider")

    async def generate_image(self, prompt: str, model: str, **kwargs) -> bytes:
        """Image generation (optional)."""
        raise NotImplementedError("Image generation not supported by this provider")

    async def edit_image(self, image: bytes, prompt: str, model: str, **kwargs) -> bytes:
        """Image editing (optional)."""
        raise NotImplementedError("Image editing not supported by this provider")

    async def create_variation(self, image: bytes, model: str, **kwargs) -> bytes:
        """Image variation (optional)."""
        raise NotImplementedError("Image variation not supported by this provider")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD."""
        return 0.0

    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """Get model information."""
        for m in self.models:
            if m.id == model:
                return m
        return None


from libs.schemas_common.responses import (
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
)