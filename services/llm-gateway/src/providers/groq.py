import structlog
import time
from typing import List, Optional, AsyncIterator, Dict, Any
from groq import AsyncGroq

from libs.llm_common.base import BaseProvider, ProviderCapabilities
from libs.schemas_common.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ChatChoice,
    ChatStreamChoice,
    ChatDelta,
    UsageInfo,
    MessageRole,
    ToolDefinition,
    ToolCall,
    FunctionCall,
)
from libs.schemas_common.embeddings import EmbeddingRequest, EmbeddingResponse
from libs.schemas_common.providers import ProviderName, ModelInfo
from ..config.settings import settings

logger = structlog.get_logger()


class GroqProvider(BaseProvider):
    """Groq provider for fast LLM inference."""

    def __init__(
        self,
        api_key: str = None,
        default_model: str = None,
        timeout: int = None,
        max_retries: int = None,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key or settings.groq_api_key,
            base_url=None,
            default_model=default_model or settings.groq_model,
            timeout=timeout or settings.groq_timeout,
            max_retries=max_retries or settings.groq_max_retries,
            **kwargs,
        )
        self._client: Optional[AsyncGroq] = None
        self._models_cache: List[ModelInfo] = []
        self._capabilities = ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_vision=False,
            supports_embeddings=False,
            supports_audio=False,
            supports_image_gen=False,
            max_context_window=131072,
            max_output_tokens=8192,
        )

    @property
    def provider_name(self) -> ProviderName:
        return ProviderName.GROQ

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def models(self) -> List[ModelInfo]:
        return self._models_cache

    async def initialize(self) -> None:
        """Initialize Groq client."""
        self._client = AsyncGroq(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        await self._fetch_models()
        logger.info("groq_provider_initialized", models=len(self._models_cache))

    async def close(self) -> None:
        """Close connections."""
        logger.info("groq_provider_closed")

    async def _fetch_models(self) -> None:
        """Define available Groq models."""
        self._models_cache = [
            ModelInfo(
                id="llama-3.1-70b-versatile",
                provider=ProviderName.GROQ,
                name="Llama 3.1 70B Versatile",
                description="High-quality general purpose",
                context_window=131072,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                input_cost_per_1k=0.59,
                output_cost_per_1k=0.79,
                tags=["chat", "tools", "fast"],
            ),
            ModelInfo(
                id="llama-3.1-8b-instant",
                provider=ProviderName.GROQ,
                name="Llama 3.1 8B Instant",
                description="Ultra-fast inference",
                context_window=131072,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                input_cost_per_1k=0.05,
                output_cost_per_1k=0.08,
                tags=["chat", "tools", "ultra_fast"],
            ),
            ModelInfo(
                id="mixtral-8x7b-32768",
                provider=ProviderName.GROQ,
                name="Mixtral 8x7B",
                description="Mixture of experts",
                context_window=32768,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                input_cost_per_1k=0.24,
                output_cost_per_1k=0.24,
                tags=["chat", "tools", "moe"],
            ),
            ModelInfo(
                id="gemma2-9b-it",
                provider=ProviderName.GROQ,
                name="Gemma 2 9B",
                description="Google's efficient model",
                context_window=8192,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                input_cost_per_1k=0.10,
                output_cost_per_1k=0.10,
                tags=["chat", "tools", "fast"],
            ),
        ]

    async def health_check(self) -> bool:
        """Check Groq health."""
        try:
            await self._client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False

    async def list_models(self) -> List[ModelInfo]:
        """List available models."""
        return self._models_cache

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send chat completion request."""
        model = request.model or self.default_model
        messages = self._convert_messages(request.messages)

        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=False,
            stop=request.stop,
            tools=self._convert_tools(request.tools) if request.tools else None,
            tool_choice=request.tool_choice,
        )

        return self._build_chat_response(response, model)

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream chat completion."""
        model = request.model or self.default_model
        messages = self._convert_messages(request.messages)

        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=True,
            stop=request.stop,
            tools=self._convert_tools(request.tools) if request.tools else None,
            tool_choice=request.tool_choice,
        )

        async for chunk in stream:
            yield self._build_stream_chunk(chunk, model)

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert internal messages to Groq/OpenAI format."""
        converted = []
        for msg in messages:
            groq_msg = {"role": msg.role.value}
            if msg.content is not None:
                groq_msg["content"] = msg.content
            if msg.name:
                groq_msg["name"] = msg.name
            if msg.tool_call_id:
                groq_msg["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                groq_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type.value,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
            converted.append(groq_msg)
        return converted

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert tools to Groq/OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters,
                    "strict": t.function.strict,
                }
            }
            for t in tools
        ]

    def _build_chat_response(self, response, model: str) -> ChatResponse:
        """Build ChatResponse from Groq response."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    function=FunctionCall(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )
                for tc in message.tool_calls
            ]

        return ChatResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            provider="groq",
            choices=[
                ChatChoice(
                    index=choice.index,
                    message=ChatMessage(
                        role=MessageRole(message.role),
                        content=message.content,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=choice.finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            ),
        )

    def _build_stream_chunk(self, chunk, model: str) -> ChatStreamChunk:
        """Build stream chunk from Groq streaming response."""
        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            return None

        delta = choice.delta
        tool_calls = None
        if delta.tool_calls:
            tool_calls = [
                {
                    "index": tc.index,
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in delta.tool_calls
            ]

        return ChatStreamChunk(
            id=chunk.id,
            created=chunk.created,
            model=chunk.model,
            provider="groq",
            choices=[
                ChatStreamChoice(
                    index=choice.index,
                    delta=ChatDelta(
                        role=MessageRole(delta.role) if delta.role else None,
                        content=delta.content,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=choice.finish_reason,
                )
            ],
        )

    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Groq doesn't support embeddings natively."""
        raise NotImplementedError("Groq does not support embeddings")

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str = None) -> float:
        """Calculate cost in USD."""
        model = model or self.default_model
        model_info = self.get_model_info(model)
        if model_info:
            return (prompt_tokens / 1000 * model_info.input_cost_per_1k) + \
                   (completion_tokens / 1000 * model_info.output_cost_per_1k)
        return 0.0