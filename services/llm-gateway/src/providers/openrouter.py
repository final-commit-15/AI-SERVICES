import structlog
import time
from typing import List, Optional, AsyncIterator, Dict, Any
from openai import AsyncOpenAI

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


class OpenRouterProvider(BaseProvider):
    """OpenRouter provider for accessing multiple models."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        default_model: str = None,
        timeout: int = None,
        max_retries: int = None,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key or settings.openrouter_api_key,
            base_url=base_url or settings.openrouter_base_url,
            default_model=default_model or settings.openrouter_model,
            timeout=timeout or settings.openrouter_timeout,
            max_retries=max_retries or settings.openrouter_max_retries,
            **kwargs,
        )
        self._client: Optional[AsyncOpenAI] = None
        self._models_cache: List[ModelInfo] = []
        self._capabilities = ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            supports_embeddings=False,
            supports_audio=False,
            supports_image_gen=False,
            max_context_window=128000,
            max_output_tokens=4096,
        )

    @property
    def provider_name(self) -> ProviderName:
        return ProviderName.OPENROUTER

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def models(self) -> List[ModelInfo]:
        return self._models_cache

    async def initialize(self) -> None:
        """Initialize OpenRouter client."""
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers={
                "HTTP-Referer": "https://agentforge.ai",
                "X-Title": "AgentForge AI Services",
            }
        )
        await self._fetch_models()
        logger.info("openrouter_provider_initialized", models=len(self._models_cache))

    async def close(self) -> None:
        """Close connections."""
        if self._client:
            await self._client.close()
        logger.info("openrouter_provider_closed")

    async def _fetch_models(self) -> None:
        """Fetch available models from OpenRouter."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                data = resp.json()
                self._models_cache = []
                for model_data in data.get("data", []):
                    model_id = model_data.get("id", "")
                    pricing = model_data.get("pricing", {})
                    self._models_cache.append(ModelInfo(
                        id=model_id,
                        provider=ProviderName.OPENROUTER,
                        name=model_data.get("name", model_id),
                        description=model_data.get("description", ""),
                        context_window=model_data.get("context_length", 4096),
                        max_output_tokens=model_data.get("top_provider", {}).get("max_completion_tokens", 4096),
                        supports_streaming=model_data.get("supported_parameters", {}).get("stream", True),
                        supports_tools="tools" in model_data.get("supported_parameters", []),
                        supports_vision="vision" in model_data.get("architecture", {}).get("input_modalities", []),
                        input_cost_per_1k=float(pricing.get("prompt", 0)) * 1000,
                        output_cost_per_1k=float(pricing.get("completion", 0)) * 1000,
                        tags=model_data.get("tags", []),
                    ))
        except Exception as e:
            logger.warning("failed_to_fetch_openrouter_models", error=str(e))
            # Default models
            self._models_cache = [
                ModelInfo(
                    id=self.default_model,
                    provider=ProviderName.OPENROUTER,
                    name=self.default_model,
                    context_window=128000,
                    max_output_tokens=4096,
                    supports_streaming=True,
                    supports_tools=True,
                    supports_vision=True,
                )
            ]

    async def health_check(self) -> bool:
        """Check OpenRouter health."""
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
        await self._fetch_models()
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
            extra_headers={
                "HTTP-Referer": "https://agentforge.ai",
                "X-Title": "AgentForge AI Services",
            },
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
            extra_headers={
                "HTTP-Referer": "https://agentforge.ai",
                "X-Title": "AgentForge AI Services",
            },
        )

        async for chunk in stream:
            yield self._build_stream_chunk(chunk, model)

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert internal messages to OpenRouter/OpenAI format."""
        converted = []
        for msg in messages:
            or_msg = {"role": msg.role.value}
            if msg.content is not None:
                or_msg["content"] = msg.content
            if msg.name:
                or_msg["name"] = msg.name
            if msg.tool_call_id:
                or_msg["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                or_msg["tool_calls"] = [
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
            if msg.images:
                if isinstance(msg.content, str):
                    content = [{"type": "text", "text": msg.content}]
                else:
                    content = msg.content or []
                for img in msg.images:
                    content.append({"type": "image_url", "image_url": {"url": img}})
                or_msg["content"] = content
            converted.append(or_msg)
        return converted

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert tools to OpenRouter/OpenAI format."""
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
        """Build ChatResponse from OpenRouter response."""
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
            provider="openrouter",
            choices=[
                ChatChoice(
                    index=choice.index,
                    message=ChatMessage(
                        role=MessageRole(message.role),
                        content=message.content,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=choice.finish_reason,
                    logprobs=choice.logprobs.model_dump() if choice.logprobs else None,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            ),
        )

    def _build_stream_chunk(self, chunk, model: str) -> ChatStreamChunk:
        """Build stream chunk from OpenRouter streaming response."""
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
            provider="openrouter",
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
        """OpenRouter doesn't provide embeddings directly."""
        raise NotImplementedError("OpenRouter does not provide embeddings directly")

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str = None) -> float:
        """Calculate cost in USD."""
        model = model or self.default_model
        model_info = self.get_model_info(model)
        if model_info:
            return (prompt_tokens / 1000 * model_info.input_cost_per_1k) + \
                   (completion_tokens / 1000 * model_info.output_cost_per_1k)
        return 0.0