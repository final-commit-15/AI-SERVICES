import structlog
import time
from typing import List, Optional, AsyncIterator, Dict, Any
from together import AsyncTogether

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
from libs.schemas_common.embeddings import EmbeddingRequest, EmbeddingResponse, EmbeddingData
from libs.schemas_common.providers import ProviderName, ModelInfo
from ..config.settings import settings

logger = structlog.get_logger()


class TogetherProvider(BaseProvider):
    """Together AI provider."""

    def __init__(
        self,
        api_key: str = None,
        default_model: str = None,
        embedding_model: str = None,
        timeout: int = None,
        max_retries: int = None,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key or settings.together_api_key,
            base_url=None,
            default_model=default_model or settings.together_model,
            timeout=timeout or settings.together_timeout,
            max_retries=max_retries or settings.together_max_retries,
            **kwargs,
        )
        self.embedding_model = embedding_model or settings.together_embedding_model
        self._client: Optional[AsyncTogether] = None
        self._models_cache: List[ModelInfo] = []
        self._capabilities = ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_vision=False,
            supports_embeddings=True,
            supports_audio=False,
            supports_image_gen=False,
            max_context_window=128000,
            max_output_tokens=4096,
        )

    @property
    def provider_name(self) -> ProviderName:
        return ProviderName.TOGETHER

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def models(self) -> List[ModelInfo]:
        return self._models_cache

    async def initialize(self) -> None:
        """Initialize Together client."""
        self._client = AsyncTogether(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        await self._fetch_models()
        logger.info("together_provider_initialized", models=len(self._models_cache))

    async def close(self) -> None:
        """Close connections."""
        logger.info("together_provider_closed")

    async def _fetch_models(self) -> None:
        """Define available Together models."""
        self._models_cache = [
            ModelInfo(
                id="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                provider=ProviderName.TOGETHER,
                name="Llama 3.1 70B Instruct Turbo",
                description="High-quality Llama 3.1",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                input_cost_per_1k=0.88,
                output_cost_per_1k=0.88,
                tags=["chat", "tools", "llama"],
            ),
            ModelInfo(
                id="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                provider=ProviderName.TOGETHER,
                name="Llama 3.1 8B Instruct Turbo",
                description="Fast Llama 3.1",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                input_cost_per_1k=0.18,
                output_cost_per_1k=0.18,
                tags=["chat", "tools", "llama", "fast"],
            ),
            ModelInfo(
                id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                provider=ProviderName.TOGETHER,
                name="Mixtral 8x7B Instruct",
                description="Mixture of experts",
                context_window=32768,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                input_cost_per_1k=0.60,
                output_cost_per_1k=0.60,
                tags=["chat", "tools", "moe"],
            ),
            ModelInfo(
                id="Qwen/Qwen2.5-72B-Instruct-Turbo",
                provider=ProviderName.TOGETHER,
                name="Qwen 2.5 72B Instruct",
                description="Strong multilingual model",
                context_window=128000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                input_cost_per_1k=0.90,
                output_cost_per_1k=0.90,
                tags=["chat", "tools", "multilingual"],
            ),
            ModelInfo(
                id="togethercomputer/m2-bert-80M-8k-retrieval",
                provider=ProviderName.TOGETHER,
                name="M2-BERT 80M",
                description="Embedding model for retrieval",
                context_window=8192,
                max_output_tokens=0,
                supports_streaming=False,
                supports_tools=False,
                supports_embeddings=True,
                tags=["embedding", "retrieval"],
            ),
        ]

    async def health_check(self) -> bool:
        """Check Together health."""
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
        """Convert internal messages to Together/OpenAI format."""
        converted = []
        for msg in messages:
            together_msg = {"role": msg.role.value}
            if msg.content is not None:
                together_msg["content"] = msg.content
            if msg.name:
                together_msg["name"] = msg.name
            if msg.tool_call_id:
                together_msg["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                together_msg["tool_calls"] = [
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
            converted.append(together_msg)
        return converted

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert tools to Together/OpenAI format."""
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
        """Build ChatResponse from Together response."""
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
            provider="together",
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
        """Build stream chunk from Together streaming response."""
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
            provider="together",
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
        """Generate embeddings using Together."""
        model = request.model or self.embedding_model
        texts = request.input if isinstance(request.input, list) else [request.input]

        response = await self._client.embeddings.create(
            model=model,
            input=texts,
        )

        return EmbeddingResponse(
            data=[
                EmbeddingData(index=d.index, embedding=d.embedding)
                for d in response.data
            ],
            model=response.model,
            usage=UsageInfo(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            ),
        )

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str = None) -> float:
        """Calculate cost in USD."""
        model = model or self.default_model
        model_info = self.get_model_info(model)
        if model_info:
            return (prompt_tokens / 1000 * model_info.input_cost_per_1k) + \
                   (completion_tokens / 1000 * model_info.output_cost_per_1k)
        return 0.0