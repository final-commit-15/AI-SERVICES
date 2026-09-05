import structlog
from typing import List, Optional, AsyncIterator, Dict, Any
import anthropic
from anthropic import AsyncAnthropic

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


class AnthropicProvider(BaseProvider):
    """Anthropic provider for Claude models."""

    def __init__(
        self,
        api_key: str = None,
        default_model: str = None,
        timeout: int = None,
        max_retries: int = None,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key or settings.anthropic_api_key,
            base_url=None,
            default_model=default_model or settings.anthropic_model,
            timeout=timeout or settings.anthropic_timeout,
            max_retries=max_retries or settings.anthropic_max_retries,
            **kwargs,
        )
        self._client: Optional[AsyncAnthropic] = None
        self._models_cache: List[ModelInfo] = []
        self._capabilities = ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            supports_embeddings=False,
            supports_audio=False,
            supports_image_gen=False,
            max_context_window=200000,
            max_output_tokens=8192,
        )

    @property
    def provider_name(self) -> ProviderName:
        return ProviderName.ANTHROPIC

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def models(self) -> List[ModelInfo]:
        return self._models_cache

    async def initialize(self) -> None:
        """Initialize Anthropic client."""
        self._client = AsyncAnthropic(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        await self._fetch_models()
        logger.info("anthropic_provider_initialized", models=len(self._models_cache))

    async def close(self) -> None:
        """Close connections."""
        logger.info("anthropic_provider_closed")

    async def _fetch_models(self) -> None:
        """Define available Claude models."""
        self._models_cache = [
            ModelInfo(
                id="claude-3-5-sonnet-20241022",
                provider=ProviderName.ANTHROPIC,
                name="Claude 3.5 Sonnet",
                description="Most intelligent model",
                context_window=200000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
                input_cost_per_1k=3.00,
                output_cost_per_1k=15.00,
                tags=["chat", "reasoning", "tools", "vision"],
            ),
            ModelInfo(
                id="claude-3-5-haiku-20241022",
                provider=ProviderName.ANTHROPIC,
                name="Claude 3.5 Haiku",
                description="Fast and efficient",
                context_window=200000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
                input_cost_per_1k=0.80,
                output_cost_per_1k=4.00,
                tags=["chat", "fast", "tools", "vision"],
            ),
            ModelInfo(
                id="claude-3-opus-20240229",
                provider=ProviderName.ANTHROPIC,
                name="Claude 3 Opus",
                description="Previous flagship model",
                context_window=200000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
                input_cost_per_1k=15.00,
                output_cost_per_1k=75.00,
                tags=["chat", "reasoning", "tools", "vision"],
            ),
        ]

    async def health_check(self) -> bool:
        """Check Anthropic health."""
        try:
            await self._client.messages.create(
                model=self.default_model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
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
        messages, system = self._convert_messages(request.messages)

        response = await self._client.messages.create(
            model=model,
            messages=messages,
            system=system,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 4096,
            stop_sequences=request.stop,
            tools=self._convert_tools(request.tools) if request.tools else None,
            tool_choice=self._convert_tool_choice(request.tool_choice) if request.tool_choice else None,
        )

        return self._build_chat_response(response, model)

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream chat completion."""
        model = request.model or self.default_model
        messages, system = self._convert_messages(request.messages)

        stream = await self._client.messages.create(
            model=model,
            messages=messages,
            system=system,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 4096,
            stop_sequences=request.stop,
            tools=self._convert_tools(request.tools) if request.tools else None,
            tool_choice=self._convert_tool_choice(request.tool_choice) if request.tool_choice else None,
            stream=True,
        )

        chunk_id = f"chatcmpl-{id(stream)}"
        created = int(time.time())

        async for event in stream:
            chunk = self._build_stream_chunk(event, chunk_id, created, model)
            if chunk:
                yield chunk

    def _convert_messages(self, messages: List[ChatMessage]) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """Convert internal messages to Anthropic format."""
        converted = []
        system = None

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system = msg.content
                continue

            anthropic_msg = {
                "role": "user" if msg.role == MessageRole.USER else "assistant",
            }

            content = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            if msg.images:
                for img in msg.images:
                    content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments),
                    })
            if msg.tool_call_id:
                content.append({
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content or "",
                })

            if content:
                anthropic_msg["content"] = content if len(content) > 1 else content[0].get("text", "")

            converted.append(anthropic_msg)

        return converted, system

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic format."""
        return [
            {
                "name": t.function.name,
                "description": t.function.description,
                "input_schema": t.function.parameters,
            }
            for t in tools
        ]

    def _convert_tool_choice(self, tool_choice: Any) -> Dict[str, Any]:
        """Convert tool choice to Anthropic format."""
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return {"type": "auto"}
            elif tool_choice == "none":
                return {"type": "none"}
            else:
                return {"type": "tool", "name": tool_choice}
        elif isinstance(tool_choice, dict):
            return tool_choice
        return {"type": "auto"}

    def _build_chat_response(self, response, model: str) -> ChatResponse:
        """Build ChatResponse from Anthropic response."""
        content = response.content
        text_parts = []
        tool_calls = []

        for block in content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    function=FunctionCall(
                        name=block.name,
                        arguments=json.dumps(block.input),
                    )
                ))

        return ChatResponse(
            id=response.id,
            created=int(time.time()),
            model=model,
            provider="anthropic",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="\n".join(text_parts) if text_parts else None,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason="tool_use" if tool_calls else response.stop_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
        )

    def _build_stream_chunk(self, event, chunk_id: str, created: int, model: str) -> Optional[ChatStreamChunk]:
        """Build stream chunk from Anthropic streaming event."""
        if event.type == "message_start":
            return None
        elif event.type == "content_block_start":
            return None
        elif event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                return ChatStreamChunk(
                    id=chunk_id,
                    created=created,
                    model=model,
                    provider="anthropic",
                    choices=[
                        ChatStreamChoice(
                            index=0,
                            delta=ChatDelta(
                                role=MessageRole.ASSISTANT,
                                content=delta.text,
                            ),
                        )
                    ],
                )
            elif delta.type == "input_json_delta":
                # Tool call streaming - would need to accumulate
                return None
        elif event.type == "content_block_stop":
            return None
        elif event.type == "message_delta":
            finish_reason = None
            if event.delta.stop_reason:
                finish_reason = "tool_calls" if event.delta.stop_reason == "tool_use" else event.delta.stop_reason
            return ChatStreamChunk(
                id=chunk_id,
                created=created,
                model=model,
                provider="anthropic",
                choices=[
                    ChatStreamChoice(
                        index=0,
                        delta=ChatDelta(),
                        finish_reason=finish_reason,
                    )
                ],
            )
        elif event.type == "message_stop":
            return ChatStreamChunk(
                id=chunk_id,
                created=created,
                model=model,
                provider="anthropic",
                choices=[
                    ChatStreamChoice(
                        index=0,
                        delta=ChatDelta(),
                        finish_reason="stop",
                    )
                ],
            )
        return None

    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Anthropic doesn't support embeddings natively."""
        raise NotImplementedError("Anthropic does not support embeddings")

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str = None) -> float:
        """Calculate cost in USD."""
        model = model or self.default_model
        model_info = self.get_model_info(model)
        if model_info:
            return (prompt_tokens / 1000 * model_info.input_cost_per_1k) + \
                   (completion_tokens / 1000 * model_info.output_cost_per_1k)
        return 0.0


import time
import json