import structlog
import time
import json
from typing import List, Optional, AsyncIterator, Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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


class GeminiProvider(BaseProvider):
    """Google Gemini provider."""

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
            api_key=api_key or settings.gemini_api_key,
            base_url=None,
            default_model=default_model or settings.gemini_model,
            timeout=timeout or settings.gemini_timeout,
            max_retries=max_retries or settings.gemini_max_retries,
            **kwargs,
        )
        self.embedding_model = embedding_model or settings.gemini_embedding_model
        self._client = None
        self._models_cache: List[ModelInfo] = []
        self._capabilities = ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            supports_embeddings=True,
            supports_audio=True,
            supports_image_gen=False,
            max_context_window=1000000,
            max_output_tokens=8192,
        )

    @property
    def provider_name(self) -> ProviderName:
        return ProviderName.GEMINI

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def models(self) -> List[ModelInfo]:
        return self._models_cache

    async def initialize(self) -> None:
        """Initialize Gemini client."""
        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(self.default_model)
        await self._fetch_models()
        logger.info("gemini_provider_initialized", models=len(self._models_cache))

    async def close(self) -> None:
        """Close connections."""
        logger.info("gemini_provider_closed")

    async def _fetch_models(self) -> None:
        """Define available Gemini models."""
        self._models_cache = [
            ModelInfo(
                id="gemini-1.5-pro",
                provider=ProviderName.GEMINI,
                name="Gemini 1.5 Pro",
                description="High-performance model with 1M context",
                context_window=1000000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
                supports_embeddings=False,
                input_cost_per_1k=1.25,
                output_cost_per_1k=5.00,
                tags=["chat", "reasoning", "tools", "vision", "long_context"],
            ),
            ModelInfo(
                id="gemini-1.5-flash",
                provider=ProviderName.GEMINI,
                name="Gemini 1.5 Flash",
                description="Fast and efficient",
                context_window=1000000,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
                supports_embeddings=False,
                input_cost_per_1k=0.075,
                output_cost_per_1k=0.30,
                tags=["chat", "fast", "tools", "vision", "long_context"],
            ),
            ModelInfo(
                id="gemini-1.0-pro",
                provider=ProviderName.GEMINI,
                name="Gemini 1.0 Pro",
                description="Balanced performance",
                context_window=32768,
                max_output_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                supports_vision=True,
                supports_embeddings=False,
                input_cost_per_1k=0.50,
                output_cost_per_1k=1.50,
                tags=["chat", "tools", "vision"],
            ),
            ModelInfo(
                id="text-embedding-004",
                provider=ProviderName.GEMINI,
                name="Text Embedding 004",
                description="Latest embedding model",
                context_window=8192,
                max_output_tokens=0,
                supports_streaming=False,
                supports_tools=False,
                supports_vision=False,
                supports_embeddings=True,
                tags=["embedding"],
            ),
        ]

    async def health_check(self) -> bool:
        """Check Gemini health."""
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            await model.generate_content_async("hi")
            return True
        except Exception:
            return False

    async def list_models(self) -> List[ModelInfo]:
        """List available models."""
        return self._models_cache

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send chat completion request."""
        model = request.model or self.default_model
        gm = genai.GenerativeModel(model)

        messages = self._convert_messages(request.messages)
        tools = self._convert_tools(request.tools) if request.tools else None

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        response = await gm.generate_content_async(
            messages,
            generation_config=genai.GenerationConfig(
                temperature=request.temperature,
                top_p=request.top_p,
                max_output_tokens=request.max_tokens,
                stop_sequences=request.stop,
            ),
            tools=tools,
            safety_settings=safety_settings,
        )

        return self._build_chat_response(response, model)

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream chat completion."""
        model = request.model or self.default_model
        gm = genai.GenerativeModel(model)

        messages = self._convert_messages(request.messages)
        tools = self._convert_tools(request.tools) if request.tools else None

        stream = await gm.generate_content_async(
            messages,
            generation_config=genai.GenerationConfig(
                temperature=request.temperature,
                top_p=request.top_p,
                max_output_tokens=request.max_tokens,
                stop_sequences=request.stop,
            ),
            tools=tools,
            stream=True,
        )

        chunk_id = f"chatcmpl-{id(stream)}"
        created = int(time.time())

        async for chunk in stream:
            yield self._build_stream_chunk(chunk, chunk_id, created, model)

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert internal messages to Gemini format."""
        converted = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # System messages become first user message with special prefix
                converted.append({
                    "role": "user",
                    "parts": [f"[SYSTEM INSTRUCTION]: {msg.content}"],
                })
                converted.append({"role": "model", "parts": ["Understood."]})
            elif msg.role == MessageRole.USER:
                parts = []
                if msg.content:
                    parts.append(msg.content)
                if msg.images:
                    for img in msg.images:
                        parts.append({"mime_type": "image/jpeg", "data": img})
                converted.append({"role": "user", "parts": parts})
            elif msg.role == MessageRole.ASSISTANT:
                parts = []
                if msg.content:
                    parts.append(msg.content)
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append({
                            "function_call": {
                                "name": tc.function.name,
                                "args": json.loads(tc.function.arguments),
                            }
                        })
                converted.append({"role": "model", "parts": parts})
            elif msg.role == MessageRole.TOOL:
                if msg.tool_call_id:
                    converted.append({
                        "role": "user",
                        "parts": [{
                            "function_response": {
                                "name": msg.name or "unknown",
                                "response": {"result": msg.content},
                            }
                        }]
                    })
        return converted

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert tools to Gemini format."""
        return [
            {
                "function_declarations": [
                    {
                        "name": t.function.name,
                        "description": t.function.description,
                        "parameters": t.function.parameters,
                    }
                ]
            }
            for t in tools
        ]

    def _build_chat_response(self, response, model: str) -> ChatResponse:
        """Build ChatResponse from Gemini response."""
        text_parts = []
        tool_calls = []

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    tool_calls.append(ToolCall(
                        id=f"call_{len(tool_calls)}",
                        function=FunctionCall(
                            name=part.function_call.name,
                            arguments=json.dumps(dict(part.function_call.args)),
                        )
                    ))

        finish_reason = "stop"
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason.name.lower()
            if finish_reason == "function_call":
                finish_reason = "tool_calls"

        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=model,
            provider="gemini",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content="\n".join(text_parts) if text_parts else None,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                completion_tokens=response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
                total_tokens=response.usage_metadata.total_token_count if response.usage_metadata else 0,
            ),
        )

    def _build_stream_chunk(self, chunk, chunk_id: str, created: int, model: str) -> ChatStreamChunk:
        """Build stream chunk from Gemini streaming response."""
        text = ""
        tool_calls = None

        for candidate in chunk.candidates:
            for part in candidate.content.parts:
                if part.text:
                    text += part.text
                if part.function_call:
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "index": len(tool_calls),
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(dict(part.function_call.args)),
                        }
                    })

        finish_reason = None
        if chunk.candidates and chunk.candidates[0].finish_reason:
            finish_reason = chunk.candidates[0].finish_reason.name.lower()
            if finish_reason == "function_call":
                finish_reason = "tool_calls"

        return ChatStreamChunk(
            id=chunk_id,
            created=created,
            model=model,
            provider="gemini",
            choices=[
                ChatStreamChoice(
                    index=0,
                    delta=ChatDelta(
                        role=MessageRole.ASSISTANT,
                        content=text if text else None,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )

    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using Gemini."""
        model = request.model or self.embedding_model
        texts = request.input if isinstance(request.input, list) else [request.input]

        embeddings = []
        for text in texts:
            result = genai.embed_content(model=model, content=text)
            embeddings.append(result["embedding"])

        return EmbeddingResponse(
            data=[
                EmbeddingData(index=i, embedding=emb)
                for i, emb in enumerate(embeddings)
            ],
            model=model,
            usage=UsageInfo(
                prompt_tokens=sum(len(t) // 4 for t in texts),
                completion_tokens=0,
                total_tokens=sum(len(t) // 4 for t in texts),
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