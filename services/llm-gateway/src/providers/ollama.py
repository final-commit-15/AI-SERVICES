import json
import structlog
from typing import List, Optional, AsyncIterator, Dict, Any
import httpx
from ollama import AsyncClient

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
)
from libs.schemas_common.embeddings import EmbeddingRequest, EmbeddingResponse
from libs.schemas_common.providers import ProviderName, ModelInfo
from libs.schemas_common.health import HealthStatus
from ..config.settings import settings

logger = structlog.get_logger()


class OllamaProvider(BaseProvider):
    """Ollama provider for local LLM inference."""

    def __init__(
        self,
        host: str = None,
        default_model: str = None,
        embedding_model: str = None,
        vision_model: str = None,
        timeout: int = None,
        **kwargs,
    ):
        super().__init__(
            base_url=host or settings.ollama_host,
            default_model=default_model or settings.ollama_model,
            timeout=timeout or settings.ollama_timeout,
            **kwargs,
        )
        self.host = host or settings.ollama_host
        self.embedding_model = embedding_model or settings.ollama_embedding_model
        self.vision_model = vision_model or settings.ollama_vision_model
        self._client: Optional[AsyncClient] = None
        self._models_cache: List[ModelInfo] = []
        self._capabilities = ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            supports_embeddings=True,
            supports_audio=False,
            supports_image_gen=False,
            max_context_window=32768,
            max_output_tokens=8192,
        )

    @property
    def provider_name(self) -> ProviderName:
        return ProviderName.OLLAMA

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def models(self) -> List[ModelInfo]:
        return self._models_cache

    async def initialize(self) -> None:
        """Initialize Ollama client and fetch available models."""
        self._client = AsyncClient(
            host=self.host,
            timeout=self.timeout,
        )
        await self._fetch_models()
        logger.info("ollama_provider_initialized", host=self.host, models=len(self._models_cache))

    async def close(self) -> None:
        """Close connections."""
        if self._client:
            await self._client.close()
        logger.info("ollama_provider_closed")

    async def _fetch_models(self) -> None:
        """Fetch and cache available models."""
        try:
            response = await self._client.list()
            self._models_cache = []
            for model in response.get("models", []):
                model_name = model.get("name", "")
                capabilities = self._infer_capabilities(model_name)
                self._models_cache.append(ModelInfo(
                    id=model_name,
                    provider=ProviderName.OLLAMA,
                    name=model_name,
                    description=f"Ollama model: {model_name}",
                    context_window=32768,
                    max_output_tokens=8192,
                    supports_streaming=True,
                    supports_tools=capabilities.get("tools", False),
                    supports_vision=capabilities.get("vision", False),
                    supports_embeddings=capabilities.get("embeddings", False),
                    tags=capabilities.get("tags", []),
                ))
        except Exception as e:
            logger.warning("failed_to_fetch_ollama_models", error=str(e))
            # Add default models
            self._models_cache = [
                ModelInfo(
                    id=self.default_model,
                    provider=ProviderName.OLLAMA,
                    name=self.default_model,
                    context_window=32768,
                    max_output_tokens=8192,
                    supports_streaming=True,
                    supports_tools=True,
                    supports_vision="llava" in self.default_model.lower(),
                    supports_embeddings="embed" in self.default_model.lower(),
                )
            ]

    def _infer_capabilities(self, model_name: str) -> Dict[str, Any]:
        """Infer model capabilities from name."""
        name_lower = model_name.lower()
        return {
            "tools": any(x in name_lower for x in ["hermes", "function", "tool", "qwen2.5", "llama3.1", "nemotron"]),
            "vision": any(x in name_lower for x in ["llava", "bakllava", "vision", "moondream"]),
            "embeddings": any(x in name_lower for x in ["embed", "nomic", "bge", "e5"]),
            "tags": [],
        }

    async def health_check(self) -> bool:
        """Check Ollama health."""
        try:
            await self._client.list()
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

        ollama_options = {
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.max_tokens:
            ollama_options["num_predict"] = request.max_tokens
        if request.stop:
            ollama_options["stop"] = request.stop

        response = await self._client.chat(
            model=model,
            messages=messages,
            stream=False,
            options=ollama_options,
            format=request.response_format,
            tools=self._convert_tools(request.tools) if request.tools else None,
            keep_alive=settings.ollama_keep_alive,
        )

        return self._build_chat_response(response, model, request)

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream chat completion."""
        model = request.model or self.default_model
        messages = self._convert_messages(request.messages)

        ollama_options = {
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.max_tokens:
            ollama_options["num_predict"] = request.max_tokens
        if request.stop:
            ollama_options["stop"] = request.stop

        stream = await self._client.chat(
            model=model,
            messages=messages,
            stream=True,
            options=ollama_options,
            format=request.response_format,
            tools=self._convert_tools(request.tools) if request.tools else None,
            keep_alive=settings.ollama_keep_alive,
        )

        chunk_id = f"chatcmpl-{id(stream)}"
        created = int(time.time())

        async for chunk in stream:
            yield self._build_stream_chunk(chunk, chunk_id, created, model)

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert internal messages to Ollama format."""
        converted = []
        for msg in messages:
            ollama_msg = {
                "role": msg.role.value,
                "content": msg.content or "",
            }
            if msg.images:
                ollama_msg["images"] = msg.images
            if msg.tool_calls:
                ollama_msg["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                ollama_msg["tool_call_id"] = msg.tool_call_id
            converted.append(ollama_msg)
        return converted

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert tools to Ollama format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters,
                }
            }
            for t in tools
        ]

    def _build_chat_response(self, response: Dict[str, Any], model: str, request: ChatRequest) -> ChatResponse:
        """Build ChatResponse from Ollama response."""
        message = response.get("message", {})
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=f"call_{i}",
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=json.dumps(tc["function"].get("arguments", {})),
                    )
                )
                for i, tc in enumerate(message["tool_calls"])
            ]

        return ChatResponse(
            id=f"chatcmpl-{id(response)}",
            created=int(time.time()),
            model=model,
            provider="ollama",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=message.get("content", ""),
                        tool_calls=tool_calls,
                    ),
                    finish_reason="tool_calls" if tool_calls else "stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
                total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            ),
        )

    def _build_stream_chunk(self, chunk: Dict[str, Any], chunk_id: str, created: int, model: str) -> ChatStreamChunk:
        """Build stream chunk from Ollama streaming response."""
        message = chunk.get("message", {})
        delta = ChatDelta(
            role=MessageRole.ASSISTANT if not message.get("role") else MessageRole(message["role"]),
            content=message.get("content", ""),
        )

        if message.get("tool_calls"):
            delta.tool_calls = [
                {
                    "index": i,
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": json.dumps(tc["function"].get("arguments", {})),
                    }
                }
                for i, tc in enumerate(message["tool_calls"])
            ]

        finish_reason = None
        if chunk.get("done"):
            finish_reason = "tool_calls" if message.get("tool_calls") else "stop"

        return ChatStreamChunk(
            id=chunk_id,
            created=created,
            model=model,
            provider="ollama",
            choices=[
                ChatStreamChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                )
            ],
        )

    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using Ollama."""
        model = request.model or self.embedding_model
        texts = request.input if isinstance(request.input, list) else [request.input]

        embeddings = []
        for text in texts:
            resp = await self._client.embeddings(model=model, prompt=text)
            embeddings.append(resp["embedding"])

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

    async def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.host}/api/ps")
                return resp.json()
        except Exception as e:
            logger.warning("failed_to_get_gpu_info", error=str(e))
            return {}

    async def pull_model(self, model_name: str) -> AsyncIterator[Dict[str, Any]]:
        """Pull a model from Ollama registry."""
        async for progress in self._client.pull(model=model_name, stream=True):
            yield progress

    async def delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama."""
        try:
            await self._client.delete(model=model_name)
            await self._fetch_models()
            return True
        except Exception as e:
            logger.error("failed_to_delete_model", model=model_name, error=str(e))
            return False


import time
from libs.schemas_common.embeddings import EmbeddingData