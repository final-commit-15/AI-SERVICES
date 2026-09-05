import structlog
from typing import List, Optional, AsyncIterator, Dict, Any
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageToolCall
from openai.types.responses import Response as OpenAIResponse

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
from libs.schemas_common.responses import (
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
    ResponsesOutputItem,
    ResponsesContent,
    ResponsesUsage,
)
from ..config.settings import settings

logger = structlog.get_logger()


class OpenAIProvider(BaseProvider):
    """OpenAI provider for GPT models."""

    def __init__(
        self,
        api_key: str = None,
        organization: str = None,
        default_model: str = None,
        embedding_model: str = None,
        timeout: int = None,
        max_retries: int = None,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key or settings.openai_api_key,
            base_url=None,
            default_model=default_model or settings.openai_model,
            timeout=timeout or settings.openai_timeout,
            max_retries=max_retries or settings.openai_max_retries,
            **kwargs,
        )
        self.organization = organization or settings.openai_org_id
        self.embedding_model = embedding_model or settings.openai_embedding_model
        self._client: Optional[AsyncOpenAI] = None
        self._models_cache: List[ModelInfo] = []
        self._capabilities = ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            supports_embeddings=True,
            supports_audio=True,
            supports_image_gen=True,
            supports_responses_api=True,
            max_context_window=128000,
            max_output_tokens=4096,
        )

    @property
    def provider_name(self) -> ProviderName:
        return ProviderName.OPENAI

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @property
    def models(self) -> List[ModelInfo]:
        return self._models_cache

    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            organization=self.organization,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        await self._fetch_models()
        logger.info("openai_provider_initialized", models=len(self._models_cache))

    async def close(self) -> None:
        """Close connections."""
        if self._client:
            await self._client.close()
        logger.info("openai_provider_closed")

    async def _fetch_models(self) -> None:
        """Fetch available models."""
        try:
            models_resp = await self._client.models.list()
            self._models_cache = []
            for model in models_resp.data:
                if any(prefix in model.id for prefix in ["gpt-", "o1-", "text-embedding", "dall-e", "whisper", "tts"]):
                    capabilities = self._infer_capabilities(model.id)
                    self._models_cache.append(ModelInfo(
                        id=model.id,
                        provider=ProviderName.OPENAI,
                        name=model.id,
                        description=f"OpenAI model: {model.id}",
                        context_window=self._get_context_window(model.id),
                        max_output_tokens=self._get_max_output(model.id),
                        supports_streaming=True,
                        supports_tools=capabilities["tools"],
                        supports_vision=capabilities["vision"],
                        supports_embeddings=capabilities["embeddings"],
                        input_cost_per_1k=capabilities.get("input_cost", 0),
                        output_cost_per_1k=capabilities.get("output_cost", 0),
                        tags=capabilities.get("tags", []),
                    ))
        except Exception as e:
            logger.warning("failed_to_fetch_openai_models", error=str(e))
            self._models_cache = [
                ModelInfo(
                    id=self.default_model,
                    provider=ProviderName.OPENAI,
                    name=self.default_model,
                    context_window=128000,
                    max_output_tokens=4096,
                    supports_streaming=True,
                    supports_tools=True,
                    supports_vision="gpt-4o" in self.default_model,
                    supports_embeddings=False,
                )
            ]

    def _infer_capabilities(self, model_id: str) -> Dict[str, Any]:
        """Infer capabilities from model ID."""
        model_lower = model_id.lower()
        if "embedding" in model_lower:
            return {"embeddings": True, "tools": False, "vision": False, "tags": ["embedding"]}
        if "dall-e" in model_lower:
            return {"embeddings": False, "tools": False, "vision": False, "tags": ["image_gen"]}
        if "whisper" in model_lower:
            return {"embeddings": False, "tools": False, "vision": False, "tags": ["audio", "stt"]}
        if "tts" in model_lower:
            return {"embeddings": False, "tools": False, "vision": False, "tags": ["audio", "tts"]}
        if "gpt-4o" in model_lower:
            return {"tools": True, "vision": True, "embeddings": False, "input_cost": 2.50, "output_cost": 10.00, "tags": ["chat", "vision", "tools"]}
        if "gpt-4" in model_lower:
            return {"tools": True, "vision": False, "embeddings": False, "input_cost": 30.00, "output_cost": 60.00, "tags": ["chat", "tools"]}
        if "gpt-3.5" in model_lower:
            return {"tools": True, "vision": False, "embeddings": False, "input_cost": 0.50, "output_cost": 1.50, "tags": ["chat", "tools"]}
        if "o1" in model_lower:
            return {"tools": True, "vision": False, "embeddings": False, "input_cost": 15.00, "output_cost": 60.00, "tags": ["reasoning", "tools"]}
        return {"tools": True, "vision": False, "embeddings": False, "tags": ["chat"]}

    def _get_context_window(self, model_id: str) -> int:
        """Get context window for model."""
        if "gpt-4o" in model_id or "o1" in model_id:
            return 128000
        if "gpt-4" in model_id and "32k" in model_id:
            return 32768
        if "gpt-4" in model_id:
            return 8192
        if "gpt-3.5" in model_id and "16k" in model_id:
            return 16384
        return 4096

    def _get_max_output(self, model_id: str) -> int:
        """Get max output tokens for model."""
        if "o1" in model_id:
            return 32768
        if "gpt-4o" in model_id:
            return 16384
        return 4096

    async def health_check(self) -> bool:
        """Check OpenAI health."""
        try:
            await self._client.models.list()
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
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            response_format=request.response_format,
            tools=self._convert_tools(request.tools) if request.tools else None,
            tool_choice=request.tool_choice,
            user=request.user,
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
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            response_format=request.response_format,
            tools=self._convert_tools(request.tools) if request.tools else None,
            tool_choice=request.tool_choice,
            user=request.user,
        )

        async for chunk in stream:
            yield self._build_stream_chunk(chunk, model)

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        converted = []
        for msg in messages:
            openai_msg = {
                "role": msg.role.value,
            }
            if msg.content is not None:
                openai_msg["content"] = msg.content
            if msg.name:
                openai_msg["name"] = msg.name
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                openai_msg["tool_calls"] = [
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
                content = msg.content or ""
                for img in msg.images:
                    content += f"\n![image]({img})"
                openai_msg["content"] = content
            converted.append(openai_msg)
        return converted

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI format."""
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

    def _build_chat_response(self, response: ChatCompletion, model: str) -> ChatResponse:
        """Build ChatResponse from OpenAI response."""
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
            provider="openai",
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
                prompt_tokens_details=response.usage.prompt_tokens_details.model_dump() if response.usage and response.usage.prompt_tokens_details else None,
                completion_tokens_details=response.usage.completion_tokens_details.model_dump() if response.usage and response.usage.completion_tokens_details else None,
            ),
            system_fingerprint=response.system_fingerprint,
        )

    def _build_stream_chunk(self, chunk: ChatCompletionChunk, model: str) -> ChatStreamChunk:
        """Build stream chunk from OpenAI streaming response."""
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
            provider="openai",
            choices=[
                ChatStreamChoice(
                    index=choice.index,
                    delta=ChatDelta(
                        role=MessageRole(delta.role) if delta.role else None,
                        content=delta.content,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=choice.finish_reason,
                    logprobs=choice.logprobs.model_dump() if choice.logprobs else None,
                )
            ],
        )

    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using OpenAI."""
        model = request.model or self.embedding_model
        texts = request.input if isinstance(request.input, list) else [request.input]

        response = await self._client.embeddings.create(
            model=model,
            input=texts,
            encoding_format=request.encoding_format,
            dimensions=request.dimensions,
            user=request.user,
        )

        return EmbeddingResponse(
            data=[
                EmbeddingData(index=d.index, embedding=d.embedding)
                for d in response.data
            ],
            model=response.model,
            usage=UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=0,
                total_tokens=response.usage.total_tokens,
            ),
        )

    async def responses(self, request: ResponsesRequest) -> ResponsesResponse:
        """OpenAI Responses API."""
        response = await self._client.responses.create(
            model=request.model,
            input=request.input,
            instructions=request.instructions,
            tools=self._convert_tools(request.tools) if request.tools else None,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            truncation=request.truncation,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            store=request.store,
            metadata=request.metadata,
            user=request.user,
        )

        return self._build_responses_response(response)

    async def responses_stream(self, request: ResponsesRequest) -> AsyncIterator[ResponsesStreamEvent]:
        """Stream Responses API."""
        stream = await self._client.responses.create(
            model=request.model,
            input=request.input,
            instructions=request.instructions,
            tools=self._convert_tools(request.tools) if request.tools else None,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            truncation=request.truncation,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            store=request.store,
            metadata=request.metadata,
            user=request.user,
            stream=True,
        )

        async for event in stream:
            yield self._build_stream_event(event)

    def _build_responses_response(self, response: OpenAIResponse) -> ResponsesResponse:
        """Build ResponsesResponse from OpenAI response."""
        return ResponsesResponse(
            id=response.id,
            created_at=response.created_at,
            status=response.status,
            model=response.model,
            output=[
                ResponsesOutputItem(
                    id=item.id,
                    type=item.type,
                    role=item.role,
                    content=[
                        ResponsesContent(type=c.type, text=c.text)
                        for c in item.content
                    ] if item.content else None,
                    name=item.name,
                    arguments=item.arguments,
                    call_id=item.call_id,
                    output=item.output,
                    status=item.status,
                )
                for item in response.output
            ],
            usage=ResponsesUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    def _build_stream_event(self, event) -> ResponsesStreamEvent:
        """Build stream event from OpenAI streaming response."""
        return ResponsesStreamEvent(
            type=event.type,
            sequence_number=event.sequence_number,
            item=event.item,
            delta=event.delta,
            snapshot=event.snapshot,
        )

    async def transcribe_audio(self, file: bytes, model: str = None, **kwargs) -> str:
        """Transcribe audio using Whisper."""
        model = model or settings.speech_stt_model
        from io import BytesIO
        audio_file = BytesIO(file)
        audio_file.name = "audio.wav"

        response = await self._client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            **kwargs,
        )
        return response.text

    async def synthesize_speech(self, text: str, model: str = None, voice: str = None, **kwargs) -> bytes:
        """Synthesize speech using TTS."""
        model = model or settings.speech_tts_model
        voice = voice or settings.speech_tts_voice

        response = await self._client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            **kwargs,
        )
        return response.content

    async def generate_image(self, prompt: str, model: str = None, **kwargs) -> bytes:
        """Generate image using DALL-E."""
        model = model or settings.image_gen_model

        response = await self._client.images.generate(
            model=model,
            prompt=prompt,
            size=kwargs.get("size", settings.image_gen_size),
            quality=kwargs.get("quality", settings.image_gen_quality),
            style=kwargs.get("style", settings.image_gen_style),
            n=kwargs.get("n", 1),
        )

        # Download the image
        import httpx
        async with httpx.AsyncClient() as client:
            img_resp = await client.get(response.data[0].url)
            return img_resp.content

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str = None) -> float:
        """Calculate cost in USD."""
        model = model or self.default_model
        model_info = self.get_model_info(model)
        if model_info:
            return (prompt_tokens / 1000 * model_info.input_cost_per_1k) + \
                   (completion_tokens / 1000 * model_info.output_cost_per_1k)
        return 0.0