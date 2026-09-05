import structlog
import time
from typing import Optional, List, AsyncIterator
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from libs.schemas_common.chat import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    ChatMessage,
    MessageRole,
    TaskType,
)
from libs.schemas_common.providers import ProviderName
from ..router.router import ModelRouter, RoutingStrategy
from ..dependencies import get_router
from ..caching.cache import get_cache
from ..guardrails.guardrails import apply_input_guardrails, apply_output_guardrails
from ..config.settings import settings

logger = structlog.get_logger()

router = APIRouter()


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    messages: List[ChatMessage]
    model: Optional[str] = None
    provider: Optional[str] = None
    task_type: Optional[str] = "general"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    response_format: Optional[dict] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[str] = None
    user: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Chat completion endpoint with intelligent routing."""
    start_time = time.time()
    request_id = getattr(http_request.state, "request_id", "unknown")

    # Convert to internal request
    internal_request = ChatRequest(
        messages=request.messages,
        model=request.model,
        provider=request.provider,
        task_type=TaskType(request.task_type) if request.task_type else TaskType.GENERAL,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream,
        stop=request.stop,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        response_format=request.response_format,
        tools=request.tools,
        tool_choice=request.tool_choice,
        user=request.user,
    )

    # Input guardrails
    if settings.enable_input_guardrails:
        last_user_msg = next((m.content for m in reversed(request.messages) if m.role == MessageRole.USER), "")
        if not await apply_input_guardrails(last_user_msg):
            raise HTTPException(status_code=400, detail="Input failed guardrails")

    # Check cache for non-streaming requests
    cache = get_cache()
    cache_key = None
    if cache and not request.stream:
        cache_key = f"chat:{request.model or 'auto'}:{hash(str(internal_request.messages))}:{request.temperature}"
        cached = await cache.get(cache_key)
        if cached:
            logger.info("cache_hit", request_id=request_id)
            return cached

    try:
        # Route to provider
        provider_name = ProviderName(request.provider) if request.provider else None
        provider = router.route(
            task_type=internal_request.task_type,
            provider=provider_name,
            model=request.model,
        )

        if request.stream:
            return StreamingResponse(
                _stream_chat(provider, internal_request, router, request_id, start_time),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

        # Non-streaming response
        response = await provider.chat(internal_request)

        # Output guardrails
        if settings.enable_output_guardrails:
            content = response.choices[0].message.content if response.choices else ""
            if not await apply_output_guardrails(content):
                raise HTTPException(status_code=500, detail="Output failed guardrails")

        # Cache response
        if cache and cache_key:
            await cache.set(cache_key, response)

        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        provider_name_used = getattr(provider, 'provider_name', 'unknown')
        await router.record_request(
            provider_name_used,
            latency_ms,
            True,
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )

        logger.info("chat_completion_success", request_id=request_id, latency_ms=latency_ms)
        return response

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("chat_completion_failed", request_id=request_id, error=str(e), latency_ms=latency_ms)
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_chat(
    provider,
    request: ChatRequest,
    router: ModelRouter,
    request_id: str,
    start_time: float,
) -> AsyncIterator[str]:
    """Stream chat completion as SSE."""
    try:
        provider_name = getattr(provider, 'provider_name', 'unknown')
        chunk_count = 0

        async for chunk in provider.chat_stream(request):
            chunk_count += 1
            # Convert to SSE format
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Send done signal
        yield "data: [DONE]\n\n"

        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        await router.record_request(provider_name, latency_ms, True)

        logger.info("chat_stream_success", request_id=request_id, chunks=chunk_count, latency_ms=latency_ms)

    except Exception as e:
        logger.error("chat_stream_failed", request_id=request_id, error=str(e))
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions_alias(
    request: ChatCompletionRequest,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """OpenAI-compatible alias for /chat."""
    return await chat_completion(request, http_request, router)