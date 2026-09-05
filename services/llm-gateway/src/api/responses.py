import structlog
import time
from typing import Optional, List, AsyncIterator
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse

from libs.schemas_common.responses import (
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
    ResponsesInputItem,
    ResponsesContent,
)
from libs.schemas_common.providers import ProviderName
from ..router.router import ModelRouter
from ..dependencies import get_router
from ..caching.cache import get_cache
from ..config.settings import settings

logger = structlog.get_logger()

router = APIRouter()


class ResponsesAPIRequest(BaseModel):
    """OpenAI Responses API request."""
    model: str
    input: List[ResponsesInputItem] | str
    instructions: Optional[str] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[str | dict] = None
    parallel_tool_calls: bool = True
    truncation: str = "auto"
    max_output_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    store: bool = True
    metadata: Optional[dict] = None
    user: Optional[str] = None
    stream: bool = False


@router.post("/responses", response_model=ResponsesResponse)
async def create_response(
    request: ResponsesAPIRequest,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """OpenAI-compatible Responses API endpoint."""
    start_time = time.time()
    request_id = getattr(http_request.state, "request_id", "unknown")

    try:
        # Convert to internal request
        internal_request = ResponsesRequest(
            model=request.model,
            input=request.input if isinstance(request.input, list) else [{"type": "message", "role": "user", "content": request.input}],
            instructions=request.instructions,
            tools=request.tools,
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

        # Check cache for non-streaming
        cache = get_cache()
        cache_key = None
        if cache and not request.stream:
            cache_key = f"responses:{request.model}:{hash(str(internal_request.input))}:{request.temperature}"
            cached = await cache.get(cache_key)
            if cached:
                logger.info("cache_hit", request_id=request_id)
                return cached

        # Route to provider with Responses API support
        provider = router.route(
            task_type="general",
            model=request.model,
        )

        # Check if provider supports Responses API
        if not hasattr(provider, 'responses') or not provider.capabilities.supports_responses_api:
            # Fallback to chat completion
            return await _fallback_to_chat(provider, internal_request)

        if request.stream:
            return StreamingResponse(
                _stream_responses(provider, internal_request, router, request_id, start_time),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

        response = await provider.responses(internal_request)

        if cache and cache_key:
            await cache.set(cache_key, response)

        latency_ms = (time.time() - start_time) * 1000
        provider_name = getattr(provider, 'provider_name', 'unknown')
        await router.record_request(provider_name, latency_ms, True)

        logger.info("responses_success", request_id=request_id, latency_ms=latency_ms)
        return response

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("responses_failed", request_id=request_id, error=str(e), latency_ms=latency_ms)
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_responses(
    provider,
    request: ResponsesRequest,
    router: ModelRouter,
    request_id: str,
    start_time: float,
) -> AsyncIterator[str]:
    """Stream Responses API as SSE."""
    try:
        provider_name = getattr(provider, 'provider_name', 'unknown')

        async for event in provider.responses_stream(request):
            yield f"data: {event.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

        latency_ms = (time.time() - start_time) * 1000
        await router.record_request(provider_name, latency_ms, True)

    except Exception as e:
        logger.error("responses_stream_failed", request_id=request_id, error=str(e))
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        yield "data: [DONE]\n\n"


async def _fallback_to_chat(provider, request: ResponsesRequest) -> ResponsesResponse:
    """Fallback to chat completion when Responses API not supported."""
    # Convert Responses request to Chat request
    from libs.schemas_common.chat import ChatRequest, ChatMessage, MessageRole

    messages = []
    for item in request.input:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                content = "\n".join(text_parts)
            messages.append(ChatMessage(role=MessageRole(role), content=content))
        elif isinstance(item, str):
            messages.append(ChatMessage(role=MessageRole.USER, content=item))

    chat_request = ChatRequest(
        messages=messages,
        model=request.model,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_output_tokens,
    )

    response = await provider.chat(chat_request)

    # Convert ChatResponse to ResponsesResponse
    return ResponsesResponse(
        id=response.id,
        created_at=int(time.time()),
        status="completed",
        model=response.model,
        output=[
            ResponsesOutputItem(
                id=f"msg_{response.id}",
                type="message",
                role="assistant",
                content=[ResponsesContent(type="text", text=response.choices[0].message.content)] if response.choices else [],
                status="completed",
            )
        ],
        usage=ResponsesUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        ),
    )


from libs.schemas_common.responses import ResponsesOutputItem, ResponsesUsage
import time