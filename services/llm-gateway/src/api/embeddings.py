import structlog
import time
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel

from libs.schemas_common.embeddings import EmbeddingRequest, EmbeddingResponse
from libs.schemas_common.providers import ProviderName
from ..router.router import ModelRouter
from ..dependencies import get_router
from ..caching.cache import get_cache
from ..config.settings import settings

logger = structlog.get_logger()

router = APIRouter()


class EmbeddingsRequest(BaseModel):
    """OpenAI-compatible embeddings request."""
    input: List[str] | str
    model: Optional[str] = None
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingsRequest,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Generate embeddings."""
    start_time = time.time()
    request_id = getattr(http_request.state, "request_id", "unknown")

    try:
        # Convert to internal request
        internal_request = EmbeddingRequest(
            input=request.input,
            model=request.model,
            encoding_format=request.encoding_format,
            dimensions=request.dimensions,
            user=request.user,
        )

        # Check cache
        cache = get_cache()
        cache_key = None
        if cache and settings.embedding_cache_enabled:
            texts = request.input if isinstance(request.input, list) else [request.input]
            cache_key = f"embed:{request.model or 'auto'}:{hash(''.join(texts))}"
            cached = await cache.get(cache_key)
            if cached:
                logger.info("embedding_cache_hit", request_id=request_id)
                return cached

        # Route to embedding-capable provider
        provider = router.route(
            task_type="embedding",
            model=request.model,
        )

        response = await provider.generate_embeddings(internal_request)

        if cache and cache_key:
            await cache.set(cache_key, response)

        latency_ms = (time.time() - start_time) * 1000
        provider_name = getattr(provider, 'provider_name', 'unknown')
        await router.record_request(provider_name, latency_ms, True)

        logger.info("embeddings_success", request_id=request_id, latency_ms=latency_ms)
        return response

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("embeddings_failed", request_id=request_id, error=str(e), latency_ms=latency_ms)
        raise HTTPException(status_code=500, detail=str(e))