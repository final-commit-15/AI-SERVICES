from typing import List, Optional
from .base import EmbeddingProvider
from .models import EmbeddingRequest, EmbeddingResponse


class EmbeddingClient:
    """High‑level client that uses a provider and handles caching."""

    def __init__(self, provider: EmbeddingProvider, cache=None):
        self.provider = provider
        self.cache = cache  # optional cache

    async def embed(self, texts: List[str], model: Optional[str] = None) -> EmbeddingResponse:
        # Check cache if available
        if self.cache:
            cache_key = f"embed:{model or ''}:{''.join(texts)}"
            cached = await self.cache.get(cache_key)
            if cached:
                return EmbeddingResponse(**cached)
        request = EmbeddingRequest(texts=texts, model=model)
        response = await self.provider.embed(request)
        if self.cache:
            await self.cache.set(cache_key, response.model_dump())
        return response