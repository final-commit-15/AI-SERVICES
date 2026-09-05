import structlog
from typing import List
from libs.embeddings_common.client import EmbeddingClient
from libs.embeddings_common.models import EmbeddingRequest

logger = structlog.get_logger()


class EmbeddingService:
    """High-level embedding service."""

    def __init__(self, client: EmbeddingClient):
        self.client = client

    async def embed_chunks(self, chunks) -> List[List[float]]:
        """Embed document chunks."""
        texts = [chunk.text for chunk in chunks]
        response = await self.client.embed(texts)
        return response.embeddings

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        response = await self.client.embed(texts)
        return response.embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        response = await self.client.embed([query])
        return response.embeddings[0] if response.embeddings else []