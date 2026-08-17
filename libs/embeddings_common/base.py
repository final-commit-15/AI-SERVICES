from abc import ABC, abstractmethod
from typing import List, Optional

from .models import EmbeddingRequest, EmbeddingResponse


class EmbeddingProvider(ABC):
    """Abstract base for all embedding providers."""

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for the given text(s)."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding for documents."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""
        pass