from typing import List, Dict, Tuple
from ..vectorstores import VectorStore
from ..embeddings.service import EmbeddingService


class Retriever:
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService, top_k: int = 5):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.top_k = top_k

    async def retrieve(self, query: str) -> List[Dict]:
        # Get embedding for query
        embeddings = await self.embedding_service.client.embed([query])
        query_vec = embeddings.embeddings[0]
        # Search – returns (metadata, score) where metadata already contains content and score
        results = await self.vector_store.search(query_vec, top_k=self.top_k)
        # Return metadata (which now includes content and score)
        return [meta for meta, _ in results]