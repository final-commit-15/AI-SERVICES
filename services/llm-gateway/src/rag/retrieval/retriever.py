import structlog
from typing import List, Dict, Optional, Any

logger = structlog.get_logger()


class Retriever:
    """Document retriever using vector store."""

    def __init__(self, vector_store, embedding_service, top_k: int = 5):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.top_k = top_k

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        k = top_k or self.top_k

        # Embed query
        query_embedding = await self.embedding_service.embed_query(query)

        # Search vector store
        results = await self.vector_store.search(
            query_vector=query_embedding,
            top_k=k,
            collection=collection,
            filter=filter,
        )

        return results