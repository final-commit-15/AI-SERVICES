import structlog
from typing import List, Dict, Optional, Any
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)

logger = structlog.get_logger()


class QdrantVectorStore:
    """Qdrant vector store implementation."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_prefix: str = "agentforge",
        dimension: int = 768,
    ):
        self.url = url
        self.api_key = api_key
        self.collection_prefix = collection_prefix
        self.dimension = dimension
        self._client: Optional[AsyncQdrantClient] = None

    async def connect(self):
        """Connect to Qdrant."""
        self._client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
        )
        # Test connection
        await self._client.get_collections()
        logger.info("qdrant_connected", url=self.url)

    def _collection_name(self, collection: Optional[str] = None) -> str:
        """Get full collection name with prefix."""
        if collection:
            return f"{self.collection_prefix}_{collection}"
        return f"{self.collection_prefix}_default"

    async def _ensure_collection(self, collection: Optional[str] = None):
        """Ensure collection exists."""
        name = self._collection_name(collection)
        collections = await self._client.get_collections()
        existing = [c.name for c in collections.collections]

        if name not in existing:
            await self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("qdrant_collection_created", collection=name)

    async def add(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: List[str],
        collection: Optional[str] = None,
    ):
        """Add vectors to collection."""
        await self._ensure_collection(collection)
        name = self._collection_name(collection)

        points = [
            PointStruct(
                id=id_,
                vector=emb,
                payload=meta,
            )
            for id_, emb, meta in zip(ids, embeddings, metadata)
        ]

        await self._client.upsert(collection_name=name, points=points)
        logger.debug("qdrant_vectors_added", count=len(points), collection=name)

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        collection: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        await self._ensure_collection(collection)
        name = self._collection_name(collection)

        qdrant_filter = None
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            qdrant_filter = Filter(must=conditions)

        results = await self._client.search(
            collection_name=name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
            }
            for hit in results
        ]

    async def list_collections(self) -> List[str]:
        """List all collections with prefix."""
        collections = await self._client.get_collections()
        return [
            c.name.replace(f"{self.collection_prefix}_", "")
            for c in collections.collections
            if c.name.startswith(self.collection_prefix)
        ]

    async def delete_collection(self, collection: str):
        """Delete a collection."""
        name = self._collection_name(collection)
        await self._client.delete_collection(collection_name=name)
        logger.info("qdrant_collection_deleted", collection=name)

    async def close(self):
        """Close connection."""
        if self._client:
            await self._client.close()