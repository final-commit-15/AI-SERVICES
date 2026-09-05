import structlog
import hashlib
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

logger = structlog.get_logger()


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class RAGPipeline:
    """End-to-end RAG pipeline."""

    def __init__(
        self,
        ingestion_pipeline,
        embedding_service,
        vector_store,
        retriever,
        reranker=None,
    ):
        self.ingestion = ingestion_pipeline
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.retriever = retriever
        self.reranker = reranker

    async def ingest(
        self,
        source: str,
        collection: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Ingest a document from file path or text."""
        chunks = await self.ingestion.run(source, metadata)
        if not chunks:
            return 0

        # Get embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = await self.embedding_service.embed_documents(texts)

        # Prepare data for vector store
        metadata_list = [chunk.metadata for chunk in chunks]
        ids = []
        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.text.encode('utf-8')).hexdigest()
            ids.append(content_hash)

        # Add to vector store
        await self.vector_store.add(
            embeddings=embeddings,
            metadata=metadata_list,
            ids=ids,
            collection=collection,
        )
        return len(chunks)

    async def query(
        self,
        query: str,
        top_k: int = 5,
        collection: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        docs = await self.retriever.retrieve(
            query=query,
            top_k=top_k,
            collection=collection,
            filter=filter,
        )
        if self.reranker:
            docs = await self.reranker.rerank(query, docs)
        return docs

    async def list_collections(self) -> List[str]:
        """List all collections."""
        return await self.vector_store.list_collections()

    async def delete_collection(self, collection: str):
        """Delete a collection."""
        await self.vector_store.delete_collection(collection)

    async def close(self):
        """Close connections."""
        if hasattr(self.vector_store, 'close'):
            await self.vector_store.close()