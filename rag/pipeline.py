import hashlib
from typing import List, Dict
from .ingestion.pipeline import IngestionPipeline
from .embeddings.service import EmbeddingService
from .vectorstores import VectorStore
from .retrieval.retriever import Retriever
from .retrieval.reranker import Reranker


class RAGPipeline:
    def __init__(self,
                 ingestion_pipeline: IngestionPipeline,
                 embedding_service: EmbeddingService,
                 vector_store: VectorStore,
                 retriever: Retriever,
                 reranker: Reranker = None):
        self.ingestion = ingestion_pipeline
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.retriever = retriever
        self.reranker = reranker

    async def ingest(self, source: str) -> int:
        """Ingest a document from a file path."""
        chunks = await self.ingestion.run(source)
        if not chunks:
            return 0

        # Get embeddings
        texts = [chunk.text for chunk in chunks]
        embedding_response = await self.embedding_service.client.embed(texts)
        embeddings = embedding_response.embeddings

        # Prepare metadata and IDs
        metadata_list = [chunk.metadata for chunk in chunks]
        # Compute content hash for each chunk to avoid duplicates
        ids = []
        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.text.encode('utf-8')).hexdigest()
            # Optionally prepend source to keep track: ids.append(f"{source}:{content_hash}")
            ids.append(content_hash)

        # Add to vector store (dedup happens inside)
        await self.vector_store.add(embeddings, metadata_list, ids)
        return len(chunks)

    async def query(self, query: str) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        docs = await self.retriever.retrieve(query)
        if self.reranker:
            docs = await self.reranker.rerank(query, docs)
        return docs