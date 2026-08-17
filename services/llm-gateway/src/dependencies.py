from rag.pipeline import RAGPipeline
from rag.ingestion.pipeline import IngestionPipeline
from rag.embeddings.service import EmbeddingService
from rag.vectorstores.faiss_store import FAISSVectorStore
from rag.retrieval.retriever import Retriever
from rag.retrieval.reranker import Reranker
from libs.embeddings_common import EmbeddingClient
from rag.embeddings.providers.ollama import OllamaEmbeddingProvider
from .config.settings import settings

rag_pipeline: RAGPipeline = None

def init_rag():
    global rag_pipeline
    if rag_pipeline is not None:
        return

    # Create embedding provider (Ollama)
    embed_provider = OllamaEmbeddingProvider(
        host=settings.ollama_host,
        model=settings.embedding_model
    )
    embed_client = EmbeddingClient(provider=embed_provider)
    embed_service = EmbeddingService(embed_client)

    # Vector store
    vector_store = FAISSVectorStore(dimension=settings.embedding_dim)

    # Retriever
    retriever = Retriever(vector_store, embed_service, top_k=settings.rag_top_k)

    # Reranker (optional, can be None)
    reranker = Reranker()

    # Ingestion pipeline
    from rag.ingestion.chunker import TextChunker
    chunker = TextChunker(
        chunk_size=settings.rag_chunk_size,
        overlap=settings.rag_chunk_overlap
    )
    ingestion = IngestionPipeline(chunker=chunker)

    # RAG pipeline
    rag_pipeline = RAGPipeline(
        ingestion_pipeline=ingestion,
        embedding_service=embed_service,
        vector_store=vector_store,
        retriever=retriever,
        reranker=reranker
    )