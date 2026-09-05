import structlog
from typing import Optional
from contextlib import asynccontextmanager

from .router.router import ModelRouter
from .config.settings import settings
from .caching.cache import Cache, RedisCache, InMemoryCache
from .rag.pipeline import RAGPipeline
from .memory.manager import MemoryManager
from .tools.registry import ToolRegistry
from .analytics.service import AnalyticsService
from .auth.service import AuthService

logger = structlog.get_logger()

# Global instances
_router: Optional[ModelRouter] = None
_cache: Optional[Cache] = None
_rag_pipeline: Optional[RAGPipeline] = None
_memory_manager: Optional[MemoryManager] = None
_tool_registry: Optional[ToolRegistry] = None
_analytics: Optional[AnalyticsService] = None
_auth_service: Optional[AuthService] = None


async def init_services():
    """Initialize all services."""
    global _router, _cache, _rag_pipeline, _memory_manager, _tool_registry, _analytics, _auth_service

    logger.info("initializing_services")

    # Initialize cache
    _cache = await _init_cache()

    # Initialize router and providers
    _router = ModelRouter()
    await _router.initialize_providers()

    # Initialize RAG pipeline
    _rag_pipeline = await _init_rag()

    # Initialize memory manager
    _memory_manager = await _init_memory()

    # Initialize tool registry
    _tool_registry = await _init_tools()

    # Initialize analytics
    _analytics = await _init_analytics()

    # Initialize auth service
    _auth_service = await _init_auth()

    logger.info("services_initialized")


async def close_services():
    """Close all services."""
    global _router, _cache, _rag_pipeline, _memory_manager, _tool_registry, _analytics, _auth_service

    logger.info("closing_services")

    if _router:
        await _router.close_providers()

    if _cache:
        await _cache.close()

    if _rag_pipeline:
        await _rag_pipeline.close()

    if _memory_manager:
        await _memory_manager.close()

    if _analytics:
        await _analytics.close()

    logger.info("services_closed")


async def _init_cache():
    """Initialize cache backend."""
    if settings.cache_redis_enabled:
        try:
            cache = RedisCache(
                url=settings.redis_url,
                max_size=settings.cache_max_size,
                ttl=settings.cache_ttl_seconds,
            )
            await cache.connect()
            logger.info("redis_cache_initialized")
            return cache
        except Exception as e:
            logger.warning("redis_cache_failed_fallback_to_memory", error=str(e))

    # Fallback to in-memory
    return InMemoryCache(
        max_size=settings.cache_max_size,
        ttl=settings.cache_ttl_seconds,
    )


async def _init_rag():
    """Initialize RAG pipeline."""
    try:
        from .rag.pipeline import RAGPipeline
        from .rag.ingestion.pipeline import IngestionPipeline
        from .rag.ingestion.chunker import TextChunker
        from .rag.embeddings.service import EmbeddingService
        from .rag.vectorstores.qdrant_store import QdrantVectorStore
        from .rag.retrieval.retriever import Retriever
        from .rag.retrieval.reranker import Reranker
        from libs.embeddings_common.client import EmbeddingClient
        from .providers.ollama import OllamaEmbeddingProvider

        # Create embedding provider
        embed_provider = OllamaEmbeddingProvider(
            host=settings.ollama_host,
            model=settings.embedding_model,
        )
        embed_client = EmbeddingClient(provider=embed_provider, cache=_cache)
        embed_service = EmbeddingService(embed_client)

        # Vector store
        vector_store = QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_prefix=settings.qdrant_collection_prefix,
            dimension=settings.embedding_dim,
        )
        await vector_store.connect()

        # Retriever
        retriever = Retriever(vector_store, embed_service, top_k=settings.rag_top_k)

        # Reranker
        reranker = Reranker(model=settings.rag_rerank_model) if settings.rag_rerank_enabled else None

        # Ingestion pipeline
        chunker = TextChunker(
            chunk_size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
        )
        ingestion = IngestionPipeline(chunker=chunker)

        # RAG pipeline
        rag = RAGPipeline(
            ingestion_pipeline=ingestion,
            embedding_service=embed_service,
            vector_store=vector_store,
            retriever=retriever,
            reranker=reranker,
        )

        logger.info("rag_pipeline_initialized")
        return rag
    except Exception as e:
        logger.warning("rag_pipeline_initialization_failed", error=str(e))
        return None


async def _init_memory():
    """Initialize memory manager."""
    try:
        memory = MemoryManager(
            redis_url=settings.redis_url,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
            short_term_ttl=settings.memory_short_term_ttl,
            long_term_ttl=settings.memory_long_term_ttl,
            max_messages=settings.memory_max_messages,
        )
        await memory.connect()
        logger.info("memory_manager_initialized")
        return memory
    except Exception as e:
        logger.warning("memory_manager_initialization_failed", error=str(e))
        return None


async def _init_tools():
    """Initialize tool registry."""
    try:
        tool_registry = ToolRegistry()
        # Register builtin tools
        from .tools.builtin import register_builtin_tools
        await register_builtin_tools(tool_registry)
        logger.info("tool_registry_initialized")
        return tool_registry
    except Exception as e:
        logger.warning("tool_registry_initialization_failed", error=str(e))
        return None


async def _init_analytics():
    """Initialize analytics service."""
    try:
        analytics = AnalyticsService(
            database_url=settings.database_url,
            retention_days=settings.analytics_retention_days,
        )
        await analytics.connect()
        logger.info("analytics_service_initialized")
        return analytics
    except Exception as e:
        logger.warning("analytics_service_initialization_failed", error=str(e))
        return None


async def _init_auth():
    """Initialize auth service."""
    try:
        auth = AuthService(
            jwt_secret=settings.jwt_secret_key,
            jwt_algorithm=settings.jwt_algorithm,
            access_token_expire=settings.jwt_access_token_expire_minutes,
            refresh_token_expire=settings.jwt_refresh_token_expire_days,
        )
        await auth.connect()
        logger.info("auth_service_initialized")
        return auth
    except Exception as e:
        logger.warning("auth_service_initialization_failed", error=str(e))
        return None


# Dependency providers
def get_router() -> ModelRouter:
    if _router is None:
        raise RuntimeError("Router not initialized")
    return _router


def get_cache() -> Optional[Cache]:
    return _cache


def get_rag_pipeline() -> Optional[RAGPipeline]:
    return _rag_pipeline


def get_memory_manager() -> Optional[MemoryManager]:
    return _memory_manager


def get_tool_registry() -> Optional[ToolRegistry]:
    return _tool_registry


def get_analytics() -> Optional[AnalyticsService]:
    return _analytics


def get_auth_service() -> Optional[AuthService]:
    return _auth_service