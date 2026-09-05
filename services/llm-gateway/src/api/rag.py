import structlog
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request, Depends, UploadFile, File, Form
from pydantic import BaseModel

from libs.schemas_common.chat import ChatMessage, MessageRole
from ..router.router import ModelRouter
from ..dependencies import get_router, get_rag_pipeline
from ..memory.manager import MemoryManager

logger = structlog.get_logger()

router = APIRouter()


class RagQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    collection: Optional[str] = None
    filter: Optional[dict] = None


class RagIngestRequest(BaseModel):
    source: str
    collection: Optional[str] = None
    metadata: Optional[dict] = None


class RagChatRequest(BaseModel):
    query: str
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.0
    top_k: Optional[int] = None
    collection: Optional[str] = None


@router.post("/rag/query")
async def rag_query(
    request: RagQueryRequest,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Query the RAG pipeline."""
    rag_pipeline = get_rag_pipeline()
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        docs = await rag_pipeline.query(
            query=request.query,
            top_k=request.top_k,
            collection=request.collection,
            filter=request.filter,
        )
        return {"documents": docs}
    except Exception as e:
        logger.error("rag_query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/ingest")
async def rag_ingest(
    request: RagIngestRequest,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Ingest a document into the RAG pipeline."""
    rag_pipeline = get_rag_pipeline()
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        count = await rag_pipeline.ingest(
            source=request.source,
            collection=request.collection,
            metadata=request.metadata,
        )
        return {"ingested": count}
    except Exception as e:
        logger.error("rag_ingest_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/ingest/file")
async def rag_ingest_file(
    file: UploadFile = File(...),
    collection: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    http_request: Request = None,
    router: ModelRouter = Depends(get_router),
):
    """Ingest a file into the RAG pipeline."""
    rag_pipeline = get_rag_pipeline()
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        import json
        meta = json.loads(metadata) if metadata else {}
        meta["filename"] = file.filename
        meta["content_type"] = file.content_type

        # Save file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        count = await rag_pipeline.ingest(source=tmp_path, collection=collection, metadata=meta)
        return {"ingested": count}
    except Exception as e:
        logger.error("rag_ingest_file_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/chat")
async def rag_chat(
    request: RagChatRequest,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Chat with RAG context."""
    rag_pipeline = get_rag_pipeline()
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        # Retrieve relevant documents
        docs = await rag_pipeline.query(
            query=request.query,
            top_k=request.top_k,
            collection=request.collection,
        )

        # Build context
        context = "\n\n".join([doc.get("content", "") for doc in docs])

        if not context:
            prompt = f"Answer the question: {request.query}"
        else:
            prompt = (
                "Answer the question based only on the following context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {request.query}"
            )

        # Route to provider
        provider_name = request.provider
        if provider_name:
            from libs.schemas_common.providers import ProviderName
            provider = router.route(provider=ProviderName(provider_name), model=request.model)
        else:
            provider = router.route(task_type="rag", model=request.model)

        # Generate answer
        from libs.schemas_common.chat import ChatRequest, ChatMessage, MessageRole
        chat_request = ChatRequest(
            messages=[ChatMessage(role=MessageRole.USER, content=prompt)],
            model=request.model,
            temperature=request.temperature,
        )

        response = await provider.chat(chat_request)

        return {
            "answer": response.choices[0].message.content if response.choices else "",
            "source_documents": docs,
            "usage": response.usage.model_dump() if response.usage else {},
        }
    except Exception as e:
        logger.error("rag_chat_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/collections")
async def list_collections(
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """List RAG collections."""
    rag_pipeline = get_rag_pipeline()
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        collections = await rag_pipeline.list_collections()
        return {"collections": collections}
    except Exception as e:
        logger.error("list_collections_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rag/collections/{collection_name}")
async def delete_collection(
    collection_name: str,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Delete a RAG collection."""
    rag_pipeline = get_rag_pipeline()
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        await rag_pipeline.delete_collection(collection_name)
        return {"status": "success", "collection": collection_name}
    except Exception as e:
        logger.error("delete_collection_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))