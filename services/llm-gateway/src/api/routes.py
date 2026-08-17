from fastapi import APIRouter, HTTPException, Body
from typing import Optional, List, Dict, Any
from ..providers.base import LLMProvider          # relative is fine for internal
from ..router.router import ModelRouter
from ..caching.cache import get_cache
from ..guardrails.guardrails import apply_input_guardrails, apply_output_guardrails
from ..prompt_registry.registry import PromptRegistry
from ..config.settings import settings
from .. import dependencies
from pathlib import Path
import logging
from .. import dependencies

logger = logging.getLogger(__name__)

router = APIRouter()
model_router = ModelRouter()
cache = get_cache()
prompt_registry = PromptRegistry(Path(__file__).parent.parent / "prompt_registry" / "templates")

@router.post("/chat")
async def chat(
    messages: List[Dict[str, str]] = Body(...),
    provider: Optional[str] = None,
    task_type: Optional[str] = None,
    temperature: float = 0.0,
    thinking: bool = False,
    response_format: Optional[Dict[str, Any]] = None,
):
    # Input guardrails
    if messages:
        last_msg = messages[-1].get("content", "")
        if not await apply_input_guardrails(last_msg):
            raise HTTPException(status_code=400, detail="Input failed guardrails")

    # Route to provider
    provider_obj = model_router.route(
        task_type=task_type or settings.default_task_type,
        provider=provider,
    )

    # Cache check
    cache_key = None
    if cache:
        # simple cache key from messages and parameters
        cache_key = f"chat:{provider_obj.model_name}:{str(messages)}:{temperature}:{thinking}"
        cached = await cache.get(cache_key)
        if cached:
            return cached

    # Call LLM
    try:
        response = await provider_obj.chat(
            messages=messages,
            temperature=temperature,
            thinking=thinking,
            response_format=response_format,
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Output guardrails
    content = response.get("message", {}).get("content", "")
    if not await apply_output_guardrails(content):
        raise HTTPException(status_code=500, detail="Output failed guardrails")

    # Cache store
    if cache and cache_key:
        await cache.set(cache_key, response)

    return response


@router.post("/generate")
async def generate(
    prompt: str = Body(...),
    provider: Optional[str] = None,
    task_type: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
):
    if not await apply_input_guardrails(prompt):
        raise HTTPException(status_code=400, detail="Input failed guardrails")

    provider_obj = model_router.route(
        task_type=task_type or settings.default_task_type,
        provider=provider,
    )

    cache_key = None
    if cache:
        cache_key = f"generate:{provider_obj.model_name}:{prompt}:{temperature}:{max_tokens}"
        cached = await cache.get(cache_key)
        if cached:
            return {"result": cached}

    try:
        result = await provider_obj.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not await apply_output_guardrails(result):
        raise HTTPException(status_code=500, detail="Output failed guardrails")

    if cache and cache_key:
        await cache.set(cache_key, result)

    return {"result": result}


@router.post("/embed")
async def embed(
    texts: List[str] = Body(...),
    model: Optional[str] = None,
):
    if dependencies.rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    response = await dependencies.rag_pipeline.embedding_service.client.embed(
        texts,
    model=model,
)
    return response.model_dump()


@router.post("/rag/query")
async def rag_query(query: str = Body(...)):
    if dependencies.rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    docs = await dependencies.rag_pipeline.query(query)
    return {"documents": docs}


@router.post("/rag/ingest")
async def rag_ingest(source: str = Body(...)):
    if dependencies.rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    count = await dependencies.rag_pipeline.ingest(source)
    return {"ingested": count}


@router.get("/models")
async def list_models(provider: Optional[str] = None):
    # Get list of models from configured providers
    # For ollama, we can list
    if provider == "ollama":
        from ..providers.local.ollama_provider import OllamaProvider
        ollama = OllamaProvider()
        try:
            resp = await ollama.list_models()
            return {"provider": "ollama", "models": [m.model for m in resp.models]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"providers": ["ollama", "openai"]}


@router.get("/health/ollama")
async def ollama_health():
    from ..providers.local.ollama_provider import OllamaProvider
    ollama = OllamaProvider()
    try:
        resp = await ollama.list_models()
        return {
            "status": "healthy",
            "provider": "ollama",
            "model": ollama.model,
            "models": [m.model for m in resp.models],
        }
    except Exception as e:
        return {"status": "unhealthy", "provider": "ollama", "error": str(e)}

@router.post("/rag/chat")
async def rag_chat(
    query: str = Body(...),
    provider: Optional[str] = None,
    temperature: float = 0.0,
    thinking: bool = False,
):
    if dependencies.rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    docs = await dependencies.rag_pipeline.query(query)
    logger.info(f"🔍 RAG Chat - query: '{query}'")
    logger.info(f"📄 docs count: {len(docs)}")
    if docs:
        logger.info(f"📄 first doc: {docs[0].get('content', '')[:100]}...")
        
    context = "\n\n".join(
        [doc.get("content", "") for doc in docs]
    )

    if not context:
        prompt = f"Answer the question: {query}"
    else:
        prompt = (
            "Answer the question based only on the following context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )

    provider_obj = model_router.route(
        task_type="rag",
        provider=provider,
    )

    answer = await provider_obj.generate(
        prompt,
        temperature=temperature,
        thinking=thinking,
    )

    return {
        "answer": answer,
        "source_documents": docs,
    }