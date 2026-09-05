import structlog
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request, Depends, Query

from libs.schemas_common.providers import ProviderName, ModelInfo, ProviderConfig, ProviderListResponse
from ..router.router import ModelRouter
from ..dependencies import get_router
from ..config.settings import settings

logger = structlog.get_logger()

router = APIRouter()


@router.get("/models", response_model=ProviderListResponse)
async def list_models(
    http_request: Request,
    router: ModelRouter = Depends(get_router),
    provider: Optional[str] = Query(None, description="Filter by provider"),
):
    """List all available models from all providers."""
    try:
        if provider:
            provider_name = ProviderName(provider)
            if provider_name not in router.providers:
                raise HTTPException(status_code=404, detail=f"Provider not found: {provider}")
            models = router.providers[provider_name].provider.models
            return ProviderListResponse(providers=[router.providers[provider_name].config], models=models)

        all_models = router.get_all_models()
        provider_configs = [state.config for state in router.providers.values()]

        return ProviderListResponse(
            providers=provider_configs,
            models=all_models,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("list_models_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Get details for a specific model."""
    provider = router.get_provider_for_model(model_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    model_info = provider.get_model_info(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model info not found: {model_id}")

    return model_info


@router.post("/models/pull")
async def pull_model(
    model_name: str,
    provider: str = "ollama",
    http_request: Request = None,
    router: ModelRouter = Depends(get_router),
):
    """Pull a model (Ollama only)."""
    if provider != "ollama":
        raise HTTPException(status_code=400, detail="Model pulling only supported for Ollama")

    if ProviderName.OLLAMA not in router.providers:
        raise HTTPException(status_code=404, detail="Ollama provider not available")

    ollama_provider = router.providers[ProviderName.OLLAMA].provider

    try:
        async for progress in ollama_provider.pull_model(model_name):
            # In production, stream progress via WebSocket or SSE
            pass
        return {"status": "success", "model": model_name}
    except Exception as e:
        logger.error("pull_model_failed", model=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    provider: str = "ollama",
    http_request: Request = None,
    router: ModelRouter = Depends(get_router),
):
    """Delete a model (Ollama only)."""
    if provider != "ollama":
        raise HTTPException(status_code=400, detail="Model deletion only supported for Ollama")

    if ProviderName.OLLAMA not in router.providers:
        raise HTTPException(status_code=404, detail="Ollama provider not available")

    ollama_provider = router.providers[ProviderName.OLLAMA].provider

    try:
        success = await ollama_provider.delete_model(model_name)
        if success:
            return {"status": "success", "model": model_name}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete model")
    except Exception as e:
        logger.error("delete_model_failed", model=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))