import structlog
import time
import psutil
from typing import List
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel

from libs.schemas_common.health import HealthCheckResponse, HealthStatus, ComponentHealth, OllamaHealthResponse
from ..router.router import ModelRouter
from ..dependencies import get_router, get_rag_pipeline
from ..config.settings import settings

logger = structlog.get_logger()

router = APIRouter()

# Track startup time
_start_time = time.time()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Comprehensive health check."""
    components = []

    # Check each provider
    for name, state in router.providers.items():
        try:
            is_healthy = await state.provider.health_check()
            components.append(ComponentHealth(
                name=name.value,
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                latency_ms=state.health.get("latency_ms"),
            ))
        except Exception as e:
            components.append(ComponentHealth(
                name=name.value,
                status=HealthStatus.UNHEALTHY,
                error=str(e),
            ))

    # Check database
    try:
        # TODO: Check PostgreSQL connection
        components.append(ComponentHealth(
            name="postgresql",
            status=HealthStatus.HEALTHY,
        ))
    except Exception as e:
        components.append(ComponentHealth(
            name="postgresql",
            status=HealthStatus.UNHEALTHY,
            error=str(e),
        ))

    # Check Redis
    try:
        # TODO: Check Redis connection
        components.append(ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
        ))
    except Exception as e:
        components.append(ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            error=str(e),
        ))

    # Check Qdrant
    try:
        # TODO: Check Qdrant connection
        components.append(ComponentHealth(
            name="qdrant",
            status=HealthStatus.HEALTHY,
        ))
    except Exception as e:
        components.append(ComponentHealth(
            name="qdrant",
            status=HealthStatus.UNHEALTHY,
            error=str(e),
        ))

    # Overall status
    unhealthy_count = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)
    degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)

    if unhealthy_count > 0:
        overall_status = HealthStatus.UNHEALTHY
    elif degraded_count > 0:
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.HEALTHY

    return HealthCheckResponse(
        status=overall_status,
        version=settings.app_version,
        uptime_seconds=time.time() - _start_time,
        components=components,
        timestamp=int(time.time()),
    )


@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@router.get("/health/ollama", response_model=OllamaHealthResponse)
async def ollama_health(
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Ollama-specific health check with GPU info."""
    if ProviderName.OLLAMA not in router.providers:
        return OllamaHealthResponse(
            status=HealthStatus.UNHEALTHY,
            error="Ollama provider not configured",
        )

    ollama_provider = router.providers[ProviderName.OLLAMA].provider

    try:
        is_healthy = await ollama_provider.health_check()
        models = await ollama_provider.list_models()
        gpu_info = await ollama_provider.get_gpu_info()

        return OllamaHealthResponse(
            status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
            models=[m.id for m in models],
            gpu_info=gpu_info,
        )
    except Exception as e:
        logger.error("ollama_health_failed", error=str(e))
        return OllamaHealthResponse(
            status=HealthStatus.UNHEALTHY,
            error=str(e),
        )


@router.get("/health/providers")
async def providers_health(
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Get health status for all providers."""
    return router.get_provider_stats()


@router.get("/stats")
async def get_stats(
    http_request: Request,
    router: ModelRouter = Depends(get_router),
):
    """Get service statistics."""
    import time
    stats = router.get_provider_stats()

    total_requests = sum(s["request_count"] for s in stats.values())
    total_errors = sum(s["error_count"] for s in stats.values())

    return {
        "uptime_seconds": time.time() - _start_time,
        "total_requests": total_requests,
        "total_errors": total_errors,
        "error_rate": total_errors / total_requests if total_requests > 0 else 0,
        "providers": stats,
    }