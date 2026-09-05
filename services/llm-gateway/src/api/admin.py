import structlog
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request, Depends, Header
from pydantic import BaseModel

from libs.schemas_common.auth import APIKey, APIKeyCreate, APIKeyResponse, UserRole
from libs.schemas_common.analytics import AnalyticsQuery, AnalyticsResponse
from ..router.router import ModelRouter
from ..dependencies import get_router, get_analytics
from ..auth.service import AuthService
from ..config.settings import settings

logger = structlog.get_logger()

router = APIRouter()


async def verify_admin_key(x_admin_key: str = Header(...)):
    """Verify admin API key."""
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")


@router.get("/admin/providers")
async def admin_list_providers(
    http_request: Request,
    router: ModelRouter = Depends(get_router),
    _: None = Depends(verify_admin_key),
):
    """List all providers with detailed status."""
    return router.get_provider_stats()


@router.post("/admin/providers/{provider_name}/enable")
async def admin_enable_provider(
    provider_name: str,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
    _: None = Depends(verify_admin_key),
):
    """Enable a provider."""
    from libs.schemas_common.providers import ProviderName
    try:
        pn = ProviderName(provider_name)
        if pn in router.providers:
            router.providers[pn].config.enabled = True
            return {"status": "enabled", "provider": provider_name}
        raise HTTPException(status_code=404, detail="Provider not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid provider name")


@router.post("/admin/providers/{provider_name}/disable")
async def admin_disable_provider(
    provider_name: str,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
    _: None = Depends(verify_admin_key),
):
    """Disable a provider."""
    from libs.schemas_common.providers import ProviderName
    try:
        pn = ProviderName(provider_name)
        if pn in router.providers:
            router.providers[pn].config.enabled = False
            return {"status": "disabled", "provider": provider_name}
        raise HTTPException(status_code=404, detail="Provider not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid provider name")


@router.get("/admin/routing-rules")
async def admin_get_routing_rules(
    http_request: Request,
    router: ModelRouter = Depends(get_router),
    _: None = Depends(verify_admin_key),
):
    """Get current routing rules."""
    return {
        task.value: {
            "preferred_providers": [p.value for p in rule.preferred_providers],
            "preferred_models": {k.value: v for k, v in rule.preferred_models.items()},
            "strategy": rule.strategy.value,
            "fallback_enabled": rule.fallback_enabled,
            "required_capabilities": rule.required_capabilities,
        }
        for task, rule in router.routing_rules.items()
    }


@router.put("/admin/routing-rules/{task_type}")
async def admin_update_routing_rule(
    task_type: str,
    rule_data: dict,
    http_request: Request,
    router: ModelRouter = Depends(get_router),
    _: None = Depends(verify_admin_key),
):
    """Update routing rule for a task type."""
    from libs.schemas_common.chat import TaskType
    from ..router.router import RoutingRule, RoutingStrategy
    from libs.schemas_common.providers import ProviderName

    try:
        tt = TaskType(task_type)
        rule = RoutingRule(
            task_type=tt,
            preferred_providers=[ProviderName(p) for p in rule_data.get("preferred_providers", [])],
            preferred_models={ProviderName(k): v for k, v in rule_data.get("preferred_models", {}).items()},
            strategy=RoutingStrategy(rule_data.get("strategy", "priority")),
            fallback_enabled=rule_data.get("fallback_enabled", True),
            required_capabilities=rule_data.get("required_capabilities"),
        )
        router.routing_rules[tt] = rule
        return {"status": "updated", "task_type": task_type}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/admin/api-keys", response_model=APIKeyResponse)
async def admin_create_api_key(
    request: APIKeyCreate,
    http_request: Request,
    auth_service: AuthService = Depends(lambda: None),  # TODO: inject properly
    _: None = Depends(verify_admin_key),
):
    """Create a new API key."""
    # TODO: Implement with auth_service
    return APIKeyResponse(
        id="temp",
        name=request.name,
        prefix="afk_",
        key="temp_key",
        roles=request.roles,
        scopes=request.scopes,
        rate_limit=request.rate_limit,
        expires_at=None,
        created_at=None,
    )


@router.get("/admin/api-keys", response_model=List[APIKey])
async def admin_list_api_keys(
    http_request: Request,
    auth_service: AuthService = Depends(lambda: None),
    _: None = Depends(verify_admin_key),
):
    """List all API keys."""
    # TODO: Implement with auth_service
    return []


@router.delete("/admin/api-keys/{key_id}")
async def admin_delete_api_key(
    key_id: str,
    http_request: Request,
    auth_service: AuthService = Depends(lambda: None),
    _: None = Depends(verify_admin_key),
):
    """Delete an API key."""
    # TODO: Implement with auth_service
    return {"status": "deleted", "key_id": key_id}


@router.get("/admin/analytics", response_model=AnalyticsResponse)
async def admin_get_analytics(
    query: AnalyticsQuery = Depends(),
    http_request: Request = None,
    analytics = Depends(get_analytics),
    _: None = Depends(verify_admin_key),
):
    """Get usage analytics."""
    return await analytics.query(query)


@router.get("/admin/costs")
async def admin_get_costs(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    http_request: Request = None,
    analytics = Depends(get_analytics),
    _: None = Depends(verify_admin_key),
):
    """Get cost breakdown."""
    # TODO: Implement cost breakdown
    return {"message": "Cost tracking endpoint"}


@router.post("/admin/cache/clear")
async def admin_clear_cache(
    http_request: Request,
    _: None = Depends(verify_admin_key),
):
    """Clear all caches."""
    from ..caching.cache import get_cache
    cache = get_cache()
    if cache:
        await cache.clear()
    return {"status": "cache_cleared"}


@router.post("/admin/providers/reload")
async def admin_reload_providers(
    http_request: Request,
    router: ModelRouter = Depends(get_router),
    _: None = Depends(verify_admin_key),
):
    """Reload provider configurations."""
    await router.close_providers()
    await router.initialize_providers()
    return {"status": "reloaded", "providers": list(router.providers.keys())}