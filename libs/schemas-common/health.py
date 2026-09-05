from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from enum import Enum


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthCheckResponse(BaseModel):
    status: HealthStatus
    version: str
    uptime_seconds: float
    components: List[ComponentHealth]
    timestamp: int


class OllamaHealthResponse(BaseModel):
    status: HealthStatus
    version: Optional[str] = None
    models: List[str] = []
    gpu_info: Optional[Dict[str, Any]] = None
    memory_usage: Optional[Dict[str, Any]] = None