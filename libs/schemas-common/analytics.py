from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class MetricType(str, Enum):
    REQUEST_COUNT = "request_count"
    TOKEN_USAGE = "token_usage"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    COST = "cost"
    PROVIDER_USAGE = "provider_usage"
    MODEL_USAGE = "model_usage"
    TASK_TYPE_USAGE = "task_type_usage"


class UsageMetrics(BaseModel):
    id: str
    timestamp: datetime
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    provider: str
    model: str
    task_type: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    status: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class AnalyticsQuery(BaseModel):
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    user_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    task_type: Optional[str] = None
    group_by: Optional[str] = None
    limit: int = 100


class AnalyticsResponse(BaseModel):
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    error_rate: float
    by_provider: Dict[str, "ProviderAnalytics"]
    by_model: Dict[str, "ModelAnalytics"]
    by_task_type: Dict[str, "TaskTypeAnalytics"]
    time_series: List["TimeSeriesPoint"]


class ProviderAnalytics(BaseModel):
    requests: int
    tokens: int
    cost_usd: float
    avg_latency_ms: float
    error_rate: float


class ModelAnalytics(BaseModel):
    requests: int
    tokens: int
    cost_usd: float
    avg_latency_ms: float


class TaskTypeAnalytics(BaseModel):
    requests: int
    tokens: int
    avg_latency_ms: float


class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    requests: int
    tokens: int
    cost_usd: float
    latency_ms: float


class CostAlert(BaseModel):
    id: str
    threshold_usd: float
    current_spend_usd: float
    period: str
    triggered_at: datetime
    acknowledged: bool = False