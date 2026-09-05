import time
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from prometheus_client import Counter, Histogram, Gauge

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"]
)
ACTIVE_REQUESTS = Gauge(
    "http_requests_active",
    "Active HTTP requests",
    ["method", "endpoint"]
)
PROVIDER_REQUESTS = Counter(
    "provider_requests_total",
    "Total provider requests",
    ["provider", "model", "status"]
)
PROVIDER_LATENCY = Histogram(
    "provider_request_duration_seconds",
    "Provider request latency in seconds",
    ["provider", "model"]
)
TOKEN_USAGE = Counter(
    "token_usage_total",
    "Total tokens used",
    ["provider", "model", "type"]
)
COST_USD = Counter(
    "cost_usd_total",
    "Total cost in USD",
    ["provider", "model"]
)
ERROR_COUNT = Counter(
    "errors_total",
    "Total errors",
    ["type", "endpoint"]
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        method = request.method
        path = request.url.path

        ACTIVE_REQUESTS.labels(method=method, endpoint=path).inc()
        start_time = time.time()

        try:
            response = await call_next(request)
            duration = time.time() - start_time

            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status=response.status_code
            ).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)

            return response
        except Exception as e:
            ERROR_COUNT.labels(type=type(e).__name__, endpoint=path).inc()
            raise
        finally:
            ACTIVE_REQUESTS.labels(method=method, endpoint=path).dec()


metrics_middleware = MetricsMiddleware