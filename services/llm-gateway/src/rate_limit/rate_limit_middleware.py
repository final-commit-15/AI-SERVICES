import time
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from collections import defaultdict
from typing import Dict, Tuple

from ..config.settings import settings

logger = structlog.get_logger()

# In-memory rate limit store (use Redis in production)
rate_limit_store: Dict[str, Dict[str, Tuple[int, float]]] = defaultdict(dict)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.rate_limit_enabled:
            return await call_next(request)

        path = request.url.path

        # Skip rate limiting for health checks
        if path.startswith("/health"):
            return await call_next(request)

        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        user_id = getattr(request.state, "user", None)
        identifier = user_id.user_id if user_id else client_ip

        # Get rate limit config
        rpm = settings.rate_limit_requests_per_minute
        rph = settings.rate_limit_requests_per_hour
        burst = settings.rate_limit_burst

        # Check custom rate limit from user
        if user_id and hasattr(user_id, "rate_limit") and user_id.rate_limit:
            rpm = user_id.rate_limit

        current_time = time.time()
        minute_key = f"{identifier}:{int(current_time / 60)}"
        hour_key = f"{identifier}:{int(current_time / 3600)}"

        # Clean old entries
        self._cleanup_store(current_time)

        # Check minute limit
        minute_count = rate_limit_store[identifier].get(minute_key, (0, current_time))[0]
        if minute_count >= rpm:
            logger.warning("rate_limit_exceeded", identifier=identifier, limit=rpm, window="minute")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "limit": rpm,
                    "window": "minute",
                    "retry_after": 60 - int(current_time % 60)
                },
                headers={
                    "X-RateLimit-Limit": str(rpm),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time / 60) * 60 + 60),
                    "Retry-After": str(60 - int(current_time % 60))
                }
            )

        # Check hour limit
        hour_count = rate_limit_store[identifier].get(hour_key, (0, current_time))[0]
        if hour_count >= rph:
            logger.warning("rate_limit_exceeded", identifier=identifier, limit=rph, window="hour")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "limit": rph,
                    "window": "hour",
                    "retry_after": 3600 - int(current_time % 3600)
                },
                headers={
                    "X-RateLimit-Limit": str(rph),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time / 3600) * 3600 + 3600),
                    "Retry-After": str(3600 - int(current_time % 3600))
                }
            )

        # Increment counters
        rate_limit_store[identifier][minute_key] = (minute_count + 1, current_time)
        rate_limit_store[identifier][hour_key] = (hour_count + 1, current_time)

        response = await call_next(request)

        # Add rate limit headers
        remaining_minute = max(0, rpm - minute_count - 1)
        remaining_hour = max(0, rph - hour_count - 1)
        response.headers["X-RateLimit-Limit-Minute"] = str(rpm)
        response.headers["X-RateLimit-Remaining-Minute"] = str(remaining_minute)
        response.headers["X-RateLimit-Limit-Hour"] = str(rph)
        response.headers["X-RateLimit-Remaining-Hour"] = str(remaining_hour)

        return response

    def _cleanup_store(self, current_time: float):
        """Remove expired entries."""
        minute_threshold = current_time - 120  # Keep 2 minutes
        hour_threshold = current_time - 7200   # Keep 2 hours

        for identifier in list(rate_limit_store.keys()):
            for key in list(rate_limit_store[identifier].keys()):
                _, timestamp = rate_limit_store[identifier][key]
                if ":0:" in key and timestamp < hour_threshold:
                    del rate_limit_store[identifier][key]
                elif timestamp < minute_threshold:
                    del rate_limit_store[identifier][key]
            if not rate_limit_store[identifier]:
                del rate_limit_store[identifier]


rate_limit_middleware = RateLimitMiddleware