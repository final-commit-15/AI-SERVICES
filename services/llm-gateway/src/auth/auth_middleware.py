import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from jose import jwt, JWTError
from typing import Optional

from ..config.settings import settings
from ..schemas.auth import TokenData, UserRole

logger = structlog.get_logger()

# Public paths that don't require authentication
PUBLIC_PATHS = {
    "/health",
    "/health/ready",
    "/health/live",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/",
}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.auth_enabled:
            return await call_next(request)

        path = request.url.path

        # Skip auth for public paths
        if any(path.startswith(p) for p in PUBLIC_PATHS):
            return await call_next(request)

        # Check for API key first
        api_key = request.headers.get(settings.api_key_header)
        if api_key:
            # Validate API key (would check against database)
            user_data = await self._validate_api_key(api_key)
            if user_data:
                request.state.user = user_data
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"}
            )

        # Check for JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                payload = jwt.decode(
                    token,
                    settings.jwt_secret_key,
                    algorithms=[settings.jwt_algorithm]
                )
                token_data = TokenData(**payload)
                request.state.user = token_data
                return await call_next(request)
            except JWTError as e:
                logger.warning("jwt_validation_failed", error=str(e))
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or expired token"}
                )

        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required"}
        )

    async def _validate_api_key(self, api_key: str) -> Optional[TokenData]:
        """Validate API key against database/cache."""
        # In production, check against database
        # For now, check against admin key
        if api_key == settings.admin_api_key:
            return TokenData(
                sub="admin",
                user_id="admin",
                roles=[UserRole.ADMIN],
                scopes=["*"],
                exp=9999999999,
                iat=0,
                jti="admin"
            )
        return None


auth_middleware = AuthMiddleware