import structlog
import traceback
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastapi import HTTPException, RequestValidationError
from pydantic import ValidationError

logger = structlog.get_logger()


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException:
            raise
        except RequestValidationError as e:
            logger.warning("validation_error", path=request.url.path, errors=e.errors())
            return JSONResponse(
                status_code=422,
                content={
                    "detail": "Validation error",
                    "errors": e.errors()
                }
            )
        except ValidationError as e:
            logger.warning("pydantic_validation_error", path=request.url.path, errors=e.errors())
            return JSONResponse(
                status_code=422,
                content={
                    "detail": "Validation error",
                    "errors": e.errors()
                }
            )
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.error(
                "unhandled_error",
                request_id=request_id,
                path=request.url.path,
                error=str(e),
                traceback=traceback.format_exc()
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "request_id": request_id
                }
            )


error_handling_middleware = ErrorHandlingMiddleware