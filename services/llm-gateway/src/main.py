import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from .config.settings import settings
from .api import chat, responses, embeddings, rag, models, health, admin, speech, vision, images, memory, tools
from .middleware import logging_middleware, metrics_middleware, error_handling_middleware
from .auth import auth_middleware
from .rate_limit import rate_limit_middleware
from .dependencies import init_services, close_services

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("starting_application", version=settings.app_version, environment=settings.environment)
    await init_services()
    logger.info("services_initialized")
    yield
    logger.info("shutting_down_application")
    await close_services()
    logger.info("services_closed")


app = FastAPI(
    title="AgentForge AI Services",
    version=settings.app_version,
    description="Unified AI Gateway for AgentForge - Production-grade LLM orchestration",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware (order matters - first added = outermost)
app.middleware("http")(error_handling_middleware)
app.middleware("http")(metrics_middleware)
app.middleware("http")(logging_middleware)
if settings.rate_limit_enabled:
    app.middleware("http")(rate_limit_middleware)
if settings.auth_enabled:
    app.middleware("http")(auth_middleware)

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc) if settings.debug else "Internal server error"}
    )

# Include routers
app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
app.include_router(models.router, prefix=settings.api_prefix, tags=["models"])
app.include_router(chat.router, prefix=settings.api_prefix, tags=["chat"])
app.include_router(responses.router, prefix=settings.api_prefix, tags=["responses"])
app.include_router(embeddings.router, prefix=settings.api_prefix, tags=["embeddings"])
app.include_router(rag.router, prefix=settings.api_prefix, tags=["rag"])
app.include_router(memory.router, prefix=settings.api_prefix, tags=["memory"])
app.include_router(tools.router, prefix=settings.api_prefix, tags=["tools"])
app.include_router(speech.router, prefix=settings.api_prefix, tags=["speech"])
app.include_router(vision.router, prefix=settings.api_prefix, tags=["vision"])
app.include_router(images.router, prefix=settings.api_prefix, tags=["images"])
app.include_router(admin.router, prefix=f"{settings.api_prefix}/admin", tags=["admin"])


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )