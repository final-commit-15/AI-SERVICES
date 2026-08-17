import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from fastapi import FastAPI
from .api.routes import router
from .config.settings import settings
from .dependencies import init_rag
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AgentForge LLM Gateway",
    version="0.2.0",
    description="Unified gateway for AI capabilities",
)

app.include_router(router, prefix="/v1")

@app.on_event("startup")
async def startup():
    logger.info("Starting LLM Gateway...")
    init_rag()
    logger.info("RAG initialized.")

@app.get("/health")
async def health():
    return {"status": "ok"}