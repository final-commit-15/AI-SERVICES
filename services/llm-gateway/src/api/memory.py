import structlog
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from datetime import datetime

from libs.schemas_common.memory import (
    MemoryType,
    MemoryEntry,
    ConversationMemory,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryStats,
)
from libs.schemas_common.chat import ChatMessage
from ..memory.manager import MemoryManager
from ..dependencies import get_memory_manager

logger = structlog.get_logger()

router = APIRouter()


class CreateConversationRequest(BaseModel):
    user_id: Optional[str] = None
    metadata: Optional[dict] = None


class AddMessageRequest(BaseModel):
    role: str
    content: str
    metadata: Optional[dict] = None


@router.post("/memory/conversations", response_model=ConversationMemory)
async def create_conversation(
    request: CreateConversationRequest,
    http_request: Request,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """Create a new conversation."""
    try:
        conversation = await memory_manager.create_conversation(
            user_id=request.user_id,
            metadata=request.metadata,
        )
        return conversation
    except Exception as e:
        logger.error("create_conversation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/conversations/{conversation_id}", response_model=ConversationMemory)
async def get_conversation(
    conversation_id: str,
    http_request: Request,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """Get conversation by ID."""
    conversation = await memory_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.post("/memory/conversations/{conversation_id}/messages")
async def add_message(
    conversation_id: str,
    request: AddMessageRequest,
    http_request: Request,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """Add a message to conversation."""
    try:
        message = await memory_manager.add_message(
            conversation_id=conversation_id,
            role=request.role,
            content=request.content,
            metadata=request.metadata,
        )
        return {"message": message}
    except Exception as e:
        logger.error("add_message_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/conversations/{conversation_id}/messages")
async def get_messages(
    conversation_id: str,
    limit: int = 50,
    offset: int = 0,
    http_request: Request = None,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """Get messages from conversation."""
    messages = await memory_manager.get_messages(conversation_id, limit, offset)
    return {"messages": messages, "total": len(messages)}


@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    request: MemorySearchRequest,
    http_request: Request,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """Search conversation memory."""
    try:
        results = await memory_manager.search(request)
        return results
    except Exception as e:
        logger.error("search_memory_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/stats", response_model=MemoryStats)
async def get_memory_stats(
    http_request: Request,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """Get memory statistics."""
    try:
        stats = await memory_manager.get_stats()
        return stats
    except Exception as e:
        logger.error("get_memory_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    http_request: Request,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """Delete a conversation."""
    try:
        await memory_manager.delete_conversation(conversation_id)
        return {"status": "success"}
    except Exception as e:
        logger.error("delete_conversation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/conversations/{conversation_id}/summarize")
async def summarize_conversation(
    conversation_id: str,
    http_request: Request,
    memory_manager: MemoryManager = Depends(get_memory_manager),
):
    """Generate summary of conversation."""
    try:
        summary = await memory_manager.summarize_conversation(conversation_id)
        return {"summary": summary}
    except Exception as e:
        logger.error("summarize_conversation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))