from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"


class MemoryEntry(BaseModel):
    id: str
    conversation_id: str
    user_id: Optional[str] = None
    type: MemoryType
    role: str
    content: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
    created_at: datetime
    expires_at: Optional[datetime] = None
    importance: float = 1.0


class ConversationMemory(BaseModel):
    conversation_id: str
    user_id: Optional[str] = None
    messages: List["ChatMessage"] = []
    summary: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    token_count: int = 0


class MemorySearchRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    memory_types: List[MemoryType] = [MemoryType.SHORT_TERM, MemoryType.LONG_TERM]
    limit: int = 10
    min_score: float = 0.7


class MemorySearchResponse(BaseModel):
    results: List[MemoryEntry]
    total: int


class MemoryStats(BaseModel):
    total_conversations: int
    total_messages: int
    short_term_entries: int
    long_term_entries: int
    semantic_entries: int
    storage_size_mb: float


from .chat import ChatMessage