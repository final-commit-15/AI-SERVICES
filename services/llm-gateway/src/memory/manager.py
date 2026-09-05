import structlog
import uuid
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

import redis.asyncio as redis
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

logger = structlog.get_logger()


@dataclass
class MemoryEntry:
    id: str
    conversation_id: str
    user_id: Optional[str]
    role: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]
    created_at: datetime
    expires_at: Optional[datetime]
    importance: float


class MemoryManager:
    """Conversation memory manager with Redis (short-term) and Qdrant (long-term/semantic)."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        short_term_ttl: int = 3600,
        long_term_ttl: int = 2592000,
        max_messages: int = 50,
        embedding_dim: int = 768,
    ):
        self.redis_url = redis_url
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.short_term_ttl = short_term_ttl
        self.long_term_ttl = long_term_ttl
        self.max_messages = max_messages
        self.embedding_dim = embedding_dim
        self._redis: Optional[redis.Redis] = None
        self._qdrant: Optional[AsyncQdrantClient] = None
        self._embedding_service = None

    async def connect(self):
        """Connect to Redis and Qdrant."""
        self._redis = redis.from_url(self.redis_url, decode_responses=True)
        await self._redis.ping()

        self._qdrant = AsyncQdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )
        await self._qdrant.get_collections()

        # Ensure collections exist
        await self._ensure_collections()

        logger.info("memory_manager_connected")

    async def _ensure_collections(self):
        """Ensure Qdrant collections exist."""
        collections = await self._qdrant.get_collections()
        existing = [c.name for c in collections.collections]

        for name in ["conversations", "memories"]:
            if name not in existing:
                await self._qdrant.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("memory_collection_created", collection=name)

    async def close(self):
        """Close connections."""
        if self._redis:
            await self._redis.close()
        if self._qdrant:
            await self._qdrant.close()

    async def create_conversation(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new conversation."""
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow()

        conversation = {
            "id": conversation_id,
            "user_id": user_id,
            "messages": [],
            "summary": None,
            "metadata": metadata or {},
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "message_count": 0,
            "token_count": 0,
        }

        # Store in Redis
        key = f"conversation:{conversation_id}"
        await self._redis.setex(
            key,
            self.short_term_ttl,
            json.dumps(conversation, default=str),
        )

        # Store in Qdrant for long-term
        point = PointStruct(
            id=conversation_id,
            vector=[0.0] * self.embedding_dim,  # Placeholder
            payload={
                "type": "conversation",
                "user_id": user_id,
                "metadata": metadata or {},
                "created_at": now.isoformat(),
            },
        )
        await self._qdrant.upsert(collection_name="conversations", points=[point])

        return conversation

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID."""
        key = f"conversation:{conversation_id}"
        data = await self._redis.get(key)
        if data:
            return json.loads(data)

        # Try Qdrant
        points = await self._qdrant.retrieve(
            collection_name="conversations",
            ids=[conversation_id],
        )
        if points:
            payload = points[0].payload
            return {
                "id": conversation_id,
                "user_id": payload.get("user_id"),
                "messages": [],
                "summary": None,
                "metadata": payload.get("metadata", {}),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("created_at"),
                "message_count": 0,
                "token_count": 0,
            }

        return None

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a message to conversation."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")

        message_id = str(uuid.uuid4())
        now = datetime.utcnow()

        message = {
            "id": message_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": now.isoformat(),
        }

        # Update conversation
        conversation["messages"].append(message)
        conversation["message_count"] += 1
        conversation["updated_at"] = now.isoformat()

        # Trim if too many messages
        if len(conversation["messages"]) > self.max_messages:
            conversation["messages"] = conversation["messages"][-self.max_messages:]

        # Store updated conversation in Redis
        key = f"conversation:{conversation_id}"
        await self._redis.setex(
            key,
            self.short_term_ttl,
            json.dumps(conversation, default=str),
        )

        # Store message in Qdrant for semantic search
        if self._embedding_service:
            embedding = await self._embedding_service.embed_query(content)
            point = PointStruct(
                id=message_id,
                vector=embedding,
                payload={
                    "type": "message",
                    "conversation_id": conversation_id,
                    "user_id": conversation.get("user_id"),
                    "role": role,
                    "content": content,
                    "metadata": metadata or {},
                    "created_at": now.isoformat(),
                },
            )
            await self._qdrant.upsert(collection_name="memories", points=[point])

        return message

    async def get_messages(
        self,
        conversation_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get messages from conversation."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return []

        messages = conversation.get("messages", [])
        return messages[offset:offset + limit]

    async def search(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search memory semantically."""
        if not self._embedding_service:
            return []

        query_embedding = await self._embedding_service.embed_query(query)

        # Build filter
        conditions = []
        if conversation_id:
            conditions.append(FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id)))
        if user_id:
            conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))

        qdrant_filter = Filter(must=conditions) if conditions else None

        results = await self._qdrant.search(
            collection_name="memories",
            query_vector=query_embedding,
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True,
            score_threshold=min_score,
        )

        return [
            {
                "id": hit.id,
                "conversation_id": hit.payload.get("conversation_id"),
                "user_id": hit.payload.get("user_id"),
                "role": hit.payload.get("role"),
                "content": hit.payload.get("content"),
                "metadata": hit.payload.get("metadata", {}),
                "created_at": hit.payload.get("created_at"),
                "score": hit.score,
            }
            for hit in results
        ]

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        # Count Redis keys
        conv_keys = await self._redis.keys("conversation:*")
        total_conversations = len(conv_keys)

        # Count Qdrant points
        collections = await self._qdrant.get_collections()
        total_messages = 0
        for coll in collections.collections:
            if coll.name in ["conversations", "memories"]:
                info = await self._qdrant.get_collection(coll.name)
                total_messages += info.points_count

        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "redis_memory_mb": 0,  # Would need Redis INFO
            "qdrant_collections": [c.name for c in collections.collections],
        }

    async def summarize_conversation(self, conversation_id: str) -> str:
        """Generate summary of conversation."""
        conversation = await self.get_conversation(conversation_id)
        if not conversation:
            return ""

        messages = conversation.get("messages", [])
        if not messages:
            return ""

        # Build summary prompt
        text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt = f"Summarize this conversation in 2-3 sentences:\n\n{text}"

        # In production, use an LLM to generate summary
        # For now, return a simple summary
        summary = f"Conversation with {len(messages)} messages. Last topic: {messages[-1]['content'][:100]}"

        # Update conversation with summary
        conversation["summary"] = summary
        key = f"conversation:{conversation_id}"
        await self._redis.setex(key, self.short_term_ttl, json.dumps(conversation, default=str))

        return summary

    async def delete_conversation(self, conversation_id: str):
        """Delete a conversation."""
        key = f"conversation:{conversation_id}"
        await self._redis.delete(key)

        # Delete from Qdrant
        await self._qdrant.delete(
            collection_name="conversations",
            points_selector=[conversation_id],
        )
        # Delete associated messages
        await self._qdrant.delete(
            collection_name="memories",
            filter=Filter(
                must=[FieldCondition(key="conversation_id", match=MatchValue(value=conversation_id))]
            ),
        )

    def set_embedding_service(self, embedding_service):
        """Set embedding service for semantic search."""
        self._embedding_service = embedding_service