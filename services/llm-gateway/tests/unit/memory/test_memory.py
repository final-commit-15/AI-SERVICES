import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from services.llm_gateway.src.memory.manager import MemoryManager, MemoryEntry


class TestMemoryManager:
    @pytest.fixture
    def mock_redis(self):
        redis = AsyncMock()
        redis.ping = AsyncMock(return_value=True)
        redis.setex = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.delete = AsyncMock()
        redis.keys = AsyncMock(return_value=[])
        redis.sadd = AsyncMock()
        redis.smembers = AsyncMock(return_value=set())
        redis.srem = AsyncMock()
        redis.hset = AsyncMock()
        redis.hgetall = AsyncMock(return_value={})
        return redis

    @pytest.fixture
    def mock_qdrant(self):
        qdrant = AsyncMock()
        qdrant.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
        qdrant.create_collection = AsyncMock()
        qdrant.upsert = AsyncMock()
        qdrant.retrieve = AsyncMock(return_value=[])
        qdrant.search = AsyncMock(return_value=[])
        qdrant.delete = AsyncMock()
        qdrant.get_collection = AsyncMock(return_value=MagicMock(points_count=0))
        qdrant.close = AsyncMock()
        return qdrant

    @pytest.fixture
    async def memory_manager(self, mock_redis, mock_qdrant):
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            with patch("qdrant_client.AsyncQdrantClient", return_value=mock_qdrant):
                manager = MemoryManager(
                    redis_url="redis://localhost:6379/0",
                    qdrant_url="http://localhost:6333",
                    embedding_dim=768,
                )
                await manager.connect()
                return manager

    @pytest.mark.asyncio
    async def test_create_conversation(self, memory_manager):
        conv = await memory_manager.create_conversation(
            user_id="user-123",
            metadata={"topic": "test"}
        )
        
        assert conv["id"] is not None
        assert conv["user_id"] == "user-123"
        assert conv["metadata"]["topic"] == "test"
        assert conv["message_count"] == 0

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self, memory_manager):
        conv = await memory_manager.get_conversation("nonexistent")
        assert conv is None

    @pytest.mark.asyncio
    async def test_add_message(self, memory_manager):
        # Create conversation first
        conv = await memory_manager.create_conversation(user_id="user-123")
        conv_id = conv["id"]
        
        # Mock Redis get to return conversation
        memory_manager._redis.get = AsyncMock(return_value='{"id": "' + conv_id + '", "user_id": "user-123", "messages": [], "message_count": 0, "metadata": {}, "created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-01T00:00:00"}')
        
        message = await memory_manager.add_message(
            conversation_id=conv_id,
            role="user",
            content="Hello!"
        )
        
        assert message["id"] is not None
        assert message["role"] == "user"
        assert message["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_get_messages(self, memory_manager):
        conv_id = "test-conv"
        memory_manager._redis.get = AsyncMock(return_value='{"id": "' + conv_id + '", "user_id": "user-123", "messages": [{"id": "msg-1", "role": "user", "content": "Hi"}], "message_count": 1, "metadata": {}, "created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-01T00:00:00"}')
        
        messages = await memory_manager.get_messages(conv_id, limit=10)
        
        assert len(messages) == 1
        assert messages[0]["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_delete_conversation(self, memory_manager):
        await memory_manager.delete_conversation("conv-123")
        
        memory_manager._redis.delete.assert_called_once()
        memory_manager._qdrant.delete.assert_called()

    @pytest.mark.asyncio
    async def test_get_stats(self, memory_manager):
        memory_manager._redis.keys = AsyncMock(return_value=["conv:1", "conv:2"])
        memory_manager._qdrant.get_collection = AsyncMock(return_value=MagicMock(points_count=100))
        
        stats = await memory_manager.get_stats()
        
        assert stats["total_conversations"] == 2
        assert stats["total_messages"] == 100