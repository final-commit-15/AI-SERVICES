import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from services.llm_gateway.src.caching.cache import InMemoryCache, RedisCache


class TestInMemoryCache:
    @pytest.fixture
    def cache(self):
        return InMemoryCache(max_size=3, ttl=60)

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, cache):
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, cache):
        cache_with_short_ttl = InMemoryCache(max_size=10, ttl=0)  # Immediate expiration
        
        await cache_with_short_ttl.set("key1", "value1")
        await asyncio.sleep(0.1)
        result = await cache_with_short_ttl.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self, cache):
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        await cache.set("key4", "value4")  # Should evict key1
        
        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        await cache.set("key1", "value1")
        await cache.delete("key1")
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.clear()
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_custom_ttl(self, cache):
        await cache.set("key1", "value1", ttl=10)
        await cache.set("key2", "value2")  # Uses default TTL
        
        # Both should exist immediately
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"


class TestRedisCache:
    @pytest.fixture
    def cache(self):
        return RedisCache(url="redis://localhost:6379/0", max_size=100, ttl=60)

    @pytest.mark.asyncio
    async def test_connect_failure(self, cache):
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Connection refused")
            
            with pytest.raises(Exception):
                await cache.connect()

    @pytest.mark.asyncio
    async def test_operations_without_connection(self, cache):
        # Should not raise, just return None/do nothing
        result = await cache.get("key1")
        assert result is None
        
        await cache.set("key1", "value1")  # Should not raise
        await cache.delete("key1")  # Should not raise
        await cache.clear()  # Should not raise