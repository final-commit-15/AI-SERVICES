import structlog
from abc import ABC, abstractmethod
from typing import Any, Optional
import time
from collections import OrderedDict

logger = structlog.get_logger()


class Cache(ABC):
    """Abstract cache interface."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        pass

    @abstractmethod
    async def delete(self, key: str):
        pass

    @abstractmethod
    async def clear(self):
        pass

    @abstractmethod
    async def close(self):
        pass


class InMemoryCache(Cache):
    """In-memory LRU cache with TTL."""

    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.default_ttl = ttl

    async def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp, ttl = self.cache[key]
            if time.time() - timestamp < ttl:
                self.cache.move_to_end(key)
                return value
            else:
                del self.cache[key]
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = (value, time.time(), ttl or self.default_ttl)
        self.cache.move_to_end(key)

    async def delete(self, key: str):
        if key in self.cache:
            del self.cache[key]

    async def clear(self):
        self.cache.clear()

    async def close(self):
        self.cache.clear()


class RedisCache(Cache):
    """Redis-backed cache."""

    def __init__(self, url: str, max_size: int = 10000, ttl: int = 3600):
        self.url = url
        self.max_size = max_size
        self.default_ttl = ttl
        self._client = None

    async def connect(self):
        import redis.asyncio as redis
        self._client = redis.from_url(self.url, decode_responses=True)
        await self._client.ping()

    async def get(self, key: str) -> Optional[Any]:
        if not self._client:
            return None
        try:
            data = await self._client.get(key)
            if data:
                import json
                return json.loads(data)
        except Exception as e:
            logger.warning("redis_get_failed", key=key, error=str(e))
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if not self._client:
            return
        try:
            import json
            await self._client.setex(key, ttl or self.default_ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.warning("redis_set_failed", key=key, error=str(e))

    async def delete(self, key: str):
        if not self._client:
            return
        try:
            await self._client.delete(key)
        except Exception as e:
            logger.warning("redis_delete_failed", key=key, error=str(e))

    async def clear(self):
        if not self._client:
            return
        try:
            await self._client.flushdb()
        except Exception as e:
            logger.warning("redis_clear_failed", error=str(e))

    async def close(self):
        if self._client:
            await self._client.close()


def get_cache() -> Optional[Cache]:
    """Get cache instance from dependencies."""
    from .dependencies import get_cache as _get_cache
    return _get_cache()