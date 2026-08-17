from typing import Any, Optional
import time
from collections import OrderedDict
from ..config.settings import settings


class InMemoryCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl  # seconds

    async def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                # Move to end (LRU)
                self.cache.move_to_end(key)
                return value
            else:
                del self.cache[key]
        return None

    async def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # remove oldest
        self.cache[key] = (value, time.time())
        self.cache.move_to_end(key)


def get_cache():
    if settings.cache_enabled:
        return InMemoryCache(max_size=settings.cache_max_size, ttl=settings.cache_ttl_seconds)
    return None