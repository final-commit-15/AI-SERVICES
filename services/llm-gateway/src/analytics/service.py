import structlog
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

logger = structlog.get_logger()


@dataclass
class UsageRecord:
    id: str
    timestamp: datetime
    user_id: Optional[str]
    conversation_id: Optional[str]
    provider: str
    model: str
    task_type: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    status: str
    error: Optional[str]
    metadata: Dict[str, Any]


class AnalyticsService:
    """Usage analytics and cost tracking service."""

    def __init__(
        self,
        database_url: str = "postgresql+asyncpg://localhost/agentforge",
        retention_days: int = 90,
    ):
        self.database_url = database_url
        self.retention_days = retention_days
        self._pool = None
        self._in_memory_buffer: List[UsageRecord] = []
        self._buffer_size = 100
        self._flush_interval = 30  # seconds

    async def connect(self):
        """Connect to database."""
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(self.database_url, min_size=2, max_size=10)
            await self._init_tables()
            logger.info("analytics_service_connected")
        except Exception as e:
            logger.warning("analytics_database_connection_failed", error=str(e))
            # Continue with in-memory only

    async def _init_tables(self):
        """Initialize database tables."""
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_records (
                    id UUID PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    user_id TEXT,
                    conversation_id TEXT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    cost_usd REAL NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    metadata JSONB DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_records(timestamp);
                CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_records(user_id);
                CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage_records(provider);
                CREATE INDEX IF NOT EXISTS idx_usage_model ON usage_records(model);
            """)

    async def record_usage(self, record: UsageRecord):
        """Record a usage event."""
        self._in_memory_buffer.append(record)

        if len(self._in_memory_buffer) >= self._buffer_size:
            await self._flush_buffer()

    async def _flush_buffer(self):
        """Flush buffer to database."""
        if not self._in_memory_buffer:
            return

        records = self._in_memory_buffer[:]
        self._in_memory_buffer.clear()

        if not self._pool:
            return

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO usage_records (
                        id, timestamp, user_id, conversation_id, provider, model,
                        task_type, prompt_tokens, completion_tokens, total_tokens,
                        latency_ms, cost_usd, status, error, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """, [
                    (
                        r.id, r.timestamp, r.user_id, r.conversation_id, r.provider, r.model,
                        r.task_type, r.prompt_tokens, r.completion_tokens, r.total_tokens,
                        r.latency_ms, r.cost_usd, r.status, r.error, r.metadata
                    )
                    for r in records
                ])
        except Exception as e:
            logger.error("analytics_flush_failed", error=str(e))
            # Put back in buffer
            self._in_memory_buffer = records + self._in_memory_buffer

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
        group_by: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Query analytics data."""
        # For now, return in-memory aggregation
        # In production, query database
        now = datetime.utcnow()
        start = start_time or (now - timedelta(days=7))
        end = end_time or now

        filtered = [
            r for r in self._in_memory_buffer
            if start <= r.timestamp <= end
            and (not user_id or r.user_id == user_id)
            and (not provider or r.provider == provider)
            and (not model or r.model == model)
            and (not task_type or r.task_type == task_type)
        ]

        return self._aggregate(filtered)

    def _aggregate(self, records: List[UsageRecord]) -> Dict[str, Any]:
        """Aggregate usage records."""
        if not records:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 0.0,
                "error_rate": 0.0,
                "by_provider": {},
                "by_model": {},
                "by_task_type": {},
                "time_series": [],
            }

        total_requests = len(records)
        total_tokens = sum(r.total_tokens for r in records)
        total_cost = sum(r.cost_usd for r in records)
        avg_latency = sum(r.latency_ms for r in records) / total_requests
        errors = sum(1 for r in records if r.status == "error")
        error_rate = errors / total_requests

        # Group by provider
        by_provider = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost_usd": 0.0, "latency_ms": 0.0, "errors": 0})
        for r in records:
            p = by_provider[r.provider]
            p["requests"] += 1
            p["tokens"] += r.total_tokens
            p["cost_usd"] += r.cost_usd
            p["latency_ms"] += r.latency_ms
            if r.status == "error":
                p["errors"] += 1

        for p in by_provider.values():
            p["avg_latency_ms"] = p["latency_ms"] / p["requests"]
            p["error_rate"] = p["errors"] / p["requests"]
            del p["latency_ms"], p["errors"]

        # Group by model
        by_model = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost_usd": 0.0, "latency_ms": 0.0})
        for r in records:
            m = by_model[r.model]
            m["requests"] += 1
            m["tokens"] += r.total_tokens
            m["cost_usd"] += r.cost_usd
            m["latency_ms"] += r.latency_ms

        for m in by_model.values():
            m["avg_latency_ms"] = m["latency_ms"] / m["requests"]
            del m["latency_ms"]

        # Group by task type
        by_task = defaultdict(lambda: {"requests": 0, "tokens": 0, "latency_ms": 0.0})
        for r in records:
            t = by_task[r.task_type]
            t["requests"] += 1
            t["tokens"] += r.total_tokens
            t["latency_ms"] += r.latency_ms

        for t in by_task.values():
            t["avg_latency_ms"] = t["latency_ms"] / t["requests"]
            del t["latency_ms"]

        # Time series (hourly buckets)
        time_series = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost_usd": 0.0, "latency_ms": 0.0})
        for r in records:
            bucket = r.timestamp.replace(minute=0, second=0, microsecond=0)
            ts = time_series[bucket]
            ts["requests"] += 1
            ts["tokens"] += r.total_tokens
            ts["cost_usd"] += r.cost_usd
            ts["latency_ms"] += r.latency_ms

        time_series_list = [
            {
                "timestamp": ts.isoformat(),
                "requests": d["requests"],
                "tokens": d["tokens"],
                "cost_usd": d["cost_usd"],
                "latency_ms": d["latency_ms"] / d["requests"] if d["requests"] > 0 else 0,
            }
            for ts, d in sorted(time_series.items())
        ]

        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "avg_latency_ms": avg_latency,
            "error_rate": error_rate,
            "by_provider": dict(by_provider),
            "by_model": dict(by_model),
            "by_task_type": dict(by_task),
            "time_series": time_series_list,
        }

    async def get_cost_breakdown(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get detailed cost breakdown."""
        records = await self.query(start_time, end_time)
        return records

    async def close(self):
        """Close connections."""
        await self._flush_buffer()
        if self._pool:
            await self._pool.close()