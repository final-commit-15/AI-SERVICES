import structlog
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

import redis.asyncio as redis
from jose import jwt, JWTError
from passlib.context import CryptContext

logger = structlog.get_logger()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@dataclass
class User:
    id: str
    username: str
    email: str
    full_name: Optional[str]
    hashed_password: str
    roles: List[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]


@dataclass
class APIKeyData:
    id: str
    name: str
    key_hash: str
    prefix: str
    user_id: str
    roles: List[str]
    scopes: List[str]
    rate_limit: Optional[int]
    expires_at: Optional[datetime]
    created_at: datetime
    last_used_at: Optional[datetime]
    is_active: bool


class AuthService:
    """Authentication and authorization service."""

    def __init__(
        self,
        jwt_secret: str,
        jwt_algorithm: str = "HS256",
        access_token_expire: int = 30,
        refresh_token_expire: int = 7,
    ):
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.access_token_expire = access_token_expire  # minutes
        self.refresh_token_expire = refresh_token_expire  # days
        self._redis: Optional[redis.Redis] = None

    async def connect(self, redis_url: str = "redis://localhost:6379/0"):
        """Connect to Redis."""
        self._redis = redis.from_url(redis_url, decode_responses=True)
        await self._redis.ping()
        logger.info("auth_service_connected")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)

    def create_access_token(self, user_id: str, roles: List[str], scopes: List[str] = None) -> str:
        """Create JWT access token."""
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.access_token_expire)
        payload = {
            "sub": user_id,
            "user_id": user_id,
            "roles": roles,
            "scopes": scopes or [],
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(uuid.uuid4()),
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token."""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.refresh_token_expire)
        payload = {
            "sub": user_id,
            "user_id": user_id,
            "type": "refresh",
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(uuid.uuid4()),
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except JWTError as e:
            logger.warning("jwt_decode_failed", error=str(e))
            return None

    async def create_api_key(
        self,
        name: str,
        user_id: str,
        roles: List[str] = None,
        scopes: List[str] = None,
        rate_limit: Optional[int] = None,
        expires_in_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a new API key."""
        import secrets
        key = f"afk_{secrets.token_urlsafe(32)}"
        key_hash = self.get_password_hash(key)
        prefix = key[:8]

        api_key = APIKeyData(
            id=str(uuid.uuid4()),
            name=name,
            key_hash=key_hash,
            prefix=prefix,
            user_id=user_id,
            roles=roles or ["user"],
            scopes=scopes or [],
            rate_limit=rate_limit,
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days) if expires_in_days else None,
            created_at=datetime.utcnow(),
            last_used_at=None,
            is_active=True,
        )

        # Store in Redis
        key_data = {
            "id": api_key.id,
            "name": api_key.name,
            "key_hash": api_key.key_hash,
            "prefix": api_key.prefix,
            "user_id": api_key.user_id,
            "roles": api_key.roles,
            "scopes": api_key.scopes,
            "rate_limit": api_key.rate_limit,
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "created_at": api_key.created_at.isoformat(),
            "last_used_at": None,
            "is_active": True,
        }
        await self._redis.hset(f"apikey:{api_key.id}", mapping=key_data)
        await self._redis.sadd(f"user_apikeys:{user_id}", api_key.id)

        return {
            "id": api_key.id,
            "name": api_key.name,
            "prefix": api_key.prefix,
            "key": key,  # Only returned once!
            "roles": api_key.roles,
            "scopes": api_key.scopes,
            "rate_limit": api_key.rate_limit,
            "expires_at": api_key.expires_at,
            "created_at": api_key.created_at,
        }

    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key."""
        # Extract prefix and lookup
        prefix = api_key[:8]
        # In production, would use prefix to find candidate keys
        # For now, scan (not efficient for production)
        keys = await self._redis.keys("apikey:*")
        for key_id in keys:
            data = await self._redis.hgetall(key_id)
            if data.get("is_active") == "True":
                if self.verify_password(api_key, data["key_hash"]):
                    # Update last used
                    await self._redis.hset(key_id, "last_used_at", datetime.utcnow().isoformat())
                    return {
                        "id": data["id"],
                        "user_id": data["user_id"],
                        "roles": data["roles"].split(",") if isinstance(data["roles"], str) else data["roles"],
                        "scopes": data["scopes"].split(",") if isinstance(data["scopes"], str) else data["scopes"],
                        "rate_limit": int(data["rate_limit"]) if data.get("rate_limit") else None,
                    }
        return None

    async def list_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """List API keys for a user."""
        key_ids = await self._redis.smembers(f"user_apikeys:{user_id}")
        keys = []
        for key_id in key_ids:
            data = await self._redis.hgetall(key_id)
            if data:
                keys.append({
                    "id": data["id"],
                    "name": data["name"],
                    "prefix": data["prefix"],
                    "roles": data["roles"],
                    "scopes": data["scopes"],
                    "rate_limit": data.get("rate_limit"),
                    "expires_at": data.get("expires_at"),
                    "created_at": data["created_at"],
                    "last_used_at": data.get("last_used_at"),
                    "is_active": data["is_active"] == "True",
                })
        return keys

    async def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key."""
        # Verify ownership
        data = await self._redis.hgetall(f"apikey:{key_id}")
        if not data or data.get("user_id") != user_id:
            return False

        await self._redis.hset(f"apikey:{key_id}", "is_active", "False")
        await self._redis.srem(f"user_apikeys:{user_id}", key_id)
        return True

    async def close(self):
        """Close connections."""
        if self._redis:
            await self._redis.close()