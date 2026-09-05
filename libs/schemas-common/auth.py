from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"


class TokenData(BaseModel):
    sub: str
    user_id: str
    roles: List[UserRole] = [UserRole.USER]
    scopes: List[str] = []
    exp: int
    iat: int
    jti: str


class APIKey(BaseModel):
    id: str
    name: str
    key_hash: str
    prefix: str
    user_id: str
    roles: List[UserRole] = [UserRole.USER]
    scopes: List[str] = []
    rate_limit: Optional[int] = None
    expires_at: Optional[datetime] = None
    created_at: datetime
    last_used_at: Optional[datetime] = None
    is_active: bool = True


class APIKeyCreate(BaseModel):
    name: str
    roles: List[UserRole] = [UserRole.USER]
    scopes: List[str] = []
    rate_limit: Optional[int] = None
    expires_in_days: Optional[int] = None


class APIKeyResponse(BaseModel):
    id: str
    name: str
    prefix: str
    key: str
    roles: List[UserRole]
    scopes: List[str]
    rate_limit: Optional[int]
    expires_at: Optional[datetime]
    created_at: datetime


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class User(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    roles: List[UserRole] = [UserRole.USER]
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None