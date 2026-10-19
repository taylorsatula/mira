"""The record types every layer of `auth` shares: DB rows, the Valkey session payload, and the two credential contexts.

`SessionData` and `APITokenContext` are the type-level distinction the credential
ladder returns — a browser/header session versus an issued API token — and
browser-session-only surfaces branch on `isinstance` to refuse an API-token
holder. Collapsing them into one type removes that refusal.

`UserRecord` mirrors `AuthDatabase._USER_RECORD_COLUMNS` field for field, and the
two change together. Pydantic ignores extra keys, so a column with no matching
model field disappears without raising; a model field with no matching column
raises when the row is constructed. Either drift is silent one way and loud the
other, never symmetric.
"""

from typing import Optional, Literal, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


SubjectKind = Literal["member", "demo"]


class UserRecord(BaseModel):
    """Database user record structure."""
    id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool
    created_at: datetime
    last_login_at: Optional[datetime] = None
    webauthn_credentials: Dict[str, Any] = Field(default_factory=dict)
    memory_manipulation_enabled: bool
    daily_manipulation_last_run: Optional[datetime] = None
    timezone: str
    subject_kind: SubjectKind
    demo_expires_at: Optional[datetime] = None


class UserProfile(BaseModel):
    """User profile data returned to client."""
    id: str
    email: str
    is_active: bool
    created_at: datetime
    subject_kind: SubjectKind
    demo_expires_at: Optional[datetime] = None


class SessionData(BaseModel):
    """Session data stored in Valkey."""
    user_id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    timezone: str
    subject_kind: SubjectKind
    demo_expires_at: Optional[str] = None
    created_at: str  # ISO format for JSON serialization
    last_activity: str  # ISO format for JSON serialization
    max_expiry: str  # ISO format for JSON serialization


class MagicLinkRecord(BaseModel):
    """Database magic link record structure."""
    id: str
    user_id: str
    email: str
    token_hash: str
    expires_at: datetime
    used_at: Optional[datetime] = None
    created_at: datetime


class CookieSettings(BaseModel):
    """Secure cookie configuration."""
    samesite: Literal["strict", "lax", "none"]
    httponly: bool
    secure: bool
    max_age: int


class APITokenContext(BaseModel):
    """Context for API token-authenticated requests (not session-based)."""
    user_id: str
    token_type: str
    token_id: str
    subject_kind: SubjectKind
    demo_expires_at: Optional[str] = None
