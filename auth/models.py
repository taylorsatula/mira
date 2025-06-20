"""
Database models for authentication.
"""

from utils.timezone_utils import utc_now
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    tenant_id = Column(String, nullable=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utc_now)
    last_login_at = Column(DateTime, nullable=True)
    
    # WebAuthn credentials stored as JSON
    webauthn_credentials = Column(JSON, nullable=True, default=dict)
    
    # Memory consolidation settings
    memory_consolidation_enabled = Column(Boolean, default=True, nullable=False)
    daily_consolidation_last_run = Column(DateTime, nullable=True)
    weekly_consolidation_last_run = Column(DateTime, nullable=True)
    monthly_consolidation_last_run = Column(DateTime, nullable=True)
    learning_reflection_last_run = Column(DateTime, nullable=True)
    last_activity_check = Column(DateTime, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "email": self.email,
            "tenant_id": self.tenant_id,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "has_webauthn": bool(self.webauthn_credentials),
            "memory_consolidation_enabled": self.memory_consolidation_enabled,
            "daily_consolidation_last_run": self.daily_consolidation_last_run.isoformat() if self.daily_consolidation_last_run else None,
            "weekly_consolidation_last_run": self.weekly_consolidation_last_run.isoformat() if self.weekly_consolidation_last_run else None,
            "monthly_consolidation_last_run": self.monthly_consolidation_last_run.isoformat() if self.monthly_consolidation_last_run else None,
            "learning_reflection_last_run": self.learning_reflection_last_run.isoformat() if self.learning_reflection_last_run else None
        }

class MagicLink(Base):
    """Magic link tokens."""
    __tablename__ = "magic_links"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    email = Column(String, nullable=False)
    token_hash = Column(String, nullable=False, unique=True, index=True)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=utc_now)
    
    def is_expired(self) -> bool:
        """Check if link is expired."""
        return utc_now() > self.expires_at
    
    def is_used(self) -> bool:
        """Check if link is already used."""
        return self.used_at is not None

class Session(Base):
    """User sessions."""
    __tablename__ = "sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    token_hash = Column(String, nullable=False, unique=True, index=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, nullable=False, default=utc_now)
    last_activity_at = Column(DateTime, nullable=False, default=utc_now)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return utc_now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "expires_at": self.expires_at.isoformat(),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat()
        }