"""
Enhanced PostgreSQL database models for secure authentication system.

This module provides a production-ready authentication system with:
- Multi-tenant user isolation
- Enhanced audit trails
- Advanced session management
- Comprehensive security logging
"""

import os
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, String, DateTime, Boolean, Integer, 
    ForeignKey, Text, Index, text, Enum as SQLEnum
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET, ARRAY
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()


class SecurityEventType(str, Enum):
    """Classification of security events for audit logging."""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    MAGIC_LINK_REQUESTED = "magic_link_requested"
    MAGIC_LINK_VERIFIED = "magic_link_verified"
    MAGIC_LINK_EXPIRED = "magic_link_expired"
    MAGIC_LINK_REUSED = "magic_link_reused"
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    SESSION_INVALIDATED = "session_invalidated"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    CONTEXT_VIOLATION = "context_violation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"


class RiskLevel(str, Enum):
    """Risk assessment levels for security events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class User(Base):
    """Enhanced user account model with tenant isolation."""
    __tablename__ = 'secure_users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    email = Column(String(320), unique=True, nullable=False, index=True)  # RFC 5321 max length
    tenant_id = Column(String(50), nullable=False, index=True, default="default")
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    failed_login_count = Column(Integer, default=0, nullable=False)
    account_locked_until = Column(DateTime(timezone=True), nullable=True)
    password_changed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP"))
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_activity_at = Column(DateTime(timezone=True), nullable=True)
    user_metadata = Column(JSONB, default=dict)
    
    # Security tracking
    trusted_device_ids = Column(ARRAY(String), default=list)
    known_ip_addresses = Column(ARRAY(INET), default=list)
    security_flags = Column(JSONB, default=dict)
    
    # Relationships
    magic_links = relationship("MagicLink", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    security_events = relationship("SecurityEvent", back_populates="user", cascade="all, delete-orphan")
    rate_limits = relationship("RateLimit", back_populates="user", cascade="all, delete-orphan")
    
    @hybrid_property
    def is_locked(self) -> bool:
        """Check if account is currently locked."""
        if not self.account_locked_until:
            return False
        return datetime.utcnow() < self.account_locked_until
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, tenant={self.tenant_id})>"


class MagicLink(Base):
    """Enhanced magic link model with additional security features."""
    __tablename__ = 'secure_magic_links'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey('secure_users.id'), nullable=False, index=True)
    email = Column(String(320), nullable=False, index=True)
    token_hash = Column(String(128), nullable=False, unique=True, index=True)
    salt = Column(String(64), nullable=False)  # Additional salt for token hashing
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    
    # Request context
    requesting_ip = Column(INET, nullable=True)
    requesting_user_agent = Column(Text, nullable=True)
    requesting_fingerprint = Column(String(128), nullable=True)
    
    # Verification context
    verified_ip = Column(INET, nullable=True)
    verified_user_agent = Column(Text, nullable=True)
    verified_fingerprint = Column(String(128), nullable=True)
    
    # Security metadata
    attempt_count = Column(Integer, default=0, nullable=False)
    max_attempts = Column(Integer, default=3, nullable=False)
    risk_score = Column(Integer, default=0, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="magic_links")
    
    @hybrid_property
    def is_expired(self) -> bool:
        """Check if magic link has expired."""
        return datetime.utcnow() > self.expires_at
    
    @hybrid_property
    def is_used(self) -> bool:
        """Check if magic link has been used."""
        return self.used_at is not None
    
    @hybrid_property
    def attempts_exceeded(self) -> bool:
        """Check if maximum attempts have been exceeded."""
        return self.attempt_count >= self.max_attempts


class UserSession(Base):
    """Enhanced session model with comprehensive context binding."""
    __tablename__ = 'secure_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey('secure_users.id'), nullable=False, index=True)
    token_hash = Column(String(128), nullable=False, unique=True, index=True)
    session_key = Column(String(128), nullable=False)  # Additional key for session binding
    
    # Device and context information
    device_fingerprint = Column(String(256), nullable=True)
    device_name = Column(String(100), nullable=True)
    device_id = Column(String(128), nullable=True, index=True)
    device_trust_score = Column(Integer, default=0, nullable=False)
    
    # Network context
    ip_address = Column(INET, nullable=True, index=True)
    ip_location = Column(JSONB, nullable=True)  # GeoIP data
    user_agent = Column(Text, nullable=True)
    user_agent_hash = Column(String(128), nullable=True)
    
    # Session lifecycle
    created_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    last_activity_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    last_refresh_at = Column(DateTime(timezone=True), nullable=True)
    invalidated_at = Column(DateTime(timezone=True), nullable=True)
    
    # Security state
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_remembered = Column(Boolean, default=False, nullable=False)
    requires_mfa = Column(Boolean, default=False, nullable=False)
    auth_method = Column(String(50), nullable=False, default='magic_link')
    
    # Context binding hashes for validation
    context_hash = Column(String(128), nullable=True)
    security_hash = Column(String(128), nullable=True)
    
    # Risk assessment
    risk_score = Column(Integer, default=0, nullable=False)
    anomaly_flags = Column(JSONB, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    @hybrid_property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() > self.expires_at
    
    @hybrid_property
    def is_valid(self) -> bool:
        """Check if session is valid and active."""
        return (
            self.is_active 
            and not self.is_expired 
            and self.invalidated_at is None
        )


class SecurityEvent(Base):
    """Comprehensive security event logging for audit trail."""
    __tablename__ = 'secure_security_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey('secure_users.id'), nullable=True, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('secure_sessions.id'), nullable=True, index=True)
    
    # Event classification
    event_type = Column(SQLEnum(SecurityEventType), nullable=False, index=True)
    risk_level = Column(SQLEnum(RiskLevel), nullable=False, default=RiskLevel.LOW, index=True)
    success = Column(Boolean, nullable=False, index=True)
    
    # Event details
    message = Column(Text, nullable=False)
    details = Column(JSONB, nullable=True)
    error_code = Column(String(50), nullable=True)
    
    # Context information
    ip_address = Column(INET, nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    device_fingerprint = Column(String(256), nullable=True)
    request_id = Column(String(128), nullable=True, index=True)
    correlation_id = Column(String(128), nullable=True, index=True)
    
    # Data access tracking
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(128), nullable=True)
    action_performed = Column(String(100), nullable=True)
    
    # Timing
    timestamp = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), nullable=False, index=True)
    duration_ms = Column(Integer, nullable=True)
    
    # Compliance and forensics
    tenant_id = Column(String(50), nullable=True, index=True)
    compliance_flags = Column(JSONB, default=dict)
    forensic_data = Column(JSONB, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="security_events")
    session = relationship("UserSession")
    
    def __repr__(self):
        return f"<SecurityEvent(type={self.event_type}, user_id={self.user_id}, success={self.success})>"


class RateLimit(Base):
    """Advanced rate limiting with adaptive thresholds."""
    __tablename__ = 'secure_rate_limits'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey('secure_users.id'), nullable=True, index=True)
    
    # Rate limit identifiers
    identifier_type = Column(String(20), nullable=False, index=True)  # 'email', 'ip', 'user_id'
    identifier_value = Column(String(320), nullable=False, index=True)
    action_type = Column(String(50), nullable=False, index=True)  # 'magic_link', 'login', 'api_call'
    
    # Time windows
    window_start = Column(DateTime(timezone=True), nullable=False, index=True)
    window_duration_seconds = Column(Integer, nullable=False)
    
    # Limits and counters
    attempt_count = Column(Integer, default=1, nullable=False)
    limit_threshold = Column(Integer, nullable=False)
    reset_count = Column(Integer, default=0, nullable=False)
    
    # Adaptive behavior
    base_threshold = Column(Integer, nullable=False)
    escalation_factor = Column(Integer, default=1, nullable=False)
    last_violation_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    user_metadata = Column(JSONB, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="rate_limits")
    
    @hybrid_property
    def is_expired(self) -> bool:
        """Check if rate limit window has expired."""
        return datetime.utcnow() > self.expires_at
    
    @hybrid_property
    def is_exceeded(self) -> bool:
        """Check if rate limit has been exceeded."""
        return self.attempt_count >= self.limit_threshold


class DeviceTrust(Base):
    """Device trust and recognition system."""
    __tablename__ = 'secure_device_trust'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey('secure_users.id'), nullable=False, index=True)
    
    # Device identification
    device_id = Column(String(128), nullable=False, unique=True, index=True)
    device_fingerprint = Column(String(256), nullable=False)
    device_name = Column(String(100), nullable=True)
    device_type = Column(String(50), nullable=True)  # 'desktop', 'mobile', 'tablet'
    
    # Trust metrics
    trust_score = Column(Integer, default=0, nullable=False)
    is_trusted = Column(Boolean, default=False, nullable=False)
    trust_established_at = Column(DateTime(timezone=True), nullable=True)
    
    # Usage tracking
    first_seen_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    last_seen_at = Column(DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP"))
    access_count = Column(Integer, default=1, nullable=False)
    
    # Security indicators
    known_ip_addresses = Column(ARRAY(INET), default=list)
    suspicious_activity_count = Column(Integer, default=0, nullable=False)
    last_anomaly_at = Column(DateTime(timezone=True), nullable=True)
    
    # Device metadata
    user_agent_history = Column(ARRAY(Text), default=list)
    location_history = Column(JSONB, default=list)
    capabilities = Column(JSONB, default=dict)
    
    # Relationships
    user = relationship("User")


# Create composite indexes for performance
Index('idx_user_email_tenant', User.email, User.tenant_id)
Index('idx_magiclink_user_active', MagicLink.user_id, MagicLink.expires_at)
Index('idx_session_user_active', UserSession.user_id, UserSession.is_active, UserSession.expires_at)
Index('idx_security_event_time_type', SecurityEvent.timestamp, SecurityEvent.event_type)
Index('idx_rate_limit_identifier_window', RateLimit.identifier_type, RateLimit.identifier_value, RateLimit.window_start)
Index('idx_device_trust_user_score', DeviceTrust.user_id, DeviceTrust.trust_score)


def get_database_url() -> str:
    """
    Construct PostgreSQL database URL from environment variables.
    
    Environment variables required:
    - AUTH_DB_HOST: Database host
    - AUTH_DB_PORT: Database port  
    - AUTH_DB_NAME: Database name
    - AUTH_DB_USER: Database username
    - AUTH_DB_PASSWORD: Database password
    
    Returns:
        PostgreSQL connection URL
        
    Raises:
        ValueError: If required environment variables are missing
    """
    host = os.environ.get('AUTH_DB_HOST')
    port = os.environ.get('AUTH_DB_PORT')
    database = os.environ.get('AUTH_DB_NAME')
    username = os.environ.get('AUTH_DB_USER')
    password = os.environ.get('AUTH_DB_PASSWORD')
    
    missing_vars = []
    if not host:
        missing_vars.append('AUTH_DB_HOST')
    if not port:
        missing_vars.append('AUTH_DB_PORT')
    if not database:
        missing_vars.append('AUTH_DB_NAME')
    if not username:
        missing_vars.append('AUTH_DB_USER')
    if not password:
        missing_vars.append('AUTH_DB_PASSWORD')
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return f"postgresql://{username}:{password}@{host}:{port}/{database}"


def create_auth_database(database_url: Optional[str] = None):
    """
    Create the authentication database with all tables and indexes.
    
    Args:
        database_url: PostgreSQL connection URL (defaults to constructing from env vars)
        
    Returns:
        SQLAlchemy engine instance
    """
    if not database_url:
        database_url = get_database_url()
    
    engine = create_engine(
        database_url,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    
    Base.metadata.create_all(engine)
    return engine


def get_database_session(database_url: Optional[str] = None):
    """
    Get a database session for the authentication database.
    
    Args:
        database_url: PostgreSQL connection URL (defaults to constructing from env vars)
        
    Returns:
        SQLAlchemy session instance
    """
    if not database_url:
        database_url = get_database_url()
        
    engine = create_engine(database_url, echo=False)
    Session = sessionmaker(bind=engine)
    return Session()


if __name__ == "__main__":
    """Create the authentication database when run directly."""
    import sys
    
    print("Creating secure authentication database...")
    print("Required environment variables:")
    print("  AUTH_DB_HOST")
    print("  AUTH_DB_PORT") 
    print("  AUTH_DB_NAME")
    print("  AUTH_DB_USER")
    print("  AUTH_DB_PASSWORD")
    print()
    
    try:
        database_url = get_database_url()
        masked_url = database_url.replace(os.environ.get('AUTH_DB_PASSWORD', ''), '***')
        print(f"Connecting to: {masked_url}")
        
        engine = create_auth_database(database_url)
        print("Secure authentication database created successfully!")
        
        # Show table information
        from sqlalchemy import inspect
        inspector = inspect(engine)
        
        print(f"\nCreated {len(inspector.get_table_names())} tables:")
        for table_name in sorted(inspector.get_table_names()):
            print(f"  - {table_name}")
            
        print(f"\nCreated {len(inspector.get_indexes('secure_users'))} composite indexes for optimized queries")
        
    except Exception as e:
        print(f"Error creating database: {e}", file=sys.stderr)
        sys.exit(1)