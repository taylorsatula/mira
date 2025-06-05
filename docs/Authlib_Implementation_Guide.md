# Authlib Implementation Guide for MIRA Authentication System

## Overview

This guide implements a production-grade authentication system using Authlib as the cryptographic foundation, while incorporating the advanced security concepts from your existing `secure_auth` system. We'll build this in `/auth/` and reference `secure_auth/` for security patterns.

## Prerequisites

- Existing PostgreSQL setup with mira_admin/mira_app users
- Your Universal Testing Guide methodology
- FastAPI application framework
- Reference to `/secure_auth/` for security architecture patterns

## Phase 1: Foundation Setup

### 1.1 Install Dependencies

```bash
pip install authlib[jose] python-multipart email-validator
```

### 1.2 Directory Structure

Create the new auth system:
```
auth/
├── __init__.py
├── models.py          # Database models (reference secure_auth/models.py for security fields)
├── crypto.py          # Authlib-based cryptographic operations  
├── magic_links.py     # Magic link generation and verification
├── sessions.py        # Session management with security context
├── rate_limiting.py   # Rate limiting with database persistence
├── audit.py          # Security event logging
├── middleware.py     # Security middleware
├── api.py            # FastAPI routes
├── config.py         # Configuration management
└── exceptions.py     # Custom exceptions
```

### 1.3 Database Models (auth/models.py)

Reference `secure_auth/models.py` for security field patterns, implement with Authlib:

```python
"""
Authentication models using secure patterns from secure_auth/models.py
with Authlib cryptographic foundation.
"""

from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import uuid

Base = declarative_base()

class User(Base):
    """
    User model with security-first design.
    References secure_auth/models.py User class for security field patterns.
    """
    __tablename__ = "users"
    
    # Core identity
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(320), unique=True, nullable=False, index=True)
    tenant_id = Column(String(100), nullable=False, default="default")
    
    # Security state (patterns from secure_auth/models.py)
    is_active = Column(Boolean, default=True, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    failed_login_count = Column(Integer, default=0, nullable=False)
    account_locked_until = Column(DateTime(timezone=True), nullable=True)
    
    # Security tracking (enhanced from secure_auth patterns)
    trusted_device_ids = Column(ARRAY(String), default=list, nullable=False)
    known_ip_addresses = Column(ARRAY(INET), default=list, nullable=False)
    security_flags = Column(JSONB, default=dict, nullable=False)
    user_metadata = Column(JSONB, default=dict, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    
    @property
    def is_locked(self) -> bool:
        """Check if account is currently locked (from secure_auth pattern)."""
        if not self.account_locked_until:
            return False
        return datetime.utcnow() < self.account_locked_until

class MagicLink(Base):
    """
    Magic link tokens with security tracking.
    Enhanced from secure_auth/models.py MagicLink with Authlib crypto.
    """
    __tablename__ = "magic_links"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True)  # Can be null for non-existent users
    email = Column(String(320), nullable=False, index=True)
    
    # Authlib-managed token (not stored directly for security)
    token_hash = Column(String(128), nullable=False, unique=True, index=True)
    
    # Security context (from secure_auth patterns)
    ip_address = Column(INET, nullable=False)
    user_agent = Column(Text, nullable=True)
    device_fingerprint = Column(String(128), nullable=True)
    
    # State tracking
    attempt_count = Column(Integer, default=0, nullable=False)
    is_used = Column(Boolean, default=False, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    used_at = Column(DateTime(timezone=True), nullable=True)

class UserSession(Base):
    """
    User sessions with security context binding.
    Enhanced from secure_auth/models.py patterns.
    """
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Authlib-managed session token (hashed for storage)
    session_token_hash = Column(String(128), nullable=False, unique=True, index=True)
    
    # Security context binding (critical from secure_auth design)
    ip_address = Column(INET, nullable=False)
    user_agent = Column(Text, nullable=False)
    device_fingerprint = Column(String(128), nullable=True)
    device_name = Column(String(200), nullable=True)
    
    # Session metadata
    auth_method = Column(String(50), nullable=False)  # "magic_link", etc.
    remember_device = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_accessed_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)

# Additional models for SecurityEvent, RateLimit, etc. following secure_auth patterns...
```

## Phase 2: Authlib Cryptographic Foundation

### 2.1 Crypto Module (auth/crypto.py)

```python
"""
Cryptographic operations using Authlib.
Replaces custom crypto from secure_auth with battle-tested Authlib implementation.
"""

import os
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

from authlib.jose import JsonWebSignature, JsonWebToken, JWTClaims
from authlib.common.security import generate_token
from authlib.common.encoding import to_bytes, to_native

class AuthCrypto:
    """Authlib-based cryptographic operations for authentication."""
    
    def __init__(self):
        self.master_key = self._get_master_key()
        self.jws = JsonWebSignature()
        self.jwt = JsonWebToken()
    
    def _get_master_key(self) -> bytes:
        """Get master key from environment (pattern from secure_auth)."""
        key_hex = os.environ.get('AUTH_MASTER_KEY')
        if not key_hex:
            raise ValueError("AUTH_MASTER_KEY environment variable is required")
        
        try:
            key_bytes = bytes.fromhex(key_hex)
        except ValueError:
            raise ValueError("AUTH_MASTER_KEY must be a valid hex string")
        
        if len(key_bytes) < 32:
            raise ValueError("AUTH_MASTER_KEY must be at least 32 bytes (64 hex chars)")
        
        return key_bytes
    
    def generate_magic_link_token(self, email: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generate magic link token using Authlib.
        Returns (token, token_hash) for secure storage.
        """
        # Generate cryptographically secure token
        token = generate_token(32)  # 32 bytes = 256 bits
        
        # Create JWT payload with security context
        payload = {
            'email': email,
            'type': 'magic_link',
            'context': context,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(minutes=15)
        }
        
        # Sign token with master key
        signed_token = self.jwt.encode(
            header={'alg': 'HS256'},
            payload=payload,
            key=self.master_key
        )
        
        # Hash for database storage (never store raw token)
        token_hash = hashlib.sha256(signed_token.encode()).hexdigest()
        
        return signed_token.decode(), token_hash
    
    def verify_magic_link_token(self, token: str, expected_hash: str) -> Optional[Dict[str, Any]]:
        """
        Verify magic link token using constant-time operations.
        Pattern from secure_auth timing attack resistance.
        """
        try:
            # Verify hash matches (constant time comparison)
            actual_hash = hashlib.sha256(token.encode()).hexdigest()
            if not hmac.compare_digest(expected_hash, actual_hash):
                return None
            
            # Verify JWT signature and expiration
            claims = self.jwt.decode(token, key=self.master_key)
            claims.validate()
            
            return dict(claims)
            
        except Exception:
            # Always take same time for invalid tokens (timing attack resistance)
            hmac.compare_digest(expected_hash, "dummy_hash_for_timing")
            return None
    
    def generate_session_token(self, user_id: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """Generate session token with security context binding."""
        payload = {
            'user_id': user_id,
            'type': 'session',
            'context': context,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        
        session_token = self.jwt.encode(
            header={'alg': 'HS256'},
            payload=payload,
            key=self.master_key
        )
        
        token_hash = hashlib.sha256(session_token).hexdigest()
        
        return session_token.decode(), token_hash
```

## Phase 3: Magic Link Implementation

### 3.1 Magic Links (auth/magic_links.py)

```python
"""
Magic link generation and verification.
Enhanced from secure_auth/auth_service.py patterns with Authlib crypto.
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from .crypto import AuthCrypto
from .models import User, MagicLink
from .rate_limiting import RateLimit
from .audit import SecurityAudit
from .exceptions import AuthError, RateLimitError

class MagicLinkService:
    """
    Magic link service with security-first design.
    References secure_auth/auth_service.py for security patterns.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.crypto = AuthCrypto()
        self.rate_limit = RateLimit(db)
        self.audit = SecurityAudit(db)
    
    def request_magic_link(
        self, 
        email: str, 
        ip_address: str, 
        user_agent: str,
        device_fingerprint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Request magic link with rate limiting and security logging.
        Enhanced from secure_auth patterns.
        """
        # Input sanitization (pattern from secure_auth)
        email = self._sanitize_email(email)
        ip_address = self._sanitize_ip(ip_address)
        
        # Rate limiting check (critical from secure_auth design)
        if not self.rate_limit.check_magic_link_request(email, ip_address):
            self.audit.log_rate_limit_violation(email, ip_address, "magic_link_request")
            raise RateLimitError("Too many magic link requests")
        
        # Security context
        context = {
            'ip_address': ip_address,
            'user_agent': user_agent,
            'device_fingerprint': device_fingerprint,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Generate token using Authlib
        token, token_hash = self.crypto.generate_magic_link_token(email, context)
        
        # Store magic link record
        magic_link = MagicLink(
            email=email,
            token_hash=token_hash,
            ip_address=ip_address,
            user_agent=user_agent,
            device_fingerprint=device_fingerprint,
            expires_at=datetime.utcnow() + timedelta(minutes=15)
        )
        
        # Set user_id if user exists
        user = self.db.query(User).filter(User.email == email).first()
        if user:
            magic_link.user_id = user.id
        
        self.db.add(magic_link)
        self.db.commit()
        
        # Security audit logging
        self.audit.log_magic_link_request(email, ip_address, magic_link.id)
        
        return {
            'magic_link_id': str(magic_link.id),
            'token': token,  # Only return for email sending
            'expires_at': magic_link.expires_at.isoformat()
        }
    
    def verify_magic_link(self, token: str, ip_address: str, user_agent: str) -> Dict[str, Any]:
        """
        Verify magic link with security validation.
        Enhanced timing attack resistance from secure_auth patterns.
        """
        # Hash token for database lookup
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        # Find magic link record
        magic_link = self.db.query(MagicLink).filter(
            MagicLink.token_hash == token_hash
        ).first()
        
        if not magic_link:
            # Timing attack resistance - always take same time
            self.crypto.verify_magic_link_token("dummy_token", "dummy_hash")
            self.audit.log_magic_link_failure("", ip_address, "invalid_token")
            raise AuthError("Invalid magic link")
        
        # Verify token cryptographically
        claims = self.crypto.verify_magic_link_token(token, token_hash)
        if not claims:
            self.audit.log_magic_link_failure(magic_link.email, ip_address, "crypto_verification_failed")
            raise AuthError("Invalid magic link")
        
        # Security checks (from secure_auth patterns)
        if magic_link.is_used:
            self.audit.log_magic_link_failure(magic_link.email, ip_address, "already_used")
            raise AuthError("Magic link already used")
        
        if datetime.utcnow() > magic_link.expires_at:
            self.audit.log_magic_link_failure(magic_link.email, ip_address, "expired")
            raise AuthError("Magic link expired")
        
        # Context validation (critical security feature from secure_auth)
        if not self._validate_security_context(claims['context'], ip_address, user_agent):
            self.audit.log_magic_link_failure(magic_link.email, ip_address, "context_mismatch")
            raise AuthError("Security context validation failed")
        
        # Mark as used
        magic_link.is_used = True
        magic_link.used_at = datetime.utcnow()
        
        # Get or create user
        user = self._get_or_create_user(magic_link.email)
        
        self.db.commit()
        
        # Security audit
        self.audit.log_successful_authentication(user.id, ip_address, "magic_link")
        
        return {
            'user_id': str(user.id),
            'email': user.email,
            'magic_link_id': str(magic_link.id)
        }
    
    def _sanitize_email(self, email: str) -> str:
        """Sanitize email input (pattern from secure_auth)."""
        # Reference secure_auth/auth_service.py for implementation
        pass
    
    def _sanitize_ip(self, ip_address: str) -> str:
        """Sanitize IP address (pattern from secure_auth)."""
        # Reference secure_auth/auth_service.py for implementation
        pass
    
    def _validate_security_context(self, stored_context: Dict, ip_address: str, user_agent: str) -> bool:
        """Validate security context matches (from secure_auth)."""
        # Reference secure_auth patterns for context validation
        pass
    
    def _get_or_create_user(self, email: str) -> User:
        """Get existing user or create new one."""
        user = self.db.query(User).filter(User.email == email).first()
        if not user:
            user = User(email=email, email_verified=True)
            self.db.add(user)
            self.db.flush()
        return user
```

## Phase 4: Session Management

### 4.1 Session Service (auth/sessions.py)

```python
"""
Session management with security context binding.
Enhanced from secure_auth/session_service.py patterns with Authlib foundation.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from .crypto import AuthCrypto
from .models import User, UserSession
from .audit import SecurityAudit
from .exceptions import AuthError, SessionError

class SessionService:
    """
    Session management with security context binding.
    References secure_auth/session_service.py for security patterns.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.crypto = AuthCrypto()
        self.audit = SecurityAudit(db)
    
    def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        device_fingerprint: Optional[str] = None,
        device_name: Optional[str] = None,
        remember_device: bool = False,
        auth_method: str = "magic_link"
    ) -> Dict[str, Any]:
        """
        Create new user session with security context binding.
        Enhanced from secure_auth/session_service.py patterns.
        """
        # Security context for token
        context = {
            'user_id': user_id,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'device_fingerprint': device_fingerprint,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Generate session token using Authlib
        session_token, token_hash = self.crypto.generate_session_token(user_id, context)
        
        # Create session record
        session = UserSession(
            user_id=user_id,
            session_token_hash=token_hash,
            ip_address=ip_address,
            user_agent=user_agent,
            device_fingerprint=device_fingerprint,
            device_name=device_name,
            auth_method=auth_method,
            remember_device=remember_device,
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
        
        self.db.add(session)
        self.db.commit()
        
        # Security audit
        self.audit.log_session_created(user_id, ip_address, session.id)
        
        return {
            'session_token': session_token,
            'session_id': str(session.id),
            'expires_at': session.expires_at.isoformat()
        }
    
    def validate_session(self, session_token: str, ip_address: str, user_agent: str) -> Optional[Dict[str, Any]]:
        """
        Validate session token with security context verification.
        Enhanced timing attack resistance from secure_auth patterns.
        """
        # Implementation following secure_auth/session_service.py patterns
        pass
```

## Phase 5: Rate Limiting

### 5.1 Rate Limiting Service (auth/rate_limiting.py)

```python
"""
Rate limiting with database persistence.
Enhanced from secure_auth/rate_limit_service.py patterns.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

class RateLimit:
    """
    Rate limiting service with database persistence.
    References secure_auth/rate_limit_service.py for implementation patterns.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def check_magic_link_request(self, email: str, ip_address: str) -> bool:
        """
        Check if magic link request is within rate limits.
        Enhanced from secure_auth rate limiting patterns.
        """
        # Reference secure_auth/rate_limit_service.py for implementation
        pass
```

## Phase 6: Security Audit

### 6.1 Audit Service (auth/audit.py)

```python
"""
Security event logging and audit trails.
Enhanced from secure_auth/audit_service.py patterns.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from uuid import UUID

class SecurityAudit:
    """
    Security audit logging service.
    References secure_auth/audit_service.py for security event patterns.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def log_magic_link_request(self, email: str, ip_address: str, magic_link_id: UUID):
        """Log magic link request event."""
        # Reference secure_auth/audit_service.py for implementation
        pass
    
    def log_magic_link_failure(self, email: str, ip_address: str, reason: str):
        """Log magic link verification failure."""
        # Reference secure_auth/audit_service.py for implementation
        pass
    
    def log_successful_authentication(self, user_id: UUID, ip_address: str, method: str):
        """Log successful authentication event."""
        # Reference secure_auth/audit_service.py for implementation
        pass
    
    def log_session_created(self, user_id: UUID, ip_address: str, session_id: UUID):
        """Log session creation event."""
        # Reference secure_auth/audit_service.py for implementation
        pass
    
    def log_rate_limit_violation(self, email: str, ip_address: str, action: str):
        """Log rate limit violation."""
        # Reference secure_auth/audit_service.py for implementation
        pass
```

## Phase 7: Exception Handling

### 7.1 Custom Exceptions (auth/exceptions.py)

```python
"""
Custom exceptions for authentication system.
Enhanced from secure_auth/exceptions.py patterns.
"""

class AuthError(Exception):
    """Base authentication error."""
    pass

class RateLimitError(AuthError):
    """Rate limit exceeded error."""
    pass

class SessionError(AuthError):
    """Session-related error."""
    pass

class TokenError(AuthError):
    """Token validation error."""
    pass
```

## Phase 8: FastAPI Integration

### 8.1 API Routes (auth/api.py)

```python
"""
FastAPI routes for authentication.
Enhanced from secure_auth/api.py patterns with proper error handling.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Dict, Any

from .magic_links import MagicLinkService
from .sessions import SessionService
from .exceptions import AuthError, RateLimitError

router = APIRouter(prefix="/auth", tags=["authentication"])

def get_db() -> Session:
    """Database dependency."""
    # Implementation for database session
    pass

def get_client_info(request: Request) -> Dict[str, str]:
    """Extract client information from request."""
    return {
        'ip_address': request.client.host,
        'user_agent': request.headers.get('user-agent', ''),
        'device_fingerprint': request.headers.get('x-device-fingerprint')
    }

@router.post("/magic-link/request")
async def request_magic_link(
    email: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Request magic link for passwordless authentication.
    Enhanced from secure_auth/api.py patterns.
    """
    try:
        client_info = get_client_info(request)
        magic_link_service = MagicLinkService(db)
        
        result = magic_link_service.request_magic_link(
            email=email,
            ip_address=client_info['ip_address'],
            user_agent=client_info['user_agent'],
            device_fingerprint=client_info.get('device_fingerprint')
        )
        
        # Don't return the actual token in the response
        return {
            'magic_link_id': result['magic_link_id'],
            'expires_at': result['expires_at'],
            'message': 'Magic link sent to email'
        }
        
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/magic-link/verify")
async def verify_magic_link(
    token: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Verify magic link and create session.
    Enhanced from secure_auth/api.py patterns.
    """
    try:
        client_info = get_client_info(request)
        magic_link_service = MagicLinkService(db)
        session_service = SessionService(db)
        
        # Verify magic link
        auth_result = magic_link_service.verify_magic_link(
            token=token,
            ip_address=client_info['ip_address'],
            user_agent=client_info['user_agent']
        )
        
        # Create session
        session_result = session_service.create_session(
            user_id=auth_result['user_id'],
            ip_address=client_info['ip_address'],
            user_agent=client_info['user_agent'],
            device_fingerprint=client_info.get('device_fingerprint'),
            auth_method="magic_link"
        )
        
        return {
            'user_id': auth_result['user_id'],
            'email': auth_result['email'],
            'session_token': session_result['session_token'],
            'expires_at': session_result['expires_at']
        }
        
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
```

## Phase 9: Testing Integration

### 9.1 Test Structure

Create comprehensive tests following your Universal Testing Guide methodology:

```
tests/test_auth/
├── conftest.py                    # Database fixtures and test setup
├── test_crypto.py                # Authlib crypto operations testing
├── test_magic_links.py           # Magic link security tests  
├── test_sessions.py              # Session management tests
├── test_rate_limiting.py         # Rate limiting validation
├── test_security_integration.py  # End-to-end security tests
├── test_api.py                   # FastAPI endpoint testing
├── test_timing_attacks.py        # Timing attack resistance validation
└── test_concurrent_operations.py # Concurrency and race condition tests
```

### 9.2 Example Test File (tests/test_auth/test_magic_links.py)

```python
"""
Security-focused tests for magic link functionality.
Following Universal Testing Guide methodology and referencing secure_auth test patterns.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch

from auth.magic_links import MagicLinkService
from auth.models import User, MagicLink
from auth.exceptions import AuthError, RateLimitError

class TestMagicLinkSecurity:
    """Test magic link security properties using real-world attack scenarios."""
    
    def test_magic_link_request_rate_limiting(self, db_session):
        """Test rate limiting prevents brute force magic link requests."""
        service = MagicLinkService(db_session)
        
        # Make requests up to the limit
        for i in range(3):  # Assuming limit is 3
            result = service.request_magic_link(
                email="test@example.com",
                ip_address="192.168.1.100",
                user_agent="Test Browser"
            )
            assert result['magic_link_id']
        
        # Fourth request should be rate limited
        with pytest.raises(RateLimitError):
            service.request_magic_link(
                email="test@example.com",
                ip_address="192.168.1.100",
                user_agent="Test Browser"
            )
    
    def test_magic_link_timing_attack_resistance(self, db_session):
        """Test that magic link verification takes constant time."""
        service = MagicLinkService(db_session)
        
        # Create valid magic link
        result = service.request_magic_link(
            email="test@example.com",
            ip_address="192.168.1.100",
            user_agent="Test Browser"
        )
        valid_token = result['token']
        
        # Measure timing for invalid token
        invalid_times = []
        for _ in range(50):
            start = time.perf_counter()
            try:
                service.verify_magic_link("invalid_token", "192.168.1.100", "Test Browser")
            except AuthError:
                pass
            invalid_times.append(time.perf_counter() - start)
        
        # Measure timing for valid token (but don't actually use it)
        valid_times = []
        for _ in range(50):
            start = time.perf_counter()
            try:
                # Use a copy of the token that will fail hash verification
                service.verify_magic_link(valid_token + "x", "192.168.1.100", "Test Browser")
            except AuthError:
                pass
            valid_times.append(time.perf_counter() - start)
        
        # Timing should be similar (within 20% variance)
        avg_invalid = sum(invalid_times) / len(invalid_times)
        avg_valid = sum(valid_times) / len(valid_times)
        
        timing_variance = abs(avg_invalid - avg_valid) / max(avg_invalid, avg_valid)
        assert timing_variance < 0.2, f"Timing variance too high: {timing_variance:.1%}"
    
    def test_magic_link_context_validation(self, db_session):
        """Test security context validation prevents replay attacks."""
        service = MagicLinkService(db_session)
        
        # Create magic link from one context
        result = service.request_magic_link(
            email="test@example.com",
            ip_address="192.168.1.100",
            user_agent="Original Browser"
        )
        token = result['token']
        
        # Try to use from different IP
        with pytest.raises(AuthError, match="Security context validation failed"):
            service.verify_magic_link(token, "192.168.1.200", "Original Browser")
        
        # Try to use from different user agent
        with pytest.raises(AuthError, match="Security context validation failed"):
            service.verify_magic_link(token, "192.168.1.100", "Different Browser")
    
    def test_magic_link_single_use_enforcement(self, db_session):
        """Test magic links can only be used once."""
        service = MagicLinkService(db_session)
        
        result = service.request_magic_link(
            email="test@example.com",
            ip_address="192.168.1.100",
            user_agent="Test Browser"
        )
        token = result['token']
        
        # First use should succeed
        auth_result = service.verify_magic_link(token, "192.168.1.100", "Test Browser")
        assert auth_result['user_id']
        
        # Second use should fail
        with pytest.raises(AuthError, match="Magic link already used"):
            service.verify_magic_link(token, "192.168.1.100", "Test Browser")
    
    def test_magic_link_expiration_enforcement(self, db_session):
        """Test magic links expire after time limit."""
        service = MagicLinkService(db_session)
        
        # Create magic link
        result = service.request_magic_link(
            email="test@example.com",
            ip_address="192.168.1.100",
            user_agent="Test Browser"
        )
        token = result['token']
        
        # Manually expire the magic link in database
        magic_link = db_session.query(MagicLink).filter(
            MagicLink.id == result['magic_link_id']
        ).first()
        magic_link.expires_at = datetime.utcnow() - timedelta(minutes=1)
        db_session.commit()
        
        # Should fail due to expiration
        with pytest.raises(AuthError, match="Magic link expired"):
            service.verify_magic_link(token, "192.168.1.100", "Test Browser")
```

## Phase 10: Configuration Management

### 10.1 Configuration (auth/config.py)

```python
"""
Configuration management for authentication system.
Enhanced from secure_auth/config.py patterns.
"""

import os
from typing import Optional

class AuthConfig:
    """Authentication system configuration."""
    
    def __init__(self):
        self.master_key = self._get_required_env('AUTH_MASTER_KEY')
        self.database_url = self._build_database_url()
        
        # Rate limiting configuration
        self.magic_link_rate_limit = int(os.environ.get('AUTH_MAGIC_LINK_RATE_LIMIT', '3'))
        self.magic_link_rate_window = int(os.environ.get('AUTH_MAGIC_LINK_RATE_WINDOW', '900'))  # 15 minutes
        
        # Session configuration
        self.session_lifetime_hours = int(os.environ.get('AUTH_SESSION_LIFETIME_HOURS', '24'))
        self.magic_link_lifetime_minutes = int(os.environ.get('AUTH_MAGIC_LINK_LIFETIME_MINUTES', '15'))
    
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable."""
        value = os.environ.get(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def _build_database_url(self) -> str:
        """Build database URL from environment variables."""
        host = os.environ.get('AUTH_DB_HOST', 'localhost')
        port = os.environ.get('AUTH_DB_PORT', '5432')
        name = os.environ.get('AUTH_DB_NAME', 'auth_db')
        user = os.environ.get('AUTH_DB_USER', 'mira_admin')
        password = os.environ.get('AUTH_DB_PASSWORD', '')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"
```

## Phase 11: Migration Strategy

### 11.1 Migration Steps

1. **Keep secure_auth/ as reference** during development
2. **Build auth/ incrementally** with tests at each step
3. **Run both systems in parallel** during transition
4. **Migrate data** from secure_auth tables to auth tables
5. **Remove secure_auth/** only after full validation

### 11.2 Data Migration Script

```python
"""
Migration script from secure_auth to auth system.
"""

def migrate_users():
    """Migrate users from secure_auth.users to auth.users."""
    # Implementation for data migration
    pass

def migrate_sessions():
    """Migrate active sessions."""
    # Implementation for session migration
    pass
```

## Phase 12: Claude Development Instructions

When implementing this authentication system, follow these guidelines:

### 12.1 Reference Guidelines

1. **Reference secure_auth/ extensively** - the security concepts are solid, just need proper implementation
2. **Follow the Universal Testing Guide** - write security-focused tests that validate real attack scenarios  
3. **Use Authlib for all cryptographic operations** - never implement custom crypto
4. **Maintain the security-first design** from secure_auth but with battle-tested foundations

### 12.2 Security Validation

1. **Test timing attack resistance** - measure actual response times like secure_auth tests do
2. **Implement rate limiting with database persistence** - learn from secure_auth's rate limiting design
3. **Add comprehensive audit logging** - follow secure_auth's security event patterns
4. **Validate all security contexts** - ensure context binding prevents replay attacks

### 12.3 Implementation Priority

1. **Phase 1-3**: Core foundation (models, crypto, magic links)
2. **Phase 4-6**: Security services (sessions, rate limiting, audit)
3. **Phase 7-8**: API integration and error handling
4. **Phase 9**: Comprehensive testing with real attack scenarios
5. **Phase 10-11**: Configuration and migration

### 12.4 Quality Gates

Before moving to the next phase:
- All tests pass using Universal Testing Guide methodology
- Security features validated with real attack simulations
- Performance characteristics measured and documented
- Code review focusing on security implications

## Conclusion

This implementation combines the excellent security architecture from `secure_auth/` with the proven cryptographic foundation of Authlib. The result will be a production-grade authentication system that is both secure and maintainable.

The key advantages of this approach:
- **Battle-tested cryptography** from Authlib
- **Advanced security features** from secure_auth design
- **Comprehensive testing** using your proven methodology
- **Clear migration path** from existing system
- **Maintainable codebase** with proper separation of concerns

Remember: The goal is to build something more secure than 95% of production authentication systems, and this approach gives you the foundation to achieve that goal.