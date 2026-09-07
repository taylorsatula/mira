"""
Auth fixtures for the new lean magic-link authentication system.

Provides real services for testing without mocks, using actual Postgres and Valkey.
"""
import pytest
import pytest_asyncio
import secrets
import hashlib
from typing import Dict, Any, Optional
from datetime import timedelta

from utils.timezone_utils import utc_now
from utils.database_session_manager import get_shared_session_manager
from utils.user_context import set_current_user_id, clear_user_context

# Test user constants (from core fixtures)
TEST_USER_ID = "443a898d-ed56-495a-b9de-0551c80169fe"
TEST_USER_EMAIL = "test@example.com"


@pytest_asyncio.fixture
async def auth_service():
    """Create auth service instance for testing."""
    from auth.service import AuthService
    return AuthService()


@pytest.fixture
def test_email():
    """Return the consistent test user email."""
    return TEST_USER_EMAIL


@pytest_asyncio.fixture
async def test_user(test_email):
    """Get the persistent test user (reuses same user across tests)."""
    from tests.fixtures.core import ensure_test_user_exists
    
    # Get the persistent test user
    user_record = ensure_test_user_exists()
    
    yield {
        "id": user_record["id"],
        "email": user_record["email"],
        "is_active": user_record["is_active"],
        "created_at": user_record["created_at"]
    }
    
    # No cleanup - we reuse the user across tests


@pytest.fixture
def captured_emails():
    """Capture emails instead of sending them."""
    captured = []
    
    class EmailCapture:
        def __init__(self):
            self.emails = captured
        
        def send_magic_link(self, email: str, token: str):
            self.emails.append({
                "to": email,
                "token": token,
                "timestamp": utc_now()
            })
            return True
    
    return EmailCapture()


@pytest_asyncio.fixture
async def magic_link_token(test_user):
    """Create a valid magic link token for testing."""
    from auth.database import AuthDatabase
    
    db = AuthDatabase()
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    
    db.create_magic_link(
        user_id=test_user["id"],
        email=test_user["email"],
        token_hash=token_hash,
        expires_at=utc_now() + timedelta(minutes=10)
    )
    
    return token


@pytest_asyncio.fixture
async def expired_magic_link_token(test_user):
    """Create an expired magic link token for testing."""
    from auth.database import AuthDatabase
    
    db = AuthDatabase()
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    
    db.create_magic_link(
        user_id=test_user["id"],
        email=test_user["email"],
        token_hash=token_hash,
        expires_at=utc_now() - timedelta(minutes=1)  # Already expired
    )
    
    return token


@pytest.fixture
def session_token(test_user, auth_service):
    """Create a valid session token for testing."""
    session_token = auth_service.create_session(
        user_id=test_user["id"],
        user_data=test_user
    )
    return session_token


@pytest.fixture
def security_test_vectors():
    """Security test vectors for input validation."""
    return {
        "valid_emails": [
            "user@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org"
        ],
        "invalid_emails": [
            "notanemail",
            "@domain.com",
            "user@",
            "user space@domain.com",
            "",
            None
        ],
        "malicious_inputs": [
            "<script>alert('xss')</script>@domain.com",
            "user'; DROP TABLE users; --@domain.com",
            "../../../etc/passwd@domain.com"
        ]
    }


@pytest_asyncio.fixture(autouse=True)
async def clean_test_valkey(request):
    """Clean Valkey test data before and after tests.

    Inert outside integration mode (O-22); when integration mode is on its
    pre-existing swallow-and-continue behaviour is unchanged.
    """
    from clients.valkey_client import get_valkey
    from tests.fixtures.infra import integration_mode

    if not integration_mode(request.config):
        yield
        return

    try:
        valkey = await get_valkey()
        # Clean any test session keys before test
        cursor = 0
        while True:
            cursor, keys = await valkey.valkey.scan(cursor, match="session:test_*", count=100)
            for key in keys:
                await valkey.delete(key)
            if cursor == 0:
                break
    except Exception:
        pass  # Valkey might not be available
    
    yield
    
    # Clean after test (same logic)
    try:
        valkey = await get_valkey()
        cursor = 0
        while True:
            cursor, keys = await valkey.valkey.scan(cursor, match="session:test_*", count=100)
            for key in keys:
                await valkey.delete(key)
            if cursor == 0:
                break
    except Exception:
        pass


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for testing rate limit scenarios."""
    class MockRateLimiter:
        def __init__(self):
            self.call_count = {}
        
        def is_allowed(self, key: str) -> tuple[bool, int]:
            self.call_count[key] = self.call_count.get(key, 0) + 1
            if self.call_count[key] > 5:
                return False, 300  # 5 minute window
            return True, 0
        
        def reset(self, key: str) -> bool:
            self.call_count.pop(key, None)
            return True
    
    return MockRateLimiter()


@pytest.fixture
def auth_cookie_settings():
    """Expected secure cookie settings."""
    return {
        "samesite": "Strict",
        "httponly": True,
        "secure": True,
        "max_age": 86400 * 7  # 7 days
    }