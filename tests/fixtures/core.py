"""
Core test fixtures for OSS mode.

Uses direct database queries instead of AuthDatabase/AuthService.
Test users (test@example.com, test2@example.com) must exist in the database.
"""
import pytest
import pytest_asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Test user constants - same as multi-user MIRA
TEST_USER_EMAIL = "test@example.com"
SECOND_TEST_USER_EMAIL = "test2@example.com"
# Restored for WP0: the OSS release scrub removed these two while leaving the
# email constants and the imports in tests/lt_memory/conftest.py and
# tests/working_memory/test_notification_center.py in place, so the suite could
# not be collected. Values match the pre-scrub file and tests/fixtures/auth.py.
TEST_USER_ID = "443a898d-ed56-495a-b9de-0551c80169fe"
SECOND_TEST_USER_ID = "7b8e4c2a-f3d1-4a5b-9e6c-1d2f3a4b5c6d"


def get_user_by_email(email: str) -> Optional[dict]:
    """
    Get a user by email directly from the database.

    This replaces AuthDatabase.get_user_by_email() for OSS mode.
    """
    from utils.database_session_manager import get_shared_session_manager

    session_manager = get_shared_session_manager()

    with session_manager.get_admin_session() as session:
        result = session.execute_single(
            "SELECT id, email, is_active, created_at, timezone, memory_manipulation_enabled "
            "FROM users WHERE email = %(email)s",
            {'email': email}
        )

        if not result:
            return None

        # Return a dict that mimics UserRecord interface
        return {
            "id": str(result['id']),
            "email": result['email'],
            "is_active": result['is_active'],
            "created_at": result['created_at'],
            "timezone": result.get('timezone', 'UTC'),
            "memory_manipulation_enabled": result.get('memory_manipulation_enabled', False)
        }


def create_test_user(email: str, first_name: str = "Test", last_name: str = "User") -> str:
    """
    Create a test user in the database.

    Returns the user ID.
    """
    import uuid
    from utils.database_session_manager import get_shared_session_manager

    session_manager = get_shared_session_manager()
    user_id = str(uuid.uuid4())

    with session_manager.get_admin_session() as session:
        session.execute_update("""
            INSERT INTO users (id, email, is_active, memory_manipulation_enabled, timezone)
            VALUES (%(id)s, %(email)s, true, false, 'UTC')
        """, {'id': user_id, 'email': email})

        logger.info(f"Created test user {email} with ID {user_id}")
        return user_id


def ensure_test_user_exists() -> dict:
    """
    Ensure the test user exists in the database, creating if necessary.
    Returns the user record with actual user ID.
    """
    user = get_user_by_email(TEST_USER_EMAIL)
    if user:
        return user

    # User doesn't exist, create new one
    try:
        user_id = create_test_user(TEST_USER_EMAIL, "Test", "User")
        user = get_user_by_email(TEST_USER_EMAIL)
        if user:
            return user
        raise RuntimeError(f"Created user but couldn't retrieve it")
    except Exception as e:
        # If creation fails (race condition), try again
        user = get_user_by_email(TEST_USER_EMAIL)
        if user:
            return user
        raise RuntimeError(f"Failed to ensure test user exists: {e}")


def ensure_test_user_ready() -> dict:
    """
    Ensure test user is fully set up and ready: user record + continuum + context.

    Returns:
        dict with 'user_id', 'email', 'continuum_id' - everything needed for testing
    """
    from cns.infrastructure.continuum_repository import get_continuum_repository
    from clients.postgres_client import PostgresClient
    from utils.user_context import set_current_user_id

    # Step 1: Ensure user exists
    user = ensure_test_user_exists()
    user_id = user["id"]

    # Step 2: Set user context for RLS
    set_current_user_id(user_id)

    # Step 3: Ensure continuum exists
    db = PostgresClient("mira_service", user_id=user_id)
    result = db.execute_query(
        "SELECT id FROM continuums WHERE user_id = %s LIMIT 1",
        (user_id,)
    )

    if result and len(result) > 0:
        continuum_id = str(result[0][0])
        logger.debug(f"Using existing continuum {continuum_id} for test user {user_id}")
    else:
        repo = get_continuum_repository()
        continuum = repo.create_continuum(user_id)
        continuum_id = str(continuum.id)
        logger.info(f"Created continuum {continuum_id} for test user {user_id}")

    # Tool schemas (contacts, pager, domaindoc) self-initialise idempotently
    # from UserDataManager._ensure_database on first connection, so there is no
    # schema-distribution step to run here.

    return {
        "user_id": user_id,
        "email": user["email"],
        "continuum_id": continuum_id,
        "is_active": user["is_active"],
        "created_at": user["created_at"]
    }


def ensure_second_test_user_exists() -> dict:
    """
    Ensure the second test user exists in the database.
    Used for RLS isolation testing.
    """
    user = get_user_by_email(SECOND_TEST_USER_EMAIL)
    if user:
        return user

    try:
        user_id = create_test_user(SECOND_TEST_USER_EMAIL, "Second", "User")
        user = get_user_by_email(SECOND_TEST_USER_EMAIL)
        if user:
            return user
        raise RuntimeError(f"Created user but couldn't retrieve it")
    except Exception as e:
        user = get_user_by_email(SECOND_TEST_USER_EMAIL)
        if user:
            return user
        raise RuntimeError(f"Failed to ensure second test user exists: {e}")


def ensure_second_test_user_ready() -> dict:
    """
    Ensure second test user is fully set up for RLS testing.
    """
    from cns.infrastructure.continuum_repository import get_continuum_repository
    from clients.postgres_client import PostgresClient
    from utils.user_context import set_current_user_id

    user = ensure_second_test_user_exists()
    user_id = user["id"]

    set_current_user_id(user_id)

    db = PostgresClient("mira_service", user_id=user_id)
    result = db.execute_query(
        "SELECT id FROM continuums WHERE user_id = %s LIMIT 1",
        (user_id,)
    )

    if result and len(result) > 0:
        continuum_id = str(result[0][0])
        logger.debug(f"Using existing continuum {continuum_id} for second test user {user_id}")
    else:
        repo = get_continuum_repository()
        continuum = repo.create_continuum(user_id)
        continuum_id = str(continuum.id)
        logger.info(f"Created continuum {continuum_id} for second test user {user_id}")

    # Tool schemas self-initialise from UserDataManager._ensure_database on the
    # first connection for this user, so no distribution step is needed here.

    return {
        "user_id": user_id,
        "email": user["email"],
        "continuum_id": continuum_id,
        "is_active": user["is_active"],
        "created_at": user["created_at"]
    }


# ================== DATABASE FIXTURES ==================

@pytest_asyncio.fixture
async def test_db():
    """
    Provide real database access with the test user context.
    """
    from clients.postgres_client import PostgresClient
    from utils.user_context import set_current_user_id, clear_user_context

    user = ensure_test_user_exists()
    user_id = user["id"]

    set_current_user_id(user_id)
    db = PostgresClient("mira_service", user_id=user_id)

    try:
        yield db
    finally:
        clear_user_context()


@pytest_asyncio.fixture
async def test_memory_db():
    """Provide real database access with test user context."""
    from clients.postgres_client import PostgresClient
    from utils.user_context import set_current_user_id, clear_user_context

    user = ensure_test_user_exists()
    user_id = user["id"]

    set_current_user_id(user_id)
    db = PostgresClient("mira_service", user_id=user_id)

    try:
        yield db
    finally:
        clear_user_context()


# ================== SERVICE FIXTURES ==================

@pytest_asyncio.fixture
async def conversation_repository():
    """
    Provide the real continuum repository singleton.
    """
    from cns.infrastructure.continuum_repository import get_continuum_repository
    return get_continuum_repository()


@pytest_asyncio.fixture
async def continuum_repo(conversation_repository):
    """Alias for conversation_repository (ContinuumRepository singleton)."""
    return conversation_repository


@pytest.fixture
def event_bus():
    """EventBus instance for CNS event handling tests."""
    from cns.integration.event_bus import EventBus
    bus = EventBus()
    yield bus
    bus.shutdown()


@pytest.fixture
def continuum_pool():
    """Mock ContinuumPool for segment collapse handler tests."""
    from unittest.mock import Mock
    pool = Mock()
    pool.invalidate.return_value = None
    pool.get_or_create.return_value = Mock()
    return pool


@pytest_asyncio.fixture
async def vault_client():
    """Provide real vault client for testing."""
    from clients.vault_client import VaultClient
    try:
        client = VaultClient()
        client.get_secret('mira/auth', 'jwt_secret_key')
        return client
    except Exception as e:
        pytest.skip(f"Vault unavailable: {e}")


@pytest_asyncio.fixture
async def valkey_client():
    """
    Provide real Valkey client for testing.
    """
    from clients.valkey_client import get_valkey_client
    client = get_valkey_client()

    if not client.valkey_available:
        pytest.skip("Valkey unavailable")

    return client


# ================== USER & AUTH FIXTURES ==================

@pytest_asyncio.fixture
async def authenticated_user():
    """
    Provide authenticated test user with context set up.

    In OSS mode, this sets up the test user context without session tokens
    since API key auth is used instead.
    """
    from utils.user_context import set_current_user_data, clear_user_context

    setup = ensure_test_user_ready()

    user_data = {
        "user_id": str(setup["user_id"]),
        "email": setup["email"],
        "continuum_id": setup["continuum_id"],
        "is_active": setup["is_active"]
    }
    set_current_user_data(user_data)

    try:
        yield user_data
    finally:
        clear_user_context()


@pytest_asyncio.fixture
async def second_authenticated_user():
    """
    Provide second authenticated test user for RLS testing.
    """
    from utils.user_context import set_current_user_data, clear_user_context

    setup = ensure_second_test_user_ready()

    user_data = {
        "user_id": str(setup["user_id"]),
        "email": setup["email"],
        "continuum_id": setup["continuum_id"],
        "is_active": setup["is_active"]
    }
    set_current_user_data(user_data)

    try:
        yield user_data
    finally:
        clear_user_context()


# ================== CONTINUUM FIXTURES ==================

@pytest_asyncio.fixture
async def realistic_conversation(conversation_repository, authenticated_user):
    """
    Load the realistic continuum from JSON fixture.
    """
    from tests.fixtures.conversation_data import load_realistic_conversation_data

    continuum = await load_realistic_conversation_data()
    return continuum


@pytest_asyncio.fixture
async def realistic_messages():
    """
    Provide just the realistic messages without creating a continuum.
    """
    import json
    from pathlib import Path
    from datetime import timedelta
    from cns.core.message import Message
    from utils.timezone_utils import utc_now

    fixture_path = Path(__file__).parent / "realistic_conversation.json"
    with open(fixture_path, 'r') as f:
        data = json.load(f)

    messages = []
    base_time = utc_now()

    for msg_data in data['messages']:
        timestamp_offset = timedelta(hours=msg_data.get('timestamp_offset_hours', 0))
        message = Message(
            content=msg_data['content'],
            role=msg_data['role'],
            created_at=base_time + timestamp_offset,
            metadata=msg_data.get('metadata', {})
        )
        messages.append(message)

    return messages


# ================== CLEANUP FIXTURES ==================

@pytest.fixture(autouse=True)
def cleanup_test_data(request):
    """
    Automatically clean up test data before and after each test.

    Inert outside integration mode: cleanup_test_user_data() opens a real
    database pool (Vault-backed).  It already swallowed its exceptions, but
    it logged an ERROR/WARNING pair and attempted a connection for every
    unit test (O-22).
    """
    from tests.fixtures.conversation_data import cleanup_test_user_data
    from tests.fixtures.infra import integration_mode

    if not integration_mode(request.config):
        yield
        return

    try:
        cleanup_test_user_data()
        logger.debug("Pre-test cleanup completed successfully")
    except Exception as e:
        logger.warning(f"Pre-test cleanup failed: {e}")

    yield

    try:
        cleanup_test_user_data()
        logger.debug("Post-test cleanup completed successfully")
    except Exception as e:
        logger.warning(f"Post-test cleanup failed: {e}")


@pytest.fixture(autouse=True)
def reset_user_context():
    """
    Ensure user context is clean for each test.
    """
    from utils.user_context import clear_user_context

    clear_user_context()
    yield
    clear_user_context()


# ================== INTEGRATION TEST FIXTURES ==================

@pytest_asyncio.fixture
async def chat_handler(conversation_repository, authenticated_user):
    """
    Provide real ChatHandler with all dependencies.
    """
    from cns.api.chat import ChatHandler
    from cns.integration.factory import create_cns_orchestrator

    orchestrator = await create_cns_orchestrator()
    return ChatHandler(orchestrator, conversation_repository)


@pytest.fixture
def orchestrator():
    """Provide real CNS orchestrator for integration tests."""
    from cns.integration.factory import create_cns_orchestrator
    from utils.user_context import set_current_user_id, clear_user_context

    user = ensure_test_user_exists()
    user_id = user["id"]

    set_current_user_id(user_id)

    try:
        orchestrator_instance = create_cns_orchestrator()
        yield orchestrator_instance
    finally:
        clear_user_context()


# ================== FASTAPI TEST CLIENT ==================

@pytest.fixture
def test_client():
    """
    Create FastAPI TestClient with real app configuration.
    """
    from fastapi.testclient import TestClient
    from main import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def authenticated_client(test_client):
    """
    Create authenticated test client with API key header.
    """
    from clients.vault_client import _ensure_vault_client

    try:
        vault_client = _ensure_vault_client()
        secret_data = vault_client.client.secrets.kv.v2.read_secret_version(
            path='mira/api_keys'
        )
        api_key = secret_data['data']['data'].get('mira_api')
        test_client.headers = {"Authorization": f"Bearer {api_key}"}
    except Exception as e:
        pytest.skip(f"Could not get API key from Vault: {e}")

    return test_client


@pytest_asyncio.fixture
async def async_client():
    """
    Create async HTTP client for testing async endpoints.
    """
    import httpx
    from main import create_app

    app = create_app()
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def authenticated_async_client():
    """
    Create authenticated async HTTP client.
    """
    import httpx
    from main import create_app
    from clients.vault_client import _ensure_vault_client

    app = create_app()

    try:
        vault_client = _ensure_vault_client()
        secret_data = vault_client.client.secrets.kv.v2.read_secret_version(
            path='mira/api_keys'
        )
        api_key = secret_data['data']['data'].get('mira_api')
    except Exception as e:
        pytest.skip(f"Could not get API key from Vault: {e}")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        client.headers = {"Authorization": f"Bearer {api_key}"}
        yield client


# ================== REQUEST DATA FIXTURES ==================

@pytest.fixture
def sample_chat_request():
    """Provide sample chat request data."""
    return {
        "message": "Hello, this is a test message",
        "continuum_id": None,
        "stream": False
    }


# ================== PERFORMANCE FIXTURES ==================

@pytest.fixture
def timer():
    """Simple timer for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()
            return self.elapsed()

        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer()
