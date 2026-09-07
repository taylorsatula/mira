"""
Test isolation fixtures for cleaning up global state between tests.

Ensures tests don't interfere with each other through shared global state.
"""
import pytest
import logging
from clients.valkey_client import get_valkey

from tests.fixtures.infra import integration_mode

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def reset_global_state(request):
    """
    Automatically reset global state before and after each test.

    This fixture runs for every test and ensures:
    - User locks are cleared from Valkey
    - Any other global state is reset

    Inert outside integration mode: get_valkey() constructs a client whose
    config comes from Vault, so it raises ValueError/PermissionError for
    every unit test when infrastructure is absent (O-22).
    """
    if not integration_mode(request.config):
        yield
        return

    # Clear distributed locks before test
    valkey = get_valkey()
    try:
        # Clear all user locks
        for key in valkey.scan_iter(match="user_lock:*"):
            valkey.delete(key)
    except Exception as e:
        logger.warning(f"Could not clear user locks: {e}")
    
    yield
    
    # Clear after test
    try:
        # Clear all user locks
        for key in valkey.scan_iter(match="user_lock:*"):
            valkey.delete(key)
    except Exception as e:
        logger.warning(f"Could not clear user locks: {e}")
    


@pytest.fixture
def clean_user_locks():
    """
    Fixture to ensure user locks are clean for specific tests.
    
    Use this when you need explicit control over lock state.
    """
    valkey = get_valkey()
    
    # Clear all user locks before test
    try:
        for key in valkey.scan_iter(match="user_lock:*"):
            valkey.delete(key)
    except Exception as e:
        logger.warning(f"Could not clear user locks: {e}")
    
    yield
    
    # Clear all user locks after test
    try:
        for key in valkey.scan_iter(match="user_lock:*"):
            valkey.delete(key)
    except Exception as e:
        logger.warning(f"Could not clear user locks: {e}")