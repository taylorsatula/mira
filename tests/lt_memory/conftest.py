"""
Fixtures specific to lt_memory module testing.

Provides session managers, database access, and test users for lt_memory tests.
"""

import pytest
from tests.fixtures.core import TEST_USER_ID, ensure_test_user_exists
from utils.user_context import set_current_user_id, clear_user_context
from utils.database_session_manager import LTMemorySessionManager


@pytest.fixture
def lt_memory_session_manager():
    """
    Provide LTMemorySessionManager for testing.

    Uses the production session manager pointing to the real mira_memory database
    with the test user.
    """
    from utils.database_session_manager import get_shared_session_manager
    return get_shared_session_manager()


@pytest.fixture
def test_user():
    """
    Provide test user record with ID for lt_memory tests.

    Automatically sets user context for the test since many db_access
    operations need it internally for activity day calculation.

    Returns dict with user_id and other user fields.
    """
    user_record = ensure_test_user_exists()

    # Set user context automatically for convenience
    set_current_user_id(user_record["id"])

    # Return a simple dict with user_id
    yield {
        "user_id": user_record["id"],
        "email": user_record.get("email", "test@example.com"),
    }

    # Cleanup happens via reset_user_context autouse fixture


@pytest.fixture
def sqlite_test_db():
    """
    Placeholder to prevent imports of the wrong fixture type.

    LT_Memory tests should use lt_memory_session_manager, not sqlite_test_db.
    """
    raise RuntimeError(
        "lt_memory tests should use 'lt_memory_session_manager' fixture, "
        "not 'sqlite_test_db'. SQLite is for tool storage, not LT_Memory."
    )


@pytest.fixture
def embeddings_provider():
    """
    Provide real HybridEmbeddingsProvider for testing.

    Returns the singleton instance with mdbr-leaf-ir-asym (768d) model,
    plus BGE reranker for testing reranking functionality.
    """
    from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
    return get_hybrid_embeddings_provider(cache_enabled=True, enable_reranker=True)


@pytest.fixture
def lt_memory_db(lt_memory_session_manager, test_user):
    """
    Provide real LTMemoryDB for testing.

    Uses actual database with test user isolation via RLS.
    User context is automatically set by test_user fixture.
    """
    from lt_memory.db_access import LTMemoryDB
    return LTMemoryDB(lt_memory_session_manager)


@pytest.fixture
def vector_ops(embeddings_provider, lt_memory_db):
    """
    Provide real VectorOps service for testing.

    Uses actual embeddings provider and database - no mocks.
    Tests will exercise real embedding generation, database queries,
    and reranking functionality.
    """
    from lt_memory.vector_ops import VectorOps
    return VectorOps(embeddings_provider, lt_memory_db)
