"""
Tests for UserDataManager connection ownership and caching.

Verifies the connection-per-instance design:
- Lazy connection creation
- Connection reuse within same manager
- Instance caching across calls to get_user_data_manager
- Proper cleanup via clear_manager_cache
- Thread-safe connection sharing (check_same_thread=False)

Following MIRA's real testing philosophy:
- No mocks, use real SQLite connections
- Test actual behavior, not implementation details
"""
import pytest
import sqlite3
from uuid import UUID

from utils.userdata_manager import (
    UserDataManager,
    get_user_data_manager,
    clear_manager_cache,
    derive_session_key,
    _manager_cache
)


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure clean cache state for each test."""
    clear_manager_cache()
    yield
    clear_manager_cache()


@pytest.fixture
def test_user_id():
    """Provide a test user ID."""
    return UUID('00000000-0000-0000-0000-000000000001')


class TestConnectionOwnership:
    """Tests for lazy connection creation and reuse."""

    def test_connection_created_lazily(self, test_user_id):
        """Connection should not exist until first database access."""
        dm = get_user_data_manager(test_user_id)

        # Connection should be None before first access
        assert dm._conn is None

        # Access connection property
        conn = dm.connection

        # Now connection should exist
        assert dm._conn is not None
        assert isinstance(conn, sqlite3.Connection)

    def test_connection_reused_within_manager(self, test_user_id):
        """Same connection should be returned on subsequent accesses."""
        dm = get_user_data_manager(test_user_id)

        conn1 = dm.connection
        conn2 = dm.connection
        conn3 = dm.connection

        assert conn1 is conn2
        assert conn2 is conn3

    def test_connection_allows_cross_thread_usage(self, test_user_id):
        """Connection should be created with check_same_thread=False."""
        dm = get_user_data_manager(test_user_id)
        conn = dm.connection

        # This would raise ProgrammingError if check_same_thread=True
        # and we're in a different thread. Since we're testing in the same
        # thread, we verify the connection works for basic operations.
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1


class TestInstanceCaching:
    """Tests for UserDataManager instance caching."""

    def test_same_user_returns_cached_instance(self, test_user_id):
        """Multiple calls with same user_id should return same instance."""
        dm1 = get_user_data_manager(test_user_id)
        dm2 = get_user_data_manager(test_user_id)
        dm3 = get_user_data_manager(test_user_id)

        assert dm1 is dm2
        assert dm2 is dm3

    def test_different_users_get_different_instances(self):
        """Different user_ids should return different instances."""
        user1 = UUID('00000000-0000-0000-0000-000000000001')
        user2 = UUID('00000000-0000-0000-0000-000000000002')

        dm1 = get_user_data_manager(user1)
        dm2 = get_user_data_manager(user2)

        assert dm1 is not dm2
        assert dm1.user_id != dm2.user_id

    def test_cached_instance_shares_connection(self, test_user_id):
        """Cached instances should share the same connection."""
        dm1 = get_user_data_manager(test_user_id)
        conn1 = dm1.connection

        dm2 = get_user_data_manager(test_user_id)
        conn2 = dm2.connection

        assert dm1 is dm2  # Same instance
        assert conn1 is conn2  # Same connection


class TestCacheCleanup:
    """Tests for clear_manager_cache functionality."""

    def test_clear_specific_user(self, test_user_id):
        """Clearing specific user should close their connection."""
        dm = get_user_data_manager(test_user_id)
        conn = dm.connection

        # Verify connection is open
        assert dm._conn is not None

        # Clear cache for this user
        clear_manager_cache(test_user_id)

        # Connection should be closed
        assert dm._conn is None

        # User should not be in cache
        assert str(test_user_id) not in _manager_cache

    def test_clear_all_users(self):
        """Clearing all users should close all connections."""
        user1 = UUID('00000000-0000-0000-0000-000000000001')
        user2 = UUID('00000000-0000-0000-0000-000000000002')

        dm1 = get_user_data_manager(user1)
        dm2 = get_user_data_manager(user2)

        # Create connections
        dm1.connection
        dm2.connection

        assert len(_manager_cache) == 2

        # Clear all
        clear_manager_cache()

        assert len(_manager_cache) == 0
        assert dm1._conn is None
        assert dm2._conn is None

    def test_new_manager_after_clear(self, test_user_id):
        """After clearing, new manager should be created with fresh connection."""
        dm1 = get_user_data_manager(test_user_id)
        conn1 = dm1.connection

        clear_manager_cache(test_user_id)

        dm2 = get_user_data_manager(test_user_id)
        conn2 = dm2.connection

        # Should be different instances
        assert dm1 is not dm2

        # Old connection should be closed, new one created
        assert dm1._conn is None
        assert dm2._conn is not None


class TestDatabaseOperations:
    """Tests for database operations through the persistent connection."""

    def test_execute_returns_results(self, test_user_id):
        """Execute should return query results as list of dicts."""
        dm = get_user_data_manager(test_user_id)

        results = dm.execute("SELECT 1 as num, 'hello' as msg")

        assert len(results) == 1
        assert results[0]['num'] == 1
        assert results[0]['msg'] == 'hello'

    def test_fetchone_returns_single_row(self, test_user_id):
        """Fetchone should return single row or None."""
        dm = get_user_data_manager(test_user_id)

        result = dm.fetchone("SELECT 42 as answer")

        assert result is not None
        assert result['answer'] == 42

    def test_fetchone_returns_none_for_empty(self, test_user_id):
        """Fetchone should return None for empty results."""
        dm = get_user_data_manager(test_user_id)

        # Create temp table, query non-existent row
        dm.execute("CREATE TABLE IF NOT EXISTS test_empty (id INTEGER)")
        result = dm.fetchone("SELECT * FROM test_empty WHERE id = 999")

        assert result is None

    def test_insert_and_select_roundtrip(self, test_user_id):
        """Data should persist through insert and be readable."""
        dm = get_user_data_manager(test_user_id)

        # Create test table
        dm.execute("""
            CREATE TABLE IF NOT EXISTS test_roundtrip (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)

        # Insert data
        dm.execute("DELETE FROM test_roundtrip")  # Clean slate
        dm.execute("INSERT INTO test_roundtrip (name) VALUES (:name)", {"name": "test_value"})

        # Read back
        result = dm.fetchone("SELECT name FROM test_roundtrip WHERE name = :name", {"name": "test_value"})

        assert result is not None
        assert result['name'] == 'test_value'


class TestContextManager:
    """Tests for context manager support."""

    def test_context_manager_closes_connection(self, test_user_id):
        """Context manager should close connection on exit."""
        session_key = derive_session_key(str(test_user_id))

        with UserDataManager(test_user_id, session_key) as dm:
            conn = dm.connection
            assert dm._conn is not None

        # After context exit, connection should be closed
        assert dm._conn is None

    def test_context_manager_allows_operations(self, test_user_id):
        """Database operations should work within context manager."""
        session_key = derive_session_key(str(test_user_id))

        with UserDataManager(test_user_id, session_key) as dm:
            result = dm.execute("SELECT 'context_test' as value")
            assert result[0]['value'] == 'context_test'
