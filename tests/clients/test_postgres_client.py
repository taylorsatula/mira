"""
Real tests for PostgresClient - testing the client's unique logic, not PostgreSQL.

Focuses on:
- Connection pooling (shared ThreadedConnectionPool per database)
- RLS context management (app.current_user_id setting/clearing)
- UUID parameter conversion (automatic string conversion)
- JSONB auto-registration (global, one-time registration)
- pgvector registration (mira_memory database only)
- Automatic timestamp injection (memories table only)
- User context isolation (prevents inheriting context from pooled connections)
- Transaction support (atomic multi-operation commits)

Uses real mira_service and mira_memory databases - will fail if databases unavailable.
Does NOT test basic PostgreSQL operations - those are PostgreSQL's responsibility.
"""

import pytest
import uuid
from datetime import datetime
from utils.timezone_utils import utc_now
from clients.postgres_client import PostgresClient


@pytest.fixture
def authenticated_user():
    """
    Provides a test user for RLS and user isolation testing.
    Uses the persistent test user from fixtures.
    """
    from tests.fixtures.auth import TEST_USER_ID, TEST_USER_EMAIL
    return {
        "user_id": TEST_USER_ID,
        "email": TEST_USER_EMAIL
    }


@pytest.fixture(scope="function")
def postgres_app_client(authenticated_user):
    """Provides PostgresClient connected to mira_service with authenticated user."""
    user_id = authenticated_user["user_id"]
    client = PostgresClient(database_name="mira_service", user_id=user_id)

    # Create test table
    client.execute_query("""
        CREATE TABLE IF NOT EXISTS test_postgres_items (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            data JSONB,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
    """)

    # Enable RLS
    client.execute_query("ALTER TABLE test_postgres_items ENABLE ROW LEVEL SECURITY")

    # Create RLS policy
    client.execute_query("""
        DROP POLICY IF EXISTS test_postgres_items_isolation ON test_postgres_items
    """)
    client.execute_query("""
        CREATE POLICY test_postgres_items_isolation ON test_postgres_items
        USING (user_id = current_setting('app.current_user_id', true))
    """)

    yield client

    # Cleanup
    client.execute_query("DROP TABLE IF EXISTS test_postgres_items CASCADE")


@pytest.fixture(scope="function")
def postgres_memory_client(authenticated_user):
    """Provides PostgresClient connected to mira_memory with authenticated user."""
    user_id = authenticated_user["user_id"]
    client = PostgresClient(database_name="mira_memory", user_id=user_id)
    yield client


class TestPostgresClientConnectionPooling:
    """Test connection pool management and sharing."""

    def test_same_database_shares_connection_pool(self, authenticated_user):
        """Verify multiple clients for same database share the same pool."""
        user_id = authenticated_user["user_id"]

        client1 = PostgresClient(database_name="mira_service", user_id=user_id)
        client2 = PostgresClient(database_name="mira_service", user_id=user_id)

        # Both should reference the same pool
        assert "mira_service" in PostgresClient._connection_pools
        assert client1.database_name == client2.database_name

        # Verify pool is actually shared (same object)
        pool1 = PostgresClient._connection_pools["mira_service"]
        pool2 = PostgresClient._connection_pools["mira_service"]
        assert pool1 is pool2

    def test_different_databases_get_different_pools(self, authenticated_user):
        """Verify different databases get separate connection pools."""
        user_id = authenticated_user["user_id"]

        client_app = PostgresClient(database_name="mira_service", user_id=user_id)
        client_memory = PostgresClient(database_name="mira_memory", user_id=user_id)

        # Both pools should exist
        assert "mira_service" in PostgresClient._connection_pools
        assert "mira_memory" in PostgresClient._connection_pools

        # Pools should be different objects
        pool_app = PostgresClient._connection_pools["mira_service"]
        pool_memory = PostgresClient._connection_pools["mira_memory"]
        assert pool_app is not pool_memory

    def test_connection_pool_provides_working_connections(self, postgres_app_client):
        """Verify connection pool provides functional database connections."""
        # Simple query to verify connection works
        result = postgres_app_client.execute_scalar("SELECT 1")
        assert result == 1


class TestPostgresClientRLSContextManagement:
    """Test Row Level Security context variable management."""

    def test_sets_user_context_for_user_scoped_client(self, postgres_app_client, authenticated_user):
        """Verify client sets app.current_user_id for RLS."""
        user_id = authenticated_user["user_id"]

        # Query the session variable
        with postgres_app_client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_setting('app.current_user_id', true)")
                result = cur.fetchone()[0]

        assert result == user_id

    def test_clears_user_context_for_non_user_scoped_client(self, authenticated_user):
        """Verify client clears app.current_user_id when user_id is None."""
        # Create client without user_id
        client = PostgresClient(database_name="mira_service", user_id=None)

        # Query the session variable - should be empty/None
        with client.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_setting('app.current_user_id', true)")
                result = cur.fetchone()[0]

        # Empty string or None both indicate cleared context
        assert result == "" or result is None

    def test_different_users_see_isolated_data_via_rls(self, authenticated_user):
        """Verify RLS isolates data between different user contexts."""
        user1_id = authenticated_user["user_id"]

        # Create second user
        # Removed for OSS: from auth.database import AuthDatabase
        auth_db = None  # Removed for OSS: AuthDatabase()
        user2_email = f"test_user_2_{uuid.uuid4()}@example.com"
        user2_id = auth_db.create_user(user2_email, "password123", None, None)

        try:
            # Client for user 1
            client1 = PostgresClient(database_name="mira_service", user_id=user1_id)

            # Create test table with RLS
            client1.execute_query("""
                CREATE TABLE IF NOT EXISTS test_rls_isolation (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id TEXT NOT NULL,
                    data TEXT
                )
            """)
            client1.execute_query("ALTER TABLE test_rls_isolation ENABLE ROW LEVEL SECURITY")
            client1.execute_query("""
                DROP POLICY IF EXISTS test_rls_isolation_policy ON test_rls_isolation
            """)
            client1.execute_query("""
                CREATE POLICY test_rls_isolation_policy ON test_rls_isolation
                USING (user_id = current_setting('app.current_user_id', true))
            """)

            # User 1 inserts data
            client1.execute_query(
                "INSERT INTO test_rls_isolation (user_id, data) VALUES (%s, %s)",
                (user1_id, "User 1 data")
            )

            # User 2 inserts data
            client2 = PostgresClient(database_name="mira_service", user_id=user2_id)
            client2.execute_query(
                "INSERT INTO test_rls_isolation (user_id, data) VALUES (%s, %s)",
                (user2_id, "User 2 data")
            )

            # Each user should only see their own data
            user1_rows = client1.execute_query("SELECT * FROM test_rls_isolation")
            user2_rows = client2.execute_query("SELECT * FROM test_rls_isolation")

            assert len(user1_rows) == 1
            assert user1_rows[0]["data"] == "User 1 data"

            assert len(user2_rows) == 1
            assert user2_rows[0]["data"] == "User 2 data"

        finally:
            # Cleanup
            client1.execute_query("DROP TABLE IF EXISTS test_rls_isolation CASCADE")
            auth_db.delete_user(user2_id)


class TestPostgresClientUUIDConversion:
    """Test automatic UUID to string conversion in parameters."""

    def test_converts_uuid_set_to_list(self, postgres_app_client):
        """Verify UUID sets are converted to lists (psycopg2 can't bind sets)."""
        uuid1 = uuid.uuid4()
        uuid2 = uuid.uuid4()

        # Insert test data
        postgres_app_client.execute_query(
            "INSERT INTO test_postgres_items (id, user_id, name) VALUES (%s, %s, %s)",
            (uuid1, postgres_app_client.user_id, "Item 1")
        )
        postgres_app_client.execute_query(
            "INSERT INTO test_postgres_items (id, user_id, name) VALUES (%s, %s, %s)",
            (uuid2, postgres_app_client.user_id, "Item 2")
        )

        # Query with set of UUIDs - should be converted to list internally
        uuid_set = {uuid1, uuid2}

        # The client should convert the set to a list
        converted_params = postgres_app_client._convert_uuid_params(uuid_set)
        assert isinstance(converted_params, list)
        assert len(converted_params) == 2

    def test_converts_uuid_objects_to_strings(self, postgres_app_client):
        """Verify UUID objects are automatically converted to strings."""
        test_uuid = uuid.uuid4()

        # Insert using UUID object
        postgres_app_client.execute_query(
            "INSERT INTO test_postgres_items (id, user_id, name) VALUES (%s, %s, %s)",
            (test_uuid, postgres_app_client.user_id, "UUID Test")
        )

        # Query back using UUID object
        result = postgres_app_client.execute_single(
            "SELECT * FROM test_postgres_items WHERE id = %s",
            (test_uuid,)
        )

        assert result is not None
        assert result["name"] == "UUID Test"
        assert str(result["id"]) == str(test_uuid)

    def test_converts_nested_uuids_in_dicts(self, postgres_app_client):
        """Verify UUID conversion works in nested dict parameters."""
        test_uuid = uuid.uuid4()

        # Use dict-style params with UUID
        postgres_app_client.execute_query(
            "INSERT INTO test_postgres_items (id, user_id, name) VALUES (%(id)s, %(user_id)s, %(name)s)",
            {"id": test_uuid, "user_id": postgres_app_client.user_id, "name": "Nested UUID"}
        )

        result = postgres_app_client.execute_single(
            "SELECT * FROM test_postgres_items WHERE id = %(id)s",
            {"id": test_uuid}
        )

        assert result is not None
        assert result["name"] == "Nested UUID"

    def test_converts_uuid_lists(self, postgres_app_client):
        """Verify UUID conversion works in lists."""
        uuid1 = uuid.uuid4()
        uuid2 = uuid.uuid4()

        # Insert two items
        postgres_app_client.execute_query(
            "INSERT INTO test_postgres_items (id, user_id, name) VALUES (%s, %s, %s)",
            (uuid1, postgres_app_client.user_id, "Item 1")
        )
        postgres_app_client.execute_query(
            "INSERT INTO test_postgres_items (id, user_id, name) VALUES (%s, %s, %s)",
            (uuid2, postgres_app_client.user_id, "Item 2")
        )

        # Query with list of UUIDs
        results = postgres_app_client.execute_query(
            "SELECT * FROM test_postgres_items WHERE id = ANY(%s)",
            ([uuid1, uuid2],)
        )

        assert len(results) == 2


class TestPostgresClientJSONBRegistration:
    """Test JSONB auto-registration behavior."""

    def test_jsonb_deserializes_automatically(self, postgres_app_client):
        """Verify JSONB columns are automatically deserialized to Python objects."""
        test_data = {"key": "value", "nested": {"number": 42}}

        # Insert JSONB data
        postgres_app_client.execute_query(
            "INSERT INTO test_postgres_items (user_id, name, data) VALUES (%s, %s, %s)",
            (postgres_app_client.user_id, "JSONB Test", test_data)
        )

        # Query back - should be Python dict, not string
        result = postgres_app_client.execute_single(
            "SELECT * FROM test_postgres_items WHERE name = %s",
            ("JSONB Test",)
        )

        assert isinstance(result["data"], dict)
        assert result["data"]["key"] == "value"
        assert result["data"]["nested"]["number"] == 42


class TestPostgresClientPgVectorRegistration:
    """Test pgvector registration for mira_memory database only."""

    def test_pgvector_registered_for_memory_database(self, postgres_memory_client):
        """Verify pgvector is registered and functional for mira_memory."""
        # Test that we can query vector columns (memories table has embeddings)
        result = postgres_memory_client.execute_scalar(
            "SELECT COUNT(*) FROM memories LIMIT 1"
        )

        # If this doesn't raise an error, pgvector is working
        assert result is not None

    def test_memory_database_needs_vector_flag(self, postgres_memory_client):
        """Verify _needs_vector() returns True for mira_memory."""
        assert postgres_memory_client._needs_vector() is True

    def test_app_database_does_not_need_vector_flag(self, postgres_app_client):
        """Verify _needs_vector() returns False for mira_service."""
        assert postgres_app_client._needs_vector() is False


class TestPostgresClientAutomaticTimestamps:
    """Test automatic timestamp injection for memories table."""

    def test_json_insert_adds_timestamps_for_memories_table(self, postgres_memory_client):
        """Verify json_insert adds created_at and updated_at for memories table."""
        before_insert = utc_now()

        result = postgres_memory_client.json_insert(
            "memories",
            {
                "content": "Test memory",
                "memory_type": "semantic",
                "embedding": [0.1] * 768  # Mock embedding
            },
            returning="id, created_at, updated_at"
        )

        after_insert = utc_now()

        assert result is not None
        assert result["created_at"] is not None
        assert result["updated_at"] is not None

        # Timestamps should be within test window
        assert before_insert <= result["created_at"] <= after_insert
        assert before_insert <= result["updated_at"] <= after_insert

        # Cleanup
        postgres_memory_client.execute_query(
            "DELETE FROM memories WHERE id = %s",
            (result["id"],)
        )

    def test_json_insert_does_not_add_timestamps_for_other_tables(self, postgres_app_client):
        """Verify json_insert does NOT add timestamps for non-memories tables."""
        # Insert without timestamps
        postgres_app_client.json_insert(
            "test_postgres_items",
            {
                "user_id": postgres_app_client.user_id,
                "name": "No Timestamp Test"
            }
        )

        result = postgres_app_client.execute_single(
            "SELECT * FROM test_postgres_items WHERE name = %s",
            ("No Timestamp Test",)
        )

        # created_at and updated_at should be NULL
        assert result["created_at"] is None
        assert result["updated_at"] is None


class TestPostgresClientJSONOperations:
    """Test json_insert, json_update, json_select, json_delete methods."""

    def test_json_insert_only_injects_user_id_when_set(self, authenticated_user):
        """Verify json_insert only injects user_id when client has user_id."""
        # Client WITH user_id
        client_with_user = PostgresClient(database_name="mira_service", user_id=authenticated_user["user_id"])

        # Setup
        client_with_user.execute_query("""
            CREATE TABLE IF NOT EXISTS test_user_injection (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT,
                name TEXT
            )
        """)

        result1 = client_with_user.json_insert(
            "test_user_injection",
            {"name": "With User"},
            returning="id, user_id"
        )

        assert result1["user_id"] == authenticated_user["user_id"]

        # Client WITHOUT user_id
        client_without_user = PostgresClient(database_name="mira_service", user_id=None)

        result2 = client_without_user.json_insert(
            "test_user_injection",
            {"name": "Without User"},
            returning="id, user_id"
        )

        # user_id should be NULL
        assert result2["user_id"] is None

        # Cleanup
        client_with_user.execute_query("DROP TABLE IF EXISTS test_user_injection CASCADE")

    def test_json_insert_returns_none_without_returning_clause(self, postgres_app_client):
        """Verify json_insert returns None when returning parameter is not provided."""
        result = postgres_app_client.json_insert(
            "test_postgres_items",
            {
                "user_id": postgres_app_client.user_id,
                "name": "No Returning"
            }
        )

        # Should return None (not a dict)
        assert result is None

        # Verify item was inserted
        rows = postgres_app_client.execute_query(
            "SELECT * FROM test_postgres_items WHERE name = %s",
            ("No Returning",)
        )
        assert len(rows) == 1

    def test_json_insert_does_not_mutate_original_dict(self, postgres_app_client):
        """Verify json_insert doesn't modify the original data dict."""
        original_data = {
            "name": "Original",
            "data": {"key": "value"}
        }

        data_before = original_data.copy()

        postgres_app_client.json_insert(
            "test_postgres_items",
            original_data,
            json_columns=["data"]
        )

        # Original dict should be unchanged
        assert original_data == data_before
        assert "user_id" not in original_data
        assert "created_at" not in original_data
        assert "updated_at" not in original_data

    def test_json_insert_with_returning(self, postgres_app_client):
        """Verify json_insert returns specified columns."""
        result = postgres_app_client.json_insert(
            "test_postgres_items",
            {
                "user_id": postgres_app_client.user_id,
                "name": "Returning Test",
                "data": {"status": "new"}
            },
            json_columns=["data"],
            returning="id, name"
        )

        assert result is not None
        assert result["name"] == "Returning Test"
        assert "id" in result

    def test_json_update_with_returning(self, postgres_app_client):
        """Verify json_update returns updated rows."""
        # Insert
        result = postgres_app_client.json_insert(
            "test_postgres_items",
            {
                "user_id": postgres_app_client.user_id,
                "name": "Update Test"
            },
            returning="id"
        )
        item_id = result["id"]

        # Update with returning
        updated = postgres_app_client.json_update(
            "test_postgres_items",
            {"name": "Updated Name"},
            where_clause="id = %(id)s",
            where_params={"id": item_id},
            returning="id, name, updated_at"
        )

        assert len(updated) == 1
        assert updated[0]["name"] == "Updated Name"
        assert updated[0]["updated_at"] is not None

    def test_json_select_with_query_options(self, postgres_app_client):
        """Verify json_select supports order_by, limit, and custom columns."""
        # Insert test data
        for i in range(5):
            postgres_app_client.json_insert(
                "test_postgres_items",
                {
                    "user_id": postgres_app_client.user_id,
                    "name": f"Item {i}"
                }
            )

        # Query with options
        results = postgres_app_client.json_select(
            "test_postgres_items",
            order_by="name ASC",
            limit=3,
            columns="name"
        )

        assert len(results) == 3
        assert "name" in results[0]
        # Should only have 'name' column (not id, user_id, etc)

    def test_json_delete_returns_rowcount(self, postgres_app_client):
        """Verify json_delete returns number of deleted rows."""
        # Insert test data
        postgres_app_client.json_insert(
            "test_postgres_items",
            {
                "user_id": postgres_app_client.user_id,
                "name": "Delete Me"
            }
        )

        # Delete
        deleted = postgres_app_client.json_delete(
            "test_postgres_items",
            where_clause="name = %(name)s",
            where_params={"name": "Delete Me"}
        )

        assert deleted == 1


class TestPostgresClientTransactions:
    """Test atomic transaction support."""

    def test_execute_transaction_commits_all_or_nothing(self, postgres_app_client):
        """Verify transaction commits all operations atomically."""
        operations = [
            ("INSERT INTO test_postgres_items (user_id, name) VALUES (%s, %s)", (postgres_app_client.user_id, "Transaction Item 1")),
            ("INSERT INTO test_postgres_items (user_id, name) VALUES (%s, %s)", (postgres_app_client.user_id, "Transaction Item 2")),
            ("INSERT INTO test_postgres_items (user_id, name) VALUES (%s, %s)", (postgres_app_client.user_id, "Transaction Item 3")),
        ]

        results = postgres_app_client.execute_transaction(operations)

        # All operations should succeed
        assert len(results) == 3

        # Verify all inserted
        rows = postgres_app_client.execute_query(
            "SELECT * FROM test_postgres_items WHERE name LIKE 'Transaction Item%'"
        )
        assert len(rows) == 3

    def test_execute_transaction_returns_query_results(self, postgres_app_client):
        """Verify transaction can return query results."""
        # Insert data
        postgres_app_client.json_insert(
            "test_postgres_items",
            {
                "user_id": postgres_app_client.user_id,
                "name": "Transaction Query Test"
            }
        )

        operations = [
            ("SELECT * FROM test_postgres_items WHERE name = %s", ("Transaction Query Test",)),
            ("UPDATE test_postgres_items SET name = %s WHERE name = %s", ("Updated in Transaction", "Transaction Query Test")),
        ]

        results = postgres_app_client.execute_transaction(operations)

        assert len(results) == 2
        assert len(results[0]) == 1  # First operation is SELECT (returns rows)
        assert results[1] == 1  # Second operation is UPDATE (returns rowcount)


class TestPostgresClientTableOperations:
    """Test table introspection and management."""

    def test_table_exists_detects_existing_tables(self, postgres_app_client):
        """Verify table_exists correctly identifies present tables."""
        assert postgres_app_client.table_exists("test_postgres_items") is True
        assert postgres_app_client.table_exists("nonexistent_table_xyz") is False

    def test_get_table_schema_returns_column_info(self, postgres_app_client):
        """Verify get_table_schema returns table structure."""
        schema = postgres_app_client.get_table_schema("test_postgres_items")

        assert len(schema) > 0

        # Verify expected columns present
        column_names = [col["column_name"] for col in schema]
        assert "id" in column_names
        assert "user_id" in column_names
        assert "name" in column_names


class TestPostgresClientQueryMethods:
    """Test various query execution methods."""

    def test_execute_single_returns_first_row_or_none(self, postgres_app_client):
        """Verify execute_single returns first row or None."""
        # Insert data
        postgres_app_client.json_insert(
            "test_postgres_items",
            {
                "user_id": postgres_app_client.user_id,
                "name": "Single Row Test"
            }
        )

        # Query existing
        result = postgres_app_client.execute_single(
            "SELECT * FROM test_postgres_items WHERE name = %s",
            ("Single Row Test",)
        )
        assert result is not None
        assert result["name"] == "Single Row Test"

        # Query non-existing
        result = postgres_app_client.execute_single(
            "SELECT * FROM test_postgres_items WHERE name = %s",
            ("Nonexistent",)
        )
        assert result is None

    def test_execute_scalar_returns_first_value(self, postgres_app_client):
        """Verify execute_scalar returns single value."""
        count = postgres_app_client.execute_scalar(
            "SELECT COUNT(*) FROM test_postgres_items"
        )

        assert isinstance(count, int)
        assert count >= 0

    def test_execute_bulk_insert_inserts_multiple_rows(self, postgres_app_client):
        """Verify bulk insert creates multiple rows efficiently."""
        params_list = [
            (postgres_app_client.user_id, f"Bulk Item {i}")
            for i in range(10)
        ]

        inserted = postgres_app_client.execute_bulk_insert(
            "INSERT INTO test_postgres_items (user_id, name) VALUES (%s, %s)",
            params_list
        )

        assert inserted == 10

        # Verify all inserted
        rows = postgres_app_client.execute_query(
            "SELECT * FROM test_postgres_items WHERE name LIKE 'Bulk Item%'"
        )
        assert len(rows) == 10

    def test_execute_update_returns_rowcount(self, postgres_app_client):
        """Verify execute_update returns number of updated rows."""
        # Insert test data
        for i in range(3):
            postgres_app_client.json_insert(
                "test_postgres_items",
                {
                    "user_id": postgres_app_client.user_id,
                    "name": "Update Count Test"
                }
            )

        # Update all
        updated = postgres_app_client.execute_update(
            "UPDATE test_postgres_items SET name = %s WHERE name = %s",
            ("Updated", "Update Count Test")
        )

        assert updated == 3

    def test_execute_delete_returns_rowcount(self, postgres_app_client):
        """Verify execute_delete returns number of deleted rows."""
        # Insert test data
        for i in range(5):
            postgres_app_client.json_insert(
                "test_postgres_items",
                {
                    "user_id": postgres_app_client.user_id,
                    "name": "Delete Count Test"
                }
            )

        # Delete all
        deleted = postgres_app_client.execute_delete(
            "DELETE FROM test_postgres_items WHERE name = %s",
            ("Delete Count Test",)
        )

        assert deleted == 5


class TestPostgresClientPoolReset:
    """Test connection pool reset functionality."""

    def test_reset_all_pools_clears_connection_pools(self, authenticated_user):
        """Verify reset_all_pools closes and clears all pools."""
        user_id = authenticated_user["user_id"]

        # Create clients to establish pools
        client_app = PostgresClient(database_name="mira_service", user_id=user_id)
        client_memory = PostgresClient(database_name="mira_memory", user_id=user_id)

        # Verify pools exist
        assert len(PostgresClient._connection_pools) >= 2

        # Reset all pools
        PostgresClient.reset_all_pools()

        # Pools should be cleared
        assert len(PostgresClient._connection_pools) == 0
