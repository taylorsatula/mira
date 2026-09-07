"""
Real tests for SQLiteClient - testing the client's unique logic, not SQLite primitives.

Focuses on:
- Automatic user_id injection in json_insert/update/delete/select
- Automatic timestamp management (created_at, updated_at)
- JSON serialization/deserialization in json_* methods
- User isolation enforcement (manual WHERE user_id filtering)
- Singleton pattern (get_sqlite_client caching per user)
- Connection context manager behavior
- Directory creation for database files

Does NOT test basic SQLite operations - those are SQLite's responsibility.
"""

import json
import pytest
import tempfile
import time
from pathlib import Path
from datetime import datetime
from utils.timezone_utils import utc_now, parse_utc_time_string
from clients.sqlite_client import SQLiteClient, get_sqlite_client, _client_cache


@pytest.fixture
def temp_db_path():
    """Provides a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield str(db_path)


@pytest.fixture
def sqlite_client(temp_db_path):
    """Provides a fresh SQLiteClient for testing."""
    client = SQLiteClient(temp_db_path, user_id="test-user-123")

    # Create test table
    client.create_table(
        "test_items",
        [
            "id INTEGER PRIMARY KEY AUTOINCREMENT",
            "user_id TEXT NOT NULL",
            "name TEXT NOT NULL",
            "data TEXT",
            "metadata TEXT",
            "created_at TEXT",
            "updated_at TEXT"
        ]
    )

    yield client


@pytest.fixture
def clean_client_cache():
    """Cleans the global client cache before and after tests."""
    _client_cache.clear()
    yield
    _client_cache.clear()


class TestSQLiteClientAutoUserIdInjection:
    """Test automatic user_id injection in json_* methods."""

    def test_json_insert_auto_injects_user_id(self, sqlite_client):
        """Verify json_insert automatically adds user_id from client."""
        data = {"name": "Test Item", "data": "some data"}

        row_id = sqlite_client.json_insert("test_items", data)

        # Verify user_id was injected
        rows = sqlite_client.execute_query(
            "SELECT * FROM test_items WHERE id = :id",
            {"id": row_id}
        )
        assert len(rows) == 1
        assert rows[0]["user_id"] == "test-user-123"
        assert rows[0]["name"] == "Test Item"

    def test_json_select_auto_filters_by_user_id(self, sqlite_client):
        """Verify json_select automatically filters to current user's data."""
        # Insert data directly with different user_ids
        sqlite_client.execute_query(
            "INSERT INTO test_items (user_id, name, created_at, updated_at) VALUES (:user_id, :name, :created_at, :updated_at)",
            {"user_id": "test-user-123", "name": "Item 1", "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z"}
        )
        sqlite_client.execute_query(
            "INSERT INTO test_items (user_id, name, created_at, updated_at) VALUES (:user_id, :name, :created_at, :updated_at)",
            {"user_id": "other-user-456", "name": "Item 2", "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z"}
        )

        # json_select should only return current user's data
        rows = sqlite_client.json_select("test_items")

        assert len(rows) == 1
        assert rows[0]["user_id"] == "test-user-123"
        assert rows[0]["name"] == "Item 1"

    def test_json_update_auto_filters_by_user_id(self, sqlite_client):
        """Verify json_update only updates current user's data."""
        # Insert data for two different users
        sqlite_client.execute_query(
            "INSERT INTO test_items (user_id, name, created_at, updated_at) VALUES (:user_id, :name, :created_at, :updated_at)",
            {"user_id": "test-user-123", "name": "User 1 Item", "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z"}
        )
        sqlite_client.execute_query(
            "INSERT INTO test_items (user_id, name, created_at, updated_at) VALUES (:user_id, :name, :created_at, :updated_at)",
            {"user_id": "other-user-456", "name": "User 2 Item", "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z"}
        )

        # Try to update all items named "Item"
        updated = sqlite_client.json_update(
            "test_items",
            {"name": "Updated Name"},
            where_clause="name LIKE '%Item%'"
        )

        # Should only update current user's row
        assert updated == 1

        # Verify only user 1's item was updated
        all_rows = sqlite_client.execute_query("SELECT * FROM test_items ORDER BY user_id")
        assert all_rows[0]["name"] == "User 2 Item"  # other-user-456 unchanged
        assert all_rows[1]["name"] == "Updated Name"  # test-user-123 updated

    def test_json_delete_auto_filters_by_user_id(self, sqlite_client):
        """Verify json_delete only deletes current user's data."""
        # Insert data for two users
        sqlite_client.execute_query(
            "INSERT INTO test_items (user_id, name, created_at, updated_at) VALUES (:user_id, :name, :created_at, :updated_at)",
            {"user_id": "test-user-123", "name": "Delete Me", "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z"}
        )
        sqlite_client.execute_query(
            "INSERT INTO test_items (user_id, name, created_at, updated_at) VALUES (:user_id, :name, :created_at, :updated_at)",
            {"user_id": "other-user-456", "name": "Delete Me", "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z"}
        )

        # Try to delete all "Delete Me" items
        deleted = sqlite_client.json_delete(
            "test_items",
            where_clause="name = :name",
            where_params={"name": "Delete Me"}
        )

        # Should only delete current user's row
        assert deleted == 1

        # Verify only user 1's item was deleted
        all_rows = sqlite_client.execute_query("SELECT * FROM test_items")
        assert len(all_rows) == 1
        assert all_rows[0]["user_id"] == "other-user-456"


class TestSQLiteClientAutoTimestamps:
    """Test automatic timestamp injection and updating."""

    def test_json_insert_adds_created_at_and_updated_at(self, sqlite_client):
        """Verify json_insert adds both timestamps."""
        before_insert = utc_now()

        row_id = sqlite_client.json_insert("test_items", {"name": "Timestamped Item"})

        after_insert = utc_now()

        rows = sqlite_client.execute_query(
            "SELECT * FROM test_items WHERE id = :id",
            {"id": row_id}
        )

        assert rows[0]["created_at"] is not None
        assert rows[0]["updated_at"] is not None

        # Parse timestamps
        created_at = parse_utc_time_string(rows[0]["created_at"])
        updated_at = parse_utc_time_string(rows[0]["updated_at"])

        # Verify timestamps are reasonable
        assert before_insert <= created_at <= after_insert
        assert before_insert <= updated_at <= after_insert

    def test_json_update_updates_only_updated_at(self, sqlite_client):
        """Verify json_update changes updated_at but not created_at."""
        # Insert initial record
        row_id = sqlite_client.json_insert("test_items", {"name": "Original"})

        original_rows = sqlite_client.execute_query(
            "SELECT * FROM test_items WHERE id = :id",
            {"id": row_id}
        )
        original_created_at = original_rows[0]["created_at"]

        # Wait a moment to ensure timestamp difference
        time.sleep(0.1)

        # Update the record
        sqlite_client.json_update(
            "test_items",
            {"name": "Updated"},
            where_clause="id = :id",
            where_params={"id": row_id}
        )

        updated_rows = sqlite_client.execute_query(
            "SELECT * FROM test_items WHERE id = :id",
            {"id": row_id}
        )

        # created_at should be unchanged
        assert updated_rows[0]["created_at"] == original_created_at

        # updated_at should be newer
        assert updated_rows[0]["updated_at"] > original_created_at

    def test_json_update_does_not_mutate_original_dict(self, sqlite_client):
        """Verify json_update doesn't modify the original data dict."""
        # Insert test data
        row_id = sqlite_client.json_insert("test_items", {"name": "Test"})

        original_data = {
            "name": "Updated Name",
            "data": {"key": "value"}
        }

        data_before = original_data.copy()

        sqlite_client.json_update(
            "test_items",
            original_data,
            where_clause="id = :id",
            where_params={"id": row_id},
            json_columns=["data"]
        )

        # Original dict should be unchanged
        assert original_data == data_before
        assert "updated_at" not in original_data
        assert "user_id" not in original_data


class TestSQLiteClientJSONSerialization:
    """Test JSON serialization and deserialization in json_* methods."""

    def test_json_insert_serializes_json_columns(self, sqlite_client):
        """Verify json_insert serializes specified columns to JSON strings."""
        data = {
            "name": "Item with JSON",
            "data": {"key": "value", "number": 42},
            "metadata": {"tags": ["tag1", "tag2"], "nested": {"deep": "value"}}
        }

        row_id = sqlite_client.json_insert(
            "test_items",
            data,
            json_columns=["data", "metadata"]
        )

        # Check raw database storage (should be JSON strings)
        rows = sqlite_client.execute_query(
            "SELECT * FROM test_items WHERE id = :id",
            {"id": row_id}
        )

        # Stored as JSON strings
        assert isinstance(rows[0]["data"], str)
        assert isinstance(rows[0]["metadata"], str)

        # Verify they're valid JSON
        parsed_data = json.loads(rows[0]["data"])
        assert parsed_data["key"] == "value"
        assert parsed_data["number"] == 42

    def test_json_insert_skips_none_values_in_json_columns(self, sqlite_client):
        """Verify json_insert does not serialize None values in json_columns."""
        data = {
            "name": "Item with None",
            "data": {"key": "value"},
            "metadata": None  # This should NOT be serialized
        }

        row_id = sqlite_client.json_insert(
            "test_items",
            data,
            json_columns=["data", "metadata"]
        )

        # Check raw database storage
        rows = sqlite_client.execute_query(
            "SELECT * FROM test_items WHERE id = :id",
            {"id": row_id}
        )

        # data should be serialized JSON string
        assert isinstance(rows[0]["data"], str)

        # metadata should be None (not serialized to string "null")
        assert rows[0]["metadata"] is None

    def test_json_insert_does_not_mutate_original_dict(self, sqlite_client):
        """Verify json_insert doesn't modify the original data dict."""
        original_data = {
            "name": "Original",
            "data": {"key": "value"}
        }

        # Make a copy to compare later
        data_before = original_data.copy()

        sqlite_client.json_insert(
            "test_items",
            original_data,
            json_columns=["data"]
        )

        # Original dict should be unchanged
        assert original_data == data_before
        assert "user_id" not in original_data
        assert "created_at" not in original_data
        assert "updated_at" not in original_data

    def test_json_select_deserializes_json_columns(self, sqlite_client):
        """Verify json_select deserializes JSON strings back to objects."""
        # Insert with JSON serialization
        data = {
            "name": "JSON Item",
            "data": {"list": [1, 2, 3], "bool": True},
            "metadata": {"count": 10}
        }

        sqlite_client.json_insert(
            "test_items",
            data,
            json_columns=["data", "metadata"]
        )

        # Retrieve with deserialization
        rows = sqlite_client.json_select(
            "test_items",
            where_clause="name = :name",
            where_params={"name": "JSON Item"},
            json_columns=["data", "metadata"]
        )

        assert len(rows) == 1

        # Should be deserialized back to dicts
        assert isinstance(rows[0]["data"], dict)
        assert isinstance(rows[0]["metadata"], dict)
        assert rows[0]["data"]["list"] == [1, 2, 3]
        assert rows[0]["data"]["bool"] is True
        assert rows[0]["metadata"]["count"] == 10

    def test_json_update_serializes_json_columns(self, sqlite_client):
        """Verify json_update serializes JSON data on update."""
        # Insert initial data
        row_id = sqlite_client.json_insert(
            "test_items",
            {"name": "Update JSON", "data": {"version": 1}},
            json_columns=["data"]
        )

        # Update with new JSON data
        sqlite_client.json_update(
            "test_items",
            {"data": {"version": 2, "updated": True}},
            where_clause="id = :id",
            where_params={"id": row_id},
            json_columns=["data"]
        )

        # Retrieve and verify
        rows = sqlite_client.json_select(
            "test_items",
            where_clause="id = :id",
            where_params={"id": row_id},
            json_columns=["data"]
        )

        assert rows[0]["data"]["version"] == 2
        assert rows[0]["data"]["updated"] is True

    def test_json_select_handles_invalid_json_gracefully(self, sqlite_client):
        """Verify json_select returns None for invalid JSON."""
        # Insert invalid JSON directly
        sqlite_client.execute_query(
            "INSERT INTO test_items (user_id, name, data, created_at, updated_at) VALUES (:user_id, :name, :data, :created_at, :updated_at)",
            {"user_id": "test-user-123", "name": "Bad JSON", "data": "not valid json{", "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z"}
        )

        # Should not crash, should return None for the column
        rows = sqlite_client.json_select(
            "test_items",
            where_clause="name = :name",
            where_params={"name": "Bad JSON"},
            json_columns=["data"]
        )

        assert len(rows) == 1
        assert rows[0]["data"] is None


class TestSQLiteClientUserIsolation:
    """Test that different users can't access each other's data."""

    def test_different_users_see_only_their_own_data(self, temp_db_path):
        """Verify user isolation through different client instances."""
        # Create two clients for different users, same database
        user1_client = SQLiteClient(temp_db_path, user_id="user-1")
        user2_client = SQLiteClient(temp_db_path, user_id="user-2")

        # Create table
        user1_client.create_table(
            "shared_items",
            [
                "id INTEGER PRIMARY KEY AUTOINCREMENT",
                "user_id TEXT NOT NULL",
                "name TEXT NOT NULL",
                "created_at TEXT",
                "updated_at TEXT"
            ]
        )

        # User 1 inserts data
        user1_client.json_insert("shared_items", {"name": "User 1 Item"})

        # User 2 inserts data
        user2_client.json_insert("shared_items", {"name": "User 2 Item"})

        # Each user should only see their own data
        user1_rows = user1_client.json_select("shared_items")
        user2_rows = user2_client.json_select("shared_items")

        assert len(user1_rows) == 1
        assert user1_rows[0]["name"] == "User 1 Item"
        assert user1_rows[0]["user_id"] == "user-1"

        assert len(user2_rows) == 1
        assert user2_rows[0]["name"] == "User 2 Item"
        assert user2_rows[0]["user_id"] == "user-2"

    def test_user_cannot_update_other_users_data(self, temp_db_path):
        """Verify updates are isolated to current user."""
        user1_client = SQLiteClient(temp_db_path, user_id="user-1")
        user2_client = SQLiteClient(temp_db_path, user_id="user-2")

        # Create table
        user1_client.create_table(
            "shared_items",
            [
                "id INTEGER PRIMARY KEY AUTOINCREMENT",
                "user_id TEXT NOT NULL",
                "name TEXT NOT NULL",
                "created_at TEXT",
                "updated_at TEXT"
            ]
        )

        # User 1 creates item
        user1_client.json_insert("shared_items", {"name": "User 1 Item"})

        # User 2 tries to update ALL items
        updated = user2_client.json_update(
            "shared_items",
            {"name": "Hacked!"},
            where_clause="1=1"  # Try to update everything
        )

        # Should update 0 rows (no rows match user_id = user-2)
        assert updated == 0

        # Verify user 1's data is unchanged
        user1_rows = user1_client.json_select("shared_items")
        assert user1_rows[0]["name"] == "User 1 Item"

    def test_user_cannot_delete_other_users_data(self, temp_db_path):
        """Verify deletes are isolated to current user."""
        user1_client = SQLiteClient(temp_db_path, user_id="user-1")
        user2_client = SQLiteClient(temp_db_path, user_id="user-2")

        # Create table
        user1_client.create_table(
            "shared_items",
            [
                "id INTEGER PRIMARY KEY AUTOINCREMENT",
                "user_id TEXT NOT NULL",
                "name TEXT NOT NULL",
                "created_at TEXT",
                "updated_at TEXT"
            ]
        )

        # User 1 creates item
        user1_client.json_insert("shared_items", {"name": "User 1 Item"})

        # User 2 tries to delete ALL items
        deleted = user2_client.json_delete(
            "shared_items",
            where_clause="1=1"
        )

        # Should delete 0 rows
        assert deleted == 0

        # Verify user 1's data still exists
        user1_rows = user1_client.json_select("shared_items")
        assert len(user1_rows) == 1


class TestSQLiteClientSingletonPattern:
    """Test get_sqlite_client singleton caching."""

    def test_same_user_and_path_returns_same_instance(self, temp_db_path, clean_client_cache):
        """Verify get_sqlite_client returns cached instance for same user+path."""
        client1 = get_sqlite_client(temp_db_path, "user-123")
        client2 = get_sqlite_client(temp_db_path, "user-123")

        assert client1 is client2

    def test_different_users_get_different_instances(self, temp_db_path, clean_client_cache):
        """Verify different users get different client instances."""
        client1 = get_sqlite_client(temp_db_path, "user-1")
        client2 = get_sqlite_client(temp_db_path, "user-2")

        assert client1 is not client2
        assert client1.user_id == "user-1"
        assert client2.user_id == "user-2"

    def test_different_paths_get_different_instances(self, clean_client_cache):
        """Verify different database paths get different instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = str(Path(tmpdir) / "db1.db")
            path2 = str(Path(tmpdir) / "db2.db")

            client1 = get_sqlite_client(path1, "user-123")
            client2 = get_sqlite_client(path2, "user-123")

            assert client1 is not client2
            assert client1.db_path == path1
            assert client2.db_path == path2


class TestSQLiteClientConnectionManagement:
    """Test connection context manager behavior."""

    def test_connection_context_manager_closes_connection(self, sqlite_client):
        """Verify connection is closed after context manager exits."""
        # Get a connection
        with sqlite_client.get_connection() as conn:
            # Connection should be open
            conn.execute("SELECT 1")
            conn_obj = conn

        # Connection should be closed after exiting context
        # Attempting to use closed connection should raise
        with pytest.raises(Exception):
            conn_obj.execute("SELECT 1")

    def test_connection_supports_dict_style_access(self, sqlite_client):
        """Verify Row factory provides dict-like access."""
        sqlite_client.json_insert("test_items", {"name": "Test Dict Access"})

        with sqlite_client.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM test_items WHERE user_id = ?", ("test-user-123",))
            row = cursor.fetchone()

            # Should support dict-style access
            assert row["name"] == "Test Dict Access"
            assert row["user_id"] == "test-user-123"

            # Should also support index access (Row feature)
            assert len(row) > 0


class TestSQLiteClientBulkOperations:
    """Test bulk insert operations."""

    def test_execute_bulk_insert_inserts_multiple_rows(self, sqlite_client):
        """Verify bulk insert creates multiple rows efficiently."""
        params_list = [
            {"user_id": "test-user-123", "name": "Item 1", "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z"},
            {"user_id": "test-user-123", "name": "Item 2", "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z"},
            {"user_id": "test-user-123", "name": "Item 3", "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z"},
        ]

        query = """
        INSERT INTO test_items (user_id, name, created_at, updated_at)
        VALUES (:user_id, :name, :created_at, :updated_at)
        """

        rowcount = sqlite_client.execute_bulk_insert(query, params_list)

        assert rowcount == 3

        # Verify all rows inserted
        rows = sqlite_client.json_select("test_items")
        assert len(rows) == 3


class TestSQLiteClientTableOperations:
    """Test table creation and introspection."""

    def test_create_table_creates_table(self, temp_db_path):
        """Verify create_table actually creates the table."""
        client = SQLiteClient(temp_db_path, "test-user")

        client.create_table(
            "new_table",
            [
                "id INTEGER PRIMARY KEY",
                "name TEXT NOT NULL",
                "value INTEGER"
            ]
        )

        assert client.table_exists("new_table")

    def test_create_table_if_not_exists_is_idempotent(self, temp_db_path):
        """Verify create_table with if_not_exists can be called multiple times."""
        client = SQLiteClient(temp_db_path, "test-user")

        # Create table twice
        client.create_table("idempotent_table", ["id INTEGER PRIMARY KEY"])
        client.create_table("idempotent_table", ["id INTEGER PRIMARY KEY"])

        # Should succeed without error
        assert client.table_exists("idempotent_table")

    def test_table_exists_detects_existing_tables(self, sqlite_client):
        """Verify table_exists correctly identifies present tables."""
        assert sqlite_client.table_exists("test_items")
        assert not sqlite_client.table_exists("nonexistent_table")

    def test_get_table_schema_returns_column_info(self, sqlite_client):
        """Verify get_table_schema returns table structure."""
        schema = sqlite_client.get_table_schema("test_items")

        assert len(schema) > 0

        # Verify column names present
        column_names = [col["name"] for col in schema]
        assert "id" in column_names
        assert "user_id" in column_names
        assert "name" in column_names


class TestSQLiteClientDirectoryCreation:
    """Test automatic directory creation for database files."""

    def test_ensures_parent_directory_exists(self):
        """Verify client creates parent directories for database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Nested path that doesn't exist
            nested_path = Path(tmpdir) / "level1" / "level2" / "test.db"

            client = SQLiteClient(str(nested_path), "test-user")

            # Parent directories should be created
            assert nested_path.parent.exists()
            assert nested_path.parent.is_dir()


class TestSQLiteClientQueryOperations:
    """Test basic query, update, delete operations."""

    def test_execute_query_returns_dict_results(self, sqlite_client):
        """Verify execute_query returns list of dicts."""
        sqlite_client.json_insert("test_items", {"name": "Query Test"})

        results = sqlite_client.execute_query(
            "SELECT * FROM test_items WHERE user_id = :user_id",
            {"user_id": "test-user-123"}
        )

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert results[0]["name"] == "Query Test"

    def test_execute_update_returns_rowcount(self, sqlite_client):
        """Verify execute_update returns number of rows affected."""
        sqlite_client.json_insert("test_items", {"name": "Update Test 1"})
        sqlite_client.json_insert("test_items", {"name": "Update Test 2"})

        rowcount = sqlite_client.execute_update(
            "UPDATE test_items SET name = :new_name WHERE user_id = :user_id",
            {"new_name": "Updated", "user_id": "test-user-123"}
        )

        assert rowcount == 2

    def test_execute_delete_returns_rowcount(self, sqlite_client):
        """Verify execute_delete returns number of rows deleted."""
        sqlite_client.json_insert("test_items", {"name": "Delete Test 1"})
        sqlite_client.json_insert("test_items", {"name": "Delete Test 2"})

        rowcount = sqlite_client.execute_delete(
            "DELETE FROM test_items WHERE user_id = :user_id",
            {"user_id": "test-user-123"}
        )

        assert rowcount == 2


class TestSQLiteClientQueryOptions:
    """Test json_select query options (order_by, limit)."""

    def test_json_select_with_order_by(self, sqlite_client):
        """Verify json_select respects order_by parameter."""
        sqlite_client.json_insert("test_items", {"name": "Zebra"})
        sqlite_client.json_insert("test_items", {"name": "Apple"})
        sqlite_client.json_insert("test_items", {"name": "Mango"})

        rows = sqlite_client.json_select("test_items", order_by="name ASC")

        assert rows[0]["name"] == "Apple"
        assert rows[1]["name"] == "Mango"
        assert rows[2]["name"] == "Zebra"

    def test_json_select_with_limit(self, sqlite_client):
        """Verify json_select respects limit parameter."""
        for i in range(10):
            sqlite_client.json_insert("test_items", {"name": f"Item {i}"})

        rows = sqlite_client.json_select("test_items", limit=3)

        assert len(rows) == 3

    def test_json_select_with_where_clause(self, sqlite_client):
        """Verify json_select applies custom where clause."""
        sqlite_client.json_insert("test_items", {"name": "Active Item", "data": "active"})
        sqlite_client.json_insert("test_items", {"name": "Inactive Item", "data": "inactive"})

        rows = sqlite_client.json_select(
            "test_items",
            where_clause="data = :status",
            where_params={"status": "active"}
        )

        assert len(rows) == 1
        assert rows[0]["name"] == "Active Item"
