"""
SQLite test database fixture that wraps the production SQLiteClient.

This fixture provides a thin wrapper around the battle-tested SQLiteClient,
adding only test-specific lifecycle management (setup/teardown).
"""

import tempfile
import shutil
from pathlib import Path
from typing import Optional, Generator, List
import pytest
import logging

from clients.sqlite_client import SQLiteClient

logger = logging.getLogger(__name__)


class SQLiteTestFixture:
    """
    Minimal wrapper around SQLiteClient for test lifecycle management.
    
    This fixture:
    - Creates a temporary database in a temp directory
    - Loads SQL schema files
    - Provides the real SQLiteClient for testing
    - Cleans up after tests (unless persist flag is set)
    """
    
    def __init__(self, temp_dir: Path, user_id: str = "test-user-123", persist: bool = False):
        """
        Initialize the test fixture.
        
        Args:
            temp_dir: Temporary directory for database files
            user_id: User ID for test isolation (default: test-user-123)
            persist: If True, don't delete database after tests (useful for debugging)
        """
        self.temp_dir = temp_dir
        self.user_id = user_id
        self.persist = persist
        self.db_path = str(temp_dir / "userdata.db")
        self.client: Optional[SQLiteClient] = None
        
    def load_schema(self, schema_file: Path):
        """
        Load a SQL schema file into the database.
        
        Args:
            schema_file: Path to SQL file containing CREATE TABLE statements
        """
        if not self.client:
            raise RuntimeError("Database not initialized. Call setup() first.")
            
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
            
        # Use the client's connection to execute the schema
        with self.client.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
            
        logger.info(f"Loaded schema from {schema_file}")
        
    def setup(self) -> SQLiteClient:
        """
        Create and initialize the test database.
        
        Returns:
            SQLiteClient instance ready for use
        """
        # Create temp directory if it doesn't exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove existing database if present
        db_file = Path(self.db_path)
        if db_file.exists():
            db_file.unlink()
            
        # Create SQLiteClient instance (it will create the database)
        self.client = SQLiteClient(self.db_path, self.user_id)
        logger.info(f"Created test database at {self.db_path} for user {self.user_id}")
        
        return self.client
        
    def teardown(self):
        """Clean up the test database unless persist flag is set."""
        self.client = None  # Let garbage collection close any connections
        
        if not self.persist and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up test database at {self.temp_dir}")
        elif self.persist:
            logger.info(f"Persisted test database at {self.db_path}")


@pytest.fixture
def sqlite_test_db(tmp_path, request) -> Generator[SQLiteClient, None, None]:
    """
    Pytest fixture that provides a temporary SQLite database for testing.
    
    The database is automatically cleaned up after the test unless the
    persist_db marker is used.
    
    Usage:
        def test_something(sqlite_test_db):
            db = sqlite_test_db
            # Use db.json_insert(), db.json_select(), etc.
            # All SQLiteClient methods are available
            
    To persist database for debugging:
        @pytest.mark.persist_db
        def test_something(sqlite_test_db):
            # Database won't be deleted after test
            
    To use a custom user_id:
        @pytest.mark.user_id("custom-user-456")
        def test_something(sqlite_test_db):
            # Database will use custom-user-456 as user_id
            
    To load specific schema files:
        @pytest.mark.schema_files(['pager_schema.sql', 'other_schema.sql'])
        def test_something(sqlite_test_db):
            # Will load these schema files from tests/fixtures/
    """
    # Check for markers
    persist = request.node.get_closest_marker('persist_db') is not None
    
    user_id_marker = request.node.get_closest_marker('user_id')
    user_id = user_id_marker.args[0] if user_id_marker else "test-user-123"
    
    schema_files_marker = request.node.get_closest_marker('schema_files')
    schema_files = schema_files_marker.args[0] if schema_files_marker else []
    
    # Create fixture
    fixture = SQLiteTestFixture(tmp_path, user_id=user_id, persist=persist)
    client = fixture.setup()
    
    # Load requested schema files
    if schema_files:
        schema_dir = Path(__file__).parent
        for schema_file in schema_files:
            schema_path = schema_dir / schema_file
            if schema_path.exists():
                fixture.load_schema(schema_path)
            else:
                logger.warning(f"Schema file not found: {schema_path}")
    
    # Provide SQLiteClient to test
    yield client
    
    # Cleanup
    fixture.teardown()


@pytest.fixture
def sqlite_test_db_factory(tmp_path, request) -> Generator:
    """
    Factory fixture for creating multiple test databases in a single test.
    
    Usage:
        def test_multiple_users(sqlite_test_db_factory):
            db1 = sqlite_test_db_factory("user-1", ["schema1.sql"])
            db2 = sqlite_test_db_factory("user-2", ["schema2.sql"])
            # Each database is isolated
    """
    persist = request.node.get_closest_marker('persist_db') is not None
    fixtures = []
    
    def create_db(user_id: str = "test-user-123", 
                  schema_files: Optional[List[str]] = None) -> SQLiteClient:
        """Create a new test database with given parameters."""
        # Create unique temp dir for this database
        db_dir = tmp_path / f"db_{user_id}_{len(fixtures)}"
        
        fixture = SQLiteTestFixture(db_dir, user_id=user_id, persist=persist)
        fixtures.append(fixture)
        client = fixture.setup()
        
        # Load schema files if provided
        if schema_files:
            schema_dir = Path(__file__).parent
            for schema_file in schema_files:
                schema_path = schema_dir / schema_file
                if schema_path.exists():
                    fixture.load_schema(schema_path)
        
        return client
    
    yield create_db
    
    # Cleanup all created databases
    for fixture in fixtures:
        fixture.teardown()