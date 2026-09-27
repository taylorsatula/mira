"""
SQLite client with explicit connection management and dict-style results.

Simple tool storage with raw SQL for performance
and explicit memory management.
"""

import sqlite3
import logging
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Global cache for SQLiteClient instances per database path
_client_cache: Dict[str, 'SQLiteClient'] = {}

class SQLiteClient:
    """
    Raw SQL client with explicit per-request connections (no pooling).

    IMPORTANT: SQLite does not support Row Level Security (RLS) like PostgreSQL.
    Each user has their own separate SQLite database file, and queries must
    filter by user_id manually to maintain consistency with the PostgreSQL patterns
    and to prevent accidental cross-user data access if database paths are misconfigured.

    This manual filtering is NOT redundant - it's the ONLY mechanism for user isolation
    in SQLite, unlike PostgreSQL where it would be redundant with RLS policies.
    The client itself does no scoping; it is raw SQL over the given db_path.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_db_directory()
        logger.debug(f"SQLite client initialized: {db_path}")

    def _ensure_db_directory(self):
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Creates fresh connection per request with sqlite3.Row factory for dict-like access."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            logger.error(f"SQLite connection failed for {self.db_path}: {e}", exc_info=True)
            raise
        finally:
            if 'conn' in locals():
                conn.close()

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or {})

            if cursor.description:
                return [dict(row) for row in cursor.fetchall()]
            else:
                conn.commit()
                return []

    def execute_insert(self, query: str, params: Optional[Dict] = None) -> str:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or {})
            conn.commit()
            return str(cursor.lastrowid)

    def execute_update(self, query: str, params: Optional[Dict] = None) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or {})
            conn.commit()
            return cursor.rowcount

    def execute_delete(self, query: str, params: Optional[Dict] = None) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or {})
            conn.commit()
            return cursor.rowcount

    def create_table(self, table_name: str, columns: List[str], if_not_exists: bool = True) -> None:
        if_not_exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        columns_str = ", ".join(columns)

        query = f"""
        CREATE TABLE {if_not_exists_clause}{table_name} (
            {columns_str}
        )
        """

        self.execute_query(query)
        logger.debug(f"Table created: {table_name}")


def get_sqlite_client(db_path: str) -> SQLiteClient:
    """
    Get a singleton SQLiteClient instance for the given database path.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        SQLiteClient instance (singleton per database path)
    """
    if db_path not in _client_cache:
        _client_cache[db_path] = SQLiteClient(db_path)
        logger.debug("Created new SQLiteClient singleton")
    else:
        logger.debug("Reusing existing SQLiteClient singleton")

    return _client_cache[db_path]
