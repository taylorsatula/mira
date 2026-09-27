"""
Database session management for LT_Memory with proper lifecycle and transaction support.

Provides session-based database access with connection pooling, automatic cleanup,
and transaction management following MIRA's architectural patterns.
"""

import time
import logging
import threading
import atexit
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Union, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout
from pgvector.psycopg import register_vector
from clients.postgres_client import PostgresClient
from config import config


logger = logging.getLogger(__name__)

# Module-level singleton instance
_shared_session_manager = None
_singleton_lock = threading.RLock()


def get_shared_session_manager() -> 'LTMemorySessionManager':
    """
    Get the shared session manager instance (singleton pattern).
    
    This ensures all components share the same connection pools,
    preventing connection exhaustion. Industry best practice for
    expensive resources like database connection pools.
    
    Returns:
        Shared LTMemorySessionManager instance
    """
    global _shared_session_manager
    if _shared_session_manager is None:
        with _singleton_lock:
            # Double-check pattern for thread safety
            if _shared_session_manager is None:
                logger.info("Creating shared session manager singleton")
                _shared_session_manager = LTMemorySessionManager()
    return _shared_session_manager


class LTMemorySessionManager:
    """
    Session manager for LT_Memory database operations.
    
    Manages connection pools with proper lifecycle, integrates with MIRA's
    authentication patterns, and provides transactional session access.
    """
    
    def __init__(self):
        """Initialize session manager with cleanup registration."""
        self._pools: Dict[str, ConnectionPool] = {}
        self._lock = threading.RLock()
        # Register cleanup to prevent connection leaks
        atexit.register(self._cleanup_all_pools)
    
    def get_session(self, user_id: str) -> 'LTMemorySession':
        """
        Get a user-scoped database session.

        Args:
            user_id: User ID (required for proper data isolation)

        Returns:
            LTMemorySession configured for the specified user

        Raises:
            ValueError: If user_id is not provided
        """
        if not user_id:
            raise ValueError("user_id is required for database operations")

        pool = self._get_or_create_pool('mira_service')
        return LTMemorySession(pool, user_id)

    def get_admin_session(self) -> 'AdminSession':
        """
        Get a database session for admin/cross-user operations.

        Use this for queries that need to access data across multiple users
        (e.g., polling batches for all users, user management operations).
        Uses mira_admin role which has BYPASSRLS privilege.

        Returns:
            AdminSession for mira_service database
        """
        pool = self._get_or_create_pool('mira_service_admin')
        return AdminSession(pool)

    def _get_or_create_pool(self, database_name: str) -> ConnectionPool:
        """
        Get existing pool or create new one for database.

        Args:
            database_name: Name of database in Vault configuration
                          'mira_service' = regular user access with RLS
                          'mira_service_admin' = admin access with BYPASSRLS

        Returns:
            Connection pool for the specified database
        """
        with self._lock:
            if database_name not in self._pools:
                try:
                    from clients.vault_client import get_database_url

                    # Determine if this is an admin pool
                    is_admin = database_name == 'mira_service_admin'
                    actual_db_name = 'mira_service'  # Both connect to same database

                    conninfo = PostgresClient._parse_database_url(
                        get_database_url(actual_db_name, admin=is_admin)
                    )

                    # Create pool with reasonable size limits
                    # Industry best practice: keep pools small, let them queue
                    pool = ConnectionPool(
                        conninfo=conninfo,
                        min_size=config.database.session_pool_min,
                        max_size=config.database.session_pool_max,
                        timeout=30,
                        max_lifetime=3600,  # Recycle connections after 1 hour
                        max_idle=300,       # Close idle connections after 5 minutes
                        check=ConnectionPool.check_connection,
                        kwargs={'options': f'-c statement_timeout={config.database.statement_timeout_ms}'}
                    )

                    self._pools[database_name] = pool

                except Exception as e:
                    logger.error(f"Failed to create connection pool for {database_name}: {e}")
                    raise ValueError(f"Database pool creation failed: {str(e)}")

            return self._pools[database_name]
    
    def _cleanup_all_pools(self):
        """Clean up all connection pools on shutdown."""
        with self._lock:
            for db_name, pool in self._pools.items():
                try:
                    pool.close(timeout=0)  # Workers are daemon threads — no need to wait
                except Exception as e:
                    logger.error(f"Error closing pool for {db_name}: {e}")

            self._pools.clear()
    
    def cleanup(self):
        """Explicit cleanup method following MIRA patterns."""
        self._cleanup_all_pools()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about current connection pools."""
        stats = {}
        with self._lock:
            for db_name, pool in self._pools.items():
                # Note: psycopg3 pools don't expose detailed stats
                # but we can at least show configuration
                stats[db_name] = {
                    "min_size": getattr(pool, 'min_size', 'unknown'),
                    "max_size": getattr(pool, 'max_size', 'unknown'),
                    "closed": getattr(pool, 'closed', False)
                }
        return stats


class LTMemorySession:
    """
    Database session with transaction support and user context.
    
    Provides connection lifecycle management, automatic pgvector registration,
    user context configuration, and nested transaction support.
    """
    
    def __init__(self, pool: ConnectionPool, user_id: str):
        """
        Initialize session with pool and user context.
        
        Args:
            pool: Connection pool to acquire connections from
            user_id: User ID for row-level security context
        """
        self.pool = pool
        self.user_id = user_id
        self._conn = None
        self._transaction_depth = 0
        self._closed = False
    
    def __enter__(self):
        """Enter session context - acquire and setup connection."""
        self._conn = self._acquire_connection_with_timeout()
        try:
            self._setup_connection()
        except:
            # Setup failed (e.g. a connection that died while idle in the
            # pool). Return the connection so the pool slot is not stranded,
            # then neutralize the session so a later stray __exit__ cannot
            # touch an already-returned connection. The exception itself
            # propagates unchanged (required-infrastructure doctrine).
            self.pool.putconn(self._conn)
            self._conn = None
            self._closed = True
            raise
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit session context - handle transactions and return connection."""
        if self._closed:
            return

        try:
            if exc_type:
                # Rollback on exception regardless of transaction depth
                self._conn.rollback()
            else:
                # Always commit on success (matches AdminSession behavior)
                # PostgreSQL connection pool requires explicit commits
                self._conn.commit()
        except Exception as e:
            if exc_type:
                # An exception is already propagating to the caller; a failed
                # rollback must not mask it. Log for observability only.
                logger.error(f"Error during session cleanup: {e}")
            else:
                # The commit failed: surface it. Every caller that reports
                # success or runs follow-ups after this block must see the
                # failure, never a clean exit. Log context, then re-raise the
                # original error so its type (e.g. a psycopg errors subclass)
                # survives for callers that classify it.
                logger.error(f"Database commit failed during session cleanup: {e}")
                raise
        finally:
            if self._conn:
                self.pool.putconn(self._conn)
                self._conn = None
            self._closed = True
    
    @contextmanager
    def transaction(self):
        """
        Transaction context manager with nested transaction support.
        
        Supports nested transactions using savepoints for inner transactions.
        """
        if self._transaction_depth == 0:
            # Start new top-level transaction (PostgreSQL does this automatically)
            pass
        else:
            # Create savepoint for nested transaction
            savepoint_name = f"sp_{self._transaction_depth}"
            self._execute(f"SAVEPOINT {savepoint_name}")
        
        self._transaction_depth += 1
        
        try:
            yield
            
            if self._transaction_depth == 1:
                # Commit top-level transaction
                self._conn.commit()
            else:
                # Release savepoint for nested transaction
                savepoint_name = f"sp_{self._transaction_depth - 1}"
                self._execute(f"RELEASE SAVEPOINT {savepoint_name}")
                
        except Exception:
            if self._transaction_depth == 1:
                # Rollback top-level transaction
                self._conn.rollback()
            else:
                # Rollback to savepoint for nested transaction
                savepoint_name = f"sp_{self._transaction_depth - 1}"
                self._execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            raise
            
        finally:
            self._transaction_depth -= 1
    
    def execute_query(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> List[Dict]:
        """
        Execute query and return results as list of dictionaries.

        Args:
            query: SQL query string
            params: Optional parameters (dict for named, tuple for positional)

        Returns:
            List of row dictionaries
        """
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)

            if cur.description:
                return cur.fetchall()
            else:
                return []
    
    def execute_single(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> Optional[Dict]:
        """
        Execute query and return single result or None.
        
        Args:
            query: SQL query string
            params: Optional parameters
            
        Returns:
            Single row dictionary or None
        """
        results = self.execute_query(query, params)
        return results[0] if results else None
    
    def execute_bulk_insert(self, query: str, params_list: List[Union[Dict, Tuple]]) -> int:
        """
        Execute bulk insert and return number of rows affected.
        
        Args:
            query: SQL insert query
            params_list: List of parameter sets
            
        Returns:
            Number of rows inserted
        """
        with self._conn.cursor() as cur:
            cur.executemany(query, params_list)
            return cur.rowcount
    
    def execute_update(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> int:
        """
        Execute update query and return number of rows affected.
        
        Args:
            query: SQL update query
            params: Optional parameters
            
        Returns:
            Number of rows updated
        """
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            return cur.rowcount
    
    def execute_delete(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> int:
        """
        Execute delete query and return number of rows affected.
        
        Args:
            query: SQL delete query
            params: Optional parameters
            
        Returns:
            Number of rows deleted
        """
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            return cur.rowcount
    
    def _acquire_connection_with_timeout(self) -> psycopg.Connection:
        """
        Acquire connection from pool with timeout and retry logic.
        
        Returns:
            Database connection
            
        Raises:
            ValueError: If connection cannot be acquired within timeout
        """
        try:
            start_time = time.time()
            timeout = 30  # seconds
            retry_interval = 0.1  # seconds
            
            while time.time() - start_time < timeout:
                try:
                    conn = self.pool.getconn()
                    if conn and not conn.closed:
                        return conn
                    elif conn and conn.closed:
                        # Return bad connection and try again
                        self.pool.putconn(conn)
                except PoolTimeout:
                    # Pool exhausted, wait and retry
                    pass

                time.sleep(retry_interval)

            # Timeout reached
            logger.error(f"Database connection timeout after {timeout}s - pool may be exhausted")
            raise ValueError(f"Database connection timeout after {timeout}s - pool may be exhausted")
        except Exception as e:
            logger.error(f"Error acquiring database connection: {e}")
            raise
    
    def _setup_connection(self):
        """
        Setup connection with pgvector, UUID support, and user context.

        Registers pgvector extension, UUID type adapter, and sets user context
        for row-level security.
        """
        # Register pgvector extension on this connection
        register_vector(self._conn)

        # Set user context for row-level security using set_config()
        # Using set_config() instead of SET because it's a proper PostgreSQL
        # function that works correctly with psycopg3 parameter binding.
        # Third parameter (false) = session-wide, not transaction-local.
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('app.current_user_id', %s, false)",
                (str(self.user_id),)
            )
            
    
    def _execute(self, sql: str):
        """
        Execute SQL statement without returning results.
        
        Args:
            sql: SQL statement to execute
        """
        with self._conn.cursor() as cur:
            cur.execute(sql)


class AdminSession:
    """
    Database session for admin/cross-user operations.

    Does not set RLS user context, allowing queries across all users.
    Use for system-level operations like batch polling, user management, etc.
    """

    def __init__(self, pool: ConnectionPool):
        """Initialize session with connection pool."""
        self.pool = pool
        self._conn = None
        self._closed = False

    def __enter__(self):
        """Enter session context - acquire and setup connection."""
        self._conn = self._acquire_connection_with_timeout()
        try:
            self._setup_connection()
        except:
            # Same leak guard as LTMemorySession.__enter__: return the
            # connection to the pool on any setup failure, mark the session
            # closed so a later stray __exit__ is a no-op, and let the
            # exception propagate to the caller.
            self.pool.putconn(self._conn)
            self._conn = None
            self._closed = True
            raise
        return self

    def _setup_connection(self):
        """Setup connection with pgvector support."""
        # Register pgvector extension
        register_vector(self._conn)

        # Don't set app.current_user_id at all - let it remain undefined
        # RLS policies check for NULL context to allow admin cross-user queries

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit session context - return connection."""
        if self._closed:
            return

        try:
            if exc_type:
                self._conn.rollback()
            else:
                self._conn.commit()
        except Exception as e:
            if exc_type:
                # An exception is already propagating; a failed rollback must
                # not mask it. Log for observability only.
                logger.error(f"Error during admin session cleanup: {e}")
            else:
                # The commit failed: surface it (same contract as
                # LTMemorySession.__exit__) so no caller reports success or
                # runs destructive follow-ups over data that never persisted.
                logger.error(f"Database commit failed during admin session cleanup: {e}")
                raise
        finally:
            if self._conn:
                self.pool.putconn(self._conn)
                self._conn = None
            self._closed = True

    def execute_query(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> List[Dict]:
        """Execute query and return results as list of dictionaries."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)

            if cur.description:
                return cur.fetchall()
            else:
                return []

    def execute_single(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> Optional[Dict]:
        """Execute query and return single result or None."""
        results = self.execute_query(query, params)
        return results[0] if results else None
    
    def execute_update(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> int:
        """Execute update query and return number of rows affected."""
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            return cur.rowcount
    
    def execute_delete(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> int:
        """Execute delete query and return number of rows affected."""
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            return cur.rowcount
    
    def _acquire_connection_with_timeout(self) -> psycopg.Connection:
        """Acquire connection from pool with timeout."""
        try:
            start_time = time.time()
            timeout = 30
            retry_interval = 0.1
            
            while time.time() - start_time < timeout:
                try:
                    conn = self.pool.getconn()
                    if conn and not conn.closed:
                        return conn
                    elif conn and conn.closed:
                        self.pool.putconn(conn)
                except PoolTimeout:
                    pass

                time.sleep(retry_interval)

            logger.error(f"Auth database connection timeout after {timeout}s")
            raise ValueError(f"Auth database connection timeout after {timeout}s")
        except Exception as e:
            logger.error(f"Error acquiring auth database connection: {e}")
            raise