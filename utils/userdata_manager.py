"""
Per-user SQLite store (``data/users/{user_id}/userdata.db``) with transparent
field encryption, plus the DDL for every tool table MIRA ships.

Encryption is driven by the column-name prefix: any column named
``encrypted__*`` holds a Fernet token over the JSON value, encrypted on insert
and update and decrypted on select, so a reader gets plaintext from ``select``
and never decrypts again (a second ``_decrypt_dict`` on a plaintext row raises
``InvalidToken``). The key is ``derive_session_key(user_id)`` — HMAC-SHA256
over the user id, keyed by the Vault ``userdata_encryption_key`` — so one
user's file is unreadable with another user's key.

Open order is load-bearing: tables (``CREATE ... IF NOT EXISTS``) → the
versioned steps in ``utils/userdata_migrations.py`` (a file created fresh is
stamped at head instead) → ``_PAGER_INDEXES``/``_DOMAINDOC_INDEXES``. An index
over a column a step adds, created before the steps run, fails on every file
older than that step.

Every connection sets ``PRAGMA foreign_keys=ON`` (per connection in SQLite: it
is off by default, which quietly disables every ``ON DELETE CASCADE``) and uses
``INTEGER PRIMARY KEY AUTOINCREMENT`` ids so a deleted row's id is never
reissued to new content.

One manager instance per user, cached process-wide under ``_manager_cache_lock``,
each owning one persistent connection; ``clear_manager_cache`` closes it and
runs when that user's segment collapses.
"""

import base64
import hashlib
import hmac
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID

from cryptography.fernet import Fernet

from utils.userdata_migrations import run_userdata_migrations, stamp_userdata_head

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cns.integration.event_bus import EventBus

_manager_cache: Dict[str, "UserDataManager"] = {}

# Guards every _manager_cache access: get_user_data_manager's check-then-act —
# create-and-cache under the lock, so concurrent first accesses cannot overwrite
# a slot and orphan a live SQLite connection — plus clear_manager_cache's
# per-user close/delete and full-cache iteration. Plain Lock suffices: no guarded
# path re-enters the cache.
_manager_cache_lock = threading.Lock()

# id columns use AUTOINCREMENT so a deleted doc's rowid is never reused
# (plain INTEGER PRIMARY KEY reuses max(rowid)+1, which can graft deleted
# content into a newly created document).
_DOMAINDOC_DDL = {
    "domaindocs": """
    CREATE TABLE IF NOT EXISTS domaindocs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT UNIQUE NOT NULL,
        encrypted__description TEXT,
        enabled BOOLEAN DEFAULT TRUE,
        archived BOOLEAN DEFAULT FALSE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "domaindoc_sections": """
    CREATE TABLE IF NOT EXISTS domaindoc_sections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domaindoc_id INTEGER NOT NULL REFERENCES domaindocs(id) ON DELETE CASCADE,
        parent_section_id INTEGER DEFAULT NULL REFERENCES domaindoc_sections(id) ON DELETE CASCADE,
        header TEXT NOT NULL,
        encrypted__content TEXT NOT NULL,
        encrypted__summary TEXT DEFAULT NULL,
        sort_order INTEGER NOT NULL,
        collapsed BOOLEAN DEFAULT FALSE,
        expanded_by_default BOOLEAN DEFAULT FALSE,
        pinned BOOLEAN DEFAULT FALSE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(domaindoc_id, parent_section_id, header)
    )
    """,
    "domaindoc_versions": """
    CREATE TABLE IF NOT EXISTS domaindoc_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domaindoc_id INTEGER NOT NULL REFERENCES domaindocs(id) ON DELETE CASCADE,
        section_id INTEGER REFERENCES domaindoc_sections(id) ON DELETE SET NULL,
        version_num INTEGER NOT NULL,
        operation TEXT NOT NULL,
        encrypted__diff_data TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(domaindoc_id, section_id, version_num)
    )
    """,
}

# Sidebar-agent tables. SidebarAgent failure records and sidebar_tool
# complete_task records both write sidebar_activity; sidebar_tool keeps working
# notes in scratchpad, cleaned up by SidebarDispatcher._maybe_cleanup().
# Created via agents.base.ensure_activity_schema() (CREATE ... IF NOT EXISTS);
# sidebar_activity is also created at every user-DB open.
_PAGER_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_pager_devices_user_id ON pager_devices(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_pager_devices_active ON pager_devices(active)",
    "CREATE INDEX IF NOT EXISTS idx_pager_trust_user_id ON pager_trust(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_pager_trust_trusting_device ON pager_trust(trusting_device_id)",
    "CREATE INDEX IF NOT EXISTS idx_pager_messages_user_id ON pager_messages(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_pager_messages_sender ON pager_messages(sender_id)",
    "CREATE INDEX IF NOT EXISTS idx_pager_messages_recipient ON pager_messages(recipient_id)",
    "CREATE INDEX IF NOT EXISTS idx_pager_messages_expires ON pager_messages(expires_at)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_pager_messages_external_id "
    "ON pager_messages(user_id, external_message_id) WHERE external_message_id IS NOT NULL",
)

_DOMAINDOC_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_domaindocs_label ON domaindocs(label)",
    "CREATE INDEX IF NOT EXISTS idx_domaindocs_enabled ON domaindocs(enabled)",
    "CREATE INDEX IF NOT EXISTS idx_domaindocs_archived ON domaindocs(archived)",
    "CREATE INDEX IF NOT EXISTS idx_sections_domaindoc ON domaindoc_sections(domaindoc_id)",
    "CREATE INDEX IF NOT EXISTS idx_sections_collapsed ON domaindoc_sections(domaindoc_id, collapsed)",
    "CREATE INDEX IF NOT EXISTS idx_sections_parent ON domaindoc_sections(parent_section_id)",
    "CREATE INDEX IF NOT EXISTS idx_versions_domaindoc ON domaindoc_versions(domaindoc_id)",
)

ACTIVITY_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS sidebar_activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interface_name TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'handled',
    escalation_reason TEXT,
    run_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(interface_name, thread_id)
)"""

ACTIVITY_INDEX_DDL = """\
CREATE INDEX IF NOT EXISTS idx_activity_interface
ON sidebar_activity(interface_name)"""

SCRATCHPAD_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS scratchpad (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    note TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)"""

SCRATCHPAD_INDEX_DDL = """\
CREATE INDEX IF NOT EXISTS idx_scratchpad_thread
ON scratchpad(thread_id)"""


class UserDataManager:
    """
    Manages per-user SQLite databases with session-based encryption.

    Connection ownership: Each instance owns a single persistent SQLite connection,
    created lazily on first database access. Use `close()` for explicit cleanup or
    context manager pattern (`with get_user_data_manager(user_id) as db:`).
    """

    def __init__(self, user_id: UUID, session_key: bytes):
        self.user_id = user_id
        self.session_key = session_key
        self.fernet = self._create_fernet()
        self.db_path = self._get_user_db_path()
        self._conn: Optional[sqlite3.Connection] = None  # Lazy persistent connection
        self._ensure_database()

    @property
    def connection(self) -> sqlite3.Connection:
        """
        Lazy persistent connection (thread-safe for cross-thread reuse).

        Connection is created on first access and reused for all subsequent operations.
        Uses check_same_thread=False to allow reuse across ThreadPoolExecutor workers.
        WAL mode enables concurrent reads/writes from multiple threads.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False  # Allow cross-thread usage
            )
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for concurrent read/write support
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            # Enforce foreign keys (per-connection in SQLite). Without this, every
            # ON DELETE CASCADE in the schema is inert and deletes orphan child rows.
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        """Close the persistent connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug(f"Closed SQLite connection for user {self.user_id}")

    def __enter__(self) -> "UserDataManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet cipher from session key."""
        key = base64.urlsafe_b64encode(self.session_key[:32])
        return Fernet(key)


    @property
    def base_dir(self) -> Path:
        # Anchor to the project root so the storage path is independent of cwd.
        project_root = Path(__file__).resolve().parent.parent
        user_dir = project_root / "data/users" / str(self.user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    def _get_user_db_path(self) -> Path:
        return self.base_dir / "userdata.db"
    
    def _ensure_database(self):
        """Ensure the database exists and its schema is at ``USERDATA_HEAD``."""
        new_file = not self.db_path.exists()
        if new_file:
            self.db_path.touch()
            logger.info(f"Created user database: {self.db_path}")
        self._initialize_tool_schemas(new_file)

    def _initialize_tool_schemas(self, new_file: bool):
        """
        Tables (CREATE IF NOT EXISTS) → versioned steps (``utils/userdata_migrations.py``;
        a new file is stamped at head instead, its tables already being today's) → indexes,
        which may reference columns a step adds.
        """
        logger.info(f"Initializing tool schemas for user {self.user_id}")

        self._init_pager_schema()
        self._init_domaindoc_schema()
        self._init_contacts_schema()
        self.connection.commit()

        if new_file:
            stamp_userdata_head(self.connection)
        else:
            run_userdata_migrations(self.connection, self._encrypt_value)

        cursor = self.connection.cursor()
        for index_sql in _PAGER_INDEXES + _DOMAINDOC_INDEXES:
            cursor.execute(index_sql)
        self.connection.commit()

        self._init_trigger_rules_schema()

        # Initialize the sidebar-activity schema so dispatcher dedup works
        # from first boot, before any agent has run.
        self._init_sidebar_activity_schema()

        logger.info("Tool schemas initialized successfully")

    def _init_sidebar_activity_schema(self):
        """Create the sidebar_activity table at every user-DB open.

        The table then exists from first boot (the dispatcher's dedup reads
        it) instead of appearing only when the first agent runs. DDL: module-level ACTIVITY_TABLE_DDL / ACTIVITY_INDEX_DDL.
        """
        cursor = self.connection.cursor()
        cursor.execute(ACTIVITY_TABLE_DDL)
        cursor.execute(ACTIVITY_INDEX_DDL)
        self.connection.commit()
    
    def _init_pager_schema(self):
        """Initialize PagerTool database schema."""
        cursor = self.connection.cursor()

        devices_sql = """
        CREATE TABLE IF NOT EXISTS pager_devices (
            id TEXT PRIMARY KEY,
            user_id UUID NOT NULL,
            encrypted__name TEXT NOT NULL,
            encrypted__description TEXT,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            device_secret TEXT NOT NULL,
            device_fingerprint TEXT NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE
        )
        """

        trust_sql = """
        CREATE TABLE IF NOT EXISTS pager_trust (
            id TEXT PRIMARY KEY,
            user_id UUID NOT NULL,
            trusting_device_id TEXT NOT NULL,
            trusted_device_id TEXT NOT NULL,
            trusted_fingerprint TEXT NOT NULL,
            encrypted__trusted_name TEXT,
            first_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_verified TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            trust_status TEXT NOT NULL DEFAULT 'trusted',
            UNIQUE(trusting_device_id, trusted_device_id)
        )
        """

        messages_sql = """
        CREATE TABLE IF NOT EXISTS pager_messages (
            id TEXT PRIMARY KEY,
            user_id UUID NOT NULL,
            sender_id TEXT NOT NULL,
            recipient_id TEXT NOT NULL,
            encrypted__content TEXT,
            encrypted__original_content TEXT,
            ai_distilled BOOLEAN NOT NULL DEFAULT FALSE,
            priority INTEGER NOT NULL DEFAULT 0,
            location TEXT,
            sent_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP WITH TIME ZONE,
            read_at TIMESTAMP WITH TIME ZONE,
            delivered BOOLEAN NOT NULL DEFAULT TRUE,
            read BOOLEAN NOT NULL DEFAULT FALSE,
            message_signature TEXT,
            sender_fingerprint TEXT,
            external_message_id TEXT
        )
        """

        cursor.execute(devices_sql)
        cursor.execute(trust_sql)
        cursor.execute(messages_sql)

    def _init_domaindoc_schema(self):
        """Initialize Domaindoc database schema for section-aware storage."""
        cursor = self.connection.cursor()

        # DDL lives in _DOMAINDOC_DDL (module level) — shared with the versions rebuild step.
        for ddl in _DOMAINDOC_DDL.values():
            cursor.execute(ddl)

    def _init_contacts_schema(self):
        """Initialize ContactsTool database schema."""
        cursor = self.connection.cursor()
        contacts_sql = """
        CREATE TABLE IF NOT EXISTS contacts (
            id TEXT PRIMARY KEY,
            encrypted__name TEXT NOT NULL,
            encrypted__email TEXT,
            encrypted__phone TEXT,
            encrypted__street TEXT,
            encrypted__city TEXT,
            encrypted__state TEXT,
            encrypted__zip TEXT,
            encrypted__pager_address TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
        cursor.execute(contacts_sql)

    def _init_trigger_rules_schema(self):
        """Initialize trigger_rules table for per-user sidebar trigger filters.

        Each row is a filter rule scoped to a specific trigger type. Field
        names are validated by the API against the trigger registry
        (agents/triggers/registry.py); rows are applied at dispatch time by
        agents/triggers/rule_filter.py, which reads them by trigger_id. The
        scope column is trigger-specific stored metadata — no registered
        trigger has a scoped surface today, so nothing matches on it.
        """
        cursor = self.connection.cursor()

        rules_sql = """
        CREATE TABLE IF NOT EXISTS trigger_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger_id TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT 'INBOX',
            field TEXT NOT NULL,
            pattern TEXT NOT NULL,
            prompt TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
        cursor.execute(rules_sql)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_trigger_rules_trigger "
            "ON trigger_rules(trigger_id, enabled)"
        )
        self.connection.commit()

    def _encrypt_value(self, value: Any) -> str:
        """Encrypt a value for storage."""
        json_str = json.dumps(value)
        encrypted_token = self.fernet.encrypt(json_str.encode())
        return encrypted_token.decode()
    
    def _decrypt_value(self, encrypted_str: str) -> Any:
        """Decrypt a value from storage. Raises on cipher or JSON failure."""
        decrypted_bytes = self.fernet.decrypt(encrypted_str.encode())
        return json.loads(decrypted_bytes.decode())

    def _encrypt_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt fields with encrypted__ prefix (prefix is kept in column name)."""
        result = {}
        for key, value in data.items():
            if key.startswith('encrypted__'):
                # Encrypt value but keep the encrypted__ prefix in the column name
                result[key] = self._encrypt_value(value) if value is not None else None
            elif value is None:
                result[key] = None
            elif isinstance(value, bool):
                # SQLite stores booleans as integers (1/0)
                result[key] = 1 if value else 0
            elif isinstance(value, (int, float)):
                # Keep numeric types as-is for proper SQL comparisons
                result[key] = value
            else:
                result[key] = str(value)
        return result
    
    def _decrypt_dict(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Decrypt fields prefixed with encrypted__. Raises on cipher failure."""
        result = {}
        for key, value in data.items():
            if key.startswith("encrypted__") and value is not None:
                result[key] = self._decrypt_value(value)
            else:
                result[key] = value
        return result
    
    def execute(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute SQL query and return results."""
        cursor = self.connection.cursor()
        cursor.execute(query, params or {})

        if cursor.description:
            results = [dict(row) for row in cursor.fetchall()]
            # Close any implicit read transaction to ensure fresh reads
            self.connection.commit()
            return results
        else:
            self.connection.commit()
            return []

    def fetchone(self, query: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Execute query and return single row."""
        results = self.execute(query, params)
        return results[0] if results else None

    def fetchall(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute query and return all rows."""
        return self.execute(query, params)

    def create_table(self, table_name: str, schema: str):
        """Create table with given schema."""
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        cursor = self.connection.cursor()
        cursor.execute(query)
        self.connection.commit()

    def insert(self, table_name: str, data: Dict[str, Any]) -> str:
        """Insert encrypted data and return row ID."""
        encrypted_data = self._encrypt_dict(data)
        columns = list(encrypted_data.keys())
        placeholders = [f":{col}" for col in columns]
        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"

        cursor = self.connection.cursor()
        cursor.execute(query, encrypted_data)
        self.connection.commit()
        return str(cursor.lastrowid)

    def select(self, table_name: str, where: str = None, params: Optional[Dict] = None, order_by: str = None) -> List[Dict[str, Any]]:
        """Select and decrypt rows from table."""
        query = f"SELECT * FROM {table_name}"
        if where:
            query += f" WHERE {where}"
        if order_by:
            query += f" ORDER BY {order_by}"
        rows = self.fetchall(query, params)
        return [self._decrypt_dict(row) for row in rows]

    def update(self, table_name: str, data: Dict[str, Any], where: str, params: Optional[Dict] = None) -> int:
        """Update rows with encrypted data."""
        encrypted_data = self._encrypt_dict(data)
        set_clauses = [f"{col} = :{col}" for col in encrypted_data.keys()]
        query = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {where}"

        all_params = encrypted_data.copy()
        if params:
            all_params.update(params)

        cursor = self.connection.cursor()
        cursor.execute(query, all_params)
        self.connection.commit()
        return cursor.rowcount

    def delete(self, table_name: str, where: str, params: Optional[Dict] = None) -> int:
        """Delete rows from table."""
        query = f"DELETE FROM {table_name} WHERE {where}"
        cursor = self.connection.cursor()
        cursor.execute(query, params or {})
        self.connection.commit()
        return cursor.rowcount
    
    def get_tool_data_dir(self, tool_name: str) -> Path:
        path = self.base_dir / "tools" / tool_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_skills_dir(self) -> Path:
        """Per-user skills root; each skill is a subdirectory containing a SKILL.md."""
        path = self.base_dir / "skills"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _ensure_credentials_table(self):
        """Helper method to ensure credentials table exists (used by UserCredentialService)."""
        schema = """
            id TEXT PRIMARY KEY,
            credential_type TEXT NOT NULL,
            service_name TEXT NOT NULL,
            encrypted__credential_value TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(credential_type, service_name)
        """
        self.create_table('credentials', schema)


# CROSS-FILE-COUPLING: scripts/export_user_bundle.py reimplements this
# derivation plus _create_fernet by hand; a change here without the same change
# there exports every encrypted__ value as an UNDECRYPTABLE__ null.
def derive_session_key(user_id: str) -> bytes:
    """Derive per-user encryption key from the Vault-backed master key."""
    from clients.vault_client import get_service_config

    master_key = get_service_config("userdata_encryption_key").encode()
    return hmac.new(master_key, str(user_id).encode(), hashlib.sha256).digest()


def get_user_data_manager(user_id: UUID) -> UserDataManager:
    """
    Get a cached UserDataManager instance with automatic encryption key derivation.

    Instances are cached per-user for connection reuse across operations.
    Use clear_manager_cache() to explicitly close connections when needed.
    """
    cache_key = str(user_id)

    with _manager_cache_lock:
        if cache_key not in _manager_cache:
            session_key = derive_session_key(cache_key)
            _manager_cache[cache_key] = UserDataManager(user_id, session_key)
            logger.debug(f"Created new UserDataManager for user {user_id}")

        return _manager_cache[cache_key]


def clear_manager_cache(user_id: Optional[UUID] = None) -> None:
    """
    Clear cached UserDataManager instances and close their connections.

    Args:
        user_id: If provided, only clear the cache for this user.
                 If None, clear all cached managers.
    """
    if user_id is not None:
        cache_key = str(user_id)
        with _manager_cache_lock:
            if cache_key in _manager_cache:
                _manager_cache[cache_key].close()
                del _manager_cache[cache_key]
                logger.debug(f"Cleared UserDataManager cache for user {user_id}")
    else:
        with _manager_cache_lock:
            for manager in _manager_cache.values():
                manager.close()
            _manager_cache.clear()
        logger.info("Cleared all UserDataManager caches")


class UserDataManagerCleanupHandler:
    """
    Handles cleanup of cached UserDataManager instances on session collapse.

    Subscribes to SegmentCollapsedEvent and closes the user's SQLite connection
    when their conversation session collapses, freeing resources.
    """

    def __init__(self, event_bus: 'EventBus'):
        self.event_bus = event_bus
        self.event_bus.subscribe('SegmentCollapsedEvent', self._handle_segment_collapsed)
        logger.info("UserDataManagerCleanupHandler subscribed to SegmentCollapsedEvent")

    def _handle_segment_collapsed(self, event) -> None:
        """Close user's SQLite connection when their session collapses."""
        try:
            user_id = UUID(event.user_id)
            clear_manager_cache(user_id)
            logger.debug(f"Closed UserDataManager for user {event.user_id} after segment collapse")
        except Exception as e:
            # Don't let cleanup failures break the collapse pipeline
            logger.warning(f"Failed to clean up UserDataManager for user {event.user_id}: {e}")
