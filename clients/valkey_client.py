"""
Unified Valkey client for system-wide use.

Combines auth operations (async) and working memory operations (sync).
Uses Vault for configuration with fail-fast semantics.
"""

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional

from config import config

if TYPE_CHECKING:
    import valkey

logger = logging.getLogger(__name__)

# Global connection pool (singleton like PostgresClient)
_valkey_pool = None


class ValkeyClient:
    """
    Production Valkey client with fail-fast semantics for required infrastructure.

    System fails to start if Valkey is unreachable. Operations fail loudly if
    Valkey goes down during runtime. Features connection pooling and retry
    patterns for transient failures.
    """

    def __init__(self):
        # Get configuration and initialize connections
        # System fails to start if Valkey is unreachable (fail-fast)
        self._load_config()
        self._init_connections()

    def _load_config(self):
        """Load Valkey URL and password from Vault. Raises if Vault is unavailable."""
        from urllib.parse import urlparse
        from clients.vault_client import get_service_config
        self.valkey_url = get_service_config('valkey_url')
        try:
            self.password = get_service_config('valkey_password')
        except KeyError:
            self.password = None
            logger.toast("Valkey running without authentication")

        parsed = urlparse(self.valkey_url)
        # TLS schemes must not be accepted-and-dropped — this client
        # implements no TLS (no ssl key reaches any connection constructor),
        # so honoring the scheme silently would connect in plaintext. Reject
        # loudly at config load instead (same fail-fast shape as the
        # query-parameter rejection in postgres_client._parse_database_url).
        if parsed.scheme in ('rediss', 'valkeys'):
            raise ValueError(
                f"Valkey URL scheme {parsed.scheme!r} requests TLS, which this client does not "
                f"implement — accepting it would silently connect in plaintext. Use a "
                f"valkey:// or redis:// URL (TLS support would need a certificate-verification "
                f"Vault surface, which does not exist): {self.valkey_url}"
            )
        if parsed.scheme not in ('redis', 'valkey'):
            raise ValueError(
                f"Valkey URL scheme {parsed.scheme!r} is not supported (expected valkey:// or "
                f"redis://): {self.valkey_url}"
            )
        self.host = parsed.hostname or 'localhost'
        self.port = parsed.port or 6379
        logger.debug(f"Valkey config loaded from Vault: {self.host}:{self.port}")

    def _init_connections(self):
        """
        Initialize Valkey connections with fail-fast semantics.

        Raises if Valkey is unreachable - system should not start without Valkey.
        """
        global _valkey_pool

        import valkey
        import valkey.asyncio as async_valkey
        from valkey.connection import ConnectionPool

        conn_params = {
            'host': self.host,
            'port': self.port,
            'password': self.password,
            'decode_responses': True,
            'health_check_interval': 30,
            'socket_keepalive': True,
            'socket_keepalive_options': {},
            'retry_on_timeout': True,
            'socket_connect_timeout': 5,
            'socket_timeout': 5
        }

        # Create thread-safe connection pool (like PostgresClient)
        if _valkey_pool is None:
            _valkey_pool = ConnectionPool(
                max_connections=config.cache.max_connections,
                **conn_params
            )
            logger.toast(f"Valkey connection pool created: {self.host}:{self.port} (max=20)")

        # Use pooled connections
        self._client = valkey.Valkey(connection_pool=_valkey_pool)
        self._client.ping()  # Raises if unreachable - system fails to start

        # Create binary client using same connection params but without decode_responses
        binary_conn_params = conn_params.copy()
        binary_conn_params['decode_responses'] = False
        self._binary_client = valkey.Valkey(**binary_conn_params)

        # Build URL for async client (with auth if password is set)
        if self.password:
            from urllib.parse import quote
            auth_url = f"valkey://:{quote(self.password, safe='')}@{self.host}:{self.port}"
        else:
            auth_url = f"valkey://{self.host}:{self.port}"
        self.valkey = async_valkey.from_url(
            auth_url,
            encoding="utf-8",
            **{k: v for k, v in conn_params.items() if k not in ['host', 'port', 'password']}
        )

        logger.toast(f"Valkey client initialized: {self.host}:{self.port}")

    @property
    def valkey_binary(self) -> "valkey.Valkey":
        """
        Access the binary client for storing raw bytes (e.g., embeddings).

        The binary client has decode_responses=False, preserving bytes as-is
        without UTF-8 decoding. Used for numpy arrays and other binary data.
        """
        return self._binary_client

    def get(self, key: str) -> Optional[str]:
        """
        Get value by key.

        Returns None if key doesn't exist.
        Raises if Valkey connection fails.
        """
        return self._client.get(key)

    def delete(self, key: str) -> bool:
        """
        Delete key.

        Returns True if key was deleted, False if key didn't exist.
        Raises if Valkey connection fails.
        """
        result = self._client.delete(key)
        return result > 0

    def exists(self, key: str) -> bool:
        """
        Check if key exists.

        Returns True if key exists, False if not.
        Raises if Valkey connection fails.
        """
        result = self._client.exists(key)
        return result > 0

    def ttl(self, key: str) -> int:
        """
        Get TTL of a key in seconds.

        Returns -1 if key has no TTL, -2 if key doesn't exist.
        Raises if Valkey connection fails.
        """
        return self._client.ttl(key)

    def scan(self, cursor: int, match: Optional[str] = None, count: int = 100) -> tuple[int, list[str]]:
        """
        Scan keys matching pattern.

        Returns (next_cursor, keys) tuple.
        Raises if Valkey connection fails.
        """
        return self._client.scan(cursor, match=match, count=count)

    def increment_with_expiry(self, key: str, expiry_seconds: int) -> int:
        """Atomic increment with expiration - ideal for rate limiting."""
        # Single EVAL so the counter can never exist without its TTL:
        # a crash between a bare INCR and a separate EXPIRE must not be
        # able to immortalize a rate-limit counter. EXPIRE NX sets the
        # TTL only when the key has none, preserving the window on
        # subsequent increments (and healing any TTL-less key).
        script = """
        local count = redis.call('INCR', KEYS[1])
        redis.call('EXPIRE', KEYS[1], ARGV[1], 'NX')
        return count
        """
        return int(self._client.eval(script, 1, key, expiry_seconds))

    def delete_and_increment(self, delete_key: str, increment_key: str) -> tuple[int, int]:
        """Atomic delete-then-increment, returning (deleted_count, new_counter).

        Single EVAL so no writer can observe or land between the DEL and the
        counter bump: a reader holding the pre-increment counter can never
        re-create the deleted key with stale content after this call returns.
        """
        script = """
        local deleted = redis.call('DEL', KEYS[1])
        local count = redis.call('INCR', KEYS[2])
        return {deleted, count}
        """
        result = self._client.eval(script, 2, delete_key, increment_key)
        return int(result[0]), int(result[1])

    def json_set(self, key: str, path: str, value: Dict[str, Any], ex: Optional[int] = None) -> bool:
        """Set JSON data, optionally with expiration.

        When path is "$", replaces entire value. When path is "$.field_name",
        updates just that field in existing JSON (read-modify-write).

        Args:
            key: The key to store JSON data
            path: "$" for full replacement, "$.field_name" for field update
            value: Value to set
            ex: Expiration in seconds. If None, preserves existing TTL (for updates)
                or uses no expiration (for new keys)

        Returns:
            True if successful, False if key doesn't exist (for field updates)
        """
        if path == "$":
            # Full replacement
            json_data = json.dumps(value)
            if ex is not None:
                return self._client.setex(key, ex, json_data)
            else:
                return self._client.set(key, json_data)

        # Field update: read-modify-write
        if not path.startswith("$."):
            raise ValueError(f"Unsupported path: {path}. Use '$' or '$.field_name'")

        current = self.json_get(key, "$")
        if current is None:
            return False

        data = current[0]
        field = path[2:]
        data[field] = value

        # Preserve existing TTL if no expiry specified
        if ex is None:
            remaining_ttl = self.ttl(key)
            ex = remaining_ttl if remaining_ttl > 0 else None

        json_data = json.dumps(data)
        if ex is not None:
            return self._client.setex(key, ex, json_data)
        else:
            return self._client.set(key, json_data)

    def json_set_with_expiry(self, key: str, path: str, value: Dict[str, Any], ex: int) -> bool:
        """Set JSON data with expiration. Alias for json_set with required ex."""
        return self.json_set(key, path, value, ex=ex)

    def json_get(self, key: str, path: str) -> Optional[list[Dict[str, Any]]]:
        """Get JSON data (returns list format for JSONPath compatibility)."""
        json_str = self._client.get(key)
        if json_str is None:
            return None

        data = json.loads(json_str)
        return [data]  # Wrap in list to match expected JSONPath result format

    def getdel_json(self, key: str) -> Optional[list[Dict[str, Any]]]:
        """Atomically get-and-delete JSON data (single-use semantics)."""
        json_str = self._client.getdel(key)
        if json_str is None:
            return None

        data = json.loads(json_str)
        return [data]

    def hset_with_retry(self, hash_key: str, field: str, value: str) -> int:
        """Hash set with retry pattern for transient failures."""
        try:
            return self._client.hset(hash_key, field, value)
        except Exception as e:
            logger.warning(f"Valkey write failed, retrying: {e}", exc_info=True)

        time.sleep(0.1)
        return self._client.hset(hash_key, field, value)  # Raises on failure

    def hget_with_retry(self, hash_key: str, field: str) -> Optional[str]:
        """Hash get with retry pattern for transient failures."""
        try:
            return self._client.hget(hash_key, field)
        except Exception as e:
            logger.warning(f"Valkey read failed, retrying: {e}", exc_info=True)

        time.sleep(0.1)
        return self._client.hget(hash_key, field)

    def hdel_with_retry(self, hash_key: str, *fields) -> int:
        """Hash delete with retry pattern for transient failures."""
        try:
            return self._client.hdel(hash_key, *fields)
        except Exception as e:
            logger.warning(f"Valkey delete failed, retrying: {e}", exc_info=True)

        time.sleep(0.1)
        return self._client.hdel(hash_key, *fields)

    def set(self, key: str, value: str, nx: Optional[bool] = None, ex: Optional[int] = None) -> bool:
        """
        Set key to value.

        Args:
            key: Key to set
            value: Value to store
            nx: If True, only set if key doesn't exist
            ex: Expiration time in seconds

        Returns:
            True if key was set, False if NX was specified and key already exists

        Raises if Valkey connection fails.
        """
        kwargs = {}
        if nx is not None:
            kwargs['nx'] = nx
        if ex is not None:
            kwargs['ex'] = ex
        return self._client.set(key, value, **kwargs)

    def compare_and_set(self, key: str, expected_value: str, new_value: str) -> bool:
        """Atomically replace a string value only when it still matches."""
        script = """
        if redis.call('GET', KEYS[1]) ~= ARGV[1] then
            return 0
        end
        redis.call('SET', KEYS[1], ARGV[2])
        return 1
        """
        return bool(self._client.eval(script, 1, key, expected_value, new_value))

    def compare_and_delete(self, key: str, expected_value: str) -> bool:
        """Atomically delete a key only when it still holds the expected value."""
        script = """
        if redis.call('GET', KEYS[1]) ~= ARGV[1] then
            return 0
        end
        redis.call('DEL', KEYS[1])
        return 1
        """
        return bool(self._client.eval(script, 1, key, expected_value))

    def compare_and_expire(self, key: str, expected_value: str, seconds: int) -> bool:
        """Atomically extend a key's TTL only when it still holds the expected value."""
        script = """
        if redis.call('GET', KEYS[1]) ~= ARGV[1] then
            return 0
        end
        return redis.call('EXPIRE', KEYS[1], ARGV[2])
        """
        return bool(self._client.eval(script, 1, key, expected_value, seconds))

    def increment(self, key: str) -> int:
        """Atomically increment a counter by one, creating it at 1 if missing."""
        return int(self._client.incr(key))

    def set_if_counter_matches(self, counter_key: str, expected_counter: int,
                               value_key: str, value: str) -> bool:
        """Atomically SET value_key only when counter_key equals expected_counter.

        A missing counter_key reads as 0, so callers capturing the epoch before
        the first invalidation can still compare-and-set. Returns False (no
        write) when the counter has moved on.
        """
        script = """
        local current = tonumber(redis.call('GET', KEYS[1]) or '0')
        if current ~= tonumber(ARGV[1]) then
            return 0
        end
        redis.call('SET', KEYS[2], ARGV[2])
        return 1
        """
        return bool(self._client.eval(script, 2, counter_key, value_key,
                                     expected_counter, value))

    def scan_iter(self, match: Optional[str] = None) -> Iterator[str]:
        """Scan iterator for keys matching pattern."""
        return self._client.scan_iter(match=match)

    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on key."""
        return self._client.expire(key, seconds)

    def setex(self, key: str, seconds: int, value: str) -> bool:
        """Set key with expiration."""
        return self._client.setex(key, seconds, value)

    def rpush(self, key: str, *values: str) -> int:
        """
        Append values to the end of a list.

        Creates the list if it doesn't exist.

        Args:
            key: List key
            *values: Values to append

        Returns:
            Length of list after push

        Raises if Valkey connection fails.
        """
        return self._client.rpush(key, *values)

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        """
        Get a range of elements from a list.

        Args:
            key: List key
            start: Start index (0-based, negative from end)
            stop: Stop index (inclusive, -1 means end)

        Returns:
            List of elements in range, empty list if key doesn't exist

        Raises if Valkey connection fails.
        """
        return self._client.lrange(key, start, stop)

    def flush_except_whitelist(self, preserve_prefixes: list[str]) -> int:
        """
        Delete all keys from Valkey except those matching whitelist prefixes.

        Used during system startup to clear all caches while preserving
        critical data like auth sessions and rate limiting.

        Args:
            preserve_prefixes: List of key prefixes to preserve (e.g., ["session:", "rate_limit:"])

        Returns:
            Number of keys deleted
        """
        if any(prefix == "" for prefix in preserve_prefixes):
            raise ValueError("Empty string in preserve_prefixes would preserve all keys")

        deleted_count = 0
        preserved_count = 0

        # Scan all keys
        for key in self.scan_iter(match="*"):
            # Check if key starts with any whitelisted prefix
            should_preserve = any(key.startswith(prefix) for prefix in preserve_prefixes)

            if should_preserve:
                preserved_count += 1
            else:
                if self._client.delete(key):
                    deleted_count += 1

        logger.toast(
            f"Flushed {deleted_count} keys from Valkey, "
            f"preserved {preserved_count} keys matching whitelist: {preserve_prefixes}"
        )
        return deleted_count

    def shutdown(self):
        """Clean shutdown of Valkey client."""
        logger.toast("Valkey client shutdown complete")


_valkey_client: Optional[ValkeyClient] = None


def get_valkey_client() -> ValkeyClient:
    global _valkey_client
    if _valkey_client is None:
        _valkey_client = ValkeyClient()
    return _valkey_client


def get_valkey() -> ValkeyClient:
    """Get Valkey client instance."""
    return get_valkey_client()
