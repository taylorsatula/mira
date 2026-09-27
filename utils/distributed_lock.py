"""
Distributed lock implementation using Valkey for multi-process concurrency control.

Provides atomic distributed locks that work across multiple worker processes,
replacing in-memory locks that only work within a single process.
"""

import logging
import threading
import uuid
from typing import Optional
from contextlib import contextmanager

from clients.valkey_client import get_valkey

logger = logging.getLogger(__name__)


class DistributedLock:
    """
    Distributed lock using Valkey's atomic SET NX operation.
    
    Ensures only one process can hold a lock for a given resource at a time,
    with automatic expiration to prevent deadlocks from crashed processes.
    """
    
    def __init__(self, lock_prefix: str = "lock:", default_ttl: int = 60):
        """
        Initialize distributed lock manager.
        
        Args:
            lock_prefix: Prefix for lock keys in Valkey
            default_ttl: Default TTL in seconds for locks (prevents deadlocks)
        """
        self.lock_prefix = lock_prefix
        self.default_ttl = default_ttl
        self._valkey = None

    @property
    def valkey(self):
        """Valkey client, resolved on first use — never at import time.

        Construction of a DistributedLock must not touch infrastructure:
        locks are constructed at module import (e.g. UserRequestLock in
        cns/api/chat.py), and import must succeed without Valkey/Vault
        credentials. Unavailability still propagates — at the first
        acquire/release/renew, never as a silent no-lock.
        """
        if self._valkey is None:
            self._valkey = get_valkey()
        return self._valkey
    
    def acquire(self, resource_id: str, ttl: int | None = None) -> str | None:
        """
        Attempt to acquire a distributed lock.

        Uses Valkey's atomic SET NX (set if not exists) operation to ensure
        only one process can acquire the lock for a resource at a time.

        Args:
            resource_id: Unique identifier for the resource to lock
            ttl: Time-to-live in seconds (uses default if not specified)

        Returns:
            The lock token if acquired, None if already locked. The token
            must be passed to release()/renew() — those operations are
            compare-and-delete / compare-and-expire against it, so a lock
            that expired and was re-acquired by another owner is never
            clobbered.

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        key = f"{self.lock_prefix}{resource_id}"
        ttl = ttl or self.default_ttl
        token = str(uuid.uuid4())

        # SET NX (set if not exists) with EX (expiration)
        # This is atomic - either we get the lock or we don't
        success = self.valkey.set(
            key,
            token,
            nx=True,  # Only set if key doesn't exist
            ex=ttl    # Set expiration time
        )

        if success:
            logger.debug(f"Acquired lock for {resource_id} with TTL {ttl}s")
            return token

        logger.debug(f"Failed to acquire lock for {resource_id} - already locked")
        return None
    
    def get_lock_owner(self, resource_id: str) -> Optional[str]:
        """
        Get the current owner (value) of a lock.

        Args:
            resource_id: Unique identifier for the resource

        Returns:
            Lock owner value if locked, None if not locked

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        key = f"{self.lock_prefix}{resource_id}"
        value = self.valkey.get(key)
        return value
    
    def release(self, resource_id: str, token: str) -> bool:
        """
        Release a distributed lock by token (atomic compare-and-delete).

        Args:
            resource_id: Unique identifier for the resource to unlock
            token: Token returned by the acquire() call that won the lock

        Returns:
            True if this owner's lock was released, False if the token did
            not match (expired and re-acquired by another owner). A False is
            a logged no-op, never an error.

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        key = f"{self.lock_prefix}{resource_id}"

        released = self.valkey.compare_and_delete(key, token)

        if released:
            logger.debug(f"Released lock for {resource_id}")
        else:
            logger.warning(
                "Lock for %s not released: token mismatch "
                "(lock expired and was re-acquired by another owner)",
                resource_id,
            )

        return released

    def renew(self, resource_id: str, token: str, ttl: int | None = None) -> bool:
        """
        Extend a held lock's TTL by token (atomic compare-and-expire).

        Args:
            resource_id: Unique identifier for the locked resource
            token: Token returned by the acquire() call that won the lock
            ttl: New TTL in seconds (uses default if not specified)

        Returns:
            True if the TTL was extended, False if ownership was lost (the
            caller logs and continues — the lock is gone either way).

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        key = f"{self.lock_prefix}{resource_id}"
        ttl = ttl or self.default_ttl

        renewed = self.valkey.compare_and_expire(key, token, ttl)

        if not renewed:
            logger.warning("Lock renewal for %s failed: ownership lost", resource_id)

        return renewed

    def start_renewal(self, resource_id: str, token: str) -> threading.Event:
        """
        Start a background daemon thread that keeps renewing a held lock
        until the returned stop Event is set (the guarded action ended) or
        ownership is lost.

        The cadence is TTL/3 — safely inside the TTL: one missed or slow
        renewal still leaves two full intervals of cover, so a live turn
        cannot outlive its own lock. Each renewal resets the full TTL via
        compare-and-expire against the token, so a lock that expired and was
        re-acquired by another owner is never extended by this thread.

        The caller MUST set the returned Event when the guarded action ends;
        the thread then exits within one cadence interval.

        Args:
            resource_id: Unique identifier for the locked resource
            token: Token returned by the acquire() call that won the lock

        Returns:
            threading.Event the owner sets to stop the renewal
        """
        interval = max(self.default_ttl // 3, 1)
        stop = threading.Event()

        def _renew_loop() -> None:
            while not stop.wait(interval):
                try:
                    if not self.renew(resource_id, token):
                        # Ownership lost (expired and re-acquired elsewhere):
                        # nothing left to renew.
                        break
                except Exception:
                    # Transient infrastructure failure: the key is not
                    # deleted by Valkey being down, so retry at the next
                    # cadence tick while cover remains.
                    logger.warning(
                        "Background lock renewal for %s failed; retrying next interval",
                        resource_id, exc_info=True,
                    )

        threading.Thread(
            target=_renew_loop,
            name=f"lock-renewal:{self.lock_prefix}{resource_id}",
            daemon=True,
        ).start()
        return stop

    def is_locked(self, resource_id: str) -> bool:
        """
        Check if a resource is currently locked.

        Args:
            resource_id: Unique identifier for the resource

        Returns:
            True if resource is locked, False otherwise

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        key = f"{self.lock_prefix}{resource_id}"
        return self.valkey.exists(key)
    
    def get_ttl(self, resource_id: str) -> int:
        """
        Get remaining TTL for a lock.

        Args:
            resource_id: Unique identifier for the resource

        Returns:
            TTL in seconds, -2 if key doesn't exist, -1 if no TTL set

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        key = f"{self.lock_prefix}{resource_id}"
        return self.valkey.ttl(key)
    
    @contextmanager
    def lock(self, resource_id: str, ttl: Optional[int] = None):
        """
        Context manager for distributed locks.

        Usage:
            with distributed_lock.lock("user_123"):
                # Critical section - only one process can be here
                process_user_request()

        Args:
            resource_id: Unique identifier for the resource to lock
            ttl: Time-to-live in seconds

        Raises:
            LockAcquisitionError: If lock can't be acquired

        Yields:
            None if lock acquired successfully
        """
        acquired_token: str | None = None
        try:
            acquired_token = self.acquire(resource_id, ttl)
            if acquired_token is None:
                raise LockAcquisitionError(f"Could not acquire lock for {resource_id}")
            yield
        finally:
            if acquired_token is not None:
                self.release(resource_id, acquired_token)


class LockAcquisitionError(Exception):
    """Raised when a distributed lock cannot be acquired."""
    pass


class UserRequestLock:
    """
    Specialized distributed lock for per-user request concurrency control.
    
    Ensures a user can only have one active chat request at a time across
    all worker processes.
    """
    
    def __init__(self, ttl: int = 60):
        """
        Initialize user request lock.
        
        Args:
            ttl: Lock timeout in seconds (protects against crashes)
        """
        self.lock = DistributedLock(lock_prefix="user_lock:", default_ttl=ttl)
        self.default_ttl = ttl
    
    
    
    def acquire(self, user_id: str) -> str | None:
        """
        Attempt to acquire lock for user.

        Args:
            user_id: User identifier

        Returns:
            Lock token if acquired, None if user has a concurrent request
        """
        token = self.lock.acquire(user_id, ttl=self.default_ttl)
        if token is not None:
            logger.debug(f"Acquired lock for user {user_id} (TTL: {self.default_ttl}s)")
        else:
            logger.debug(f"Failed to acquire lock for user {user_id} - concurrent request in progress")
        return token

    def release(self, user_id: str, token: str) -> bool:
        """
        Release lock for user by token (atomic compare-and-delete).

        Args:
            user_id: User identifier
            token: Token returned by acquire()

        Returns:
            True if this owner's lock was released
        """
        return self.lock.release(user_id, token)

    def renew(self, user_id: str, token: str) -> bool:
        """
        Extend the held lock's TTL by token (atomic compare-and-expire).

        Returns:
            True if renewed, False if ownership was lost
        """
        return self.lock.renew(user_id, token, ttl=self.default_ttl)

    def start_renewal(self, user_id: str, token: str) -> threading.Event:
        """
        Start background renewal of the held lock (see
        DistributedLock.start_renewal). The caller sets the returned Event
        when the turn ends; the renewal thread stops within one cadence
        interval (TTL/3).
        """
        return self.lock.start_renewal(user_id, token)
    
    def is_locked(self, user_id: str) -> bool:
        """
        Check if user currently has an active request.
        
        Args:
            user_id: User identifier
        
        Returns:
            True if user has active request
        """
        return self.lock.is_locked(user_id)
    
    @contextmanager
    def lock_user(self, user_id: str):
        """
        Context manager for user request locks.

        Args:
            user_id: User identifier

        Raises:
            LockAcquisitionError: If lock can't be acquired

        Yields:
            None if lock acquired successfully
        """
        with self.lock.lock(user_id):
            yield