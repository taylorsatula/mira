"""
User context management using contextvars.

This module provides transparent user context that works for both:
- Single-user scenarios (CLI): Context set once and persists
- Multi-user scenarios (web): Context isolated per request automatically

Uses Python's contextvars which provides automatic isolation for
concurrent operations while working identically for single-threaded use.
"""

import contextvars
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Literal, Optional

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

# Context variable for current user data
_user_context: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    'user_context',
    default=None
)

# Context variable for current segment ID (set during conversation processing)
_current_segment_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'current_segment_id',
    default=None
)

# Context variable for request cancellation (set by WebSocket handler, checked by LLM provider)
_cancel_event: contextvars.ContextVar[Optional[threading.Event]] = contextvars.ContextVar(
    'cancel_event',
    default=None
)


def set_cancel_event(event: threading.Event) -> None:
    """Set the cancellation event for the current request."""
    _cancel_event.set(event)


def get_cancel_event() -> Optional[threading.Event]:
    """Get the cancellation event for the current request, or None if not set."""
    return _cancel_event.get(None)


def set_cancel_reason(reason: Literal["halt", "disconnect"]) -> None:
    """Attach the Halt reason to the shared cancellation signal."""
    if reason not in {"halt", "disconnect"}:
        raise ValueError("Cancellation reason must be halt or disconnect")
    event = _cancel_event.get(None)
    if event is None:
        raise RuntimeError("Cannot set a cancellation reason without an active signal")
    event.mira_stop_reason = reason


def get_cancel_reason() -> Literal["halt", "disconnect"]:
    """Return the active signal's exact persistence reason."""
    event = _cancel_event.get(None)
    reason = getattr(event, "mira_stop_reason", "halt") if event is not None else "halt"
    if reason not in {"halt", "disconnect"}:
        raise RuntimeError(f"Invalid cancellation reason on active signal: {reason}")
    return reason


def check_cancelled() -> None:
    """Raise GenerationCancelled if the current request has been cancelled.

    Lightweight check — call between stream chunks, before tool execution,
    and at agentic loop boundaries.
    """
    evt = _cancel_event.get(None)
    if evt is not None and evt.is_set():
        from clients.llm.events import GenerationCancelled
        raise GenerationCancelled()


def set_current_user_id(user_id: str) -> None:
    """
    Set current user ID in context (standardized key: 'user_id').
    """
    current = _user_context.get() or {}
    current["user_id"] = user_id
    _user_context.set(current)


def get_current_user_id() -> str:
    """
    Get current user ID from context (reads 'user_id').
    """
    context = _user_context.get()
    if not context or "user_id" not in context:
        raise RuntimeError("No user context set. Ensure authentication is properly initialized.")
    return context["user_id"]


def set_current_user_data(user_data: Dict[str, Any]) -> None:
    """
    Set complete user data in context.
    Standardizes to 'user_id' and does not maintain legacy 'id'.
    """
    data = user_data.copy()
    if "user_id" not in data and "id" in data:
        # Normalize legacy key to standardized key
        data["user_id"] = data.pop("id")
    current = _user_context.get() or {}
    current.update(data)
    _user_context.set(current)


def get_current_user() -> Dict[str, Any]:
    """
    Get current user data from context.

    Returns:
        Copy of current user data dictionary

    Raises:
        RuntimeError: If no user context is set
    """
    context = _user_context.get()
    if not context:
        raise RuntimeError("No user context set. Ensure authentication is properly initialized.")
    return context.copy()


def update_current_user(updates: Dict[str, Any]) -> None:
    """
    Update current user data with new values.

    Args:
        updates: Dictionary of updates to apply
    """
    current = _user_context.get() or {}
    current.update(updates)
    _user_context.set(current)


def clear_user_context() -> None:
    """
    Clear the current user context.

    Useful for cleanup or testing scenarios.
    """
    _user_context.set(None)


def has_user_context() -> bool:
    """
    Check if user context is currently set.

    Returns:
        True if user context exists, False otherwise
    """
    context = _user_context.get()
    return context is not None and "user_id" in context


def get_current_segment_id() -> Optional[str]:
    """
    Get current segment ID from context.

    Returns None if no segment is active (first message before segment creation).
    """
    return _current_segment_id.get()


def set_current_segment_id(segment_id: str) -> contextvars.Token:
    """
    Set current segment ID in context.

    Called at conversation entry points (websocket/HTTP chat handlers)
    when the active segment is known.

    Returns:
        Token that can be used with reset_current_segment_id()
    """
    return _current_segment_id.set(segment_id)


# ============================================================
# ModelConfig - fixed database-backed model routes
# ============================================================

@dataclass(frozen=True)
class ModelConfig:
    """One of MIRA's five fixed model routes.

    Routes are capability-addressed: a caller names the capability it needs
    and the model_configs row owns dialect, model, endpoint, Vault key,
    default effort, and output-token ceiling for it. `other` is seeded to a
    different vendor than `primary` so it can consult an outside model.

    `api_key_name` is None when the route requires no credential. The column is
    `text NOT NULL` and stores '' for that case, so `load_model_configs()`
    normalises the sentinel to None here — the only place that has to know it.
    """

    name: str
    model: str
    dialect_name: str
    endpoint_url: str
    api_key_name: str | None
    effort: str | None
    max_tokens: int


_model_config_cache: dict[str, ModelConfig] | None = None
_MODEL_CONFIG_NAMES = frozenset({"primary", "fast", "batch", "assessment", "other"})


def load_model_configs() -> None:
    """Load and validate the complete fixed model-config set at startup."""
    global _model_config_cache
    from clients.llm.resolver import optional_api_key_name
    from clients.postgres_client import PostgresClient
    db = PostgresClient("mira_service")

    results = db.execute_query(
        "SELECT name, model, dialect_name, endpoint_url, api_key_name, effort, max_tokens FROM model_configs"
    )

    loaded = {
        row["name"]: ModelConfig(
            name=row["name"],
            model=row["model"],
            dialect_name=row["dialect_name"],
            endpoint_url=row["endpoint_url"],
            api_key_name=optional_api_key_name(
                row["api_key_name"],
                f"model_config '{row['name']}' api_key_name",
            ),
            effort=row["effort"],
            max_tokens=row["max_tokens"],
        )
        for row in results
    }

    if set(loaded) != _MODEL_CONFIG_NAMES:
        raise RuntimeError(
            "model_configs must contain exactly primary, fast, batch, assessment, "
            f"and other; found {sorted(loaded)}"
        )

    # D14 seeding constraint: `other` exists to consult an outside model. If
    # it resolves to primary's model, phoneafriend_tool degenerates into
    # self-review. Warn rather than fail — an operator with a single provider
    # available must still be able to boot (O-3).
    if loaded["other"].model == loaded["primary"].model:
        logger.warning(
            "model_configs route 'other' is seeded to the same model as 'primary' (%s); "
            "outside-model consultation will be consulting the same model it is asking for help",
            loaded["other"].model,
        )

    from clients.llm.resolver import ModelSelection
    from clients.llm.types import coerce_dialect_name, coerce_effort

    for model_config in loaded.values():
        dialect_name = coerce_dialect_name(model_config.dialect_name)
        effort = coerce_effort(model_config.effort) if model_config.effort else None
        ModelSelection(
            dialect_name=dialect_name,
            model=model_config.model,
            endpoint_url=model_config.endpoint_url,
            api_key_name=model_config.api_key_name,
            max_tokens=model_config.max_tokens,
            effort=effort,
            model_config_name=model_config.name,
        )

    from config import config

    config.api.validate_compaction_budget(loaded["primary"].max_tokens)
    _model_config_cache = loaded


def get_model_configs() -> dict[str, ModelConfig]:
    """Return all fixed model routes after startup loading."""
    if _model_config_cache is None:
        raise RuntimeError("Model configs not loaded. Call load_model_configs() at startup.")
    return dict(_model_config_cache)


def get_model_config(name: str) -> ModelConfig:
    """Resolve one fixed model route by name or raise."""
    configs = get_model_configs()
    try:
        return configs[name]
    except KeyError as error:
        raise KeyError(f"Unknown model_config '{name}'") from error


# ============================================================
# UserPreferences - Database-backed user settings
# ============================================================

def _config_default_timezone() -> str:
    """Timezone default from system config; lazy import avoids a module-level cycle."""
    from config.config_manager import config
    return config.system.timezone


class UserPreferences(BaseModel):
    """
    User preferences and profile data loaded from database.
    Cached in Valkey with invalidation on updates.
    """
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    timezone: str = Field(default_factory=_config_default_timezone)
    temperature_unit: str = Field(default="fahrenheit")
    memory_manipulation_enabled: bool = Field(default=True)
    created_at: Optional[datetime] = None


def get_user_preferences() -> UserPreferences:
    """
    Get current user's preferences with Valkey caching.

    Cache hierarchy:
    1. Valkey (shared across all contexts - WebSocket, HTTP, etc.)
    2. Database (source of truth)

    Valkey cache is invalidated on preference updates, ensuring
    all contexts see changes immediately.
    """
    import json
    from clients.valkey_client import get_valkey_client
    from clients.postgres_client import PostgresClient

    user_id = get_current_user_id()
    cache_key = f"user_prefs:{user_id}"

    # Check Valkey cache first (shared across all contexts)
    valkey = get_valkey_client()
    cached = valkey.get(cache_key)
    if cached:
        data = json.loads(cached)
        return UserPreferences(**data)

    # Cache miss - fetch from database
    db = PostgresClient('mira_service', user_id=user_id)
    result = db.execute_single(
        """SELECT first_name, last_name, timezone, temperature_unit, memory_manipulation_enabled, created_at
           FROM users WHERE id = %s""",
        (user_id,)
    )

    prefs = UserPreferences(
        first_name=result.get('first_name'),
        last_name=result.get('last_name'),
        timezone=result.get('timezone') or _config_default_timezone(),
        temperature_unit=result.get('temperature_unit') or 'fahrenheit',
        memory_manipulation_enabled=result.get('memory_manipulation_enabled', True),
        created_at=result.get('created_at'),
    )

    # Cache in Valkey with 5-minute TTL (safety net - invalidation handles freshness)
    valkey.set(cache_key, prefs.model_dump_json(), ex=300)

    return prefs


def invalidate_user_preferences_cache(user_id: str) -> None:
    """
    Invalidate the Valkey cache entry for a user's preferences.

    user_context owns the ``user_prefs:{user_id}`` cache-key format; callers
    that write preferences (e.g. the update_profile action) call this after
    persisting so every context sees the change immediately.
    """
    from clients.valkey_client import get_valkey_client
    get_valkey_client().delete(f"user_prefs:{user_id}")


# ============================================================
# Activity tracking (not a preference - computed value)
# ============================================================

def get_user_cumulative_activity_days() -> int:
    """
    Get current user's cumulative activity days with context caching.

    This is the canonical way to get "how many days" for scoring calculations.
    Returns activity days (not calendar days) to ensure vacation-proof decay.

    Context caching ensures we only query the database once per session,
    with subsequent calls returning the cached value.

    Returns:
        Cumulative activity days for current user

    Raises:
        RuntimeError: If no user context is set
    """
    # Check if already cached in context
    try:
        user_data = get_current_user()
        if 'cumulative_activity_days' in user_data:
            return user_data['cumulative_activity_days']
    except RuntimeError:
        raise RuntimeError("No user context set. Cannot get activity days without user context.")

    # Not cached - query user activity module and cache result
    user_id = get_current_user_id()

    from utils.user_activity import get_user_cumulative_activity_days as get_activity_days
    activity_days = get_activity_days(user_id)

    # Cache for subsequent calls
    update_current_user({'cumulative_activity_days': activity_days})

    return activity_days
