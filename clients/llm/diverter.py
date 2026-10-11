"""Provider failover diverter: process-global trip state for the backup target.

One global backup provider (config.api.failover_*) serves all five routes
while the diverter is tripped. Trip/flip mechanics:

- Trip: N consecutive `ProviderRetryableError`s (stalls included) recorded
  against the same primary endpoint. Callers — `clients/llm_provider.py` —
  record only for the primary selection's endpoint, and only when not already
  tripped, so a single success resets the counter and backup-endpoint failures
  are structurally unrecordable: there is exactly one diversion level and no
  recursion into failover-of-failover.
- Flip back: the recovery probe job (`utils/failover_jobs.py`) probes every
  deduped primary target while tripped; `failover_flipback_consecutive_passes`
  consecutive full passes return traffic to the primary. One failed probe
  resets the pass counter.

State is process-local by design (a restart re-parks the boot gate on the
primary until it answers, then the counter re-trips within seconds — the
accepted trade for zero new infrastructure). `ProviderAuthError` and
`ProviderContextOverflowError` are never recorded: a misconfigured credential
or an oversized request must not divert around itself.

The diverter holds no infrastructure and publishes no events; the TOAST journal
lines are the operator surface. `utils/power_on_self_test.py` (boot gate) and
the resolver consult `tripped()`; at boot the state is always clean.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from config import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_consecutive_failures: dict[str, int] = {}
_tripped = False
_tripped_since: Optional[float] = None
_trip_reason: Optional[str] = None
_recovery_passes = 0


def enabled() -> bool:
    """True when the diverter feature is configured on."""
    return config.api.failover_enabled


def tripped() -> bool:
    """True while traffic should divert to the failover target."""
    return _tripped


def trip_reason() -> Optional[str]:
    """Human-readable trip cause for logs; None while healthy."""
    return _trip_reason


def tripped_since() -> Optional[float]:
    """`time.time()` of the trip, for diagnostics; None while healthy."""
    return _tripped_since


def record_success(endpoint: str) -> None:
    """Reset the failure counter for a primary endpoint after a clean call.

    No-op while tripped (traffic is on the backup; nothing about the backup's
    health is counted here) and when the feature is disabled.
    """
    if _tripped or not config.api.failover_enabled:
        return
    with _lock:
        _consecutive_failures.pop(endpoint, None)


def record_failure(endpoint: str, error: BaseException) -> None:
    """Count one retryable primary failure; trip at the configured threshold."""
    if _tripped or not config.api.failover_enabled:
        return
    threshold = config.api.failover_trip_consecutive_failures
    with _lock:
        count = _consecutive_failures.get(endpoint, 0) + 1
        _consecutive_failures[endpoint] = count
        if count < threshold:
            logger.warning(
                "Provider failure %d/%d for %s: %s",
                count, threshold, endpoint, error,
            )
            return
        _trip_locked(endpoint, error)


def _trip_locked(endpoint: str, error: BaseException) -> None:
    global _tripped, _tripped_since, _trip_reason, _recovery_passes
    _tripped = True
    _tripped_since = time.time()
    _trip_reason = f"{type(error).__name__} from {endpoint}: {error}"
    _recovery_passes = 0
    logger.toast(
        "PROVIDER FAILOVER TRIPPED: routing all LLM routes to the backup target "
        "(%s @ %s). Cause: %s. The recovery probe rechecks the primary every "
        "%ss.",
        config.api.failover_model,
        config.api.failover_endpoint_url,
        _trip_reason,
        config.api.failover_recovery_probe_seconds,
    )


def record_probe_pass() -> bool:
    """Count one full passing primary probe; flip back at the threshold.

    Returns True when this pass completed the flip-back.
    """
    global _recovery_passes
    if not _tripped:
        return False
    threshold = config.api.failover_flipback_consecutive_passes
    with _lock:
        _recovery_passes += 1
        if _recovery_passes < threshold:
            logger.toast(
                "Failover recovery probe passed (%d/%d) — staying on the backup "
                "until the primary passes %d consecutive probes.",
                _recovery_passes, threshold, threshold,
            )
            return False
        _flip_locked()
        return True


def record_probe_fail(error: BaseException) -> None:
    """Reset the recovery pass counter on a failed primary probe."""
    global _recovery_passes
    if not _tripped:
        return
    with _lock:
        if _recovery_passes:
            logger.warning(
                "Failover recovery probe failed (%s) — pass counter reset.",
                error,
            )
        _recovery_passes = 0


def _flip_locked() -> None:
    global _tripped, _tripped_since, _trip_reason, _recovery_passes
    _tripped = False
    _tripped_since = None
    _trip_reason = None
    _recovery_passes = 0
    _consecutive_failures.clear()
    logger.toast(
        "PROVIDER FAILOVER RESTORED: primary targets passed the recovery probe; "
        "all LLM routes returned to model_configs."
    )


def reset() -> None:
    """Clear all diverter state (probe/test isolation; not used at runtime)."""
    global _tripped, _tripped_since, _trip_reason, _recovery_passes
    with _lock:
        _tripped = False
        _tripped_since = None
        _trip_reason = None
        _recovery_passes = 0
        _consecutive_failures.clear()
