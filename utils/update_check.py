"""
Release-availability check: fetches the newest published release tag from
GitHub, compares it against this tree's ``VERSION``, and caches the verdict
for the HTTP surface the TUI polls (``cns/api/update.py``
``GET /update_status``). The identity halves both live in
``utils/release_identity.py`` (stdlib-only, shared with the TUI's
``mira update`` subcommand); this module owns the cadence and the cache.

Cadence: one calendar job every 24 h, plus one check at boot. The boot check is
run by the caller (``main.py``) rather than by the job, because an
``IntervalTrigger``'s first fire is one full interval out — without it a
restarted instance would show no verdict for a day.
"""
import logging

from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel, Field

from utils.logging_config import TOAST
from utils.release_identity import (
    fetch_latest_release_tag,
    get_current_version,
    version_sort_key,
)
from utils.timezone_utils import utc_now, format_utc_iso

logger = logging.getLogger(__name__)

CHECK_INTERVAL_HOURS = 24


class UpdateStatus(BaseModel):
    """Cached verdict of the most recent release check."""

    update_available: bool = Field(
        description="True when the newest published release sorts above this instance's VERSION"
    )
    current_version: str = Field(description="This instance's VERSION file contents")
    latest_version: str | None = Field(
        default=None,
        description="Newest published release name, without the leading v; None until a check succeeds",
    )
    checked_at: str | None = Field(
        default=None,
        description="ISO-8601 UTC time of the last successful check, or None when none has succeeded",
    )
    detail: str | None = Field(
        default=None,
        description="Why no verdict is available, when the most recent check did not produce one",
    )


_status_cache: UpdateStatus | None = None


def check_for_update() -> UpdateStatus:
    """
    Run one release check, cache the verdict, and return it.

    Never raises. The notice is advisory: a provider outage, a rate limit, or a
    malformed response must not take down a scheduled job or the boot path, so
    a failed check returns the last known verdict with ``detail`` naming the
    failure and logs the consequence — how long until the next attempt.

    Returns:
        The newly computed verdict, or the previous one when this check failed.
    """
    global _status_cache

    try:
        current = get_current_version()
        latest = fetch_latest_release_tag()
        current_key = version_sort_key(current)
        latest_key = version_sort_key(latest)
        if current_key is None or latest_key is None:
            raise ValueError(
                f"unparseable release name (current={current!r}, latest={latest!r})"
            )

        available = latest_key > current_key
        _status_cache = UpdateStatus(
            update_available=available,
            current_version=current,
            latest_version=latest,
            checked_at=format_utc_iso(utc_now()),
        )
        if available:
            # Always-visible: an operator should learn this from the journal,
            # not only from the TUI status row.
            logger.log(
                TOAST,
                "Update available: MIRA %s is published (this instance runs %s). "
                "Update in place with `mira update`.",
                latest, current,
            )
        else:
            logger.info("Update check: %s is the newest published release", current)
    except Exception as error:
        logger.warning(
            "Update check failed (%s: %s); no update will be offered until the "
            "next check in %dh",
            type(error).__name__, error, CHECK_INTERVAL_HOURS,
        )
        if _status_cache is None:
            try:
                fallback_version = get_current_version()
            except OSError:
                fallback_version = ""
            _status_cache = UpdateStatus(
                update_available=False,
                current_version=fallback_version,
                detail=f"{type(error).__name__}: {error}",
            )

    return _status_cache


def get_update_status() -> UpdateStatus:
    """
    The cached verdict, or a never-checked status when no check has completed.

    ``update_available`` is False in the never-checked case: absence of a
    verdict is not evidence of an available update, and a consumer that shows a
    notice must never show one on a guess.
    """
    if _status_cache is not None:
        return _status_cache
    try:
        current = get_current_version()
    except OSError:
        current = ""
    return UpdateStatus(
        update_available=False,
        current_version=current,
        detail="no release check has completed yet",
    )


def register_update_check_job(scheduler_service) -> None:
    """
    Register the 24-hour release check.

    The immediate boot check is the caller's (`main.py`), because an
    `IntervalTrigger` fires its first time one full interval from registration.
    """
    scheduler_service.register_job(
        job_id="update_check",
        func=check_for_update,
        trigger=IntervalTrigger(hours=CHECK_INTERVAL_HOURS),
        component="system",
        description=f"Check GitHub for a newer published release every {CHECK_INTERVAL_HOURS}h",
    )
    logger.info("Registered release update check (%dh interval)", CHECK_INTERVAL_HOURS)
