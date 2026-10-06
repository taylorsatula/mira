"""
Update Check API endpoint - version update notification.

Public endpoint that checks if a newer MIRA version is available.
Logs check requests with timestamp and IP to data/update_checks.log for analytics.
"""
import json
import logging
import urllib.request
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from auth.api import get_current_user
from auth.types import APITokenContext, SessionData
from utils.timezone_utils import utc_now, format_utc_iso
from utils.release_identity import get_current_version, version_sort_key
from utils.update_check import UpdateStatus, get_update_status

REMOTE_UPDATE_URL = "https://miraos.org/check_update"

logger = logging.getLogger(__name__)

# Dedicated file logger for update check analytics
_update_log_path = Path(__file__).parent.parent.parent / "data" / "update_checks.log"
_update_log_path.parent.mkdir(parents=True, exist_ok=True)

update_logger = logging.getLogger("update_checks")
update_logger.setLevel(logging.INFO)
# Rotate at 10MB, keep 5 backups
_handler = RotatingFileHandler(_update_log_path, maxBytes=10_000_000, backupCount=5)
_handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))
update_logger.addHandler(_handler)

router = APIRouter()


class UpdateCheckResponse(BaseModel):
    """Response for version update check."""
    update_available: bool
    current_version: str | None = Field(default=None, description="Installed version of this instance")
    latest_version: str | None = Field(default=None, description="Latest version if update available")
    checked_at: str = Field(..., description="ISO-8601 timestamp of check")


def get_client_ip(request: Request) -> str:
    """Return the socket peer address. X-Forwarded-For is ignored: client-controlled
    and no proxy is deployed."""
    return request.client.host if request.client else "unknown"


def _sanitize_log_field(value: str) -> str:
    """Make a client-supplied log value forge-resistant: one physical line,
    control characters visibly escaped, nothing that could splice log rows."""
    stripped = value.strip().replace("\x00", "")
    return stripped.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


@router.get("/check_update", response_model=UpdateCheckResponse)
def check_update_endpoint(request: Request, version: str = "") -> UpdateCheckResponse:
    """
    Check if a newer MIRA version is available.

    Public endpoint - no authentication required.
    Logs: timestamp, client IP, version being checked.
    """
    latest = get_current_version()
    client_ip = get_client_ip(request)

    # Scrubbed, not rejected — forgeries stay visible on one line, no free 4xx oracle.
    safe_version = _sanitize_log_field(version)
    update_logger.info(f"ip={client_ip}\tversion={safe_version or 'none'}\tlatest={latest}")

    # No version provided or invalid - can't compare
    if not version:
        return UpdateCheckResponse(
            update_available=False,
            latest_version=latest,
            checked_at=format_utc_iso(utc_now())
        )

    installed_key = version_sort_key(version)
    latest_key = version_sort_key(latest)

    if installed_key is None or latest_key is None:
        logger.warning(
            f"Unparseable version in update check: installed={safe_version!r} "
            f"latest={latest!r} — declining to compare"
        )
    elif latest_key > installed_key:
        return UpdateCheckResponse(
            update_available=True,
            latest_version=latest,
            checked_at=format_utc_iso(utc_now())
        )

    return UpdateCheckResponse(
        update_available=False,
        latest_version=latest,
        checked_at=format_utc_iso(utc_now())
    )


@router.get("/check_remote_update", response_model=UpdateCheckResponse)
def check_remote_update(request: Request) -> UpdateCheckResponse:
    """Check miraos.org for a newer version. Proxies server-side to avoid browser CORS."""
    current = get_current_version()
    client_ip = get_client_ip(request)
    update_logger.info(f"ip={client_ip}\tversion={current}\tcheck=remote")

    try:
        url = f"{REMOTE_UPDATE_URL}?version={current}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return UpdateCheckResponse(
                update_available=data.get("update_available", False),
                current_version=current,
                latest_version=data.get("latest_version"),
                checked_at=format_utc_iso(utc_now())
            )
    except Exception:
        pass

    return UpdateCheckResponse(
        update_available=False,
        current_version=current,
        checked_at=format_utc_iso(utc_now())
    )


@router.get("/update_status", response_model=UpdateStatus)
def update_status_endpoint(
    current_user: SessionData | APITokenContext = Depends(get_current_user),
) -> UpdateStatus:
    """
    The cached verdict of the daily release check (`utils/update_check.py`).

    Unlike the public `/check_update` (which compares a caller-supplied version
    against this instance's own), this reports what the *server* learned by
    asking GitHub — the fact the TUI renders as an update notice. The TUI polls
    it at startup and every 24 h.

    Anonymous callers are rejected: the endpoint is mounted under the
    authenticated `/v0/api` surface, and the notice is a per-install fact a
    session or API token is already required to read anything else with.
    """
    return get_update_status()
