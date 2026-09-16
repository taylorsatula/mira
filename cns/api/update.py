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

from fastapi import APIRouter, Request
from packaging import version as pkg_version
from pydantic import BaseModel, Field

from utils.timezone_utils import utc_now, format_utc_iso

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


def get_latest_version() -> str:
    """Read latest version from VERSION file."""
    version_file = Path(__file__).parent.parent.parent / "VERSION"
    return version_file.read_text().strip()


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
    latest = get_latest_version()
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

    try:
        installed = pkg_version.parse(version)
        latest_parsed = pkg_version.parse(latest)

        if latest_parsed > installed:
            return UpdateCheckResponse(
                update_available=True,
                latest_version=latest,
                checked_at=format_utc_iso(utc_now())
            )
    except pkg_version.InvalidVersion as e:
        logger.warning(f"Invalid version format in update check: {version} - {e}")

    return UpdateCheckResponse(
        update_available=False,
        checked_at=format_utc_iso(utc_now())
    )


@router.get("/check_remote_update", response_model=UpdateCheckResponse)
def check_remote_update(request: Request) -> UpdateCheckResponse:
    """Check miraos.org for a newer version. Proxies server-side to avoid browser CORS."""
    current = get_latest_version()
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
