"""
The startup announcement every user sees until an operator clears it.

The announcement lives in `config/announcement.json` — `announcement.sample.json`
is its filled shape — and `load_announcement()` reads it once into a module global
from the `main.py` lifespan. `get_cached_announcement()` serves that cache and
loads it defensively on a first call that arrives before startup, so posting or
removing an announcement takes a restart to take effect.

The announcement is an optional feature, and this is a sanctioned degrade: a
missing file logs at info, malformed JSON logs a warning, and either leaves the
cache empty. A null `id` or `message` means no active announcement. None of these
fails boot; the surfaces that serve the cache tolerate absence.
"""
import json
import logging
from pathlib import Path
from typing import Optional, TypedDict

logger = logging.getLogger(__name__)

_cached_announcement: Optional[dict] = None
_loaded = False


class Announcement(TypedDict):
    id: str
    message: str


def load_announcement() -> None:
    """Load announcement from config/announcement.json into module cache.

    Called once at startup. If file is missing, malformed, or has null id/message,
    the cached value stays None (no announcement).
    """
    global _cached_announcement, _loaded

    config_path = Path(__file__).parent / "announcement.json"

    try:
        data = json.loads(config_path.read_text())
        ann_id = data.get("id")
        ann_message = data.get("message")

        if ann_id and ann_message:
            _cached_announcement = {"id": ann_id, "message": ann_message}
            logger.info(f"Announcement loaded: id={ann_id!r}")
        else:
            _cached_announcement = None
            logger.info("No active announcement (id or message is null)")
    except FileNotFoundError:
        _cached_announcement = None
        logger.info("No announcement config file found")
    except (json.JSONDecodeError, Exception) as e:
        _cached_announcement = None
        logger.warning(f"Failed to parse announcement config: {e}")

    _loaded = True


def get_cached_announcement() -> Optional[Announcement]:
    """Return the cached announcement, or None if no active announcement."""
    if not _loaded:
        load_announcement()
    return _cached_announcement
