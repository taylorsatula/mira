"""
Self-edit rollback: the Python half of the launcher's trial-boot contract.

MIRA can edit its own code tree with bash_tool. The tree is a git repository
(deploy/python.sh initializes it and commits every install), so "last good" is
simply HEAD and an untested edit is simply an uncommitted change. Three actors
share a small state directory outside the tree (``MIRA_SELF_EDIT_STATE_DIR``,
set only by the supervisor configs deploy/finalize.sh writes):

- ``deploy/mira-launch.sh`` (before Python starts): an edited tree with no
  ``booting`` marker gets the marker and a trial boot; an edited tree whose
  marker survived from the previous start means that trial never completed, so
  the launcher stashes the change (never deletes it), writes ``result`` as
  failed with the tail of the boot log, and starts on the clean tree.
- ``main.py`` lifespan, at startup complete: ``commit_trial_if_booting()``
  commits the tree that just proved it boots, writes ``result`` as applied,
  and removes the marker.
- The running app: ``selfedit_tool`` records who asked for the restart, the
  restart handler exits the process with ``RESTART_EXIT_CODE`` once the turn is
  committed, and the trinket / heartbeat digest / turn-completion handler
  deliver and then clear ``result``.

``result`` format (written by both the launcher and this module): the first
line is ``<status> <ref>`` where status is ``applied`` (ref = commit hash) or
``failed`` (ref = stash commit hash); every following line is detail text.

Inactive mode: with the env var unset (dev checkout, Docker, an unsupervised
launch) or no ``.git`` in the tree, every entry point here is a no-op or
refusal — nothing commits, the POST gate keeps its configured failure action,
and the tool refuses with the reason.
"""
import logging
import os
import re
import signal
import subprocess
import threading
from pathlib import Path
from typing import Literal, TypedDict

from utils.timezone_utils import format_utc_iso, utc_now

logger = logging.getLogger(__name__)

STATE_DIR_ENV = "MIRA_SELF_EDIT_STATE_DIR"

# The running code tree. Self-edit operates on exactly the tree this process
# was loaded from — never a configured path that could point elsewhere.
APP_ROOT = Path(__file__).resolve().parents[1]

# Exit code for a requested restart. Must be non-zero: systemd
# (Restart=on-failure) and launchd (KeepAlive SuccessfulExit=false) restart
# only on a failed exit. 75 is EX_TEMPFAIL.
RESTART_EXIT_CODE = 75

# Every git call is bounded; a hung git must never wedge startup or a tool call.
GIT_TIMEOUT_SECONDS = 60

_BOOTING = "booting"
_RESULT = "result"
_REQUESTED_BY = "requested_by"
_OFFERED = "offered"

_restart_requested = threading.Event()

# Terminal color codes the console log handler emits into the boot log.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class SelfEditResult(TypedDict):
    """Outcome of the last trial boot, as recorded in the state directory."""
    status: Literal["applied", "failed"]
    ref: str
    detail: str
    requested_by: str | None


def state_dir() -> Path | None:
    """The state directory, or None when the supervisor did not configure one."""
    value = os.environ.get(STATE_DIR_ENV)
    return Path(value) if value else None


def inactive_reason() -> str | None:
    """Why self-edit is unavailable in this process, or None when it is active."""
    directory = state_dir()
    if directory is None:
        return (
            f"{STATE_DIR_ENV} is not set: MIRA was not started by a supervisor "
            "configured for self-edit rollback (dev checkout, Docker, or an "
            "unsupervised launch). Code changes cannot be applied or rolled back here."
        )
    if not directory.is_dir():
        return f"self-edit state directory {directory} does not exist"
    if not (APP_ROOT / ".git").is_dir():
        return f"the code tree {APP_ROOT} is not a git repository"
    return None


def is_active() -> bool:
    return inactive_reason() is None


def _require_state_dir() -> Path:
    reason = inactive_reason()
    if reason is not None:
        raise RuntimeError(f"Self-edit is unavailable: {reason}")
    directory = state_dir()
    assert directory is not None
    return directory


def git(*args: str) -> str:
    """Run one git command in the app tree; return stdout or raise with stderr."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(APP_ROOT), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"git {' '.join(args)} exceeded {GIT_TIMEOUT_SECONDS}s") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout


def uncommitted_changes() -> list[str]:
    """Porcelain status lines for every uncommitted change, untracked files included."""
    output = git("status", "--porcelain", "--untracked-files=all")
    return [line for line in output.splitlines() if line.strip()]


def protected_tree_names() -> frozenset[str]:
    """Top-level names in the tree that git does not track, so no rollback covers them.

    Read from ``.git/info/exclude`` — the installed copy deploy/python.sh writes,
    outside the editable tree — so an edit to a tracked file cannot widen what
    bash_tool may delete. Glob lines (``*.pyc``) name no single path and are
    skipped. Raises when the file is missing: an active self-edit tree without
    its exclude list is a broken install, not an empty protection set.
    """
    exclude = APP_ROOT / ".git" / "info" / "exclude"
    names = {".git"}
    for raw in exclude.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or any(ch in line for ch in "*?[!"):
            continue
        names.add(line.strip("/").split("/")[0])
    return frozenset(names)


# -- trial boot -------------------------------------------------------------


def trial_in_progress() -> bool:
    """True while an edited tree is mid-boot (the launcher's marker is present)."""
    directory = state_dir()
    return directory is not None and (directory / _BOOTING).exists()


def commit_trial_if_booting() -> None:
    """At startup complete: commit the tree that just proved it boots.

    Commit first, then write ``result``, then remove the marker: a crash between
    the commit and the marker removal leaves a clean tree plus a marker, which
    the launcher resolves on the next start (it records HEAD as applied). A
    commit failure removes the marker anyway — the code booted, so the launcher
    must not stash it; the change stays uncommitted and the next restart runs it
    as a trial again.
    """
    if not is_active() or not trial_in_progress():
        return
    directory = _require_state_dir()
    try:
        if uncommitted_changes():
            git("add", "-A")
            git("commit", "--quiet", "-m", f"self-edit applied {format_utc_iso(utc_now())}")
        commit = git("rev-parse", "HEAD").strip()
        _write_result("applied", commit, "")
        logger.warning("Self-edit trial boot passed; committed %s", commit)
    except RuntimeError as exc:
        logger.error(
            "Self-edit trial boot passed but the commit failed (%s). The change "
            "stays uncommitted and will be re-tried as a trial on the next restart; "
            "the user is told it booted.", exc,
        )
        _write_result("applied", "uncommitted", f"commit failed: {exc}")
    finally:
        (directory / _BOOTING).unlink(missing_ok=True)


# -- restart ----------------------------------------------------------------


def record_restart_request(user_id: str) -> None:
    """Remember who asked, so the outcome is delivered to that user."""
    directory = _require_state_dir()
    (directory / _REQUESTED_BY).write_text(user_id, encoding="utf-8")


def request_process_restart() -> None:
    """Mark the restart and SIGTERM this process; main() exits with RESTART_EXIT_CODE."""
    _restart_requested.set()
    logger.warning("Self-edit restart: sending SIGTERM to pid %d", os.getpid())
    os.kill(os.getpid(), signal.SIGTERM)


def restart_requested() -> bool:
    return _restart_requested.is_set()


# -- result -----------------------------------------------------------------


def _write_result(status: str, ref: str, detail: str) -> None:
    """Record a new outcome; the newest outcome replaces any undelivered one."""
    directory = _require_state_dir()
    path = directory / _RESULT
    tmp = directory / f".{_RESULT}.tmp"
    tmp.write_text(f"{status} {ref}\n{detail}", encoding="utf-8")
    tmp.replace(path)
    (directory / _OFFERED).unlink(missing_ok=True)


def read_result() -> SelfEditResult | None:
    """The pending outcome, or None when there is nothing to report.

    Raises ValueError on a malformed first line: the launcher and this module
    are the only writers, so a bad tag is a defect to surface, not a no-op.
    """
    directory = state_dir()
    if directory is None:
        return None
    path = directory / _RESULT
    if not path.exists():
        return None
    head, _, detail = path.read_text(encoding="utf-8", errors="replace").partition("\n")
    status, _, ref = head.strip().partition(" ")
    if status not in ("applied", "failed") or not ref:
        raise ValueError(f"Malformed self-edit result header in {path}: {head!r}")
    requested_by_path = directory / _REQUESTED_BY
    requested_by = (
        requested_by_path.read_text(encoding="utf-8").strip()
        if requested_by_path.exists() else None
    )
    return SelfEditResult(
        status=status,  # type: ignore[typeddict-item]
        ref=ref.strip(),
        detail=_ANSI_ESCAPE.sub("", detail).strip(),
        requested_by=requested_by or None,
    )


def result_for_user(user_id: str) -> SelfEditResult | None:
    """The pending outcome if it belongs to this user.

    An outcome with no recorded requester (a restart nobody requested through
    selfedit_tool) is shown to every user — on a single-account install, the one.
    """
    result = read_result()
    if result is None:
        return None
    if result["requested_by"] is not None and result["requested_by"] != user_id:
        return None
    return result


def heartbeat_delivery_pending(user_id: str) -> bool:
    """True when this user's outcome has not yet been offered to a heartbeat wake."""
    directory = state_dir()
    if directory is None or result_for_user(user_id) is None:
        return False
    return not (directory / _OFFERED).exists()


def mark_offered_to_heartbeat() -> None:
    directory = _require_state_dir()
    (directory / _OFFERED).touch()


def clear_result() -> None:
    """Delivered: remove the outcome and its delivery bookkeeping."""
    directory = _require_state_dir()
    for name in (_RESULT, _REQUESTED_BY, _OFFERED):
        (directory / name).unlink(missing_ok=True)
