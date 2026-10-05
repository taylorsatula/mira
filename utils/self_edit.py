"""
Self-edit rollback: the Python half of the launcher's trial-boot contract.

MIRA can edit its own code tree with bash_tool. The tree is a git repository
(deploy/python.sh initializes it and commits every install), so "last good" is
simply HEAD and an untested edit is simply an uncommitted change. The launcher
and this module share a small state directory outside the tree
(``MIRA_SELF_EDIT_STATE_DIR``, set only by the supervisor configs
deploy/finalize.sh writes):

- ``deploy/mira-launch.sh`` (before Python starts): an edited tree with no
  ``booting`` marker gets the marker and a trial boot; an edited tree whose
  marker survived from the previous start means that trial never completed, so
  the launcher stashes the change (never deletes it), writes ``result`` as
  failed with the tail of the boot log, and starts on the clean tree.
- ``main.py`` lifespan, at startup complete: ``commit_trial_if_booting()``
  commits the tree that just proved it boots and writes ``result`` as applied;
  ``deliver_result()`` then turns ``result`` into an activity-feed item for the
  user who asked — the HUD (AsyncActivityTrinket), the heartbeat digest, and
  sidebaragents_tool's dismiss/resolve already serve that feed — and makes the
  user's heartbeat due now so the outcome is announced unprompted.
- ``selfedit_tool`` request_restart: records the requester and
  ``schedule_restart_after_turn()`` exits the process once the turn is over.

``result`` format (written by both the launcher and this module): the first
line is ``<status> <ref>`` where status is ``applied`` (ref = commit hash) or
``failed`` (ref = stash commit hash); every following line is detail text.

Inactive mode: with the env var unset (dev checkout, Docker, an unsupervised
launch) or no ``.git`` in the tree, every entry point here is a no-op or
refusal — nothing commits, the POST gate keeps its configured failure action,
and the tool refuses with the reason.
"""
import contextvars
import logging
import os
import re
import signal
import subprocess
import threading
import time
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

# Upper bound on waiting for the requesting turn to release its lock before
# restarting. A turn holding it longer is wedged; the restart proceeds.
_TURN_WAIT_SECONDS = 600
_TURN_POLL_SECONDS = 0.5

# Activity-feed item for the outcome. The feed renders the escalation reason in
# the HUD on every turn until the item is resolved, so the boot-log tail is cut
# to its end — where a traceback names the failure.
_ACTIVITY_INTERFACE = "self_edit"
_ACTIVITY_AGENT = "mira-launch"
_REASON_CHARS = 2000

_BOOTING = "booting"
_RESULT = "result"
_REQUESTED_BY = "requested_by"

_restart_scheduled = threading.Event()
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


def schedule_restart_after_turn(user_id: str) -> None:
    """Restart once the requesting user's turn is over; record who asked.

    The tool call runs inside the turn, which holds the user's request lock
    until its terminal frame is queued (shutdown drains queued frames), so a
    background thread waits — bounded — to acquire that lock, holds it so no
    new turn starts in the shutdown window (the startup Valkey flush clears
    it), and then restarts. Every path ends in the restart: the user was told
    MIRA is restarting, and a lock that cannot be waited on only costs the tail
    of the reply. Repeat requests in one process are no-ops.
    """
    directory = _require_state_dir()
    (directory / _REQUESTED_BY).write_text(user_id, encoding="utf-8")
    if _restart_scheduled.is_set():
        return
    _restart_scheduled.set()
    context = contextvars.copy_context()
    threading.Thread(
        target=context.run, args=(_restart_when_turn_ends, user_id),
        name="self-edit-restart", daemon=True,
    ).start()


def _restart_when_turn_ends(user_id: str) -> None:
    from utils.distributed_lock import UserRequestLock

    try:
        lock = UserRequestLock(ttl=_TURN_WAIT_SECONDS)
        deadline = time.monotonic() + _TURN_WAIT_SECONDS
        while time.monotonic() < deadline:
            if lock.acquire(user_id) is not None:
                break
            time.sleep(_TURN_POLL_SECONDS)
        else:
            logger.error(
                "Self-edit restart: user %s's turn still held its lock after %ds; "
                "restarting anyway — an in-flight turn, if any, is cut off",
                user_id, _TURN_WAIT_SECONDS,
            )
    except Exception:
        logger.error(
            "Self-edit restart: waiting on user %s's turn lock failed; restarting "
            "without waiting — the end of the reply may not reach the client",
            user_id, exc_info=True,
        )
    _restart_requested.set()
    logger.warning("Self-edit restart: sending SIGTERM to pid %d", os.getpid())
    os.kill(os.getpid(), signal.SIGTERM)


def restart_requested() -> bool:
    return _restart_requested.is_set()


# -- result -----------------------------------------------------------------


def _write_result(status: str, ref: str, detail: str) -> None:
    directory = _require_state_dir()
    tmp = directory / f".{_RESULT}.tmp"
    tmp.write_text(f"{status} {ref}\n{detail}", encoding="utf-8")
    tmp.replace(directory / _RESULT)


def read_result() -> SelfEditResult | None:
    """The undelivered outcome, or None when there is nothing to report.

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


def deliver_result() -> None:
    """At startup: hand the outcome to the user who asked, through the activity feed.

    Writes one ``sidebar_activity`` item (applied → ``handled``; failed →
    ``escalated`` with the boot-log tail as the reason) and stamps the user's
    heartbeat wake to now, so the next dispatcher pass wakes MIRA to announce
    it. An outcome with no recorded requester (a restart that did not come
    through selfedit_tool) goes to every user with an active segment — on a
    single-account install, the one. The ``result`` file is removed only after
    every item is written; a failure leaves it for the next startup to retry and
    never fails startup, because the code it reports on is already running.
    """
    try:
        result = read_result()
        if result is None or not is_active():
            return
        from agents.base import ensure_activity_schema, upsert_activity_record
        from cns.infrastructure.continuum_repository import get_continuum_repository
        from utils.user_context import clear_user_context, set_current_user_id
        from utils.userdata_manager import get_user_data_manager

        repository = get_continuum_repository()
        active = {
            str(segment["user_id"]): str(segment["continuum_id"])
            for segment in repository.find_all_active_segments_admin()
        }
        recipients = [result["requested_by"]] if result["requested_by"] else list(active)

        if result["status"] == "applied":
            status, reason = "handled", None
            summary = (
                f"Your code change was applied: MIRA restarted on it and started "
                f"(commit {result['ref']}). Starting is not working — check the "
                f"change does what was asked, tell the user, then resolve this item "
                f"(thread_id {result['ref']})."
            )
        else:
            status, reason = "escalated", result["detail"][-_REASON_CHARS:] or None
            summary = (
                f"Your code change FAILED: MIRA could not start on it. The change is "
                f"stashed as {result['ref']} and MIRA is running the previous code. "
                f"Tell the user what failed and whether to fix and retry or drop it, "
                f"then resolve this item (thread_id {result['ref']})."
            )

        for user_id in recipients:
            set_current_user_id(user_id)
            try:
                db = get_user_data_manager(user_id)
                ensure_activity_schema(db)
                upsert_activity_record(
                    db,
                    interface_name=_ACTIVITY_INTERFACE,
                    thread_id=result["ref"],
                    agent_id=_ACTIVITY_AGENT,
                    summary=summary,
                    status=status,
                    escalation_reason=reason,
                )
                if user_id in active:
                    repository.set_heartbeat_wake_at(
                        active[user_id], user_id, format_utc_iso(utc_now())
                    )
            finally:
                clear_user_context()

        directory = _require_state_dir()
        for name in (_RESULT, _REQUESTED_BY):
            (directory / name).unlink(missing_ok=True)
        logger.warning(
            "Self-edit outcome %s %s delivered to %d user(s)",
            result["status"], result["ref"], len(recipients),
        )
    except Exception:
        logger.error(
            "Self-edit outcome could not be delivered; the user is not told whether "
            "their code change applied until a later startup retries "
            "(the result file stays in %s)", state_dir(), exc_info=True,
        )
