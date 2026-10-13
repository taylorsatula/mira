"""Process-self restart: the one sanctioned way MIRA restarts its own process.

bash_tool's ``restart_mira`` parameter (operation ``run``) is the only
in-process path that requests a restart. A request is a process-wide latch,
not an immediate action: the requesting turn must still deliver its tool
result and final reply, so the restart fires only once every turn in flight
has finished. The liveness gauge is maintained at the two turn boundaries
(``enter_turn``/``exit_turn``): the WebSocket turn
(``cns/api/websocket_chat.py:process_turn``) and the HTTP chat turn
(``cns/api/chat.py:process_request`` — the MCP check-in rides this endpoint);
heartbeat brief runs are agent-side on a report-only schema and cannot call
bash_tool, so no third boundary exists. Together those cover
every surface that can run bash_tool, so a restart request cannot fire ahead
of the turn that requested it — not even when a concurrent other-user turn
ends first in multi-user mode — and it cannot outlive that turn by more than
one quiescence boundary.

The mechanism is a graceful SIGTERM to this process followed by a dedicated
non-zero exit code. Hypercorn installs the SIGTERM handler (see main.py), so
the signal runs the graceful shutdown — in-flight requests drain (bounded by
hypercorn's ``graceful_timeout``), the lifespan shutdown closes WebSocket
connections — and main() then exits ``RESTART_EXIT_CODE`` so the deployment
supervisor starts a fresh process. Every supervised layout treats that exit
as a restartable failure: systemd ``Restart=on-failure`` (deploy/finalize.sh),
the macOS LaunchAgent ``KeepAlive`` with ``SuccessfulExit=false``, and s6,
which restarts on any exit. A foreground run without a supervisor simply
exits and stays down; the latch is still honoured (the process stops
serving), but re-launching is the operator's step.

The restart drops every client connection. Clients reconnect automatically
(the TUI's reconnect loop covers the supervisor delay plus the boot gate), and
all durable state — conversations, memories, reminders, background-job
records — lives in Postgres/Valkey/user SQLite, so a restart loses only
in-process state.
"""

import logging
import os
import signal
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Exit code main() uses when the server shut down because a restart was
# requested. Deliberately distinct from every other exit path (1 = startup
# failure) so journalctl can tell a requested restart from a crash.
RESTART_EXIT_CODE = 75

_lock = threading.Lock()
_requested_reason: Optional[str] = None
_triggered = False
_active_turns = 0


def request_restart(reason: str) -> None:
    """Record that this process should restart once the turns in flight end.

    Idempotent: the first request wins and later ones are no-ops, so any
    number of restart_mira calls in one turn schedule exactly one restart.
    """
    global _requested_reason
    with _lock:
        if _requested_reason is not None:
            return
        _requested_reason = reason
    logger.warning("In-process restart requested: %s", reason)


def restart_pending() -> bool:
    """True when a restart has been requested (fired or not).

    main() reads this after hypercorn's graceful shutdown returns: a pending
    request means the shutdown was self-initiated and the process must exit
    RESTART_EXIT_CODE so the supervisor respawns it.
    """
    with _lock:
        return _requested_reason is not None


def enter_turn() -> None:
    """Register one turn as in flight.

    Called at every turn boundary that can contain a bash_tool call, before
    the turn body runs. The gauge is what keeps a requested restart from
    firing under a still-streaming turn: ``trigger_restart_if_pending()``
    fires only at zero, so the last turn out is always the one that signals.
    """
    global _active_turns
    with _lock:
        _active_turns += 1


def exit_turn() -> None:
    """Unregister one turn. An unpaired exit (more exits than enters) is a
    wiring defect that could let a restart fire under a live turn, so it is
    logged at error and the count is clamped rather than driven negative."""
    global _active_turns
    with _lock:
        if _active_turns == 0:
            logger.error(
                "exit_turn called with no turn registered: a turn boundary "
                "calls enter/exit unevenly; restart gating may misfire"
            )
            return
        _active_turns -= 1


def trigger_restart_if_pending() -> None:
    """Fire a pending restart once no turn is in flight: SIGTERM to this
    process, exactly once.

    Called at turn end, after that turn's output is delivered or persisted.
    With another turn still in flight the call is a no-op — that turn's own
    end re-checks, so the restart waits for process quiescence. The signal
    is delivered to the process and handled on the event loop (hypercorn's
    native handler), so this returns immediately and the graceful shutdown
    proceeds while in-flight requests drain. Repeat calls after the first are
    no-ops.
    """
    global _triggered
    with _lock:
        if _requested_reason is None or _triggered or _active_turns > 0:
            return
        _triggered = True
        reason = _requested_reason
    logger.warning("Restarting MIRA: SIGTERM to self (requested: %s)", reason)
    os.kill(os.getpid(), signal.SIGTERM)
