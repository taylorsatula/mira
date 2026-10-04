"""
Background-activity digest: pluggable sources for the heartbeat wake briefing.

The heartbeat's stimulus asks the model to decide keepsleeping/breakout against
a digest of recent background activity. That digest is assembled here from
registered contributors, so a background surface an install runs can brief the
model at wake time without the heartbeat knowing its specifics.

A contributor owns one source and its presentation:

    class BackgroundDigestContributor(Protocol):
        def name(self) -> str: ...
        def contribute(self, user_id: str) -> list[str]: ...

Contract:
- ``contribute`` returns the section's lines with the header line first.
  Return ``[]`` to contribute nothing on this wake.
- A contributor either returns lines or raises. Required infrastructure
  (per-user SQLite, the service DB) may propagate: the wake fails and retries
  on the next tick. An optional external source should catch its own failure
  and render it as a line ("surface unavailable: ...") so the wake still
  happens and the model sees the outage as state.
- Contributors are process-global singletons shared across users: keep all
  per-user state derived from the ``user_id`` argument, never on ``self``.

Registration: the built-in contributors self-register at module import. An
install adding its own source registers it in factory wiring
(``register_background_digest_contributor(MyContributor())``) or in the module
that imports it first.

Worked example — a contributor over an optional external surface. The shape
to copy (this sketch is complete in shape; a real implementation is a fetch,
an in-band failure line, and a bounded row render):

    class JobsContributor:
        def name(self) -> str:
            return "mlfactory_jobs"

        def contribute(self, user_id: str) -> list[str]:
            try:
                records = fetch_live_jobs(user_id)          # transport + parse
            except FacadeTransportError as exc:
                return [f"mlfactory jobs surface unavailable: {exc}"]
            if not records:
                return ["No mlfactory jobs on record."]
            return ["Mlfactory jobs (live from the supervisor):"] + [
                f"- job {r['job_id']} run {r['run_id']} "
                f"[{r['status']}, alive={r['process_alive']}] spec={r['spec']}"
                for r in records[:_LIMIT]
            ]
"""
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class BackgroundDigestContributor(Protocol):
    """One bounded source of recent background activity for the wake digest."""

    def name(self) -> str:
        """Stable contributor identifier, for logs and registry diagnostics."""
        ...

    def contribute(self, user_id: str) -> list[str]:
        """Return this section's lines (header first), or [] to stay silent."""
        ...


_contributors: list[BackgroundDigestContributor] = []


def register_background_digest_contributor(
    contributor: BackgroundDigestContributor,
) -> None:
    """Register a digest contributor. Order of registration = order in the digest."""
    _contributors.append(contributor)


def build_background_digest(user_id: str) -> str:
    """Assemble the wake briefing from every registered contributor.

    Contributors speak for their own failure handling (see the module
    docstring): infrastructure failures propagate, optional sources render
    their outage in band.
    """
    sections: list[list[str]] = []
    for contributor in _contributors:
        lines = contributor.contribute(user_id)
        if lines:
            sections.append(lines)
    return "\n".join(line for section in sections for line in section)


class SidebarActivityContributor:
    """Recent terminal sidebar-agent activity from the user's SQLite store.

    ``interface_name`` is the provenance marker: it distinguishes production
    rows from rows created by tests or probes, so the model can weigh them
    accordingly.
    """

    # 24h covers a sleep of any realistic length; 15 rows keeps the digest
    # bounded no matter how chatty the agents were.
    _WINDOW_HOURS = 24
    _LIMIT = 15
    _SNIPPET_CHARS = 200

    def name(self) -> str:
        return "sidebar_activity"

    def contribute(self, user_id: str) -> list[str]:
        from utils.userdata_manager import get_user_data_manager

        db = get_user_data_manager(user_id)
        rows = db.fetchall(
            "SELECT interface_name, agent_id, status, summary, updated_at "
            "FROM sidebar_activity "
            "WHERE updated_at >= datetime('now', ?) "
            "ORDER BY updated_at DESC LIMIT ?",
            (f"-{self._WINDOW_HOURS} hours", self._LIMIT),
        )
        if not rows:
            return [f"No sidebar agent activity in the last {self._WINDOW_HOURS}h."]
        lines = [
            f"Sidebar agent activity, last {self._WINDOW_HOURS}h "
            "(interface_name is the provenance: rows from tests or probes "
            "carry non-production interface names):"
        ]
        for row in rows:
            summary = (row.get("summary") or "")[: self._SNIPPET_CHARS]
            lines.append(
                f"- [{row.get('updated_at')}] {row.get('interface_name')}/"
                f"{row.get('agent_id')} ({row.get('status')}): {summary}"
            )
        return lines


class HeartbeatDecisionsContributor:
    """The heartbeat's own decision record, for continuity across wakes."""

    _LIMIT = 5
    _SNIPPET_CHARS = 200

    def name(self) -> str:
        return "heartbeat_decisions"

    def contribute(self, user_id: str) -> list[str]:
        from tools.implementations.heartbeat_tool import ensure_heartbeat_log
        from utils.userdata_manager import get_user_data_manager

        db = get_user_data_manager(user_id)
        ensure_heartbeat_log(db)
        decisions = db.fetchall(
            "SELECT decision, reason, wake_in_seconds, created_at FROM heartbeat_log "
            "ORDER BY created_at DESC LIMIT ?",
            (self._LIMIT,),
        )
        if not decisions:
            return []
        lines = ["Recent heartbeat decisions:"]
        for row in decisions:
            wake_in = row.get("wake_in_seconds")
            sleep_note = f" (slept {wake_in}s)" if wake_in else ""
            lines.append(
                f"- [{row.get('created_at')}] {row.get('decision')}{sleep_note}: "
                f"{(row.get('reason') or '')[: self._SNIPPET_CHARS]}"
            )
        return lines


class SelfEditOutcomeContributor:
    """The outcome of a self-edit restart, offered to exactly one wake.

    MIRA restarted itself to apply a code change (utils/self_edit.py); the user
    is waiting to hear whether it worked. The heartbeat dispatcher treats a
    pending outcome as due now (heartbeat_service.heartbeat_tick), and this
    contributor puts it in that wake's digest once — marking it offered — so a
    breakout announces it without every later wake re-announcing it. The HUD
    section (working_memory/trinkets/self_edit_trinket.py) keeps it visible
    until the user's next turn either way.
    """

    def name(self) -> str:
        return "self_edit_outcome"

    def contribute(self, user_id: str) -> list[str]:
        from utils import self_edit

        if not self_edit.heartbeat_delivery_pending(user_id):
            return []
        result = self_edit.result_for_user(user_id)
        if result is None:
            return []
        self_edit.mark_offered_to_heartbeat()
        outcome = (
            "it APPLIED: MIRA started on the changed code"
            if result["status"] == "applied"
            else "it FAILED: MIRA could not start on it, so the change was stashed "
            "and MIRA is running the previous code"
        )
        return [
            "Self-edit restart outcome (details in your HUD self_edit_status section):",
            f"- MIRA restarted to apply a change to its own code, and {outcome}. "
            "The user asked for this change and is waiting to hear the result: "
            "break out and tell them.",
        ]


register_background_digest_contributor(SelfEditOutcomeContributor())
register_background_digest_contributor(SidebarActivityContributor())
register_background_digest_contributor(HeartbeatDecisionsContributor())
