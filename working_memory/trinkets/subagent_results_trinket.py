"""Unified subagent results trinket.

Renders every subagent result into one notification-center section (slot
`subagent_results`). Terminal results come from the durable `agent_results`
ledger (`agents/results_store.py`) — segment collapse no longer destroys
them, and they survive until the 14-day archive sweep (nothing is ever
deleted). In-flight states (pending / in_progress) are transient and live
in memory only; they never reach the ledger.

Per agent, the five most recent unarchived results render, followed by a
`<more>` line carrying the hidden and archived counts. `agent_tool recall`
(P6) unarchives rows back into the default view.
"""
import html
import logging
from typing import Dict, Any, TYPE_CHECKING

from working_memory.trinkets.base import StatefulTrinket, TRINKET_KEY_PREFIX
from utils.user_context import get_current_user_id

if TYPE_CHECKING:
    from cns.core.events import UpdateTrinketEvent
    from cns.integration.event_bus import EventBus
    from working_memory.core import WorkingMemory

logger = logging.getLogger(__name__)

# Slot names of the result trinkets this one replaced. The old trinkets are
# unregistered, so nothing else ever cleans these Valkey hash fields —
# without this sweep, API reads would serve their stale content forever.
_LEGACY_SLOT_FIELDS = ("forage_results", "whilethecatsaway_results", "memory_curation")

_PER_AGENT_LIMIT = 5


class SubagentResultsTrinket(StatefulTrinket):
    """Renders all subagent results into one prompt section."""

    variable_name = "subagent_results"

    def __init__(self, event_bus: 'EventBus', working_memory: 'WorkingMemory'):
        super().__init__(event_bus, working_memory)
        # Transient in-flight entries only (pending / in_progress). Terminal
        # results live in the ledger and are read at render time.
        self._user_results: dict[str, Dict[str, Dict[str, Any]]] = {}
        self._legacy_fields_cleaned: set[str] = set()

    @property
    def active_results(self) -> Dict[str, Dict[str, Any]]:
        """Get the active user's in-flight results. Lazy-initializes on access."""
        user_id = get_current_user_id()
        if user_id not in self._user_results:
            self._user_results[user_id] = {}
        return self._user_results[user_id]

    def _persist_to_valkey(self, content: str) -> None:
        super()._persist_to_valkey(content)
        user_id = get_current_user_id()
        if user_id in self._legacy_fields_cleaned:
            return
        from clients.valkey_client import get_valkey_client

        valkey = get_valkey_client()
        valkey.hdel_with_retry(
            f"{TRINKET_KEY_PREFIX}:{user_id}", *_LEGACY_SLOT_FIELDS
        )
        self._legacy_fields_cleaned.add(user_id)

    def _expire_items(self) -> bool:
        # Retention lives in the ledger's 14-day archive sweep; nothing
        # in-memory expires on a turn clock.
        return False

    def _clear_all_state(self) -> None:
        """Drop in-flight entries on segment collapse. The ledger persists."""
        results = self._user_results.pop(get_current_user_id(), None)
        if results:
            logger.info(f"Clearing {len(results)} in-flight subagent entries on segment collapse")

    def handle_update_request(self, event: 'UpdateTrinketEvent') -> None:
        """Process incoming subagent results by agent_type and status."""
        context = event.context
        task_id = context.get('task_id')
        status = context.get('status')

        if not task_id:
            # Lifecycle refresh from StatefulTrinket — just regenerate content
            super().handle_update_request(event)
            return

        if status == 'dismissed':
            self.active_results.pop(task_id, None)
            self._archive_ledger_rows(task_id)
            super().handle_update_request(event)
            return

        agent_type = context.get('agent_type')
        if agent_type == 'forage':
            self._handle_forage(task_id, status, context, event)
        elif agent_type == 'whilethecatsaway':
            self._handle_whilethecatsaway(task_id, status, context, event)
        elif agent_type == 'memory_curator':
            self._handle_memory_curator(task_id, status, context, event)
        elif agent_type and agent_type.startswith('agent:'):
            # Templated agents publish terminal events only; the ledger row
            # is written by the base class before the publish.
            if status in ('success', 'timeout', 'failed', 'rejected', 'skipped'):
                self.active_results.pop(task_id, None)
                super().handle_update_request(event)
                return
            super().handle_update_request(event)
        else:
            # Unknown agent_type: a terminal event still lands in the ledger;
            # drop any in-flight entry and re-render.
            if status in ('success', 'timeout', 'failed', 'rejected', 'skipped'):
                self.active_results.pop(task_id, None)
            super().handle_update_request(event)

    def _archive_ledger_rows(self, task_id: str) -> None:
        try:
            from agents.results_store import archive_by_task_id
            from utils.userdata_manager import get_user_data_manager

            db = get_user_data_manager(get_current_user_id())
            archive_by_task_id(db, task_id)
        except Exception:
            logger.exception("Failed to archive ledger rows for task %s", task_id)

    def _handle_forage(
        self, task_id: str, status: str, context: Dict[str, Any], event,
    ) -> None:
        if status == 'in_progress':
            # Stack summaries — accumulate per-iteration lines so Mira
            # sees the full research arc on next working memory refresh
            summary = context.get('summary', '')
            iteration = context.get('iteration', 0)
            existing = self.active_results.get(task_id)
            if existing and existing['type'] in ('success', 'failed', 'timeout'):
                return
            if existing and existing['type'] == 'in_progress':
                summaries = existing['data'].get('summaries', [])
            else:
                summaries = []
            summaries.append({'iteration': iteration, 'text': summary})
            self.active_results[task_id] = {
                'type': 'in_progress',
                'data': {
                    'agent_type': 'forage',
                    'subject': context.get('subject', ''),
                    'iteration': iteration,
                    'max_iterations': context.get('max_iterations', 0),
                    'summaries': summaries,
                },
                'received_turn': self.current_turn,
            }
            super().handle_update_request(event)
            return

        if status == 'pending':
            if task_id in self.active_results:
                # Pending publishes after the thread starts (the spawn seam);
                # an existing entry is the agent's own earlier update — a
                # late pending must not overwrite it.
                super().handle_update_request(event)
                return
            self.active_results[task_id] = {
                'type': 'pending',
                'data': context,
                'received_turn': self.current_turn,
            }
            super().handle_update_request(event)
            return

        # Terminal: the ledger row is the record; drop any in-flight entry.
        self.active_results.pop(task_id, None)
        super().handle_update_request(event)

    def _handle_whilethecatsaway(
        self, task_id: str, status: str, context: Dict[str, Any], event,
    ) -> None:
        if status == 'pending':
            if task_id in self.active_results:
                super().handle_update_request(event)
                return
            self.active_results[task_id] = {
                'type': 'pending',
                'data': context,
                'received_turn': self.current_turn,
            }
            super().handle_update_request(event)
            return
        # Terminal: ledger is the record.
        self.active_results.pop(task_id, None)
        super().handle_update_request(event)

    def _handle_memory_curator(
        self, task_id: str, status: str, context: Dict[str, Any], event,
    ) -> None:
        # The curator publishes once at completion; the ledger is the record.
        self.active_results.pop(task_id, None)
        super().handle_update_request(event)

    def _ledger_rows(self) -> tuple[list[dict], dict[str, dict[str, int]]]:
        """Read unarchived rows (capped per agent) and per-agent counts."""
        from agents.results_store import counts_by_agent, unarchived
        from utils.userdata_manager import get_user_data_manager

        db = get_user_data_manager(get_current_user_id())
        rows = unarchived(db, per_agent_limit=_PER_AGENT_LIMIT)
        counts = counts_by_agent(db)
        return rows, counts

    def _mark_seen(self, rows: list[dict]) -> None:
        try:
            from agents.results_store import mark_seen
            from utils.userdata_manager import get_user_data_manager

            unseen = [r["run_id"] for r in rows if not r.get("seen_at")]
            if unseen:
                mark_seen(get_user_data_manager(get_current_user_id()), unseen)
        except Exception:
            logger.exception("Failed to mark agent results seen")

    def generate_content(self, context: Dict[str, Any]) -> str:
        """Generate XML content showing in-flight and recent ledger results."""
        parts = []

        for task_id, result in self.active_results.items():
            parts.append(self._format_in_flight(task_id, result))

        try:
            rows, counts = self._ledger_rows()
        except Exception:
            logger.exception("SubagentResultsTrinket: ledger read failed; showing in-flight only")
            rows, counts = [], {}

        by_agent: dict[str, list[dict]] = {}
        for row in rows:
            by_agent.setdefault(row["agent"], []).append(row)

        # Most recently active agent first
        def _latest(items: list[dict]) -> str:
            return max((r.get("created_at") or "") for r in items)

        for agent in sorted(by_agent, key=lambda a: _latest(by_agent[a]), reverse=True):
            agent_rows = by_agent[agent]
            for row in agent_rows:
                parts.append(self._format_ledger_row(row))
            total = counts.get(agent, {}).get("unarchived", len(agent_rows))
            archived = counts.get(agent, {}).get("archived", 0)
            hidden = total - len(agent_rows)
            if hidden > 0 or archived > 0:
                text = f"+{hidden} more" if hidden else ""
                if archived:
                    text += f" ({archived} archived)" if text else f"{archived} archived"
                parts.append(
                    f'<more agent="{html.escape(agent, quote=True)}" '
                    f'hidden="{hidden}" archived="{archived}">'
                    f"{text}"
                    f"</more>"
                )

        self._mark_seen(rows)

        alerts = self._agent_memory_alerts()
        if alerts:
            parts.append(
                "<agent_memory_alerts>\n"
                + "\n".join(alerts)
                + "\n</agent_memory_alerts>"
            )

        if parts:
            return "<subagent_results>\n" + "\n".join(parts) + "\n</subagent_results>"
        return ""

    def _agent_memory_alerts(self) -> list[str]:
        """Surface a 'too many memories' alert when an agent exceeds the
        configured active-memory threshold."""
        try:
            from config import config
            from utils.user_context import get_current_user_id
            from utils.userdata_manager import get_user_data_manager

            threshold = config.sidebar_dispatcher.agent_memory_max_count
            db = get_user_data_manager(get_current_user_id())
            rows = db.execute(
                "SELECT agent_id, COUNT(*) AS c FROM agent_memory "
                "WHERE status = 'active' GROUP BY agent_id"
            )
            alerts: list[str] = []
            for row in rows:
                if int(row.get("c", 0)) > threshold:
                    alerts.append(
                        f"agent {row.get('agent_id')} has "
                        f"{row.get('c')} active memories (threshold "
                        f"{threshold}); consider compacting or deprecating."
                    )
            return alerts
        except Exception:
            return []

    def _format_in_flight(self, task_id: str, result: Dict[str, Any]) -> str:
        result_type = result['type']
        data = result['data']
        agent_type = data.get('agent_type', 'unknown')

        if agent_type == 'forage' and result_type == 'in_progress':
            return self._format_forage_in_progress(task_id, data)
        if result_type == 'pending':
            return self._format_pending(task_id, data, agent_type)
        return ""

    def _format_ledger_row(self, row: Dict[str, Any]) -> str:
        agent = row.get("agent", "unknown")
        status = row.get("status") or "unknown"
        subject = row.get("subject") or ""
        summary = row.get("encrypted__summary") or ""
        run_id = row.get("run_id", "")
        subject_attr = f' subject="{html.escape(subject, quote=True)}"' if subject else ""
        return (
            f'<result agent_type="{html.escape(agent, quote=True)}" '
            f'type="{html.escape(status, quote=True)}" run_id="{html.escape(run_id, quote=True)}"'
            f'{subject_attr}>\n'
            f"{html.escape(summary)}\n"
            f"</result>"
        )

    def _subject_attr(self, data: Dict[str, Any]) -> str:
        subject = data.get('subject', '')
        return f' subject="{html.escape(subject, quote=True)}"'

    def _format_forage_in_progress(self, task_id: str, data: Dict[str, Any]) -> str:
        iteration = data.get('iteration', 0)
        max_iter = data.get('max_iterations', 0)
        summaries = data.get('summaries', [])
        lines = [f"[{s['iteration']}] {html.escape(s['text'])}" for s in summaries]
        return (
            f'<result agent_type="forage" type="in_progress" task_id="{task_id}"'
            f'{self._subject_attr(data)} iteration="{iteration}/{max_iter}">\n'
            + "\n".join(lines) + "\n"
            "</result>"
        )

    def _format_pending(
        self, task_id: str, data: Dict[str, Any], agent_type: str
    ) -> str:
        verb = "Foraging" if agent_type == 'forage' else "Working"
        return (
            f'<result agent_type="{agent_type}" type="pending" task_id="{task_id}"'
            f'{self._subject_attr(data)}>\n'
            f"{verb} in progress...\n"
            f"</result>"
        )
