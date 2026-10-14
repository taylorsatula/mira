"""Unified subagent results trinket.

Renders every model-visible subagent result — forage research,
while-the-cats-away curiosity research, and memory-curation runs — into one
notification-center section (slot `subagent_results`), keyed by `task_id`,
carrying `agent_type` for per-type policy. Per-type lifecycles are read from
`agents/specs.py:SUBAGENT_SPECS`:

  forage           pending -> in_progress (stacked per-iteration summaries,
                   terminal-state guard against late overwatch downgrades)
                   -> success (persists until dismissed or collapse) |
                   timeout/failed (error TTL). Explicit `dismissed` removal.
  whilethecatsaway every status (pending | success | timeout | failed)
                   auto-expires (result TTL for pending/success, error TTL
                   for timeout/failed); `dismissed` removal.
  memory_curator   success persists (the next run's stable task_id replaces
                   it); timeout/failed (error TTL); unknown statuses drop
                   without re-render. No pending/in_progress path — the
                   curator publishes once at completion.

phoneafriend never appears here: its consults are blocking runs, and
`agents/base.py` suppresses every trinket publication for blocking runs —
the response travels entirely through the tool result.
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



class SubagentResultsTrinket(StatefulTrinket):
    """Renders all subagent results into one prompt section."""

    variable_name = "subagent_results"

    def __init__(self, event_bus: 'EventBus', working_memory: 'WorkingMemory'):
        super().__init__(event_bus, working_memory)
        self._user_results: dict[str, Dict[str, Dict[str, Any]]] = {}
        self._legacy_fields_cleaned: set[str] = set()

    @property
    def active_results(self) -> Dict[str, Dict[str, Any]]:
        """Get the active user's results. Lazy-initializes on access to support in-place mutation."""
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
        """Remove results past their per-type display window."""
        expired = [
            task_id for task_id, result in self.active_results.items()
            if 'display_until_turn' in result
            and self.current_turn > result['display_until_turn']
        ]
        for task_id in expired:
            del self.active_results[task_id]
        return bool(expired)

    def _clear_all_state(self) -> None:
        """Clear results for the current user on segment collapse."""
        results = self._user_results.pop(get_current_user_id(), None)
        if results:
            logger.info(f"Clearing {len(results)} subagent results on segment collapse")

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
            if task_id in self.active_results:
                del self.active_results[task_id]
                logger.debug(f"Dismissed subagent result {task_id[:8]}")
            super().handle_update_request(event)
            return

        agent_type = context.get('agent_type')
        if agent_type == 'forage':
            self._handle_forage(task_id, status, context, event)
        elif agent_type == 'whilethecatsaway':
            self._handle_whilethecatsaway(task_id, status, context, event)
        elif agent_type == 'memory_curator':
            self._handle_memory_curator(task_id, status, context, event)
        else:
            logger.warning(
                "SubagentResultsTrinket: no lifecycle for agent_type %r "
                "(task %s) — update dropped",
                agent_type, task_id[:8],
            )

    def _handle_forage(
        self, task_id: str, status: str, context: Dict[str, Any], event,
    ) -> None:
        if status == 'in_progress':
            # Don't downgrade terminal states (late overwatch thread)
            existing = self.active_results.get(task_id)
            if existing and existing['type'] in ('success', 'failed', 'timeout'):
                return
            # Stack summaries — accumulate per-iteration lines so Mira
            # sees the full research arc on next working memory refresh
            summary = context.get('summary', '')
            iteration = context.get('iteration', 0)
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
        elif status == 'success':
            self.active_results[task_id] = {
                'type': 'success',
                'data': context,
                'received_turn': self.current_turn,
            }
        elif status in ('timeout', 'failed'):
            from agents.specs import SUBAGENT_SPECS

            self.active_results[task_id] = {
                'type': status,
                'data': context,
                'received_turn': self.current_turn,
                'display_until_turn': (
                    self.current_turn + SUBAGENT_SPECS['forage'].error_ttl_turns
                ),
            }
        else:
            # Unknown status: no state change, but still re-render
            super().handle_update_request(event)
            return
        super().handle_update_request(event)

    def _handle_whilethecatsaway(
        self, task_id: str, status: str, context: Dict[str, Any], event,
    ) -> None:
        from agents.specs import SUBAGENT_SPECS

        if status not in ('pending', 'success', 'timeout', 'failed'):
            super().handle_update_request(event)
            return
        if status == 'pending' and task_id in self.active_results:
            # Same race guard as forage's pending: the agent's own update
            # arrived first; never overwrite it with the stale pending.
            super().handle_update_request(event)
            return
        # All statuses auto-expire — informational, not a working document
        spec = SUBAGENT_SPECS['whilethecatsaway']
        ttl_turns = (
            spec.result_ttl_turns
            if status in ('pending', 'success')
            else spec.error_ttl_turns
        )
        self.active_results[task_id] = {
            'type': status,
            'data': context,
            'received_turn': self.current_turn,
            'display_until_turn': self.current_turn + ttl_turns,
        }
        super().handle_update_request(event)

    def _handle_memory_curator(
        self, task_id: str, status: str, context: Dict[str, Any], event,
    ) -> None:
        from agents.specs import SUBAGENT_SPECS

        if status == 'success':
            self.active_results[task_id] = {
                'type': 'success',
                'data': context,
                'received_turn': self.current_turn,
            }
        elif status in ('timeout', 'failed'):
            self.active_results[task_id] = {
                'type': status,
                'data': context,
                'received_turn': self.current_turn,
                'display_until_turn': (
                    self.current_turn + SUBAGENT_SPECS['memory_curator'].error_ttl_turns
                ),
            }
        else:
            # Unknown statuses drop without re-render — the curator
            # publishes once at completion; anything else is a producer bug
            return
        super().handle_update_request(event)

    def generate_content(self, context: Dict[str, Any]) -> str:
        """Generate XML content showing all active subagent results."""
        parts = []

        for task_id, result in self.active_results.items():
            if 'display_until_turn' in result and (
                self.current_turn > result['display_until_turn']
            ):
                continue
            parts.append(self._format_entry(task_id, result))

        if parts:
            return "<subagent_results>\n" + "\n".join(parts) + "\n</subagent_results>"
        return ""

    def _format_entry(self, task_id: str, result: Dict[str, Any]) -> str:
        result_type = result['type']
        data = result['data']
        agent_type = data.get('agent_type', 'unknown')

        if agent_type == 'forage':
            if result_type == 'in_progress':
                return self._format_forage_in_progress(task_id, data)
            if result_type == 'pending':
                return self._format_pending(task_id, data, 'forage')
            if result_type == 'success':
                return self._format_forage_success(task_id, data)
            return self._format_error(task_id, result, agent_type)

        if agent_type == 'whilethecatsaway':
            if result_type == 'pending':
                return self._format_pending(task_id, data, 'whilethecatsaway')
            if result_type == 'success':
                return self._format_success(task_id, data, 'whilethecatsaway')
            return self._format_error(task_id, result, agent_type)

        if agent_type == 'memory_curator':
            if result_type == 'success':
                return self._format_curator_success(task_id, data)
            return self._format_error(task_id, result, agent_type)

        # An entry with no lifecycle should never be stored; render nothing
        # rather than fabricate a shape for it.
        return ""

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
        verb = "Foraging" if agent_type == 'forage' else "Researching"
        return (
            f'<result agent_type="{agent_type}" type="pending" task_id="{task_id}"'
            f'{self._subject_attr(data)}>\n'
            f"{verb} in progress...\n"
            f"</result>"
        )

    def _format_forage_success(self, task_id: str, data: Dict[str, Any]) -> str:
        written_result = data.get('result', '')
        iterations = data.get('iterations', 0)
        return (
            f'<result agent_type="forage" type="success" task_id="{task_id}"'
            f'{self._subject_attr(data)} iterations="{iterations}">\n'
            f"{html.escape(written_result)}\n"
            f"</result>"
        )

    def _format_success(
        self, task_id: str, data: Dict[str, Any], agent_type: str
    ) -> str:
        summary = data.get('result', '')
        return (
            f'<result agent_type="{agent_type}" type="success" task_id="{task_id}"'
            f'{self._subject_attr(data)}>\n'
            f"{html.escape(summary)}\n"
            f"</result>"
        )

    def _format_curator_success(self, task_id: str, data: Dict[str, Any]) -> str:
        mode = data.get('mode', 'unknown')
        summary = data.get('summary', '')
        segment_id = data.get('segment_id', '')

        header = (
            f'<result agent_type="memory_curator" type="success" '
            f'task_id="{task_id}" mode="{html.escape(mode, quote=True)}"'
        )
        if segment_id:
            header += f' segment="{html.escape(segment_id, quote=True)}"'
        header += '>'

        return f"{header}\n{html.escape(summary)}\n</result>"

    def _format_error(
        self, task_id: str, result: Dict[str, Any], agent_type: str
    ) -> str:
        data = result['data']
        result_type = result['type']
        turns_remaining = result['display_until_turn'] - self.current_turn

        if result_type == 'timeout':
            if agent_type == 'forage':
                # Producer sends `iterations` (count) + `error`; never elapsed/iteration
                iterations = data.get('iterations', 0)
                error = data.get('error', 'Iteration limit exceeded')
                return (
                    f'<result agent_type="forage" type="timeout" task_id="{task_id}"'
                    f'{self._subject_attr(data)} turns_remaining="{turns_remaining}">\n'
                    f"Forage timed out after {iterations} iteration(s): {html.escape(error)}\n"
                    f"</result>"
                )
            if agent_type == 'memory_curator':
                return (
                    f'<result agent_type="memory_curator" type="timeout" '
                    f'task_id="{task_id}" mode="{data.get("mode", "unknown")}" '
                    f'turns_remaining="{turns_remaining}">\n'
                    f"Memory curator timed out.\n"
                    f"</result>"
                )
            return (
                f'<result agent_type="whilethecatsaway" type="timeout" task_id="{task_id}"'
                f'{self._subject_attr(data)} turns_remaining="{turns_remaining}">\n'
                f"Background research timed out.\n"
                f"</result>"
            )

        if agent_type == 'forage':
            error = data.get('error', 'Unknown error')
            error_type = data.get('error_type', 'Error')
            return (
                f'<result agent_type="forage" type="failed" task_id="{task_id}"'
                f'{self._subject_attr(data)} turns_remaining="{turns_remaining}">\n'
                f"Forage failed: {html.escape(error_type)} — {html.escape(error)}\n"
                f"</result>"
            )
        if agent_type == 'memory_curator':
            return (
                f'<result agent_type="memory_curator" type="failed" '
                f'task_id="{task_id}" mode="{data.get("mode", "unknown")}" '
                f'turns_remaining="{turns_remaining}">\n'
                f"Memory curator failed.\n"
                f"</result>"
            )
        error = data.get('error', 'Unknown error')
        return (
            f'<result agent_type="whilethecatsaway" type="failed" task_id="{task_id}"'
            f'{self._subject_attr(data)} turns_remaining="{turns_remaining}">\n'
            f"Background research failed: {html.escape(error)}\n"
            f"</result>"
        )
