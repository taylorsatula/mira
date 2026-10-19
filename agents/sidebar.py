"""
Sidebar dispatcher -- the trigger protocol, the work item, and the poll loop.

`WorkItem` is the contract between the two halves of this directory: a trigger
produces one per piece of discovered work, keyed by an `item_id` that stays
stable across polls, and the agent run consumes its `context`. `SidebarTrigger`
is structural, with no base class to inherit: `check_for_new_items` returns
candidate work and nothing else, because the dispatcher alone decides what
actually runs.

`SidebarDispatcher.poll` is one APScheduler tick. It resolves eligible users
(active recently, plus anyone holding an armed heartbeat brief), sets the user
contextvar per user, and walks the registered triggers. `_dispatch_decision` is
the whole dedup rule: no prior row dispatches, a terminal status skips forever,
and a `failed` row retries only while `run_count` is still within the agent
class's `max_retries`. `_get_prior_run` reads back just the two fields that rule
needs, so the `prior_run` handed to the agent is not a copy of the row.

Two bounds live here. `MAX_ITEMS_PER_USER_PER_POLL` caps dispatches per trigger
per user per tick, and `_maybe_cleanup` deletes old activity rows — which is what
makes a long-finished item discoverable again later.

A trigger that raises aborts the rest of that user's triggers for the tick: the
per-user handler logs it and moves to the next user, so one broken trigger hides
every trigger ordered after it for the users it breaks on.
"""
import logging
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

from pydantic import BaseModel

from utils.timezone_utils import utc_now

if TYPE_CHECKING:
    from agents.base import SidebarAgent
    from tools.repo import ToolRepository
    from cns.integration.event_bus import EventBus

logger = logging.getLogger(__name__)


class WorkItem(BaseModel):
    """A unit of work detected by a trigger."""
    item_id: str         # Unique ID for dedup (e.g. IMAP folder:uid)
    interface_name: str  # Key for AsyncActivityTrinket (e.g. "email_watcher")
    context: dict[str, Any]  # Trigger-specific payload passed to the agent


@runtime_checkable
class SidebarTrigger(Protocol):
    trigger_id: str
    interface_name: str
    agent_class: type['SidebarAgent']

    def check_for_new_items(self, user_id: str) -> list[WorkItem]:
        """Poll for new work. Return ALL discovered items -- dispatcher handles dedup."""
        ...

    def on_dispatched(self, user_id: str, item_id: str) -> None:
        """Post-dispatch hook for trigger-specific side effects (e.g. IMAP flag).

        NOT for dedup -- the dispatcher owns that via sidebar_activity.
        Must be idempotent (safe to call on retries).
        """
        ...


# Per-user-per-poll cap to prevent runaway cost from misconfigured triggers
MAX_ITEMS_PER_USER_PER_POLL = 5

_TERMINAL_STATUSES = frozenset({'handled', 'escalated', 'resolved', 'dismissed'})

# Records older than this are deleted during cleanup. Not tied to any
# trigger's discovery window -- purely a retention hygiene bound.
_CLEANUP_RETENTION_DAYS = 30

_CLEANUP_INTERVAL_SECONDS = 86400


class SidebarDispatcher:
    """Iterates users and triggers, spawns sidebar agent threads.

    Owns dedup via sidebar_activity SQLite table + in-flight tracking.
    Triggers are pure discovery -- they return all candidate items and
    the dispatcher decides what to act on.
    """

    def __init__(
        self,
        tool_repo: 'ToolRepository',
        event_bus: 'EventBus',
    ):
        self.tool_repo = tool_repo
        self.event_bus = event_bus
        self._triggers: list[SidebarTrigger] = []
        self._last_cleanup: dict[str, float] = {}

    def register_trigger(self, trigger: SidebarTrigger) -> None:
        self._triggers.append(trigger)
        logger.info(f"Registered sidebar trigger: {trigger.trigger_id}")

    def poll(self) -> None:
        """Single poll cycle. Called by APScheduler on each tick."""
        from utils.user_context import set_current_user_id, clear_user_context

        users = self._get_eligible_users()
        if not users:
            return

        for user in users:
            user_id = str(user['id'])
            set_current_user_id(user_id)
            try:
                self._poll_user(user_id)
            except Exception as e:
                logger.error(
                    f"Sidebar dispatcher error for user {user_id}: {e}",
                    exc_info=True,
                )
            finally:
                clear_user_context()

    def _poll_user(self, user_id: str) -> None:
        """Check all triggers for a single user, with SQLite-backed dedup."""
        from utils.userdata_manager import get_user_data_manager
        from agents.base import ensure_activity_schema
        from agents.spawn import is_in_flight

        db = get_user_data_manager(user_id)
        ensure_activity_schema(db)
        self._maybe_cleanup(db, user_id)

        for trigger in self._triggers:
            items = trigger.check_for_new_items(user_id)
            if not items:
                continue

            dispatched = 0
            for item in items:
                if dispatched >= MAX_ITEMS_PER_USER_PER_POLL:
                    break

                if is_in_flight(item.interface_name, item.item_id):
                    continue

                prior_run = self._get_prior_run(db, item)
                decision = self._dispatch_decision(
                    prior_run, trigger.agent_class,
                )
                if decision == 'skip':
                    continue

                if decision == 'retry' and prior_run:
                    item.context['run_count'] = prior_run.get('run_count', 1) + 1
                    item.context['prior_run'] = dict(prior_run)
                else:
                    item.context['run_count'] = 1

                if not self._spawn_agent(trigger, item):
                    continue
                trigger.on_dispatched(user_id, item.item_id)
                dispatched += 1

    def _get_prior_run(self, db, item: WorkItem) -> dict | None:
        """Look up existing activity record for this item."""
        rows = db.execute(
            "SELECT status, run_count FROM sidebar_activity "
            "WHERE interface_name = :iface AND thread_id = :tid",
            {'iface': item.interface_name, 'tid': item.item_id},
        )
        return dict(rows[0]) if rows else None

    def _dispatch_decision(
        self,
        prior_run: dict | None,
        agent_class: type,
    ) -> str:
        """Decide: 'dispatch' (first run), 'retry', or 'skip'."""
        if prior_run is None:
            return 'dispatch'

        status = prior_run.get('status', '')
        if status in _TERMINAL_STATUSES:
            return 'skip'

        if status == 'failed':
            run_count = prior_run.get('run_count', 1)
            max_retries = agent_class.max_retries
            if run_count <= max_retries:
                return 'retry'

        return 'skip'

    def _spawn_agent(
        self,
        trigger: SidebarTrigger,
        item: WorkItem,
    ) -> bool:
        """Dispatch an item through the shared spawn seam.

        Returns True if the agent was spawned, False if the process is at
        the cap or the item is already in flight — either way the item is
        left unmarked and rediscovered next poll.
        """
        from agents.spawn import spawn_subagent
        from agents.specs import spec_for_agent_class

        spec = spec_for_agent_class(trigger.agent_class)
        return spawn_subagent(
            spec, item, self.event_bus, tool_repo=self.tool_repo
        )

    def _maybe_cleanup(self, db, user_id: str) -> None:
        """Delete old activity + scratchpad records, rate-limited per user."""
        from datetime import timedelta
        now = utc_now()
        last = self._last_cleanup.get(user_id)
        if last and (now.timestamp() - last) < _CLEANUP_INTERVAL_SECONDS:
            return

        self._last_cleanup[user_id] = now.timestamp()
        cutoff = (now - timedelta(days=_CLEANUP_RETENTION_DAYS)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        try:
            db.execute(
                "DELETE FROM scratchpad WHERE thread_id IN ("
                "  SELECT thread_id FROM sidebar_activity"
                "  WHERE updated_at < :cutoff"
                ")",
                {'cutoff': cutoff},
            )
            db.execute(
                "DELETE FROM sidebar_activity WHERE updated_at < :cutoff",
                {'cutoff': cutoff},
            )
        except Exception as e:
            logger.warning(f"Sidebar cleanup failed for user {user_id}: {e}")

    def _get_eligible_users(self) -> list[dict]:
        """Users the dispatcher polls: anyone active in the last two days,
        plus anyone with an armed heartbeat brief."""
        from utils.database_session_manager import get_shared_session_manager

        session_manager = get_shared_session_manager()
        with session_manager.get_admin_session() as session:
            return session.execute_query("""
                SELECT id FROM users
                WHERE is_active = TRUE
                  AND (
                      last_activity_date >= CURRENT_DATE - INTERVAL '2 days'
                      OR id IN (SELECT user_id FROM heartbeat_armed_users)
                  )
            """)
