"""
HeartbeatTrigger -- dispatcher-driven wake-up source for due briefs.

Surfaces one work item per user per poll when that user has armed briefs due
(now >= next_wake_at). Pure deterministic discovery: one SQLite query, no
LLM. All evaluation and mission work belongs to the HeartbeatAgent.

One work item batches ALL of the user's due briefs: briefs are rare, and
batching avoids dispatching N agent runs for a morning cluster. The work
item's context carries each due brief's full payload (mission,
condition_check, bail_if, prior last_result for continuity) so the agent
acts on what is handed to it, never reconstructs intent from archaeology.

Dedup: item_id is stable per due-window (brf_{user_id}_{minute_key}),
derived from the earliest due brief's next_wake_at truncated to the poll
minute. A re-arm stamps a new next_wake_at, so the next due-window is a
genuinely new work item; the same window never double-dispatches.

Eligibility: users with armed briefs remain pollable by the dispatcher
even when dormant (the heartbeat_armed_users UNION in
agents/sidebar.py:_get_eligible_users).
"""
import logging
from typing import Any

from agents.sidebar import WorkItem
from agents.triggers.rule_filter import apply_trigger_rules

logger = logging.getLogger(__name__)


class HeartbeatTrigger:
    """Surfaces due armed briefs as one batched work item per user."""

    trigger_id = "heartbeat"
    interface_name = "heartbeat"

    # No scalar context keys to match rules against (the briefs list is the
    # payload); declared empty per the trigger-rules contract.
    FILTERABLE_FIELDS = ()

    # Lazily imported: a module-level import here would pull the agent (and
    # heartbeat_tool's Postgres session machinery) into
    # utils/sidebar_jobs.py's registration import path.
    @property
    def agent_class(self) -> type:
        from agents.implementations.heartbeat_agent import HeartbeatAgent

        return HeartbeatAgent

    def check_for_new_items(self, user_id: str) -> list[WorkItem]:
        from config.config_manager import config

        # Poll-time half of the opt-in double gate
        if not config.heartbeat.enabled:
            return []

        from utils.timezone_utils import format_utc_iso, parse_utc_time_string, utc_now
        from utils.userdata_manager import get_user_data_manager
        from tools.implementations.heartbeat_tool import ensure_briefs_table

        db = get_user_data_manager(user_id)
        ensure_briefs_table(db)

        now = utc_now()
        rows = db.fetchall(
            "SELECT * FROM briefs WHERE state = 'armed' ORDER BY next_wake_at ASC"
        )
        if not rows:
            return []

        due = []
        for row in rows:
            try:
                wake_at = parse_utc_time_string(row["next_wake_at"])
            except (ValueError, TypeError):
                logger.warning(
                    "Unparseable next_wake_at %r on brief %s; skipping",
                    row.get("next_wake_at"), row.get("id"),
                )
                continue
            if wake_at <= now:
                due.append(row)

        if not due:
            return []

        # Stable item_id per due-window: the earliest due brief's wake time
        # truncated to the poll minute. Every brief in this batch shares it.
        minute_key = due[0]["next_wake_at"][:16]  # 'YYYY-MM-DDTHH:MM'
        item_id = f"brf_{user_id}_{minute_key}"

        # Dispatch stamp: marks these briefs as handed to this run. The
        # HeartbeatAgent compares against it on failure to find briefs that
        # never received a report; every report overwrites it.
        dispatch_stamp = format_utc_iso(now)
        for r in due:
            db.execute(
                "UPDATE briefs SET last_wake_at = :stamp WHERE id = :id",
                {"stamp": dispatch_stamp, "id": r["id"]},
            )

        context: dict[str, Any] = {
            "briefs": [
                {
                    "id": r["id"],
                    "type": r["type"],
                    "mission": r.get("encrypted__mission"),
                    "condition_check": r.get("encrypted__condition_check"),
                    "bail_if": r.get("encrypted__bail_if"),
                    "schedule": r.get("schedule"),
                    "interval_seconds": r.get("interval_seconds"),
                    "last_result": r.get("encrypted__last_result"),
                    "dispatch_stamp": dispatch_stamp,
                }
                for r in due
            ],
        }

        items = [
            WorkItem(
                item_id=item_id,
                interface_name=self.interface_name,
                context=context,
            )
        ]
        return apply_trigger_rules(user_id, self.trigger_id, items)


def publish_device_wake_aggregate() -> None:
    """Publish the earliest armed-brief wake time to the device binding.

    Reads min(earliest_wake_at) from heartbeat_armed_users and hands it to
    notify_heartbeat_sleep; an empty table publishes nothing (device stays
    at full power). notify never raises, so this function owns only the
    aggregation query.
    """
    from utils.database_session_manager import get_shared_session_manager
    from utils.device_binding import notify_heartbeat_sleep

    session = get_shared_session_manager()
    with session.get_admin_session() as admin:
        rows = admin.execute_query(
            "SELECT user_id, earliest_wake_at FROM heartbeat_armed_users "
            "ORDER BY earliest_wake_at ASC LIMIT 1"
        )
    if not rows:
        return
    earliest = rows[0]
    notify_heartbeat_sleep(
        earliest["earliest_wake_at"],
        {"kind": "heartbeat_sleep", "user_wake_times": {
            str(earliest["user_id"]): earliest["earliest_wake_at"]
        }},
    )


def register_device_wake_job(scheduler_service) -> None:
    """Register the device-binding aggregation job (once, at boot).

    Own job rather than dispatcher-poll piggyback: the aggregate is a
    cross-user Postgres read, while the dispatcher is per-user and
    SQLite-oriented; coupling them would give the dispatcher a heartbeat
    import it does not otherwise need. Cadence matches the sidebar poll
    (1m) so wake obligations propagate promptly without a second cadence
    knob.
    """
    from apscheduler.triggers.interval import IntervalTrigger
    from config.config_manager import config

    if not config.heartbeat.enabled:
        return
    scheduler_service.register_job(
        job_id="heartbeat_device_wake_aggregate",
        func=publish_device_wake_aggregate,
        trigger=IntervalTrigger(
            seconds=config.sidebar_dispatcher.poll_interval_minutes * 60
        ),
        component="agents",
        description="Publish earliest armed-brief wake time to the device binding",
    )
