"""Deterministic dispatch of brief-bound agents.

When the heartbeat gate reports a brief `fired` and the brief names an
`agent_id`, this module decides — in plain code, no model involved —
whether the agent actually runs, and spawns it when it should:

1. the definition must exist and be `active` (missing/paused/retired → skip);
2. `min_interval_sec` since the definition's `last_run_at` must have elapsed;
3. no run of this agent may already be in flight (spawn registry).

Every outcome is appended to the brief's `last_result` so the skip reason
is visible wherever the brief is read. Dispatch runs at the gate's
`on_completion`, after every brief in the batch has been reported.
"""
from __future__ import annotations

import logging
from typing import Any

from utils.timezone_utils import utc_now, parse_utc_time_string

logger = logging.getLogger(__name__)


def dispatch_brief_agent(
    *,
    db,
    event_bus,
    tool_repo,
    brief: dict,
    mission: str | None,
    condition_check: str | None,
    last_result: str | None,
    bypass_min_interval: bool = False,
    trigger: str = "heartbeat",
) -> dict[str, Any]:
    """Spawn the agent a fired brief names, or record why not.

    `brief` is the work-item context entry (carries id, agent_id,
    dispatch_stamp). Returns a small outcome dict for logging/tests.
    """
    from agents.definitions import get_definition
    from agents.spawn import is_in_flight, spawn_subagent
    from agents.implementations.templated_agent import definition_spec
    from agents.sidebar import WorkItem
    from tools.implementations.heartbeat_tool import append_brief_note

    def _note(note: str) -> None:
        if not brief_id:
            return
        try:
            append_brief_note(db, brief_id, note)
        except Exception:
            logger.exception("brief %s: failed to append dispatch note", brief_id)

    brief_id = brief.get("id")
    agent_key = brief.get("agent_id")
    if not agent_key:
        return {"dispatched": False, "reason": "no agent_id on brief"}

    definition = get_definition(db, agent_key)
    if definition is None:
        note = f"dispatch skipped: agent {agent_key} not found"
        logger.warning("brief %s: %s", brief_id, note)
        _note(note)
        return {"dispatched": False, "reason": "not_found"}

    name = definition["name"]
    state = definition.get("state")
    if state != "active":
        note = f"dispatch skipped: agent {name} is {state}"
        logger.info("brief %s: %s", brief_id, note)
        _note(note)
        return {"dispatched": False, "reason": state}

    last_run = definition.get("last_run_at")
    min_interval = int(definition.get("min_interval_sec") or 3600)
    if last_run and not bypass_min_interval:
        try:
            elapsed = (utc_now() - parse_utc_time_string(last_run)).total_seconds()
        except (ValueError, TypeError):
            elapsed = None
        if elapsed is not None and elapsed < min_interval:
            wait = int(min_interval - elapsed)
            note = f"dispatch skipped: agent {name} ran {int(elapsed)}s ago (min_interval {min_interval}s; {wait}s to go)"
            logger.info("brief %s: %s", brief_id, note)
            _note(note)
            return {"dispatched": False, "reason": "min_interval", "retry_after_sec": wait}

    if is_in_flight("agent", definition["agent_id"]):
        note = f"dispatch skipped: agent {name} is already running"
        logger.info("brief %s: %s", brief_id, note)
        _note(note)
        return {"dispatched": False, "reason": "in_flight"}

    spec = definition_spec(definition)
    work_item = WorkItem(
        item_id=definition["agent_id"],
        interface_name="agent",
        context={
            "subject": mission or "",
            "detail": condition_check or "",
            "last_result": last_result or "",
            "brief_id": brief_id,
            "trigger": trigger,
        },
    )
    spawned = spawn_subagent(spec, work_item, event_bus, tool_repo=tool_repo)
    if not spawned:
        note = f"dispatch skipped: agent {name} not spawned (capacity or duplicate)"
        logger.warning("brief %s: %s", brief_id, note)
        _note(note)
        return {"dispatched": False, "reason": "spawn_refused"}

    note = f"dispatched to agent {name}"
    logger.info("brief %s: %s", brief_id, note)
    _note(note)
    return {"dispatched": True, "agent": name, "agent_id": definition["agent_id"]}


def dispatch_fired_briefs(
    *,
    event_bus,
    tool_repo,
    work_item,
) -> list[dict[str, Any]]:
    """Dispatch every brief in this gate run that reported `fired`.

    A brief counts as fired-this-run when its row shows last_outcome
    'fired' and last_wake_at differs from the dispatch stamp the trigger
    wrote (a report overwrites the stamp; an unreported brief keeps it).
    """
    from utils.userdata_manager import get_user_data_manager
    from utils.user_context import get_current_user_id

    db = get_user_data_manager(get_current_user_id())
    outcomes: list[dict[str, Any]] = []
    for brief in work_item.context.get("briefs", []):
        agent_key = brief.get("agent_id")
        if not agent_key:
            continue
        brief_id = brief.get("id")
        stamp = brief.get("dispatch_stamp")
        row = db.fetchone(
            "SELECT last_outcome, last_wake_at FROM briefs WHERE id = :id",
            {"id": brief_id},
        )
        if row is None:
            continue
        if row.get("last_outcome") != "fired":
            continue
        if stamp and row.get("last_wake_at") == stamp:
            # Reported in an earlier run; not this one.
            continue
        outcomes.append(
            dispatch_brief_agent(
                db=db,
                event_bus=event_bus,
                tool_repo=tool_repo,
                brief=brief,
                mission=brief.get("mission"),
                condition_check=brief.get("condition_check"),
                last_result=brief.get("last_result"),
            )
        )
    return outcomes
