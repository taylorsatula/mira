"""Trigger package root.

Each trigger discovers work items for a sidebar agent. Triggers are pure
discovery with no judgment — the dispatcher owns dedup via
`sidebar_activity`, and the agent owns every decision.

Nothing in the repo imports this package root: consumers import
`agents.triggers.registry` (dispatch and the rules API),
`agents.triggers.heartbeat_trigger` (the wake job), and
`agents.triggers.rule_filter` directly. The re-export below is still the
add-a-trigger walkthrough's registration step, so a new trigger class joins
it even though no live code path reads it here. Registering a class means
adding it to `agents.triggers.registry.TRIGGER_CLASSES`; a class that exists
only in this `__all__` is unreachable by both the dispatcher and the
trigger-rules API.
"""
from agents.triggers.memory_floor_trigger import MemoryFloorTrigger
from agents.triggers.heartbeat_trigger import HeartbeatTrigger

__all__ = ["MemoryFloorTrigger", "HeartbeatTrigger"]
