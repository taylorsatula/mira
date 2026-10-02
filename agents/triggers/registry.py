"""Runtime sidebar trigger registry — the single source of truth for which
triggers exist and which of their work-item context fields are filterable.

Both consumers import this module and nothing else:
- dispatch: utils/sidebar_jobs.py:register_sidebar_jobs instantiates and
  registers every class in TRIGGER_CLASSES on the SidebarDispatcher — the
  runtime registry the dispatcher consults on each poll;
- API: cns/api/trigger_rules.py derives the valid trigger_id and field sets
  for rule validation from registered_trigger_specs() at request time.

There is deliberately no second, hand-maintained table anywhere: adding a
class to TRIGGER_CLASSES makes it dispatchable AND rule-addressable in the
same change; removing one makes the API reject its trigger_id in the same
change.
"""
from dataclasses import dataclass

from agents.triggers.memory_floor_trigger import MemoryFloorTrigger

# Every class here is instantiated and registered on the SidebarDispatcher at
# startup by utils/sidebar_jobs.py:register_sidebar_jobs. Order is dispatch
# order. A trigger class not in this list is unreachable by the dispatcher
# AND rejected by the trigger-rules API.
TRIGGER_CLASSES: list[type] = [
    MemoryFloorTrigger,
]


@dataclass(frozen=True)
class TriggerSpec:
    """Class-level metadata for a registered trigger (no instantiation)."""
    trigger_id: str
    interface_name: str
    # Regex-matchable scalar context keys of this trigger's WorkItems.
    # Declared per trigger class as FILTERABLE_FIELDS, derived from its
    # work-item context shape (see each trigger's declaration).
    filterable_fields: tuple[str, ...]


def registered_trigger_specs() -> list[TriggerSpec]:
    """Trigger specs for every registered class, derived at call time."""
    specs: list[TriggerSpec] = []
    for cls in TRIGGER_CLASSES:
        specs.append(TriggerSpec(
            trigger_id=cls.trigger_id,
            interface_name=cls.interface_name,
            filterable_fields=tuple(getattr(cls, "FILTERABLE_FIELDS", ())),
        ))
    return specs


def _validate_registry() -> None:
    """Fail loud on duplicate identity keys — a duplicate would silently
    shadow one trigger in the dispatcher and key-collide rules in the API."""
    trigger_ids = [s.trigger_id for s in registered_trigger_specs()]
    interface_names = [s.interface_name for s in registered_trigger_specs()]
    if len(set(trigger_ids)) != len(trigger_ids):
        raise ValueError(f"Duplicate trigger_id in TRIGGER_CLASSES: {trigger_ids}")
    if len(set(interface_names)) != len(interface_names):
        raise ValueError(
            f"Duplicate interface_name in TRIGGER_CLASSES: {interface_names}"
        )


_validate_registry()
