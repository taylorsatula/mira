# agents/triggers/ — dispatcher work-item discovery triggers

## Rules

- Triggers are pure discovery: `check_for_new_items(user_id)` returns `WorkItem`s or `[]`, makes no judgment calls, and performs no dedup — the dispatcher owns dedup via `sidebar_activity` (`agents/sidebar.py`). `on_dispatched()` is for side effects only; `MemoryFloorTrigger.on_dispatched()` is a no-op.
- The floor cadence gate is an inline per-user modular check (`activity_days % cfg.floor_use_days`) using `get_user_cumulative_activity_days()`, not a call to `get_users_due_for_job()` — the dispatcher already iterates users, so re-querying all users per poll is both wasteful and wrong for this shape.
- `item_id` must stay stable per activity-day cycle (`floor_{user_id}_{activity_days}`) — the dispatcher's `sidebar_activity` dedup depends on the same key reproducing within a cycle; changing the key format silently breaks dedup.
- `agent_class` is a lazy `@property` that imports `MemoryCuratorAgent` at access time — resolve it to a module-level import and the `memory_curator_agent` → `tools.implementations.memory_tool` → `lt_memory` import cycle breaks module load.
- Failed floor runs are not retried: `max_retries=0` by dispatcher policy; the next due activity day re-samples. A retry path here would double-process the same sample.

## Files

- `__init__.py` — re-exports `MemoryFloorTrigger` only.
- `memory_floor_trigger.py` — `MemoryFloorTrigger` (`trigger_id = "memory_floor"`, `interface_name = "memory_curator_floor"`). Deterministic SQL discovery, no LLM: `LTMemoryDB.get_floor_candidates()` samples low-importance memories un-tended for `cfg.floor_unseen_days` (`config.config.MemoryCuratorConfig`), then `get_memories_by_ids()` fetches texts/scores so the agent judges without a read round-trip. Emits a single floor-mode `WorkItem` whose context matches `FloorContext` (owned by `agents/implementations/memory_curator_agent.py`). Consumers: `agents/sidebar.py` (SidebarDispatcher poll loop) → spawns `MemoryCuratorAgent` in floor mode.
