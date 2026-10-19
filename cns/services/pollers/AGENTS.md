# cns/services/pollers/ — pollers whose lifetime is a conversation segment: a per-user thread starts on the segment's first compose and stops on collapse

## Files

- `__init__.py` — docstring only; no re-exports.
- `segment_poller.py` — the poller base class. Open it before adding a poller or changing how one starts, stops, or publishes.
- `inbox_poller.py` — IMAP unread-header polling for the active segment. Open it before changing which header fields the trinket has to render, or how a cycle classifies an IMAP failure.

## Wiring

- Lifecycle in: `cns/services/orchestrator.py:_compose_llm_messages` publishes `ComposeSystemPromptEvent` inside `working_memory.core._compose_lock` on every LLM request → `SegmentPoller._on_compose`; `cns/services/segment_collapse_handler.py` publishes `SegmentCollapsedEvent` on both the success and the tombstone path → `SegmentPoller._on_segment_collapsed`. A collapse path that skips its publish leaves the poller polling a dead segment. Counterpart: `cns/services/AGENTS.md`.
- Data out: `SegmentPoller._poll_loop` → `UpdateTrinketEvent.create(target_trinket=...)` → `working_memory/core.py:_handle_update_trinket` → the target's `handle_update_request`. Routing resolves the target by registered class name, so a `target_trinket` naming no constructed trinket drops every update with one warning and that prompt slot never refreshes. What arrives inside `context` is the trinket's own contract, not this directory's: renaming a payload key here reaches `working_memory/trinkets/email_trinket.py:generate_content` and silently blanks the slot, and the trinket must also tolerate the `context={}` refresh `_handle_compose_prompt` publishes to every trinket. Counterparts: `working_memory/AGENTS.md`, `working_memory/trinkets/AGENTS.md`.
- Construction: `cns/integration/factory.py:_initialize_inbox_poller` constructs the target trinket and its poller together, after working memory exists and before the orchestrator: the trinket's constructor needs working memory, and a poller built before its target trinket publishes into a class name nothing has registered, so every update is dropped behind one warning and that prompt slot never refreshes. Counterpart: `cns/integration/AGENTS.md`.
- Config chain: `inbox_poller._try_load_config` → `utils/tool_config_store.py:load_user_tool_config("email_tool", hydrate_secrets=True)`. Which fields hydrate is decided by `json_schema_extra={"secret": True}` on `EmailToolConfig`, so a rename in `tools/implementations/email_tool.py` stops hydration and the poller reports an unconfigured user instead of a schema error. Counterparts: `utils/AGENTS.md`, `tools/implementations/AGENTS.md`.

## Rules

- Adding a poller is four coordinated edits: subclass `SegmentPoller` with `poll_interval_seconds`, `target_trinket` (the target trinket's class `__name__`, verbatim) and `poller_name`; implement `_try_load_config` and `_poll_once`; construct exactly one instance in `cns/integration/factory.py` beside its target trinket; and confirm the trinket tolerates a refresh whose `context` carries no `'data'` key and wraps poller data as untrusted content. Each skipped step fails silently: no construction line and the poller never runs; a `target_trinket` that is not a registered class name and every update is dropped; a second instance and the bus calls both, giving one user two threads and duplicated external calls.
- `_try_load_config` runs on the composing thread inside `working_memory.core._compose_lock`, and the reverse trip takes the same lock — the poll thread's `UpdateTrinketEvent` publish enters `_handle_update_trinket` under it. A network call in `_try_load_config` stalls every compose in the process; a compose in flight delays any poll publish. The unconfigured path caches nothing, so it re-reads the config store on every compose.
- Removing or bypassing `utils/distributed_lock.py:UserRequestLock` in either chat transport is a change to poller safety, not just to turn serialization. Keeping one turn per user is what stops a second `_on_compose` start from overwriting the first's state entry and orphaning a thread that then polls until process exit, duplicating external calls and trinket publishes; the registry alone does not stop it, for the reason `segment_poller.py`'s docstring gives.

Elsewhere: this directory owns a poller's lifetime and its handoff to a trinket, not the events that drive it, the trinket that renders its data, or the config it reads. Event definitions are `cns/core/events.py`; the subscriber and publisher sets are projections — `rg 'ComposeSystemPromptEvent'`, `rg 'SegmentCollapsedEvent'`; why a rule here reads as it does is `git log`. Every other destination this directory reaches is an edge — see `## Wiring` above.

## One poller's life, from compose to collapse

The case that decides most of the Rules above:

1. `cns/integration/factory.py:_initialize_inbox_poller` constructs the target trinket first and the poller second. The trinket registers itself into working memory under its class name; the poller's constructor subscribes to two events and starts no thread.
2. A turn reaches `cns/services/orchestrator.py:_compose_llm_messages`, which takes `_compose_lock` and publishes `ComposeSystemPromptEvent`. The bus calls `_on_compose` inline on that thread, still inside the lock.
3. `_on_compose` finds no entry for the user, so it calls `_try_load_config()` — here the user's `email_tool` config, or None. None ends this turn's involvement: no thread, no error, and the same check runs again on the next compose because nothing recorded the absence.
4. With a config, a stop event and a copied context are made, the state is inserted under the lock, and a daemon thread starts. The copied context is what carries the user id into `UpdateTrinketEvent.create` and into the trinket's per-user snapshot writes.
5. Each cycle calls `_poll_once` and publishes only when the result is not None. The publish is routed by `_handle_update_trinket`, which takes `_compose_lock`, so a poll publish and a compose serialize against each other.
6. The segment collapses. `SegmentCollapsedEvent` reaches `_on_segment_collapsed`, which pops the state, sets the stop event, and joins with a bounded timeout. A cycle already blocked in IMAP outlives that join, may publish once after the collapse, and is still running when the next compose starts a replacement; on exit it pops the registry entry only if that entry still holds its own stop event, so it cannot evict the replacement's.
7. Step 6 happens only if some path publishes the collapse event. The tombstone path in `cns/services/segment_collapse_handler.py` publishes the same event the success path does for exactly this reason.
