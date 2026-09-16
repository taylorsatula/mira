# working_memory/ — Event-driven system prompt composition via trinkets

Trinkets live in `trinkets/` and are documented in `working_memory/trinkets/AGENTS.md` (base classes, per-trinket behavior, persistence, result-feed state machine). This map owns composition, event routing, and collapse flush.

## Rules

- Placement is controlled exclusively by `SECTION_LAYOUT` in `composer.py`. Sections not listed there default to `system` placement at the end, with a warning — a new trinket works immediately but lands in the wrong slot until its `variable_name` is added to the appropriate layout list. Do not route placement inside `generate_content()` or a trinket's render path.
- Trinket registration happens outside this directory: trinkets self-register via `working_memory.register_trinket()` (`working_memory/trinkets/base.py`) and are instantiated by the CNS factory — `cns/integration/AGENTS.md` owns the factory/trinket-registration ordering. Do not instantiate trinkets elsewhere.
- `WorkingMemory._handle_update_trinket()` is the sole isolation boundary: it catches every exception from `handle_update_request()` so one failing trinket cannot break the compose broadcast. It classifies failures by exception type name (`Database` / `Valkey` / `Connection` substrings → `infrastructure` category, everything else `logic`) — an infrastructure exception class whose name matches none of those substrings is misclassified as `logic`. Trinkets must still propagate infrastructure failures; never catch inside a trinket.
- `invalidate_trinket(trinket_name, user_id)` temporarily swaps the user contextvar (`set_current_user_id(user_id)`) around `trinket._clear_from_valkey()` because that method reads the user from context. External services calling it must pass the correct user explicitly; the swap is restored in a `finally`.
- Trinket state persists in the Valkey hash `trinkets:{user_id}` — key prefix contract owned by `working_memory/trinkets/AGENTS.md` (`working_memory/trinkets/base.py:TRINKET_KEY_PREFIX`). The read side here (`get_trinket_state()` / `get_all_trinket_states()`) builds the same key; changing one side without the other silently reads empty state.

## Files

- `types.py` — `ComposedPrompt`, `TrinketState`, `TrinketStatesMeta`, `AllTrinketStates` TypedDicts for the dict shapes produced by `composer.py` and `core.py`
- `composer.py` — `SystemPromptComposer`: section collection via `add_section()` / `set_base_prompt()`, and placement/ordering authority `SECTION_LAYOUT` + `compose()` routing into `ComposedPrompt` fields (`cached_content`, `non_cached_content`, `conversation_prefix_items`, `post_history_items`, `notification_center`). `set_base_prompt()` appends the `═`-delimiter scaffolding note; `_build_notification_center()` wraps notification parts in `<mira:hud>`.
- `core.py` — `WorkingMemory`: event subscriptions (`ComposeSystemPromptEvent` → `_handle_compose_prompt()`, `UpdateTrinketEvent` → `_handle_update_trinket()`, `TrinketContentEvent` → `_handle_trinket_content()`, `SegmentCollapsedEvent` → `_flush_stateful_trinkets()`), trinket registry (`register_trinket()`), external entry points (`publish_trinket_update()`, `invalidate_trinket()`, `invalidate_portrait()`, `get_trinket()`), Valkey state reads (`get_trinket_state()`, `get_all_trinket_states()`), and `_portrait_cache`. Gotcha: `_handle_compose_prompt()` also performs template substitution on the base prompt — see the deep-dive.
- `__init__.py` — re-exports `WorkingMemory` only; registration happens via the CNS factory (`cns/integration/AGENTS.md`).
- `trinkets/` — self-registering prompt-section components; one `variable_name` slot each. See `trinkets/AGENTS.md`.

## Wiring

Composition flow (all synchronous, single request; owned here):

`ComposeSystemPromptEvent` → `WorkingMemory._handle_compose_prompt()` substitutes base-prompt templates, calls `composer.set_base_prompt()` + `clear_sections(preserve_base=True)`, broadcasts `UpdateTrinketEvent` per registered trinket → each trinket's `handle_update_request()` renders and publishes `TrinketContentEvent` → `_handle_trinket_content()` calls `composer.add_section()` → `composer.compose()` routes by `SECTION_LAYOUT` → `SystemPromptComposedEvent`

The round-trip producer is `cns/services/orchestrator.py` (`_compose_llm_messages()`): it nulls its `_cached_content` / `_non_cached_content` / `_conversation_prefix_items` / `_post_history_items` / `_notification_center` fields, publishes `ComposeSystemPromptEvent`, then reads those same fields — which `_handle_system_prompt_composed()` (subscribed to `SystemPromptComposedEvent`) has just refilled. Delivery is synchronous inside `cns/integration/event_bus.py:publish()`, so the fields are populated by the time `publish()` returns; there is no timeout or retry. The bus swallows subscriber exceptions (logged, at-most-once, remaining subscribers still run — `cns/integration/AGENTS.md` owns the bus contract), and `_handle_update_trinket()` additionally catches every per-trinket exception — so a failed compose or failed trinket degrades SILENTLY: the orchestrator falls back to empty-string defaults and the prompt ships without the failed section or, if `_handle_compose_prompt()` itself dies, without any trinket content at all. Failure visibility is logs only (`error_category` extra on trinket failures). `cns/services/pollers/segment_poller.py` also subscribes `ComposeSystemPromptEvent` (poller lifecycle start — `cns/services/AGENTS.md` owns it).

Ordering invariants in this flow: `clear_sections()` must run before the trinket broadcast (stale sections would otherwise persist into the new prompt); the broadcast completes synchronously before `compose()`, so every trinket's content is present at composition time; `publish_trinket_update()` is a no-op with a warning until `_handle_compose_prompt()` sets `_current_continuum_ids[user_id]` (per-user dict — multi-user composes cannot steal each other's continuum context) — external callers cannot trigger trinket updates before the first compose of the process for their user. Trinket registration order (factory instantiation order in `cns/integration/factory.py`) does NOT determine slot order — it only sets the `UpdateTrinketEvent` broadcast order; the composed sequence is solely `composer.py:SECTION_LAYOUT` iterating placement lists in dict order. Reordering factory instantiations changes nothing in the prompt; adding a trinket without a `SECTION_LAYOUT` entry lands it in system-at-end with a warning (see Rules).

Collapse flow: `SegmentCollapsedEvent` (published by the collapse path in `cns/services/`) → `WorkingMemory._flush_stateful_trinkets()` calls `_clear_all_state()` + `clear_user_turn()` + `_clear_from_valkey()` on every registered `StatefulTrinket`, then `invalidate_portrait(user_id)`. Per-trinket expiry behavior is owned by `working_memory/trinkets/AGENTS.md` (`base.py`).

Compose is serialized process-wide by the module-level `_compose_lock` (an `RLock` in `core.py`) held across three windows: `orchestrator._compose_llm_messages()` from field-reset through reading the five composed-prompt fields into locals, the whole `_handle_compose_prompt()` handler, and `_handle_trinket_content()` / `_handle_update_trinket()` (mid-turn `publish_trinket_update()` mutates the same composer). RLock, not Lock, because compose re-enters synchronously through the event bus on the same thread.

External update flow: services that mutate trinket inputs publish `UpdateTrinketEvent` (or call `publish_trinket_update()`) with a `target_trinket` class name; `cns/services/AGENTS.md` owns the producer side of those events. The trinket must be registered by class name or the update is dropped with a warning.

## Base-prompt template substitution

`_handle_compose_prompt()` replaces template variables in the event's base prompt before composition. The contract is split: the template strings live in the base-prompt source (`config/system_prompt.txt` and related prompt files in `config/`), the substitution values are sourced here:

| Template | Substituted with | Source |
|---|---|---|
| `{first_name}` | user's first name, falling back to `"friend"` when unset/blank | `get_user_preferences().first_name` |
| `{user_context}` | portrait text prefixed with a newline, or empty string | `cns/services/portrait_service.read_portrait()`, cached in `_portrait_cache` per user until `invalidate_portrait()` |
| `{relative time since account creation}` | humanized duration, or `"some time"` when `created_at` is unset | `format_relationship_duration(prefs.created_at)` |
| `{model_id}` / `{model_name}` | primary route's model id and name | `get_model_config("primary")` — fixed for the process lifetime; per-user model switching is not a feature |

A portrait is cached per user in `_portrait_cache` and only invalidated by `invalidate_portrait()` (called from `_flush_stateful_trinkets()` and external callers). Editing a portrait's source data does not appear in prompts until something calls `invalidate_portrait()`.
