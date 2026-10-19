# cns/core/ — the data a turn is carried in: continuum aggregate, frozen message and event types, the provider-neutral request form, and the session cache rebuild

## Files

- `__init__.py` — package root re-exports; nothing in the repo imports the root, so the only reason to open it is a boot-time ImportError whose traceback runs through this chain.
- `continuum.py` — the aggregate root and its hot message cache. Open it before adding anything that appends to a conversation; `add_tool_message`, `discard_transient_user_message` and its `to_dict`/`from_dict` pair have no caller.
- `events.py` — the event taxonomy and the `create()` factories that build events. Open it before publishing, subscribing, or naming a new event.
- `message.py` — the `Message` value object. Open it before adding a metadata field, or before reading message content on a model's behalf.
- `message_formatter.py` — message list to neutral request dicts. Open it before changing what the model receives as history.
- `segment_cache_loader.py` — the cache-miss load, and the past conversation a returning session opens with. Open it before changing what a new session sees of older history.
- `state.py` — the frozen identity dataclass for one continuum. Open it when constructing or persisting a continuum's identity; its `to_dict`/`from_dict` pair is unexercised.

## Wiring

- Clock and identity: every event `create()` and every rendered timestamp resolves `get_current_user_id` / `get_user_preferences` at call time, so both must run on the request thread or a context-copying thread. `SegmentTimeoutEvent.create` is the deliberate exception and takes `user_id` as a parameter, because the timeout sweep and the queued manual-collapse path run off the request context, where `get_current_user_id()` raises; a new event reached from a scheduled path takes the same form.
- Cache lifecycle: `ContinuumPool.get_or_create` → `SegmentCacheLoader.load_session_cache` (cache miss only) → `Continuum.apply_cache` → `ValkeyMessageCache.set_continuum_if_epoch`. The epoch is read before the database load and the write is a compare-and-set, so a collapse landing mid-load is not clobbered by the list that loaded before it. Counterpart: `cns/infrastructure/AGENTS.md`.
- Persistence: `ContinuumRepository` constructs `ContinuumState` and `Message` directly and stamps `content_type` through `with_metadata(...).to_db_tuple`, while `ValkeyMessageCache` is the only caller of `Message.to_dict`/`from_dict`. The repository never calls the dict pair, so a field the dict pair drops survives a database reload and disappears from the hot cache. Counterpart: `cns/infrastructure/AGENTS.md`.
- Request build: `LiveContextCompactionService.format_messages_for_api` drops covered messages and calls `message_formatter.format_messages_for_api`, which delegates summary rendering to `segment_helpers` — imported function-local, and that placement is what keeps `cns.services` off this package's import chain. Counterpart: `cns/services/AGENTS.md`.
- Reasoning round-trip: `message.py:MessageMetadata.thinking_signatures` / `reasoning_details` are read off the message dict by `clients/llm/dialects/anthropic.py:_accepted_round_trip_fields` and its OpenRouter twin, while `message_formatter.format_messages_for_api` is the writer. A key renamed here reaches neither, and the signatures reach the provider as unsigned reasoning. Counterpart: `clients/llm/dialects/AGENTS.md`.
- Sentinel vocabulary: `cns/services/segment_helpers.py` writes the sentinel metadata this directory selects on and displays; it owns the key set, and readers here track it rather than the reverse. Counterpart: `cns/services/AGENTS.md`.
- Compose round-trip: `ComposeSystemPromptEvent` → `working_memory/core.py:_handle_compose_prompt` → one `UpdateTrinketEvent` per registered trinket → `TrinketContentEvent` → `SystemPromptComposedEvent`. All of it runs synchronously on the publisher's thread under `_compose_lock`, so content arriving after `composer.compose()` misses this prompt. Counterpart: `working_memory/AGENTS.md`.
- Event-name registry: `EventBus.subscribe` validates every subscription string against `cns/integration/event_bus.py:_iter_event_names`, which walks this taxonomy's subclasses — this directory is the bus's name registry. Counterpart: `cns/integration/AGENTS.md`.
- Signup bootstrap: `auth/database.py:_prepopulate_welcome_content` appends straight to `continuum._message_cache`, bypassing the aggregate's own methods. Each row is staged and committed afterwards, so it works; do not extend the pattern.
- Neutral types: `clients/llm/types.py:CacheTTL` rides on the four cacheable content blocks — `TextBlock`, `ImageBlock`, `DocumentBlock`, `FileRefBlock`, and no other — and each dialect translates it to its provider's cache control.
- Primer text: `config/prompts/loader.py:load_prompt` supplies `behavioral_primer.txt`; the block format is parsed here, the words are owned there. Counterpart: `config/prompts/AGENTS.md`.
- Digest consumers: a projection, not a list — `rg 'preprocess_content_blocks'`. Counterpart: `cns/services/AGENTS.md`.

## Rules

- No module-level import of `cns.services` on the `cns/core/__init__.py` chain (`message`, `continuum`, `state`, `events`): importing any `cns.services` module runs `cns/services/__init__.py`, which imports the orchestrator, whose own `from cns.core.continuum import Continuum` re-enters this half-built package and raises ImportError at boot. Keep such imports function-local, the way `message_formatter.format_messages_for_api` does. `segment_cache_loader.py` is the one module here with a module-level `cns.services` import, and it is safe only because nothing reachable from `cns/services/__init__.py` imports the loader back; a module-level importer added inside `cns/services/` closes that cycle and breaks boot.
- A message no user typed must carry `system_notification: True` before it enters the history. `continuum_repository.save_message`'s `is_real_message` test, the formatter's timestamp injection, and `UnitOfWork._get_real_messages` all key on that flag; unflagged, the row counts as a real turn, opens a segment through `_ensure_active_segment`, and lands inside compaction ranges. The deliberate exception is the signup intro pair in `auth/database.py:_prepopulate_welcome_content` — the "Hi, my name is…" turn and MIRA's reply carry `system_generated` alone and are meant to be the account's first real turns.
- `SegmentCacheLoader._primer_turns` holds one list of primer `Message` instances that every load with at least one collapsed segment puts into that user's cache, so an in-place metadata write on one of them reaches every other user's primer turns; stamp with `with_metadata` and copy with `dict(message.metadata)` where isolation is the point.

Elsewhere: this directory owns the shapes a turn is carried in, not the work done with them once they exist — orchestration, segment collapse, memory surfacing, and persistence all sit outside its boundary. Why a rule on this page reads as it does is `git log`. Every destination this directory reaches is an edge — see `## Wiring` above.

## A past segment reaching the model

The case behind most of the invariants above: history the user can no longer see still arrives in the next request.

1. `create_segment_boundary_sentinel` opens the segment: it writes `is_segment_boundary: True`, `status: 'active'` and `segment_id`, and sets `created_at` one microsecond before the first message so a chronological query orders it ahead of the messages it brackets. Collapse later UPDATEs that same row: `collapse_segment_sentinel` writes the summary into `content` and stamps `precis`, `display_title`, `complexity_score` and `status: 'collapsed'`, carrying the original `created_at` through.
2. `find_collapsed_segments` selects on those two metadata keys, newest first, and reverses the rows — the candidate pool the loader receives is chronological.
3. Tier selection walks newest-first, then reverses each tier, and `_load_segment_summaries` stamps `display_mode` and concatenates tier 2 ahead of tier 1. That concatenation is what leaves the list chronological end to end, and the newest segment at `[-1]`.
4. `create_session_boundary_marker` reads that last element's `segment_end_time` for the gap it states, falling back to the sentinel's own timestamp — reorder step 3 and the boundary reports the wrong session's end.
5. At request build the sentinel stops being a summary: `format_messages_for_api` emits an assistant `tool_call` on `continuum_tool` and a matching tool result, so past segments reach the model framed as retrieved data rather than as prior assistant prose.
