# How to Build a Trinket

*Guide for creating event-driven system prompt components*

## What is a Trinket?

A **trinket** is an event-driven component that owns exactly one named section of the composed system prompt. Unlike tools (which execute user requests), trinkets **passively reflect system state** in the model's context.

**Key Differences from Tools:**

| Aspect | Tools | Trinkets |
|--------|-------|----------|
| Purpose | Execute user operations | Display system state in prompt |
| Trigger | Model invokes via function call | Compose broadcast + producer events |
| Interaction | Active command execution | Passive observation and reporting |
| Output | Result dict returned to the model | One `variable_name` section in the system prompt |
| Lifetime | Fresh instance per `get_tool()` call | Process-global singleton, shared across all users |
| Examples | Send email, search web | Inbox status, active reminders, forage results |

For autonomous multi-step work with no user present, build a **sidebar agent** instead — see `agents/HOW_TO_BUILD_AN_AGENT.md` for the three-way comparison.

## When to Build a Trinket

Build a trinket when you need to:

✅ **Display dynamic state** in the system prompt (inbox, reminders, background task results)
✅ **Update the prompt on producer events** (a tool, service, poller, or agent publishes at you)
✅ **Provide ambient awareness** (what's running, what's available, recent activity)
✅ **Render session-stable context** (persona directives, user-model observations, domaindoc content)

❌ **Don't build a trinket** for:
- Executing user commands (use a tool instead)
- One-time data retrieval (use a tool)
- Processing user input (use a tool)
- Autonomous multi-step work (use a sidebar agent)
- Persistent storage (trinket state is in-memory and dies on segment collapse; the producer persists)

## Essential Patterns

### Pattern 0: Pick Your Base Class

| Base class | Use when | You must implement | You get free |
|---|---|---|---|
| `EventAwareTrinket` | Stateless render, or state that survives segment collapse | `generate_content()` | Valkey persist/clear, self-registration, publish cycle |
| `StatefulTrinket` | Turn-scoped state that must expire or reset on collapse | `generate_content()`, `_expire_items()`, `_clear_all_state()` | `current_turn` (per-user), `TurnCompletedEvent` subscription, `_refresh_content()` |

Live split: `manifest`, `time_manager`, `location`, `persona`, `lora`, `proactive_memory`, `reminder_manager`, `domaindoc`, `asyncactivity` are plain; `email`, `forage`, `whilethecatsaway`, `memory_curator`, `peanutgallery`, `live_context_compaction` are stateful.

### Pattern 1: Basic Trinket Structure

**See:** `time_manager.py` (smallest), `manifest_trinket.py` (query + format)

```python
from typing import Dict, Any
from .base import EventAwareTrinket

class MyTrinket(EventAwareTrinket):
    """Brief description of what this trinket displays."""

    # REQUIRED class attribute -- base __init__ raises TypeError if falsy.
    # Dual-purpose: names the SECTION_LAYOUT slot in composer.py AND the
    # Valkey hash field under trinkets:{user_id}. Must be globally unique.
    variable_name = "my_section"

    # True only for content that rarely changes (prompt-caching hint)
    cache_policy = False

    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate the content for this trinket.

        Args:
            context: Update context with any relevant data

        Returns:
            Formatted string for system prompt, or "" if legitimately no content
        """
        return "<my_section>content here</my_section>"
```

`variable_name` is a **class attribute**, not a method. There is no `_get_variable_name()` in this codebase — defining one silently leaves the required attribute unset and the trinket raises `TypeError` at instantiation.

### Pattern 2: Event Subscription

**See:** `working_memory/trinkets/base.py:StatefulTrinket._on_turn_completed` — the only subscription a trinket should normally need

**Default: subscribe to nothing.** Most trinkets are pure renderers driven by the compose broadcast and by producers publishing `UpdateTrinketEvent` at them.

| Event | Who handles it | You |
|---|---|---|
| `ComposeSystemPromptEvent` | `working_memory/core.py` broadcasts `UpdateTrinketEvent` per registered trinket | implement `generate_content()` |
| `UpdateTrinketEvent` | `working_memory/trinkets/base.py:handle_update_request()` | override only to consume context first (Pattern 3) |
| `TrinketContentEvent` | `working_memory/trinkets/base.py` publishes it for you | never publish directly |
| `TurnCompletedEvent` | `StatefulTrinket` already subscribes | inherit `StatefulTrinket`, do not re-subscribe |
| `SegmentCollapsedEvent` | `WorkingMemory._flush_stateful_trinkets()` calls `_clear_all_state()` | implement `_clear_all_state()`, never subscribe |

Subscribing to `TurnCompletedEvent` or `SegmentCollapsedEvent` yourself duplicates the central lifecycle and is a review defect. If you need turn-driven expiry, that is `StatefulTrinket._expire_items()`.

To trigger a re-render from inside the trinket:

```python
self.working_memory.publish_trinket_update(
    target_trinket="MyTrinket",          # class name, must be registered
    context={"action": "lifecycle_refresh"}
)
```

`publish_trinket_update()` is a no-op with a warning until the first compose of the process sets `_current_continuum_id`.

### Pattern 3: Custom Update Handling

**See:** `email_trinket.py:handle_update_request` (snapshot store), `forage_trinket.py` (status state machine)

Ordering is load-bearing: **consume the event context first, then call `super()`** so the render sees the mutation. Rendering before mutating leaves the prompt one cycle behind the producer.

```python
def handle_update_request(self, event) -> None:
    """Store producer data, then delegate to parent for render + publish."""
    data = event.context.get('data')

    # A context with no action/data key is a LIFECYCLE REFRESH (compose
    # broadcast or _expire_items re-render) -- re-render from existing
    # state, never treat it as new input.
    if data is not None:
        self._snapshots[get_current_user_id()] = data

    super().handle_update_request(event)
```

Result-feed trinkets that key on `task_id` return `super().handle_update_request(event)` early when the key is absent — see `forage_trinket.py`, `memory_curator_trinket.py`, `whilethecatsaway_trinket.py`.

### Pattern 4: State Management

**See:** `email_trinket.py:_inbox_snapshots`, `peanutgallery_trinket.py:_user_guidance`, `forage_trinket.py:_user_results`

Trinket instances are **process-global singletons shared across every user**. All mutable state must be a dict keyed by the user-id contextvar — never a bare instance attribute, or user A's state renders into user B's prompt.

```python
def __init__(self, event_bus, working_memory):
    super().__init__(event_bus, working_memory)
    self._items: dict[str, Dict[str, Any]] = {}   # user_id -> state

def generate_content(self, context):
    items = self._items.get(get_current_user_id(), {})
    ...
```

Turn counters come from `StatefulTrinket.current_turn` (already per-user); do not shadow it with your own attribute.

**Important:** this state is **in-memory only** and dies on collapse via `_clear_all_state()`. For data that must survive, the producer persists it (tool SQLite, Valkey, Postgres) and the trinket re-reads it per render — `asyncactivity_trinket.py` reads `sidebar_activity` on every render for exactly this reason.

### Pattern 5: Content Formatting

**See:** `email_trinket.py:generate_content`, `manifest_trinket.py:_format_manifest`

Every live trinket renders **XML**, not banner text:

```python
def generate_content(self, context: Dict[str, Any]) -> str:
    snapshot = self._snapshots.get(get_current_user_id(), [])
    if not snapshot:
        return ""                      # legitimately empty, not a failure

    lines = ['<inbox_status>']
    lines.append(f'<unread count="{len(snapshot)}">')
    for item in snapshot:
        lines.append(f'<email from="{_xml_attr_escape(item["from"])}"/>')
    lines.append('</unread>')
    lines.append('</inbox_status>')
    return "\n".join(lines)
```

**Formatting rules:**
- One root element named after the section; nested elements for structure
- Escape user-controlled text before interpolation: `_xml_attr_escape()` (`email_trinket.py`) for attribute values, `html.escape()` (`asyncactivity_trinket.py`) for text nodes. Result-feed trinkets currently interpolate `query`/`topic` unescaped — a known gap, not a license
- Return `""` when there is genuinely nothing to show; the composer strips empty sections and `working_memory/trinkets/base.py` clears the stale Valkey field
- Never wrap your own output in `---` separators or placement scaffolding — `composer.py` owns that

### Pattern 6: Caching

**See:** `location_trinket.py`, `domaindoc_trinket.py`, `persona_trinket.py`, `lora_trinket.py` — the four that set it

`cache_policy` is a **provider prompt-cache placement flag**, not a result cache. `composer.py:_route_section` sends `cache_policy=True` sections into `ComposedPrompt.cached_content` (the cache-eligible prefix) instead of `non_cached_content`.

```python
class MyTrinket(EventAwareTrinket):
    variable_name = "my_section"
    cache_policy = True   # stable within a session -- don't bust the prompt cache
```

**Only affects `PLACEMENT_SYSTEM` sections.** `post_history`, `conversation_prefix`, and `notification` placements are routed before the cache check, so `cache_policy` on a HUD trinket is a no-op. Placement is decided by `SECTION_LAYOUT` in `composer.py` (see Registration below).

**Set True for:** user preferences, persona/LoRA directives, domaindoc content, location — anything stable within a session.

**Leave False for:** turn-scoped state, active results, anything time-sensitive. A volatile section marked cached invalidates the provider cache for everything after it.

### Pattern 7: Error Handling

**See:** `working_memory/trinkets/base.py:handle_update_request` docstring — "let infrastructure failures propagate"

**Do not wrap `generate_content()` in a blanket try/except that returns `""`.** `WorkingMemory._handle_update_trinket()` is the *sole* isolation boundary: it catches every exception from `handle_update_request()` so one failing trinket cannot break the compose broadcast, and it classifies the failure (`Database`/`Valkey`/`Connection` in the exception class name → `infrastructure`, else `logic`). Swallowing inside the trinket hides an outage as "no content" and defeats that classification.

```python
def generate_content(self, context: Dict[str, Any]) -> str:
    user_id = get_current_user_id()

    # Infrastructure failures RAISE -- core.py isolates and logs them.
    segments = get_manifest_query_service().get_segments(user_id)

    if not segments:
        logger.debug("No segments available for manifest")
        return ""       # legitimately empty, NOT a failure

    return self._format_manifest(segments)
```

| Situation | Return |
|---|---|
| No data yet (new user, empty inbox, no segments) | `""` — and say so in a `logger.debug` |
| DB/Valkey/embeddings failure | raise — never catch |
| One malformed row inside an otherwise good list | skip the row, `logger.warning`, render the rest (`manifest_trinket.py:_format_time_range` returns `"[Unknown]"`) |
| Missing user context | raise — a trinket rendering with no user is a wiring bug, not an empty state |

Narrow `except` is fine around a *single* formatting call whose failure should not discard the whole section. It is not fine around the data fetch.

### Pattern 8: Database Queries

Pick the store by where the data lives. User context is already set by the compose path, so RLS and per-user scoping apply automatically.

**User SQLite (tool-owned data)** — `asyncactivity_trinket.py`, `domaindoc_trinket.py`:

```python
from utils.userdata_manager import get_user_data_manager
from utils.user_context import get_current_user_id

db = get_user_data_manager(get_current_user_id())
rows = db.execute("SELECT * FROM sidebar_activity WHERE status != 'dismissed'")
```

**Decrypt or die, literally:** only `db.select(table, ...)` decrypts `encrypted__` columns. `execute()` / `fetchone()` / `fetchall()` return rows **verbatim** — any `encrypted__` field comes back as a Fernet ciphertext blob. If your table has encrypted columns, use `select()`; if you need raw SQL beyond what `select()` supports, route the rows through `db._decrypt_dict(row)` (the same private helper `domaindoc_tool.py` uses throughout). Putting ciphertext into `generate_content()` fails noisily at the model, not at the read -- nothing warns you at fetch time.

Cross-user reads (shared domaindocs) fetch the owner's manager explicitly: `get_user_data_manager(share.owner_user_id)`.

**Postgres (service data)** — go through the owning service, not raw SQL. `manifest_trinket.py` calls `get_manifest_query_service().get_segments(user_id)`; `persona_trinket.py` calls `PersonaRepository.get_current_revision()`. A trinket that hand-writes `SELECT ... WHERE user_id = %(user_id)s` duplicates an RLS guarantee the client already provides.

**Valkey (request-scoped cache)** — `location_trinket.py` reads the `location:{user_id}` key written by `cns/api/location.py`.

`utils/database_session_manager.get_shared_session_manager()` is for services that own a `LTMemoryDB`/repository (e.g. `agents/triggers/memory_floor_trigger.py`), not for trinket renders.

### Pattern 9: Timezone Handling

**See:** `manifest_trinket.py:_group_segments_by_date` / `_format_time_range`, `time_manager.py`

Stored timestamps are UTC; everything the model reads is user-local. Parse with `parse_utc_time_string()` (never bare `datetime.fromisoformat`, which drops the tz contract):

```python
from utils.timezone_utils import parse_utc_time_string, convert_from_utc, format_datetime
from utils.user_context import get_user_preferences

def _format_timestamp(self, utc_timestamp: str) -> str:
    user_tz = get_user_preferences().timezone   # raises without user context -- let it
    local_dt = convert_from_utc(parse_utc_time_string(utc_timestamp), user_tz)
    return format_datetime(local_dt, "date_time_short")
```

`manifest_trinket.py` catches `Exception` around the tz lookup and falls back to `'UTC'` for date *grouping* labels only — a wrong bucket label is cosmetic, a wrong wall time is not. Do not copy that fallback into anything the model acts on.

### Pattern 10: Triggering Updates from Tools

**See:** `forage_tool.py` (publishes `UpdateTrinketEvent` on the event bus), `sidebaragents_tool.py` (uses `agents/base.py:_publish_trinket_refresh`)

Two producer shapes, pick by what the tool already holds:

```python
# A: tool was DI-injected with WorkingMemory (constructor arg)
self._working_memory.publish_trinket_update(
    target_trinket="MyTrinket",              # CLASS NAME, must be registered
    context={"action": "state_updated", "data": payload},
)

# B: tool holds the event bus (e.g. spawning a background agent)
from cns.core.events import UpdateTrinketEvent
self.event_bus.publish(UpdateTrinketEvent.create(
    continuum_id=continuum_id,               # 'sidebar' for agent-driven updates
    target_trinket="MyTrinket",
    context={"task_id": task_id, "status": "in_progress", "summary": summary},
))
```

`target_trinket` is matched against registered **class names**. A name with no registered trinket is dropped with a warning and the refresh silently never happens — `punchclock_tool.py` publishes to `PunchclockTrinket`, which does not exist, and is a live example of that dead target. Verify the class name before shipping.

Working Memory injection into a tool is signature-driven: declare `working_memory: Optional["WorkingMemory"] = None` in the tool's `__init__` and `ToolRepository.get_tool()` supplies it (see `tools/HOW_TO_BUILD_A_TOOL.md`).

## Complete Example: Result-Feed Trinket

A `StatefulTrinket` fed by a producer tool, with per-turn expiry. Mirrors
`whilethecatsaway_trinket.py` / `forage_trinket.py` — the shape you want for any
"background task publishes results into the prompt" trinket.

```python
"""Task result trinket -- renders background task outcomes into the HUD."""
import html
import logging
from typing import Any, Dict, TypedDict

from .base import StatefulTrinket
from utils.user_context import get_current_user_id

logger = logging.getLogger(__name__)

RESULT_TTL_TURNS = 8      # successes auto-expire after this many turns
ERROR_TTL_TURNS = 5       # failures expire sooner


class _Result(TypedDict):
    status: str           # pending | in_progress | success | timeout | failed
    query: str
    summary: str
    expires_at_turn: int | None


class MyTaskTrinket(StatefulTrinket):
    """Background task results. Fed by UpdateTrinketEvent from the producer."""

    variable_name = "my_task_results"    # must be a SECTION_LAYOUT slot, globally unique
    cache_policy = False                 # HUD placement -- cache_policy is a no-op here

    def __init__(self, event_bus, working_memory):
        super().__init__(event_bus, working_memory)
        self._user_results: dict[str, Dict[str, _Result]] = {}   # user_id -> task_id -> result

    def handle_update_request(self, event) -> None:
        """Consume producer context FIRST, then render via super()."""
        task_id = event.context.get('task_id')
        if not task_id:
            # Lifecycle refresh (compose broadcast or _expire_items) -- re-render only.
            return super().handle_update_request(event)

        results = self._user_results.setdefault(get_current_user_id(), {})
        status = event.context.get('status', 'pending')
        prior = results.get(task_id)

        # Never downgrade a terminal state with a late in_progress update.
        if prior and prior['status'] in ('success', 'timeout', 'failed', 'dismissed'):
            if status == 'in_progress':
                return super().handle_update_request(event)

        results[task_id] = {
            'status': status,
            'query': event.context.get('query', ''),
            'summary': event.context.get('summary') or event.context.get('error', ''),
            'expires_at_turn': (
                self.current_turn + (ERROR_TTL_TURNS if status in ('timeout', 'failed')
                                     else RESULT_TTL_TURNS)
                if status in ('success', 'timeout', 'failed') else None
            ),
        }
        super().handle_update_request(event)

    def _expire_items(self) -> bool:
        """Drop results past their TTL. True triggers a mid-turn re-render."""
        results = self._user_results.get(get_current_user_id(), {})
        expired = [
            tid for tid, r in results.items()
            if r['expires_at_turn'] is not None and self.current_turn > r['expires_at_turn']
        ]
        for tid in expired:
            del results[tid]
        return bool(expired)

    def _clear_all_state(self) -> None:
        """Called by WorkingMemory on segment collapse."""
        self._user_results.clear()

    def generate_content(self, context: Dict[str, Any]) -> str:
        results = self._user_results.get(get_current_user_id(), {})
        if not results:
            return ""                      # legitimately empty, not a failure

        lines = ['<my_task_results>']
        for task_id, r in results.items():
            lines.append(
                f'<task id="{html.escape(task_id[:8])}" status="{r["status"]}">'
                f'{html.escape(r["query"])}'
            )
            if r['summary']:
                lines.append(f'<summary>{html.escape(r["summary"])}</summary>')
            lines.append('</task>')
        lines.append('</my_task_results>')
        return "\n".join(lines)
```

Every line here is load-bearing against a real failure mode: per-user dict
(cross-user leak), context-before-super (prompt lags a cycle), terminal-state
guard (late overwatch update resurrects a finished task), `_expire_items` /
`_clear_all_state` (state outlives its segment), `html.escape` (untrusted query
text into prompt XML).

## Development Workflow

### 1. Define Purpose

Four decisions before writing code — record them in the module docstring:

```python
"""
Slot:      my_section          (unique; must be added to SECTION_LAYOUT)
Placement: notification        (system | conversation_prefix | post_history | notification)
Base:      StatefulTrinket     (turn-scoped state) | EventAwareTrinket (stateless / collapse-surviving)
Updates:   producer publishes UpdateTrinketEvent on task status change
Cache:     False               (True only for session-stable system-placement content)
"""
```

### 2. Identify the Producer

Who mutates your inputs, and how does it reach you? Almost always: a tool, service,
poller, or agent publishes `UpdateTrinketEvent` with `target_trinket="MyTrinket"`.
Turn-driven expiry is not an event you subscribe to — it is `StatefulTrinket._expire_items()`.
If nothing publishes at you and the data is already in a store, render from that store
per compose (`asyncactivity_trinket.py`, `reminder_manager.py`) and skip the producer entirely.

### 3. Design Content Format

One root XML element named after the section:

```xml
<my_section>
  <item id="a1b2c3d4" status="active">label</item>
</my_section>
```

### 4. Implement Core Pattern

```python
class MyTrinket(EventAwareTrinket):
    variable_name = "my_section"     # class attribute, not a method
    cache_policy = False

    def generate_content(self, context: Dict[str, Any]) -> str:
        return "<my_section>content</my_section>"
```

### 5. Add State and Lifecycle

Per-user dict keyed by `get_current_user_id()`. If state is turn-scoped, switch the
base to `StatefulTrinket` and implement `_expire_items()` / `_clear_all_state()`
instead of subscribing to events yourself.

### 6. Register (two files, both required)

**A. Instantiate** — `cns/integration/factory.py:_get_working_memory()`. Import
alongside the other trinket imports, then construct in the registration block:

```python
from working_memory.trinkets.my_trinket import MyTrinket
# ...
MyTrinket(event_bus, self._working_memory)   # self-registers via register_trinket()
```

Registration order in that block is deliberate and manual — there is no
auto-discovery. Construction alone registers the trinket; do not also assign it
to an attribute unless something needs a direct handle.

**B. Place** — `working_memory/composer.py:SECTION_LAYOUT`. Add `variable_name`
to exactly one placement list:

| Placement | Lands in | Typical content |
|---|---|---|
| `PLACEMENT_SYSTEM` | `cached_content` / `non_cached_content` (by `cache_policy`) | stable orientation: persona, LoRA, domaindoc, manifest, location |
| `PLACEMENT_CONVERSATION_PREFIX` | before conversation history | compaction brief |
| `PLACEMENT_POST_HISTORY` | after history | domaindoc body |
| `PLACEMENT_NOTIFICATION` | `<mira:hud>` assistant message that slides forward each turn | time, reminders, inbox, task results, surfaced memories |

A `variable_name` absent from `SECTION_LAYOUT` still renders — appended to
`system` at the end with a `logger.warning`. That is the "content appears in the
wrong place" failure, not a crash.

**C. Document** — add the `## Files` bullet to `working_memory/trinkets/AGENTS.md`
in the same commit (root map maintenance trigger table).

## Verification

No mocks, no test files — this is live-or-it-doesn't-count (root AGENTS.md).

| Tier | When | Do |
|---|---|---|
| 1 | Edit inside an existing trinket | Boot MIRA, run one turn, read the composed prompt and confirm your section rendered with real data |
| 2 | New trinket, new `variable_name`, or new `SECTION_LAYOUT` slot | Tier 1, plus: publish a real `UpdateTrinketEvent` from the producer path and confirm the render; confirm `_expire_items()` fires on turn advance; confirm `_clear_all_state()` on segment collapse; confirm the Valkey field `trinkets:{user_id}` → your `variable_name` is written and cleared |

Reading back the persisted state:

```python
working_memory.get_trinket_state("my_section")   # TrinketState | None from Valkey
```

A trinket whose content only appears on the compose broadcast but never survives
a Valkey round-trip is unverified — `get_trinket_state()` is the check.

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| `TypeError: must define 'variable_name'` | Implemented `_get_variable_name()` or forgot the attribute | Set the `variable_name` **class attribute** |
| Content never appears | Not instantiated in `factory.py:_get_working_memory()` | Add the construction (step 6A) |
| Content appears at the end of `system` + warning log | `variable_name` missing from `SECTION_LAYOUT` | Add it to the right placement list (step 6B) |
| Producer updates ignored | `target_trinket` name ≠ registered class name | Class name exactly; a mismatch is dropped with a warning |
| Prompt lags one turn behind producer | Called `super().handle_update_request()` before mutating state | Consume context first, then `super()` |
| Empty section after data existed | Treated a lifecycle refresh as new input and overwrote state | Missing action/data key ⇒ re-render only |
| One user's data in another's prompt | State on a bare instance attribute | Dict keyed by `get_current_user_id()` |
| Section persists past segment collapse | Plain `EventAwareTrinket` holding turn-scoped state | Inherit `StatefulTrinket`, implement `_clear_all_state()` |
| Another trinket's content vanished | Duplicate `variable_name` — shared Valkey hash field | Pick a unique slot name |
| Outage looks like "no data" | `try/except: return ""` around the fetch | Let infrastructure failures propagate; `core.py` isolates |
| Model acts on garbage in your section | Unescaped user-controlled text in XML | `_xml_attr_escape()` / `html.escape()` |

## Best Practices

1. **Return `""` only for legitimately empty** — never as an error path
2. **Propagate infrastructure failures** — `WorkingMemory._handle_update_trinket()` is the isolation boundary, not your `generate_content()`
3. **Render XML** — one root element named after the section
4. **Key all state by user id** — instances are process-global singletons
5. **Use `StatefulTrinket` for turn-scoped state** — never re-subscribe to `TurnCompletedEvent` / `SegmentCollapsedEvent`
6. **Escape user-controlled text** before interpolating into prompt XML
7. **`cache_policy=True` only for session-stable `system`-placement content**
8. **Read via the owning service** — no hand-written `WHERE user_id = ...` SQL
9. **Update `working_memory/trinkets/AGENTS.md`** in the same commit

## Reference Implementations

| Trinket | Base | Key patterns |
|---|---|---|
| `time_manager.py` | plain | Smallest possible trinket; stateless per-compose render |
| `manifest_trinket.py` | plain | Service query, user-tz date grouping, narrow `except` on formatting only |
| `location_trinket.py` | plain | Valkey-backed, `cache_policy=True` |
| `domaindoc_trinket.py` | plain | `cache_policy=True`, section-tree render, cross-user share reads |
| `asyncactivity_trinket.py` | plain | Reads SQLite per render (no in-memory state), `html.escape` |
| `reminder_manager.py` | plain | Invokes a tool directly (`ReminderTool.run()`) |
| `email_trinket.py` | stateful | Snapshot store, `_xml_attr_escape`, `_expire_items()` returns `False` |
| `forage_trinket.py` | stateful | Task-status state machine, terminal-state guard, per-iteration summary stacking |
| `whilethecatsaway_trinket.py` | stateful | Same machine, all results TTL-expire, no dismiss path |
| `peanutgallery_trinket.py` | stateful | TTL-in-turns expiry, constructor arg beyond the base two |
| `live_context_compaction_trinket.py` | stateful | `conversation_prefix` placement, `_clear_all_state()` deletes a Valkey artifact |

## Summary

Trinkets are **passive prompt composers** that:
- Own exactly one `variable_name` slot in the composed system prompt
- Render on the compose broadcast and on producer `UpdateTrinketEvent`s
- Keep all mutable state in dicts keyed by user id
- Return `""` when legitimately empty and raise on infrastructure failure
- Render escaped XML, placed by `SECTION_LAYOUT`

For user-facing operations, build a **tool** (`tools/HOW_TO_BUILD_A_TOOL.md`).
For autonomous multi-step work, build a **sidebar agent**
(`agents/HOW_TO_BUILD_AN_AGENT.md`).
