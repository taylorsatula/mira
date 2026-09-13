# agents/implementations/ — concrete SidebarAgent implementations

## Rules

- Every class here sets `model_config_name` to an existing `model_configs` route — batch research agents use `"batch"` (`forage_agent.py`, `whilethecatsaway_agent.py`); `MemoryCuratorAgent` uses `"primary"`. Never invent a route name.
- Completion trinket publishing goes through the `_get_completion_trinket()` / `_build_completion_context()` overrides (`agents/base.py` hooks), one trinket per agent: `ForageTrinket`, `WhileTheCatsAwayTrinket`, `MemoryCuratorTrinket`. Do not publish terminal results by any other path.
- All three classes set `inherit_base_prompt = False` — each loads a self-contained rubric prompt (including its own loop/complete_task framing) from `config/prompts/agents/` via `load_agent_prompt()` in `get_agent_prompt()`. Adding an agent without a prompt file fails at runtime, not import.
- `MemoryCuratorAgent.get_agent_prompt()` and `build_initial_message()` raise `ValueError` on any `work_item.context['mode']` other than `'integration'` or `'floor'` — an unknown mode is a trigger bug and must fail loud.
- `MemoryCuratorAgent` never creates memories: `tool_schema_overrides = {"memory_tool": CURATOR_MEMORY_SCHEMA}` (`CURATOR_MEMORY_SCHEMA` in `tools/implementations/memory_tool.py`) replaces the memory_tool schema handed to the LLM with one excluding `create_memory`. Do not restore the full schema.
- `MemoryCuratorAgent.on_completion()` stamps `last_tended_at` (via `LTMemoryDB.update_last_tended`) only when `status == 'success'` — a failed/timeout run leaves memories un-tended so the floor trigger re-samples them later. Stamp failure is logged and swallowed; it must never mask a successful run.
- `ForageAgent.on_overwatch_update()` swallows publish failures at `debug` level — overwatch is observability, not results; do not let it raise into the agent loop.

## Files

- `__init__.py` — docstring only, no re-exports; agents are imported directly by spawn paths (`agents/sidebar.py` dispatcher via `trigger.agent_class`, direct callers like `ForageTool`).
- `forage_agent.py` — `ForageAgent`: background research, 20-iteration cap, on `batch`. Owns the per-iteration progress bar: `_iteration_status()` injects a Unicode bar (width `_PROGRESS_BAR_WIDTH = 20`) plus a wind-down directive into the system prompt each turn. `build_initial_message()` branches on `work_item.context['previous_result']` for refinement runs. `on_overwatch_update()` publishes `UpdateTrinketEvent` per-iteration summaries to `ForageTrinket` (`continuum_id='sidebar'`). Tools: `continuum_tool`, `memory_tool`, `web_tool`. Consumers: `tools/implementations/forage_tool.py` (direct spawn path), dispatcher triggers.
- `whilethecatsaway_agent.py` — `WhileTheCatsAwayAgent`: open-ended curiosity research, 25-iteration cap, on `batch`. Minimal overrides — no sentry, no overwatch, no `_iteration_status`. Reads `work_item.context['topic']`/`['context']`. Tools: `web_tool`, `memory_tool`, `continuum_tool`. Consumers: dispatcher.
- `memory_curator_agent.py` — `MemoryCuratorAgent`: memory-graph curation in two modes (`work_item.context['mode']`), 8-iteration ceiling, on `primary`, only `memory_tool` in `available_tools`. Owns the trigger→agent context contract TypedDicts (see Mode Contract below) and is the sole link typer. `sanitize_untrusted_input = False`. Consumers: `agents/triggers/memory_floor_trigger.py` (floor mode), segment-collapse spawn hook (integration mode).

## Wiring

- Spawn paths arrive from outside this directory: the dispatcher (via a trigger's `agent_class`) or direct invocation. Triggers populate `work_item.context` to the TypedDict shapes owned here; `build_initial_message()` renders it, `on_completion()` reads memory IDs back out.
- Forage overwatch publishes `UpdateTrinketEvent` on `continuum_id='sidebar'` targeting `ForageTrinket`; completion publishes to the same trinket via the base hook. Late-arriving overwatch updates must not downgrade terminal trinket states (invariant owned by `working_memory/trinkets/AGENTS.md`).

## Mode Contract (MemoryCuratorAgent)

`work_item.context['mode']` selects one of two TypedDict shapes, both defined in `memory_curator_agent.py`:

**Integration mode** (`IntegrationContext`): spawned at the segment-collapse hook when new memories are extracted. Fields: `segment_id`, `new_memories` (list of `NewMemory`), `candidate_hints` (dict keyed by full-UUID string → list of `CandidateRef`). The agent decides MERGE, LINK (with exact link_type), or STAND ALONE for each new memory. `CandidateRef` carries a SHORT id (`mem_XXXXXXXX`) because it feeds the LLM's tool calls; `NewMemory.memory_id` is a FULL UUID because `on_completion` needs it for the `last_tended_at` stamp (short IDs are an irreversible prefix). `build_initial_message()` formats full UUIDs to short display form via `_short()` → `utils.tag_parser.format_memory_id`.

**Floor mode** (`FloorContext`): dispatcher-driven via `agents/triggers/memory_floor_trigger.py`. Fields: `memories` (list of `FloorMemory`, each with FULL-UUID `memory_id`, `text`, `importance_score`). The agent decides ARCHIVE or SALVAGE per memory — the harsher mode, since all sampled items already score low.

ID-format asymmetry is load-bearing: discovery-side structures may use either form, but anything stamped in `on_completion` must be a full UUID; `_tended_uuids_for_status()` parses and skips unparseable IDs with a warning.
