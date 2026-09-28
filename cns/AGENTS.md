# cns/ — Conversation orchestration layer: domain model, application services, persistence, API routing, and wiring

## Rules

- Layering: `core/` is pure domain (no I/O), `services/` holds application logic and calls repositories, `infrastructure/` owns all persistence, `api/` only validates and delegates. The no-I/O constraint inside `core/` is owned by `cns/core/AGENTS.md`; the SQL-in-repositories rule is owned by `cns/services/AGENTS.md`.
- All inter-component communication goes through the event bus (`cns/integration/event_bus.py`); all construction goes through the factory (`cns/integration/factory.py`). Both contracts — the four-category event taxonomy and the sole-construction-path rule — are owned by `cns/integration/AGENTS.md`.
- Every LLM call in this tree passes `model_config='<route>'` on `generate_response()` / `stream_events()` — never hardcode model names. The five fixed routes (`primary`, `fast`, `batch`, `assessment`, `other`) and stall behavior are owned by `clients/llm/AGENTS.md`.
- All LLM output parses `<mira:tag>` XML via regex with flexible whitespace. Use `utils.tag_parser.TagParser` for standard tags; follow the custom-regex patterns in `cns/services/subcortical.py` and `cns/services/assessment_extractor.py` for custom tags.
- Background threads spawned anywhere in this tree must propagate user context explicitly: `contextvars.copy_context().run(fn)`. This rule is owned here; the canonical pattern is `cns/services/tool_loop.py`, and the event-handler variant (async work inside a synchronous subscriber) is owned by `cns/integration/AGENTS.md`.

## Files

- `__init__.py` — Empty file, no docstring, no re-exports; nothing to initialize here. CNS graph assembly happens in `cns/integration/factory.py`.
- `core/` — Immutable domain model: Continuum aggregate, message value objects, domain events, message formatting, segment cache reconstruction. Map: `cns/core/AGENTS.md`.
- `services/` — Application services: turn orchestration, subcortical processing, summary generation, segment collapse, memory surfacing, user-model/Persona/portrait pipelines, pollers. Map: `cns/services/AGENTS.md`.
- `infrastructure/` — PostgreSQL repositories, Valkey continuum cache, `UnitOfWork`, segment sentinel lifecycle. Map: `cns/infrastructure/AGENTS.md`.
- `integration/` — Event bus and the CNS dependency-graph factory. Map: `cns/integration/AGENTS.md`.
- `api/` — FastAPI routers: chat (HTTP + WebSocket), actions, data, health, tool config, triggers, location, files, federation. Map: `cns/api/AGENTS.md`.

## Wiring

Request flow across the tree: `api/` handler → user context set → `services/orchestrator.process_message()` → `core/` domain objects mutate in memory → `infrastructure/UnitOfWork.commit()` persists (DB then Valkey). Events published by services are routed by `cns/integration/event_bus.py` to handlers registered in `cns/integration/factory.py`. The collapse chain and the memory-surfacing pipeline are owned by `cns/services/AGENTS.md` (Wiring); the sentinel lifecycle and commit ordering by `cns/infrastructure/AGENTS.md`; the factory's load-bearing initialization order (session cache before collapse handler before summarizer/barrier) by `cns/integration/AGENTS.md`.

Inbound edges (outside consumers of this tree):
- `main.py` builds the CNS graph via `create_cns_orchestrator()` (`cns/integration/factory.py`) and registers every `cns/api` router — the sole-construction-path rule is owned by `cns/integration/AGENTS.md`; `utils/power_on_self_test.py` is the other `create_cns_orchestrator()` caller.
- `agents/` binds to the bus and event classes, not to services: `agents/base.py` and `agents/sidebar.py` import `EventBus` (`cns/integration/event_bus.py`), and `agents/base.py`, `agents/implementations/forage_agent.py`, `agents/implementations/memory_curator_agent.py` import event classes from `cns/core/events.py`. Agent map: `agents/AGENTS.md`.
- `tools/implementations/` imports repositories and services directly: `tools/implementations/continuum_tool.py` → `cns/infrastructure/continuum_repository.py`, `tools/implementations/domaindoc_tool.py` → `cns/services/domaindoc_summary_service.py`, `tools/implementations/forage_tool.py` → `cns/core/events.py`. Tool map: `tools/implementations/AGENTS.md`.
- `working_memory/` imports `cns/core` types and events (`working_memory/core.py`, `working_memory/trinkets/base.py`); the trinket contracts hang off `UpdateTrinketEvent`/`ComposeSystemPromptEvent` published from `cns/services`. Map: `working_memory/AGENTS.md`.

Outbound edges: `cns/services` and `cns/integration` reach into `lt_memory/` (extraction submission via `lt_memory/factory.py`, proactive retrieval via `cns/services/memory_relevance_service.py` → `lt_memory.ProactiveService`) — owned by `cns/services/AGENTS.md` and `lt_memory/AGENTS.md`; `cns/api` delegates auth to `auth/api.py` — owned by `cns/api/AGENTS.md`. `cns/` also imports `tools/` (the factory builds `ToolRepository` via `tools/repo.py`; `cns/api/tool_config.py` uses `tools/registry.py`; `cns/api/actions.py` and `cns/api/federation.py` lazily import tool implementations) and, at the collapse chain's callback seam only, `agents/` (`cns/services/segment_collapse_handler.py` lazily imports `MemoryCuratorAgent` + `WorkItem` inside `on_memories_stored`) — the seam ownership lives in `cns/services/AGENTS.md`.
