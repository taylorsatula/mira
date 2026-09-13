# cns/ — Conversation orchestration layer: domain model, application services, persistence, API routing, and wiring

## Rules

- Layering: `core/` is pure domain (no I/O), `services/` holds application logic and calls repositories, `infrastructure/` owns all persistence, `api/` only validates and delegates. The no-I/O constraint inside `core/` is owned by `cns/core/AGENTS.md`; the SQL-in-repositories rule is owned by `cns/services/AGENTS.md`.
- All inter-component communication goes through the event bus (`cns/integration/event_bus.py`); all construction goes through the factory (`cns/integration/factory.py`). Both contracts — the four-category event taxonomy and the sole-construction-path rule — are owned by `cns/integration/AGENTS.md`.
- Every LLM call in this tree passes `model_config='<route>'` on `generate_response()` / `stream_events()` — never hardcode model names. The five fixed routes (`primary`, `fast`, `batch`, `assessment`, `other`) and stall behavior are owned by `clients/llm/AGENTS.md`.
- All LLM output parses `<mira:tag>` XML via regex with flexible whitespace. Use `utils.tag_parser.TagParser` for standard tags; follow the custom-regex patterns in `cns/services/subcortical.py` and `cns/services/assessment_extractor.py` for custom tags.
- Background threads spawned anywhere in this tree must propagate user context explicitly: `contextvars.copy_context().run(fn)`. The canonical pattern is `cns/services/peanutgallery_service.py`; the event-handler variant is owned by `cns/integration/AGENTS.md`.

## Files

- `__init__.py` — Empty file, no docstring, no re-exports; nothing to initialize here. CNS graph assembly happens in `cns/integration/factory.py`.
- `core/` — Immutable domain model: Continuum aggregate, message value objects, domain events, message formatting, segment cache reconstruction. Map: `cns/core/AGENTS.md`.
- `services/` — Application services: turn orchestration, subcortical processing, summary generation, segment collapse, memory surfacing, user-model/Persona/portrait pipelines, pollers. Map: `cns/services/AGENTS.md`.
- `infrastructure/` — PostgreSQL repositories, Valkey continuum cache, `UnitOfWork`, segment sentinel lifecycle. Map: `cns/infrastructure/AGENTS.md`.
- `integration/` — Event bus and the CNS dependency-graph factory. Map: `cns/integration/AGENTS.md`.
- `api/` — FastAPI routers: chat (HTTP + WebSocket), actions, data, health, tool config, triggers, location, files, federation. Map: `cns/api/AGENTS.md`.

## Wiring

Request flow across the tree: `api/` handler → user context set → `services/orchestrator.process_message()` → `core/` domain objects mutate in memory → `infrastructure/UnitOfWork.commit()` persists (DB then Valkey). Events published by services are routed by `cns/integration/event_bus.py` to handlers registered in `cns/integration/factory.py`. The collapse chain and the memory-surfacing pipeline are owned by `cns/services/AGENTS.md` (Wiring); the sentinel lifecycle and commit ordering by `cns/infrastructure/AGENTS.md`.
