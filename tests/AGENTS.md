# tests/ — Greenfield contract verification

- `test_greenfield_schema.py` — Static fresh-schema, auth-column, legacy-absence, fixed-model, RLS, and least-privilege assertions.
- `test_model_routing.py` — Fixed route-name, row-selected dialect/defaults, caller-override, and fail-loud lifecycle tests; scans retained call sites for only `primary`, `fast`, or `batch`.
- `test_openrouter_reasoning_roundtrip.py` — Streaming `reasoning_details` coalescing and opaque signature round-trip coverage for OpenRouter tool continuations.
- `test_openai_tool_schema_validation.py` — Full JSON-Schema validation of provider-returned tool arguments on the OpenAI Chat transport.
- `test_orchestrator_tool_loop.py` — Orchestrator tool-loop contracts, including circuit-breaker finalization with a terminal turn.
- `test_cognitive_feature_bypasses.py` — Strict environment switches, dependency-graph omission for subcortical/Peanut Gallery, empty no-memory fastpath, and enabled-failure propagation.
- `test_direct_extraction.py` — Proves memory extraction calls the direct provider with `model_config="batch"` and no remote batch coordinator.
- `test_persona_service.py` — Persona evidence parsing, seven-use-day retry/publication, preview TTL, accept/decline, rollback, and cache invalidation.
- `test_auth_graft.py` — Hosted session hashing, logout-others, member provisioning compensation, development-session creation/repair and dev-only routing (mode-gated), FastAPI surface mounting with `single`-mode identity, WebSocket typed user context from a cookie session, single-mode bootstrap contracts, and explicit-RLS identity helpers.
- `test_history_cursor.py` — Opaque history cursor validation, tied-timestamp ordering, chronological server pages, concurrent-insert stability, and offset rejection.
- `test_ordered_turn_persistence.py` — Stable client/turn/segment IDs, provider-step ordering, monotonic message batches, partial Halt persistence, transient tool-loader continuation cleanup, and post-commit event behavior.
- `test_viewcard_content.py` — Typed immutable cards plus accepted/rejected text, Markdown, HTML, CSS, URL, data-image, and inline-SVG content policy.
- `test_web_frontend_protocol.py` — WebSocket frame validation, busy/malformed input behavior, sole reader/writer ownership, terminal turn frames, and module/CSP cutover contracts.
- `test_tool_config_resolution.py` — Per-user Tool Settings override resolution over global tool defaults.
- `test_sidebar_agent_tool_loop.py` — SidebarAgent invalid provider tool-call recovery: schema-invalid tool calls become repair feedback and never execute or poison continuation history.

Tests are unit/static by default. The schema must additionally be loaded into a disposable PostgreSQL database for release verification.

Infrastructure-dependent tests are auto-classified as `integration` (fixture closure + per-file table in `tests/fixtures/infra.py`): `pytest tests/` runs the unit set and reports the rest SKIPPED without touching Vault/Postgres/Valkey; run them with `pytest --integration` or with `VAULT_ADDR`/`VAULT_ROLE_ID`/`VAULT_SECRET_ID` exported. New live-infra tests need no conftest edits — either request an existing infra fixture or add `pytestmark = pytest.mark.integration`.
