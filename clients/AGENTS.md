# clients/ — External infrastructure clients and service adapters

## Rules

All secrets come from Vault via `get_database_url()`, `get_api_key()`, or `get_service_config(field)`. `VaultClient` is the only place env vars (`VAULT_ADDR`, `VAULT_ROLE_ID`, `VAULT_SECRET_ID`) are the primary config source — all other clients delegate to it.

Most clients are module-level singletons accessed through factory functions (`get_valkey()`, `get_hybrid_embeddings_provider()`, `get_lattice_client()`). `LLMProvider` follows the same pattern: `get_llm_provider()` is the single construction point for the process-wide shared instance. Direct `LLMProvider()` instantiation in application code is prohibited — call `get_llm_provider()` (construction failures propagate, fail-fast).

All LLM calls in application code go through `LLMProvider.generate_response()` or `LLMProvider.stream_events()`. Provider-specific code belongs in `clients/llm/dialects/`.

`ModelResolver` in `clients/llm/resolver.py` is the single source of truth for resolving MIRA's five fixed `model_configs` routes (`primary`, `fast`, `batch`, `assessment`, `other`) to dialect names, model IDs, endpoint URLs, Vault key names, max tokens, and effort. Callers pass `model_config=<route name>` with optional per-request `effort=`/`max_tokens=` overrides; they do not fetch provider API keys or route by provider enum.

Adding a new provider is a drop-a-file operation: write a new `clients/llm/dialects/<name>.py`, declare a `Dialect` subclass with `dialect_name` in `DialectName`, implement `from_selection`, and `DialectRegistry` picks it up at startup. The registry validates each candidate (fail-loud) — malformed dialects raise at boot, not at first request. Adding a new dialect_name value also requires extending the `DialectName` literal in `clients/llm/types.py` and the CHECK constraint in the schema.

`SQLiteClient` has no RLS. Manual `user_id` filtering in queries is the only isolation mechanism — it is not redundant.

`PostgresClient` pools are class-level, keyed by database name. `admin=True` uses a separate `{name}_admin` pool with BYPASSRLS role.

## Files

- `vault_client.py` — Vault AppRole auth and secret retrieval; the only permitted source of credentials for all other clients. `preload_secrets()` bulk-loads at startup and raises `RuntimeError` on any failure.
- `postgres_client.py` — Pooled raw-SQL client with automatic RLS context (`SET app.current_user_id`) on connection checkout. All `execute_*` methods are monkey-patched by `utils/perf.py` when `mira.perf` logger is at INFO or DEBUG.
- `valkey_client.py` — Caching, sessions, and rate limiting. Exposes sync, async, and binary clients from a single module-level pool. Pings Valkey on construction — fails immediately if unreachable.
- `sqlite_client.py` — Per-user tool data storage. No pooling; fresh connection per request. Factory `get_sqlite_client(db_path, user_id)` is cached per `user_id:path`.
- `llm_provider.py` — Universal LLM entry point. Builds neutral `Request` objects, resolves model selection through `ModelResolver`, instantiates dialects via `DialectRegistry`, and delegates one provider request's fail-loud completion policy (response timeout and stall detection, no model fallback) to `LLMLifecycle`. Returns `clients.llm.types.Result`. Public methods accept `effort=`/`thinking_tokens=` convenience kwargs OR `thinking=ThinkingConfig(...)` directly (mixing the two raises `ValueError`). Exposes the module-level `get_llm_provider()` accessor (thread-safe, lazy) as the only permitted construction path.
- `llm/` — Provider-neutral LLM layer. `types.py` owns strict `Request`, `Result`, `ThinkingConfig`, `Usage`, `ToolDefinition`, `ToolCall`, `ToolResult`, and `ReasoningArtifact` contracts plus the `DialectName` literal; `thinking.py` owns the `TranslationNote` frozen dataclass and the Anthropic-specific `uses_adaptive_thinking()` model classifier; `events.py` owns provider-neutral streaming event dataclasses; `lifecycle.py` owns one live provider request policy; `resolver.py` owns model selection (`ModelSelection` carries `dialect_name`); `dialect_registry.py` discovers concrete `Dialect` subclasses by walking the `dialects/` package at first access; `dialects/` owns provider serialization/parsing.
- `llm/dialects/` — Four concrete dialect modules plus shared infrastructure. `base.py` declares the `Dialect` ABC (`native_thinking_fields`, `from_selection`, `_log_translation`). `openai_chat_base.py` extracts the OpenAI Chat Completions transport (message conversion, streaming SSE, full JSON-Schema validation of provider tool arguments, tool parsing, usage parsing), writes raw streaming JSON chunks to the opt-in LLM tap before normalization, coalesces consecutive OpenRouter `reasoning.text` deltas before round-trip persistence, and exposes hook methods for dialect-specific thinking serialization, reasoning extraction, cache fields, and foreign-field stripping. `anthropic.py` (Anthropic native, `("effort", "budget")`), `openai.py` (top-level `reasoning_effort`, `("effort",)`), `openrouter.py` (nested `reasoning` block + `reasoning_details`, `("effort", "budget")`), and `groq.py` (`reasoning_effort` + `reasoning_format=parsed`, `("effort",)`). Each dialect translates non-native thinking knobs and emits a `TranslationNote` at WARNING when information is lost.
- `hybrid_embeddings_provider.py` — Local asymmetric embeddings (`mdbr-leaf-ir-asym`, 768-dim) with Valkey-backed cache. `encode_realtime()` for queries; `encode_deep()` for documents.
- `lattice_client.py` — Thin HTTP client for Lattice federation. Only consumed by `pager_tool`. Not exported from `__init__.py`.
- `__init__.py` — Re-exports `HybridEmbeddingsProvider`, `get_hybrid_embeddings_provider`, `LLMProvider`, `PostgresClient`, `SQLiteClient`, `ValkeyClient`, `get_valkey`, `get_valkey_client`, and selected vault functions. `LatticeClient` is excluded — import directly.

## Wiring

Local tool execution is orchestrator-owned. `cns/services/tool_loop.py` propagates contextvars to `ThreadPoolExecutor` workers via `contextvars.copy_context()`. Any new threaded tool execution path must do the same for RLS enforcement to hold.

Cost visibility is a slim `LLMLifecycle` hook: each completed result's usage is recorded into `utils.cost_accumulator`, keyed by `model_configs` route name against the `usage_pricing` table. The accumulator is inert unless a request handler called `start()`. Billing machinery is out of scope for the OSS build.
