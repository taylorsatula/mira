# lt_memory/processing/ — Extraction pipeline

## Rules

- Memory extraction has exactly one execution path: `DirectExecutionStrategy.execute_extraction()` calls `LLMProvider.generate_response(model_config="batch")` synchronously per chunk and stores through `store_and_tend_extraction()`. Never reintroduce a deferred/remote batch submission layer.
- All LLM routing comes from `model_config='<route>'` on `generate_response()`. Never pass model, endpoint, or API key explicitly.
- `store_and_tend_extraction()` in `execution_strategy.py` is the single source of truth for memory storage — called by `DirectExecutionStrategy._process_and_store_memories()`. It stores memories with embeddings, persists LLM-extracted entities, builds typeless candidate hints, and notifies the integration curator via `LTMemoryFactory.on_memories_stored`. Never duplicate this logic.
- No typed links are written by extraction/storage code. Extraction-time `related_memory_ids` + bonds flow to the `MemoryCuratorAgent` as `CandidateRef` hints (discovery_signal `"extraction"`), not as edges. Relationship typing is the agent's job.
- `MemoryProcessor` has no side effects — pure data transformation. All DB writes happen in callers.
- Model-supplied temporal fields (`happens_at`/`expires_at` on `ExtractedMemory`) are the **user's local wall time**, never UTC as-instant: `MemoryProcessor.process_extraction_response()` resolves the timezone once per response via `validate_timezone(get_user_preferences().timezone)` (broad-except → `"UTC"` — extraction is a background durability path, not a fail-loud path) and parses each field through `_parse_model_temporal_field()` → `ensure_utc(parse_time_string(value, tz_name=...))`. Never hand a raw naive string to `ExtractedMemory`: a naive datetime binds to `timestamptz` under whatever `TimeZone` Postgres happens to have, and a `ValidationError` raised at construction aborts every memory in the segment batch (no per-memory catch upstream). An unparseable value logs the raw value + timezone + parser reason and the memory is stored without that field; text is never lost. No daylight-saving strictness here — that belongs to the model-facing scheduling tools.
- `ExtractionEngine` has no LLM calls — pure payload construction. LLM calls happen in strategies.

## Files

- `orchestrator.py` — Owns the segment extraction lifecycle: load messages from `ContinuumRepository`, build `ProcessingChunk`, run the direct strategy, mark `memories_extracted=true`. Two entry points: `submit_segment_extraction()` (per-segment) and `extract_unprocessed_segments()` (6-hour safety-net sweep).
- `execution_strategy.py` — Owns the `DirectExecutionStrategy` and the `create_execution_strategy()` factory, plus the module-level `store_and_tend_extraction()` / `_persist_llm_entities()` / `_build_candidate_hints()` helpers. `execute_extraction()` processes every chunk synchronously (LLM call on the `batch` route per chunk) and returns `str` (`direct_<uuid>`), or raises `ValueError` when no valid payload was built or any dependency fails.
- `extraction_engine.py` — Owns `ExtractionPayload` construction: prompt loading, UUID shortening/mapping via `format_memory_id()`, memory context retrieval from `ProcessingChunk.memory_context_snapshot`, and message formatting via `preprocess_content_blocks()`. File-local types: `ExtractionMessage`, `ExtractionPayload`.
- `memory_processor.py` — Owns LLM response parsing: JSON repair fallback, short→full UUID remapping, field validation/sanitization, model-supplied temporal-field parsing in the user's timezone (see Rules), and fuzzy+vector duplicate detection. File-local types: `DuplicateCheckResult`, `RawMemoryDict`.
- `consolidation_handler.py` — Owns memory merge execution: link bundle transfer (inbound, outbound, entity), outbound-link rewriting on source memories, and archival of old memories. Pure business logic — no routing decisions, no LLM calls. Called by `memory_tool.merge_memories` (agent-invoked).

## Wiring

**One strategy, built at init:**
`LTMemoryFactory` calls `create_execution_strategy()` once at startup, producing the single `DirectExecutionStrategy` as `ExtractionOrchestrator.execution_strategy`. Segment collapse calls `submit_segment_extraction()` and extraction completes inline — memories are available before the user's next conversation, which is why the path is synchronous.

**`store_and_tend_extraction()` call sites:**
- `DirectExecutionStrategy._process_and_store_memories()` — called inline after each `generate_response()` chunk, via the shared helper.

It passes `segment_id` so the integration curator work-item can record the source segment. The helper's `on_memories_stored` callback (late-registered by `SegmentCollapseHandler`) spawns the `MemoryCuratorAgent` in integration mode; `lt_memory` never imports from `agents/`.

**`chunk.segment_id` pipeline:**
`DirectExecutionStrategy` carries `str(chunk.segment_id)` from `ProcessingChunk` through storage so each memory gets `source_segment_id`. Required for segment-scoped memory cleanup on session resume.
