# lt_memory/processing/ — the extraction pipeline from a collapsed segment to stored memories, consolidation, and the entity-linking helpers both store paths share

## Files

- `__init__.py` — re-export surface only; in-repo callers import the submodules directly.
- `orchestrator.py` — the extraction lifecycle and the retry sweep that re-enters it, with the segment's failure budget. Open it before changing when a segment is walked, marked, retried, or given up on.
- `execution_strategy.py` — the extraction execution path and the storage path it writes through. Open it before changing what happens to memories between the model's response and the database.
- `extraction_engine.py` — payload construction: what the model is shown, and the ID maps its answer is read back through. Open it before changing what the model is shown.
- `memory_processor.py` — the extraction response parser: what the model's reply becomes, and what is dropped on the way. Open it before changing what output is accepted or what counts as a duplicate.
- `consolidation_handler.py` — merge execution: what a merged memory inherits from its sources, links included. Open it before changing what a merge carries forward or discards.

## Wiring

- Extraction: `orchestrator.submit_segment_extraction` → `extraction_engine.build_extraction_payload` → `execution_strategy.execute_extraction` → `memory_processor.process_extraction_response` → `execution_strategy.store_and_tend_extraction` → the `memories_extracted` mark. The mark follows the whole store call, never precedes it: `store_memories_with_embeddings` is a plain insert, so a raise between the two leaves the segment unmarked and the next sweep inserts the same memories a second time.
- Submit side: `cns/services/segment_collapse_handler.py` calls `submit_segment_extraction` from `_trigger_downstream_processing`; a raise is caught at that call site, logs CRITICAL, flags `downstream_failed` on the already-saved sentinel, and leaves the segment to the registered retry sweep `extract_unprocessed_segments` — the collapse itself stands. Counterpart: `cns/services/AGENTS.md`.
- Consolidation: reached only through `tools/implementations/memory_tool.py:_merge_memories` (the `merge_memories` operation), which owns the 2+-distinct-non-archived precondition and stamps `update_last_tended` on the new id once `execute_consolidation` returns. Counterpart: `tools/implementations/AGENTS.md`.
- Construction and the curator seam: `lt_memory/factory.py` builds every class here and holds `on_memories_stored`, which `store_and_tend_extraction` reads from the factory and calls None-safe. Counterpart: `lt_memory/AGENTS.md`.
- Failure budget: the sweep reads `find_failed_extraction_segments` and `EXTRACTION_MAX_CONTENT_FAILURES` from the continuum repository and writes its metadata marks through `get_user_db_client`. That query drops a segment once `extraction_content_failures` reaches the budget or `extraction_abandoned` is true; `orchestrator.py` is the only writer of both marks. Counterpart: `cns/infrastructure/AGENTS.md`.
- Prompt files: the `memory_extraction_*` templates load through `load_prompt` and raise at construction when absent. Counterpart: `config/prompts/AGENTS.md`.

## Rules

- `persist_llm_entities` is the one entity-linking route, and `get_or_create_entity`/`link_memory_to_entity` have exactly one call site in the repo behind it. A second linking path forks entity rows under names this path already resolved, and the memory/entity table splits.
- `sanitize_entity_list` in `memory_processor.py` is the single shape contract for both entity sources — this directory's extraction responses and the pending-drain entity pass in `cns/services/segment_collapse_handler.py`, which imports it directly. It is what keeps model noise from spending a drain item's content-failure budget, so a loosened sanitizer reintroduces spend on LLM noise in a path this directory does not own. Counterpart: `cns/services/AGENTS.md`.
- The extraction path writes no typed link. `related_memory_ids` and their bonds leave as `CandidateRef` hints tagged `discovery_signal: "extraction"`, merged with `LinkingService.find_candidate_hints`. `db.create_links` has two callers, found by `rg 'create_links'`: the manual-memory supersedes link in `cns/services/segment_collapse_handler.py` and the memory tool's link write. An edge written from extraction sidesteps the replace semantics and the archived-row guard, so the two sides of the relationship disagree about whether it exists.
- `consolidation_handler.execute_consolidation` is the exception to that: it rewrites `inbound_links`/`outbound_links`/`entity_links` directly through `db.update_memory` on both sides of every affected memory, bypassing `create_links`. That is why it carries its own dedupe of inbound/outbound entries — the replace semantics it would have inherited are not in play.
- `referenced_memories` in message metadata must hold full UUIDs: `_build_memory_context` calls `UUID(ref_id)` with no guard. A short id raises `ValueError` there. On the collapse path the handler catches it, flags `downstream_failed`, and the segment is retried by the sweep; in the sweep the same `ValueError` classifies as content-caused, so three sweeps reach `extraction_abandoned` and the segment is never extracted. Any new writer resolves through `utils/tag_parser.py` first.
- Temporal parsing stays in `process_extraction_response` and is never delegated to Pydantic field coercion — every producer that builds an `ExtractedMemory` inherits that, including ones outside this directory.
- Any new raise site in this pipeline must be classified by `_is_content_caused_extraction_failure`. Types it does not recognize fall through to `RuntimeError`/`ValueError` and spend the segment's content budget, and a spent budget writes `extraction_abandoned`, which no sweep revisits. A new content-shaped error raised as a plain `ValueError` therefore erases the segment after three attempts.

Elsewhere: this map does not own what happens to memories after the curator receives them, nor the collapse that submits the segment — both are other directories' mechanisms, so the curator's consumption belongs to `agents/` and the submit side is one of the edges in `## Wiring`. The curator's use of `new_memories` and `candidate_hints` is owned by `agents/implementations/AGENTS.md`; the producer side this directory fills is keyed by each new memory's full UUID string, with items tagged `discovery_signal: "extraction"`, and `CandidateRef.memory_id` on the consuming side is a short id. The short-ID format is `utils/tag_parser.py`. Every other destination this directory reaches is an edge — see `## Wiring` above.

## One segment's state across retries

The path that decides most of the Rules: a segment that fails extraction and is offered to the sweep again.

1. Collapse finishes and `submit_segment_extraction` runs. The boundary row is missing → `RuntimeError`; or the segment has message rows but nothing extractable → the boundary is marked extracted and the segment is never revisited.
2. The LLM response is degenerate (empty, whitespace-only, unrepairable JSON) → `process_extraction_response` raises `LLMResponseFormatError` before the mark is written, so the segment keeps its unmarked state and the next sweep retries it.
3. `extract_unprocessed_segments` finds the segment through `find_failed_extraction_segments`, bumps `extraction_attempts` (observability only), and calls back in.
4. On failure it asks `_is_content_caused_extraction_failure`. `LLMResponseFormatError` and `ProviderError` are excluded ahead of the classes they subclass, so step 2 and every provider outage leave `extraction_content_failures` untouched and the segment is retried indefinitely.
5. A failure the classifier does count — missing boundary row, no-payload chunk — increments `extraction_content_failures`. At `EXTRACTION_MAX_CONTENT_FAILURES` the sweep writes `extraction_abandoned`, and `find_failed_extraction_segments` stops returning the segment.
6. The two counters are separate on purpose: the attempt counter answers "how often was this tried", the content counter answers "how often did this segment's own data fail". Collapsing them lets three provider blips consume a data-bearing budget and silence a recoverable segment.
