# WP0 triage manifest — recovered test suite

Work package WP0 of the crm_mira → mira-OSS 2.0 backport
(`.pi/plans/20260905-172000-crm-mira-backport-bisect.md` §3, §12, open item O-19).

WP0's gate is "the suite runs". That means **pytest can enumerate the tests**, not that they pass.
Nothing below was repaired to green, and nothing in the programme should read a red suite as WP0
having failed: O-19 makes this a re-baseline, not a repair.

## Totals

| Measure | Value |
|---|---|
| Files recovered into `tests/` | **73** |
| — test files (`test_*.py`) | 53 |
| — support files (`__init__.py` ×8, `conftest.py` ×2, fixture modules ×7, `realistic_conversation.json`, `pager_schema.sql`, `AGENTS.md`) | 20 |
| Pre-existing tracked test files, untouched | 2 (`tests/utils/test_pinned_http.py`, `tests/utils/test_url_safety.py`) |
| Test files now present in `tests/` | **55** |
| Files modified to unblock collection | 2 (`tests/fixtures/core.py`, `tests/clients/test_postgres_client.py`) |
| Tests enumerated | **689** |

`python3 -m pytest tests/ --collect-only -q` → **exit 2**, `689 tests collected, 18 errors`.
With `--continue-on-collection-errors` → **exit 1**, same counts. A non-zero collection exit is the
expected end state of WP0: 13 of the 18 errors are tests whose subject code arrives in WP2–WP5, and
the remaining 5 are 1.x subsystems that `ee44b18` deleted.

## Status counts (55 test files)

| Status | Count |
|---|---|
| `COLLECTS` | 32 |
| `COLLECT-ERROR` | 18 |
| `NEEDS-REPAIR` | 3 |
| `ASSERTS-REMOVED-CONTRACT` | 2 |

---

## A. crm_mira recoveries, grouped by work package

This is the manifest's useful output: for each phase, which characterisation contract is already
waiting for it. 15 test files from crm_mira — 13 generic files from `95254a5^` (the tree immediately
before crm deleted its suite) and 2 still present at `crm_mira/crm_mira` HEAD.

| File | Origin | Status | Blocks / validates | Note |
|---|---|---|---|---|
| `tests/test_model_routing.py` | crm `95254a5^` | `COLLECT-ERROR` | **WP2** `model_configs`, five-route contract | `ImportError: ModelConfig` from `utils.user_context` — the type WP2 introduces. Imports `clients.llm.{dialects.base,lifecycle,resolver,types}` which do exist. |
| `tests/test_greenfield_schema.py` | crm `95254a5^` | `NEEDS-REPAIR` | **WP-S** the greenfield DDL deliverable itself | Collects (9 tests) only because `deploy/mira_service_schema.sql` exists; it reads that file at import. **See the regex trap in §D — satisfying this test as written is not the same as passing it.** |
| `tests/test_direct_extraction.py` | crm `95254a5^` | `COLLECT-ERROR` | **WP2 / batch removal** direct-execution path | `ImportError: DirectExecutionStrategy` from `lt_memory.processing.execution_strategy`; the module exists, the strategy class does not. |
| `tests/test_sidebar_agent_tool_loop.py` | crm `95254a5^` | `COLLECTS` | **WP1** items 5, 11 (`agents/base.py` tool loop) | 1 test. Imports `agents.base.SidebarAgent`, `_init_trace`, `agents.sidebar.WorkItem` — all present at HEAD. |
| `tests/test_persona_service.py` | crm `95254a5^` | `COLLECT-ERROR` | **WP5** Persona as second system | `ModuleNotFoundError: cns.infrastructure.persona_repository` (and `cns.services.persona_service`) — modules WP5 ports. |
| `tests/test_openrouter_reasoning_roundtrip.py` | crm `95254a5^` | `COLLECTS` | **WP1** item 12 reasoning-delta coalescing | 6 tests. Recovered copy carries **both** `f099b5f` (coalesce consecutive `reasoning.text` deltas) and `922b1e5` (reject invalid provider tool-call arguments); both verified as ancestors of `95254a5^`. |
| `tests/test_viewcard_content.py` | crm `95254a5^` | `COLLECT-ERROR` | **D-4** deferred `viewcard_content.py` | `ModuleNotFoundError: tools.implementations.viewcard_content`; `viewcard_tool` exists, the content normaliser does not. |
| `tests/test_auth_graft.py` | crm `95254a5^` | `NEEDS-REPAIR` | **WP3** multi-user auth | 16 tests collect. Only **part** of the file is generic: ~7 tests exercise CRM-only surface — `import billing`, `from cns.api import demo`, crm workspace tokens, crm page visibility — which §8.1 and §0.2 omit from OSS. WP3 should keep session hashing, `logout_others`, cleanup ordering, compensation-on-failure, the removed-OSS-auth contracts and the explicit-RLS-identity checks, and excise the rest. |
| `tests/test_ordered_turn_persistence.py` | crm `95254a5^` | `COLLECT-ERROR` | **WP4** ordered turn persistence | `ImportError: AssistantStep` from `cns.services.orchestrator`. Note it imports `CircuitBreaker`/`ToolLoopExecutor` from `cns.services.tool_loop`, which does exist — crm's placement is the proposal WP4 adopts or renames. |
| `tests/test_web_frontend_protocol.py` | crm `95254a5^` | `COLLECTS` | **WP4** frontend protocol compat — the D2 risk | 7 tests. The highest-value file for the D2 decision: it pins the strict frame contract the ~150-line frontend patch must satisfy. |
| `tests/test_history_cursor.py` | crm `95254a5^` | `COLLECTS` | **WP4** keyset pagination (D-2) | 6 tests against `ContinuumRepository`; enumerates today, so it can gate the cursor-only `get_history` the moment WP4 lands. |
| `tests/test_cognitive_feature_bypasses.py` | crm `95254a5^` | `COLLECTS` | **WP1** item 16 feature flags (93 L) | 6 tests. `MIRA_PERSONA_ENABLED` and the flag-omission graph WP1 generalises. |
| `tests/test_tool_config_resolution.py` | crm `95254a5^` | `COLLECTS` | **WP1** per-user tool config (22 L) | 1 test. Registers `tools.implementations.web_tool` for its configuration side effect. |
| `tests/test_openai_tool_schema_validation.py` | crm HEAD | `COLLECTS` | **WP1** tool-call robustness | 2 tests against `clients.llm.dialects.openrouter`. |
| `tests/test_orchestrator_tool_loop.py` | crm HEAD | `COLLECTS` | **WP1** tool-call robustness | 3 tests. Imports `ContinuumOrchestrator`, `TurnAccumulator` — both present at HEAD. |
| `tests/AGENTS.md` | crm `95254a5^` | — | orientation | Directory doc. Carries crm's own terminology; §0.2 says crm names are not authoritative, so treat as context, not spec. |

## B. mira-OSS pre-`ee44b18` suite

38 test files recovered verbatim from `ee44b18^`. These characterise **1.x** behaviour. Under §0.1
some assert contracts 2.0 removes; where the subject module is already gone the file is a
`COLLECT-ERROR`, and the note names the deleted subsystem.

| File | Origin | Status | Blocks / validates | Note |
|---|---|---|---|---|
| `tests/api/test_actions_endpoint.py` | mira `ee44b18^` | `COLLECTS` | 1.x REST surface, actions | 32 tests. |
| `tests/api/test_chat_endpoint.py` | mira `ee44b18^` | `COLLECTS` | 1.x chat endpoint | 18 tests. No WebSocket frame assertions — unaffected by D2. |
| `tests/api/test_data_endpoint.py` | mira `ee44b18^` | `ASSERTS-REMOVED-CONTRACT` | 1.x offset history API | **Removed thing: offset/search pagination on `?type=history`.** `test_history_respects_offset_parameter`, `test_history_supports_search_query` and the `"offset" in pagination` assertion are exactly what D-2 deletes — crm's `_get_history()` *raises* on `offset`/`search`. Rest of file (memories, other types) is unaffected. |
| `tests/api/test_health_endpoint.py` | mira `ee44b18^` | `COLLECTS` | health endpoint | 10 tests. |
| `tests/api/test_websocket_endpoint.py` | mira `ee44b18^` | `ASSERTS-REMOVED-CONTRACT` | 1.x WS frame vocabulary | **Removed thing: the pre-D2 protocol.** Asserts `type` in {`text`, `complete`, `pong`} and a `ping` handler; D2's strict validator answers `assistant_delta` / `turn_complete` / `halt` and rejects these frames. Superseded by `test_web_frontend_protocol.py`. |
| `tests/clients/embeddings/test_bge_reranker.py` | mira `ee44b18^` | `COLLECT-ERROR` | 1.x embeddings | `No module named 'clients.embeddings'` — package deleted in `ee44b18`, absent at HEAD. |
| `tests/clients/embeddings/test_openai_embeddings.py` | mira `ee44b18^` | `COLLECT-ERROR` | 1.x embeddings | Same missing package. |
| `tests/clients/test_generic_streaming.py` | mira `ee44b18^` | `COLLECT-ERROR` | 1.x streaming events | `No module named 'cns.core.stream_events'` — deleted in `ee44b18`. |
| `tests/clients/test_hybrid_embeddings_provider.py` | mira `ee44b18^` | `COLLECTS` | hybrid embeddings provider | 28 tests. |
| `tests/clients/test_llm_provider.py` | mira `ee44b18^` | `COLLECT-ERROR` | 1.x `clients.llm_provider` | `ImportError: CircuitBreaker` from `clients.llm_provider`; that class now lives at `cns.services.tool_loop.CircuitBreaker`. A moved-symbol repoint would make it collect, but the whole module is restructured by WP2, so repairing it now is likely wasted. |
| `tests/clients/test_postgres_client.py` | mira `ee44b18^` | `NEEDS-REPAIR` | Postgres client, RLS isolation | Collects (33 tests) after the WP0 syntax fix (§C.3). `test_different_users_see_isolated_data_via_rls` now gets `auth_db = None` and will fail at run time — it needs the `AuthDatabase` call WP3 restores. |
| `tests/clients/test_sqlite_client.py` | mira `ee44b18^` | `COLLECTS` | SQLite client | 33 tests. |
| `tests/clients/test_valkey_client.py` | mira `ee44b18^` | `COLLECT-ERROR` | 1.x Valkey TTL API | `ImportError: create_ttl_persistence_setup` — zero definitions anywhere outside `tests/`. |
| `tests/clients/test_vault_client.py` | mira `ee44b18^` | `COLLECT-ERROR` | 1.x Vault DB-credential path | `ImportError: get_database_credentials` — removed; WP-S's externally-provisioned roles replace the mechanism. |
| `tests/cns/services/test_fingerprint_generator.py` | mira `ee44b18^` | `COLLECTS` | fingerprint generation | 16 tests. |
| `tests/cns/services/test_memory_retention.py` | mira `ee44b18^` | `COLLECTS` | memory retention | 15 tests. |
| `tests/cns/services/test_segment_collapse_handler.py` | mira `ee44b18^` | `COLLECT-ERROR` | 1.x segment collapse | `ImportError: add_tools_to_segment` from `cns.services.segment_helpers`. WP5 hand-edits this handler (retain `_process_feedback_loop`, `_init_feedback_loop`, `_invalidate_lora_trinket_cache`) — do not take crm's file. |
| `tests/cns/services/test_segment_helpers.py` | mira `ee44b18^` | `COLLECT-ERROR` | 1.x segment helpers | Same missing `add_tools_to_segment`. |
| `tests/cns/services/test_segment_timespan.py` | mira `ee44b18^` | `COLLECTS` | segment timespan | 3 tests. |
| `tests/lt_memory/test_db_access.py` | mira `ee44b18^` | `COLLECT-ERROR` | 1.x batch coordinator | `ImportError: PostProcessingBatch` from `lt_memory.models` — the Anthropic Batch API surface §0.1/§8.4 removes. |
| `tests/lt_memory/test_models.py` | mira `ee44b18^` | `COLLECT-ERROR` | 1.x batch models | Same missing `PostProcessingBatch`. |
| `tests/lt_memory/test_vector_ops.py` | mira `ee44b18^` | `COLLECTS` | vector operations | 33 tests. |
| `tests/tools/implementations/test_contacts_tool.py` | mira `ee44b18^` | `COLLECTS` | contacts tool | 46 tests. Uses the `schema_files` marker (unregistered; warns) pointing at `tools/implementations/schemas/contacts_tool.sql`. |
| `tests/tools/implementations/test_continuum_tool.py` | mira `ee44b18^` | `COLLECTS` | continuum tool | 51 tests. |
| `tests/tools/implementations/test_domaindoc_tool.py` | mira `ee44b18^` | `COLLECTS` | domaindoc tool | 38 tests. |
| `tests/tools/implementations/test_web_tool.py` | mira `ee44b18^` | `COLLECTS` | web tool | 57 tests. **Invariant:** `tools/implementations/web_tool.py`, `utils/url_safety.py`, `utils/http_client.py` stay at main's version — they carry `e401d59`; this file plus the two pre-existing tests are the net for that. |
| `tests/tools/test_gated_tools.py` | mira `ee44b18^` | `COLLECTS` | tool gating | 16 tests. Adjacent to WP1 item 16 feature flags. |
| `tests/tools/test_getcontext_tool.py` | mira `ee44b18^` | `COLLECT-ERROR` | 1.x getcontext tool | `No module named 'tools.implementations.getcontext_tool'` — deleted in `ee44b18`. |
| `tests/utils/test_document_processing.py` | mira `ee44b18^` | `COLLECTS` | document processing | 28 tests. |
| `tests/utils/test_generic_openai_client.py` | mira `ee44b18^` | `COLLECT-ERROR` | 1.x generic OpenAI client | `No module named 'utils.generic_openai_client'` — deleted in `ee44b18`. |
| `tests/utils/test_image_compression.py` | mira `ee44b18^` | `COLLECTS` | image compression | 20 tests. |
| `tests/utils/test_prompt_injection_defense.py` | mira `ee44b18^` | `COLLECT-ERROR` | prompt-injection defense | **Environmental, not a recovery defect.** A module-level `@pytest.mark.skipif(get_api_key("openrouter_key") is None, ...)` calls Vault at import time; needs `VAULT_ADDR` + `VAULT_ROLE_ID` + `VAULT_SECRET_ID`. One-line lazy-guard fix if WP1 wants it enumerated without live infra. |
| `tests/utils/test_tag_parser_complexity.py` | mira `ee44b18^` | `COLLECTS` | tag parser | 9 tests. |
| `tests/utils/test_tag_parser_memory_id.py` | mira `ee44b18^` | `COLLECTS` | tag parser | 20 tests. |
| `tests/utils/test_userdata_manager_connection.py` | mira `ee44b18^` | `COLLECTS` | userdata manager | 15 tests. |
| `tests/working_memory/test_notification_center.py` | mira `ee44b18^` | `COLLECTS` | notification centre | 25 tests. |
| `tests/working_memory/test_trinket_access.py` | mira `ee44b18^` | `COLLECTS` | trinket access | 10 tests. Relevant to WP5 (both trinkets must render). |
| `tests/working_memory/test_user_name_substitution.py` | mira `ee44b18^` | `COLLECTS` | name substitution | 10 tests. |
| `tests/utils/test_pinned_http.py` | pre-existing, **not** recovered | `COLLECTS` | `e401d59` SSRF fix | 4 tests. Left byte-identical. |
| `tests/utils/test_url_safety.py` | pre-existing, **not** recovered | `COLLECTS` | `e401d59` SSRF fix | 14 tests. Left byte-identical. |

Support files recovered (not test files): `tests/__init__.py`, `tests/{api,clients,cns,cns/api,fixtures,lt_memory,working_memory}/__init__.py`,
`tests/conftest.py`, `tests/lt_memory/conftest.py`, `tests/fixtures/{__init__,conversation_data,core,failure_simulation,isolation,reset,sqlite_test_db,auth}.py`,
`tests/fixtures/realistic_conversation.json`, `tests/fixtures/pager_schema.sql`.

---

## C. Repairs applied to unblock collection

Only three defects blocked *enumeration*, and all three were pre-existing in mira-OSS — none came
from a bad recovery. Every recovery was verified to parse as Python before being committed.

### C.1 `tests/fixtures/auth.py` — recovered (absent at `ee44b18^`)

`tests/conftest.py` does `from tests.fixtures.auth import *`, but `auth.py` was deleted in `273d415`
("chore: OSS release preparation", 2025-12-18) and conftest was never updated. Because conftest loads
for every test, this aborted the whole run with exit 4 before a single test could be enumerated.
Fixed by recovering the last-existing copy from `273d415^` (207 L); its module-level imports
(`utils.timezone_utils`, `utils.database_session_manager`, `utils.user_context`) all resolve at HEAD,
and its lazy `tests.fixtures.core.ensure_test_user_exists` target exists too.

**Consequence for the record:** the pre-`ee44b18` suite could not collect even at `ee44b18^`. It had
been dead for over two months before `ee44b18` deleted it.

### C.2 `tests/fixtures/core.py` — two constants restored

The same `273d415` scrub deleted `TEST_USER_ID` and `SECOND_TEST_USER_ID` while leaving
`TEST_USER_EMAIL` / `SECOND_TEST_USER_EMAIL` in place and leaving consumers importing the deleted
names (`tests/lt_memory/conftest.py`, `tests/working_memory/test_notification_center.py`). Restored
to the pre-scrub values, verified identical against `273d415^` and against `auth.py`. `SECOND_TEST_USER_ID`
has no consumer; it came back to keep the scrubbed pair symmetrical.

### C.3 `tests/clients/test_postgres_client.py:159` — syntax error

`273d415` commented out an import but left the assignment dangling:

```python
auth_db = # Removed for OSS: AuthDatabase()
```

This is a `SyntaxError` (`ast.parse` fails), so pytest could not import the module at all. Changed to
`auth_db = None  # Removed for OSS: AuthDatabase()`, which restores enumeration and leaves the one
RLS test failing at run time for a reason WP3 can fix. This was the **only** syntax-broken file: all
54 recovered files were scanned with `ast.parse`.

### C.4 Not repaired

`tests/fixtures/*.py` carry trailing whitespace inherited from origin. Left unformatted deliberately —
reformatting would break clean diffs against the origin blobs, which is the point of verbatim recovery.

---

## D. Findings that change a later work package

### D.1 `test_greenfield_schema.py` gives WP-S a false green light

The recovered test matches tables with `rf"CREATE TABLE\s+{table}\b"`, but the two repos write DDL
differently: crm's `deploy/mira_service_schema.sql` uses bare `CREATE TABLE users (` (22 occurrences,
**zero** `IF NOT EXISTS`), while mira-OSS's uses `CREATE TABLE IF NOT EXISTS users (`.

Against mira's file the regex never matches, so `test_removed_tables_are_absent` **passes vacuously**
even though the tables it demands absent are present:

| Table the test demands absent | Present in mira-OSS `deploy/mira_service_schema.sql` |
|---|---|
| `conversation_llm` | yes (L102) |
| `internal_llm` | yes (L122) |
| `usage_pricing` | yes (L163) |
| `users_trash` | yes (L234) |
| `domain_knowledge_blocks` | yes (L307) |
| `domain_knowledge_block_content` | yes (L325) |
| `feedback_synthesis_tracking` | yes (L733) |
| `extraction_batches` | yes (L648) |

`_table_body()` has the same coupling and will raise `table X is missing` for every table it inspects.

**Action for WP-S:** make both regexes tolerate `IF NOT EXISTS` (or standardise the new file on crm's
bare style) *before* trusting this test as the schema gate.

### D.2 The test's omission list contradicts WP-S's add-back list

`test_removed_tables_are_absent` is crm's omission list, and §12 requires OSS to **add back** four of
those tables: `usage_pricing`, `domain_knowledge_blocks`, `domain_knowledge_block_content`,
`feedback_synthesis_tracking`. Per §0.2 ("crm's names and contracts are not authoritative"), WP-S must
amend this test rather than satisfy it. `feedback_signals` is also an OSS add-back that the file's
`users`-column assertions do not model. The genuinely shared omissions are `conversation_llm`,
`internal_llm`, `users_trash`, `extraction_batches`, `post_processing_batches` and the Stripe objects.

`tests/test_auth_graft.py` likewise uses `conversation_llm` correctly — as a *negative* assertion that
`ensure_single_user` no longer writes it, matching §0.1's requirement to rewrite `main.py`'s
`UPDATE users SET ... conversation_llm = ...`.

---

## E. Deliberately skipped

### E.1 crm_mira tests (out of scope — CRM or billing product)

| File at `95254a5^` | Reason skipped |
|---|---|
| `tests/test_crm_ui_contract.py` | CRM UI contract; OSS ships `oss_ui` only. |
| `tests/test_monthly_billing.py` | Billing; decision D7 omits billing from OSS. |
| `tests/test_container_appliance.py` | CRM container appliance. Noted rather than dropped silently because `97951be` (the cognitive-flags commit that also produced `test_cognitive_feature_bypasses.py`) modified it, +12/−3 — so if the flag graph changes again this file is the other place that moved. |

| Files at crm HEAD | Reason skipped |
|---|---|
| `test_billing_tool.py`, `test_crm_bulk_actions.py`, `test_crm_appointment_time_contract.py`, `test_square_import.py`, `test_square_oauth.py`, `test_square_review.py`, `test_sms_channel_consumer.py`, `test_ticket_closeout_contract.py`, `test_workphone.py` | Nine CRM-specific tests; explicitly out of scope. |

### E.2 mira-OSS assets at `ee44b18^`

| Path | Reason skipped |
|---|---|
| `tests/.DS_Store`, `tests/cns/.DS_Store`, `tests/cns/api/.DS_Store` | macOS junk. |
| `tests/lt_memory/VALIDATION_models.md` | Prose validation note, not part of the executable suite. |

`tests/fixtures/realistic_conversation.json` and `pager_schema.sql` **were** recovered: the JSON is read
by path from `tests/fixtures/conversation_data.py` and `core.py`, and the SQL is named by the
`sqlite_test_db` `schema_files` marker contract.

---

## F. What WP0 does not do

* No test was made to pass. 689 tests enumerate; their pass rate is WP1–WP6's problem.
* No product code outside `tests/` was modified.
* The 18 collection errors remain, 13 of them intentionally: they are the definition of "arrives in
  WP2–WP5", written down before that code exists.
* `pyproject.toml` / `pytest.ini` do not exist in this repo. Collection works because `tests/` is a
  package (`tests/__init__.py`), so pytest's rootdir insertion makes `tests.fixtures.*` importable. A
  future package should register the `integration` and `schema_files` markers to silence those warnings;
  WP0 did not add config files, since that is outside `tests/`.
