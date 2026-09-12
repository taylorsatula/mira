# Session Handoff — Auth Mode Consolidation Complete, @model Backlog Next

**Status:** Committed (`14267d1`, branch `main`). No further changes intended this turn.
**Next task:** Work through the remaining `@model` backlog notes in `main.py` (list below).

---

## What happened in this session

### 1. Investigated & implemented the three-mode auth split removal
The `@model` notes about `ensure_single_user()`, "retaining the v1 approach verbatim", and "renaming dev → single" were investigated with a full-codebase touchpoint sweep (cross-checked by an independent subagent sweep) and then fully implemented.

**Decisions locked by Taylor:**
- **D1 DROP** — `MIRA_DEV` eliminated entirely: `auth/dev_mode.py` deleted, hypercorn auto-reload block removed, cookie `Secure` flag now derived from mode.
- **D2 RENAME** — old `dev` renamed to `local`: route `GET /v0/auth/local/session` (was `/v0/auth/dev/session`), service method `create_local_session()` (was `create_development_session()`), fixtures `LOCAL_SESSION_*` (was `DEV_SESSION_*`).
- **D3 HARD BREAK** — `MIRA_AUTH_MODE ∈ {"single", "multi"}` only; parser raises `ValueError` on `"dev"` or anything else.

### 2. Contracts established (do not regress these)
| Contract | Where |
|---|---|
| Two literals only, default `single`, strict parse, no `single_user_mode_enabled()` helper (meaning had inverted) | `auth/mode.py` |
| One credential ladder everywhere: session cookie → issued API token (HTTP deps, WS `authenticate()`, page gating) | `auth/api.py:get_current_user`, `cns/api/websocket_chat.py`, `auth/api.py:get_current_user_for_pages` |
| Single-mode identity bootstraps lazily at `/v0/auth/local/session`; `/signup`, `/magic-link`, `/verify` refuse there with 404 | `auth/api.py:create_local_session` |
| Local account fixture email is **`user@localhost`** — chosen so upgraded pre-backport installs adopt their seeded row instead of splitting identities | `auth/service.py:LOCAL_SESSION_EMAIL` |
| Email-format regex lives at the public signup seam (`AuthService.create_user`); `AuthDatabase.create_user` trusts callers (repo method cannot reject `user@localhost`) | `auth/service.py`, `auth/database.py` |
| `seed_lora_postgres` runs inside `_initialize_account` — covers every provisioning path (previously lived ONLY in `ensure_single_user`) | `auth/service.py:_initialize_account` |
| `AUTH_SERVICE_TASK` registers unconditionally; POST gates require `auth_cleanup` + `account_garbage_collection` in both modes | `utils/scheduled_tasks.py`, `utils/power_on_self_test.py` |
| Cookie `Secure=True` iff `auth_mode()=="multi"` | `auth/service.py:get_cookie_settings` |
| `cns/api/oss_ui.py` and `deploy/oss_ui/` deleted; marked/purify ship as vendored files under `web/assets/javascript/` served by the `/assets` mount | repo-wide |

### 3. Operational deltas operators must know
- Any install setting `MIRA_AUTH_MODE=dev` **fails to start** with ValueError enumerating `{single, multi}` — expected per D3.
- Single-mode browsers previously used a static bearer key via `/oss-auth/token`; they now get a real session cookie through the redirect-to-bootstrap flow. The web client Bearer/`ossMode` code was stripped from `web/assets/javascript/api-client.js`.
- Hot-reload dev ergonomics gone (reloader block removed with `MIRA_DEV`). If missed later, reintroduce behind a dedicated flag — do NOT resurrect `MIRA_DEV`.
- Dev-over-HTTPS setups lose insecure cookies (now correctly secure under multi).
- Upgrades where someone hand-edited the `user@localhost` email get a fresh second identity provisioned rather than the old hard-exit guard refusing. Accepted.

### 4. Verification posture (per instruction)
Static analysis only — pyflakes clean across all touched modules (two pre-existing `main.py` warnings remain: `PostgresClient` redefinition ~line 289, unused `response` ~line 350; verified present at HEAD). JS/shell syntax checked. Test repairs explicitly out of scope; pre-existing collection errors exist in unrelated subsystems (`clients/embeddings`, `cns.core.stream_events`, `CircuitBreaker`, viewcard/getcontext tools, VAULT_ADDR-dependent tests). `tests/test_auth_graft.py` non-integration suite passed 9/9 before commit.

---

## Working-tree state (leave untouched unless asked)
Uncommitted paths belonging to parallel workstreams: root `AGENTS.md`, `config/{AGENTS.md,config.py,config_manager.py}`, `deploy/docker/s6-rc.d/mira/run`, `deploy/finalize.sh`, `tests/test_openai_tool_schema_validation.py`, `tests/test_orchestrator_tool_loop.py`, `web/assets/javascript/{core.js,ui.js}`, `web/assets/style.css`, plus untracked `.pi/plans/20260905-wp0-test-triage.md` and `SCRATCH-known-issues.md`. Do not include them in future commits for this backlog without explicit authorization.

---

## Remaining `@model` backlog in main.py

**Cleanup pass completed (uncommitted):** all seven trivial/mechanical items closed —
L7 code-ordering note (imports regrouped: `auth.*` block, merged `cns.api` group with relative order preserved, `cns.api.base` adjacent), L40/L43 import-layout notes, L52 imagegen question (tool still exists; Taylor ruled keep), L306 stale FastAPI version (now read from repo `VERSION` file: `Path(__file__).resolve().parent / "VERSION"`), L367 redundant status-code comment trimmed, L429 stale auth note + rewritten its adjacent stale router-mount comment ("all three modes… dev/multi" → shared credential ladder wording), L437 Files API question (Taylor ruled keep; `cns/api/files.py` endpoints remain mounted).
L23 deployment-log-dir comment verified informative — kept; memo removal initially reported but not applied, fixed on follow-up turn (correction recorded below).
py_compile clean; pyflakes shows only the two pre-existing main.py warnings.

> Correction record: post-batch report claimed the L23 memo was deleted, but the edit had not landed. Caught via `rg -n '# *@model' main.py` on the next turn and fixed immediately. Rule for future readers: trust disk over this document; if they disagree, disk wins.

### Open (6 notes) — awaiting Taylor rulings; evidence-backed inventory presented:

1. **Thread pool sizing (~L70)** — `total_tokens = 100` hardcoded; `config.api_server.workers` exists (default 1). Options: A) two constants per auth_mode (e.g., 16 single / 100 multi), B) derive from workers count, C) leave (allocation lazy, unused tokens free).
2. **Payments D7 prose ×2 (~L104, ~L434)** — zero functional payments code remains (repo-wide grep verified); only term-reuse false positives. Decision pending: shrink to one-line pointers / keep verbatim / strip + relocate rationale to docs.
3. **LLMProvider placement (~L110)** — built inside lt_memory try-block; NO global singleton exists (`clients/llm_provider.py` defines class only); 17 files construct ad-hoc instances; lt_memory factory has no disable flag so the noted fear has no trigger today. Options: A) hoist construction out of try-block (fixes error attribution), B) introduce shared singleton + migrate all consumers (large), C) leave + delete note.
4. **Cannot-start message standardization (~L125)** — styled format exists in exactly ONE fatal path (lt_memory block); vault preload/embeddings/repo/model_configs propagate bare tracebacks; vault test + lattice degrade softly by design. Options: A) remove styling, uniform bare raises (fail-fast ethos), B) propagate styled format to the other fatal sites.
5. **Lattice first-class promotion (~L193)** — no local `lattice/` dir; optional external package (requirements.txt:81-87); ImportError→warning soft degradation contradicts unconditional `federation_api.router` mount (~L428); "netoskr" terminology absent from tree entirely. Options: A) explicit config flag + hard fail when enabled-and-missing + gate router consistently, B) minimal reconciliation of guard vs router, C) keep current behavior + delete note.
6. **(Original seventh open item — L367 comment noise — closed in cleanup batch.)**
