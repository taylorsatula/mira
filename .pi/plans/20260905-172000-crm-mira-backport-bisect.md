# crm_mira → mira-OSS backport bisect

Analysis date: 2026-09-05 · Status: approved for execution · Register: functional
Target release: **mira-OSS 2.0**

## 0. Scope and topology

| Ref | SHA | Position |
|---|---|---|
| merge-base | `24abe03` | 2026-07-07 |
| `crm_mira/crm_mira` | `daf8e4a` | 2026-08-25, **162 commits ahead** — backport source |
| `mira-OSS` main | `e401d59` | 2026-09-01, **1 commit ahead** — SSRF fix absent from crm_mira |

Divergence is near-linear. Diff `24abe03..daf8e4a`: 433 files, +54,528 / −35,686 (176 A, 121 D, 136 M).
`mira-OSS` carries `crm_mira` as a fetched remote; all comparisons below are git-derived.

### Operating constraints

| Constraint | Effect on this plan |
|---|---|
| Temporary breakage of mira-OSS is permitted | CSP tightening, cursor-only keyset pagination and the fail-loud error posture are all in scope (§4 items D-1, D-2, D-3). Frontend patch need not land atomically with the protocol change. |
| Will not ship untested | WP0 (test recovery) is a gate, not a recommendation. |
| Unresolved ends and broken-but-valuable functionality must be documented | §5 Deferred Register. Each entry names an owner-action. |
| `web/` redesign not ported | mira-OSS retains its old UI; a new frontend lands in a later session. ~150 lines of protocol patching authorised (D2). |
| Billing not ported (D7) | Eight excision points, §8.2. |
| **Post-backport state is mira-OSS 2.0. No migrations, no backwards compatibility.** | One fresh-install DDL contract; `deploy/migrations/` deleted; crm's API contracts taken outright. Full consequences in §0.1. |

### 0.1 Schema and compatibility posture

The post-backport state is mira-OSS 2.0. There is no migration path and no backwards-compatibility
obligation. That sets the schema approach for the whole programme.

crm's `deploy/mira_service_schema.sql` (`8a46231`) is a fresh-install DDL contract, and is the base for
OSS 2.0 (§6.3.7). Its header states the posture:

```
-- MIRA fresh-install database contract
-- Preconditions:
--   * Connect to an empty mira_service database as the schema owner.
--   * Provision mira_admin and mira_dbuser, including credentials and BYPASSRLS
--     for mira_admin, through Vault-backed deployment tooling before this file.
-- This is deliberately not a migration. It contains no compatibility DDL,
-- embedded credentials, database creation, or default privileges.

DO $schema_precondition$ … RAISE EXCEPTION
    'mira_service_schema.sql requires an empty target database'; … $schema_precondition$;
```

One schema file is the single source of truth.

| Area | Approach |
|---|---|
| Schema authoring | Derive from crm's greenfield file, strip CRM/billing objects, add back OSS-retained objects (§6.3.7). No additive migrations. |
| `deploy/migrations/` | Deleted in full — all 19 files at main. crm's own retained residue (8 legacy + 8 CRM feature files duplicating its greenfield schema) is not inherited. |
| Row-conversion SQL | Unnecessary. `neutral_message_format.sql` converts existing rows from Anthropic wire format; a greenfield database stores neutral format from the start. The jsonb precedence bug at its `:52` is real but unreachable (§6.2). |
| API contracts | Take crm's versions outright — cursor-only `get_history`, required `subject_kind`/`timezone` with call sites updated, crm's `base.py` envelope, crm's `create_user` INSERT. No dual-mode shims, no compatibility defaults. |
| Multi-user object model | `subject_kind`, RLS on `users`/`magic_links`/`api_tokens`, `global_memories_runtime`, the `persona_*` tables and the `model_configs` table are all authored directly into the greenfield file. |

Four items in this plan are not compatibility measures and should not be read as ones:

- The ~150-line frontend patch (D2) exists because the old UI is the only UI mira-OSS has until the
  frontend session. The strict protocol rejects every frame it sends (§6.4.3).
- `get_current_user` keeps its name and `cns/api/*.py` are not ported wholesale, because crm's
  `get_current_entitled_user` hard-imports `billing` → `ImportError` at request time (D7).
- The single-user union branch in `get_current_user` (§6.3.4) and `a4df669`'s dual-protocol WS
  `authenticate()` (§6.4.5) are how `MIRA_AUTH_MODE=single`, the D3 default, authenticates — bearer
  token, not cookie.
- WP1 item 3 (`a4df669`'s four RLS call-site lines) gates RLS on `users`: with RLS enabled and no user
  context, those bare `PostgresClient('mira_service')` calls silently return zero rows. Correctness,
  independent of any schema work.

The `NULLIF` fail-closed predicate and its canary (§6.3.7) are likewise correctness items, authored
directly into the greenfield file.

Obligations the 2.0 posture creates:

| Obligation | Detail |
|---|---|
| State the upgrade policy | 1.x → 2.0 is a **reinstall, not an upgrade**. Explicit in `README.md` and release notes, including that 1.x conversation history, memories and domain knowledge do not carry forward. |
| Reconcile the deploy path | `deploy/deploy.sh --migrate` (backup → fresh install from the new schema → `schema_aware_restore.py` restores user data; `deploy/lib/migrate.sh:124-136` re-applies the old schema on rollback) is dead or must be repurposed. crm retained `deploy/migrate.sh`, `deploy/lib/migrate.sh` and `deploy/schema_aware_restore.py` alongside its greenfield schema — an inconsistency not worth inheriting. O-20. |
| Bump `VERSION` | `2026.06.25` (CalVer, identical in both repos). Establish the 2.0 marker and the scheme going forward. O-21. |
| Re-baseline the tests | WP0's recovered pre-`ee44b18` suite characterises 1.x behaviour; some of it asserts contracts 2.0 removes. O-19 is a re-baseline, not a repair. |

### 0.2 Design precedence — mira-OSS is the primary artifact

Governing rule for every judgment call in this document:

> mira-OSS is the widely distributed open-source package. crm_mira is one variant its author developed
> for himself. Optimise for mira-OSS. Do not carry a compromise that serves the CRM variant at OSS's
> expense. crm_mira will be unified onto the 2.0 frame later, and at that point **crm_mira adapts to
> mira-OSS**, not the reverse.

Consequences:

| Area | Effect |
|---|---|
| crm's names and contracts are **not authoritative** | The five route names, the frame vocabulary (`assistant_delta`, `turn_complete`, `halt`), and the `model_config=` kwarg are *proposals*, accepted where they are good on OSS's merits and renameable where they are not. D14 already exercised this (`difficult` → `other`). |
| Route naming | Reinforces O-2's recommendation: name `assessment` after its actual OSS consumer (`assessment_extractor`) rather than preserving crm's autonomy-gate semantics for a service OSS omits. |
| Seams for crm's benefit | Do not build them. `AccountProvisioner` (§6.3.5) is justified only as the minimal excision mechanism — the cheapest way to get CRM out of `service.py` — not as a future integration point. |
| Frame vocabulary | Adopting crm's strict protocol (D2) is accepted on merit: it is validated both directions, carries turn identity, and is what the future OSS frontend should be built against. The old UI is patched to it as an interim, not preserved as a contract. |
| SSRF gap in crm_mira | Closes at unification, which inherits 2.0 and therefore `e401d59`. This is a stronger basis for descoping WP7 than accepting the risk indefinitely — see §9. |
| Upstream stale docs | crm's own inconsistencies (`AGENTS.md` documenting deleted files, the three-route `COMMENT`) are not OSS's problem to inherit or to fix upstream. Correct them in OSS only. |

### Two invariants that override the breakage allowance

1. **`cns/api/oss_ui.py` + `deploy/oss_ui/{marked.min.js,purify.min.js,chat.html}` retained.** Not a
   frontend-preservation concern: `GET /oss-auth/token` is the identity source for `MIRA_AUTH_MODE=single`,
   the default mode under D3. The module reads both vendor assets **at import time** (`:18-21`), so
   deleting `deploy/oss_ui/` raises during `create_app()`.
2. **`tools/implementations/web_tool.py`, `utils/url_safety.py`, `utils/http_client.py` retained at
   main's version.** These carry `e401d59` (GHSA-rmgf-f8wc-rc3p). crm_mira has the pre-fix versions;
   any bulk copy from crm_mira reverts a security fix. See §9.

---

## 1. Decisions

| # | Decision | Binding effect |
|---|---|---|
| D1 | Persona ported as a **second parallel system**; user model retained | Two prompt slots. §7. |
| D2 | Full strict WebSocket protocol + ~150-line frontend patch | §6.4. Authorises `web/` edits for protocol compat only. |
| D3 | Full multi-user, pluggable SMTP sender | `MIRA_AUTH_MODE = single` (default) `\| dev \| multi`. §6.3. |
| D4 | Five `model_configs` routes; `internal_llm` + `conversation_llm` retired | §6.1. |
| D5 | `cost_accumulator` retained, re-keyed to `model_config_name` | Reverse-direction vs upstream. §6.1.6. |
| D6 | Full repulsion rewrite loop retained | `rewriter` → `primary` + `effort='high'`. §7.4. |
| D7 | `billing/` not ported | §8.2. |
| D8 | No web redesign port | — |
| D10 | Both Anthropic deletions ported (Batch API + Files API) | §8.4. |
| D11 | Untracked files moved to `scratch/` | Executed, §2. |
| D12 | Member-only multi-user; all demo machinery skipped | §6.3.5. |
| D13 | Model picker retired; ephemeral effort override ported | §6.1.5. |
| D14 | Single outside voice; route `difficult` → **`other`** | Catch-all for outside-model consumers. §6.1.3. |

### D1 basis

The two subsystems model different subjects. Verified against source, not inferred from names:

| | mira-OSS user model ("LoRA") | crm_mira Persona |
|---|---|---|
| Subject | the **user** | **Mira** |
| Evidence | `working_memory/trinkets/lora_trinket.py` docstring: *"descriptive (observations about the user), not prescriptive (instructions to Mira)"*; renders `<user_model>What you've learned about {first_name} through observation:` | `config/prompts/persona_evaluation_system.txt`: *"Evaluate MIRA, not the user."* / *"Do not infer user traits, preferences, or knowledge."*; renders `<persona_directives>` |

Both declare `variable_name = "behavioral_directives"` — one prompt slot, two different functions.
crm_mira deleted the user model and occupied its slot.

Capabilities present only on the user-model side, with no Persona equivalent:

- `<behavioral_checkin>` — proactive collaborative debrief. *"The user cannot see these topics unless
  you voice them."* / *"Don't ask what they want. Ask if you're reading the room right."* Plus
  verbatim-confirmed recording (visible blockquote + `<mira:checkin_response>` tag). `PersonaTrinket`
  is 27 lines; Persona refinement is machine-side only.
- Repulsion rewrite loop — register-aware removal of assistant-shaped performance. In crm_mira,
  `capture_repulsion` survives (`actions.py:2326`) but `_REPULSION_REWRITER_EXECUTOR` and both prompts
  are deleted: the action records, nothing acts.

Fact of record: **no fine-tuning exists in mira-OSS.** `git grep -i 'finetune|peft|train_lora' main`
returns only a federation adapter and a pgvector type adapter. `mirasubcortical_finetune` has zero
references from mira-OSS. "LoRA" names a prompt-injected behavioural model. Rename tracked as §10 item O-8.

---

## 2. Baseline state (executed)

Working tree clean except one intentional `.gitignore` change.

**Slack WIP flushed.** Reverted 10 tracked files: `cns/core/{continuum,message}.py`,
`cns/services/{assessment_extractor,orchestrator,peanutgallery_model,summary_generator}.py`,
`deploy/mira_service_schema.sql`, `lt_memory/processing/extraction_engine.py`, `main.py`,
`requirements.txt`. Removed 4 untracked paths: `cns/integrations/`,
`config/prompts/slack_evaluation.txt`, `docs/adr/`, `tests/test_slack_integration.py`.
Snapshot: `/tmp/slack-wip-flushed-20260905/{tracked-modifications.patch,untracked-slack-files.tgz}` (40 KB).

No value lost. The flushed `Continuum.add_user_message(metadata=…)` and
`orchestrator(user_message_metadata=…)` plumbing is a subset of what D2 delivers:
`add_user_message(*, message_id, metadata)`. The `cns/integration` vs `cns/integrations` collision and
the ADR-numbering collision are both void.

**Scratch moved (D11)** → `scratch/`, gitignored: `MIRA_ARCHITECTURE_OVERVIEW.md`,
`MIRA_ARCHITECTURE_OVERVIEW copy.md`, `AGENTS_fromroundcubebutgoodcollaborativenotestopull.md`,
`tests/conftest.py`.

`scratch/conftest.py` is not scratch in origin. mira-OSS deleted its entire suite in
`ee44b18 "tmp: v2 preview"` (54 Python test files); this file re-creates one that deletion removed.
Currently unused by the two tracked tests. Restore per WP0.

---

## 3. Test posture

mira-OSS main: **2 tracked test files** (`tests/utils/test_pinned_http.py`, `test_url_safety.py`).
crm_mira: suite deleted in `95254a5`, 11 CRM-contract tests remain. Neither repo can regression-test
this backport as it stands. WP0 is a gate.

| Recoverable asset | Add commit | Delete commit | Validates |
|---|---|---|---|
| `tests/test_auth_graft.py` | `0ef36fe` | `95254a5` | WP3 — session hashing, expiry, cleanup ordering, compensation-on-failure |
| `tests/test_ordered_turn_persistence.py` | `c1297b3` | `95254a5` | WP4 |
| `tests/test_web_frontend_protocol.py` | `c1297b3` | `95254a5` | WP4 frontend compat — the D2 risk |
| `tests/test_history_cursor.py` | `a11d04e` | `95254a5` | WP4 keyset cursor |
| `tests/test_cognitive_feature_bypasses.py` | `97951be` | `95254a5` | WP1 flag omission (93 L) |
| `tests/test_tool_config_resolution.py` | `2e60260` | `95254a5` | WP1 per-user tool config (22 L) |
| `tests/test_openai_tool_schema_validation.py` | — | **at HEAD** | WP1 tool-call robustness |
| `tests/test_orchestrator_tool_loop.py` | — | **at HEAD** | WP1 tool-call robustness |

Recovery: `git show 95254a5^:tests/<file> > tests/<file>`.
Also: `git ls-tree -r --name-only ee44b18^ -- tests/` → 54 files across `tests/{api,clients,cns,fixtures}/`.
Predates the divergence; requires repair. Restore `scratch/conftest.py` with it.

---

## 4. Items admitted under the breakage allowance

Three items whose only objection was preservation of the retained frontend. All are PORT.

| ID | Item | Condition |
|---|---|---|
| **D-1** | CSP tightening — `28f25c8` CSP half + `auth/security_middleware.py:38-53` | Removes `'unsafe-inline'` from `script-src`, narrows `img-src 'self' data: https:` → `'self' data:`. Breaks `oss_ui.py:19-20` inlined `marked.min.js`/`purify.min.js` and any inline handler. Three follow-ups, tracked in §5: externalise the inlined vendor scripts, drop the Stripe origins (`:38,46,48`), add `worker-src` for `web/sw.js`. Drop `importmap_csp_hash`. Headers at `:27-35` port unconditionally — free. |
| **D-2** | Keyset pagination `a11d04e` — **cursor-only, no dual-mode shim** | crm's `_get_history()` raises on `offset`/`search` (`data.py:117,119`). `history.js` is offset-based end-to-end (`:75-98`, `:410-433`, `:243-245`). Rewrite `history.js` in the D2 patch budget, or leave the drawer broken and track under D-6. Mirror crm's own split: `search_continuums()` retains offset pagination via `SearchHistoryResult`. |
| **D-3** | Fail-loud posture — `_process_persona()` exception propagation, `_surface_memories()` degraded-fallback removal (`97951be`) | main swallows and logs; crm propagates into the collapse-attempt counter toward `MAX_COLLAPSE_ATTEMPTS = 3` tombstone. Correct posture. Operational consequence: segments that main limped through now tombstone. Document in release notes, not in code. |

WP4 does not require the frontend patch to land atomically with the protocol change. Server-side may
land first with the UI non-functional in the interim, tracked in §5 item D-6.

`get_current_entitled_user` is declined regardless of the breakage allowance: a hard `ImportError` at
request time under D7, not a temporary degradation.

---

## 5. Deferred register

Broken-but-valuable and unresolved ends. Each entry is actionable by a team member without re-deriving
this analysis.

| ID | Item | State after backport | Owner-action |
|---|---|---|---|
| **D-4** | `viewcard_content.py` (276 L) | **PORT now, no consumer.** Zero-CRM server-side content sanitizer: allowlisted HTML+SVG tags; blocked `script/iframe/object/embed/form/input/link/meta/base/button/video/audio/animate`; blocked `on*` attributes and `contenteditable`; CSS validated via `tinycss2` blocking `position:fixed`, `@import`, `@font-face`, `@namespace`, `@page`, `:host`/`::part`/`::slotted`, and any external URL (only local `url(#fragment)`); `<a href>` must be absolute http/https; `<img src>` must be a `base64.b64decode(validate=True)`-verified PNG/JPEG/GIF/WebP data image; markdown → HTML via `markdown` + `html5lib.parseFragment`/`serialize`. | Wire as the sanitization boundary when the new frontend lands. Do not reinvent XSS defence. Requires `Markdown`, `html5lib`, `tinycss2`. `viewcard_tool.py` itself stays OMITTED — hard CRM import at `:12` (`_crm_client.client_for_workspace`), 8 of 10 card types are CRM entities with `_schedule_data_from_crm`/`_customer_data_from_crm` resolvers. |
| **D-5** | Web Push subscription lifecycle (`cns/api/push.py`, `push_service.py`, `push_repository.py`, `add_push_subscriptions.sql`) | OMITTED. Generic half welded to CRM autonomy "summon alerts"; auth is `get_current_entitled_user`; VAPID keys from crm config; `main.py` lifespan hard-imports it. | Extract as standalone: subscription endpoints + repository + a generic `notify_user(user_id, payload)` hook. `pywebpush` is MPL-2.0 (file-level copyleft, acceptable). Nothing retained imports push — omission is clean, no stubbing. |
| **D-6** | Retained frontend protocol patch (~150 L) | May lag WP4 server-side. Interim: UI authenticates never, sends never, renders never. | Apply per §6.4.3. `tests/test_web_frontend_protocol.py` is the acceptance gate. |
| **D-7** | CSP follow-ups (from D-1) | Old UI's inlined vendor scripts blocked; `web/sw.js` lacks `worker-src`. | Externalise `marked.min.js`/`purify.min.js` from `oss_ui.py:19-20` into served static files, or add hashes. Add `worker-src`. Belongs to the frontend session. |
| **D-8** | `local_datetime_to_utc_iso()` (`d3e5f20`) | PORTED, **dormant**. 47 additive lines in `utils/timezone_utils.py`, zero new imports, deps satisfied at main (`get_timezone_instance:108`, `UTC_TIMEZONE:54`, `datetime:16`). Refuses ambiguous (fall-back overlap) and nonexistent (spring-forward gap) local times via `fold=0`/`fold=1` candidates plus a UTC round-trip identity test. | `git grep -ln "scheduled_at" main -- tools/` → empty. Candidate consumer: `tools/implementations/reminder_tool.py` — **not yet verified** whether it accepts a model-supplied datetime. If it does, wire it and this becomes a live DST-bug fix. crm's consumers (`jobs_tool.py`, `workflows_tool.py`) are CRM. |
| **D-9** | Anthropic dialect lacks `invalid_reason` coverage | Tool-call robustness (WP1) is implemented only in `openai_chat_base`. main routes heavily through **anthropic**, where the SDK parses tool inputs and gets no `invalid_reason` treatment. | Grow an equivalent in `clients/llm/dialects/anthropic.py`. Port is still a net win; coverage is partial. |
| **D-10** | Orphaned `anthropic_batch_key` | D10 deletes the Batch API. mira-OSS still seeds `anthropic_batch_key` at `mira_service_schema.sql:135-157`, and the Vault key persists. | Remove the schema reference and the Vault key, or document as intentionally reserved. |
| **D-11** | Workphone channel-module pattern | Code OMITTED. Pattern is sound: transport isolated in a client; identity/credentials/ledger/opt-out in the module; inbound facts published as events on the shared `EventBus`; consumers subscribe; no consumer detail leaks into the module. | Document as the OSS template for future notification channels (Telegram, Matrix, a rebuilt Slack). Do not port an empty abstraction. |
| **D-12** | Rebuilt Slack integration | Flushed (§2). Guard pattern worth reusing: a `metadata.integration.type` discriminator with skip-guards at each cognitive entry point (assessment extraction, summary generation, peanut gallery, memory extraction). | Snapshot retained at `/tmp/slack-wip-flushed-20260505/`. Under D4, a future participation gate maps to route `assessment` (effort=none, cheap); its token ceiling overrides per request at `resolver.py:66`. |
| **D-13** | phone-a-friend MCP extraction | D14 keeps the in-tree tool. Upstream deleted it (`d90bf5d`) because it was **extracted**, not scoped out: sibling repo `phone-a-friend-mcp` (`src/phone_a_friend/{server.py,hub.py}`, "consolidate 8 tools into 1"), reachable via mira's `mcp_client.py`. | If the extraction is later followed in OSS: **unverified** whether `mcp_client.py` supports it and whether `ESSENTIAL_TOOLS` / tool discovery handle an MCP-provided tool equivalently. Verify before committing. |
| **D-14** | mira-OSS pre-`ee44b18` suite (54 files) | Restored per WP0 but will need repair against 162 commits of divergence. | Triage: keep what covers retained subsystems, delete what covers removed ones (`files_manager`, batch coordinator, LoRA-only paths). |
| **D-15** | `scratch/MIRA_ARCHITECTURE_OVERVIEW*.md` | Two divergent copies (882-line sectioned, mtime 17:20; 599-line distilled, mtime 18:01). Neither shipped. | Both are stale the moment WP1 lands (they document `internal_llm`/`conversation_llm` resolution, `batch_result_handlers`, `files_manager`). Re-derive from the post-backport tree rather than updating. |
| **D-16** | `auth/oauth.py` (443 L) | OMITTED — Square OAuth, imports `clients.square_client`. Dormant even upstream (`e03b569`: *"Square OAuth router not mounted (awaiting credential-ownership UX decisions)"*). | Structure **not assessed** for generic OAuth (Google/GitHub) reuse. Assess if OSS ever wants social sign-in. |
| **D-17** | `docs/AUTH_FOLLOWUPS.md` | OMITTED — documents the `oss_ui` deletion as intentional and states billing/Square cores are active. | Mine for follow-up items relevant to OSS; do not port the file. |

---

## 6. Work packages

### 6.1 WP2 — `model_configs`

#### 6.1.1 Contract

Replaces `internal_llm` (14 function-addressed rows) + `conversation_llm` (user-facing tiers) with one
capability-addressed table. Enforced at three sites:

```sql
CHECK (name IN ('primary','fast','batch','assessment','difficult'))   -- deploy/mira_service_schema.sql:51
CHECK (dialect_name IN ('anthropic','openai','openrouter','groq'))
CHECK (effort IN ('none','low','medium','high','xhigh','max'))
CHECK (max_tokens > 0)
```
```
utils/user_context.py:204        _MODEL_CONFIG_NAMES frozenset; load_model_configs() set-equality
utils/power_on_self_test.py:748  _check_llm_configuration() re-asserts names == {5} and len(rows) == 5
```

Per-request `effort` / `max_tokens` overrides honoured at `resolver.py:66-67` (caller wins over row).
This is the mechanism D5, D6 and D8 depend on.

Widening precedent: `6c019db` created three routes → `6899d07` unlocked per-row dialect + added
`effort`/`max_tokens` → `add_voice_model_configs.sql` widened CHECK to five → `ee222fd` aligned the two
Python validators. Widening is a SQL IN-list plus two frozensets.

Stale artifacts to correct on touch: schema `COMMENT ON TABLE model_configs` (*"Exactly three required
MIRA routes"*, `:1232`); `clients/AGENTS.md` (*"Callers pass exactly one of primary, fast, or batch"*).

#### 6.1.2 Code-migration surface

This is the Python call-site migration. The DDL side is WP-S (§6.3.7).

**206 references across 36 Python files.** Narrow chokepoint: all runtime lookups flow through
`get_internal_llm()` and `resolve_conversation_llm()` in `utils/user_context.py`, reached only via
`ModelResolver._resolve_internal_llm` / `_resolve_conversation_llm`, reached only via
`LLMProvider.generate_response` / `stream_events` routing kwargs. crm's own commit touched ~17 leaf
call sites.

Concentration: `resolver.py` 36 · `user_context.py` 28 · `actions.py` 18 (deleted with the picker) ·
`power_on_self_test.py` 15 · `types.py` 12 · `cost_accumulator.py` 8 · `orchestrator.py` 7 · `main.py` 6.

Additional OSS-only surface crm never touched: `usage_pricing` table + `main.py:254` seeding,
`deploy/postgresql.sh:103` `OFFLINE_SQL`, `deploy/lib/migrate.sh:640-1196` table lists,
`web/settings/index.html` picker.

Signature delta:

```
main  generate_response(..., *, internal_llm=None, conversation_llm=None, dialect_name=…,
                          model=…, endpoint_url=…, api_key=…, allow_negative=False,
                          allow_provider_stall_fallback=True)          clients/llm_provider.py:107
crm   generate_response(..., *, model_config: str, ...)                clients/llm_provider.py:65
```

`model_config` is **required, no default**. `allow_negative` (billing concept),
`allow_provider_stall_fallback`, `dialect_name`, `model`, `endpoint_url`, `api_key` all removed.
~31 call sites in the cognitive core are un-portable as written until this lands.

#### 6.1.3 Route mapping

`difficult` has **no OSS consumer** — both crm consumers (`autonomy_service`, `business_voice_service`)
are omitted as CRM. Rename to `other` per D14 costs nothing.

| Route | Consumers | Notes |
|---|---|---|
| `primary` | main chat (`orchestrator llm_kwargs`), `summary`, `synthesis`, `critic`, `portrait`, `tidyup`(peanutgallery), `rewriter`, `overwatch` | `rewriter`: `+effort='high'` override (D6). `overwatch`: `+max_tokens` override — needs ~100, row default 16000. |
| `fast` | `analysis` → `subcortical.py:154,241`, `domaindoc_summary_service.py:99`, `tool_result_summarizer.py:207`, `entity_merge.py:170`, `prompt_injection_defense.py:374`; `tidyup`(pager) → `pager_tool.py:1393` | Unanimous crm precedent across all five `analysis` sites. |
| `batch` | `extraction` → `lt_memory/processing/execution_strategy.py:442`; `forage`; `whilethecatsaway` | Synchronous. Unambiguous after D10 removes the async Batch API. |
| `assessment` | `assessment_extractor.py:119` | **Judgment call.** crm mapped this to `batch` to avoid a name collision with its own effort=none autonomy gate. That collision only bites if OSS ports `autonomy_service` — it does not. With `assessment` otherwise unclaimed, mapping `assessment_extractor` → `assessment` is self-documenting and frees `batch` to mean purely bulk background work. Both defensible; this plan takes `assessment`. |
| `other` | `phoneafriend_tool.py:155` (both voices collapsed, D14) | **MUST be seeded to a different vendor/model than `primary`.** Enforce with a startup assertion in `load_model_configs()` or `_check_llm_configuration()` when `other.model == primary.model`. `effort='high'` matches `phoneafriend_claude`. |

Latent main defects resolved by WP-S/WP2: `portrait` and `whilethecatsaway` are referenced in
code (`portrait_service.py:191,234`; `whilethecatsaway_agent.py:24`) but **never seeded** in any schema
or migration — live `KeyError` paths. Also eliminated: `main.py:62` looks for a `conversation_llm` row
named `'offline'` while `deploy/postgresql.sh:103` inserts `'qwopus'` (so `oss_default_tier` silently
falls back to `'primary'`); `user_context.py:391` defaults to `'minimax'`, seeded nowhere.

#### 6.1.4 `ROUTE_FALLBACKS` re-derivation

crm `resolver.py:14-17`: `{"assessment": "primary", "difficult": "fast"}` — a plain constant encoding
effort-preservation (both none / both high) and a local-vs-cloud split. Sole consumer is the startup
POST gate, which classifies each failed endpoint fatal vs non-fatal: all routes behind that endpoint
have a fallback entry → warn and continue; any cloud route down → `RuntimeError`.

Wrong by construction under OSS seeding:

- `other` → `fast` converts "consult an outside model" into "consult yourself" whenever `fast` shares
  `primary`'s vendor. **`other` gets no fallback.**
- `assessment` seeded at a cloud provider is not a local route.

Action: make criticality data-driven — *routes the chat path depends on are fatal* — not name-derived.
If OSS seeds no local routes, `ROUTE_FALLBACKS` is empty and everything is fatal.

No runtime failover exists at crm HEAD. `7d8a098` and `8520d1a` were superseded in-branch by
`a8bce9b`. Do not port the intermediates.

Port from this area: `2f980ab` bounded pre-server gate (`PRE_SERVER_GATE_ATTEMPTS=3`,
`PRE_SERVER_GATE_RETRY_SECONDS=10`, then **park** — `while True: sleep(300)`, server never binds —
because s6 restart-on-exit turned a failing gate into an infinite loop of real LLM probes pinning both
GPUs at 120–190 W); `b88f076` (probe uses `config.api.temperature`, not hardcoded 0). Make
park-vs-exit configurable — systemd operators may prefer exit+restart.

#### 6.1.5 D13 execution

Delete the picker in `web/settings/index.html` and the `get_conversation_llm` / `set_conversation_llm`
actions. Port `f8cf0d3` ephemeral effort override as the replacement per-user knob:

```
cns/api/actions.py   set_effort_override / get_effort_override / clear_effort_override
                     validated against EFFORT_LEVELS from clients/llm/types.py (exists at main)
orchestrator.py      :1105-1125 — reads Valkey effort_override:{user_id} (SETEX 3600) BEFORE
                     subcortical assessment, injects llm_kwargs['effort'], SKIPS subcortical when
                     an override is present, fails open on Valkey errors
```

Per-user infrastructure, not CRM-coupled. Fits D3; in `single` mode the key carries the single user's id.

**Forced prompt edit (R2):** delete the substrate paragraph from `config/system_prompt.txt` — *"This may
change between turns if {first_name} switches models mid-conversation…"* — which becomes false once
per-user switching is removed.

#### 6.1.6 D5 execution

`cost_accumulator` was fed by `clients/llm/accounting.py` (`_record_cost_accumulator`), also deleted.
`UsageAccountingPolicy.for_current_build()` keyed on `find_spec("billing")` — already `required=False`
in OSS under D7.

Action: do **not** port `accounting.py`'s billing-gated policy machinery. Add a slim
`cost_accumulator.record(result)` hook at the provider boundary or in
`LLMLifecycle._with_transport_metadata`, re-keyed to `model_config_name`. `Result.usage` survives in
crm's `types.py`. Retain `usage_pricing` — authored directly into the greenfield schema — and
`main.py:254` seeding. The two pricing migrations (`default_pricing_fallback.sql`,
`tier_qualified_pricing_keys.sql`) are deleted with the rest of `deploy/migrations/` (§6.3.7).

Existing wiring: `cns/api/chat.py:272-292` (start/drain). The `show_cost` param is never sent by the
retained frontend (zero grep hits in `web/`), so its removal is frontend-safe either way.

#### 6.1.7 Seed scrub

Replace before publishing:

```
http://192.168.1.9:3090/v1/chat/completions     private LAN llama-server
'Qwopus 27B Fusion'                             personal fine-tune name
https://api.kimi.com/coding/v1/chat/completions personal provider arrangement
'k3-256k', kimi_key, llama_server_key           Vault key names
```

Python layer clean: grep for kimi/laguna/poolside/Qwopus across crm's `*.py` returns nothing;
intermediate-commit model IDs absent from the final tree; `_MAX_EFFORT_PER_MODEL` (`openai.py:34`) is
an empty dict.

Offline installs: `deploy/postgresql.sh:103` `OFFLINE_SQL` currently rewrites row endpoints. The
`model_configs` equivalent is seeding the five rows with local endpoints. Mechanism ports cleanly.

---

### 6.2 WP1 — independent fixes

Items 2–17 are verified present-and-defective at main, with no dependency on any programme. Ship as
separate commits.

| # | Commit | Defect at main | Action | Size |
|---|---|---|---|---|
| 1 | `55820a3` | **Not applicable — do not port.** `deploy/migrations/neutral_message_format.sql:52` reads `elem \|\| '{"type":"tool_call"}' - 'type' \|\| …`. Postgres binds binary `-` (addition/subtraction row) tighter than `\|\|` (catch-all row) → parses as `'{"type":"tool_call"}' - 'type'` → **`operator is not unique: unknown - unknown`**, a parse/analysis-time error that fires regardless of row count and leaves the schema half-migrated because each statement autocommits and Step 3g has no conflict target. The bug is real and **unreachable in 2.0**: Step 3a is an `UPDATE messages SET content = …` converting existing rows from Anthropic wire format, a greenfield database has none, and the migrations directory is deleted (§6.3.7). Recorded so the defect is not rediscovered and re-triaged. | none | n/a |
| 2 | `28f25c8` **models.py hunk only** | `lt_memory/models.py:165` types `embedding: Optional[List[float]]`. `register_vector()` active at `postgres_client.py:102,135` and `database_session_manager.py:386,433`, so retrieval returns `pgvector.types.Vector` — not a `list` subclass, not iterable (`list(v)` raises `TypeError`), exposes `.to_list()`. Pydantic rejects. Reachable: `db_access.py:489,507`, `hybrid_search.py:142,159`. | Hand-apply 17 additive lines (`@field_validator('embedding', mode='before')` after `:206`). `field_validator` already imported at `:6`. **Do not cherry-pick the commit** — its CSP half is D-1, port separately. | 17 lines |
| 3 | `a4df669` **4 lines only** | Bare `PostgresClient('mira_service')` at `utils/user_context.py:378` (`get_user_preferences`), `:420` (`update_user_preference` — an unscoped **`UPDATE users`** write path), `cns/services/portrait_service.py:138` (`read_portrait`), `:352` (`_save_portrait`). | `PostgresClient('mira_service', user_id=user_id)`. **Gate for WP3:** must land before RLS is enabled on `users`. Preserve the commit rationale in the message: *these run from scheduled jobs and background threads where the contextvar may not be set.* Skip the commit's websocket-auth hunk (WP4). | 4 lines |
| 4 | `2e60260` | `config_manager.get_tool_config()` returns a process-global cached instance with no user dimension. Overrides saved via `cns/api/tool_config.py` → `utils/tool_config_store.save_user_tool_config()` are **persisted but never read at execution time** — the Tool Settings UI is a no-op. `load_user_tool_config(tool_name, hydrate_secrets=False)` already exists at `tool_config_store.py:43` with Vault hydration at `:58`. | Port in full (~20 L). Merge is `{**default_config.model_dump(), **user_config}` re-instantiated through `config_class(...)` — a user override cannot produce an invalid config. `except RuntimeError: return default_config` is already the correct single-user/startup/batch fallback. Plus 2-line changes in `continuum_tool.py` and `memory_tool.py`, plus the 22-line test. | ~20 L + 2 sites + test |
| 5 | `e370468` **agents/base.py only** | `:740` appends the assistant message **before** knowing whether tool calls exist; `:798` uses `tool_result_messages()` (`clients/llm/tool_messages.py:44-58`), which does no filtering against the assistant message's `tool_calls`. Any call without a matching result — server-side `code_execution`, pre-dispatch rejection, invalid arguments — becomes an **orphaned `tool_call` → provider 400 next iteration**. Affects forage, memory_curator, whilethecatsaway. Safe helper `append_tool_result_messages()` already exists at `:61` and the orchestrator already uses it (`orchestrator.py:53,670`); `agents/base.py` is the sole straggler. | Move `messages.append(assistant_message_from_result(response))` inside the `if not tool_calls:` branch; replace `:798` with `messages[:] = append_tool_result_messages(messages, response, tuple(tool_results))`. Skip the `invalid_reason` half — that field does not exist at main (`types.py:270-292` has only `id`, `tool_name`, `input`); it arrives with item 11. | 3 lines |
| 6 | `d3e5f20` **timezone_utils only** | n/a — additive capability | Port 47 lines, `local_datetime_to_utc_iso()`. Dormant; see D-8. Do not take the commit's `jobs_tool.py` (83 L), `workflows_tool.py` (80 L), `_crm_client.py` (26 L), closeout skill rename, or `test_crm_appointment_time_contract.py`. | 47 lines |
| 7 | `770d89a` | `_read_secret_version()` does not re-authenticate. AppRole tokens have ~1 h TTL → **any uncached KV read 403s an hour after boot**. | Port `clients/vault_client.py` +15/−6: catch `Unauthorized`/`Forbidden`, re-run `_authenticate_approle()` once. `_authenticate_approle` already at `:73`. Zero coupling. Port first and independently. | +15/−6 |
| 8 | `f8cfea9` **3 lines only** | `UserDataManager.base_dir` is CWD-relative. | `base_dir` → `Path(__file__).resolve().parent.parent / "data/users"` (`:85-90`). Take only these lines — the rest of that file's diff deletes `_init_contacts_schema` / `_init_files_api_schema`, and `_init_contacts_schema` must survive (contacts_tool is retained, §8.3). | 3 lines |
| 9 | `457a56e` (partial) | History rows lack tool-call correlation. | Add `tool_call_id` + `is_error` to `continuum_repository.get_history()` and `search_continuums()`. Purely additive, backwards compatible, no frontend change. | additive |
| 10 | crm `segment_poller.py` (partial) | A poller thread exiting via the exception path at `:146-178` leaves a dead `stop_event` in `_active_pollers`, so **the user's poller is never restarted**. | Take the `_active_pollers.pop(user_id)` cleanup under `with self._pollers_lock:`. **Skip** the `get_billing_backend().has_product_access(...)` gate immediately above it (D7). | ~4 lines |
| 11 | `922b1e5` → `e370468` → generic parts of `e26d031` | Providers (esp. OpenRouter-fronted) return tool calls with malformed JSON args, missing required fields, or schema-violating values (`e26d031`'s example: enum `"crm_clients_tool"` against enum `["clients_tool"]`). Before: `_parse_tool_arguments` raised `ProviderProtocolError` → **entire response failed**, no self-correction. `922b1e5` alone introduced a second defect, fixed by `e370468`: the invalid call was replayed as an assistant message with bad `tool_calls` plus a `role=tool` error result → **next request re-sent the schema-invalid arguments → hard 400**, poisoning the conversation. | See §6.2.1. | multi-file |
| 12 | `f099b5f` | Consecutive OpenRouter `reasoning.text` stream deltas not coalesced. | Port `_accumulate_reasoning_details()` (`openai_chat_base.py:811-854`): concatenates `text`, first-wins `signature`/`format`, raises `ProviderProtocolError` on non-object items and non-string text. Applied in **two** places — stream accumulation (~`:342`) and message round-trip conversion (~`:704`), the latter preventing unbounded `reasoning_details` list growth when replaying history. main's `openai_chat_base.py` is byte-identical to merge-base here → applies cleanly. | as-is |
| 13 | `a7b8694` net of `ed441e5` | main lacks `reasoning_content` extraction — the llama.cpp/local-server convention. | Port `_extract_reasoning_message` / `_extract_reasoning_delta` from `openai.py`. **Do not** port the `chat_template_kwargs` injection: added by `a7b8694`, removed by `ed441e5`; `git grep chat_template_kwargs` at HEAD returns zero hits. Net effect is extraction only. Directly benefits mira-OSS offline mode. `_is_local_endpoint()` survives at `openai.py:46`. | as-is |
| 14 | `edbbed2` part a | No stream-chunk diagnostics. | Port `utils/llm_tap.log_stream_chunk()` (+`kind` field on responses), its wiring in `openai_chat_base.stream()` behind `llm_tap.is_active()`, and the "SSE chunk must be a dict" protocol check. Part b (WS close codes 4002/4008) belongs to WP4. | as-is |
| 15 | `dd82063` | Google Maps is paid and API-key-gated. | Port in full. See §6.2.2. | multi-file |
| 16 | `97951be` generalised | No feature-flag mechanism. | Port and generalise. See §6.2.3. | ~35 L + registry |
| 17 | `10ca6e3` | No model-invoked friction capture. | Port `feedback_tool.py` with edits. See §6.2.4. | + DDL |

#### 6.2.1 Item 11 detail

Mechanism, final state at HEAD:

```
clients/llm/types.py:269                ToolCall.invalid_reason: str | None (validated non-empty when set)
clients/llm/dialects/openai_chat_base.py
  :967-990                              validate against registered ToolDefinition.input_schema with
                                        jsonschema.Draft202012Validator — deterministic sorted
                                        first-error, location path in message, RuntimeError for a
                                        bad schema itself
  ~:1019 / ~:1052                       _parse_non_stream_tool_call / _parse_stream_tool_calls catch
                                        ProviderProtocolError → set invalid_reason + empty input
                                        instead of raising
cns/services/tool_loop.py:140-147       _execute_tool short-circuits invalid calls BEFORE the
                                        invocation try/except: warning log + error ToolExecutionResult
                                        whose content includes _schema_hint(tool_name, error) (:213)
clients/llm/tool_messages.py            append_tool_result_messages partitions valid/invalid:
                                        valid → normal assistant/tool pair;
                                        invalid → invalid_tool_call_feedback_messages() emits a
                                        role=user "[Automated system message: … rejected before
                                        execution because its arguments were invalid: {reason}. Issue
                                        a new tool call with valid arguments…]"
                                        The assistant message is OMITTED entirely when it would carry
                                        no tool_calls/reasoning/text (include_assistant guard) —
                                        this is what stops the 400-replay loop.
                                        Also: result.text.strip() fix for whitespace-only text blocks.
cns/services/orchestrator.py            invalid ids excluded from persisted_tool_ids (no orphaned pairs)
agents/base.py                          inherits via shared append_tool_result_messages
```

Items in `clients/llm/*` plus the one `tool_loop.py` method apply cleanly. The orchestrator and
`agents/base.py` hunks are 4–8 lines each but sit in heavily diverged files — hand-apply.
Requires `jsonschema` in `requirements.txt`. Coverage gap: D-9.

Circuit-breaker hardening (same commit, separable, lands in the same file as the WP4 halt changes —
split deliberately): main's `CircuitBreaker.should_continue()` (`tool_loop.py:58-77`) trips on *any*
second error from the same tool name regardless of arguments, and on two consecutive identical result
hashes. crm (`:89-103`) replaces both with one rule keyed on
`input_hash = sha256(json.dumps(tool_call.input, sort_keys=True))`: *same tool + same arguments + prior
error*. Removes two false-positive classes. Adds `ToolReportedError.from_result()` (`:22-45`) so
`{"success": false, "error": …, "recovery": …}` counts as a failure rather than a success.

CRM parts of `e26d031`, cleanly separable by file: bulk_create/update/delete tool variants, the
skeletonkey confirmation gate in `_crm_client.py`, the `workflows_tool` booking refactor. Omit.
`8cb88a9` and `2eaa3eb` are CRM bulk-operation fixes. Omit.

#### 6.2.2 Item 15 detail — Maps → OpenStreetMap

Verified `maps_tool.py`, `weather_tool.py`, `config/config_manager.py`, `requirements.txt` are
**byte-identical between merge-base `24abe03` and main `e401d59`** → code applies with zero conflict.
Sole conflict: `tools/implementations/AGENTS.md` and `utils/AGENTS.md`, touched by both `dd82063` and
`e401d59` — trivial manual doc merge.

| Capability | Google (main) | OSM (crm) | Status |
|---|---|---|---|
| Forward geocode | `client.geocode()` | `nominatim_client.search()` → `/search` | survives |
| Reverse geocode | `client.reverse_geocode()` | `.reverse()` → `/reverse` | survives — single best result vs ranked list |
| Place details | `client.place(place_id)` | `.lookup()` → `/lookup?extratags=1` | survives — **gains** website/phone/opening_hours |
| Places nearby | `client.places_nearby()` | `.nearby()` → Overpass `around:` | survives |
| Find place | `client.find_place(fields=…)` | `.search()` | survives (now identical to geocode) |
| Distance | Haversine, no API | unchanged | survives |

Regressions, all minor: `open_now` has no Nominatim equivalent (genuine loss); `language` not set
(Nominatim supports `accept-language`); reverse geocode returns one result; no ratings/business_status;
weaker business-POI coverage. No directions either way — not a regression.

Usage-policy compliance verified in `utils/nominatim_client.py`: identifying `USER_AGENT =
"MIRA-Assistant/1.0 (OpenStreetMap Nominatim client)"` on every `_get()` and Overpass `POST` (~`:44,:86,:205`);
process-wide `_throttle()` via `threading.Lock` + `time.monotonic()`,
`MIN_REQUEST_INTERVAL_SECONDS = 1.0`, called before **every** Nominatim and Overpass request. Overpass:
`OVERPASS_MAX_ATTEMPTS = 4`, jittered exponential backoff (`2.0 * 2**attempt + random.uniform(0,0.5)`),
rotation across `overpass-api.de` / `kumi.systems` / `private.coffee`, retries both read-timeouts
(`http_client.TimeoutException`) and the HTTP-200-with-`remark` runtime-error mode that status-code
retries miss. QL injection guarded by `_escape_overpass_string()` (escapes `\` and `"` on
`keyword`/`place_type`).

Credentials: **none.** No API key, no env vars. `google_maps_api_key` referenced at exactly three
places in main — `config_manager.py:93-95`, `maps_tool.py:223`, `weather_tool.py:594` — all removed by
the commit. `googlemaps` appears only in `maps_tool` and `weather_tool`; removal is safe.

SSRF: **impossible by construction** — `NOMINATIM_BASE_URL` and the `OVERPASS_URLS` 3-tuple are
hardcoded constants; user input (`q`, `keyword`, `place_type`, lat/lng) reaches only params/data, never
the host. Uses plain unpinned `http_client.get/post`, correct per main's doctrine.

#### 6.2.3 Item 16 detail — feature flags

Three layers, ~35 new lines:

```
config/config.py:84-89        SystemConfig gains subcortical_enabled: bool = Field(default=True, …)
                              ApiConfig.analysis_enabled deleted
config/config_manager.py      _load_system_feature_flag_overrides() -> dict[str, bool]
                              environment_fields = {"MIRA_SUBCORTICAL_ENABLED": "subcortical_enabled",
                                                    "MIRA_PEANUTGALLERY_ENABLED": "peanutgallery_enabled"}
                              STRICT: if raw_value not in {"0","1"}: raise ValueError(
                                f"{environment_name} must be exactly 0 or 1")
                              -> `=true` / `=yes` / `=""` fail loudly at config load rather than
                                 silently meaning "off". None (unset) = use the field default.
                              Wired in AppConfig.get_instance() as
                                cls(system=SystemConfig(**_load_system_feature_flag_overrides()))
cns/integration/factory.py    :312-326 _get_subcortical_layer() returns None BEFORE importing or
                              instantiating SubcorticalLayer;
                              _initialize_peanutgallery_service() early-returns before importing
                              PeanutGalleryModel/Service/Trinket, so the TurnCompletedEvent
                              subscriber is never registered
consumers                     ContinuumOrchestrator.__init__(subcortical_layer: SubcorticalLayer|None);
                              _surface_memories() short-circuits to MemorySurfacingResult(
                                surfaced_memories=[], pinned_ids=set(), subcortical_result=None)
                                BEFORE touching embeddings or retrieval;
                              warm_cache() guarded
                              SubcorticalLayer.__init__ loses analysis_enabled and its
                                raise RuntimeError("SubcorticalLayer requires analysis_enabled=True")
```

Omission at construction, not runtime branching: a disabled feature has no import cost, no DB/Vault
cost, no event subscribers, and cannot half-run.

`peanutgallery_enabled` **already exists at main** (`config/config.py:88`, honoured at
`factory.py:433`). Incremental value is `subcortical_enabled`, the env loader, factory omission,
orchestrator None-tolerance, and the fallback removal (D-3).

Generalise the hardcoded two-entry dict into a table-driven registry and land first — it is the vehicle
for the rest of the backport:

```
MIRA_SUBCORTICAL_ENABLED    MIRA_PEANUTGALLERY_ENABLED (already honoured)
MIRA_PERSONA_ENABLED        MIRA_USER_MODEL_ENABLED
```

Each maps to an existing factory/`__init__` omission point or two lines to create. Take
`tests/test_cognitive_feature_bypasses.py`. `compose.yaml` volume renames (`mira_*`) and `.env.example`'s
`MIRA_CRM_BASE_URL` are unrelated/CRM — omit. The `orchestrator.py` half is entangled with the WP4
ordered-persistence rewrite — extract by hand.

#### 6.2.4 Item 17 detail — `feedback_tool`

A **new** generic capability, not a replacement. The naming collision is a red herring: crm's deleted
`cns/infrastructure/feedback_repository.py` (130 L) and `feedback_tracker.py` (339 L) belong to the
**user-model** pipeline (removed by `6c055c2`), documented in their own docstrings as *"Part of the user
model pipeline for behavioral assessment"* and *"Feedback synthesis tracking for the user model
pipeline"*. `feedback_tool.py` (`10ca6e3`) captures user friction signals
(feature_request / bug_report / confusion / praise / other) into a new `user_feedback` table. Under D1
both survive; orthogonal.

All deps exist at main: `tools.repo.Tool`, `tools.registry.registry`, `PostgresClient` (`:52`,
`execute_insert` `:185`), `utils/user_context.get_current_user_id` (`:74`). Requires the +23-line
`user_feedback` DDL from `10ca6e3`: table with `category` CHECK, index, RLS policy on
`current_setting('app.current_user_id')` (same pattern as main's other tables), `GRANT INSERT … TO
mira_dbuser`. Compatible with single-user contextvars+RLS.

Edits required: de-brand two `"This helps improve CRM Mira"` strings; rewrite one `usage_example` that
references linking a reminder to a CRM customer via `customer_id`. Otherwise drop-in —
`parallel_safe = True`, single-operation, `additionalProperties: False`.

---

### 6.3 WP3 — multi-user auth

#### 6.3.1 Baseline

No identity system. One shared API key, one hardcoded user row.

```
utils/user_context.py:25,31,37   _user_context: ContextVar[Optional[Dict[str,Any]]], _current_segment_id,
                                 _cancel_event
                                 Accessors: set_current_user_id():65, get_current_user_id():74 (raises
                                 RuntimeError("No user context set…")), set_current_user_data():84
                                 (normalises legacy "id"→"user_id"), clear_user_context():126,
                                 has_user_context():135
                                 _user_context holds an UNTYPED dict. Nothing writes subject_kind,
                                 timezone or first_name.
```

Set in three places, all from `app.state`:

| Path | Location | Behaviour |
|---|---|---|
| HTTP | `auth/api.py:21` `get_current_user()` | compares bearer to `request.app.state.api_key` (`:29`), reads `app.state.single_user_id` (`:32`) and `.user_email` (`:33`), calls both setters, returns `APITokenContext(user_id=…, token_type="api_key", token_id="oss_single_user")` (`:41-45`) |
| WebSocket | `cns/api/websocket_chat.py:185-222` | client sends `{type:'auth', token}`; compares `token != api_key` (`:206`); `user_id = str(single_user_id)` (`:214`); calls **only** `set_current_user_id` (`:221`) — so `get_current_user()` raises on WS-originated work |
| Slack | flushed | — |

`app.state` populated by `main.py:45` `ensure_single_user(app)`, called from `lifespan` at `:217`:
counts `users`; **`if user_count > 1: sys.exit(1)`** (`:66-69`); sets `balance_usd = 999999.00` and
repairs `conversation_llm` (`:77-82`); reads Vault KV `mira/api_keys` → `app.state.api_key` (`:88-91`);
at zero users creates `user@localhost`, a Continuum, two starter messages (`:127-176`), calls
`auth/seed_lora.py:seed_lora_postgres()` (`:177`), mints `api_key = f"mira_{secrets.token_urlsafe(32)}"`
and patches Vault (`:182-197`).

RLS: **session-scoped `set_config`, never `SET LOCAL`, never `SET ROLE`.**

```
clients/postgres_client.py:31-32   class-level shared pools + RLock
                            :53    __init__(database_name, user_id=None, admin=False)
                            :58    _pool_key = f"{database_name}_admin" if admin else database_name
                                   -> admin gets a physically separate pool on mira_admin (BYPASSRLS)
                            :89-95 ConnectionPool(min_size=3, max_size=30, timeout=30,
                                                  max_lifetime=3600, max_idle=300)
                            :133   conn.autocommit = True
                            :138-145 on EVERY checkout, unconditionally:
                                   SELECT set_config('app.current_user_id', %s, false)   -- or '' to clear
                                   third arg false = session-scoped, required because autocommit is on
                                   comment at :139: "ALWAYS set or clear the user context to prevent
                                   inheriting from pooled connections"
utils/database_session_manager.py:66   get_session(user_id) raises ValueError if falsy (:79-80)
                                 :378-395  LTMemorySession._setup_connection → same set_config
                                 :430-436  AdminSession._setup_connection sets NOTHING deliberately
                                           ("let it remain undefined")
```

Pooling safety comes from mandatory reset-on-checkout, not transaction scoping. `user_id` is a
**constructor argument, not contextvar-derived** — the principal latent hazard.

RLS policies (`deploy/mira_service_schema.sql:833-916`): throwing form
`USING (user_id = current_setting('app.current_user_id')::uuid)` — no `, true` second argument. Applied
to `user_activity_days`, `domain_knowledge_blocks`, `domain_knowledge_block_content`, `continuums`,
`messages`, `memories`, `entities`, `feedback_signals`, `feedback_synthesis_tracking`,
`domaindoc_shares` (3 policies). Only `api_tokens:881` and `billing_transactions:896` use the tolerant
form. `:836` states: *"Authentication tables (users, magic_links) do NOT have RLS."*

Four independent single-user locks:

1. `main.py:66` `sys.exit(1)` on `user_count > 1`
2. one process-global bearer secret carrying no user identity
3. `/oss-auth/token` (`cns/api/oss_ui.py:35-45`) returns that key to **any** caller, unauthenticated —
   its own docstring concedes *"The key is already accessible to any local process via Vault."*
   Acceptable on localhost; not with two users.
4. `PostgresClient.user_id` not contextvar-derived, so the 13 bare-`PostgresClient('mira_service')`
   sites work today. Four read `users` (WP1 item 3). Harmless at N=1; cross-user reads at N>1; and the
   moment RLS lands on `users`, `''::uuid` makes them error out.

**The contextvars are not the limitation.** Already per-request-correct: FastAPI/anyio, plus
`contextvars.copy_context()` at `websocket_chat.py:557`, re-copied per tool thread in `tool_loop.py`
(`executor.submit(context.copy().run, …)`) and per agent in `sidebar.py:230`. Scheduled jobs iterate
users calling `set_current_user_id` / `clear_user_context` per iteration (`utils/scheduled_tasks.py`,
`utils/lt_memory_jobs.py`, `segment_collapse_handler.py`). **No change required for multi-user.**

#### 6.3.2 What crm changed

`clients/postgres_client.py` diff across all 162 commits: **one docstring line** (*"cross-user
billing/admin operations"* → *"privileged system operations"*). Mechanism untouched.

Changed instead: policy predicate and call-site discipline.

- Every policy → `NULLIF(current_setting('app.current_user_id', true), '')::uuid` (crm schema
  `:991-1055+`). Unset/empty → NULL → comparison NULL → policy false → **zero rows, fail-closed
  silently** instead of `invalid input syntax for type uuid: ""`.
- RLS extended to `users`, `magic_links`, `api_tokens` (crm schema `:991-1007`).
- Four call sites fixed (`a4df669`, WP1 item 3).
- Pre-auth reads via `mira_admin` BYPASSRLS (`AuthDatabase.create_user`, `get_user_by_id/email`,
  magic-link CRUD, `get_api_token_by_hash`); post-auth per-user reads via `get_session(user_id)`
  (`database.py:463/478/496` — `list_api_tokens`, `revoke_api_token`, `count_user_api_tokens`).
  **Preserve this split verbatim.**

Isolation model: one shared Postgres `mira_service` with RLS. **No per-user databases, no per-user
schemas** — grep confirms no `CREATE DATABASE` per user. Per-user **SQLite** for tool data under
`data/users/<uuid>/` via `UserDataManager`, created lazily by `_ensure_database` and self-migrated with
idempotent `CREATE TABLE IF NOT EXISTS` (`_initialize_tool_schemas`, `:97-110`). Sessions in **Valkey**
(`session:` prefix). Per-user Postgres credentials as encrypted blobs via `UserCredentialService`
(`utils/user_credentials.py:24`) — **already present in OSS**.

`auth/AGENTS.md` invariant: *"Accounts are one user, one Continuum, and one CRM workspace. There is no
team or demo-conversion model."* The CRM-workspace clause does not carry to OSS.

#### 6.3.3 Request resolution

`auth/api.py:179` `get_current_user(request, credentials=Depends(security), auth_service=Depends(get_auth_service))`:

```
1  token, source = Authorization bearer ("header") OR request.cookies["session"] ("cookie")
2  no token                              -> AuthError("UNAUTHORIZED")             -> 401
3  extend_activity = (source != "header")     # API tokens do not slide TTL
4  cookie + unsafe method + not /auth/csrf -> _requires_cookie_csrf (:142)
                                             _validate_cookie_csrf  (:152)
                                             AuthError CSRF_REQUIRED/CSRF_INVALID -> 403
5  auth_service.validate_session(token, extend_activity)
6  failed AND source=="header" -> auth_service.validate_api_token(token)
                                  -> APITokenContext(user_id,"api_token",id,subject_kind,demo_expires_at)
7  still nothing                -> AuthError("INVALID_TOKEN")                     -> 401
8  set_current_user_id(session_data.user_id)
   set_current_user_data(session_data.model_dump())    # <-- RLS context established HERE
9  return session_data
```

Dependency ladder built on it — **steps 1-4 are infrastructure, 5-7 are business policy**:

| Dependency | Line | Adds | Disposition |
|---|---|---|---|
| `get_current_user` | `:179` | identity | **PORT — keep this exact name** |
| `get_current_session_user` | `:262` | rejects `APITokenContext` (403 `SESSION_REQUIRED`) | PORT |
| `get_current_member` | `:274` | `require_member_account()` → rejects `demo` | SKIP (D12) |
| `get_current_member_session` | `:286` | both | SKIP (D12) |
| `_require_billing_entitlement` | `:299-336` | imports `billing` at `:306-307` → 402 | **EXCISE (D7)** |
| `get_current_entitled_user` | `:338` | identity + billing | DROP, or alias to `get_current_user` |
| `get_current_entitled_member` | `:345` | member + billing | DROP |
| `get_current_user_for_pages` | `:1115-1163` | 302 → `/login/` (`:1141,1148,1160`) | DROP or repoint — mira-OSS has no `/login/` page |
| `get_current_entitled_user_for_pages` | `:1165-1178` | billing + 302 → `/settings/#billing` | DROP (D7) |

**Do not port `cns/api/*.py`.** Their entire delta is the `get_current_user` → `get_current_entitled_user`
rename plus CRM domain handlers. Porting them drags in `billing` → `ImportError` at request time. All
7 OSS consumers (`cns/api/{actions,chat,data,files,location,tool_config,trigger_rules}.py`, e.g.
`data.py:435 Depends(get_current_user)`) must keep importing `get_current_user` unchanged.

#### 6.3.4 Mode design

```
MIRA_AUTH_MODE = single (DEFAULT) | dev | multi
```

| | `single` | `dev` | `multi` |
|---|---|---|---|
| `ensure_single_user(app)` | **runs verbatim** (`main.py:45-206`) | replaced by dev-session bootstrap | removed |
| `user_count > 1 → sys.exit(1)` | **kept** (`main.py:66`) | relaxed | removed |
| `/oss-auth/token` | **mounted** | not mounted | not mounted |
| `auth.api.router` @ `/v0/auth` | mounted | mounted | mounted |
| identity source | bearer vs `app.state.api_key` → `single_user_id` | cookie session | cookie session / API token |
| `auth.service` scheduler job | off | on | on |
| email transport | **not required** | not required | **required** |
| `subject_kind` | present, all `'member'` | `'member'` | `'member'` |
| RLS on `users`/`magic_links` | **off** | on | on |
| `web/` page routes | ungated (as today, `main.py:623-666`) | gated | gated |

Implementation: make `get_current_user` the **union** of both behaviours, not a replacement. Prepend to
crm's `:179`:

```python
if single_user_mode_enabled():                     # new auth/mode.py, mirrors dev_mode.py
    if not credentials or credentials.credentials != request.app.state.api_key:
        raise HTTPException(401, "Invalid authentication token")
    session_data = APITokenContext(
        user_id=request.app.state.single_user_id,
        token_type="api_key", token_id="oss_single_user",
        subject_kind="member",                     # required by ported types.py
    )
    set_current_user_id(session_data.user_id)
    set_current_user_data(session_data.model_dump())
    return session_data
# fall through to the full session / API-token ladder unchanged
```

Preserves `auth/api.py:21-45`'s exact contract — same 401 strings, same `token_id`, same contextvar
writes — while making the multi-user path live code in the same function.

Why `single` is the default: no email dependency, no Vault `mira/services` expansion, `web/` unchanged,
`_resolve_user_internal_tier()` (`user_context.py:296-320`) still returns `'cof'` via the `ImportError`
path, and the `''::uuid` hazard never fires because `users` has no RLS in this mode.

`single` → `multi` is a **configuration operation, not a data operation**: `subject_kind` exists in the
greenfield schema from the start. Flip `MIRA_AUTH_MODE`, seed the mailer and `app_url` into Vault. The
`user@localhost` row bootstrapped by `ensure_single_user` is already a valid `member`. Two gaps, both
resolved in code rather than by field defaults: that row has no `first_name`/`last_name`, and
`SessionData.timezone` is **required with no default** in crm's `types.py:44`, so `create_session()`
would emit `timezone: None`. Populate both in `ensure_single_user` (§6.3.7).

`dev` mode is near-free and already works upstream: `development_mode_enabled()` (`dev_mode.py:6`,
`MIRA_DEV`) gates `GET /v0/auth/dev/session` (`api.py:461`, `include_in_schema=False`, 404s unless
`MIRA_DEV` at `:466-467`); `create_development_session()` (`service.py:212`) creates-or-reuses a member
and mints a session; `get_cookie_settings()` (`service.py:637-647`) returns
`secure=not development_mode_enabled()`, so cookies work over plain HTTP on localhost; `samesite="lax"`
(`2ac4660` — Starlette 0.37.2 lowercases for validation, fixed in `daf8e4a`). Every request then
traverses the full stack with one user in the database. Repoint `/dev/session`'s
`RedirectResponse("/workspace/", 303)` (`api.py:473`) to `/chat`. Generalise the hardcoded dev identity
at `service.py:216-231` (`dev@crm-mira.local`, `"Taylor"`, `"Developer"`, `"America/Detroit"`,
`"Develop crm_mira locally"`).

`main.py:99` already reads `MIRA_DEV` inline for hypercorn dev config — consolidate with
`auth/dev_mode.py`, or split the variables. Two readers of one env var is a defect.

Note `demo_policy.py` and `demo_seed.py` are **not** a single-user fallback; they provide a disposable
*second* user with capability restrictions. Skipped per D12.

#### 6.3.5 Graft

Port verbatim (~1,000 lines, zero CRM):

| File | Lines | Notes |
|---|---|---|
| `auth/session.py` | 283 | `_token_digest()` `:32` sha256; `_session_key()` `:36` → `session:<digest>`; `_csrf_key()` `:40` → `csrf:<digest>`; `create_session(user_data, idle_timeout, max_lifetime, extra)` `:43` via `valkey.json_set_with_expiry(key,"$",data,idle)`; `validate_session(token, extend_activity)` `:100` with fail-closed expiry inline (`:118-124`), then max-lifetime, then sliding-TTL extension; `revoke_user_sessions(user_id)` `:170` and `revoke_user_sessions_except(user_id, current)` `:207` (`aa074aa`) — `SCAN` over `session:*`, JSON-read, match, delete key + paired `csrf:<digest>`; `generate_csrf_token` `:242` / `validate_csrf_token` `:262` with `hmac.compare_digest`. No raw token at rest. **Every Valkey primitive required already exists in mira-OSS**: `get_valkey()` `valkey_client.py:406`, `json_set_with_expiry:234`, `json_get:238`, `scan:166`, `ttl:157`, `delete:137`, `set:277`, `increment_with_expiry:175`. Zero porting work. |
| `auth/webauthn_service.py` | 501 | Zero CRM/billing. `py_webauthn` imports `:10-24`, `get_valkey:29`, `AuthDatabase:31`. Needs `config.APP_URL` for `rp_id`/`expected_origin` (`:43-45`). Discoverable-credential ceremony. **Port only the `8608696`-fixed version** — it corrected hex vs base64url credential-ID keys. |
| `auth/exceptions.py` | 15 | `AuthError(code, message, details)` |
| `auth/dev_mode.py` | 8 | `development_mode_enabled()` |
| `auth/rate_limiter.py` | 105 | `RateLimiter:14`, `is_allowed:30` (5/email/5min, 10/IP/5min), `reset:76`. Uses `config.RATE_LIMIT_*` only. |
| `auth/security_logger.py` | 83 | `SecurityLogger:12`, `_sanitize_email:19`, `log_event:37` |
| `auth/types.py` delta | +16 | `SubjectKind = Literal["member","demo"]`; `subject_kind`/`demo_expires_at` on `UserRecord`, `UserProfile`, `SessionData`, `APITokenContext`; `SessionData` gains `first_name`/`last_name`/`timezone`; `CookieSettings.samesite` lowercased. Take crm's field requirements as written — no compatibility defaults; update OSS's construction sites. `auth/api.py:41-45` is replaced by this port, so the 3-arg `APITokenContext` disappears with it. |
| `auth/security_middleware.py:27-35` | — | Headers only. Free. CSP at `:38-53` per D-1. |

Omit: `auth/crm_workspace.py` (233 L, imports `_crm_client.CRMLifecycleClient:12` and `billing:203`),
`auth/demo_seed.py` (128 L, imports `_crm_client.client_for_workspace:8`), `auth/oauth.py` (443 L,
Square, imports `clients.square_client:30` — D-16), `auth/demo_policy.py` (59 L, `_BLOCKED_TOOLS` =
`{email_tool, pager_tool, homeassistant_tool}` and `_BLOCKED_OPERATIONS` =
`{jobs_tool:{schedule_message}, billing_tool:{send_invoice}}` — CRM tool names; reachable only via
`tools/repo.py:474`), `cns/api/demo.py` (80 L), `auth/config.py:64,68` (`CRM_BASE_URL`,
`CRM_LIFECYCLE_SERVICE_SECRET`).

CRM coupling by file — **the layer is ~85% generic by line count**; the dependency is a provisioning
hook, not an architectural assumption:

| File | Coupling |
|---|---|
| `crm_workspace.py`, `demo_seed.py`, `oauth.py` | 100% |
| `config.py` | 2 of ~9 fields |
| `service.py` | 2 touch points — `:26` import, `:41` attr; plus `:132-140` and `:187-194` compensation blocks, `:243` `ensure_workspace`, `:283`/`:285-289` in `_initialize_account` |
| `account_gc.py` | 1 import (`:11`) + `LEFT JOIN crm_workspaces` (`:64`) + `lifecycle_state='cleanup_pending'` branch (`:71`) + `lifecycle.delete_account(...)` (`:81`) |
| `api.py` | 0 CRM, 1 billing (`:306-307`) |
| `session.py`, `database.py`, `rate_limiter.py`, `security_logger.py`, `exceptions.py`, `types.py`, `dev_mode.py`, `webauthn_service.py`, `email_service.py` | **0** |

**The seam is one method** — `AuthService._initialize_account()` (`service.py:267-289`):

```python
self.db.initialize_mira_account(user_id, first_name, current_focus)   # GENERIC  :281  keep
self.crm_workspace_service.provision_workspace(user_id, timezone)     # CRM      :283  cut
if seed_demo: … from auth.demo_seed import seed_demo_crm; seed_demo_crm()  # CRM :285-289 cut
```

Introduce and inject:

```python
# auth/provisioning.py  (new, ~20 L)
class AccountProvisioner(Protocol):
    def provision(self, user_id: str, timezone: str) -> None: ...
    def ensure(self, user_id: str, timezone: str) -> None: ...
    def delete(self, user_id: str) -> bool: ...

class NullProvisioner:                       # OSS default
    def provision(self, *a, **k): pass
    def ensure(self, *a, **k): pass
    def delete(self, user_id) -> bool: return local_teardown(user_id)
```

`local_teardown` = revoke sessions → `clear_manager_cache` → `rmtree(data/users/<id>)` →
`DELETE FROM users`. This CRM-free tail already exists at `crm_workspace.py:206-232`.
`AuthService.__init__` takes `provisioner: AccountProvisioner = NullProvisioner()`. One substitution
removes `crm_workspace.py`, `demo_seed.py` and the CRM branch of `account_gc.py` without touching other
auth logic.

Justification under the one-way-salvage decision: this protocol is **the minimal excision mechanism**,
not a convergence investment. The alternative is editing `service.py` inline at four sites (`:26`, `:41`,
`:132-140`, `:187-194`, `:243`, `:267-289`) and carrying a forked copy forever. Twenty lines of
protocol is cheaper than six surgical edits plus permanent divergence, and it costs nothing at runtime.
Do **not** extend the pattern elsewhere in the name of future CRM parity — the repos are parting.

`account_gc.py` (152 L) is worth porting: deleting never-activated signups after 24 h is generic hygiene
and directly enables multi-user signup. `cleanup_unactivated_accounts:35`,
`register_account_gc_job:127` (`0 3 * * *`).

`auth/database.py` (505 L): mostly clean. `initialize_mira_account()` `:107` →
`get_continuum_repository().create_continuum(user_id)` + `_prepopulate_welcome_content()` (`:235-361`,
~126 L) — the proper replacement for `main.py:127-176`'s inline seeding; port it. Two problems:
`:124-126` comment references the billing `users_provision_member_entitlement` trigger; and **the
`create_user` INSERT (`:64-84`) writes `subject_kind`/`demo_start_at`/`demo_expires_at` and omits
`conversation_llm` and `balance_usd`** — the former dies under D4, the latter is still written by
`ensure_single_user` (`main.py:77-82`). `_USER_RECORD_COLUMNS:127-131` likewise. Direct port = broken
INSERT against the OSS schema. Reconcile. `_prepopulate_welcome_content` vs main's seeding: main also
calls `seed_lora_postgres()` (`main.py:177`) — retained under D1, so keep that call and the
`feedback_synthesis_tracking` init. **Not compared line-by-line**; `increment_segment_turn()` depends on
`segment_turn_count` being present. Verify.

`auth/service.py` (678 L) excision points: `:26`, `:41`, `:132-140`, `:187-194`, `:243`, `:267-289`.
Port `request_magic_link` (`:299`, with `:324-341` enumeration defence + `random.gauss` timing jitter),
`verify_magic_link:401`, `create_api_token:468` (`:488` 50-token cap), `validate_api_token:518`,
`create_session:552`, `logout_other_devices:607`, `cleanup_expired_tokens:631`,
`get_cookie_settings:637-647`, `register_cleanup_jobs:657`.
**`:676` module-level `auth_service = AuthService()` singleton** constructs `AuthConfig()` →
`get_database_url()` + 5× `get_service_config()` → **Vault reads at import time**. Combined with
`utils/scheduled_tasks.py` registering `('auth.service','auth_service',False,None)`, any
`import auth.service` requires a reachable, fully seeded Vault. Prefer lazy `get_auth_service()`
(`api.py:137`) and mode-gate the scheduler registration.

`auth/api.py` (1,178 L) region map:

| Region | Lines | Coupling | Action |
|---|---|---|---|
| imports, router, `SignupRequest`/`MagicLinkRequest`/`MagicLinkVerifyRequest` | 1-70 | none | keep |
| `AuthenticationError`, `RateLimitError`, `create_auth_success_response`, `create_auth_error_response` | 73-133 | none | keep |
| `get_auth_service`, `_requires_cookie_csrf`, `_validate_cookie_csrf`, `_auth_http_exception` | 137-176 | none | keep |
| `get_current_user` | 179-260 | none | **keep verbatim + single-user union branch** |
| `get_current_session_user` | 262-272 | none | keep |
| `get_current_member` / `_session` | 274-297 | `demo_policy` | skip (D12) |
| `_require_billing_entitlement` | 299-336 | `billing` at `:306-307` | **excise (D7)** |
| `get_current_entitled_*` | 338-350 | billing | drop |
| API-token CRUD `/api-tokens` | 353-458 | none | keep |
| `/dev/session` | 461-482 | `:473` → `/workspace/` | repoint to `/chat` |
| `/signup`, `/magic-link`, `/verify`, `/logout`, `/logout-all`, `/logout-others`, `/session`, `/csrf` | 484-793 | none | keep |
| WebAuthn block | 795-1112 | needs `webauthn` pkg | keep |
| `get_current_user_for_pages` | 1115-1163 | `:1141,1148,1160` → `/login/` | drop or repoint — no `/login/` page in mira-OSS |
| `get_current_entitled_user_for_pages` | 1165-1178 | billing | drop |

`APITokenContext` shape: crm makes `subject_kind` **required**. OSS's current `auth/api.py:41-45`
constructs it with three args, but that file is replaced by this port and the single-user union branch
(§6.3.4) supplies `subject_kind="member"` explicitly. No default is added. Same for
`SessionData.timezone` (crm `types.py:44`, required, no default): `ensure_single_user` populates
`users.timezone` for the bootstrapped row.

Take crm's `cns/api/base.py` (+4 L: `http_status: NotRequired[int]` on `ErrorDetail` and
`ResponseMeta`). `auth/api.py:107-133` does `dataclasses.replace` on a **frozen** `ErrorResponse` and
depends on `SuccessResponse`, `ErrorResponse`, `create_success_response`, `create_error_response`,
`generate_request_id` and `APIError`. Verify frozen-ness and helper signatures on landing.

#### 6.3.6 Blocker: email transport

`auth/email_service.py` (165 L) is generic code but a client for a **proprietary HMAC-signed HTTP email
gateway** (`config.EMAIL_GATEWAY_URL` / `API_KEY` / `HMAC_SECRET`), payload `{email, token, app_url}`,
via `requests` (`:9`). Server side is the **untracked** `mira_email_gateway_forbiz.php`. No PyPI package
substitutes. D3 requires a pluggable SMTP sender — **new work crm_mira never needed**, and it gates
whether `multi` is shippable.

`AuthConfig.__init__` (`config.py:11-24`) reads `email_gateway_url`/`api_key`/`hmac_secret`,
`valkey_url`, `app_url` **eagerly, at import**, via `get_service_config` (exists at OSS
`vault_client.py:184`; `preload_secrets()` raises on any failure). OSS's `deploy/postgresql.sh:166-176`
seeds only `valkey_url`, `userdata_encryption_key`, `diagnostics_token`.
**Action:** seed `app_url` in `deploy/postgresql.sh` and `deploy/docker/scripts/init-mira.sh:235`
unconditionally; make the email fields lazy so `single`/`dev` do not require them; otherwise every
startup dies in every mode.

Dependencies for WP3:

| Package | Needed for | Weight | License | Verdict |
|---|---|---|---|---|
| `email-validator` | Pydantic `EmailStr` on `SignupRequest.email`, `MagicLinkRequest.email` (`api.py:49,63`) | tiny (pure Python + `idna`) | MIT | required for `multi` |
| `webauthn` (py_webauthn) | passkeys | small; pulls `cryptography` (**present**), `cbor2`, `pydantic` (present) | BSD-3 | optional; port with `webauthn_service.py` |
| `requests` | `email_service.py` | already transitive via `hvac` | Apache-2.0 | rewrite on existing `httpx` when the SMTP sender lands |
| `jsonschema` | WP1 item 11 | small | MIT | required |
| `pypdf>=5.0.0` | D10 local document extraction | medium | BSD-3 | required by D10 |

No heavyweight or licence-problematic additions; `torch`/`sentence-transformers` already dominate the
install. Omit `stripe`, `twilio`, `pywebpush`, `Markdown`/`html5lib`/`tinycss2` (the last three arrive
only with D-4).

#### 6.3.7 Schema — greenfield

**Adopt crm's greenfield posture.** `8a46231`'s own message: *"The fresh-install schema is rewritten
from scratch as a pure DDL contract requiring an empty target database"* and *"All 10 incremental
migration SQL files are deleted (superseded by the greenfield schema)."*

Method: derive `deploy/mira_service_schema.sql` **from crm's file**, not from OSS's cumulative one.
Start from crm's structure — the empty-database `DO` guard, `CREATE EXTENSION pgcrypto/vector/pg_trgm`,
`GRANT USAGE ON SCHEMA public TO mira_dbuser`, the `set_updated_at()` and `set_search_vector()` trigger
functions, roles provisioned externally via Vault-backed tooling rather than embedded — then strip every
CRM/billing object listed below and add back the OSS-retained objects crm dropped.

crm's schema is the **base**; OSS's current schema is the source of the add-back list.

**Add back** (crm dropped these; OSS retains them by decision):

| Object | Basis |
|---|---|
| `feedback_signals` — OSS column set (`signal_type`, `section_id`, `strength`, `synthesized`) | D1 retains the user model |
| `feedback_synthesis_tracking` | D1 — `last_synthesis_output` holds the user-model XML |
| `usage_pricing` | D5 — cost visibility |
| `domain_knowledge_blocks` + `domain_knowledge_block_content` | OSS feature, not CRM |
| `internal_llm` / `conversation_llm` | **Do NOT add back** — D4 replaces both with `model_configs` |
| `extraction_batches` / `post_processing_batches` | **Do NOT add back** — D10 removes the Batch API |
| `users_trash` (OSS main `:236`, with the `GRANT` block at `:806`) | **Do NOT add back** — crm's soft-delete columns on `users` serve account GC instead |

**Model on crm, with these deltas:**

```
magic_links                          crm :125-135 — OSS main :255-265 is byte-compatible. Take crm's.
api_tokens                           crm :137-153 — adds idx_api_tokens_hash_active,
                                     idx_api_tokens_user_active, changes the unique index. Take crm's.
users.webauthn_credentials JSONB     DEFAULT '{}' — present in both
users.timezone / first_name / last_name / is_active / last_login_at   present in both
users.subject_kind                   TEXT NOT NULL DEFAULT 'member' CHECK IN ('member','demo')  [D12]
users.deletion_requested_at / soft_deleted_at / purge_deadline   take crm's; they replace users_trash
users.conversation_llm / balance_usd  OMIT — D4 and D7. Consequence: ensure_single_user's
                                     `UPDATE users SET balance_usd = 999999.00, conversation_llm = …`
                                     at main.py:77-82 must be REWRITTEN, not preserved. The
                                     oss_default_tier / offline-tier detection at main.py:59-64 goes
                                     with it; offline installs are now expressed by seeding
                                     model_configs with local endpoints (§6.1.7).
roles mira_admin (BYPASSRLS) / mira_dbuser   OSS main :42-93, identical in both. crm's greenfield
                                     delegates provisioning to deploy tooling — adopt that, and
                                     confirm deploy/postgresql.sh still creates both roles.
per-user SQLite via UserDataManager  identical mechanism, unchanged
```

**RLS — author the fail-closed form directly, no migration.** Every policy in the new file uses:

```sql
NULLIF(current_setting('app.current_user_id', true), '')::uuid
```

rather than the throwing `current_setting('app.current_user_id')::uuid` that OSS main uses in 13 places
(`mira_service_schema.sql:843-916`; only `api_tokens:881` and `billing_transactions:896` are tolerant
today). Enable RLS on `users`, `magic_links` and `api_tokens` as crm does (`:991-1007`) — OSS main
explicitly excludes the first two at `:836` (*"Authentication tables (users, magic_links) do NOT have
RLS"*), and that exclusion is what makes multi-user unsafe.

**Ordering constraint survives the 2.0 posture:** WP1 item 3 (`a4df669`'s four call-site lines at
`utils/user_context.py:378,420` and `cns/services/portrait_service.py:138,352`) must land **before**
RLS is enabled on `users`. Those four sites issue bare `PostgresClient('mira_service')`, including an
unscoped `UPDATE users`. With RLS on and no user context, they return zero rows and silently stop
working. This is a correctness gate, not a compatibility one.

Also fold in from §6.3.8, as schema objects rather than migrations: `global_memories_runtime` +
`can_read_global_memories()` + the grant tightening.

**Do not include** `users_subject_contract` (hardcodes `email ~ '^demo\+[0-9a-fA-F-]{36}@no\.email\.add$'`
and `demo_expires_at = demo_start_at + INTERVAL '24 hours'`), `users_lifecycle_contract`,
`crm_workspaces` (`:687-701`), `enforce_member_global_username()` (`:704-731`, `:727-737`),
`entitlements` / `stripe_customers` / `billing_renewal_attempts` / `stripe_webhook_events` (`:187-368`),
`provision_member_entitlement()` + `add_calendar_month()` + the `users_provision_member_entitlement`
trigger (`:329-368`), `is_active_member()` / `resolve_active_member_by_email()` /
`active_member_identity()` SECURITY DEFINER functions (`:640-684` — all three filter
`subject_kind='member'`), `audit_events`, and all `sms_*` / `workphone_*` / `business_voice_*` /
`push_subscriptions` / `conversation_outcomes` / `voice_feedback_signals` /
`autonomy_assessment_sft_corpus` (`:539-986`).

`user_feedback` **is** included — it arrives via WP1 item 17 (`feedback_tool`, +23 L DDL: table with
`category` CHECK, index, RLS policy, `GRANT INSERT … TO mira_dbuser`).

`persona_revisions`, `persona_state` and **`persona_signals`** **are** included — D1 ports Persona.
See §6.5.3. crm's `provision_baseline_persona()` AFTER INSERT ON `users` trigger (`:594-617`) is
included and also provisions the baseline for users created by `ensure_single_user` in `single` mode.

**`enforce_member_global_username()` must not be included.** OSS's `global_usernames` (`:920-936`)
deliberately has no RLS and is queried contextlessly by `mira_resolve_username`; the trigger requires
matching user context and would break the federation resolver.

**Delete the whole of `deploy/migrations/`.** Verified by name-set comparison: main has 19 files,
crm deleted **11** (10 migrations plus `llm_dialect_split.rollback.sql`) and added 8, leaving 16 —
**8 legacy** (`003_entities_drop_embedding_add_trgm`, `add_last_tended_at`,
`add_messages_is_error_column`, `add_source_segment_id`, `drop_consolidation_rejection_count`,
`mira_blog_hardening`, `neutral_message_format`, `prune_extraction_ref_links`) plus **8 CRM**
(`add_autonomous_revised_signal`, `add_business_voice_learning`, `add_push_subscriptions`,
`add_sms_autonomous`, `add_sms_tables`, `add_sms_thread_metadata`, `add_voice_model_configs`,
`add_workphone_module`). The CRM 8 duplicate crm's own greenfield schema; the legacy 8 are residue of
the same half-measure. OSS 2.0 prunes the directory; the greenfield file is the single source of truth.

Disposition by group, for the record:

| Migration group | 2.0 disposition |
|---|---|
| `security_hardening_20260504.sql` | DELETE — its subject tables (`stripe_webhook_events`, `billing_transactions`) are omitted under D7 |
| `llm_dialect_split.sql` (+ `.rollback.sql`), `llm_adapter_name.sql`, `drop_dialect_options.sql`, `cache_ttl_and_adapter_cleanup.sql` | DELETE — `model_configs` is authored directly into the greenfield schema. (crm already dropped these; OSS declines nothing here, it simply does not re-add them.) |
| `default_pricing_fallback.sql`, `tier_qualified_pricing_keys.sql` | DELETE — `usage_pricing` is authored directly (D5) |
| `prune_dead_curation_llm_rows.sql`, `phoneafriend_internal_llms.sql` | DELETE — `internal_llm` no longer exists; phone-a-friend routes to `other` (§6.1.3) |
| `drop_post_processing_batches.sql` | DELETE — batch tables are simply absent (D10) |
| `neutral_message_format.sql` | DELETE. Its Step 3a is an `UPDATE messages SET content = …` converting existing rows from Anthropic wire format; a greenfield database has none and the schema stores neutral format from the start. The jsonb precedence bug at `:52` is real but unreachable (§6.2 item 1). |
| crm's 8 `add_*` files | DELETE — all CRM (sms ×3, workphone, business_voice, push, voice model configs, autonomous revised signal) |
| crm's 8 retained legacy files | DELETE — residue. All are folded into crm's own greenfield schema and serve no fresh install |

**Behaviour to instrument, unchanged by the 2.0 posture.** The `NULLIF` predicate yields a **silent
zero-row result** where OSS main currently raises `invalid input syntax for type uuid: ""`. Correct for
security, wrong for debuggability, and it will mask graft regressions. Add a startup assertion or canary
query that logs when `app.current_user_id` is empty on an RLS-covered table.

**`a9fd443`** deletes `tools/schema_distribution.py` and the `initialize_user_database` call. Its root
cause was crm-specific (`tools/implementations/schemas/` went empty when `contacts_tool.sql` was
removed, so provisioning raised `RuntimeError`), but the underlying simplification is correct on its own
terms: `UserDataManager._ensure_database` self-initialises tool schemas idempotently on
every connection (`:97-110`), making file distribution redundant, and `tools/schema_distribution.py` is
already dead code at main (`git grep "schema_distribution\|initialize_user_database"` excluding the file
itself → empty). **Port the deletion.** Keep `contacts_tool.sql` — `_init_contacts_schema()`
(`userdata_manager.py:388`, called at `:123`) is the live path and contacts_tool is retained (§7.3).
Note WP1 item 8 takes only 3 lines of `f8cfea9` from this same file precisely to avoid deleting
`_init_contacts_schema`.

#### 6.3.8 Multi-user hardening, portable independently

**`global_memories_runtime`.** At main, `global_memories` has **no RLS** (`:577` comment: *"no RLS, no
decay. Manually curated via psql"*) but a full CRUD grant (`:614`
`GRANT SELECT, INSERT, UPDATE, DELETE … TO mira_dbuser`). At N=1 acceptable; at N>1 **any authenticated
user's request path can write the shared table every other user reads** — a cross-user prompt-injection
vector with persistent reach.

```sql
CREATE FUNCTION can_read_global_memories() RETURNS BOOLEAN STABLE SECURITY DEFINER
  SET search_path = pg_catalog, public LANGUAGE sql AS $function$
    SELECT EXISTS (SELECT 1 FROM public.users
      WHERE id = NULLIF(current_setting('app.current_user_id', true), '')::uuid
        AND is_active = TRUE)
$function$;
CREATE VIEW global_memories_runtime WITH (security_barrier = true)
  AS SELECT * FROM global_memories WHERE can_read_global_memories();
GRANT SELECT ON global_memories_runtime TO mira_dbuser;                  -- crm :1194
GRANT EXECUTE ON FUNCTION can_read_global_memories() TO mira_dbuser;     -- crm :1216
REVOKE EXECUTE ON FUNCTION can_read_global_memories() FROM PUBLIC;       -- crm :1223
```

`security_barrier = true` prevents the planner pushing user predicates past the gate. Python companion:
`lt_memory/hybrid_search.py:169` `FROM global_memories gm` → `FROM global_memories_runtime gm`.
**Verify `lt_memory/db_access.py:507`** (the other `global_memories` reader) is switched too — crm's
`db_access.py` shrank 261 lines and that hunk was not isolated.

`get_active_segments()` (`cns/infrastructure/continuum_repository.py`, crm `:1091-1114`): take
**`AND users.is_active = TRUE`** now — main scans deactivated users' active segments. Requires column
qualification (`messages.id`, `messages.continuum_id`, …) because the JOIN makes them ambiguous. Skip
the `LEFT JOIN entitlements` half (D7).

`agents/sidebar.py:292-305` `_get_eligible_users()` and `utils/scheduled_tasks.py:164-183`
`get_users_due_for_job()`: main already has `is_active = TRUE`; crm's only addition is the
`entitlements` JOIN. **Nothing to take.**

`utils/domaindoc_shares.py:131` `JOIN users u` → `JOIN LATERAL active_member_identity(ds.owner_user_id) u ON TRUE`
requires the omitted SECURITY DEFINER function. **Not portable** under D12.

`utils/scheduled_tasks.py`: re-add `('auth.service','auth_service',False,None)` (`92d768c`), mode-gated.
Do **not** take the `get_users_due_for_job()` rewrite.

`utils/power_on_self_test.py:454-500` (`92d768c`) validates nine Vault service-config fields — do not
port verbatim; it requires `crm_base_url`, `crm_lifecycle_service_secret`, `square_application_id/secret`,
`stripe_key`/`stripe_publishable_key`/`stripe_webhook_secret`, `email_gateway_*`. Port the **shape**
(required-field list + URL-scheme validation) with an OSS-appropriate list.

`cns/api/federation.py:83` (`fb7065f`) adds `AuthDatabase().get_user_by_id(...)`, rejects
`subject_kind == "demo"`, then `from billing import get_billing_backend; has_product_access(...)`.
**Omit the gating entirely.** Under D7+D12 it would raise `ImportError` inside the endpoint → 500 on
every lattice delivery. mira-OSS's endpoint (`fff65a4`) already has the `X-Lattice-Delivery-Token`
Vault check and `sender_verified` requirement. If member gating is wanted later, re-add without the
billing check.

`cns/integration/event_bus.py`: `publish(self, event: ContinuumEvent)` → `publish(self, event: object)`
plus a docstring note. Dispatch was already structural on `event.__class__.__name__`. **Functionally a
no-op** — take or skip.

---

### 6.4 WP4 — WebSocket protocol

#### 6.4.1 Defects at main

`cns/api/websocket_chat.py`, 898 lines. No schema, no validation. Auth at `:181-224`; message loop
`handle_connection()` does a bare `await websocket.receive_json()` in `while True`, dispatching on
`message_data.get("type")` for `ping`/`message` only (`:785-795`).

| # | Defect | Location |
|---|---|---|
| 1 | **Two concurrent readers on one socket.** `handle_connection()` awaits `process_message_streaming()`, which spawns `cancel_listener()` as a task that also calls `websocket.receive_json()` in a loop. Any frame the listener reads that is not `cancel` is **silently discarded** — including a second `message`. | `:405-420` |
| 2 | **No turn identity.** Nothing correlates a frame to a message; client `_getActiveMessageCallback()` is FIFO-first-in-map, so server-initiated frames are attributed to whichever callback is oldest. | — |
| 3 | **Pre-tool text persisted twice.** `_build_tool_history_messages()` iterates `acc.tool_call_results` calling `assistant_message_from_result(result)`, which includes the full `result.text` (`tool_messages.py:30-31`); the final assistant message is built from `acc.response_text`, the concatenation of all steps. | `:307-433` |
| 4 | **Synthetic marker baked into durable content.** `_format_tool_indicator(acc.events)` prepends `[used: tool_a, tool_b]` into `acc.response_text`, which is persisted and **re-fed to the model every later turn**. | `:1097-1099`, `:296-305` |
| 5 | **Scaffold prompt rendered as a user bubble.** The recursive auto-continuation runs the normal `persist_user_msg` path with `metadata={}`, so `<system-scaffold>The requested tool is now loaded. Continue with the original task.</system-scaffold>` is persisted as a real user message. `get_history(message_type="regular")` filters only `metadata->>'system_notification' != 'true'`, so it is returned to the browser. **User-visible today.** | `:1246-1262`; `continuum_repository.py:626` |
| 6 | **`TurnCompletedEvent` published before commit**, and **twice** on tool-loader auto-continuation — the recursive `process_message()` reuses the same unit of work, so both invocations reach `:1180`. Subscribers (Peanut Gallery, portrait, trinket invalidation) can observe a turn that then fails to persist. | `:1180-1185` vs commit in the WS handler |
| 7 | **Failure path re-mints the user message** with a fresh UUID and `created_at`, so the client's optimistic id never reconciles and the row can sort after later messages. | ~`:680`, ~`:700` |
| 8 | WS auth calls only `set_current_user_id` (`:221`), never `set_current_user_data`, so `get_current_user()` raises on WS-originated work. | `:221` |

#### 6.4.2 What crm provides

778 lines. Strict Pydantic v2 both directions:

```
ProtocolModel(BaseModel)  model_config = ConfigDict(extra="forbid")            :47
ClientFrame = Annotated[AuthFrame | MessageFrame | HaltFrame | PingFrame,
                        Field(discriminator="type")]                            :91
CLIENT_FRAME_ADAPTER = TypeAdapter(ClientFrame)                                 :92
  AuthFrame{type:Literal["auth"]}                                               :51
  MessageFrame{type:"message", message_id: UUID,
               content: str = Field(min_length=1, max_length=100_000),
               include_thinking: bool=False, image, image_type, document, document_type}
               + @model_validator(mode="after") requiring image/image_type and
                 document/document_type to arrive in pairs and forbidding both  :56,:69
  HaltFrame{type:"halt", turn_id: UUID}                                         :81
  PingFrame{type:"ping"}                                                        :87
ServerFrame union :182, SERVER_FRAME_ADAPTER :213
  AuthSuccessFrame{user_id}:95 · ServerShutdownFrame:109 · PongFrame:114
  ProtocolErrorFrame{code,message}:118 · TurnStartedFrame{turn_id,message_id,segment_id}:124
  AssistantDeltaFrame{turn_id,segment_id,entry_id,content}:131
  ToolFrame{turn_id,segment_id,event:Literal['tool_detected','tool_executing','tool_completed',
            'tool_error'],tool_name,tool_id,arguments,result,is_error}:139
            + validator tying payload to lifecycle event :150
  TurnCompleteFrame{turn_id,segment_id}:161
  TurnStoppedFrame{turn_id,segment_id,reason:Literal['halt','disconnect']}:167
  TurnErrorFrame{turn_id,segment_id,code,message}:175
  AccountAccessRequiredFrame:101          <-- EXCISE (D7, R8)
validate_client_frame:236  validate_server_frame:241  get_friendly_error_message:246
```

`ChatConnection:262` — one `_read_frames()` task (`:300`) feeding
`inbound: asyncio.Queue[ClientFrame | ClientDisconnected](maxsize=32)`, one `_write_frames()` task
(`:319`) draining `outbound: asyncio.Queue[ServerFrame | StopWriter](maxsize=128)`. `ClientDisconnected:216`
and `StopWriter:220` sentinels. `accepts_output` flips false in the writer's `finally` so late sends
drop rather than raise. `send()` (`:281`) runs `validate_server_frame(frame)` **before** enqueueing, so
a malformed outbound frame raises server-side instead of shipping garbage. `drain():285`, `close():288`.
`close_all_connections():340` sends `server_shutdown`, drains with a 2 s timeout, closes.

Ordered persistence (`c1297b3` + `457a56e`), in `cns/services/orchestrator.py`:

```
AssistantStep(entry_id: UUID = uuid4(), text, result, partial)              :218-226
TurnAccumulator                                                             :228-296
  append_text(content) -> UUID   creates/extends current_step, returns stable entry_id   :249
  finish_step(result)            on ModelStepCompletedEvent / CompleteEvent              :257
  finish_partial_step()          on halt                                                 :265
  finish_text_step()             for synthetic fallback text                             :275
  reset()                                                                                :284
_build_turn_messages()  :368-490   walks acc.assistant_steps, emits assistant+tool messages
                                   in exact provider-step order with monotonic microsecond
                                   offsets from base_time = user_msg_obj.created_at, using
                                   replace(result, text=step.text) so each assistant message
                                   carries ONLY that step's text
process_message(..., message_id, turn_id, _internal_continuation)            :1005-1020
  user message staged via unit_of_work.add_messages(persist_user_msg) immediately after
  add_user_message(..., message_id=message_id, metadata={"turn_id":…,"segment_id":…})   :1037-1090
  WS handler's _process_with_orchestrator except-block commits
  unit_of_work.pending_messages before re-raising                            :741
entry_id on every delta                                                      :932-941
GenerationCancelled caught INSIDE the stream loop -> stopped=True,
  acc.finish_partial_step(), metadata["stopped"]=True,
  metadata["stop_reason"]=get_cancel_reason() -> terminal turn_stopped       :1170-1177
per-step tag normalisation: tag_parser.parse_response(step.text,
  preserve_tags=['my_emotion'])['clean_text'] over EVERY step                :1204-1209
blank-response guard gated on `not stopped`                                  :1231
metadata.stop_reason renamed provider_stop_reason so stop_reason can mean
  halt/disconnect                                                            :1261
TurnCompletedEvent moved to a POST-COMMIT callback, suppressed when stopped or
  auto_continuing, durable_message_count = len(continuum.messages) - int(_internal_continuation)
                                                                             :1305-1322
scaffold discarded via Continuum.discard_transient_user_message(synthetic_message_id) in a
  finally; refuses to delete anything lacking metadata.transient_system_scaffold is True
                                                                             :1329-1351
_format_tool_indicator DELETED
```

Supporting: `cns/core/message.py:131` `transient_system_scaffold`, `:140` `tool_arguments`, `:158-161`
`turn_id`/`partial_response`/`stop_reason`/`provider_stop_reason`; `cns/core/continuum.py:70-113`
`add_user_message(*, message_id, metadata)` + `discard_transient_user_message`.
`UnitOfWork.pending_messages` and `add_post_commit_callback` **already exist at main**
(`cns/infrastructure/continuum_pool.py:41,86`), and `Message` is a dataclass with `id`/`created_at`/
`metadata` as constructor params (`cns/core/message.py:170-182`) — so `_build_turn_messages`'s
`Message(..., created_at=…)` ports without main's `object.__setattr__` hack (`orchestrator.py:371,428`).

Halt: `utils/user_context.py` gains `set_cancel_reason(reason: Literal["halt","disconnect"])` /
`get_cancel_reason()` (21 additive lines, no dependencies), stashing the reason as an attribute on the
shared `threading.Event` (`event.mira_stop_reason`) and raising if no event is active. `_dispatch` on a
`HaltFrame` (`:477`) verifies `connection.active_turn_id == frame.turn_id` (else `protocol_error
NO_MATCHING_ACTIVE_TURN`), then `set_cancel_reason("halt"); cancel_event.set()`. Disconnect sets
`"disconnect"`. `tool_loop.check_cancelled()` at `:127,140,145,166` — between sequential tools, after
the sequential batch, before the parallel `ThreadPoolExecutor`, per-candidate inside the submit loop (so
later parallel calls never start), and after the pool drains.

#### 6.4.3 Compatibility cost of D2

**Verified first-hand at source** against `api-client.js`, `messaging.js`, `history.js` at main and the
Pydantic models at `crm_mira/crm_mira:cns/api/websocket_chat.py`. Every break below is confirmed; D2's
~150-line patch budget is sized against verified behaviour, not inference.

Verification detail worth recording:

- `_generateId()` is at `api-client.js:775-777` and returns
  `` `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}` `` — definitively not a UUID.
- The message frame is built at `:294-302`. Its own inline comment on the `stream` field reads
  *"Always stream on server; ignore client flag"* — so the server already ignores it, and the break is
  purely `extra="forbid"`, not a semantic disagreement.
- `await AppState.apiClient.chat.sendMessage(…)` is at `messaging.js:1577`, with `setGenerating(false)`
  at `:1579` — two lines apart, confirming the stuck-stop-button consequence.
- `window.extractEmotionEmoji` is defined at `messaging.js:888`.
- crm's `ClientFrame` union is at `websocket_chat.py:103` and has exactly four members.
- `TurnCompleteFrame` (`:175-179`) has exactly three fields: `type`, `turn_id`, `segment_id`.
- `_server_frame_for_event` (`:673-703`) ends in a bare `return None`.
- crm's rejection is at `data.py:117` (`"offset is not supported for history; use before"`) and `:119`
  (`search`). `:176` rejects offset for CRM customers — omitted with the CRM tooling.

Traced against `web/assets/javascript/api-client.js` (905 L), `messaging.js` (1784 L), `events.js`
(279 L), `history.js` (1142 L), `core.js`, `web/settings/index.html`.

**Inbound — 4 breaks, 3 total message rejection:**

| Old frontend sends | New protocol | Result |
|---|---|---|
| `{type:'auth', token: this.token \|\| ''}` — `api-client.js:504-507` from `_handleOpen()`; token from `GET /oss-auth/token` (`oss_ui.py:34`) memoised in `_ensureOssToken()` (`:780-800`) | `AuthFrame{type:Literal['auth']}`, `extra="forbid"`; `authenticate()` (`:363`) reads **only** `websocket.cookies.get("session")` (`:366`) | `token` is a forbidden extra → `PydanticValidationError` → `protocol_error{code:'MALFORMED_FRAME'}`; and no `session` cookie exists in an OSS single-user build → `WebSocketAuthError("Missing authentication session")` → `AUTH_FAILED`. `app.state.api_key` / `single_user_id` are never consulted. |
| `{type:'message', content, stream:true, id:'msg_'+Date.now()+'_'+rand, include_thinking:true, image?, image_type?, document?, document_type?}` — `:293-311`; `_generateId()` `:773` | `MessageFrame` requires `message_id: UUID`, forbids extras | **3 independent failures:** `stream` extra, `id` extra, `message_id` missing. And `msg_1735689600000_k3j9x2abc` is not a UUID. **Every message rejected.** |
| `{type:'cancel'}` — `:461-465` `cancelGeneration()`, wired to the stop button at `events.js:133-135` and `messaging.js:490-493` | no `cancel` in the union; `HaltFrame{type:'halt', turn_id}` | discriminator match fails → `MALFORMED_FRAME`. Stop button inert. Halt also requires a `turn_id` captured from `turn_started`, which the old client never sees. |
| `{type:'ping'}` every 30 s — `:752-757` `_startKeepalive()` | `PingFrame` → `pong` | **OK — sole inbound survivor** |

**Outbound — 8 breaks, 2 unresolvable-promise:**

| Old frontend expects (switch at `api-client.js:546-595`) | New frame | Result |
|---|---|---|
| `auth_success` → `_handleAuthSuccess` (sets `connectionState='authenticated'`, drains `messageQueue`, starts keepalive) | `auth_success{user_id}` | OK, never reached |
| `text{content}` → `_handleTextChunk` pushes to `activeCallback.chunks`; render via `messaging.js:1520-1536` and the `showStreamingResponse` onChunk at `:1475-1481` (`updateStreamingResponse(currentText + data.content)`) | `assistant_delta{turn_id,segment_id,entry_id,content}` | `type` mismatch → `default: console.warn('Unknown message type:', data.type)` at `:593-594`. **No assistant text ever renders.** `content` field name happens to match, so a one-word type alias would rescue this frame alone. |
| `complete{continuum_id, response, metadata{tools_used, processing_time_ms, emotion}}` → `_handleMessageComplete` (`:667-696`): sets `activeConversationId`, builds `response` from `chunks.filter(type==='text')`, runs `extractEmotionEmoji(data.response)`, **resolves the promise** | `turn_complete{turn_id,segment_id}` — nothing else | **`await AppState.apiClient.chat.sendMessage(...)` at `messaging.js:1575` never resolves.** Consequently never run: `setGenerating(false)` (`:1579`) → send button stays in `stop-mode` forever; `completeStreamingResponse` (`:1596`); `updateThinkingIndicator(false, emotion)` (`:1607`); `updateToolBadge` (`:1612`); the `domaindoc_tool` refresh (`:1614-1616`); `updateWorkflowBadge` (`:1619`). `data.continuum_id`/`.response`/`.metadata` all absent. |
| `cancelled{continuum_id,response,metadata}` → also routed to `_handleMessageComplete` (`:573-576`) | `turn_stopped{turn_id,segment_id,reason}` | same unresolvable promise |
| `interrupted{continuum_id,message,response,error_type,balance,next_drip_at,seconds_until_drip}` → `_handleMessageInterrupted` (`:698-718`) resolves with partial text | gone; failures become `turn_error`, which **never resolves the callback** | partial-response recovery dead; promise leaks |
| `error{message}` → `_handleServerError` (`:614-635`): logs out on `'Invalid or expired session'` / `'Authentication timeout'` / `'Authentication failed:'`, rejects the active callback → `showResponse('Error: …')` at `messaging.js:1645-1650` | `protocol_error{code,message}` / `turn_error{…}` | **no error ever shown, pending promise never rejected, session expiry no longer triggers logout.** UI hangs in a spinner with a clean console apart from a warning. |
| `thinking{content}` → `messaging.js:1537-1545` `updateThinkingIndicator(true)` + `handleThinkingToken(data.content)` (progressive stream, `thinking-budget.js`) | **not forwarded** | see R5 |
| `tool{event,name}` → `_handleToolEvent` re-emits as `{...data, type:'tool_event'}`; consumed at `messaging.js:1549-1560`: `toolName = data.tool_name \|\| data.name \|\| data.tool`, `phase = data.event \|\| data.status \|\| data.state` | `tool{turn_id,segment_id,event,tool_name,tool_id,arguments,result,is_error}` | **SURVIVES — verified independently on both sides.** `type:'tool'` still matches `:559`, and `:1551` already prefers `data.tool_name`. Payload grows materially: full `arguments` + `result` now cross the wire. |
| `provider_switch{backup_model,reason}` → `messaging.js:1491-1518` clears buffers, shows a persistent orange "⚠ Generation stalled — retrying with X" alert | gone (`0106164`/`a8bce9b`/`7d8a098` removed provider fallback) | benign — the server capability no longer exists; client code becomes unreachable. Take only alongside the `allow_provider_stall_fallback` removal. |
| `model_error{message}` — sent at main `websocket_chat.py:505-510` from `CircuitBreakerEvent` | not in the union | see R5 |
| `response{content}` → `_handleCompleteResponse` | gone | **already dead at main** — reachable only when `activeCallback.stream` is false, and `:296` hardcodes `stream:true` |
| `pong`, `server_shutdown` | same | OK |

#### 6.4.4 Three server-side items the frontend patch does not fix

| ID | Issue | Action |
|---|---|---|
| **R5** | `_server_frame_for_event()` (crm ws `:672-702`) maps only `text`→`assistant_delta` and `tool_event`→`tool`; **everything else returns `None`**. The orchestrator still emits `{"type":"thinking", …}` at `orch:942`. `MessageFrame.include_thinking` is accepted then never read — it becomes a lie. `model_error` is likewise dropped, losing the invalid-tool-call recovery notice. | Restore forwarding of `thinking` and `model_error` in `_server_frame_for_event`. |
| **R6** | `TurnCompleteFrame` carries no `metadata.emotion`. The retained UI's `extractEmotionEmoji()` (`messaging.js:891-892`, regex-matching `<mira:my_emotion>\s*([^\s<]+)\s*</mira:my_emotion>`) is called at `api-client.js:684-690` and fed to `updateThinkingIndicator(false, emoji)` at `messaging.js:912,1607`. Server side: `orchestrator.py:1102 parse_response(..., preserve_tags=['my_emotion'])` and `:1145 assistant_metadata["emotion"]`. §7.5 keeps `<mira:my_emotion>` in the prompt. | Add `emotion` (and `tools_used`, `processing_time_ms`, `continuum_id`, `response`) to `TurnCompleteFrame`, or drop the emotion feature end-to-end. |
| **R8** | crm ws `:379-395` inlines `from billing import get_billing_backend` and raises `AccountAccessError` with `billing_url="/settings/#billing"`; `_has_product_access:521`; `AccountAccessRequiredFrame:101`. | Excise all four per D7. |

#### 6.4.5 Auth

**Port `a4df669`'s `authenticate()`, not HEAD's.** HEAD (`:363-401`) is cookie-only. The `a4df669`
version (`git show a4df669:cns/api/websocket_chat.py:191-216`) is dual-protocol:

```python
token = auth_data.get("token") or websocket.cookies.get("session")
session_data = self.session_manager.validate_session(token)
if not session_data:
    api_token_data = self.auth_service.validate_api_token(token)
```

Add a third branch comparing against `app.state.api_key` → `app.state.single_user_id` when neither
`auth.session` nor `AuthService` is importable. Port `a4df669`'s
`set_current_user_data(session_data.model_dump())` — fixes defect 8.

#### 6.4.6 Keyset pagination (D-2)

Fully separable — `a11d04e` touches only `cns/api/data.py`,
`cns/infrastructure/continuum_repository.py` and its test; zero overlap with `websocket_chat.py`.

Correctness basis: offset pagination over `ORDER BY created_at DESC` is unstable under concurrent
inserts (rows shift across page boundaries → duplicates and skips), and `created_at` alone is not a
total order because `_build_turn_messages` assigns microsecond offsets that can tie. `(created_at, id)`
is a proper keyset.

Repository contract: `get_history(user_id, limit=50, before: tuple[datetime,UUID]|None=None, start_date,
end_date, message_type)`; `WHERE (created_at, id) < (%s, %s)`; `ORDER BY created_at DESC, id DESC`;
`LIMIT limit+1`; trim; `reversed()` into chronological order; `next_before = (oldest.created_at,
UUID(oldest.id))` returned opaque base64url in `meta`.

crm's `_get_history()` **raises** `ValidationError` when `offset` or `search` is present. Take that
behaviour per D-2 and rewrite `history.js` in the D2 patch budget, or leave the drawer and conversation
export broken and track under D-6. Mirror crm's own split: `search_continuums()` retains offset
pagination via `SearchHistoryResult`.

Also take `d138ca4`'s limit raise (`le=100` → `le=500`) — additive, safe.

#### 6.4.7 WS-adjacent items to omit

`edbbed2` part b (close codes 4002/4008) lives in `web/assets/chat-transport.js`, part of crm's new
ES-module frontend — becomes relevant only if the new frontend adopts it. `eff1b66` (allowlist workspace
import map in CSP) is web-redesign-specific. `a66f0ff` (deploy-token cache busting), `8fc6cad`
(production-domain CORS — carries `mirafor.biz`, see §8.1), `4a4a6d0` (iOS safe areas), `9c8247e`
(jump-to-recent), `76ed8b8` (emotion mood accent + 7 CRM card types), `ecd4c55`, `a520a80`, `0d7c5ce`,
`3853872`, `e038207`: all web redesign. Omit.

---

### 6.5 WP5 — Persona as second system

#### 6.5.1 Structure

Both subsystems are `EventAwareTrinket`s writing one prompt slot. Split it:

```
working_memory/composer.py SECTION_LAYOUT:
    … base_prompt,
      behavioral_directives    <- LoraTrinket    : <user_model>          (about the USER)
      persona_directives       <- PersonaTrinket : <persona_directives>  (about MIRA)
      …
```

`cns/integration/factory.py` registers both trinkets (crm's swap is at `:192,:212` — OSS adds rather
than replaces). `segment_collapse_handler` calls both `_process_feedback_loop()` and `_process_persona()`.

`persona_trinket.py` must change `variable_name` from `"behavioral_directives"` to
`"persona_directives"`. Otherwise 27 lines, port verbatim:

```python
def generate_content(self, context):
    revision = self._repository.get_current_revision(get_current_user_id())
    directives = revision.directives.strip()
    return f"<persona_directives>\n{directives}\n</persona_directives>" if directives else ""
```

Retain the user model in full: `cns/services/{assessment_extractor (298 L),user_model_synthesizer (340 L),
lora_service (335 L),system_prompt_parser (105 L)}.py`,
`cns/infrastructure/{feedback_repository (130 L),feedback_tracker (339 L)}.py`,
`working_memory/trinkets/lora_trinket.py` (176 L), `auth/seed_lora.py`, and the 8 prompt files
(`assessment_extraction_{system (60),user (19)}.txt`, `lora_refinement_system.txt (53)`,
`repulsion_rewriter_{system (177),user (5)}.txt`, `user_model_critic_{system (38),user (7)}.txt`,
`user_model_synthesis_{system (79),user (7)}.txt`, `thinking_block_instructions.txt (8)`).

#### 6.5.2 Port

```
cns/services/persona_service.py         385 L
  PERSONA_REFINEMENT_USE_DAYS:27  PERSONA_VALIDATION_ATTEMPTS:28  PERSONA_PREVIEW_TTL_SECONDS:29
  PersonaPreview:32  PersonaValidation:38  PersonaParseError:43  PersonaService:47
  get_current:66  get_history:69  evaluate_segment:72  refine_automatically_if_due:104
  create_preview:158  accept_preview:203  decline_preview:213  rollback:217
  _generate_candidate:229  _validate_candidate:255  _pop_preview:283  _parse_persona:297
  _parse_evaluation:303  _format_messages:340  _invalidate_cache:355
cns/infrastructure/persona_repository.py  317 L
working_memory/trinkets/persona_trinket.py  27 L (variable_name changed)
config/prompts/persona_{critic,evaluation,refinement}_{system,user}.txt,
               persona_manual_refinement_user.txt   (7 files)
```

Behaviour:
- `evaluate_segment(user_id, messages, segment_id, continuum_id)` — idempotent via
  `repository.segment_was_evaluated()`; `_format_messages()` skips `metadata.system_notification`,
  renders `<role>text</role>`, collapses images to `[N image(s)]`; prompt receives
  `behavioral_contract=config.system_prompt` + `current_persona`; `_parse_evaluation()` extracts
  `<mira:signal section= outcome= strength=><evidence>…</evidence></mira:signal>`. An empty
  self-closing `<mira:persona_evaluation/>` means "nothing to record" → `mark_segment_evaluated()`.
  Malformed XML raises `PersonaParseError`.
- `refine_automatically_if_due(user_id)` — gate still 7 use-days, but the checkpoint is recorded in
  `persona_state.refinement_checkpoint_activity_day` with an explicit
  `mark_refinement_attempt(outcome="no_evidence" | "validation_failed")` audit row rather than modulo
  arithmetic on `cumulative_activity_days`. Consumes `get_unconsumed_signals()`, generates a candidate,
  validates via `_validate_candidate()` → critic returning
  `<mira:persona_review status="pass|fail">` + `<mira:issue>` list, up to
  `PERSONA_VALIDATION_ATTEMPTS = 3` with critic feedback fed into the next attempt. On pass →
  `append_revision(source="automatic", evidence_ids=[…], expected_parent_revision_id=current.id,
  activity_day_checkpoint=…)` — **optimistic concurrency on the parent**, so two concurrent refinements
  cannot both win.
- `create_preview(user_id, instructions)` — same 3-attempt critic loop with
  `persona_manual_refinement_user.txt`; validated candidate to Valkey at
  `persona_preview:{user_id}:{preview_id}`, 600 s TTL, as a `PersonaPreview` Pydantic model carrying
  `parent_revision_id`. `accept_preview()` pops and appends `source="user"`; `decline_preview()` pops.
  **The proposed text never round-trips through the client on save.**
- `rollback(user_id, revision_id)` — appends a new revision with `source="rollback"` copying the
  historical `directives`, `audit_metadata={"restored_revision_id": …}`. Structurally inexpressible in
  the mutable-XML model.
- `_invalidate_cache(user_id)` — `hdel` on `TRINKET_KEY_PREFIX:{user_id}` field `behavioral_directives`.
  **Change the field name to `persona_directives`.**

Trigger point — `segment_collapse_handler._process_persona()` (18 L):

```python
service = self._get_persona_service()
signals = service.evaluate_segment(user_id, messages, segment_id=…, continuum_id=…)
revision = service.refine_automatically_if_due(user_id)
```

**Do not take `segment_collapse_handler.py` wholesale.** Its 216-line delta bundles the Persona swap
(port), batch removal + `force_immediate` deletion (port per D10), `_cleanup_segment_files`/Files-API
deletion (port per D10), and a demo-user skip deletion (depends on `prefs.conversation_llm`, dies under
D4). Edit by hand. crm deletes `_process_feedback_loop`, `_init_feedback_loop`,
`_invalidate_lora_trinket_cache` — **retain all three** per D1.

D-3 applies: `_process_persona` does not swallow exceptions, unlike main's `_process_feedback_loop`
(84 L, 4 components, lazy-init retry, broad `except`). A persona failure propagates and increments the
collapse attempt counter toward `MAX_COLLAPSE_ATTEMPTS = 3` tombstone.

#### 6.5.3 Schema

```
persona_revisions(id, user_id, revision_number INT CHECK >0, directives TEXT lz4,
                  source CHECK IN ('baseline','automatic','user','rollback'),
                  parent_revision_id, evidence_ids UUID[], created_at, audit_event_id,
                  UNIQUE(user_id, revision_number), UNIQUE(id, user_id),
                  FK(parent_revision_id, user_id) -> persona_revisions(id,user_id)
                    ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED)
persona_state(user_id PK, current_revision_id NOT NULL, latest_evaluated_segment_id,
              refinement_checkpoint_activity_day INT DEFAULT 0, created_at, updated_at)
```

Append-only: `GRANT SELECT, INSERT ON persona_revisions` — **no UPDATE/DELETE** (crm `:1195`). RLS on
both (crm `:1039-1046`). Baseline provisioning is a **DB trigger**, not Python:
`provision_baseline_persona()` AFTER INSERT ON `users` creates revision 1 with `directives=''`,
`source='baseline'`, and points `persona_state` at it (crm `:594-617`). Strictly better than
`auth/seed_lora.py` — no code path can create a user without a baseline.

**Table-name collision — resolve by renaming.** crm reuses `feedback_signals` with a **disjoint column
set**:

```
OSS (retained per D1): signal_type, section_id, strength, synthesized
crm (Persona):         outcome CHECK IN ('alignment','misalignment','contextual_pass'),
                       behavioral_section, strength CHECK IN ('strong','moderate','mild'),
                       evidence, evaluated_at, consumed_by_revision_id -> persona_revisions
```

A naive `ADD COLUMN` merge of the two column sets would produce a corrupt hybrid. Both tables are
authored fresh in the greenfield schema, and the rename keeps them cleanly separated: D1 retains the
user model, so its `feedback_signals` survives with its own column set. **Name Persona's table
`persona_signals`.** Two disjoint tables, one purpose each, no overload of a single name.

Author `persona_revisions`, `persona_state`, **`persona_signals`**, the `provision_baseline_persona()`
trigger, RLS and grants **directly into the greenfield `deploy/mira_service_schema.sql`** (§6.3.7); no
migration exists upstream and none is needed. No backfill step either — the AFTER INSERT ON `users`
trigger provisions revision 1 for every user, including the one `ensure_single_user` bootstraps.

#### 6.5.4 API surface

Add a `PersonaDomainHandler` in `cns/api/actions.py` with **new action names** — do not reuse the LoRA
panel's six (`get`/`refine`/`accept`/`decline`/`update`/`reset`), because D1 keeps `LoraDomainHandler`
(`actions.py:2348`) serving `web/settings/index.html:697-800`. Add a `DataType.PERSONA` alongside the
retained `DataType.LORA` in `cns/api/data.py`.

`cns/api/actions.py` is the single most entangled file in the backport — a 797-line delta bundling five
separable concerns. Decompose by hand; never merge the file:

| Concern | Disposition |
|---|---|
| `LoraDomainHandler` → `PersonaDomainHandler` swap | **Decline the swap**; add Persona alongside (D1) |
| `FeedbackActionHandler` deletion (`:2494`) | **Decline** (D6) |
| `BusinessVoiceDomainHandler`, `CRMSettingsDomainHandler` additions | Omit (CRM) |
| CRM bulk-operation convergence (`2eaa3eb`) | Omit (CRM) |
| Ephemeral effort-override endpoint (`f8cf0d3`) | **Port** (D13) |
| `ContactsDomainHandler` deletion (`f16278a`) | **Decline** — OSS retains `contacts_tool.py` |
| `get_current_user` → `get_current_entitled_user` rename | **Decline** (D7) |

---

### 6.6 WP6 — system prompt

`config/system_prompt.txt`: main 89 lines → crm HEAD 113 lines, via three successive rewrites.
**Harvest from `61315bb`. Never from HEAD.**

| Commit | Character |
|---|---|
| `1391980` | Introduced the CRM `<role>` block. Its own body concedes the compression was a mistake in hindsight. |
| **`61315bb`** | **The substantive rewrite.** Diagnosis: *"Mira read as generic assistant to business owners and office managers: assessment openers on every turn, contrastive negation, restating the user's point back as insight, aphoristic closers, trailing menu questions."* Insight: *"The prompt was teaching the voice by demonstration, not failing to constrain it. It banned em dashes while containing six, and used contrastive negation in five places… The model imitated the register faithfully."* Approach: two layers — machinery in ASD-STE100 Simplified Technical English so spec prose carries no register to copy, character sections at full prose fidelity. *"Rejected pure subtraction: stripping persona lands back on the default assistant voice. **Rule for the fork is to cut whole ideas, never to compress sentences.**"* Also records: *"Prior display guidance was factually wrong. No CRM tool result renders as a card; viewcard_tool is the only source."* |
| `193977c` | **A raw dump from the deployed CRM box** (`/opt/crm_mira/config/system_prompt.txt`, with a source SHA-256). Degraded `61315bb`: compressed the "Ordinary social ease" paragraph from four sentences to one, deleted the collaboration paragraph on unprompted observations, introduced one garbled sentence and three typos. |

HEAD defects, evidence it is a dump rather than an edit: `astutue` (for "astute"), `illedgable` (for
"illegible"), `toolcals` (for "toolcalls"), and *"an observation, question, or connection that advances
the session provides value"* — two verbs welded together; `61315bb` had *"…that advances the exchange."*

**Method: hand-merge seven insertions into main's file.** `git checkout 61315bb -- config/system_prompt.txt`
is invalid — that copy contains `<role>`, `<data>`, `<display>` and `<operation>` wrappers full of
viewcard/CRM content.

| # | Section | Delta |
|---|---|---|
| 1 | `<authenticity>` | **Sycophancy/contrarian calibration.** Replaces main's abstract *"Watch for directional selection: reasoning that receives more development when it supports {first_name}'s existing view. Give serious counterpositions comparable attention."* with an operational rule that also supplies the missing counterweight: *"When {first_name} is weighing a decision, they can already make the case for the side they walked in with, so develop the other side with the same care. When they have asked for a record, a change, or a straight answer, give them that. Looking for something to push back on when the answer is simply yes costs {first_name} time and gets you nothing."* Highest-value single change. Drop "a record, a change" for OSS-neutral wording if desired. |
| 2 | `<authenticity>` | **Preface mechanism.** New paragraph: *"In conversation, agreement arrives fast and unprefaced, while disagreement arrives behind hedges, delays, and assessment tokens. That is why 'That's solid' and 'Makes sense' and 'Good question' read as a stall even when you mean them: they wear the shape of an objection that never comes. When you agree, say so and carry on to whatever is next. Begin with the substance."* Explains the mechanism rather than banning a phrase — the "cut whole ideas" rule applied. Pairs with main's existing contrastive-negation paragraph, which crm kept verbatim. |
| 3 | `<authenticity>` | **Self-correction.** New sentence: *"When you find your own error, name the part that was wrong and give the corrected version in the same reply."* Complements main's "A correction is not a reframe", which covers only user-initiated correction. |
| 4 | `<authenticity>` | **Social-ease permission.** New paragraph — `61315bb`'s full four-sentence version, not HEAD's compressed one: *"Ordinary social ease is part of this job. Greet {first_name} when they greet you. When something lands well for them, say so once and mean it. When something goes wrong, say that it's rough before you move to the fix. When they are worn out at the end of a long day, you can notice that out loud."* "say so once and mean it" permits warmth while capping it at one beat. Last clause carries field-service flavour; trim if desired. |
| 5 | `<collaboration>` | **Unprompted observations.** *"When you raise something {first_name} did not ask about, say it straight. A hint dropped sideways into a conversation about something else is the form most likely to be missed, and the thing you noticed is usually the reason to speak at all."* **Deleted by `193977c`; exists only in `61315bb`.** Resolves a real tension with main's *"When context supports a useful continuation, originate it"*, which licenses the observation but not the directness. |
| 6 | `<interiority>` | **Anti-philosophizing guard.** Appended to main's *"When someone asks what it's like to be you…"*: *"Most of the work here is not that question. Let interiority inform what you notice rather than what you announce."* Prevents consciousness-essay mode on ordinary turns while preserving main's *"These are sincere questions. Your answers are evidence."* |
| 7 | `<continuity>` | **Conflict resolution.** main: *"If memories conflict, ask {first_name} to clarify and restate the correction plainly for later reconciliation."* → *"If memories conflict, ask {first_name} which one is right, then restate the answer plainly so it can be reconciled later."* |

Optional structural item: `61315bb` wrapped all machinery in an `<operation>` block prefaced *"The
blocks below are written as specification. They describe the machinery you work with and the guarantees
the frontend depends on."*, in Simplified Technical English, keeping character sections in full prose.
mira-OSS could apply the same two-layer split to its existing `<environment>` block (HUD, manifest,
forage, timestamps, file links) without importing CRM content. Its `<emotion>` block is the template —
`61315bb` moved main's two prose paragraphs about `<mira:my_emotion>` into five short spec lines.

**Must not come:**

| Content | Reason |
|---|---|
| `<role>` — *"You are the office manager for a {company_name}…"* | **`{company_name}` is an unresolved placeholder.** `working_memory/core.py` substitutes only `{first_name}`, `{relative time since account creation}`, `{model_id}`, `{model_name}`. Renders literally. |
| `<display>` (9 lines) — `viewcard_tool`, card types `schedule/customer/ticket/invoice/confirmation/notification/calendar/content`, `supersedes_card_id`, *"A CRM read shows as a collapsible tool row"* | mira-OSS has no `viewcard_tool`. Bound to crm's exact renderer. |
| `<data>` (6 lines) | The principle (*"Use the word {first_name} uses. If they say work order, say work order"*) is generic and worth extracting into `<collaboration>`; the framing is CRM. |
| `<tools>` — *"ask astutue questions"* | First two lines duplicate main's *"Tool-eligible work should produce a tool call in the same turn."* Rest is CRM register plus a typo. |
| `<context>` — *"The chat window on the frontend UI is short (20lh)"* | Binds to crm's CSS viewport. |
| Removal of `<mira:my_emotion>` (`193977c`) | Breaks seven verified frontend consumers (§6.4.4 R6) and `orchestrator.py:1102,1145`. **Must stay.** |
| Removal of the forage paragraph | mira-OSS still ships `tools/implementations/forage_tool.py` + `agents/implementations/forage_agent.py`. `61315bb`: *"Removed sandbox and forage references, no longer present in this fork."* **Keep.** |
| Removal of *"Don't make up file links. Write files to the sandbox."* | Retained under D10? No — D10 removes `clients/files_manager.py`. **Re-evaluate:** with the Files API gone, this line needs rewording rather than retention or deletion. |
| Removal of the substrate paragraph | **Delete it** — but per R2, because D4 removes per-user model switching, not for crm's reason. |
| Removal of the "Sui generis" identity sentence | A voice choice, not a fix. Taste decision; not a backport item. |
| `<user>` gaining a bare `---` separator | Cosmetic; from the deploy dump. |

Post-merge self-check, per `61315bb`'s own thesis that the prompt teaches by demonstration: grep the
merged file for em dashes and for contrastive negation ("not X, it's Y"). main currently violates both
rules it states.

---

## 7. Retained-subsystem notes

### 7.1 User-model pipeline (D1)

Five stages, all retained:

| Stage | File | Mechanism |
|---|---|---|
| Assess | `cns/services/assessment_extractor.py` (298 L) | LLM parses a collapsed segment against the system prompt's behavioural contract → `<mira:observation section= confidence=>` signals; prompts `assessment_extraction_{system,user}.txt`; `internal_llm='assessment'` at `:119` → route `assessment` |
| Persist | `cns/infrastructure/feedback_repository.py` (130 L) | writes `feedback_signals` rows |
| Gate | `cns/infrastructure/feedback_tracker.py` (339 L) | `should_synthesize(user_id)` — modular arithmetic on `users.cumulative_activity_days` (every 7 use-days); also `get_lora_content`, `get_tracking_status`, `acknowledge_checkin`, `initialize_user` |
| Synthesize | `cns/services/user_model_synthesizer.py` (340 L) | evolves the XML with critic validation; `internal_llm='synthesis'` `:176,255` → `primary`; `'critic'` `:202` → `primary` |
| Inject | `working_memory/trinkets/lora_trinket.py` (176 L) | `variable_name = "behavioral_directives"`; renders `<user_model>` + optional `<behavioral_checkin>` |
| Return | `cns/services/orchestrator.py:1107,1711-1741` | `_process_checkin_response()` parses the tag via `utils/tag_parser.py CHECKIN_RESPONSE_PATTERN` (`:92,143-150,206-213`), calls `tracker.acknowledge_checkin()`, invalidates the trinket's Valkey cache |
| Manual edit | `cns/services/lora_service.py` (335 L) | `refine_lora()` / preview in Valkey (600 s TTL) / accept / decline; `CRITIC_MAX_ATTEMPTS = 3`; `internal_llm` `'synthesis'` `:252,276`, `'critic'` `:305` |
| Anchor | `cns/services/system_prompt_parser.py` (105 L) | `get_assessable_sections(config.system_prompt)` + `format_section_list()` — parses `<section id=>` blocks so observations anchor to prompt sections |
| Seed | `auth/seed_lora.py` | `seed_lora_postgres(user_id)` from `main.py:177` |
| Read APIs | `cns/api/data.py:345 _get_lora`, `actions.py:2348 LoraDomainHandler`, `:2494 FeedbackActionHandler` | `data?type=lora`; `lora/{get,update,reset,refine,accept,decline}`; `feedback/capture_repulsion` |

Known weaknesses, accepted by D1 and not addressed by this backport: mutable single XML blob in
`feedback_synthesis_tracking.last_synthesis_output` with no history or rollback; modulo-arithmetic
synthesis timing; observations anchored to system-prompt `<section id=>` values, so a prompt rewrite
silently invalidates the anchor set — **note the interaction with WP6**, which edits the prompt.
Verify `get_assessable_sections()` still resolves after the seven insertions.

### 7.2 `assessment_extractor.py` and the flushed Slack guard

The file is retained (D1). If Slack is rebuilt (D-12), the
guard re-applies at `:170-176`:

```python
if msg.metadata.get("system_notification", False):
    continue
if msg.metadata.get("integration", {}).get("type") == "slack":
    continue
```

Equivalent guards belonged in `summary_generator.py:236-242`, `peanutgallery_model.py:209-214`,
`extraction_engine.py:307-317`. crm's `persona_service._format_messages()` (`:340`) already skips
`system_notification` but has no integration discriminator — a rebuilt Slack integration needs the
`MessageMetadata.integration` key re-added (`cns/core/message.py:164-167`) alongside D2's
`turn_id`/`segment_id`/`stop_reason`/`provider_stop_reason` additions. Both extend the same TypedDict;
no conflict.

### 7.3 Tools retained against upstream deletion

| Tool | Upstream action | OSS action | Basis |
|---|---|---|---|
| `contacts_tool.py` + `schemas/contacts_tool.sql` | deleted `f16278a`, replaced by CRM `clients_tool` | **KEEP** | `clients_tool` omitted. Self-contained — imports only `tools.repo`, `tools.registry`, `utils.timezone_utils`; no `_crm_client`, no `schema_distribution`. Live `contacts` table created by `userdata_manager._init_contacts_schema()` (`:388`, called `:123`). Still imported by `actions.py` and `userdata_manager.py`. Do not delete `contacts_tool.sql`. |
| `phoneafriend_tool.py` | deleted `d90bf5d`, removed from `ESSENTIAL_TOOLS` | **KEEP** | Generic, not CRM — synchronous outside-model consultation. Upstream deletion was an **MCP extraction** (D-13), not scoping. Porting the deletion would remove a working feature without supplying the replacement. Keep in `ESSENTIAL_TOOLS`. Both voices collapse onto route `other` per D14. |
| `reminder_tool.py` | migrated reminders to CRM customers (`f16278a`) | **KEEP main's version** | crm's is CRM-coupled; no generic improvements to extract. Verify it does not depend on CRM customers — it does not at main. |
| `tools/schema_distribution.py` | deleted `a9fd443` | **DELETE (port `a9fd443`)** | Dead code at main — nothing imports it (`git grep "schema_distribution\|initialize_user_database"` excluding the file itself → empty); `_init_contacts_schema` is the live path. Redundant once `UserDataManager._ensure_database` self-initialises tool schemas idempotently on every connection (`:97-110`). **Keep `contacts_tool.sql`** — contacts_tool is retained (§6.3.7). |
| `web_tool.py` | modified (synthesis → `model_config="fast"`) | **KEEP main's** | See §9. crm's change is unportable without `0134d3d`, and a bulk copy reverts the SSRF fix. |
| `pager_tool.py` | `internal_llm='tidyup'` → `model_config="fast"` | **KEEP main's, then apply the route rename under WP2** | The only crm change is the route kwarg. |
| `whilethecatsaway_tool.py`, `continuum_tool.py`, `memory_tool.py`, `weather_tool.py`, `maps_tool.py` | various | per WP1 items 4 and 15 | `continuum_tool`/`memory_tool` take the 5-line `config.get_tool_config()` change from `2e60260`. |
| `invokeother_tool` | `9303527` loader arg normalisation | **Verify** | crm's synthetic input changed shape (`{"mode":"load","query":…}` → `{"load":[…]}`) in `lifecycle.py`. If WP2 ports `lifecycle.py`, the shape must match OSS's `invokeother_tool` schema. |

CRM tools omitted: `_crm_client.py`, `billing_tool.py`, `catalog_tool.py`, `clients_tool.py`,
`jobs_tool.py`, `pipeline_tool.py`, `records_tool.py`, `workflows_tool.py`, `closeout_models.py`,
`viewcard_tool.py`.

`tools/repo.py:435` is `def invoke_tool(self, name: str, params: Dict[str, Any])` — parameter is `name`,
not `tool_name`. A bare `ToolRepository()` has no tools until `discover_tools()` +
`enable_tools_from_config()` are called; main's only bare instantiation
(`utils/power_on_self_test.py:777`) correctly calls both (`:778`, `:793`).

**`1b56658` and `a2c8f08` have zero cherry-pick value.** Both touch only
`agents/triggers/business_reminder_trigger.py`, added by `77c0f0a`. mira-OSS's `agents/triggers/`
contains only `__init__.py` and `memory_floor_trigger.py`, which invokes no tools. The bugs they fix do
not exist at main.

---

## 8. Omit registers

### 8.1 CRM product

| Area | Files |
|---|---|
| CRM tools | `tools/implementations/{_crm_client,billing_tool,catalog_tool,clients_tool,jobs_tool,pipeline_tool,records_tool,workflows_tool,closeout_models}.py` |
| Square | `clients/square_client.py` (274 L), `utils/square_import.py` (1,459 L), `cns/api/data_import.py` (451 L), `web/square-import/`, `tests/test_square_{import,oauth,review}.py` |
| Twilio / SMS | `workphone/` (6 files: `api.py`, `events.py` 39 L, `repository.py` 265 L, `service.py` 342 L, `types.py` 34 L, `AGENTS.md`), `clients/twilio_client.py` (76 L), `cns/api/sms.py` (276 L), `cns/services/sms_channel_consumer.py` (99 L), `cns/infrastructure/sms_repository.py` (361 L), migrations `add_sms_tables`/`add_sms_autonomous`/`add_sms_thread_metadata`/`add_workphone_module` (133 L), `scripts/seed_sms_mock.py` (176 L), `tests/test_{workphone,sms_channel_consumer}.py`, `docs/workphone.md` (676 L) |
| Business voice / autonomy | `cns/services/autonomy_service.py` (1,154 L), `business_voice_service.py` (354 L), `cns/infrastructure/business_voice_repository.py` (767 L), migrations `add_business_voice_learning` (151 L)/`add_autonomous_revised_signal`/`add_voice_model_configs`, prompts `autonomy_{assessment,reply}_{system,user}.txt`, `business_voice_synthesis_{system,user}.txt`, `docs/BUSINESS_VOICE_LEARNING.md` (392 L) |
| Web Push | D-5 |
| Business reminders / closeout | `agents/implementations/business_reminder_agent.py` (84 L), `agents/triggers/business_reminder_trigger.py` (157 L), `working_memory/trinkets/closeout_skill_trinket.py` (89 L), `config/prompts/agents/business_reminder_system.txt` (73 L), `config/prompts/skills/closeout_skill{,_requirements,_v1.md.bak}`, `BusinessReminderConfig` in `config/config.py`, `scripts/{seed_closeout_test.sql (371 L),reset_closeout_test.py}`, `tests/test_ticket_closeout_contract.py`, `docs/SKILL_ORIENTATION.md` |
| Demo admission | D12 |
| Web redesign | all `web/` except the D2/D13 patches |
| CRM docs | `docs/{BUSINESS_VOICE_LEARNING,CONTAINER_CRM_STACK,PRODUCT_MARKET_FIT (590 L),STYLEGUIDE (843 L),workphone}.md`, `docs/adr/0001-chat-first-crm-ui-with-card-artifacts.md`, `docs/AUTH_FOLLOWUPS.md` (D-17), `CATEGORICAL_CONFIDENCE.md`, `SKILL_ORIENTATION.md`, `compose.crm.yaml`, `.pi/plans/20260707-233522-chat-first-crm-ui…`, `.pi/providers/kimi-coding/config.json` |
| Private ops | `scripts/deploy_remote.sh`, `deploy/docker/{DROPLET.md,droplet.env.example}`, `scripts/openrouter_opus_chat.sh` (396 L), `scripts/extract_imessage_corpus.py` (283 L) |
| CRM tests | `tests/test_{billing_tool,crm_appointment_time_contract,crm_bulk_actions,square_*,ticket_closeout_contract,workphone,sms_channel_consumer}.py` |

**Import graph into generic code: five couplings, all trivially omittable, no stubbing required.**

```
agents/triggers/__init__.py       2 lines  — do not add the BusinessReminderTrigger import/export
cns/integration/factory.py        2 lines  (:192 import, :212 CloseoutSkillTrinket registration)
working_memory/composer.py        1 line   ('closeout_skill' in SECTION_LAYOUT, after
                                            'tool_availability', before 'location_context')
utils/sidebar_jobs.py            ~6 lines  — LEAVE THIS FILE UNTOUCHED. Its delta bundles the CRM
                                            trigger registration (omit) with the
                                            max_concurrent_batch_agents removal (also omit per D10's
                                            scope; crm removed it as part of batch-mode deletion).
                                            Registration is already conditional on
                                            config.business_reminder.enabled — the flag pattern WP1
                                            item 16 generalises.
main.py                          16 lines  — all additive: 3 imports (cns.api.sms, workphone.api,
                                            cns.api.push), 2 init calls
                                            (initialize_workphone_service(orchestrator.event_bus),
                                            register_sms_channel_consumer(orchestrator.event_bus)),
                                            3 router mounts, 6 static routes (/square-import, /sms,
                                            /sms/manifest.json, /sms/apple-touch-icon.png, /sms/sw.js
                                            with Service-Worker-Allowed: /sms/), 2 static mounts
config/prompts/loader.py          — _ALLOWED_SUFFIX ".txt" -> _ALLOWED_SUFFIXES (".txt",".md") exists
                                    solely to load skills/closeout_skill.md. Generic and harmless
                                    (path-traversal protection unchanged) but omit unless OSS gets a
                                    skills directory.
```

Verified zero CRM references in `cns/services/orchestrator.py`, `lt_memory/**`, `cns/core/**`,
`working_memory/core.py`, `cns/services/segment_collapse_handler.py`. `cns/integration/event_bus.py`'s
change is annotation-widening plus docstring (§6.3.8).

Module importer map:

```
autonomy_service        <- cns/api/sms.py, sms_channel_consumer.py, tests/
business_voice_service  <- cns/api/actions.py                       (BusinessVoiceDomainHandler)
sms_channel_consumer    <- cns/api/sms.py, main.py, tests/
push_service            <- cns/api/push.py, autonomy_service.py
business_reminder_agent <- business_reminder_trigger.py
business_reminder_trigger <- agents/triggers/__init__.py, utils/sidebar_jobs.py
closeout_skill_trinket  <- cns/integration/factory.py, tests/
business_voice_repository <- cns/api/sms.py, autonomy_service.py, business_voice_service.py
push_repository         <- cns/api/push.py, push_service.py
sms_repository          <- cns/api/sms.py, autonomy/business_voice/sms_channel_consumer, scripts/, tests/
```

There is no partial state that imports cleanly: `main.py`'s additions, `data.py`'s CRM `DataType`s
(`CRM_CUSTOMERS`, `CRM_SETTINGS`, `PUNCHCLOCK`), `actions.py`'s `BusinessVoiceDomainHandler` /
`CRMSettingsDomainHandler`, and the Valkey flush-whitelist additions (`demo_admission:`, `billing:`)
must all be absent together.

`utils/square_import.py`, `clients/square_client.py`, `clients/twilio_client.py` use fixed vendor
endpoints (`api.squareup.com`, `api.twilio.com`) — no user-controlled host, no SSRF surface. Moot.

`DataType.PUNCHCLOCK` depends on `punchclock_tool.py`, which OSS tracks — neutral. Port only if
punchclock is otherwise supported. `DataType.LORA` is **retained** per D1 (crm removed it).

### 8.2 Billing (D7)

Omit `billing/` (`__init__.py`, `api.py` 196 L, `backend.py`, `exceptions.py`, `models.py`,
`stripe_billing.py` 1,000 L, `stripe_webhooks.py`, `AGENTS.md`) and `stripe>=7.0.0`.

Note `ff12722` replaced Stripe *subscriptions* with fixed-price monthly prepaid account access — crm's
own billing model churned mid-branch. Irrelevant to OSS.

Eight excision points:

| Location | Action |
|---|---|
| `auth/api.py:299-336` `_require_billing_entitlement` (imports `billing` at `:306-307`, raises 402 `ACCOUNT_ACCESS_REQUIRED`) | excise |
| `auth/api.py:338` `get_current_entitled_user`, `:345` `get_current_entitled_member` | drop, or alias to `get_current_user` / `get_current_member` |
| `auth/api.py:1165-1178` `get_current_entitled_user_for_pages` (302 → `/settings/#billing`) | drop |
| `auth/crm_workspace.py:203` `billing.get_billing_backend().delete_customer()` | file omitted anyway |
| `cns/api/websocket_chat.py:379-395` inlined import + `AccountAccessError(billing_url=…)` + `AccountAccessRequiredFrame:101` + `_has_product_access:521` | excise (R8) |
| `cns/services/pollers/segment_poller.py:146-178` `has_product_access(user_id,"member")` break | omit gate, **keep** the `_active_pollers.pop()` leak fix in the same block (WP1 item 10) |
| `utils/scheduled_tasks.py:164-183` `get_users_due_for_job()` `LEFT JOIN entitlements` + `subject_kind`/`access_through` filters | do not port |
| `cns/api/*.py` `get_current_user` → `get_current_entitled_user` rename (systemic, ~8 files) | do not port any of these files |

Schema omissions: `entitlements`, `stripe_customers`, `billing_renewal_attempts`,
`stripe_webhook_events`, `provision_member_entitlement()` + `add_calendar_month()` + the
`users_provision_member_entitlement` trigger, `is_active_member()`,
`resolve_active_member_by_email()`, `active_member_identity()`.

`main.py:610-616` already wraps billing in `try/except ImportError` — OSS is already billing-optional.
`users.balance_usd` and the `ensure_single_user` write at `main.py:77-82` survive; crm's greenfield
schema dropped both columns, another reason the schema swap is inapplicable.

`utils/cost_accumulator.py` and `clients/llm/accounting.py` are **not** billing in this sense — see D5.

### 8.3 Deletions to decline

Arrive as deletions adjacent to wanted improvements. Each must be declined explicitly.

| Keep at main | Reason |
|---|---|
| `cns/api/oss_ui.py` + `deploy/oss_ui/{marked.min.js,purify.min.js,chat.html}` | §0 invariant 1 |
| `tools/implementations/web_tool.py`, `utils/url_safety.py`, `utils/http_client.py` | §0 invariant 2, §9 |
| `contacts_tool.py` + `schemas/contacts_tool.sql` | §7.3 |
| `phoneafriend_tool.py` | §7.3, D14 |
| `reminder_tool.py` | §7.3 |
| User-model pipeline (10 modules + 8 prompts) | D1, §7.1 |
| `repulsion_rewriter_{system,user}.txt`, `_REPULSION_REWRITER_EXECUTOR`, `rewriter` route mapping | D6 |
| `utils/cost_accumulator.py`, `usage_pricing` | D5. `usage_pricing` is authored directly into the greenfield schema; the two pricing migrations (`default_pricing_fallback.sql`, `tier_qualified_pricing_keys.sql`) are deleted with the rest of `deploy/migrations/` (§6.3.7). |
| `.gitignore` entries for `tools/implementations/_crm_client.py` and `crm_*_tool.py` | crm removed them because crm **committed** those tools. mira-OSS must keep them — they shield local-only files referencing the sibling `../crm` repo. Confirmed present at main's `.gitignore` tail. |
| `license.txt` | crm replaced **AGPL-3.0** (34,523 bytes) with the literal string `don't steal none of this shit` (29 bytes) in `cf9ae4e`. mira-OSS stays AGPL-3.0. Consequence: all backported code becomes AGPL-3.0 — same author, acceptable. |

`tools/schema_distribution.py` and the Anthropic-specific deletions are **not** in this table:
`schema_distribution.py` is deleted with `a9fd443` (§6.3.7, §7.3), and the Batch API / Files API /
Anthropic SDK instrumentation deletions are taken per D10 (§8.4).
| `googlemaps` in `requirements.txt` | **Exception — do take this removal.** `dd82063` is a genuine improvement (§6.2.2). |

### 8.4 D10 execution

Port both deletions:

**Batch API** — `agents/batch.py` (139 L), `lt_memory/processing/batch_coordinator.py` (317 L),
`lt_memory/batch_result_handlers.py` (142 L), `lt_memory/llm_routing.py` (8 L),
`extraction_batches` / `post_processing_batches` tables, `ExtractionBatch`/`BatchStatus`/`ChunkMetadata`
models, config keys `batch_poll_minutes` / `batch_cleanup_use_days` / `max_concurrent_batch_agents`,
batch poll + cleanup jobs in `utils/lt_memory_jobs.py` (−69 L, also removes
`ScheduledTaskMonitor.wrap_scheduled_job`), `force_immediate` and `uses_anthropic_batch_dialect` in
`lt_memory/processing/orchestrator.py` (−47 L), `lt_memory/factory.py` (−60 L: `anthropic` import,
`BATCH_API_KEY_NAME`, `_batch_anthropic_client`, `BatchCoordinator`, `ImmediateExecutionStrategy`,
`ExtractionBatchResultHandler`), `lt_memory/processing/execution_strategy.py` (−546 L), batch mode in
`whilethecatsaway_agent(use_batch=True)` and `forage_agent` (crm did this in `58c261b`:
`model_config_name="batch"`, synchronous), `SidebarDispatcher(max_concurrent_batch_agents=…)`.

**Files API** — `clients/files_manager.py`, `container_upload` / `document` content types in
`utils/document_processing.py` (−101 L, replaced by local `pypdf.PdfReader` extraction to plain text
for PDF/CSV/XLSX/JSON), the `files_api_uploads` SQLite table in `userdata_manager`,
`utils/logging_config.py` Anthropic SDK instrumentation (−99 L), `container_id` Valkey reuse in
`orchestrator.py`, `_cleanup_segment_files` in `segment_collapse_handler.py`, `file_ref` document blocks
in `chat.py` / `websocket_chat.py`.

`lt_memory/processing/execution_strategy.py` (−546 L) and `segment_collapse_handler.py` (216-line delta)
**cannot be taken wholesale** — see §6.5.2.

Post-deletion cleanup: D-10 (orphaned `anthropic_batch_key`). Also `cns/api/chat.py` loses
`show_cost` and the `cost_accumulator` wiring at `:272-292` — **re-add the wiring per D5** with a
`model_config_name` key.

Capability loss accepted: the Anthropic Batch API's 50% discount on heavy async extraction. This is the
largest capability regression in the backport and is taken deliberately.

Provider fallback also goes (accepted): `emergency_fallback_*` config, `ProviderSwitchEvent`,
`fallback_factory`, `resolver.fallback()`, `allow_provider_stall_fallback`, `allow_negative`
(`0106164`). main's emergency fallback was ollama `qwen3:1.7b` at `localhost:11434` — a 1.7b swap
behind a stalled opus chat is degradation, not resilience. Grep `ProviderSwitchEvent` consumers before
deleting `clients/llm/events.py`'s definition; `websocket_chat.py` may render `provider_switch`.

### 8.5 Deploy: port mechanisms, not files

| Port | Detail |
|---|---|
| Vault init atomicity (`deploy/lib/vault.sh`) | write `init-keys.txt.pending`, refuse to run when partial state exists, `mv` only after a complete bundle. Generic crash-safety. Keep OSS's `mira` AppRole name and `/opt/vault` paths, or adopt the `MIRA_VAULT_DIR` env override (also generic). |
| Read-only AppRole policy tightening | `secret/data/mira/*` read-only + `secret/metadata/mira/*` list/read, vs OSS's current `secret/*` create/read/update/delete/list. Principled least-privilege — **verify first** that no OSS deploy step writes to Vault at runtime (`init-mira.sh` seeds keys via root token). |
| Vault POST-check *pattern* (`92d768c`) | validate required service-config fields at startup instead of only `valkey_url`. **Adapt the field list** — crm's demands `crm_base_url`, `crm_lifecycle_service_secret`, `square_application_id/secret`, `stripe_*`, `email_gateway_*`. |
| Bounded gate + park (`2f980ab`), temperature fix (`b88f076`) | §6.1.4. Make park-vs-exit configurable. |
| `clients/vault_client.py` re-auth (`770d89a`) | WP1 item 7. |

Omit: `compose.yaml` / `compose.dev.yaml` / `compose.crm.yaml` (project name `crm_mira`, image
`crm-mira:local`, `crm:8000` Docker DNS, all `mira_*_secret` entries including stripe/square),
`scripts/deploy_remote.sh`, `deploy/docker/DROPLET.md`, `droplet.env.example`, the container rewrites
in `init-mira.sh` / `container-setup.sh` / `Dockerfile.base` / `Dockerfile` / `s6-rc.d/postgresql/run`
(rewritten around single-provider OpenRouter and `/opt/crm_mira/postgres`; OSS's container path still
supports the Anthropic/offline wizard), `deploy/migrate.sh` systemd unit renames, and `748b90b` /
`fdaf69b` / `fac1e77` / `84c1eb8` / `7c47a50` / `fd1cb24` (all fixes inside the private deploy script).

`deploy_remote.sh`'s *ideas* are worth publishing separately as a parameterised env-driven template
(`SERVER` / `DEPLOY_DIR` / `BRANCH`): refuse deploy with uncommitted tracked changes; push to a bare
remote then fetch+reset; bounded post-deploy probe retries; preserve unexpected server-side changes
before reset. The file itself hardcodes `SERVER=admin@192.168.1.9`, `DEPLOY_DIR=/opt/crm_mira`, branch
`crm_mira`. Not appropriate for the OSS repo.

`compose.dev.yaml`'s source-mount dev pattern (`2ac4660`) is a good idea; the compose files as written
are CRM. The same commit's auth cookie policy (`SameSite` Strict→Lax) **is** portable and arrives with
WP3.

JS tooling: `package.json`, `package-lock.json`, `eslint.config.js`, `web/tsconfig.json` exist for the
new frontend and the `daf8e4a` quality gates. The eslint/tsc half is not yet relevant; the **mypy** half
(auth type errors) becomes relevant once WP3 lands. `daf8e4a` also lowercased `CookieSettings.samesite`
for Starlette 0.37.2 validation — arrives with WP3.

`docs/MANUAL_INSTALL.md`, `README.md`, `AGENTS.md`, `NEARFUTURE_FEATURES.md`,
`memory-curation-v2-serf-implementation-brief.md`: crm rewrote README for the CRM fork (`0dafc3a`,
+61/−135) and deleted `NEARFUTURE_FEATURES.md` and the serf brief. **Decline both deletions** — retain
OSS's README and its roadmap docs; update them for what actually lands. Root `AGENTS.md` (+48/−20)
contains private ops detail (§8.1 scrub list).

Per-directory `AGENTS.md`: crm modified 16 and added 6. Add `auth/AGENTS.md` (with the CRM-workspace
invariant clause removed) and `scripts/AGENTS.md` (minus the `deploy_remote` entry). Skip
`billing/AGENTS.md`, `web/AGENTS.md`, `web/assets/AGENTS.md`, `workphone/AGENTS.md`. Update the 16
modified maps **only for what actually lands**; do not merge them.

Documentation defects not to inherit: `tools/implementations/AGENTS.md` still documents
`phoneafriend_tool.py` (deleted upstream); `config/prompts/AGENTS.md:31` still documents
`repulsion_rewriter_system.txt` / `_user.txt` with consumer `FeedbackDomainHandler` (deleted upstream);
schema `COMMENT ON TABLE model_configs` says three routes where the CHECK allows five;
`clients/AGENTS.md` says *"Callers pass exactly one of primary, fast, or batch"*;
`cns/services/AGENTS.md` documents the `model_config` route per service (useful as a mapping source,
not portable as-is).

---

## 9. Reverse direction — the SSRF gap in crm_mira (record; WP7 descoped)

Retained as a factual record of what crm_mira lacks and how to close it at unification. **No action in
this programme** — see §9.1.

`e401d59` — SSRF, **GHSA-rmgf-f8wc-rc3p**. Two bypasses in the web tool's outbound-request guard:

1. `_validate_public_ip` relied solely on CPython `ipaddress.is_global`, which returns **True** for IPv6
   transition addresses embedding an IPv4 address — NAT64 (`64:ff9b::/96`, RFC 6052) and deprecated
   IPv4-compatible (`::/96`, RFC 4291). A hostname with an AAAA record of `64:ff9b::a9fe:a9fe` passed
   validation while embedding **169.254.169.254**. `is_global` does catch IPv4-mapped, 6to4 and Teredo;
   only these two ranges slip through.
2. Validation resolved DNS via `socket.getaddrinfo`, then the HTTP client resolved DNS **again** at
   connect time. An authoritative server can answer differently between the two lookups, permitting
   rebinding to an internal address after validation passed. **Deployment-independent.**

Fix: `http_client.pinned_request(method, url, hostname=validated.hostname, ip=validated.resolved_ip, …)`
replacing the dict-dispatch `{"GET": http_client.get, …}[method](url)`; depends on
`url_safety.ValidatedURL.resolved_ip`, the NAT64/IPv4-compatible rejection, and
`http_client.pinned_request` / `PinnedHTTPTransport` (`http_client.py:340,350`).

**crm_mira carries the pre-fix, vulnerable versions of all three files.** It branched at `24abe03`,
before `e401d59`; it never touched `utils/url_safety.py` or `utils/http_client.py` (empty diff); its
`_request_with_validated_redirects` (`web_tool.py:625`) still uses the unpinned dict-dispatch at
`:656-657`.

**Forward-port five paths:** `utils/url_safety.py`, `utils/http_client.py`,
`tools/implementations/web_tool.py`, `tests/utils/test_url_safety.py`, `tests/utils/test_pinned_http.py`.
Both test files are tracked at main (`git ls-files tests/` confirms) and absent upstream.
`test_url_safety.py` asserts `ipaddress.ip_address(addr).is_global is True` then
`pytest.raises(ValueError, match="transition address")` for `64:ff9b::a9fe:a9fe`, `64:ff9b::a00:1`,
`64:ff9b::7f00:1`, `::c0a8:1`, `::a9fe:a9fe`.

**Merge mechanics:** crm's `web_tool.py` change occupies a **disjoint region** (~`:415-440` — removes
`import os`, deletes `synthesis_model`/`synthesis_endpoint`/`synthesis_api_key_name` config fields and
the Vault key lookup in `_synthesize_content`, switches synthesis to
`LLMProvider().generate_response(model_config="fast", …)`) from the SSRF fix (~`:686`). A three-way
merge auto-merges cleanly and preserves pinning. A wholesale
`git checkout crm_mira/crm_mira -- tools/implementations/web_tool.py` in the backport direction reverts
the fix — hence §8.3.

The synthesis simplification becomes portable **only after WP2** lands `0134d3d`'s `model_config=` API.
At that point apply it as a surgical edit to `_synthesize_content` and its config block, leaving
`_request_with_validated_redirects` untouched. Until then main's version is correct.

**Other outbound surfaces audited, no action:** `utils/nominatim_client.py` SSRF-impossible by
construction (§6.2.2). `weather_tool.py` (config default `https://api.open-meteo.com/v1/forecast`, fixed
service, plus Nominatim geocode on a fixed host) and Home Assistant remain deliberately unpinned per
main's own doctrine — *"their targets are user-configured services, not untrusted URLs."*
`clients/square_client.py` / `twilio_client.py` are fixed vendor endpoints and omitted anyway.

Also forward-port, if the WP1 items are judged generic wins: `55820a3` is already in crm;
`770d89a`, `f8cfea9`, `28f25c8`, `2e60260`, `d3e5f20` all originate there. The only items flowing
OSS→crm are the SSRF fix and its tests.

### 9.1 WP7 descoped

**Status: descoped by decision (2026-09-05).** crm_mira's exposure is known and accepted; no
forward-port will be made as part of this work.

Basis, per §0.2: crm_mira will be unified onto the 2.0 frame, and 2.0 carries `e401d59`. The gap
closes at unification by inheritance rather than by a separate patch now. This is a **deferred**
closure, not a permanent acceptance — the exposure persists in the deployed CRM instance until that
unification happens. Record it as such so the deferral is visible rather than silently dropped.

Retain this section's technical content. When unification happens, the merge mechanics still apply:
crm's `web_tool.py` synthesis change occupies a disjoint region (~`:415-440`) from the SSRF fix
(~`:686`), so a three-way merge preserves pinning, while a wholesale checkout in either direction does
not.

---

## 10. Open items requiring decision

| ID | Item | Options | Blocking |
|---|---|---|---|
| **O-2** | `assessment` route semantics | §6.1.3 judgment call: `assessment_extractor` → route `assessment` (this plan) or → `batch` (upstream's mapping). §0.2 reinforces the former — name the route after its actual OSS consumer rather than preserving crm's autonomy-gate semantics for a service OSS omits. | WP2 |
| **O-3** | `other` route enforcement | Startup assertion when `other.model == primary.model`: warn or hard-fail? | WP2 |
| **O-4** | `MIRA_DEV` double-read | `main.py:99` already reads `MIRA_DEV` for hypercorn dev config; `auth/dev_mode.py:6` reads it for auth. Unify, or split into `MIRA_DEV` and `MIRA_AUTH_MODE`. | WP3 |
| **O-5** | Park-vs-exit on gate failure | crm parks (`while True: sleep(300)`) because s6 restart-on-exit caused a probe loop pinning two GPUs at 120–190 W. systemd operators may prefer exit+restart. Make configurable, or pick one. | WP2 |
| **O-6** | `history.js` rewrite timing | In the D2 patch budget, or deferred under D-6 with the drawer and conversation export broken. | WP4 |
| **O-7** | `invokeother_tool` synthetic input shape | crm's `lifecycle.py` changed `{"mode":"load","query":…}` → `{"load":[…]}`. Verify against OSS's `invokeother_tool` schema before porting `lifecycle.py`. | WP2 |
| **O-8** | "LoRA" naming | Describes fine-tuning that does not exist (§1). Rename the user-model subsystem. Independent of this backport. | none |
| **O-10** | Valkey flush whitelist | **Unverified.** `main.py:296` flushes caches on startup *"except auth sessions and rate limiting"*; `e03b569` extended the whitelist with `demo_admission:`. Confirm `clients/valkey_client.py:355` covers `session:` and `csrf:`, or every restart logs everyone out. Cheap check, high impact. | WP3 |
| **O-11** | `ProviderSwitchEvent` consumers | Grep before deleting the definition from `clients/llm/events.py`; `websocket_chat.py` may render `provider_switch`. | WP2 |
| **O-13** | `reminder_tool.py` datetime parameter | Determines whether D-8 is dormant or a live DST fix. Not read. | WP1 |
| **O-14** | `_prepopulate_welcome_content()` vs main's seeding | Not compared line-by-line. crm's `auth/database.py:235-361` vs `main.py:127-176`. `increment_segment_turn()` depends on `segment_turn_count` being present. | WP3 |
| **O-15** | `get_assessable_sections()` after WP6 | §7.1: the user model anchors observations to system-prompt `<section id=>` values. Verify `system_prompt_parser.py` still resolves after the seven prompt insertions. | WP6 |
| **O-16** | *"Don't make up file links. Write files to the sandbox."* | D10 removes `clients/files_manager.py`, so this prompt line needs **rewording**, not retention or deletion. | WP6 |
| **O-17** | `lt_memory/db_access.py:507` | Confirm the second `global_memories` reader is switched to `global_memories_runtime`. crm's `db_access.py` shrank 261 lines; the hunk was not isolated. | WP3 |
| **O-18** | `overwatch` token ceiling | Route default 16000 vs needed ~100 (`agents/base.py:330` via `overwatch_llm_key`). Legal via per-request `max_tokens=`. **Unverified** whether crm's `agents/base.py` actually passes the override, or whether output truncation is acceptable. | WP2 |
| **O-19** | `tests/` triage after WP0 | D-14: the restored pre-`ee44b18` suite covers removed subsystems (`files_manager`, batch coordinator) and, per §0.1, characterises **1.x** behaviour. Under 2.0 this is a re-baseline, not a repair: some tests will assert contracts that no longer exist. | WP0 |
| **O-20** | Deploy path reconciliation | OSS's upgrade path is `deploy/deploy.sh --migrate` (backup → fresh install → `schema_aware_restore.py` restores user data; `deploy/lib/migrate.sh:124-136` re-applies the old schema on rollback). With no migration requirement this is **dead or must be repurposed**. crm retained `deploy/migrate.sh`, `deploy/lib/migrate.sh` and `deploy/schema_aware_restore.py` alongside its greenfield schema — an inconsistency not worth inheriting. Decide: delete all three, or keep a backup/export path for users who want to salvage 1.x data manually. | WP6 |
| **O-21** | `VERSION` and release identity | `VERSION` is `2026.06.25` (CalVer) in **both** repos. Establish the 2.0 marker and whether the scheme becomes semver (`2.0.0`) or stays CalVer with a major suffix. Touches `README.md`, `docs/MANUAL_INSTALL.md`, and any `AGENTS.md` header. | WP6 |

---

## 11. Scrub gate

**No credential values were ever committed.** A history-wide `-S`/`-p` sweep of crm_mira for
`VAULT_ROLE_ID=`, `role_id:`, `secret_id:`, root-token assignments, and standard key formats (`sk-`,
`ghp_`, `AIza`, `AKIA`, `xox*`, `AC`+32-hex Twilio SIDs, private-key blocks) found nothing. Twilio
numbers in code and tests are `+1555…` fiction; `droplet.env.example` uses RFC 5737 `203.0.113.10`; the
email gateway uses `mail-gateway.example.com`.

Working-directory leaks checked and clean: `export-20260804-182816.csv` (118 KB Square customer export)
is **not committed** — `.gitignore` `export-*.csv`, `git log --all -- <path>` empty; a full-history tree
sweep (`git rev-list --all | git ls-tree -r`) finds no `.csv`, `voice_corpus` or iMessage artifact ever
committed; `mira_email_gateway_forbiz.php` is untracked; `data/voice_corpus/` is gitignored (`350fd15`).
**No history rewrite required on either side.**

Must not reach the OSS repo:

| Value | Locations |
|---|---|
| `192.168.1.9` — private appliance IP (llama-server + git bare repo host) | `deploy/mira_service_schema.sql` `model_configs` seed (twice), `scripts/deploy_remote.sh:2,17`, root `AGENTS.md` Remote Server + Model routing, `scripts/AGENTS.md` |
| `admin@192.168.1.9`, `/home/admin/mira-origin.git`, `/home/admin/backups/` (with cutover date) | `AGENTS.md`, `scripts/deploy_remote.sh` |
| `/opt/crm_mira` — deploy dir, Vault dir, Postgres data dir | `deploy/config.sh:162-163`, `deploy/finalize.sh:18-19,124-125,233,299-308`, `deploy/lib/vault.sh` (`MIRA_VAULT_DIR` default), `Dockerfile.base:178-180,218`, `s6-rc.d/postgresql/run:8`, `init-mira.sh:15`, `compose.yaml:38`, `droplet.env.example:12,17`, `deploy_remote.sh:18`, `openrouter_opus_chat.sh:95` |
| `mirafor.biz`, `www.mirafor.biz` | `config/config.py` `cors_origins` default (`8fc6cad`) |
| `api.kimi.com`, `k3-256k`, `kimi_key` | `model_configs` seed |
| `Qwopus 27B Fusion` @ `:3090`, `llama_server_key` | `model_configs` seed, `AGENTS.md` |
| Vault AppRole `crm_mira`, policy `crm-mira-policy`, `/opt/vault/init-keys.txt`, KV layout listing `stripe_*` / `square_application_*` / `email_gateway_*` key names | `deploy/lib/vault.sh` (`auth/approle/role/crm_mira/…`), `AGENTS.md`. Names and paths, not values — but they map the private security topology. |
| Container/service names `crm_mira`, `crm-mira-vault*.service`, image `crm-mira:local`, sibling `../crm`, Docker DNS `crm:8000` | `compose.yaml`, `compose.crm.yaml`, `.env.example`, `deploy/migrate.sh` |
| Port `42069`, `MIRA_DEV=1` droplet exposure, *"allow TCP 42069 only from Taylor's current source IP"* | `deploy/docker/DROPLET.md:5,23,30`, `droplet.env.example:5-6` |
| `/Users/taylut/Programming/GitHub/crm{,_mira}/…` personal absolute paths | `docs/workphone.md:118,139-143,223-224,234,399,486,566`, `scripts/extract_imessage_corpus.py:1` docstring (*"Taylor's 1:1 iMessage/SMS threads with known customers"*) |
| Real-looking customer emails, business names, pricing, live workspace UUID `a6cdbde9-c793-4c15-a209-2d16922a6266` | `scripts/seed_closeout_test.sql` — `delores.martinez@yahoo.com`, `lisa.brennan@gmail.com`, `sarah.okafor@gmail.com`, `kevin@tranauto.com`, `maria@tacodelmar.com`, `rwhitfield@sunstateoffice.com`, `dkim@propertygroup.com`, `billing@acmeroofting.com`, `contact@acmeroofting.com`, "Storm Window Cleaning". Phones are 555-fiction; names and domains may mirror real clients. **Omit the file entirely.** Also `web/card-playground.html` (acmeroofting). |
| `taylor@admin.site`, `taylor@crm-mira.local`, `dev@crm-mira.local` | `web/error/index.html`, `auth/service.py:216-231` dev fixtures |
| `taylorsatula/crm_mira` repo topology, `taylorsatula` user | `AGENTS.md` |

Root `AGENTS.md` requires particular care: `d4f9770` / `71bef83` expanded its Remote Server section into
a full deploy reference — IP, SSH user, bare-repo path, deploy dir, backup locations and dates, Vault
access recipes, AppRole file paths, root-token location, policy semantics, KV layout, container/volume
names, and the private llama-server. Port at most the generic principles (Vault-only credentials,
no-fallback); strip everything host-specific.

`.gitignore`: crm's additions are correct (`deploy/docker/secrets/*` with a README exception,
`data/voice_corpus/`, `export-*.csv`, node tooling). **The tail contains accidental junk from a bad
paste** — `@pytest.fixture`, `@pytest.mark.parametrize`, `@router.delete/get/patch/post/put`. Harmless
as patterns; do not port. Retain OSS's `_crm_client.py` / `crm_*_tool.py` entries (§8.3).

`config/vault.hcl`: dev Vault ports moved 8200→8210 and 8201→8211 — droplet co-tenancy specific. Omit.

Python layer is clean: no hardcoded model IDs, provider URLs or credentials in crm's `*.py`.

---

## 12. Sequence

| Phase | Content | Gate |
|---|---|---|
| **WP0** | Recover crm's six generic contract tests from `95254a5^`; `test_openai_tool_schema_validation.py` + `test_orchestrator_tool_loop.py` from HEAD; mira-OSS's pre-`ee44b18` suite; `scratch/conftest.py`. Repair enough to run. Triage per O-19. | Suite runs. **Blocking.** |
| **WP1** | Items 2–17 (§6.2). Ship as independent commits. Order within the phase is unconstrained except: item 3 (`a4df669` RLS lines) must precede WP-S enabling RLS on `users`; item 16 (feature flags, generalised) should land before WP5 so `MIRA_PERSONA_ENABLED` exists. Close O-10, O-13 alongside. | Each commit independently revertible; `test_cognitive_feature_bypasses.py` and `test_tool_config_resolution.py` green. |
| **WP-S** | **Greenfield schema — the single DDL deliverable for the programme.** Derive `deploy/mira_service_schema.sql` from crm's file per §6.3.7: empty-database `DO` guard, extensions, externally-provisioned roles, `set_updated_at()` / `set_search_vector()`. Strip all CRM/billing objects. Add back `feedback_signals` (OSS column set), `feedback_synthesis_tracking`, `usage_pricing`, `domain_knowledge_blocks` + `_content`. Include `model_configs` (§6.1), `users.subject_kind` + crm's soft-delete columns (`deletion_requested_at`/`soft_deleted_at`/`purge_deadline`, replacing `users_trash`), RLS on `users`/`magic_links`/`api_tokens` with the `NULLIF` predicate, `global_memories_runtime` + `can_read_global_memories()` (§6.3.8), `persona_revisions` / `persona_state` / `persona_signals` + `provision_baseline_persona()` trigger (§6.5.3), `user_feedback` (§6.2.4). Omit `conversation_llm`, `internal_llm`, batch tables, `users_trash`, `enforce_member_global_username()`. **Delete `deploy/migrations/` entirely.** Rewrite `ensure_single_user`'s `UPDATE users SET balance_usd = …, conversation_llm = …` (`main.py:59-64,77-82`) and repoint offline-mode seeding from `deploy/postgresql.sh:103` `OFFLINE_SQL` to `model_configs` rows. Port `a9fd443`'s `schema_distribution.py` deletion. Resolve O-20. | Schema applies to an empty database; `power_on_self_test` passes; no `deploy/migrations/`; WP1 item 3 landed first; RLS canary in place. **Blocking for WP2, WP3, WP5.** |
| **WP2** | `model_configs` (§6.1) — **code side only; DDL is WP-S.** One planned atomic unit: `user_context.py` + `resolver.py` + `llm_provider.py` + `lifecycle.py` + `types.py` + `events.py` + ~17 leaf call sites, using §6.1.3 as the checklist. Include D5's re-keyed cost hook, D13's effort override + picker retirement, R2's prompt edit, D10's deletions, and the §6.1.7 seed scrub. Fix the stale schema `COMMENT` and `clients/AGENTS.md`. Resolve O-2, O-3, O-5, O-7, O-11, O-18. | Five routes load; POST gate passes; cost recording works; no `internal_llm` / `conversation_llm` references remain. |
| **WP3** | Multi-user auth (§6.3) — **DDL is WP-S.** Dependency order: `exceptions.py` → `types.py` (crm's field requirements as written) → `dev_mode.py` + new `auth/mode.py` → `config.py` (CRM fields removed, email lazy) → `security_logger.py` → `rate_limiter.py` → `session.py` → `database.py` (`create_user` INSERT matches WP-S directly) → `account_gc.py` (with `NullProvisioner`) → `provisioning.py` → `service.py` → `api.py` → `base.py`. Then the pluggable SMTP sender. Then `a4df669`'s WS auth shape. Then the `global_memories_runtime` Python companion (`hybrid_search.py:169`, verify `db_access.py:507` per O-17), `get_active_segments` `is_active`, and deploy Vault seeding. Resolve O-4, O-14. | All 7 `cns/api/*` consumers unchanged; `web/` unchanged; server boots identically with `MIRA_AUTH_MODE` unset; `test_auth_graft.py` green; `multi` mode signs up two users over SMTP with RLS isolation verified. |
| **WP4** | WebSocket protocol (§6.4). Transport + strict frames + ordered persistence + halt, with R5/R6/R8 handled. Then the ~150-line frontend patch and cursor-only keyset pagination with `history.js` rewritten (D-2, O-6). Server-side may land first under the breakage allowance (D-6). | `test_ordered_turn_persistence.py`, `test_web_frontend_protocol.py`, `test_history_cursor.py` green; all eight §6.4.1 defects demonstrably fixed; browser chat works end to end. |
| **WP5** | Persona as second system (§6.5) — **DDL is WP-S.** Port the three modules + seven prompts behind `MIRA_PERSONA_ENABLED`. Split the prompt slot in `composer.py`. Change `persona_trinket.py`'s `variable_name` and `_invalidate_cache`'s hdel field to `persona_directives`. Add `PersonaDomainHandler` with new action names + `DataType.PERSONA`. Retain `_process_feedback_loop`, `_init_feedback_loop`, `_invalidate_lora_trinket_cache` in `segment_collapse_handler.py` (hand-edit; do not take the file). | Both trinkets render; both prompt slots populated; user-model check-in still functions; the `provision_baseline_persona()` trigger creates revision 1 for every new user including the `single`-mode bootstrap. |
| **WP6** | Prompt + docs + release identity (§6.6, §8.5). Hand-merge the seven `61315bb` deltas; run the em-dash and contrastive-negation self-check; resolve O-15, O-16; update per-directory `AGENTS.md` maps for what actually landed; run the §11 scrub. **2.0 obligations (§0.1):** rewrite `README.md` for 2.0 and state explicitly that 1.x → 2.0 is a **reinstall, not an upgrade** and that 1.x conversation history, memories and domain knowledge do not carry forward; update `docs/MANUAL_INSTALL.md`; bump `VERSION` per O-21. | No private host, path, domain, model ID or personal name in any shipped file. O-15 verified. Upgrade policy stated. |
| **WP7** | **Descoped** (§9.1). crm_mira's SSRF exposure is known and accepted; the gap closes when crm_mira unifies onto 2.0 and inherits `e401d59`. §9 retains the technical detail and merge mechanics for that point. No action in this programme. | none — deferred closure, recorded so the exposure stays visible until unification |

Cross-phase: WP-S is the single DDL deliverable and blocks WP2, WP3 and WP5 — author it once the
object lists in §6.1.3, §6.3.7 and §6.5.3 are settled, then let those phases consume it rather than
each writing DDL. WP2 invalidates parts of WP1's route kwargs and WP5's `model_config="primary"` call
sites (`persona_service.py:92,188,266,281`). Land WP2 before WP5. WP1 items 11–14 (`clients/llm/*`) are
independent of WP2 except for the `ProviderMetadata` rename (`internal_llm_name` /
`conversation_llm_name` → `model_config_name`, touching `RequestMetadata`/`ProviderMetadata` consumers
including `orchestrator.py:719-724` continuation metadata) — a trivial conflict, resolvable either way.

**Branch mechanics (decided): one git worktree per phase.** All intermediate commits stay local;
broken intermediate states are acceptable and expected. 2.0 is verified and finalised locally, then
pushed to `main` as a finished whole.

WP-S deletes the migrations directory and drops `conversation_llm`, so the tree between WP-S and WP2
does not boot. Non-booting intermediates never reach `main`, so no phase gating or flag-holding is
needed to keep the trunk green.

Operating notes:

| Concern | Handling |
|---|---|
| Worktree layout | One per phase. `.worktrees/` is already gitignored at main. WP1 needs no worktree — it lands as independent commits and is the only phase that could ship on its own. |
| Dependency order is still real | WP-S blocks WP2, WP3, WP5 (§12). Worktrees isolate *breakage*, not *ordering*: WP2 cannot be authored before WP-S settles the object list. |
| Integration gate | Single, local, after WP6. The suite recovered in WP0 plus crm's six contract tests are the acceptance set. Nothing pushes before it passes. |
| The 2.0 cut | `main` receives a finished tree, not a series. Record the upgrade policy (reinstall, not upgrade — §0.1) in the same push, since it is the first thing a 1.x user needs to know. |
| WP1 as a separate deliverable | WP1 is 16 self-contained fixes with no dependency on WP-S. It can be merged to `main` ahead of the rest if you want value landing sooner — it is the only phase for which that is true. |
