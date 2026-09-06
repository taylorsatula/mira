# Session handoff — mira-OSS 2.0 backport execution state

Written 2026-09-05. This document carries **execution state only**. It deliberately does not restate
specification, because two other documents on disk survive compaction and are authoritative for that:

| Document | Role |
|---|---|
| `.pi/plans/20260905-172000-crm-mira-backport-bisect.md` | **The specification.** Divergence topology, decisions D1–D14, deferred register D-1…D-17, work packages WP0–WP7 with acceptance gates, omit/keep registers, scrub gate, open items O-1…O-24. ~2,090 lines. Read §0.1 and §0.2 first — they set the posture and the precedence rule everything else depends on. |
| `.pi/plans/20260905-REMAINING-WP-PROMPTS.md` | **Paste-ready subagent prompts** for WP2-A, WP2-B, WP3-A, WP4, WP5, WP6, plus a skeleton for WP3-B and the shared conventions block (authority, report format, commit convention, differential-test methodology). |
| this file | **Where execution actually stands**, what is verified, what is assumed, and how to resume. |

All three live under `/Users/taylut/Programming/GitHub/mira-OSS/.pi/plans/`.

---

## 1. Mission

Backport general-purpose improvements from `crm_mira` (proprietary descendant, 162 commits ahead of
merge-base `24abe03`) into `mira-OSS` (the distributed open-source package), omitting the CRM product.
The result is **mira-OSS 2.0**: no migration path, no backwards compatibility, fresh install only.

Two standing constraints from the user that shape everything:

- **mira-OSS is the primary artifact.** crm_mira is one variant its author built for himself. Optimise
  for mira-OSS; do not carry compromises that serve the CRM variant. crm_mira will later be unified onto
  the 2.0 frame, and at that point **crm adapts to mira-OSS**, not the reverse. Consequently crm's names
  and contracts are proposals, not authority (plan §0.2).
- **Temporary breakage is permitted.** mira-OSS may be left non-booting between packages. All
  intermediate commits stay local. 2.0 is verified and finalised locally, then pushed to `main` as a
  finished whole. **Nothing has been pushed and nothing may be pushed.**

---

## 2. Repository topology

Root: `/Users/taylut/Programming/GitHub/mira-OSS`. Reference-only sibling:
`/Users/taylut/Programming/GitHub/crm_mira` (never modified; read via `git show` against the fetched
`crm_mira` remote, which resolves from inside every worktree).

| Path | Branch | HEAD | Status |
|---|---|---|---|
| `.` (main checkout) | `main` | `2bb2807` | clean. Carries the three plan documents and the `scratch/` gitignore rule. **No 2.0 code.** |
| `.worktrees/wp1` | **`2.0/integration`** | **`ec5e50b`** | **clean — this is the current tip of all 2.0 work.** 31 commits ahead of main. Note the directory is still named `wp1`; the branch was renamed to `2.0/integration` in place. |
| `.worktrees/wp-s` | `2.0/wp-s` | `1326633` | clean, 7 commits. **Merged** into integration. |
| `.worktrees/wp0b-testsplit` | `2.0/wp0b-testsplit` | `013476c` | clean, 1 commit. **Merged** into integration. |
| `.worktrees/wp0-tests` | `2.0/wp0-tests` | `17d6f7b` | clean, 4 commits. **Merged** (into `2.0/wp1`, which integration descends from). |
| `.worktrees/wp1-surgical` | `2.0/wp1-surgical` | `1d6dae0` | clean, 7 commits. **Merged.** |
| `.worktrees/wp1-tools-config` | `2.0/wp1-tools-config` | `882c94c` | clean, 4 commits. **Merged.** |

`2.0/integration` vs `main`: **137 files changed, +27,669 / −2,036.**

Branch mechanics decided by the user: **one git worktree per phase**, all intermediate commits local,
broken intermediates acceptable, single local integration gate before `main` receives a finished tree.
`.worktrees/` is gitignored. Merged sub-branches are retained (not deleted) so any package can be
inspected or reverted in isolation.

Recreate a worktree for the next package with:

```
cd /Users/taylut/Programming/GitHub/mira-OSS
git worktree add .worktrees/<name> -b 2.0/<name> 2.0/integration
```

---

## 3. Completed and verified

### WP0 — test asset recovery ✅ (`2.0/wp0-tests`, 4 commits, merged)

74 files recovered. **13** generic crm_mira contract tests from `95254a5^` (the original brief named
only 6; the omission was caught and corrected mid-flight) plus `tests/AGENTS.md`, **2** from crm HEAD,
and **38** mira-OSS test files from `ee44b18^`. All 9 CRM/billing tests correctly excluded.

Result: **689 tests enumerated**, 18 collection errors correctly attributed. Produced
`tests/TRIAGE.md` (~20 KB) classifying all 55 test files and mapping each to the package it gates.

Its most valuable finding: `tests/test_greenfield_schema.py` **passed vacuously** against mira-OSS. It
matched tables with `rf"CREATE TABLE\s+{table}\b"` while mira-OSS writes `CREATE TABLE IF NOT EXISTS`,
so its absent-table assertions held while all eight tables it demanded absent were present. That would
have given WP-S a false green light. Fixed by WP-S; non-vacuity now proven by mutation test (§4).

### WP1 — all 16 items ✅ (three sub-branches, 15 commits, merged)

Split by file ownership into three disjoint packages so they could run in parallel; all three merged
with **zero conflicts**.

**WP1-A** (`2.0/wp1-surgical`, 7 commits) — items 2, 3, 6, 7, 8, 9, 10. Verified independently:
exactly 8 files touched; all 7 specified exclusions empty; `_init_contacts_schema` survived the
`f8cfea9` trap; no `billing` import in the poller fix; the `state.stop_event is stop_event` identity
guard present. The DST utility was **functionally** tested, not just compiled: correct EDT (−4) and EST
(−5) offsets, raises on the 2026-03-08 spring-forward gap and the 2026-11-01 fall-back overlap, rejects
`Z`-suffixed input.

**WP1-B** (`2.0/wp1-tools-config`, 4 commits) — items 4, 15, 16, 17. Verified: `google_maps_api_key` → 0
refs; `analysis_enabled` → 0 refs; `config.api.max_tokens` → **3 refs preserved**, proving it declined
the WP2 fragment of `6899d07` as instructed; zero `model_config` leakage into `orchestrator.py`; all
three contaminated `97951be` orchestrator hunks correctly declined (`gpt-legacy` block, invokeother
argument shape, `max_tokens = 31999` all still present). Nominatim policy compliance verified on all
seven points; hosts are hardcoded constants so SSRF is impossible by construction. `feedback_tool.py`
fully de-branded (0 CRM refs).

**WP1-C** (`2.0/wp1`, 2 commits) — items 5, 11, 12, 13, 14. **This package's subagent was killed by
provider quota exhaustion after applying all seven patch steps but before committing.** I completed it
manually: the missing `clients/AGENTS.md` hand-edit, verification, and both commits. Verified:
`persisted_tool_ids` absent (WP4 hunk correctly declined); `chat_template_kwargs` absent (proving
`ed441e5` netted out `a7b8694`); `memory-curation-v2-serf-implementation-brief.md` preserved; `tests/`
and `web/` untouched.

Whole-tree checks on the merged WP1 line: **all 277 Python files compile**, all six Vault-independent
modules import cleanly, zero dangling references to removed symbols.

### WP-S — greenfield schema ✅ (`2.0/wp-s`, 7 commits, merged)

The critical-path package; WP2, WP3 and WP5 all depend on it. Verified against the specification:

- **Exactly the specified 21 tables**: `model_configs`, `usage_pricing`, `users`, `magic_links`,
  `api_tokens`, `user_activity_days`, `domain_knowledge_blocks`, `domain_knowledge_block_content`,
  `domaindoc_shares`, `continuums`, `messages`, `memories`, `global_memories`, `entities`,
  `persona_revisions`, `persona_state`, **`persona_signals`**, `feedback_signals`,
  `feedback_synthesis_tracking`, `global_usernames`, `user_feedback`.
- All CRM, billing, channel, `audit_events`, `conversation_llm`, `internal_llm`, `users_trash`,
  `extraction_batches`, `users_subject_contract` and `enforce_member_global_username` **absent**. The
  single `users_trash` string occurrence is a comment explaining what replaced it.
- `persona_signals` rename applied (8 refs) so it no longer collides with the retained user-model
  `feedback_signals`.
- **RLS: 38 fail-closed `NULLIF(current_setting('app.current_user_id', true), '')` predicates, 0
  throwing-form predicates.**
- `model_configs` CHECK is `('primary','fast','batch','assessment','other')` — **`other`, not crm's
  `difficult`**, per D14. `COMMENT ON TABLE` corrected from crm's stale "Exactly three" to an accurate
  five-route description naming `other` as "a sidebar turn routed to an outside model, deliberately a
  different vendor from primary".
- **Seed is fully scrubbed** and matches the specification exactly. `primary` = openrouter
  `openai/gpt-5.5` / `provider_key` / high / 16000; `fast` = groq `qwen/qwen3.6-27b` /
  `subcortical_key` / none / 4096; `batch` = anthropic `claude-sonnet-4-6` / `anthropic_batch_key` /
  high / 16000; `assessment` = anthropic `claude-opus-4-6` / `anthropic_key` / none / 10000;
  `other` = openrouter `google/gemini-3.1-pro-preview` / `provider_key` / high / 10000. Every model and
  Vault key name is one mira-OSS already referenced at main. **`other` (gemini) differs in family from
  `primary` (gpt-5.5), satisfying D14's constraint.**
- `deploy/migrations/` deleted in full. `tools/schema_distribution.py` deleted (`a9fd443`), with
  `contacts_tool.sql` retained.
- `global_memories_runtime` + `can_read_global_memories()` + `security_barrier` present.
  `provision_baseline_persona()` trigger present.
- `main.py` `ensure_single_user()` rewritten for the retired tier columns; `deploy/postgresql.sh`
  offline seeding repointed at `model_configs`.
- RLS canary added (`1326633`) warning when a query runs without user context.
- **Scrub gate clean**: no `192.168.1.9`, `/opt/crm_mira`, `mirafor.biz`, `Qwopus`, `k3-256k`,
  `kimi_key` or `llama_server_key` anywhere in `deploy/`, `main.py` or `utils/`.

I steered this agent mid-run with one addition it would otherwise have missed: `utils/power_on_self_test.py`
holds a hardcoded RLS `expected_tables` list (and a matching SQL `IN (...)` list) containing
`billing_transactions`, which the new schema drops — left alone it would fail the power-on self-test at
**every startup**. Both lists were updated (verified: `billing_transactions` → 0 refs,
`persona_signals`/`user_feedback` → 4 refs). This is now recorded in plan §6.3.7.

### O-22 — test harness split ✅ (`2.0/wp0b-testsplit`, 1 commit, merged)

The recovered suite could not run at all: `tests/conftest.py`'s autouse `reset_test_environment` calls
`full_reset()`, which builds a Vault client and DB pools, so **all 689 tests errored at setup**. The
failure chain is `VAULT_ADDR` → `VAULT_ROLE_ID`/`VAULT_SECRET_ID` → real AppRole login →
`PermissionError` on connection refusal.

Fixed by splitting the suite on an `integration` marker with an env-probe (not a network probe). Touched
only harness paths: `tests/conftest.py`, `tests/AGENTS.md`, `tests/fixtures/{auth,core,infra,isolation}.py`.
No test assertions modified.

---

## 4. Verification methodology — reuse this

Two techniques proved their worth and should be applied to every remaining package.

### Differential testing

The suite contains tests for code that does not exist yet, so a raw pass/fail count is meaningless.
Compare two branches differing only in the package under test:

| Branch state | passed | failed | skipped | errors |
|---|---|---|---|---|
| tests + harness, **WP1 code absent** | 150 | 143 | 396 | 18 |
| tests + harness, **WP1 code present** | 164 | 129 | 396 | 18 |
| **+ WP-S and O-22 (current `2.0/integration`)** | **195** | **121** | **396** | **18** |

Each step improved the delta with **no increase in failures and an identical skip/error count** — the
signature of a clean package. A package that raises the failure count has regressed something.

Command: `python3 -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -3` (~9 s).

To construct a baseline branch for a future package, branch from `2.0/integration` and revert only that
package's commits, or cherry-pick the recovered tests onto an earlier base (this is how the WP1 baseline
was made: copy the harness files into `.worktrees/wp0-tests`, run, then `git checkout -- tests/ && git
clean -fdq tests/` to restore).

### Mutation testing a gate

A green gate proves nothing unless it can go red. For `test_greenfield_schema.py` I appended
`CREATE TABLE conversation_llm` and `CREATE TABLE crm_workspaces` to the schema and confirmed **3
failures** including `test_crm_and_billing_sql_does_not_reappear`, then restored from a backup and
confirmed 32 passed and a clean tree. Apply the same discipline to any gate that asserts absence.

### Residual failure accounting (current, 121 failed / 18 errors)

- ~23 are **characterization tests for unlanded packages** and *should* fail: `test_web_frontend_protocol.py`
  (7) → WP4, `test_history_cursor.py` (6) → WP4, `test_orchestrator_tool_loop.py` (2) → O-23/O-7, plus
  `test_model_routing.py` / `test_direct_extraction.py` → WP2 and `test_persona_service.py` /
  `test_viewcard_content.py` → WP5 / D-4 (these are collection errors).
- ~98 are **pre-existing 1.x breakage**, proven pre-existing by the differential baseline above. The
  mira-OSS suite was already failing when `ee44b18` deleted it. Largest concentrations:
  `clients/test_sqlite_client.py` (28), `working_memory/test_notification_center.py` (22),
  `cns/services/test_fingerprint_generator.py` (16), `working_memory/test_user_name_substitution.py` (8),
  `working_memory/test_trinket_access.py` (7), `tools/test_gated_tools.py` (6).
- Plan O-19 makes the 1.x suite a **re-baseline, not a repair**. Nobody has done that triage yet; it is
  unassigned and should be scheduled, ideally alongside WP6.
- The 18 collection errors include `tests/utils/test_prompt_injection_defense.py`, which still raises
  `ValueError: VAULT_ADDR` — the O-22 harness did not cover it. Worth a follow-up.

### Static gates that require no infrastructure

`python3 -m compileall -q .` (277 files, exit 0 on the current tree) and targeted import smoke tests of
Vault-independent modules. Use these every package; they catch merge damage cheaply.

---

## 5. What remains

Dependency chain, revised 2026-09-06: **WP2-A ✅ → WP2-C ✅ → WP2-B ✅ → {WP3-B ∥ WP5 ∥ WP2-D}
(in flight) → WP4 → WP6.** WP7 is descoped.

Also landed outside the original chain: **R-3** (deploy audit + three fixes), the **`effort='none'`
install-blocker fix**, and **WP1-D** (the dormant DST utility wired to its two real consumers, closing
D-8 and O-13). **WP2-D** was chartered mid-flight for a silent regression no gate could catch.

**Integration tip `47ff2da`, 69 commits ahead of main, 205 files, clean, compiles, differential
120 failed / 200 passed / 396 skipped / 16 errors.** For orientation: the programme began at
150/143/396/18.

The revision is structural, not cosmetic. Plan §12 treats WP2 as *one atomic unit that includes D10's
deletions*; the WP2-A/WP2-B split allocated the chokepoint and the leaf call sites but left D10 —
~1,900 lines of Batch removal plus ~500 of Files removal across 21 files — owned by nobody, **11 of
which no package claims at all**. That gap is now WP2-C, and it sequences *between* WP2-A and WP2-B
because WP2-B's headline gate (zero `internal_llm` refs tree-wide) is unreachable while
`lt_memory/llm_routing.py` still holds two refs that nothing else owns.

WP3-B and WP5 are parallel in wave 4 — verified disjoint owned sets. WP4 follows WP3-B because both own
`cns/api/websocket_chat.py`. WP3-B is prioritised ahead of WP4 deliberately: WP3-A's merge leaves
`auth/api.py:41-45` constructing a 3-arg `APITokenContext` against a type whose `subject_kind` is now
required, which raises `ValidationError` at **request time** — so `single` mode cannot serve any
authenticated request until WP3-B lands the §6.3.4 union branch. Permitted breakage, but it is the
highest-risk open item and should not sit.

| Package | Scope | Gate | Prompt |
|---|---|---|---|
| **WP2-C** | D10: delete the Anthropic Batch API transport and the Files API **upload** transport. 11 orphan files plus six deltas §8.4 never named. Upstream `5c50dfc`, `0134d3d`, `58c261b`. | `uses_anthropic_batch_dialect`/`batch_coordinator`/`extraction_batches`/`files_manager`/`force_immediate` → 0; the KEEP list still present; `test_direct_extraction.py` collects; `test_greenfield_schema.py` still 32 passed | complete, written 2026-09-06 |

WP3-A is disjoint from WP2 (new `auth/*.py` files only) and *could* be parallelised with WP2-A; it was
not launched because the user asked for drafted prompts rather than more work in flight.

| Package | Scope | Gate | Prompt |
|---|---|---|---|
| **WP2-A** | `model_configs` chokepoint: `utils/user_context.py` `ModelConfig`/`load_model_configs`/`get_model_config`, `clients/llm/{resolver,types,lifecycle,events}.py`, `clients/llm_provider.py`, `config/config.py` (`validate_compaction_budget`, remove `api.max_tokens`), `power_on_self_test._check_llm_configuration`, `cost_accumulator` re-key | `test_model_routing.py` progresses; 0 `internal_llm` refs in the chokepoint files | complete, in the prompts file |
| **WP2-B** | ~22 leaf call sites, the `agents/base.py` dataclass fields (`internal_llm_key`, `overwatch_llm_key`), 4 agent subclasses, `phoneafriend_tool` contract change, D13 picker retirement + effort override, `main.py` `usage_pricing` seeding, system-prompt substrate paragraph | `git grep -E 'internal_llm\|conversation_llm' -- '*.py'` → 0 tree-wide | complete |
| **WP3-A** | auth primitives: `session.py`, `exceptions.py`, `dev_mode.py`, `rate_limiter.py`, `security_logger.py`, `webauthn_service.py`, `types.py` delta, `config.py` (lazy email), plus **two files with no upstream equivalent**: `auth/mode.py` (MIRA_AUTH_MODE) and `auth/provisioning.py` (AccountProvisioner + NullProvisioner + local_teardown) | `import auth.mode, auth.provisioning, …` succeeds with no Vault and no env set | complete |
| **WP3-B** | `auth/{database,service,api,account_gc}.py`, the **new pluggable SMTP sender**, `main.py` three-mode bootstrap, WS auth via `a4df669`'s shape, deploy Vault seeding, excise ~7 CRM tests from `test_auth_graft.py` | `test_auth_graft.py` green; `multi` mode signs up two users over SMTP with RLS isolation verified | **skeleton only** — write the full brief after WP3-A reports, because it depends on how WP3-A resolved `app_url` laziness and the `auth.config` import constraint |
| **WP4** | strict WS protocol, ordered turn persistence, halt, keyset cursor, the ~150-line frontend patch, plus **R5/R6/R8** and closing **O-23** | `test_ordered_turn_persistence.py`, `test_web_frontend_protocol.py`, `test_history_cursor.py` green; all eight §6.4.1 defects demonstrably fixed | complete |
| **WP5** | Persona as a **second** system: `persona_service.py`, `persona_repository.py`, `persona_trinket.py` (with `variable_name = "persona_directives"`), 7 prompts, `MIRA_PERSONA_ENABLED`, `PersonaDomainHandler`, `DataType.PERSONA` | both prompt slots populated; user model provably untouched | complete |
| **WP6** | prompt harvest from `61315bb` (seven deltas, never HEAD), docs pass, **§11 scrub gate against the whole tree**, VERSION bump, README reinstall-not-upgrade policy, O-15/O-16/O-20/O-21/O-24 | scrub grep empty; em-dash and contrastive-negation self-check | complete |
| ~~WP7~~ | ~~forward-port the SSRF fix to crm_mira~~ | **DESCOPED** — crm_mira's exposure is known and accepted; the gap closes when crm unifies onto 2.0 and inherits `e401d59`. Plan §9.1. | n/a |

Two invariants that survive every package (plan §0):

1. **`cns/api/oss_ui.py` + `deploy/oss_ui/{marked.min.js,purify.min.js,chat.html}` must be retained.**
   `GET /oss-auth/token` is the identity source for `MIRA_AUTH_MODE=single`, the default. The module
   reads both vendor assets **at import time**, so deleting `deploy/oss_ui/` crashes `create_app()`.
2. **`tools/implementations/web_tool.py`, `utils/url_safety.py`, `utils/http_client.py` stay at main's
   version.** They carry `e401d59` (GHSA-rmgf-f8wc-rc3p). crm_mira has the pre-fix, vulnerable versions;
   a bulk copy from crm reverts a security fix.

---

## 6. Decisions taken during execution that are not in the plan

Recorded here because they were made under time pressure and are not yet folded into the specification.

| # | Decision | Basis |
|---|---|---|
| E-1 | **O-3 resolved**: the `other` ≠ `primary` seeding constraint is enforced by a startup **WARNING**, not a hard failure. | An operator with a single available provider must still be able to boot. Specified in the WP2-A prompt. |
| E-2 | **O-2 resolved**: `assessment_extractor` maps to route `assessment`, not `batch`. | crm mapped it to `batch` to avoid a name collision with its own effort=none autonomy gate. OSS omits `autonomy_service`, so the name is unclaimed and self-documenting. WP-S seeded it accordingly. |
| E-3 | **`phoneafriend_tool` loses its model-choice parameter** in WP2-B. | D14 collapses both voices onto one route, so `MODEL_INTERNAL_LLMS[model_choice]` at `:151` chooses nothing. A parameter that silently does nothing is worse than no parameter — the model would reason about a distinction that does not exist. Recorded in plan §6.1.3. |
| E-4 | **`billing_transactions` and `stripe_webhook_events` are dropped.** | I verified there are **no live application readers** in mira-OSS Python — the only references were `power_on_self_test.py` (updated) and `test_greenfield_schema.py` (amended). |
| E-5 | **`deploy/config.sh`, `deploy/deploy_database.sh`, `deploy/docker/scripts/init-mira.sh`, `deploy/finalize.sh` were modified by WP-S.** | Not explicitly in my brief; they reference migrations or roles and had to follow the schema. **Not yet reviewed line-by-line** — see §8 risk R-3. |
| E-6 | **`2.0/integration` is the de-facto integration branch**, created by renaming `2.0/wp1` in place. | The worktree directory is still `.worktrees/wp1`, which is confusing. Consider renaming the directory or documenting the mismatch. |
| E-7 | **All five routes are critical at startup** (user-confirmed). `ROUTE_FALLBACKS` is deleted outright with **no replacement criticality set**, and any provider probe failure parks boot. | crm's `ROUTE_FALLBACKS` was overloaded: it encoded a local-llama-server vs cloud split that WP-S's all-cloud OSS seeding makes vacuous, *and* `_check_llm_provider_reachability` used `c in ROUTE_FALLBACKS` purely as a criticality classifier. With every route critical there is nothing to classify. Accepted consequence: an install missing any one vendor's credentials cannot boot. No soft-failure escape hatch. The raised message must name each failed route, vendor and model so an operator can act on it. Resolves O-3 and O-11-adjacent concerns. |
| E-8 | **`MailSender` is an abstract interface with a stdlib `smtplib` default backend** (user-confirmed), sitting beside WP3-A's `AccountProvisioner`. | Matches the brief's "pluggable SMTP sender" wording and the provisioner pattern. Keeps OSS vendor-neutral (plan §0.1): no new dependency, works against any relay the operator already has. An HTTP backend (Resend/SES/Postmark) can be registered by an operator but is not shipped. To be written into WP3-B's brief. |
| E-9 | **No test support outside the migration scope** (user directive). Agents must not author tests, fixtures, conftest helpers or coverage; tests the migration invalidates are **deleted**, not replaced. | This codebase chooses fail-fast over extensive performative tests. Encoded once in the prompts file's shared conventions so every dispatch inherits it. Consequence: **O-19 (re-baselining the ~98 pre-existing 1.x failures) is dropped from the inline work list** as test work outside the migration. |
| E-10 | **WP2-C chartered for D10** (Batch API + Files API upload), sequencing between WP2-A and WP2-B. | See §5. Plan §12 specified WP2 as one atomic unit including D10; the A/B split orphaned it. Eleven files were claimed by no package. |
| E-11 | **mira-OSS keeps Anthropic code execution. Only the Files API *upload* side is removed.** `FileRefBlock`/`DocumentBlock`, the `file_ref` defensive handling, `anthropic.py`'s `file_ref→container_upload` translation, `_file_artifact_events` (the download side), `FILES_API_BETA_FLAG` and the `container_id` **write** side all stay. | OSS's anthropic routes are live (`batch` = claude-sonnet-4-6, `assessment` = claude-opus-4-6) and code execution is a retained feature. crm removed the read side only because *all* its routes were openai — porting crm's full container hunk would inherit a CRM artifact, contrary to §0.2. |
| E-12 | **`anthropic_batch_key` is kept and documented as reserved**, not renamed (closes deferred item D-10). | D10 removes the Batch API but the key stays live as the `batch` *route's* credential. Renaming would touch WP-S's live-validated schema, its greenfield test and four deploy scripts for no functional gain; the name usefully isolates batch-route rate limits from `assessment`'s `anthropic_key`. |
| E-13 | **`utils/logging_config.py` is excluded from D10.** Plan §8.4's "Anthropic SDK instrumentation (−99 L)" under the Files API is a **misattribution**. | Verified: the 99-line removal is crm's `ff12722`, the billing/prepaid commit; `0134d3d` does not touch the file. `instrument_anthropic_client` is still defined at crm HEAD, still called by `clients/llm/dialects/anthropic.py:129-131`, and mira-OSS's WP1-landed `utils/llm_tap.py:156` documents that it is attached via that function. Porting §8.4's hunk would have broken shipped WP1 observability. Belongs to §8.2/D-14 triage instead. |
| E-14 | **Domaindoc sharing gets a minimal SECURITY DEFINER lookup** exposing only `id, email, first_name` for active users, with no member gating (user-approved). | RLS on `users` is unconditional in the landed schema (`:553,559,665`), so three cross-user readers silently return nothing in `multi` mode: `cns/api/actions.py:1849` looks a collaborator up **by email** (how a share target is added — sharing becomes impossible), and `actions.py:1928` plus `utils/domaindoc_shares.py:132` join to another user's row for share listings, so collaborator names and emails vanish. Plan §6.3.8 already concluded crm's fix (`JOIN LATERAL active_member_identity(...)`) is **not portable**, because that function filters `subject_kind='member'` and D12 omits the demo machinery it belongs to. So there is no upstream answer. Rejected alternative: routing these reads through an admin session, which would bypass RLS entirely for a user-facing query. New open item **O-25**; WP3-B owns it. |
| E-15 | **`effort='none'` means thinking is DISABLED** (user ruling) — not provider default, not lowest level, not "omit and let the model decide". | Resolves the design question behind the install blocker. The greenfield schema seeds `fast` (groq `qwen/qwen3.6-27b`) and `assessment` (anthropic `claude-opus-4-6`) at `effort='none'`, but `clients/llm/types.py:14`'s `EffortLevel` Literal has never included `'none'`, so `coerce_effort('none')` raises `ValueError` from `load_model_configs()` at `utils/user_context.py:248` and **every fresh install aborts at startup**. Plan §5 D-12's "effort=none, cheap" and §6.1.3's effort=none gate both assume the semantic. crm's `98a8f28` fix is a one-line Literal widening that **cannot be ported alone**: `EFFORT_LEVEL_ORDER` derives from `get_args`, so `'none'` sorts first, and crm HEAD's `_LEGACY_BUDGET_PER_EFFORT` has no `'none'` entry — making `anthropic.py:343-344`'s clamp loop `KeyError` on its first iteration for any budget. Open sub-question: Qwen3 on Groq disables thinking via `chat_template_kwargs: {"enable_thinking": false}`, which mira-OSS deliberately does not carry (crm added it in `a7b8694`, removed it in `ed441e5`). If the `fast` route genuinely needs it, that is a reinstatement of a previously declined mechanism and comes back for a decision rather than being added unilaterally. |
| E-16 | **Accept the chat output ceiling dropping from 31999 to 16000** (WP2-B decision D-2, ratified). | WP2-B deleted `llm_kwargs['max_tokens'] = 31999` ("Frontend generation ceiling"), a caller-side override on a contract whose whole point is that the row owns the ceiling (§6.1.1). crm's post-image omits it too. Worse, `primary` is seeded `openai/gpt-5.5` at `max_tokens=16000`, and per-request overrides win over the row, so 31999 exceeded the model's own ceiling and risked provider 400s on long turns. **Reversible as a one-line row change** in `deploy/mira_service_schema.sql` if longer replies are wanted — that is the correct place, not a caller override. Flagged by the agent as the one user-visible decision it made without instruction. |
| E-17 | **O-18 is closed, and the briefed fix would have made it worse.** The overwatch ceiling is **80**, not 100. | §10 recorded O-18 as "unverified whether crm's `agents/base.py` actually passes the override". Verified: `_run_overwatch` already passes `max_tokens=self.overwatch_max_tokens` = **80**, and precedence was identical at main, so the observer's effective ceiling was always 80 — the row's 16000 was never reachable. Passing 100 as the brief instructed would have **raised** it 25%. The agent declined the instruction, documented the override as load-bearing so a future cleanup cannot delete it and open the real 160x hole, and closed the item with a value. |
| E-18 | **A vocabulary gate cannot catch signature breakage.** After a required-kwarg migration, the correct sweep is "find every caller of the changed function", not "grep for the retired names". | `tools/implementations/web_tool.py::_synthesize_content` passed four kwargs WP2-A removed (`endpoint_url=`, `dialect_name=`, `model=`, `api_key=`), so every call raised `TypeError` — swallowed by its own `except Exception: return None`, leaving long-page synthesis silently dead while the fetch still succeeded. The headline gate stayed clean because the file contains no `internal_llm` string, and the differential stayed clean because no test covers it. Found by WP2-B reporting residue, not by any gate. Chartered WP2-D to fix it and to sweep the tree for other callers of `generate_response`/`stream_events` still passing a removed kwarg. |

---

## 7. Open items status

Plan §10 carries O-1…O-24. Current disposition:

**Closed:** O-1 (crm's soft-delete columns adopted, `users_trash` dropped), O-2 (see E-2), O-9 (crm's
`base.py` taken), O-12 (void — no migrations), O-22 (harness landed), **O-3** (E-1, warning not hard
failure — landed in WP2-A's `load_model_configs()`), **O-5** (park-vs-exit landed as
`MIRA_POST_GATE_FAILURE_ACTION=park|exit`, strictly parsed, default `park`, with `2f980ab`'s
`PRE_SERVER_GATE_ATTEMPTS=3`/`RETRY=10` and `b88f076`'s `config.api.temperature` probe), **O-11**
(`ProviderSwitchEvent` consumers enumerated: `events.py:137` definition and `orchestrator.py:44,889`
branch left dead for WP2-B; emission removed from `lifecycle.py`; the `provider_switch` frame also
renders at `websocket_chat.py:510-512`, `api-client.js:589`, `messaging.js:1491` — all WP4), **O-18**
(overwatch ceiling — passed to WP2-B), **O-20** (R-3's verdict: `deploy/migrate.sh`,
`deploy/lib/migrate.sh` and `deploy/schema_aware_restore.py` are **dead — delete all three plus the
`--migrate`/`--dry-run` branch and usage text in `deploy.sh:7-13,44-77`**. `migrate.sh:575`
`if nohup mira >/dev/null 2>&1 &; then` does not parse, on `main` as well, so the whole family including
its rollback path has been unreachable since `0254a8c1` in 2025-12. **Do not fix the syntax error** —
deleting is the disposition. WP6 owns it).

**O-10 is a confirmed defect, not merely unverified.** `main.py:290` calls
`flush_except_whitelist(preserve_prefixes=["session:", "rate_limit:"])` — **`csrf:` is missing**, while
`auth/session.py`'s `_csrf_key()` writes `csrf:<digest>` paired with `session:<digest>`. Every restart
flushes CSRF tokens and keeps sessions, so the first unsafe cookie-authenticated request 403s until the
client re-fetches `/csrf`. Fix is one list entry, but `main.py` is **WP3-B's** file and WP3-A is barred
from it — so it is recorded for WP3-B's brief rather than steered into WP3-A. This is the answer to the
question WP3-A explicitly left open.

**O-7 is closed in both halves.** The shape half: WP2-A ported crm's synthetic `{"load": [tool]}` form
at `clients/llm/lifecycle.py:108`, and I verified it against OSS's tool — `invokeother_tool.py:94-103`
declares exactly `load` and `load_for_rest_of_session` and `:110` takes those two parameters, so the
shapes match exactly and the old `{"mode":"load","query":…}` form would not have. The detection half:
`orchestrator.py:772-777` gated `acc.invoked_tool_loader` on `event.arguments.get("mode", "")` against
`["load","fallback","prepare_code_execution"]` — but **no `mode` property exists in the tool's schema**,
so the flag was never set and loader auto-continuation never fired. Steered to WP2-B (it owns the file,
and §10 assigns O-7 to WP2) as a three-line fix detecting on the parameters the tool actually declares.
`"fallback"` and `"prepare_code_execution"` are legacy vocabulary nothing emits.

**O-23's remaining half is WP4's and unaffected:** the circuit-breaker finalization from `e26d031` and
`e370468`'s `persisted_tool_ids` hunk. Of `test_orchestrator_tool_loop.py`'s two known failures, O-7's
fix should clear `test_successful_tool_loader_triggers_auto_continuation` while
`test_circuit_breaker_remains_latched_after_final_no_tools_pass` stays red until WP4.

**Plan cross-reference defect: §7.4 does not exist.** Decision D6 cites "§7.4" for the rewriter route
mapping, but §7 has only 7.1–7.3. The information is present elsewhere — §6.1.3's route table maps
`rewriter` to `primary` with an `effort='high'` override, §7.1 covers the user-model pipeline including
the retained `repulsion_rewriter_*` prompts, and §8.3 lists them as deletions to decline. Worth fixing
the reference in the plan rather than leaving a dangling citation.

**§9's `web_tool.py` synthesis simplification is now portable and needs an owner.** §9 records that crm's
change (removing `synthesis_model`/`synthesis_endpoint`/`synthesis_api_key_name` config fields and the
Vault lookup in `_synthesize_content`, switching synthesis to `LLMProvider().generate_response(
model_config="fast", …)`) "becomes portable only after WP2 lands `0134d3d`'s `model_config=` API".
**Both preconditions are now met**: `model_config=` landed with WP2-A and `0134d3d` landed with WP2-C.
It is a surgical edit to `_synthesize_content` and its config block, leaving
`_request_with_validated_redirects` untouched — which matters because that function carries the
`e401d59` SSRF pinning (§0 invariant 2). WP2-B was told it is optional and to report either way; if it
declines, assign it to WP6.

**Plan claims found false during wave 1 — correct these in the bisect before WP4/WP5/WP6 are briefed:**

| Plan location | Claim | Verified reality |
|---|---|---|
| §4 D-1, §5 D-7 | CSP tightening "breaks `oss_ui.py:19-20` inlined `marked.min.js`/`purify.min.js`"; follow-up is to "externalise the inlined vendor scripts" | **Wrong.** `oss_ui.py:19-20` *reads* both files at import and `:23-25` *serves* them at `/oss-assets/*.js`; `web/chat/index.html:219-220` loads them by `src`. What `script-src 'self'` actually rejects is **8 inline `<script>` blocks** (2 each in `web/{chat,settings,domaindocs,memories}/index.html`) and **32 `onclick=` attributes** (2 in chat, 30 in domaindocs). D-7 is a materially larger job than the plan states and does not fit WP4's ~150-line budget. |
| §6.3.6 | "`deploy/postgresql.sh:166-176` seeds only `valkey_url`, `userdata_encryption_key`, `diagnostics_token`"; "Action: seed `app_url`" | **Stale.** Both installers already seed four fields including `app_url="http://localhost:1993"` (`postgresql.sh:197`, `init-mira.sh:242`). The action is already satisfied at main; delete it from WP3-B's brief. |
| §8.4 | `logging_config.py` −99 L belongs to the Files API | **Misattributed** — see E-13. |
| §8.4 | "batch mode in … `forage_agent`" | `forage_agent.py` has **no** `use_batch` in mira-OSS; crm-side artifact, nothing to remove. |
| §8.4 | Omits two mandatory edit sites | `clients/llm_provider.py::build_batch_params` (:54-90) and **`power_on_self_test.py:902,905,972`**, whose required-job lists raise `RuntimeError` at `:911` — a boot failure if D10 deletes the jobs without editing them. |
| §8.4 | Names no upstream SHA for the Batch deletion | `5c50dfc` (Batch, 16 files +153/−1413), `0134d3d` (Files, 7 files +82/−695). `58c261b` was already named. |
| §6.3.4 / O-4 | `main.py:99` reads `MIRA_DEV` | It is **`main.py:647`**, in the `__main__` hypercorn block. |
| §6.1.4 / WP2-A brief | `ROUTE_FALLBACKS` must be re-derived from `utils/user_context.py` | It lives in crm's `clients/llm/resolver.py:17` and **never existed in mira-OSS at all** — OSS's `_check_llm_provider_reachability` already raised unconditionally. The instruction was a no-op; the real work was deleting the `source`/`hidden` classification. |

**New facts established, not previously recorded anywhere:**

- `lt_memory/db_access.py:1409-1647` (~238 L) queries `extraction_batches`, a table WP-S's greenfield
  schema does not create. That code is **already runtime-dead against a fresh install**, so D10 is a
  correctness fix as well as a scope reduction.
- `config.batching.batch_max_age_hours`, consumed at `utils/lt_memory_jobs.py:143`, **does not exist**
  (`LTMemoryFactory` has no `.config`; there is no `BatchingConfig`). The batch cleanup job is already
  AttributeError-dead.
- The mira-OSS tree is **byte-identical to crm's pre-deletion state** for 13 of 16 Batch files and 4 of 7
  Files files, which makes D10 close to a mechanical patch replay. Drift only in `lt_memory/models.py`
  (17 lines), `agents/base.py` (84), `segment_collapse_handler.py` (53+), `userdata_manager.py` (4).
- §8.1's "`sidebar_jobs.py` — LEAVE THIS FILE UNTOUCHED" and §8.4's `SidebarDispatcher(
  max_concurrent_batch_agents=…)` deletion are a **paper conflict**: §8.1's own parenthetical assigns
  that line to D10, and crm deleted exactly one line there in `58c261b`. WP2-C owns `sidebar_jobs.py:29`
  and nothing else in the file; left behind it is a boot-time `AttributeError` whenever
  `sidebar_dispatcher.enabled` is true.
- `tests/test_direct_extraction.py` is byte-identical to crm's copy at `5c50dfc` and asserts precisely
  the post-deletion `DirectExecutionStrategy` shape, which confirms that removing the batch transport is
  what makes a direct path necessary — `ImmediateExecutionStrategy` is not sufficient because it is
  dialect-conditional and passes `internal_llm='extraction', allow_negative=True`.
- `deploy/oss_ui/chat.html` has **no Python consumer** (`git grep chat.html -- '*.py'` → empty). WP6
  scrub/docs should rule on it.
- `init-mira.sh:249-252` writes `/opt/vault/provider_endpoint.txt` and `provider_model.txt` that
  **nothing in the repo reads** — a pre-existing dead mechanism. Consequence: non-Groq container installs
  keep the groq endpoint on `fast` and no container flow rewrites the `primary` row. WP6 should implement
  or delete it.
- `auth/security_middleware.py` now carries a `MIRA_CSP=off|strict` knob (default `off`, strictly
  parsed at middleware-stack build). Its strict value adds `worker-src 'self'`, which **closes part of
  D-7**, and omits `frame-src` rather than setting `'none'` — crm set it to third-party payment origins
  only, which under `default-src 'self'` actively blocked same-origin frames.

**New open item O-26 — naive local wall times are attributed to UTC. Larger than the DST bug it was
found beside, and unassigned.** The WP1-D agent surveyed every tool for model-supplied wall times and
found a different defect class in the *stored-timestamp* paths:

- `memory_tool.create_memory(happens_at=, expires_at=)` takes model-supplied strings, queues them
  verbatim to Valkey, and they are parsed later at `cns/services/segment_collapse_handler.py:605,610` by
  `parse_time_string(mem.happens_at)` **with no `tz_name` argument**. `get_default_timezone()`
  (`utils/timezone_utils.py:99`) returns `"UTC"`. So a naive local wall time is attributed to UTC —
  **wrong by the user's full offset, every day of the year**, which in some zones exceeds the one-hour
  DST error. The surrounding `except Exception: logger.warning(...)` then **silently drops the field**.
- `continuum_tool.search_messages(start_time, end_time)` and its `reference_time` are parsed by
  `parse_utc_time_string`, i.e. naive input read as UTC. Same class, milder in practice: the schema
  instructs the model to copy segment-summary `time_boundaries`, which are `Z`-bearing UTC strings, so
  the normal flow carries no local wall time.

**DST strictness is the wrong remedy here**, and this is the interesting part: by the time these strings
are parsed the model is no longer on the call stack, so raising loses the memory instead of prompting a
clarifying question. The correct fix is to parse in the **user's** timezone (available via
`get_user_preferences().timezone`) rather than the system default, and to stop discarding the field on
failure. That is a different change from WP1-D's and belongs in its own small package. It touches
`segment_collapse_handler.py`, which **WP5 is editing right now** — so sequence it after WP5 merges.

**Plan defect: §6.1.5 and the WP2-B brief both name the wrong file for the model picker.** They say the
picker UI is in `web/settings/index.html`. Verified: that file contains **no** picker (zero `tier` or
`conversation_llm` hits). The real consumers were `web/chat/index.html`,
`web/assets/javascript/thinking-budget.js` (181 L, deleted) and `web/assets/style.css:1974-2005`
(`.thinking-popover`, `.thinking-options`). A related trap: `data-indicator="tier_btn"` is **also** the
live thinking/emotion indicator — `messaging.js:896` looks it up by `id="thinking-indicator"` — so
deleting the button along with the popover would have broken the thinking stream. Only `#tier-popover`
is the picker. Residue left for WP4: the dead picker CSS, and stale tier-label comments at
`messaging.js:899,916`.

**Resolved by WP-S but not yet marked in the plan:** O-20 is *partly* addressed — WP-S deleted
`deploy/migrations/` and reported on the `deploy/*migrate*` scripts, but the decision whether to delete
`deploy/migrate.sh`, `deploy/lib/migrate.sh` and `deploy/schema_aware_restore.py` outright is still open
and belongs to WP6.

**Open and blocking a package:** O-3 (resolved as E-1, needs applying in WP2-A), O-7 (invokeother
argument shape — blocks half of O-23), O-18 (overwatch `max_tokens` override — WP2-B), O-10 (Valkey
flush whitelist covering `session:`/`csrf:` — **must be checked before WP3-B**, cheap and high impact:
if absent, every restart logs everyone out), O-17 (`lt_memory/db_access.py:507` second
`global_memories` reader), O-13 (`reminder_tool` datetime parameter — determines whether the dormant
DST utility becomes a live fix), O-14 (`_prepopulate_welcome_content` vs main's seeding), O-15
(`get_assessable_sections()` after the prompt edit), O-16 (the file-links prompt line needs rewording
after D10), O-21 (VERSION), O-24 (the serf brief).

**Open and unassigned:** O-19 (the 1.x suite re-baseline — ~98 pre-existing failures), O-5
(park-vs-exit on gate failure), O-6 (`history.js` rewrite timing — folded into WP4), O-8 (rename the
misnamed "LoRA" subsystem), O-11 (`ProviderSwitchEvent` consumers — WP2-A must grep before deleting),
O-23 (circuit-breaker finalization — assigned to WP4).

**Deferred register D-1…D-17** in plan §5 is unchanged and still accurate; D-4 (`viewcard_content.py`
as the future frontend's sanitization boundary) and D-9 (anthropic dialect lacks `invalid_reason`
coverage) are the two most likely to be forgotten.

---

## 8. Risks and known-weak points

**R-1 — RETIRED 2026-09-06.** The greenfield schema has now been applied to a live PostgreSQL 17.11
server and the security model behaviourally verified. 17 checks, all passing:

- Apply is clean: **0 errors** both with and without `ON_ERROR_STOP=1`, on a fresh database created
  exactly as `deploy/postgresql.sh` does (roles `mira_admin LOGIN BYPASSRLS` and `mira_dbuser LOGIN`
  via its `DO $roles$` block, then `CREATE DATABASE mira_service OWNER mira_admin`). Result: 21 tables,
  19 policies, 17 RLS-enabled tables, 1 view, 10 triggers, 65 indexes, 5 `model_configs` rows,
  6 `usage_pricing` rows.
- The handoff's worry about unbalanced `$$` quoting was **aimed at the wrong delimiter**: the file uses
  8 balanced `$function$` tags plus a `$schema_precondition$` guard, and no bare `$$`.
- **RLS fail-closed confirmed.** As `mira_dbuser` with the GUC never set → 0 rows; with
  `app.current_user_id = ''` (the exact value `clients/postgres_client.py:201` writes for the no-user
  state) → 0 rows.
- **Cross-user isolation confirmed.** Alice sees only her 2 memories, Bob only his 1. As Bob: UPDATE of
  Alice's row → `UPDATE 0`; DELETE → `DELETE 0`; INSERT with `user_id` = Alice → `ERROR: new row
  violates row-level security policy` (WITH CHECK enforced).
- **The `global_memories` view gate is not bypassable.** `mira_dbuser` holds no direct grant on
  `global_memories` — direct SELECT → `ERROR: permission denied for table global_memories`, while
  `global_memories_runtime` (`security_barrier=true`) returns rows. Deactivating the user closes the
  gate: `can_read_global_memories()`'s `is_active = TRUE` check drops the view from 1 row to 0.
- The 4 tables without RLS — `global_memories`, `global_usernames`, `model_configs`, `usage_pricing` —
  are global by design and correct: config is `GRANT SELECT` only, and `global_memories` is reachable
  solely through its gated view.
- **All triggers fire.** `provision_baseline_persona()` created 2 `persona_state` + 2 `persona_revisions`
  rows for 2 inserted users. `set_updated_at()` leaves `updated_at` NULL on insert and advances it past
  `created_at` on UPDATE. `set_search_vector()` populated `search_vector` on 3/3 memories.
- The emptiness guard works: re-applying to a populated database raises
  `mira_service_schema.sql requires an empty target database`. The schema is deliberately **not**
  idempotent — this is a fresh-install contract, consistent with plan §0.1.
- `model_configs` CHECK constraints confirmed live: route names carry **`other`**, not `difficult`
  (D14 enforced at the database level); `dialect_name` limited to anthropic/openai/openrouter/groq;
  `effort` to none/low/medium/high/xhigh/max; `max_tokens > 0`.

Residual: this validated SQL correctness and RLS semantics, **not** application behaviour against the
database — no Postgres, Vault or Valkey is reachable to the Python code, and none of the connection
paths in `clients/` have been exercised. That gap is closed only at first real install.

**R-2 — WP3's SMTP sender is genuinely new design work.** crm_mira posts to a private HMAC-signed HTTP
gateway whose server side is an untracked PHP file (`mira_email_gateway_forbiz.php`). No upstream
reference exists. Until it lands, `MIRA_AUTH_MODE=multi` cannot sign anyone up, so WP3's headline
acceptance gate is unachievable without that design.

**R-3 — Four `deploy/` scripts were modified by WP-S outside my brief's explicit list** (E-5). They are
plausibly correct consequences but have not been reviewed line-by-line. Read the diff of
`3711360` and `c4f58b2` for `deploy/config.sh`, `deploy/deploy_database.sh`,
`deploy/docker/scripts/init-mira.sh` and `deploy/finalize.sh` before trusting them, and re-run the §11
scrub grep over `deploy/` specifically.

**R-4 — The retained frontend is still unpatched.** WP1–WP-S did not touch `web/`. The old UI currently
works against the old backend. WP4 changes the protocol and *will* break it until the ~150-line patch
lands. Plan §6.4.3 has the verified frame-by-frame break list.

**R-5 — `cns/api/actions.py` is the most entangled file in the backport** and has not been decomposed.
A 797-line upstream delta bundling five separable concerns. WP2-B touches it (picker removal, effort
override) and WP5 touches it (PersonaDomainHandler). Coordinate, or expect a conflict.

**R-6 — Three of my own subagent briefs contained errors**, all caught by the verification loop rather
than by the agents' output: WP0's recovery list named 6 generic tests where 13 exist; WP1-A's brief said
"eight items" while listing seven and mis-described item 10's defect mechanism; and I over-deferred
`e26d031`'s circuit-breaker finalization by wrongly claiming a WP4 dependency. The pattern is that
**briefs derived from the earlier analysis inherit its errors.** Read the prompts file critically before
dispatching, and pre-verify apply-cleanliness with `git show <sha> -- <path> | git apply --check` in a
throwaway worktree, which is what caught the ordering and contamination problems in WP1-B and WP1-C.

**R-7 — Provider quota.** `qwen-token-plan` exhausted its weekly quota mid-WP1-C (reset was 09-12
20:02 UTC). The user switched the parent session to OpenRouter `qwen/qwen3.8-max` and directed
subagents to `qwen/qwen3.8-flash`. If quota fails again mid-package, the recovery procedure is proven:
inspect the worktree with `git status --porcelain` and `git diff`, determine which patch steps landed,
verify exclusions, and finish manually. WP1-C's agent had applied all seven steps correctly and only the
commits were missing.

**R-8 — critical dependencies ship unpinned while the code requires recent SDK features.**
`requirements.txt:27` lists `anthropic` with no version constraint, and
`clients/llm/dialects/anthropic.py:341` sends `output_config` — a parameter the installed 0.52.2 does
not have (`inspect.signature(...create)` confirms no `output_config`, though `thinking` and
`ThinkingConfigDisabledParam` are both present). Verified **pre-existing, not introduced by 2.0**: the
`output_config` reference is present at `main` and at crm HEAD, and crm's `requirements.txt:30` is
likewise unpinned.

So this is *not* an install blocker — a fresh `pip install -r requirements.txt` resolves to 1.4.0
(latest), which is new enough. It is a **reproducibility and release-quality issue** for a distributed
package whose stated posture is fresh-install-only (§0.1): the code depends on a feature whose minimum
SDK version is nowhere recorded, so an install pinned by a distro, a lockfile or an old cache breaks at
runtime with `TypeError` on the `batch` route rather than at install time. Note the `assessment` route
is unaffected — `effort='none'` emits `thinking={"type":"disabled"}` and never reaches `output_config`.

**WP6 action:** establish the minimum `anthropic` version that provides `output_config` and pin a floor
(`anthropic>=<version>`), and audit the other unpinned entries the same way — `openai`, `httpx[http2]`,
`psycopg`, `valkey` — for any feature the code uses that a floor would protect. Do not pin exact
versions; a floor is enough and avoids fighting the rest of the dependency graph.

---

## 9. Environment and tooling facts

- Subagent model: **`qwen/qwen3.8-flash`** (OpenRouter). 1 M context, 131 k output, reasoning-capable.
  Parent session: `qwen/qwen3.8-max`. Do **not** use `qwen-token-plan/*` — quota exhausted.
- `pytest 8.3.5`, Python 3.12 (conda env `mira`).
- No Vault, Postgres or Valkey reachable. `VAULT_ADDR`, `VAULT_ROLE_ID`, `VAULT_SECRET_ID` unset.
- Subagent results are retrieved with `get_subagent_result`, but entries are **cleaned up quickly** —
  twice an agent id returned "Agent not found". The durable fallback is the task output file:
  `/var/folders/wd/gk07dvzx6yg5fkz7wb59b14w0000gn/T/pi-subagents-501/Users-taylut-Programming-GitHub/01a07325-efc2-7c32-b9b4-4d810524a837/tasks/<agent-id>.output`
  (JSONL; the final assistant text block is the report). Extract with a small Python walk over the JSON.
- Concurrency limit observed: 4 background agents, the 5th queues.
- Commits require the `git-workflow` skill at `/Users/taylut/.pi/agent/skills/git-workflow/SKILL.md`.
  The user has explicitly authorised commits. **Pushing is not authorised.**
- `scratch/` (gitignored, in the main checkout) holds four files the user judged not relevant:
  two divergent `MIRA_ARCHITECTURE_OVERVIEW*.md`, a Roundcube agent-guide note, and a re-created
  `tests/conftest.py`. Both overview docs are stale against 2.0 and should be re-derived rather than
  updated. The flushed Slack WIP snapshot is at
  `/tmp/slack-wip-flushed-20260905/` and **will not survive a reboot**.

---

## 10. Resume procedure

```bash
cd /Users/taylut/Programming/GitHub/mira-OSS

# 1. Orient
git worktree list
git -C .worktrees/wp1 log --oneline main..2.0/integration | head -40   # 31 commits
git -C .worktrees/wp1 diff --shortstat main..2.0/integration           # 137 files

# 2. Confirm the tree is healthy
cd .worktrees/wp1
python3 -m compileall -q . && echo "compiles"
python3 -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -3
#    expect approximately: 195 passed, 121 failed, 396 skipped, 18 errors

# 3. Read the specification and the prompts
#    .pi/plans/20260905-172000-crm-mira-backport-bisect.md   (§0.1, §0.2 first)
#    .pi/plans/20260905-REMAINING-WP-PROMPTS.md

# 4. Create the next worktree and dispatch from the prompts file
cd /Users/taylut/Programming/GitHub/mira-OSS
git worktree add .worktrees/wp2a -b 2.0/wp2a 2.0/integration
```

Then dispatch the WP2-A prompt verbatim from the prompts file, with `model: qwen/qwen3.8-flash`,
`subagent_type: general-purpose`, `run_in_background: true`.

**Supervision discipline that worked, and should continue:**

1. Before dispatching, pre-verify every patch application with
   `git show <sha> -- <path> | git apply --check` in a throwaway worktree
   (`git worktree add /tmp/probe -b probe <base>`, then `git worktree remove /tmp/probe --force`).
   This caught a three-way ordering dependency in WP1-B and every contaminated hunk in WP1-C. Put the
   result in the prompt as "applies cleanly" versus "hand-edit, here is the exact change".
2. State exclusions as explicitly as inclusions, with the reason. Most upstream commits bundle wanted
   and unwanted changes; agents that are told only what to take will take the rest.
3. Verify independently after each package: file list against expectation, exclusion greps,
   `py_compile`, import smoke tests, and the differential test delta. Do not trust the report.
4. Give the agent a precise stop condition. "Report the mismatch rather than improvising a semantically
   different change" produced three honest partial-result reports and zero silent errors.
5. Keep parallel agents on **disjoint file sets**. Every collision I found was resolved by re-splitting
   the work, not by merge skill. `config/config_manager.py` and `cns/services/orchestrator.py` are the
   two recurring collision points.

---

## 11. Commit inventory on `2.0/integration`

25 non-merge commits, newest first:

```
1326633 security(rls): warn when a query runs without user context
7ed24e1 fix(tests): make the greenfield schema gate assert mira-OSS's contract
3711360 fix(deploy): seed offline and custom providers through model_configs
57c4ebc fix(startup): bootstrap the single user without retired tier columns
9ca5295 refactor(tools): remove redundant schema_distribution module
6e3354e chore(deploy)!: delete deploy/migrations
c4f58b2 feat(database-schema)!: author the 2.0 greenfield install contract
013476c test(harness): run unit tests without live infra; skip integration
0c28aaf docs(clients): record tool-argument validation and reasoning coalescing
b80d752 fix(llm): reject invalid provider tool calls with repair feedback
882c94c feat(tools): add feedback_tool for friction and feature-request capture
2fc1fd9 feat(maps): replace Google Maps with OpenStreetMap, no API key
cfd923c fix(config): resolve per-user tool config over global default
d13ba97 feat(config): add strict cognitive feature-flag env overrides
1d6dae0 fix(pollers): unregister segment poller state when the poll loop exits
17d6f7b test(tests): make the recovered suite collect and record its triage
696239f fix(history): surface tool_call_id and is_error on history rows
a54a7ca fix(userdata): anchor the per-user data directory to the project root
347ace9 fix(vault): re-authenticate AppRole when the token expires
9f86a15 feat(timezone): add DST-aware local wall time to UTC conversion
6e588f3 fix(rls): pass user_id to PostgresClient in preference and portrait paths
48ca716 fix(memory): coerce pgvector Vector to list on Memory.embedding
42a0d5b chore(tests): recover seven further generic crm_mira contract tests
ddd1dff chore(tests): recover the pre-ee44b18 mira-OSS test suite
1b8747b chore(tests): recover eight generic crm_mira contract tests
```

Plus 6 merge commits (`ec5e50b`, `77e4e4b`, `12c2aa5`, `b85c839`, `8f9a627`, `0384829`).

On `main` (not part of the 2.0 line): `2bb2807` plan revisions, `840ba72` the bisect document,
`e19667b` the `scratch/` gitignore rule, above `e401d59` (the SSRF fix that is main's only unique
commit relative to the merge-base).

**The three plan documents are committed on `main` but the prompts file and this handoff are not yet
committed** — commit them before relying on them surviving a `git clean`.
