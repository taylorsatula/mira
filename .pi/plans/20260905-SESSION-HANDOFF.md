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

Dependency chain: **WP2-A → WP2-B → WP3-A → WP3-B → WP4 → WP5 → WP6.** WP7 is descoped.

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

---

## 7. Open items status

Plan §10 carries O-1…O-24. Current disposition:

**Closed:** O-1 (crm's soft-delete columns adopted, `users_trash` dropped), O-2 (see E-2), O-9 (crm's
`base.py` taken), O-12 (void — no migrations), O-22 (harness landed).

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

**R-1 — Nothing has been executed against a live database.** No Postgres, Vault or Valkey was available
in this environment. The 21-table greenfield schema has never been applied to a real server. Static
review and the (regex-based) schema test are the only evidence it is valid SQL. **The single highest-value
next verification is `psql` against an empty database.** Balanced `$$` quoting, FK ordering, policy
targets and role grants are all plausible failure points that static reading does not reliably catch.

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
