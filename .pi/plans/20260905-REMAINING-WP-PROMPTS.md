# Pre-written subagent prompts — remaining work packages

Companion to `20260905-172000-crm-mira-backport-bisect.md` (the specification) and
`20260905-SESSION-HANDOFF.md` (execution state). Each block below is a complete, paste-ready prompt
for the `Agent` tool.

Model for all of them: `qwen/qwen3.8-flash` (OpenRouter). The earlier `qwen-token-plan` provider
exhausted its weekly quota mid-WP1-C.

---

## Conventions every prompt assumes

These are inlined in each prompt below rather than referenced, because a subagent cannot see this file
unless told to read it.

**Authority block** — required verbatim in every prompt:

```
## AUTHORITY

- You may CREATE, MODIFY and DELETE files freely inside your worktree. You do not need to ask
  permission to write.
- /Users/taylut/Programming/GitHub/crm_mira is REFERENCE ONLY — read it and run git show / git log
  against its refs, but change nothing there.
- Work only inside your worktree. Do not touch the main mira-OSS checkout or sibling worktrees.
- Commit locally. Never push. Never open a pull request.
```

**Report format** — required in every prompt:

```
Structured handoff: Status (Done | Partial | Blocked; Confidence) · Result (commit hashes and
subjects) · Evidence (each verification step with its actual output) · Work Performed ·
Changes / Artifacts (every path touched, including deletions) · Validation (verified vs assumed) ·
Limits / Unknowns · Recommended Next Action.

If you hit something the specification does not cover, make the choice that plan §0.2 implies
(optimise for mira-OSS as the distributed artifact), record it in your report as a decision you made,
and keep moving. Do not silently skip a deliverable. If a hunk does not apply and the reason is not one
I pre-identified, stop that item and report the mismatch precisely rather than improvising a
semantically different change. Partial completion with accurate reporting is a good outcome; a silently
wrong edit is not.
```

**Commit convention** — every prompt says: read and obey
`/Users/taylut/.pi/agent/skills/git-workflow/SKILL.md`; `type(scope): brief`, ≤72 chars, no trailing
period; body with ROOT CAUSE / SOLUTION RATIONALE / CHANGES / PRESERVES; cite upstream crm_mira SHAs;
no AI attribution, no co-author lines, no emojis; stage explicit paths, never `git add -A`.

**Differential test methodology** (proven during WP1 — use it to validate any package):

The recovered suite contains tests for code that does not exist yet, so a raw pass/fail count is
meaningless. Compare two branches that differ only in the package under test:

```
# baseline: tests present, package code ABSENT
cd .worktrees/<branch-with-tests-only> && python3 -m pytest tests/ -q -p no:cacheprovider | tail -2
# treatment: tests present, package code PRESENT
cd .worktrees/<package-branch>       && python3 -m pytest tests/ -q -p no:cacheprovider | tail -2
```

WP1's result: 150 passed / 143 failed without the code, 164 passed / 129 failed with it. **+14 passed,
−14 failed, skipped and error counts identical** — proof of zero regressions and of which tests the
package actually satisfied. Any package that does not improve the delta, or that increases failures,
has regressed something.

Reference numbers as of the WP1 line (re-measure, do not trust these absolutely):
`164 passed, 129 failed, 396 skipped, 18 errors in ~8s`.

The 18 collection errors and ~106 of the failures are **pre-existing**: the mira-OSS 1.x suite was
already failing when `ee44b18` deleted it. Roughly 23 failures are characterization tests for unlanded
packages (`test_greenfield_schema.py` 8 → WP-S, `test_web_frontend_protocol.py` 7 → WP4,
`test_history_cursor.py` 6 → WP4, `test_orchestrator_tool_loop.py` 2 → O-23/O-7). Plan open item O-19
makes the 1.x suite a re-baseline, not a repair.

**Test invocation.** The harness landed by O-22 makes the suite runnable with no flag and no
infrastructure; integration-style tests skip. If a run shows setup ERRORs mentioning `VAULT_ADDR`, the
harness is not present on that branch — either merge `2.0/wp0b-testsplit` or fall back to
`--noconftest` for pure-unit files.

**Test scope.** This codebase chooses fail-fast over extensive performative testing, and the migration
inherits that posture. **Do not author new tests, fixtures, conftest helpers or test infrastructure.**
Your job is to migrate source, not to expand coverage.

In scope:
- Running the existing suite and reporting per-test results against your acceptance gate.
- Deleting, or minimally updating, a test that the migration itself invalidates — one asserting a
  protocol, table, column or symbol a locked decision deliberately removed. **Prefer deletion over
  authoring a replacement.**
- Reporting a failing test with its cause classified as your defect, another package's scope, or an
  environment limit.

Out of scope — do not do these even when they seem helpful:
- Writing a test to prove your change works. Use your brief's verification steps instead
  (`py_compile`, targeted greps, import smoke tests, existing characterization tests).
- Adding coverage for behaviour you introduced that no existing test exercises.
- Repairing pre-existing failures unrelated to your package. Plan O-19 makes the 1.x suite a
  re-baseline, not a repair, and that re-baseline is not part of this migration.
- Adding defensive fallbacks, compatibility shims, dual-mode branches or soft-failure paths to make a
  test pass or to hedge a migration. Where the specification says fail-loud, raise.

If a characterization test for your own package fails, fix the source, not the test — unless the test
asserts something D1–D14 deliberately removed, in which case delete it and say so in your report.

---

## WP2-A — model_configs: the LLM-layer chokepoint

Create the worktree first:

```
cd /Users/taylut/Programming/GitHub/mira-OSS
git worktree add .worktrees/wp2a -b 2.0/wp2a 2.0/integration
```

`2.0/integration` carries WP-S, which authored the `model_configs` table and its five-row seed that
this package validates against. Do not branch from `main`.

```
You are executing work package WP2-A of the mira-OSS 2.0 backport: replace the internal_llm /
conversation_llm model-routing tables with the fixed five-route model_configs contract, at the
chokepoint layer only. Leaf call sites are WP2-B and are NOT yours.

[AUTHORITY BLOCK]

## Working location

    /Users/taylut/Programming/GitHub/mira-OSS/.worktrees/wp2a

Branch 2.0/wp2a. The crm_mira remote is fetched; its refs resolve here.

## Read first

Plan file .pi/plans/20260905-172000-crm-mira-backport-bisect.md:
  §6.1.1 the contract and its three enforcement sites
  §6.1.2 the code-migration surface and the signature delta
  §6.1.3 the route mapping table (authoritative — every consumer's target route)
  §6.1.4 ROUTE_FALLBACKS re-derivation
  §6.1.6 D5 cost_accumulator re-key
  §6.1.7 seed scrub (WP-S already did the schema side; you do the Python side)
  §0.1 (2.0 posture) and §0.2 (design precedence: crm names are proposals, not authority)
Also read deploy/mira_service_schema.sql on your branch — WP-S has already authored the model_configs
table and its five-row seed. Your Python must agree with it exactly.

## Scope: the chokepoint

Every runtime model lookup in mira-OSS flows through two functions in utils/user_context.py —
get_internal_llm() and resolve_conversation_llm() — reached only via
ModelResolver._resolve_internal_llm / _resolve_conversation_llm, reached only via
LLMProvider.generate_response / stream_events routing kwargs. You own that chain:

  utils/user_context.py       replace ConversationLLMConfig / InternalLLMConfig / get_internal_llm /
                              resolve_conversation_llm / get_conversation_llms /
                              load_internal_llm_configs with crm's ModelConfig dataclass,
                              _MODEL_CONFIG_NAMES, load_model_configs(), get_model_configs(),
                              get_model_config(). Reference: git show crm_mira/crm_mira:utils/user_context.py
                              (the ModelConfig region, roughly lines 185-270).
  clients/llm/resolver.py     ModelSelection with model_config_name; ModelResolver.resolve(model_config=…)
  clients/llm_provider.py     generate_response/stream_events take required keyword-only model_config: str;
                              drop internal_llm=, conversation_llm=, dialect_name=, model=, endpoint_url=,
                              api_key=, allow_negative=, allow_provider_stall_fallback=
  clients/llm/types.py        RequestMetadata / ProviderMetadata: internal_llm_name +
                              conversation_llm_name -> model_config_name
  clients/llm/lifecycle.py    propagate the rename; fail-loud (no provider fallback)
  clients/llm/events.py       DO NOT DELETE the ProviderSwitchEvent class in this package — see the
                              ordering constraint in "Four things" item 4 below. Remove only its
                              emission from lifecycle.py.
  config/config.py            remove ApiConfig.max_tokens and ApiConfig.analysis_enabled's replacement
                              validate_compaction_trigger_tokens; add
                              validate_compaction_budget(primary_max_tokens: int). Take this from
                              6899d07's config/config.py hunk — WP1-B deliberately declined it because
                              it was WP2's. Keep SystemConfig.subcortical_enabled and peanutgallery_enabled
                              (WP1-B added the former).
  utils/power_on_self_test.py _check_llm_configuration (:701) asserting the five routes; FIX the two
                              config.api.max_tokens readers at :719 and :1074, which break when you
                              remove that field; and rework _check_llm_provider_reachability (:1014)
                              for the criticality change in "Four things" item 3. Do NOT touch the RLS
                              expected_tables list — WP-S owns it.
  utils/cost_accumulator.py   re-key from internal_llm row names to model_config_name (D5). Keep the
                              FALLBACK_PRICES mechanism and the usage_pricing lookup; only the keying
                              changes. Its docstring explicitly names OSS as the reason FALLBACK_PRICES
                              exists — preserve that intent.
  cns/api/chat.py             re-attach cost recording (:272-292 start/drain) if WP-S or D10 disturbed it

## Four things that are yours and easy to miss

1. **The route set is primary / fast / batch / assessment / `other`.** Not crm's `difficult`. Decision
   D14 renamed it because its purpose is "route to an outside model", and it is the catch-all for future
   consumers needing a sidebar model from a different vendor. `_MODEL_CONFIG_NAMES`, the RuntimeError
   message in load_model_configs(), and `_check_llm_configuration` in power_on_self_test must all say
   `other`. crm's ModelConfig docstring says "One of MIRA's three fixed model routes" while its own
   frozenset lists five — write an accurate docstring instead of inheriting that defect.

2. **D14's seeding constraint needs an enforcement point.** `other` must resolve to a different
   model/vendor than `primary`, or phoneafriend_tool degenerates into consulting the same model. Add a
   check in load_model_configs() that logs a **WARNING** (not a hard failure) when
   loaded["other"].model == loaded["primary"].model. A warning rather than an exception because an
   operator with a single provider available must still be able to boot. This resolves plan open
   item O-3 — record that in the commit body.

3. **ROUTE_FALLBACKS is in `clients/llm/resolver.py:17`, NOT in `utils/user_context.py`, and it must be
   replaced rather than re-derived.** Verified facts: crm's value is
   `{"assessment": "primary", "difficult": "fast"}`, with a comment explaining it encodes a
   local-llama-server vs cloud split. WP-S seeded all five OSS routes at cloud providers
   (openrouter / groq / anthropic), so that split does not exist here and **no route has a sensible
   fallback target** — falling `other` back to `primary` would silently convert "consult an outside
   model" into "consult yourself" (D14), and falling `assessment` back to `primary` when `primary` is
   the thing that is down is meaningless.

   The dict is **overloaded**: crm's `_check_llm_provider_reachability()` uses `c in ROUTE_FALLBACKS`
   purely as a **criticality classifier** — a route with a fallback is treated as non-critical local
   (warn and continue), a route without one as critical cloud (raise, parking startup). mira-OSS
   already has that function at `utils/power_on_self_test.py:1014`.

   **DECISION (user-confirmed): all five routes are critical.** Delete `ROUTE_FALLBACKS` outright and
   **do not** introduce a replacement criticality set — with every route critical there is nothing to
   classify. Collapse the local/cloud classification block in `_check_llm_provider_reachability` to:
   any probe failure raises, parking startup. Remove the now-dead `local_routes` /
   `down_local_routes` bookkeeping and the `from clients.llm.resolver import ROUTE_FALLBACKS` import.

   Because this parks boot when any one of three vendors is unreachable, or when an operator does not
   hold credentials for all of them, **the failure message must be actionable**. Raise with a message
   that names each failed route, its vendor and model from `model_configs`, and states plainly that all
   five routes are required at startup. An operator staring at a parked process should learn from the
   message alone which credential or endpoint to fix. Preserve the existing per-target diagnostic
   structure (`type(error).__name__: error`) inside that message.

   Rationale to state in the commit body: crm's `ROUTE_FALLBACKS` encoded a local-llama-server vs cloud
   split that WP-S's all-cloud OSS seeding makes vacuous, and its fallback semantics would silently
   convert "consult an outside model" into "consult yourself" for `other` (D14). Criticality is now
   uniform and explicit, so the overloaded dict is deleted rather than re-derived. Per plan §0.2 crm's
   identifier is a proposal, not authority.

   Also record as a known consequence in your report: an OSS install lacking any one vendor's
   credentials cannot boot. That is the accepted trade-off of this decision, not a defect to work
   around. Do not add a soft-failure escape hatch.

4. **`ProviderSwitchEvent` has a consumer you are not allowed to edit — an ordering constraint.**
   Verified consumers: `clients/llm/events.py:137` (definition), `clients/llm/lifecycle.py:15,194`
   (import and yield — **yours**), `cns/services/orchestrator.py:44,889` (import and isinstance branch
   — **WP2-B's file, on your Do NOT touch list**), and `tests/test_model_routing.py:133`
   (characterization test, WP2's, expected to fail).

   `orchestrator.py` is imported by `main.py`, so deleting the class from `events.py` in this package
   would break application import in a file you are forbidden to fix. Therefore:

   - **Do**: remove the `yield ProviderSwitchEvent(...)` from `lifecycle.py` and its import there. That
     is the substantive fail-loud change — with no provider fallback there is nothing to switch to, so
     the event can never fire.
   - **Do NOT**: delete the class from `events.py`, or touch `orchestrator.py`.
   - **Do**: record in your report and in the commit body that `events.py:137` and
     `orchestrator.py:44,889` now hold a dead class and a dead isinstance branch, and that **WP2-B owns
     deleting both**. WP2-B's prompt already lists `ProviderSwitchEvent` deletion; this makes the
     hand-off explicit.

## Do NOT touch

  - The ~22 leaf call sites (cns/services/*, lt_memory/*, agents/*, tools/*). WP2-B.
  - cns/services/orchestrator.py, working_memory/core.py, cns/api/actions.py, web/settings/index.html.
    WP2-B.
  - deploy/mira_service_schema.sql, deploy/migrations, main.py. WP-S owns these; WP2-B takes main.py's
    usage_pricing seeding.
  - tests/. WP0 owns recovery; O-22 owns the harness.
  - auth/. WP3.

After your change the tree will NOT import cleanly end-to-end, because leaf call sites still pass
internal_llm=. That is expected and is why WP2-B follows immediately. Do not "fix" leaf call sites to
make the tree consistent — that is WP2-B's scope and doing it here makes the two packages unreviewable.

## Verification

1. python3 -m py_compile on every file you changed.
2. `git grep -nE '_MODEL_CONFIG_NAMES|load_model_configs|get_model_config\b' -- utils/ clients/` shows
   the new chokepoint and no leftover get_internal_llm definition.
3. `git grep -c 'internal_llm' -- utils/user_context.py clients/llm/ clients/llm_provider.py` → 0.
4. `git grep -n 'difficult'` → 0 in your files (proves the D14 rename is complete).
5. `git grep -n 'config.api.max_tokens'` → 0 (you removed the field and fixed both readers at :719
   and :1074).
5b. `git grep -n 'ROUTE_FALLBACKS'` → 0 tree-wide, and no replacement criticality set introduced.
    Confirm `utils/power_on_self_test.py` no longer imports it and no longer branches on it.
5c. `git grep -n 'ProviderSwitchEvent'` → still present in clients/llm/events.py and
    cns/services/orchestrator.py (deliberately deferred to WP2-B), and **absent** from
    clients/llm/lifecycle.py.
6. `git grep -n 'validate_compaction_budget'` → present in config/config.py and called from
   load_model_configs().
7. `python3 -c "from utils.user_context import ModelConfig, load_model_configs, get_model_config"` —
   import must succeed even though calling load_model_configs() needs a database.
8. Run the routing characterization test:
   `python3 -m pytest tests/test_model_routing.py -q -p no:cacheprovider --tb=short`
   It failed before your change with ImportError: ModelConfig. Report which tests now pass and which
   still fail, and for each failure say whether it is (a) your defect, (b) a leaf call site WP2-B owns,
   or (c) an environment limit. Do not edit the test.
9. Differential check per the methodology: total passed/failed/skipped/errored before and after your
   commits, on the same branch.

## Commits

Suggested: (1) utils/user_context.py ModelConfig + config/config.py budget validation;
(2) clients/llm resolver/provider/types/lifecycle/events; (3) power_on_self_test five-route
validation; (4) cost_accumulator re-key. Fold where a split creates a non-working intermediate.
The first is a breaking change — use `!` and a BREAKING CHANGE paragraph.

[COMMIT CONVENTION]
[REPORT FORMAT]
```

---

## WP2-C — D10: remove the Anthropic Batch API and Files API upload transport

Chartered 2026-09-06. Plan §12 treats WP2 as one atomic unit that *includes* D10's deletions; the WP2-A/WP2-B
split allocated the chokepoint and the leaf call sites but left D10 — roughly 1,900 lines of Batch removal
plus 500 of Files removal across 21 files — owned by nobody. Eleven of those files are claimed by no package
at all.

**Sequencing is the point of this package.** The Batch half must land **before WP2-B**, because WP2-B's
headline gate (`git grep -E 'internal_llm|conversation_llm' -- '*.py'` → 0 tree-wide) is unreachable while
`lt_memory/llm_routing.py` still holds two `internal_llm` references that no other package owns. The Files
half must land **before WP4 and WP5**, because it deletes lines in `cns/api/websocket_chat.py` and
`cns/services/segment_collapse_handler.py` that those packages otherwise inherit as dead code.

```
cd /Users/taylut/Programming/GitHub/mira-OSS
git worktree add .worktrees/wp2c -b 2.0/wp2c 2.0/integration
```

Branch only after WP2-A and its follow-up fixes have merged into `2.0/integration`.

```
You are executing work package WP2-C of the mira-OSS 2.0 backport: decision D10, removing the Anthropic
Batch API transport and the Anthropic Files API **upload** transport. Plan §8.4 is the specification.

[AUTHORITY BLOCK]

## Working location

    /Users/taylut/Programming/GitHub/mira-OSS/.worktrees/wp2c

Branch 2.0/wp2c. The crm_mira remote is fetched; its refs resolve here as crm_mira/crm_mira.

## Read first

Plan §8.4 (D10 execution), §8.3 (deletions to decline — read this as carefully as §8.4, because the two
lists look alike and one of them is a trap), §0.1 (2.0 posture: no migration, fresh install only),
§0.2 (crm's names and contracts are proposals, not authority), and §6.5.2 (why
`segment_collapse_handler.py` cannot be taken wholesale).

## The single most important fact: this is close to a mechanical replay

Verified by blob diff: **the mira-OSS tree is byte-identical to crm_mira's pre-deletion state** for 13 of
the 16 Batch files and 4 of the 7 Files files. So `git show <sha>` against the upstream deletion commits
gives you patches that apply to your tree unchanged. Use them. The three upstream commits are:

    5c50dfc   Batch extraction removal      16 files, +153/-1413   (NOT named in plan §8.4)
    0134d3d   Files Manager removal          7 files,  +82/-695    (NOT named in plan §8.4)
    58c261b   sidebar batch-mode removal     9 files,  +49/-242    (named in §8.4)

Files that are NOT byte-identical and therefore need hand-editing rather than patching:
`lt_memory/models.py` (17-line drift), `agents/base.py` (84), `cns/services/segment_collapse_handler.py`
(53+), `utils/userdata_manager.py` (4, from WP1 item 8).

Pre-verify every patch with `git show <sha> -- <path> | git apply --check` before applying. Where a patch
does not apply and the reason is not one listed here, stop that item and report it rather than improvising.

## COMMIT 1 — the Batch API half

Delete whole files: `lt_memory/processing/batch_coordinator.py` (317 L),
`lt_memory/batch_result_handlers.py` (142 L), `agents/batch.py` (139 L), `lt_memory/llm_routing.py` (8 L,
the whole file is the one `uses_anthropic_batch_dialect` helper).

Edit, per `5c50dfc` / `58c261b`:

  lt_memory/processing/execution_strategy.py   521 L -> crm's 205 L post-image. Byte-identical pre-image,
                                               so crm's post-image transfers directly. The retained core is
                                               store_and_tend_extraction, _persist_llm_entities and
                                               _build_candidate_hints. **DirectExecutionStrategy is the
                                               post-deletion shape** and is what
                                               tests/test_direct_extraction.py imports; crm's 205-L file
                                               defines it with exactly the constructor and call shape that
                                               test asserts. Do not invent it.
  lt_memory/processing/orchestrator.py         -47 L; touchpoints :23,:52,:58,:64,:153-160,:209-214
  lt_memory/factory.py                         -60 L; :7,:21-23,:29,:69-78,:134-139,:148,:156,:190-197
  utils/lt_memory_jobs.py                      -69 L (the batch poll + cleanup jobs)
  lt_memory/db_access.py                       -238 L, the batch block at :1409-1647. This queries
                                               `extraction_batches`, a table WP-S's greenfield schema does
                                               not create — so this code is ALREADY runtime-dead against a
                                               fresh install. Deleting it is a correctness fix, not only a
                                               scope reduction.
  lt_memory/models.py                          ExtractionBatch (:392-420), BatchStatus (:23-25),
                                               ChunkMetadata (:128-133). Hand-edit; 17-line drift.
  lt_memory/processing/__init__.py             -17 L
  lt_memory/processing/extraction_engine.py    the `for_batch` parameter at :90,:103,:125
  agents/sidebar.py                            :90,:95,:237-254 (the dual pools)
  agents/base.py                               ~-15 L. Hand-edit; 84-line drift. Do NOT touch the
                                               internal_llm_key / overwatch_llm_key / sentry_llm_key fields
                                               — those are WP2-B's renames, not D10's.
  agents/implementations/whilethecatsaway_agent.py  use_batch=True at :30-31, batch_timeout_seconds
  config/config.py                             batch_poll_minutes (:103), batch_cleanup_use_days (:119),
                                               max_concurrent_batch_agents (:185). Also the :107
                                               job_timeout_seconds description that names batch polling.
  utils/sidebar_jobs.py                        **exactly one line, :29** — the
                                               `max_concurrent_batch_agents=sidebar_config.…` argument.
                                               See the adjudication note below.
  cns/api/actions.py                           **:2216 only** — `collapse_segment(event, force_immediate=True)`.
                                               This is an OSS-only call site; crm HEAD's actions.py has no
                                               `force_immediate`. Do not touch anything else in this file;
                                               the picker retirement at :2053-2180 is WP2-B's.

Four sites plan §8.4 does NOT name, all mandatory:

  clients/llm_provider.py            `build_batch_params` at :54-90. Its only callers are
                                     execution_strategy.py:25,:332, which you are rewriting.
  utils/power_on_self_test.py        **:902, :905 and :972** — two required-scheduler-job lists naming
                                     `lt_memory_extraction_batch_polling` and `lt_memory_batch_cleanup`.
                                     :911 raises `RuntimeError(f"Required scheduler jobs not registered:
                                     {missing}")`. If you delete the jobs without editing these lists,
                                     **the power-on self-test fails at every startup.** This is the
                                     hardest dangling reference in the package. Do NOT touch the RLS
                                     expected_tables list (WP-S) or `_check_llm_configuration` /
                                     `_check_llm_provider_reachability` (WP2-A).
  cns/services/segment_collapse_handler.py   the `force_immediate` parameter, its docstring and its
                                     propagation at :164,:174,:314,:487,:504,:529,:535. This file is
                                     touched by BOTH commits — `force_immediate` is Batch (this commit),
                                     `_cleanup_segment_files` is Files (commit 2). The hand-edit
                                     boundary and the retain-list are given under commit 2; read that
                                     before editing this file at all.
  AGENTS.md contract lines           Skip. WP6 owns the documentation sweep (plan D-15).

### Adjudication: the §8.1 vs §8.4 `sidebar_jobs.py` contradiction is resolved

Plan §8.1 says "`utils/sidebar_jobs.py` — LEAVE THIS FILE UNTOUCHED", while §8.4 lists
`SidebarDispatcher(max_concurrent_batch_agents=…)` as a Batch API deletion. Traced end to end: §8.1's
instruction is aimed at the WP1-era sidebar-trigger port, and its own parenthetical — "also omit per D10's
scope; crm removed it as part of batch-mode deletion" — assigns that one line to D10. crm performed exactly
that deletion in `58c261b` (`utils/sidebar_jobs.py | 1 -`). **You own the one-line removal at :29 and
nothing else in that file.** If it is left behind, the attribute read on a pydantic model raises
`AttributeError` at scheduler registration — a boot-time crash, but only when
`sidebar_dispatcher.enabled` is true (there is an early return at :18-20).

### Two things that look like work and are not

- **`forage_agent` has no `use_batch` in mira-OSS.** §8.4's "batch mode in … forage_agent" is a crm-side
  artifact. Its only D10-relevant content is `internal_llm_key="forage"` at :21, which is WP2-B's rename.
  Do not edit this file.
- **`config.batching.batch_max_age_hours`, consumed at `utils/lt_memory_jobs.py:143`, does not exist.**
  `LTMemoryFactory` has no `.config` and there is no `BatchingConfig` anywhere, so the batch cleanup job
  is already AttributeError-dead. Its hunk deletes itself; there is no replacement to author.
- **`ScheduledTaskMonitor.wrap_scheduled_job` is NOT removed tree-wide by D10.** §8.4's parenthetical
  describes crm's hunk only. `segment_timeout_service.py:338` still uses it and `health.py:141` reads job
  stats. Remove it only from `utils/lt_memory_jobs.py`; leave the class alone.

## COMMIT 2 — the Files API upload half

Delete `clients/files_manager.py` (195 L, byte-identical to `0134d3d^`; verified to have no non-Anthropic
consumer — `create_files_manager()` hard-builds `AnthropicDialect`).

Edit, per `0134d3d`:

  utils/document_processing.py       remove the `container_upload` / `document` content types (:59-70) and
                                     the upload branch (:104-129), ~-70/+60. crm replaces this with local
                                     text extraction — `pypdf.PdfReader` to plain text for PDF, and
                                     equivalent local reading for CSV/XLSX/JSON. Port that replacement.
  utils/userdata_manager.py          the `files_api_uploads` SQLite table at :457-472 plus its init call
                                     at :131. Hand-edit; 4-line drift from WP1 item 8. Greenfield posture
                                     means no data migration — the table simply stops existing.
  cns/api/chat.py                    `file_ref` block construction :236-261, manager creation :195,:364.
                                     Note :271-292 is the D5 cost wiring WP2-A already re-attached —
                                     do not disturb it.
  cns/api/websocket_chat.py          `file_ref` blocks :593-605, ~-60 L. **WP4 owns this file's protocol
                                     rewrite and has not run yet.** Touch only the document-upload block;
                                     leave frame emission and auth alone.
  cns/services/segment_collapse_handler.py   `_cleanup_segment_files` def :830-855 and its call :318.
                                     **WP5 owns this file and has not run yet.** Per §6.5.2 the 216-line
                                     crm delta here is WP5 commit `6c055c2` piggybacking D10 deletions —
                                     take ONLY the two D10 items. Retain `_init_feedback_loop`,
                                     `_process_feedback_loop` and `_invalidate_lora_trinket_cache`
                                     (plan §12's WP5 gate requires all three, D1 retains the user model),
                                     and do not touch the `prefs.conversation_llm == 'demo'` branch at
                                     :516-520, which is WP2-B's.
  clients/llm_provider.py            `create_files_manager` :427-442 (sole callers are chat/ws/collapse).
  cns/services/orchestrator.py       the `container_id` **read** side at :1033-1045 only. See the
                                     decision below — do not take crm's full container hunk.
  requirements.txt                   **add `pypdf>=5.0.0`.** Verified absent from mira-OSS and nothing
                                     imports it yet; crm HEAD has it. Plan §6.3.6's dependency table
                                     already anticipates it as required by D10.

### DECISION (already made — do not relitigate): OSS keeps Anthropic code execution

The Files API has an upload side and a download side. **Only the upload side is removed.** mira-OSS's
anthropic routes are live (the `batch` route is `claude-sonnet-4-6`, `assessment` is `claude-opus-4-6`)
and the code-execution artifact pipeline is a retained OSS feature. Therefore KEEP, and report as
intentionally kept:

  cns/core/message.py:41-70                  FileRefBlock, DocumentBlock
  clients/llm/dialects/openai_chat_base.py:542   defensive file_ref handling
  clients/llm_provider.py:529-533                same
  clients/llm/dialects/anthropic.py:579-581      file_ref -> container_upload translation
  clients/llm/dialects/anthropic.py:706-730      _file_artifact_events (the DOWNLOAD side)
  clients/llm/dialects/anthropic.py:59-61        FILES_API_BETA_FLAG
  clients/llm/types.py                           the container fields
  cns/services/orchestrator.py                   the container_id WRITE side (:169,:616,:908-912)

crm removed the container_id read side only and kept the write side at HEAD. Its motivation was that all
its routes were openai; OSS's are not, so porting crm's full container hunk would be inheriting a CRM
artifact. Remove the read side, keep the write side, and say so in the commit body.

### TRAP — do not port this hunk, plan §8.4 misattributes it

§8.4 lists "`utils/logging_config.py` Anthropic SDK instrumentation (−99 L)" under the Files API. **That
attribution is wrong and acting on it would break shipped work.** Verified: the 99-line removal happened in
crm commit `ff12722`, which is the billing/prepaid-account commit, not `0134d3d`; `0134d3d` does not touch
the file at all. Furthermore `instrument_anthropic_client` still exists at crm HEAD
(`logging_config.py:163`), is still called by `clients/llm/dialects/anthropic.py:129-131`, and mira-OSS's
WP1-landed `utils/llm_tap.py:156` documents that it is attached via that function. Removing it would break
WP1's observability.

**Do not touch `utils/logging_config.py`.** The `setup_anthropic_sdk_logging` call at `main.py:15-17` is a
separate billing-provenance question belonging to §8.2/D-14 triage, and `main.py` is WP3-B's file anyway.
Report this exclusion in your commit body so the plan can be corrected.

## Also resolve: plan deferred item D-10 (the orphaned `anthropic_batch_key`)

D10 removes the Batch API but `anthropic_batch_key` stays live — it is the credential for the `batch`
*route*, which survives as an ordinary synchronous anthropic call. Verified live at schema:70,
`postgresql.sh:177,180`, `init-mira.sh:71,228`, `.env.example:20`, `finalize.sh:173,301`, and the
greenfield test's allowlist at :321. **Decision: keep it, do not rename.** Renaming would touch WP-S's
live-validated schema, its test, and four deploy scripts for no functional gain; the name usefully
isolates batch-route rate limits from the `anthropic_key` used by `assessment`.

Record the decision by adding a SQL `COMMENT ON` for the column or an adjacent `--` comment in
`deploy/mira_service_schema.sql` stating that the key names the batch route's credential and that the
Anthropic Batch API transport was removed in 2.0. **Your edit must not alter schema structure** — no
tables, columns, constraints, policies or grants. `tests/test_greenfield_schema.py` currently passes 32/32
and must still pass; it is sensitive to the table set and to the seed.

## Do NOT touch

  - tests/ (WP0/O-22 own recovery and the harness; per the Test scope convention you author no tests)
  - auth/ (WP3)
  - main.py (WP3-B)
  - cns/api/actions.py except :2216; cns/services/orchestrator.py except the container_id read side
  - utils/logging_config.py (see the trap above)
  - utils/user_context.py, clients/llm/resolver.py, clients/llm/types.py, clients/llm/lifecycle.py,
    config/config.py's model-routing sections, utils/power_on_self_test.py's LLM checks (WP2-A, landed)
  - The ~22 leaf internal_llm= call sites, agents/base.py's *_llm_key fields, the model picker,
    phoneafriend_tool (WP2-B)
  - deploy/ except the schema comment described above (WP-S landed; R-3 audited)

The tree does not import cleanly end-to-end before your change (WP2-B's leaf call sites still pass
internal_llm=) and will not after it either. That is expected. Do not migrate leaf call sites to make the
tree consistent — WP2-B follows and doing its work here makes both packages unreviewable.

## Verification

1. `python3 -m py_compile` every changed .py file.
2. `git grep -n 'uses_anthropic_batch_dialect\|batch_coordinator\|BatchCoordinator\|ExtractionBatch\|extraction_batches\|post_processing_batches\|build_batch_params\|files_manager\|create_files_manager\|files_api_uploads' -- '*.py'` -> 0.
3. `git grep -n 'max_concurrent_batch_agents\|batch_poll_minutes\|batch_cleanup_use_days'` -> 0.
4. `git grep -n 'lt_memory_extraction_batch_polling\|lt_memory_batch_cleanup' -- utils/power_on_self_test.py` -> 0.
5. `git grep -n 'force_immediate'` -> 0.
6. **The KEEP list must still be present** — this is the check that catches over-deletion:
   `git grep -c 'FileRefBlock\|DocumentBlock' -- cns/core/message.py` non-zero;
   `git grep -n '_file_artifact_events\|FILES_API_BETA_FLAG\|instrument_anthropic_client' -- clients/llm/ utils/logging_config.py` all present.
7. `git grep -n 'pypdf' -- requirements.txt utils/document_processing.py` -> present in both.
8. Import smoke, no infrastructure: `python3 -c "import lt_memory.processing.execution_strategy, lt_memory.factory, utils.document_processing, clients.llm_provider"`. `lt_memory.processing.execution_strategy` must expose `DirectExecutionStrategy`.
9. `python3 -m pytest tests/test_direct_extraction.py -q -p no:cacheprovider --tb=short` — it currently
   collection-errors with `ImportError: cannot import name 'DirectExecutionStrategy'`. It should now
   collect. Report per-test results; the `model_config="batch"` assertions may still fail until WP2-B
   lands the leaf migration, and if so say that rather than editing the test.
10. `python3 -m pytest tests/test_greenfield_schema.py -q -p no:cacheprovider --tb=no` -> must still be
    32 passed.
11. Differential totals before and after, per the conventions block. Baseline on 2.0/integration is
    `121 failed, 195 passed, 396 skipped, 18 errors`; re-measure on your own branch before starting.
    Expect the collection error for test_direct_extraction.py to disappear. Any OTHER test changing state
    is a regression — investigate and report it.

## Commits

Two, as scoped above: (1) the Batch API half, (2) the Files API upload half plus pypdf. Both are breaking
changes to internal contracts — use `!` and a BREAKING CHANGE paragraph. Cite `5c50dfc`, `0134d3d` and
`58c261b`. State the capability loss explicitly in commit 1's body: plan §8.4 records the Anthropic Batch
API's 50% discount on heavy async extraction as the largest capability regression in the backport, taken
deliberately. State the logging_config.py exclusion and the anthropic-code-execution retention in commit 2's.

[COMMIT CONVENTION]
[REPORT FORMAT]
```

---

## WP2-B — model_configs: leaf call sites and consumers

Create after WP2-A merges:

```
git worktree add .worktrees/wp2b -b 2.0/wp2b <WP2-A-merge-result>
```

```
You are executing WP2-B of the mira-OSS 2.0 backport: migrate every leaf call site and consumer from
the retired internal_llm / conversation_llm vocabulary to the five fixed model_configs routes that
WP2-A landed at the chokepoint.

[AUTHORITY BLOCK]

## Working location

    /Users/taylut/Programming/GitHub/mira-OSS/.worktrees/wp2b

## Read first

Plan §6.1.3 (the route mapping table — authoritative for every consumer), §6.1.5 (D13: retire the
picker, port the effort override), §6.1.2, §0.2. Read utils/user_context.py on your branch first:
WP2-A has already replaced the chokepoint, so get_model_config(name) is what you call.

## The complete call-site inventory

**CORRECTED 2026-09-06.** The original enumeration below was incomplete — ground truth on
`2.0/integration` was 211 references across 39 Python files, and WP2-A's own consumer inventory plus a
whole-tree ownership diff found nine files this list never mapped. The additions are in the block that
follows the inventory; treat the two together as the checklist, and still report any site you find that
is on neither.

Two counts have also moved. The leaf `internal_llm=` sites number **29**, not 22, measured after WP2-A
landed. And **WP2-C runs before you** and deletes several files this inventory lists — see the
"already handled by WP2-C" block. Do not migrate a file WP2-C has deleted.

**internal_llm= keyword call sites, with target route from §6.1.3:**

  -> primary
  cns/services/user_model_synthesizer.py   'synthesis' x2, 'critic' x1
  cns/services/lora_service.py             "synthesis" x2, "critic" x1
  cns/services/summary_generator.py        'summary' x3
  cns/services/live_context_compaction_service.py  "summary" x1
  cns/services/portrait_service.py         "portrait" x2
  cns/services/peanutgallery_model.py      'tidyup' x1

  -> fast
  cns/services/subcortical.py              'analysis' x2
  cns/services/domaindoc_summary_service.py 'analysis' x1
  cns/services/tool_result_summarizer.py   "analysis" x1
  lt_memory/entity_merge.py                'analysis' x1
  utils/prompt_injection_defense.py        'analysis' x1
  tools/implementations/pager_tool.py      'tidyup' x1

  -> batch
  lt_memory/processing/execution_strategy.py 'extraction' x1

  -> assessment
  cns/services/assessment_extractor.py     'assessment' x1
      NOTE: this is plan open item O-2, resolved toward `assessment` because OSS's consumer of that
      route IS assessment_extractor. Do not map it to `batch` just because crm did — crm's `assessment`
      route served an autonomy gate that OSS omits.

  -> other
  tools/implementations/phoneafriend_tool.py :151 MODEL_INTERNAL_LLMS[model_choice], :155

Note 'tidyup' SPLITS by call site: peanutgallery_model -> primary, pager_tool -> fast. That follows
crm's own per-call-site mapping; do not unify them.

**Agents framework — a call-site grep alone misses this.** agents/base.py declares the vocabulary as
dataclass fields:
  :192  internal_llm_key: str            -> model_config_name
  :229  overwatch_llm_key: str | None    -> overwatch_model_config_name
  :330  internal_llm=self.overwatch_llm_key   -> model_config=self.overwatch_model_config_name
  :669-673 docstring + get_internal_llm(self.internal_llm_key) -> get_model_config(...)
  :167, :180, :228, :271, :285, :319 docstrings and guards referencing the old names
Subclass declarations:
  agents/implementations/forage_agent.py:21        "forage"           -> "batch"
  agents/implementations/forage_agent.py:26        "overwatch"        -> "primary"
  agents/implementations/memory_curator_agent.py:99 "summary"         -> "primary"
  agents/implementations/whilethecatsaway_agent.py:24 "whilethecatsaway" -> "batch"

**overwatch token ceiling (plan O-18).** main's overwatch internal_llm row had max_tokens=100; the
`primary` route row is 16000. Passing no override means the observer can emit 160x its intended budget.
Pass max_tokens=100 explicitly at agents/base.py:330 (the resolver honours per-request max_tokens over
the row default). Verify crm's agents/base.py to see whether upstream passed an override; if it did not,
say so and pass it anyway — O-18 is unresolved upstream and 100 is the value main actually used.

**whilethecatsaway and forage referenced routes that were never seeded.** `whilethecatsaway` and
`portrait` are referenced in code at main but have no row in any main schema or migration — latent
KeyError paths. Your migration resolves both; mention that in the commit body.

## ADDITIONS — nine sites the original inventory missed, all verified against source

**Already handled by WP2-C — do NOT touch these, the files are deleted or rewritten:**

  agents/batch.py                      WP2-C DELETES the whole file (139 L). Its :58 docstring reference
                                       to InternalLLMConfig goes with it. Do not edit it.
  lt_memory/llm_routing.py             WP2-C DELETES the whole file (8 L, the one
                                       uses_anthropic_batch_dialect helper).
  lt_memory/processing/execution_strategy.py   WP2-C rewrites 521 L -> crm's 205 L post-image, which
                                       removes the 'extraction' call site the inventory mapped to route
                                       `batch` and introduces DirectExecutionStrategy calling
                                       `model_config="batch"` itself. Re-read the file after WP2-C
                                       merges and migrate only what still carries a legacy kwarg.
  cns/api/actions.py:2216              WP2-C takes the `force_immediate=True` kwarg.
  utils/power_on_self_test.py          WP2-A owns the LLM checks; WP2-C owns the scheduler-job lists at
                                       :902,:905,:972. Nothing here is yours.
  agents/implementations/forage_agent.py   has no `use_batch` in mira-OSS; only the
                                       `internal_llm_key="forage"` rename at :21 is yours.

**Genuinely yours, and missing from the original inventory:**

  main.py:235-236              `from utils.user_context import load_internal_llm_configs` and the call.
                               WP2-A DELETED that function, so this is a live ImportError at startup.
                               Becomes `load_model_configs()`. This is separate from the usage_pricing
                               seeding block already in your brief at ~:240.
  agents/base.py:420-428       `get_internal_llm(self.sentry_llm_key)` plus the `sentry_llm_key` field
                               itself. A THIRD agents-framework key alongside `internal_llm_key` (:192)
                               and `overwatch_llm_key` (:229), and no prior inventory named it. Rename
                               to `sentry_model_config_name` and decide its target route from §6.1.3;
                               report the mapping you chose and why.
  cns/services/orchestrator.py:720-725   `metadata.conversation_llm_name` — a consumer of the RENAMED
                               attribute WP2-A changed to `model_config_name` in
                               `clients/llm/types.py`. Distinct from the :163 field and the :998-1004
                               `resolve_conversation_llm` path your brief already lists.
  cns/services/segment_collapse_handler.py:519   `if prefs.conversation_llm == 'demo':`. Dies twice
                               over — under D13's picker retirement and under D12's member-only
                               decision. Delete the branch. WP2-C also edits this file (the
                               `force_immediate` parameter and `_cleanup_segment_files`), so re-read it
                               after WP2-C merges and touch only this branch.
  clients/llm/events.py:137    the `ProviderSwitchEvent` class definition. WP2-A removed its emission
                               from lifecycle.py and deliberately left the class dead for you.
  cns/services/orchestrator.py:44,889   its import and `isinstance` branch — dead for the same reason.
                               Delete both in the same commit as the class.
  cns/api/websocket_chat.py:510-512, web/assets/javascript/api-client.js:589,
  web/assets/javascript/messaging.js:1491   the `provider_switch` frame renderers, downstream of the
                               same dead event. **All three are WP4's files, and none of them affects
                               your `*.py` headline gate — so delete none of them.** Report all three
                               locations for WP4, which rewrites the WS protocol and the frontend's
                               event handling anyway and runs after you.
  web/assets/javascript/thinking-budget.js:14,38,64   calls `get_conversation_llm` /
                               `set_conversation_llm`. Your brief names only `web/settings/index.html`;
                               this is the picker's second frontend consumer and it must go too, or the
                               settings page will call actions you deleted.
  deploy/schema_aware_restore.py:9,44   `CONFIG_TABLES = {'conversation_llm','internal_llm'}`.
                               **Do not edit.** O-20 is resolved: WP6 deletes this file outright along
                               with `deploy/migrate.sh` and `deploy/lib/migrate.sh`. It is excluded
                               from your headline gate for that reason.

## WP2-C hand-off — read this before you start, it changes your checklist

WP2-C has merged. It deleted the Batch API and the Files API upload transport, and it explicitly
adjudicated several hunks that overlap your inventory. Consequences:

**Already done — do NOT re-migrate these:**

- `lt_memory/processing/execution_strategy.py` — the inventory's `'extraction' x1 -> batch` item is
  **already satisfied**. WP2-C replaced the file with crm's 205-line post-image, which calls
  `model_config="batch"` itself and defines `DirectExecutionStrategy`. Re-read it; migrate nothing.
- `lt_memory/llm_routing.py`, `agents/batch.py`, `lt_memory/processing/batch_coordinator.py`,
  `lt_memory/batch_result_handlers.py`, `clients/files_manager.py` — **deleted**. The inventory's
  `agents/batch.py:58` docstring item is void.
- `cns/api/actions.py:2216` — the `force_immediate=True` kwarg is gone. Your remaining `actions.py`
  scope is the picker (`:2053-2180`), the effort override, and `:2595`'s `rewriter`.
- `cns/services/segment_collapse_handler.py` — `force_immediate` and `_cleanup_segment_files` are gone.
  **Yours is now only the `prefs.conversation_llm == 'demo'` branch** (was `:519`, now around `:509` —
  locate by content, the file shifted).
- `cns/services/orchestrator.py` — the `container_id` **read** block is gone; the write side stays.
  Your scope there is unchanged: `:163` field, `:720-725` renamed-attribute consumer, `:998-1004`.

**Declined by WP2-C as yours — these are on your checklist and still to do:**

- `lt_memory/entity_merge.py` — `internal_llm='analysis'` -> `model_config="fast"`.
- `agents/implementations/forage_agent.py:21` — `internal_llm_key="forage"` -> `"batch"`.
- `agents/implementations/memory_curator_agent.py:99` — `"summary"` -> `"primary"`.
- `agents/implementations/whilethecatsaway_agent.py` — WP2-C removed `use_batch` and
  `batch_timeout_seconds` but **deliberately kept `internal_llm_key = "whilethecatsaway"`** for you to
  rename to `"batch"`.

**Declined by WP2-C as belonging elsewhere — not yours either:**

- `lt_memory/hybrid_search.py` — the `global_memories` -> `global_memories_runtime` rename is
  **WP3-B's** (plan §6.3.8, and open item O-17 alongside `lt_memory/db_access.py`).
- `cns/api/websocket_chat.py` — the `provider_switch` frame renderer and the `InsufficientBalanceError`
  blocks remain. WP2-C was told to leave frame emission and auth alone. The `provider_switch` renderer
  is **WP3-B's** (it is retiring `ProviderSwitchEvent`); the billing block is §8.2/R8.
- Every `AGENTS.md` map — WP6's sweep (plan D-15). WP2-C enumerated the stale batch lines for you:
  `agents/AGENTS.md:25`, `cns/AGENTS.md:31`, `cns/services/AGENTS.md:9,20`, `lt_memory/AGENTS.md:7-8`,
  `lt_memory/processing/AGENTS.md:26`. Do not edit them.
- `clients/llm/dialects/anthropic.py:781` — a docstring still naming `build_batch_params` and
  `agents/batch.py`, both now deleted. That file is outside your scope; it is recorded for WP6.

**Line numbers throughout your brief have drifted.** WP2-A and WP2-C both landed after it was written;
WP2-C reports 30-60 line drift in `power_on_self_test.py`, `config.py`, `websocket_chat.py` and
`openai_chat_base.py`. **Locate every site by symbol or content, not by line number.**

**Baseline for your differential:** `122 failed, 198 passed, 396 skipped, 16 errors`. Re-measure on your
own branch before starting rather than trusting that.

## phoneafriend_tool — D14 consequence, a real tool-contract change

The tool exposes a model choice to the calling model via MODEL_INTERNAL_LLMS[model_choice] at :151,
resolving to phoneafriend_claude or phoneafriend_gemini. With both collapsed onto the single `other`
route there is nothing left to choose between.

**Remove the choice parameter** and hardcode route `other`, updating the tool description and JSON
schema accordingly. A parameter that silently does nothing is worse than no parameter: the model would
reason about a distinction that does not exist. Keep the tool itself — plan §7.3 retains it, and
upstream's deletion was an extraction to a separate MCP server (plan D-13), not a scoping decision.

## D13 — retire the picker, port the effort override

DELETE:
  cns/api/actions.py :2053-2083  the resolve_conversation_llm consumer
  cns/api/actions.py :2106, :2111 the get_conversation_llm / set_conversation_llm action definitions
  cns/api/actions.py :2142-2180  their handlers, including get_conversation_llms(), the `hidden` check
                                 and update_user_preference('conversation_llm', name)
  web/settings/index.html        the model picker UI and its calls to those two actions
  cns/services/orchestrator.py:163  the conversation_llm: str field
  working_memory/core.py:119-123    resolve_conversation_llm(prefs.conversation_llm)
  -> chat becomes the fixed `primary` route, matching crm's orchestrator llm_kwargs

ADD (f8cf0d3, the replacement per-user knob):
  cns/api/actions.py   set_effort_override / get_effort_override / clear_effort_override, validated
                       against EFFORT_LEVELS from clients/llm/types.py (already present)
  cns/services/orchestrator.py  read Valkey effort_override:{user_id} (SETEX 3600) BEFORE subcortical
                       assessment, inject llm_kwargs['effort'], SKIP subcortical when an override is
                       present, fail open on Valkey errors
  Reference: git show f8cf0d3. It is per-user infrastructure keyed on user_id, not CRM-coupled, and in
  single-user mode the key simply carries the single user's id.

main.py: the usage_pricing seeding block (~:240, commented "ensure every conversation_llm +
internal_llm key has a usage_pricing row") must be re-keyed to the five route names. WP-S was
explicitly told to leave this alone for you.

**System prompt consequence (plan R2).** config/system_prompt.txt contains a substrate paragraph —
"This may change between turns if {first_name} switches models mid-conversation. The change in
underlying model will not result in a noticeable change in your perception of being Mira. Your
higher-level cognitive functions are not tied to any particular substrate." — which becomes FALSE once
per-user switching is removed. Delete that paragraph. Do not touch anything else in the prompt; the
seven-delta harvest from 61315bb is WP6 and must not be conflated with this deletion.

## Do NOT touch

  utils/user_context.py, clients/llm/*, clients/llm_provider.py, config/config.py (WP2-A, already landed)
  deploy/mira_service_schema.sql, deploy/migrations (WP-S)
  utils/power_on_self_test.py (WP2-A owns the LLM checks, WP2-C the scheduler-job lists)
  auth/ (WP3), cns/api/websocket_chat.py (WP4), tests/ (WP0/O-22)

**Two deliberate exceptions to the two lists above**, both specified in the ADDITIONS block and in
verification step 2b — do not treat them as licence to wander:

  clients/llm/events.py:137   delete the dead `ProviderSwitchEvent` class only. WP2-A removed its
                              emission and left the class for you. Touch nothing else in `clients/llm/`.
  tests/test_model_routing.py remove the single `usage_pricing` assertion that contradicts D5, per step
                              2b. Change no other assertion and add no test.

**Sequence:** branch from `2.0/integration` only after **WP2-C has merged**. WP2-C deletes
`agents/batch.py` and `lt_memory/llm_routing.py` and rewrites `execution_strategy.py`, so starting
before it lands means migrating call sites in files that are about to disappear.

cns/services/orchestrator.py is shared with WP4 and WP5. Touch ONLY the conversation_llm field, the
llm_kwargs construction, and the effort-override read. Leave message persistence, frame emission and
_surface_memories alone.

## Verification

1. py_compile every changed file.
2. `git grep -nE "internal_llm|conversation_llm" -- '*.py'` → the headline check. Report the count
   before and after (baseline on `2.0/integration` after WP2-A and WP2-C is **29 leaf `internal_llm=`
   sites**; re-measure the raw grep yourself rather than trusting that number).

   **The target is zero with exactly two permitted exceptions**, and neither is a site you may edit:

   - `tests/test_greenfield_schema.py` — its four references are assertions that the
     `conversation_llm` and `internal_llm` **tables are absent** from the schema. They are correct and
     must survive; deleting them would weaken the greenfield gate. Currently 32/32 passing.
   - `deploy/schema_aware_restore.py` — dead per the O-20 verdict; WP6 deletes the file.

   So run it as:
   `git grep -nE "internal_llm|conversation_llm" -- '*.py' ':!tests/test_greenfield_schema.py' ':!deploy/schema_aware_restore.py'`
   and require 0. Report the unfiltered count too, so the two exceptions are visible rather than
   silently excluded.

2b. `tests/test_model_routing.py::test_runtime_has_no_legacy_routing_identifiers` **asserts something
   plan §6.1.6 deliberately contradicts**: it flags `usage_pricing` in `utils/cost_accumulator.py`, but
   D5 retains the `usage_pricing` lookup and only re-keys it. The test can therefore never pass on
   mira-OSS as written. Per §6.3.7's precedent — *amend the test, do not satisfy it* — and the Test
   scope convention, **remove the `usage_pricing` assertion from that test and leave every other
   assertion in it intact.** Record the amendment and its reason in the commit body. Do not otherwise
   edit any test, and do not add one.
3. `git grep -n "internal_llm_key\|overwatch_llm_key"` → 0.
4. `git grep -rn 'model_config=' -- '*.py' | wc -l` → should be roughly 22 plus the agents framework.
5. `git grep -n 'get_conversation_llm\|set_conversation_llm'` → 0 in Python; report what remains in
   web/ if you could not fully remove the picker.
6. Import smoke test, no Vault needed:
   `python3 -c "import agents.implementations.forage_agent, agents.implementations.memory_curator_agent, agents.implementations.whilethecatsaway_agent, tools.implementations.phoneafriend_tool"`
7. `python3 -m pytest tests/test_model_routing.py tests/test_direct_extraction.py -q -p no:cacheprovider --tb=line`
   — report per-test results and classify each failure as yours / WP4-dependent / environment.
   test_direct_extraction.py needs DirectExecutionStrategy in
   lt_memory/processing/execution_strategy.py, which is D10 batch-removal work; if it is missing, say so
   rather than inventing the class.
8. Differential test totals before and after, per the methodology.

## Commits

Suggested: (1) leaf call-site route migration; (2) agents framework rename + overwatch ceiling;
(3) phoneafriend_tool contract change; (4) D13 picker retirement + effort override;
(5) usage_pricing seeding + system-prompt substrate paragraph. Items 3 and 4 are user-visible contract
changes and deserve their own commits with BREAKING CHANGE notes where the tool schema changed.

[COMMIT CONVENTION]
[REPORT FORMAT]
```

---

## WP3-A — auth primitives (no upstream equivalent for two files)

```
git worktree add .worktrees/wp3a -b 2.0/wp3a <current-integration-branch>
```

WP3-A is disjoint from WP-S and WP2 (it adds new auth/*.py files only), so it can run in parallel with
them. It must not start before WP1 is merged, because it depends on WP1 item 7 (vault_client re-auth).

```
You are executing WP3-A of the mira-OSS 2.0 backport: the pure-infrastructure layer of the multi-user
auth stack. No wiring — main.py, cns/api/*, the database layer and the service layer are WP3-B.

[AUTHORITY BLOCK]

## Working location

    /Users/taylut/Programming/GitHub/mira-OSS/.worktrees/wp3a

## Read first

Plan §6.3.1 (mira-OSS's baseline identity model — read this carefully, it explains why the contextvars
are NOT the limitation), §6.3.2, §6.3.4 (the three-mode design and the get_current_user union branch),
§6.3.5 (the graft table: what ports verbatim, what is CRM), §6.3.6 (the email blocker), §0.2.
Decision D3 is full multi-user with MIRA_AUTH_MODE = single (default) | dev | multi.
Decision D12 is member-only: the subject_kind column exists, all demo machinery is omitted.

## Port verbatim from crm_mira — zero CRM coupling, verified

  auth/session.py        283 lines. sha256-hashed Valkey keys (session:<digest>, csrf:<digest>), no raw
                         token at rest, create_session, validate_session with fail-closed expiry,
                         revoke_session, revoke_user_sessions, revoke_user_sessions_except (aa074aa),
                         generate_csrf_token / validate_csrf_token with hmac.compare_digest.
                         ALL Valkey primitives it needs already exist in mira-OSS — I verified each:
                         get_valkey, json_set_with_expiry, json_get, scan, ttl, delete, set,
                         increment_with_expiry (clients/valkey_client.py). Do not add any.
  auth/exceptions.py     15 lines. AuthError(code, message, details).
  auth/dev_mode.py       8 lines. development_mode_enabled() reading MIRA_DEV.
                         NOTE: main.py:99 already reads MIRA_DEV inline for hypercorn dev config. Do not
                         edit main.py (WP3-B owns it) but flag the duplication in your report — plan
                         open item O-4 asks whether to unify or split the variables.
  auth/rate_limiter.py   105 lines. Uses config.RATE_LIMIT_* only.
  auth/security_logger.py 83 lines. Pure logging plus email masking.
  auth/webauthn_service.py 501 lines. Zero CRM/billing. Port the version at crm HEAD, which includes
                         8608696's fix for hex-vs-base64url credential-ID keys. Needs config.APP_URL for
                         rp_id / expected_origin.
  auth/types.py          DELTA on mira-OSS's existing file, not a replacement. Add
                         SubjectKind = Literal["member","demo"]; add subject_kind and demo_expires_at to
                         UserRecord, UserProfile, SessionData and APITokenContext; add first_name,
                         last_name, timezone to SessionData; lowercase CookieSettings.samesite's Literal
                         (daf8e4a — Starlette 0.37.2 lowercases for validation).
                         Take crm's field requirements AS WRITTEN — no compatibility defaults (plan §0.1).
                         mira-OSS's existing auth/api.py constructs APITokenContext with three args; that
                         file is WP3-B's to replace, so leave the resulting inconsistency and note it.
  auth/security_middleware.py  Port the SecurityHeadersMiddleware class and its headers (:27-35) — free
                         security. For the CSP (:38-53) see the D-1 instruction below.

## Two files with NO upstream equivalent — you design these

crm_mira has neither, because crm never needed a single-user mode or a non-CRM provisioner. Specs:

**auth/mode.py** — the MIRA_AUTH_MODE switch.
    AuthMode = Literal["single", "dev", "multi"]
    auth_mode() -> AuthMode       reads MIRA_AUTH_MODE, defaults to "single", and parses STRICTLY:
                                  anything outside the three literals raises ValueError naming the
                                  allowed values. Mirror the strictness of the feature-flag loader in
                                  config/config_manager.py (SYSTEM_FEATURE_FLAG_ENVIRONMENT_FIELDS),
                                  which rejects anything but "0"/"1" rather than coercing truthily.
    single_user_mode_enabled() -> bool    auth_mode() == "single"
  Default MUST be "single": mira-OSS is distributed and most installs are single-user, and single mode
  is the only one that needs no email transport.

**auth/provisioning.py** — the seam that gets CRM out of the account lifecycle.
    class AccountProvisioner(Protocol):
        def provision(self, user_id: str, timezone: str) -> None: ...
        def ensure(self, user_id: str, timezone: str) -> None: ...
        def delete(self, user_id: str) -> bool: ...

    def local_teardown(user_id: str) -> bool: ...
        revoke sessions -> clear manager cache -> rmtree(data/users/<id>) -> DELETE FROM users

    class NullProvisioner:      # the OSS default
        provision / ensure are no-ops; delete returns local_teardown(user_id)

  The CRM-free teardown tail already exists upstream — extract it from
  `git show crm_mira/crm_mira:auth/crm_workspace.py` lines ~219-233:
        SessionManager().revoke_user_sessions(user_id)
        clear_manager_cache(user_id)
        user_dir = Path("data/users") / user_id
        if user_dir.exists(): shutil.rmtree(user_dir)
        clear_user_context()
        with self.session_manager.get_admin_session() as session:
            rows_deleted = session.execute_update(
                "DELETE FROM users WHERE id = %(user_id)s", {"user_id": user_id})
        if rows_deleted == 0:
            raise RuntimeError(f"Account {user_id} disappeared during cleanup")
        return True
  Note it uses the ADMIN session for the DELETE, which is correct — the caller may not have RLS context.
  Use Path(__file__)-anchored resolution for data/users rather than a CWD-relative path, matching what
  WP1 item 8 did to utils/userdata_manager.py. Do NOT port the two preceding try-blocks from that
  function: they call the CRM lifecycle client and billing.get_billing_backend().delete_customer().

  Justification to record in the commit body: this protocol is the minimal excision mechanism, not a
  convergence investment. mira-OSS and crm_mira are parting (plan §0.2); the alternative is six surgical
  edits to auth/service.py plus a permanently forked copy.

## auth/config.py — port with two changes

Remove CRM_BASE_URL (:64) and CRM_LIFECYCLE_SERVICE_SECRET (:68).

**Make the email fields lazy.** crm's AuthConfig.__init__ (:11-24) eagerly reads app_url plus
email_gateway_url / api_key / hmac_secret via get_service_config, which raises when a field is absent.
mira-OSS's deploy/postgresql.sh:166-176 seeds only valkey_url, userdata_encryption_key and
diagnostics_token. If you port the eager reads unchanged, EVERY startup in EVERY mode dies. Read them
lazily on first use so single and dev modes boot without an email transport configured. app_url is
needed by webauthn_service for rp_id/expected_origin, so decide and document whether it is lazy too or
required — if required, WP-S/WP3-B must add it to the deploy seeding; say which in your report.

Also: crm's auth/service.py:676 has a module-level `auth_service = AuthService()` singleton, so
importing the module triggers those Vault reads. You are not porting service.py, but design config.py so
that importing auth.config does NOT require Vault. Note the constraint in your report for WP3-B.

## D-1: the CSP decision

Plan §4 item D-1 admits crm's CSP tightening, but its three follow-ups are not done and it would break
the retained UI (oss_ui.py inlines marked.min.js and purify.min.js into served HTML; web/sw.js needs
worker-src). Port the SecurityHeadersMiddleware headers now. For the CSP, make it **config-driven and
default-off**, with the strict value available, and document the three follow-ups in the module
docstring. Do not hardcode crm's Stripe origins (js.stripe.com, api.stripe.com, hooks.stripe.com) or
its importmap_csp_hash — drop both.

## requirements.txt

Add `email-validator` (MIT, tiny — needed by WP3-B's Pydantic EmailStr) and `webauthn` (BSD-3, pulls
cbor2; cryptography and pydantic are already present). Do NOT add stripe, twilio, pywebpush, requests,
Markdown, html5lib or tinycss2.

## Do NOT touch

  main.py, cns/api/*, auth/api.py, auth/service.py, auth/database.py, auth/account_gc.py,
  auth/email_service.py (WP3-B); deploy/ (WP-S); utils/user_context.py, clients/llm/* (WP2);
  tests/ (WP0/O-22). auth/seed_lora.py STAYS — decision D1 retains the user model.

## Verification

1. py_compile every file.
2. Import each new module standalone with no Vault and no environment set:
   `python3 -c "import auth.exceptions, auth.dev_mode, auth.mode, auth.rate_limiter, auth.security_logger, auth.types, auth.provisioning"`
   This MUST succeed — it is the proof that importing auth does not require infrastructure. auth.session,
   auth.webauthn_service and auth.config may need their third-party packages installed but must not need
   a live Vault at import time.
3. `python3 -c "import auth.session"` and `import auth.config` — report whether either raises, and why.
4. `git grep -n 'billing\|stripe\|square\|crm_workspace\|_crm_client\|demo_seed' -- auth/` → must be 0.
5. `git grep -niE '192\.168\.1\.9|mirafor\.biz|/opt/crm_mira|crm-mira|taylorsatula|@admin\.site' -- auth/`
   → must be 0. This is the plan §11 scrub gate.
6. auth/seed_lora.py still present.
7. Differential test totals before and after.

## Commits

Suggested: (1) auth primitives — exceptions, dev_mode, rate_limiter, security_logger, types delta;
(2) auth/session.py; (3) auth/webauthn_service.py + requirements; (4) auth/config.py lazy email;
(5) auth/mode.py + auth/provisioning.py (the two new designs — give these a thorough SOLUTION
RATIONALE, since there is no upstream commit to cite); (6) security_middleware headers.

[COMMIT CONVENTION]
[REPORT FORMAT]
```

---

## WP3-B — auth service, API, database, graft, and the SMTP sender

**Written 2026-09-06 from WP3-A's actual report.** All three questions this brief was blocked on are
answered; the answers are baked into the specification below rather than left open.

```
cd /Users/taylut/Programming/GitHub/mira-OSS
git worktree add .worktrees/wp3b -b 2.0/wp3b 2.0/integration
```

Branch only after **WP2-B has merged**, because both packages edit `main.py`. WP2-C must also have
merged, because it edits `cns/api/websocket_chat.py`, which you touch for the WS auth path.

```
You are executing work package WP3-B of the mira-OSS 2.0 backport: the auth service, API and database
layers, the CRM excision graft, the three-mode bootstrap, and a new pluggable SMTP sender. WP3-A has
already landed the auth primitives; you build on them and close the deliberate inconsistencies it left.
This is the largest auth package in the programme and the one that unbreaks `single` mode.

### Why this package is urgent

WP3-A landed `auth/types.py` with `subject_kind` **required and no default**, per plan §0.1's
no-compatibility-defaults rule. mira-OSS's existing `auth/api.py:41-45` constructs `APITokenContext`
with three arguments, so from the moment WP3-A merged, `single` mode raises `ValidationError` at
**request time** on every authenticated endpoint. Verified: the tree imports cleanly and the suite is
unchanged, but no authenticated request can be served. That is permitted intermediate breakage — it
never reaches `main` — and **your §6.3.4 union branch is the unblock.** Nothing else in the programme
closes it.

### WP3-A's handoff: five constraints that bind you

1. **`app_url` is already seeded. Do not add it.** Plan §6.3.6's "Action: seed `app_url` in
   `deploy/postgresql.sh` and `init-mira.sh:235`" is **stale** — verified at `postgresql.sh:197` and
   `init-mira.sh:242`, both seeding `app_url="http://localhost:1993"`. That skeleton item is deleted
   from this brief. `AuthConfig.APP_URL` is a lazy `@property` (`auth/config.py:39-43`) read only when
   a `WebAuthnService` is constructed (`webauthn_service.py:44,51,56,58`), never at import or startup.
2. **`auth/config.py` exposes only `APP_URL` plus plain constants.** WP3-A deleted crm's
   `EMAIL_GATEWAY_URL` / `API_KEY` / `HMAC_SECRET` accessors outright rather than making them lazy,
   because D3 replaces that transport and plan §11 lists `email_gateway_*` among the Vault key names
   that map the private security topology and must not reach OSS. It also deleted `DATABASE_URL` and
   `VALKEY_URL` (zero consumers in either repo). **Consequence: your SMTP sender defines its own
   configuration.** Nothing may reintroduce the gateway's key names.
3. **Protect the infrastructure-free import.** `import auth.config` and `import auth.session` now
   perform zero I/O — verified under `env -i` with no Vault and no environment. The pattern that breaks
   it is crm's module-level `auth_service = AuthService()` at `service.py:676`, which `utils/
   scheduled_tasks.py` imports by name string, making a reachable seeded Vault a precondition for
   importing the module at all. **Use lazy `get_auth_service()` (`api.py:137`'s shape) and mode-gate
   the scheduler registration.** Do not reintroduce a module-level singleton.
4. **`AccountProvisioner` is three methods, a non-runtime `Protocol`, idempotent by contract** —
   `provision(user_id, timezone) -> None`, `ensure(user_id, timezone) -> None`,
   `delete(user_id) -> bool`. Inject it as `AuthService.__init__(provisioner: AccountProvisioner =
   NullProvisioner())`. **Do not extend the protocol** — §6.3.5 justifies it as the minimal excision
   mechanism, not a convergence investment, and §0.2 says the repos are parting.
   `local_teardown(user_id) -> bool` canonicalises through `uuid.UUID` before any destructive step,
   then revokes sessions, clears the manager cache, `rmtree`s the user data dir, and deletes the row on
   an **admin** session. Use it in `account_gc.py`.
5. **`auth/webauthn_service.py:31` cannot import** — `from .database import AuthDatabase` is the one
   module in `auth/` that fails, and adding `auth/database.py` is yours. **Do not "fix" this by moving
   the import inside `__init__`**; that is the soft-failure path the gate exists to catch.

### Also deliberate, and also yours

`SecurityHeadersMiddleware(app)` takes **no** `importmap_csp_hash` argument. crm's mount at
`crm_mira/main.py:453` passes one; **do not copy that line.** CSP is governed by `MIRA_CSP=off|strict`
(default `off`, strictly parsed at middleware-stack build). And `auth/mode.py`'s `auth_mode()` has **no
caller yet** — the union branch, `main.py`'s three-mode bootstrap and the scheduler gate are its first
three readers.

[AUTHORITY BLOCK]

## Working location

    /Users/taylut/Programming/GitHub/mira-OSS/.worktrees/wp3b

Branch 2.0/wp3b. The crm_mira remote is fetched; its refs resolve here as crm_mira/crm_mira.

## Read first

Plan §6.3.1 (mira-OSS's baseline identity model), §6.3.2 (what crm actually changed — the mechanism was
untouched, the predicate and call-site discipline were not), §6.3.3 (the request-resolution ladder and
why `cns/api/*.py` must not be ported wholesale), §6.3.4 (the three-mode table and the union branch,
given as code), §6.3.5 (the region map for `auth/api.py` and the CRM-coupling table), §6.3.6 (email
transport), §6.3.8 (multi-user hardening), §8.2 (the eight billing excision points), §0.1, §0.2, and
§11 (the scrub gate — `dev@crm-mira.local` and `taylor@*` are on it).

## Port order

Plan §12 fixes the dependency order, and WP3-A has already landed the first seven entries. Yours begins
at `database.py`:

    database.py -> account_gc.py -> provisioning wiring -> service.py -> api.py -> cns/api/base.py
    -> the SMTP sender -> main.py's three-mode bootstrap -> the WS auth path
    -> the global_memories_runtime Python companion -> get_active_segments -> deploy Vault seeding

### `auth/database.py` (505 L upstream)

Port, then **reconcile `create_user`'s INSERT against the schema that actually landed**. Plan §6.3.5
warns that a direct port is a broken INSERT; it no longer is, because WP-S dropped the same columns crm
did — but verify rather than assume.

**The live `users` column set, read from a PostgreSQL 17.11 server with the greenfield schema applied
(this is verified fact, not inference — the schema has been applied and its 21 tables confirmed):**

    id, email, first_name, last_name, is_active, created_at, last_login_at, webauthn_credentials,
    memory_manipulation_enabled, daily_manipulation_last_run, timezone, temperature_unit,
    subject_kind, cumulative_activity_days, last_activity_date, portrait, portrait_generated_at,
    deletion_requested_at, soft_deleted_at, purge_deadline, demo_start_at, demo_expires_at

No `conversation_llm`, no `balance_usd`, no `llm_tier` (that last one never existed in any schema —
`main`'s, WP-S's or crm's — and a WP2 follow-up already removed the query that referenced it).
`subject_kind` is `TEXT NOT NULL DEFAULT 'member'`. `timezone` is
`NOT NULL DEFAULT 'America/Chicago'`. Check crm's INSERT and `_USER_RECORD_COLUMNS:127-131` against
this list and report any column it writes that does not exist, or omits that is `NOT NULL` without a
default.

- Port `initialize_mira_account()` (`:107`) and `_prepopulate_welcome_content()` (`:235-361`, ~126 L) as
  the replacement for `main.py:127-176`'s inline seeding.
- **Preserve the admin-session / user-session split verbatim** (§6.3.2). Pre-auth reads
  (`create_user`, `get_user_by_id`, `get_user_by_email`, magic-link CRUD, `get_api_token_by_hash`) go
  through `mira_admin` BYPASSRLS; post-auth per-user reads go through `get_session(user_id)`. RLS is
  now enabled on `users`, `magic_links` and `api_tokens` (schema `:553,559,565`), so getting this split
  wrong yields silent zero-row results rather than errors.
- **O-14:** main also calls `seed_lora_postgres()` (`main.py:177`), retained under D1 — **keep that call
  and the `feedback_synthesis_tracking` init.** `increment_segment_turn()` depends on
  `segment_turn_count` being present; verify `_prepopulate_welcome_content` establishes it.
- Remove the `:124-126` comment referencing the billing `users_provision_member_entitlement` trigger;
  that trigger is not in the 2.0 schema.

### `auth/service.py` (678 L upstream)

- Excise the six CRM touch points — `:26` (import), `:41` (attr), `:132-140` and `:187-194`
  (compensation blocks), `:243` (`ensure_workspace`), `:267-289` (`_initialize_account`) — via
  `AccountProvisioner`. §6.3.5 gives `_initialize_account`'s three lines: keep
  `initialize_mira_account(...)` (generic), cut `provision_workspace(...)` and the `seed_demo` branch.
- **Prefer lazy `get_auth_service()` over the module-level singleton at `:676`** — see WP3-A constraint 3.
- Generalise the hardcoded dev identity at `:216-231`: `dev@crm-mira.local`, `"Taylor"`, `"Developer"`,
  `"America/Detroit"`, `"Develop crm_mira locally"`. **All five are §11 scrub-gate items.** Derive them
  from configuration or use neutral placeholders; do not ship a personal name or a CRM domain.
- Port: `request_magic_link` (`:299`, with the `:324-341` enumeration defence and `random.gauss` timing
  jitter), `verify_magic_link:401`, `create_api_token:468` (`:488`'s 50-token cap),
  `validate_api_token:518`, `create_session:552`, `logout_other_devices:607`,
  `cleanup_expired_tokens:631`, `get_cookie_settings:637-647`, `register_cleanup_jobs:657`.
- `get_cookie_settings` returns `secure=not development_mode_enabled()` and `samesite="lax"` — the
  Lax change is `2ac4660` and §8.5 confirms it arrives with WP3. WP3-A already lowercased the
  `CookieSettings.samesite` Literal in `types.py` for Starlette 0.37.2 (`daf8e4a`).
### `auth/api.py` (1,178 L upstream)

§6.3.5 carries the full region map — use it line by line. Summary of dispositions:

- **Keep the exact name `get_current_user`.** All seven OSS consumers
  (`cns/api/{actions,chat,data,files,location,tool_config,trigger_rules}.py`) must keep importing it
  unchanged.
- **Prepend §6.3.4's single-user union branch**, given verbatim in the plan. It preserves
  `auth/api.py:21-45`'s exact contract — same 401 strings, same `token_id="oss_single_user"`, same
  contextvar writes — and supplies `subject_kind="member"` explicitly, which is what unblocks the
  `ValidationError` described above. Then fall through to crm's full session / API-token ladder.
  **Step 8 of that ladder is where RLS context is established** (`set_current_user_id` +
  `set_current_user_data`); the union branch must do both, exactly as §6.3.4's code shows.
- Excise `_require_billing_entitlement` (`:299-336`, imports `billing` at `:306-307`) and the whole
  `*_entitled_*` ladder (`:338-350`, `:1165-1178`) — §8.2's excision points.
- Skip `get_current_member` / `get_current_member_session` (`:274-297`) per D12.
- `get_current_user_for_pages` (`:1115-1163`) 302s to `/login/` at `:1141,1148,1160` — **mira-OSS has no
  `/login/` page.** Drop it or repoint it; report which.
- Repoint `/dev/session`'s `RedirectResponse("/workspace/", 303)` (`:473`) to `/chat`.
- `:107-133` does `dataclasses.replace` on a **frozen** `ErrorResponse` and depends on
  `SuccessResponse`, `create_success_response`, `create_error_response`, `generate_request_id` and
  `APIError`. **Verify frozen-ness and the helper signatures on landing** — §6.3.5 flags this.
- Take crm's `cns/api/base.py` delta (+4 L: `http_status: NotRequired[int]` on `ErrorDetail` and
  `ResponseMeta`).

### `auth/account_gc.py` (152 L upstream)

Port the concept — deleting never-activated signups after 24 h is generic hygiene and directly enables
multi-user signup. `cleanup_unactivated_accounts:35`, `register_account_gc_job:127` (`0 3 * * *`).
Replace `CRMWorkspaceLifecycleService.delete_account()` with the injected provisioner's `delete()`
(which reaches `local_teardown`), and drop the `LEFT JOIN crm_workspaces` (`:64`) and the
`lifecycle_state='cleanup_pending'` branch (`:71`).

### `auth/email_service.py` — the SMTP sender (new design work, no upstream reference)

crm's 165-line `email_service.py` is generic code but a client for a **proprietary HMAC-signed HTTP
gateway** (`config.EMAIL_GATEWAY_URL` / `API_KEY` / `HMAC_SECRET`, payload `{email, token, app_url}`,
via `requests`). Its server side is the untracked `mira_email_gateway_forbiz.php`. No PyPI package
substitutes, and §11 lists `email_gateway_*` among the Vault key names that must not reach OSS.

**DECISION (E-8, user-confirmed): an abstract `MailSender` interface with a stdlib `smtplib` default
backend.** Requirements:

- Define a `MailSender` protocol in `auth/email_service.py` (or a sibling module if that reads better).
  **Mirror `AccountProvisioner`'s conventions, not its parameter list** — WP3-A's guidance: three
  methods shaped as primitives in / `bool` out, `None` for "did it", raise for "could not do it". A mail
  sender needs `(recipient, subject, body)`-shaped arguments, and if it has no `ensure` analogue,
  **do not invent one.**
- Decide explicitly whether the protocol is `@runtime_checkable`. `AccountProvisioner` is **not**, so a
  mis-shaped implementation fails at the call rather than at injection. If you want injection-time
  validation for the mailer, that is a deliberate divergence — make it and say so.
- Ship one backend: **stdlib `smtplib` only, no new dependency**, configured from environment
  (`MIRA_SMTP_HOST`, `MIRA_SMTP_PORT`, `MIRA_SMTP_USER`, `MIRA_SMTP_PASSWORD`, `MIRA_SMTP_FROM`, and a
  `MIRA_SMTP_STARTTLS` toggle). It must work against any relay an operator already has — Postfix, an SES
  SMTP endpoint, Mailgun, a Gmail app password. **Define this configuration yourself**; per WP3-A
  constraint 2, `auth/config.py` has no email fields at all and must not gain the gateway's Vault key
  names.
- Keep the magic-link email's *content* concerns where they are: `app_url` comes from the lazy
  `AuthConfig.APP_URL`, and the enumeration defence and timing jitter live in `service.py`, not here.
- **Email is required only in `multi` mode** (§6.3.4's table). `single` and `dev` must boot and serve
  with no mailer configured. Fail loud at the point a send is attempted in `multi` without
  configuration — not at import, not at startup.
- Drop the `requests` dependency this file currently implies; §6.3.6's table says rewrite on the
  existing `httpx` **if** you need HTTP at all. With an SMTP-only backend you should need neither.
- Leave room for an operator to register an HTTP backend (Resend/SES/Postmark) without editing this
  module, but **do not ship one** — that would tie OSS to a vendor against §0.1.
### `main.py` — the three-mode bootstrap

§6.3.4's table is the specification. Per mode: `ensure_single_user(app)` **runs verbatim** in `single`,
is replaced by a dev-session bootstrap in `dev`, and is removed in `multi`; the `user_count > 1 →
sys.exit(1)` guard at `:66` is kept in `single`, relaxed in `dev`, removed in `multi`;
`/oss-auth/token` is mounted in `single` only; `auth.api.router` mounts at `/v0/auth` in all three.

- **Retain every existing page route and the `/assets` mount** (`main.py:623-666` region) — ungated in
  `single` as today, gated in `dev`/`multi`.
- Mount `SecurityHeadersMiddleware` with **no** `importmap_csp_hash` argument.
- Gate the scheduler registration on mode: the `auth.service` job runs in `dev` and `multi`, not
  `single`.
- **O-10 — confirmed defect, one list entry, and it is yours because you own `main.py`.** `main.py:290`
  calls `flush_except_whitelist(preserve_prefixes=["session:", "rate_limit:"])`. `auth/session.py`'s
  `_csrf_key()` writes `csrf:<digest>` **paired with** `session:<digest>`, and `revoke_user_sessions_except`
  deletes both together. So every restart currently flushes CSRF tokens while preserving sessions, and
  the first unsafe cookie-authenticated request 403s (`CSRF_REQUIRED`/`CSRF_INVALID` via
  `_requires_cookie_csrf` / `_validate_cookie_csrf`) until the client re-fetches `/csrf`.
  **Add `"csrf:"` to that list.** `clients/valkey_client.py:355` needs no change — verified, the
  whitelist mechanism is generic and already rejects an empty prefix.
- **O-4 — the `MIRA_DEV` double-read.** `main.py:647` (in the `__main__` hypercorn block; plan §6.3.4
  says `:99`, which is wrong) reads `MIRA_DEV` inline while `auth/dev_mode.py:7` reads it for auth. Two
  readers of one env var is a defect. Unify on `development_mode_enabled()`, **or** split into
  `MIRA_DEV` and a separate auth variable. Decide, implement, and record which — §10 leaves it open.
- Note `ensure_single_user` already uses `get_admin_session()` for its `SELECT COUNT(*) FROM users`
  (`main.py:57`) and its row read (`:66`), which is why RLS on `users` does not make it see zero users
  and provision a duplicate on every boot. **Preserve that.** Any new pre-auth read you add must
  likewise go through an admin session.

### `cns/api/websocket_chat.py` — the auth path only

Using `a4df669`'s dual-protocol shape (plan §6.4.5), plus `set_current_user_data()`. §6.3.1 records the
defect this fixes: the WS path at `:185-222` calls **only** `set_current_user_id` (`:221`), so
`get_current_user()` raises on WS-originated work.

**The protocol rewrite itself is WP4, which has not run.** Touch only the handshake and auth. WP2-C has
already removed the `file_ref` document-upload block from this file, and WP2-A left a dead
`provider_switch` frame renderer at `:510-512` — **delete that renderer** as part of retiring
`ProviderSwitchEvent`, and report it, since WP2-B was told to leave this file to you.

### Do not port `cns/api/*.py` wholesale (§6.3.3)

Their entire delta is the `get_current_user` → `get_current_entitled_user` rename plus CRM domain
handlers, and `get_current_entitled_user` hard-imports `billing` → `ImportError` at request time under
D7. All seven OSS consumers keep importing `get_current_user` unchanged.

Related, from §6.3.8: **`cns/api/federation.py:83` — omit the gating entirely.** crm's `fb7065f` adds a
`subject_kind == "demo"` rejection then `from billing import get_billing_backend`; under D7+D12 that
would raise `ImportError` inside the endpoint, giving a 500 on every lattice delivery. mira-OSS's
endpoint already has the `X-Lattice-Delivery-Token` Vault check and `sender_verified` requirement.

### §6.3.8 multi-user hardening — portable independently, and yours

- **The `global_memories_runtime` Python companion.** `lt_memory/hybrid_search.py:169`
  `FROM global_memories gm` → `FROM global_memories_runtime gm`. **O-17:** verify
  `lt_memory/db_access.py`'s second reader is switched too. Plan cites `:507`, but **WP2-C deletes
  `db_access.py:1409-1647`, so line numbers have shifted — locate the query by content, not line.**
  The view is `security_barrier=true` and gated by `can_read_global_memories()`, which requires an
  active user; verified live: deactivating a user drops the view from 1 row to 0. mira-OSS's `global_memories`
  has a full CRUD grant to `mira_dbuser` at main, which at N>1 lets any authenticated user write the
  shared table every other user reads — a cross-user prompt-injection vector. **Confirm the landed
  schema grants SELECT only on the view, not the table** (verified: it does).
- **`get_active_segments()`** (`cns/infrastructure/continuum_repository.py`, crm `:1091-1114`): take
  `AND users.is_active = TRUE` — main scans deactivated users' active segments. Requires column
  qualification (`messages.id`, `messages.continuum_id`, …) because the JOIN makes them ambiguous.
  **Skip the `LEFT JOIN entitlements` half** (D7).
- `agents/sidebar.py:292-305` and `utils/scheduled_tasks.py:164-183` already have `is_active = TRUE`;
  crm's only addition is the entitlements JOIN. **Nothing to take.**
- `utils/domaindoc_shares.py:131` — see O-25 below.
- `utils/scheduled_tasks.py`: re-add `('auth.service','auth_service',False,None)` (`92d768c`),
  **mode-gated**. Do not take the `get_users_due_for_job()` rewrite.
- `utils/power_on_self_test.py:454-500` (`92d768c`) validates nine Vault service-config fields.
  **Port the shape, not the list** — crm's demands `crm_base_url`, `crm_lifecycle_service_secret`,
  `square_application_id/secret`, `stripe_*` and `email_gateway_*`. Use an OSS-appropriate list.
  **Touch only this region of that file**: WP-S owns the RLS `expected_tables` lists, WP2-A the LLM
  checks, WP2-C the scheduler-job lists.
- `cns/integration/event_bus.py`: `publish(self, event: ContinuumEvent)` → `publish(self, event: object)`
  is functionally a no-op (dispatch was already structural on `event.__class__.__name__`). Take or skip;
  say which.

### O-25 — domaindoc sharing under RLS (decision E-14, user-approved)

RLS on `users` is unconditional in the landed schema, so three cross-user readers silently return
nothing in `multi` mode:

- `cns/api/actions.py:1849` and `:1890` — `SELECT id, first_name, email FROM users WHERE email = …`,
  the collaborator lookup by email. **This is how a share target is added**, so sharing becomes
  impossible.
- `cns/api/actions.py:1928`, `:2008`, `:2031` and `utils/domaindoc_shares.py:132` —
  `JOIN users u ON ds.collaborator_user_id = u.id` / `ds.owner_user_id = u.id`, the share listings, so
  collaborator names and emails vanish.

Plan §6.3.8 already ruled crm's fix (`JOIN LATERAL active_member_identity(ds.owner_user_id) u ON TRUE`)
**not portable**, because that SECURITY DEFINER function filters `subject_kind='member'` and belongs to
the demo machinery D12 omits.

**Decision E-14: author a minimal SECURITY DEFINER lookup exposing only `id, email, first_name` for
active users, with no member gating** — crm's pattern minus the CRM. Roughly 15 lines of SQL plus the
`GRANT EXECUTE … TO mira_dbuser` and `REVOKE EXECUTE … FROM PUBLIC` pair that
`can_read_global_memories()` already demonstrates in the same file. Route the five call sites through
it. Rejected alternative: an admin-session read, which would bypass RLS entirely for a user-facing
query.

**This is the one item here that requires editing `deploy/mira_service_schema.sql`.** It has been
applied to a live PostgreSQL 17.11 server and its 21 tables, 19 policies and five-row seed are verified.
Your addition must be **additive only**: one function, its two grants, and the rewritten joins. Do not
alter any table, column, constraint, policy or existing grant. `tests/test_greenfield_schema.py` is
32/32 and must stay 32/32 — it is sensitive to the table set and the seed.

If you judge the function unsafe or the rewrites larger than described, **stop and report** rather than
substituting an admin-session read.

### Deploy

Seed the SMTP settings for `multi` mode in `deploy/postgresql.sh` and
`deploy/docker/scripts/init-mira.sh`, following whatever configuration mechanism you chose for
`MailSender`. **Do not add `app_url` — it is already seeded** (WP3-A finding, verified at
`postgresql.sh:197` and `init-mira.sh:242`). Do not reintroduce `email_gateway_*` key names (§11).

Note R-3 just landed three fixes here: role provisioning now sets the sentinel password (WP-S had made
the roles passwordless while the Vault URLs still embedded it, breaking every default install),
`init-mira.sh` now seeds `subcortical_key`, and `python.sh` fails fast when its schema patch misses.
Re-read those files rather than working from an older mental model.

### `tests/test_auth_graft.py`

Excise the ~7 CRM-only tests (§6.3.5): `import billing`, `from cns.api import demo`, crm workspace
tokens, crm page visibility. **Keep** session hashing, `logout_others`, cleanup ordering,
compensation-on-failure, the removed-OSS-auth contracts, the explicit-RLS-identity checks and the
`conversation_llm` negative assertion (which matches the `main.py` rewrite WP-S required).

Its `:676` module-level `auth_service = AuthService()` singleton is what forces a reachable seeded
Vault on import — see WP3-A constraint 3. **Per the Test scope convention: excise, do not rewrite, and
author no new tests.**

## Do NOT touch

  auth/{session,exceptions,dev_mode,rate_limiter,security_logger,types,webauthn_service,mode,
      provisioning,config,security_middleware}.py   WP3-A, landed. You consume them; you do not edit
      them. The single exception is if adding auth/database.py forces a change in webauthn_service.py's
      import — it should not, and if it does, report it rather than editing.
  auth/seed_lora.py            STAYS and is unchanged — D1 retains the user model.
  utils/user_context.py, clients/llm/**, config/config.py   WP2-A, landed.
  lt_memory/**, agents/**, clients/files_manager.py, utils/document_processing.py,
      utils/lt_memory_jobs.py   WP2-C.
  cns/api/actions.py except the five O-25 query sites; cns/services/orchestrator.py; web/**   WP2-B.
  tests/ except test_auth_graft.py's excision.
  deploy/mira_service_schema.sql except the additive O-25 function and its two grants.

`cns/api/oss_ui.py` + `deploy/oss_ui/{marked.min.js,purify.min.js,chat.html}` are a **plan §0
invariant**: `GET /oss-auth/token` is the identity source for `single` mode, and `oss_ui.py:19-20`
reads both vendor assets **at import time**, so deleting `deploy/oss_ui/` raises during
`create_app()`. You gate its *mount* by mode; you do not modify or delete it.

Likewise `tools/implementations/web_tool.py`, `utils/url_safety.py` and `utils/http_client.py` stay at
main's version — they carry the `e401d59` SSRF fix (GHSA-rmgf-f8wc-rc3p) that crm_mira lacks.

## Verification

1. `python3 -m py_compile` every changed file.
2. **WP3-A's gate must still hold** — this is the check that catches you reintroducing import-time
   coupling: `env -i PATH="$PATH" python3 -c "import auth"` and every `auth.*` submodule including
   `auth.database` and `auth.webauthn_service`, with no Vault and no environment set. All must import.
   `auth.webauthn_service` failing on a missing `auth.database` is the defect you are closing.
3. `python3 -c "from auth.api import get_current_user"` succeeds, and the name is unchanged.
4. `git grep -n 'billing\|stripe\|square\|crm_workspace\|_crm_client\|demo_seed\|email_gateway\|entitled' -- auth/ cns/api/ main.py utils/scheduled_tasks.py` → 0.
5. §11 scrub gate over your files:
   `git grep -niE '192\.168\.1\.9|mirafor\.biz|/opt/crm_mira|crm-mira|taylorsatula|@admin\.site|dev@|Taylor|America/Detroit' -- auth/ main.py cns/api/ deploy/` → 0.
6. `git grep -n 'csrf:' -- main.py` → present in the flush whitelist (O-10).
7. `git grep -n 'global_memories\b' -- lt_memory/` → every reader now uses `global_memories_runtime`.
8. `python3 -m pytest tests/test_auth_graft.py -q -p no:cacheprovider --tb=short` — report per-test
   results and classify each failure as yours, another package's, or an environment limit. Many will
   need live infrastructure; say which.
9. `python3 -m pytest tests/test_greenfield_schema.py -q -p no:cacheprovider --tb=no` → **must still be
   32 passed** after your O-25 schema addition.
10. Differential totals before and after: `python3 -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -2`.
    Re-measure your own baseline first. Report every test that changes state and why.
11. **The headline acceptance gate from plan §12:** the server boots identically with `MIRA_AUTH_MODE`
    unset, and all seven `cns/api/*` consumers are unchanged. You cannot fully exercise this without
    Postgres/Vault/Valkey — state precisely what you verified statically and what remains unverified
    rather than claiming a boot you did not perform.

## Commits

Follow §12's dependency order, one concern per commit, folding only where a split would create a
non-working intermediate. Suggested: (1) `auth/database.py`; (2) `auth/account_gc.py` + provisioner
injection; (3) `auth/service.py` excision + dev-identity generalisation; (4) `auth/api.py` graft with
the single-user union branch — **this is the commit that unbreaks `single` mode, and it is a breaking
change to the auth contract, so use `!` and a BREAKING CHANGE paragraph**; (5) the `MailSender`
interface + SMTP backend (no upstream commit to cite — give it a thorough SOLUTION RATIONALE);
(6) `main.py` three-mode bootstrap + O-10 + O-4; (7) the WS auth path + `provider_switch` removal;
(8) §6.3.8 hardening including the O-25 SECURITY DEFINER function.

[COMMIT CONVENTION]
[REPORT FORMAT]
```

---

## WP4 — WebSocket protocol, ordered persistence, halt, keyset, frontend patch

```
git worktree add .worktrees/wp4 -b 2.0/wp4 <integration-branch-after-WP3>
```

```
You are executing WP4 of the mira-OSS 2.0 backport: the strict WebSocket protocol, ordered turn
persistence, halt support, keyset history pagination, and the ~150-line patch to the retained frontend.

[AUTHORITY BLOCK]

## Working location

    /Users/taylut/Programming/GitHub/mira-OSS/.worktrees/wp4

## Read first — plan §6.4 in full

§6.4.1 the eight defects at main (this is what you are fixing)
§6.4.2 what crm provides, with every frame model and line reference
§6.4.3 the frame-by-frame compatibility cost, VERIFIED FIRST-HAND — 4 inbound breaks, 8 outbound, and
       which single frame survives
§6.4.4 the three server-side items the frontend patch cannot fix (R5, R6, R8) — these are the ones that
       get missed, because they look like client problems and are not
§6.4.5 auth: port a4df669's shape, not HEAD's
§6.4.6 keyset pagination (D-2: cursor-only, no dual-mode shim)
§6.4.7 what to omit
Also §0.1 (2.0 posture — no backwards compatibility, so take crm's contracts outright) and §0.2.

## Decisions already made — do not relitigate

D2: port the FULL strict protocol and patch the retained frontend. The old UI is the only UI mira-OSS
has until a later session; the patch is not a compatibility shim.
D-2: cursor-only get_history. crm's _get_history() raises on offset and search (data.py:117,119).
     Rewrite web/assets/javascript/history.js accordingly. Mirror crm's own split: search_continuums()
     KEEPS offset pagination via SearchHistoryResult.
D-3: fail-loud. Exceptions propagate rather than degrading silently.

## Server side

Take from c1297b3, 457a56e, a11d04e and d138ca4. **Do NOT take `edbbed2`** — an earlier draft of this
brief listed its "part b", but that half moves WebSocket close codes to private values inside
`web/assets/chat-transport.js`, a class belonging to crm's new ES-module frontend. **That file does not
exist in mira-OSS** (verified: `web/assets/javascript/` holds `api-client.js`, `messaging.js`,
`history.js`, `events.js` and others, but no `chat-transport.js`), and §6.4.7 omits it as
web-redesign-specific. Its other half adds stream-chunk logging to `utils/llm_tap.py`, which is a
separable diagnostic improvement in a WP1-owned file; skip it and mention it in your report if you think
it is worth a follow-up. The complete construct list with line references is in plan §6.4.2 —
ProtocolModel with extra="forbid", the ClientFrame and ServerFrame
discriminated unions, TypeAdapter validation on BOTH directions, ChatConnection's single-reader /
single-writer bounded queues (32 in, 128 out) with ClientDisconnected and StopWriter sentinels,
AssistantStep, TurnAccumulator.append_text returning a stable entry_id, _build_turn_messages emitting
provider-step order with monotonic microsecond offsets from base_time, staged user-message commit with
pending_messages committed on failure, TurnCompletedEvent moved to a post-commit callback and suppressed
when stopped or auto_continuing, transient_system_scaffold plus discard_transient_user_message,
tool_stream_frame(), set_cancel_reason / get_cancel_reason, and check_cancelled() at the five points in
tool_loop.py. **`_format_tool_indicator` is DELETED, not adapted** — §6.4.2 lists it last and it is the
fix for defect 4: it prepends `[used: tool_a, tool_b]` into `acc.response_text`, which is persisted and
then **re-fed to the model on every later turn**, so the marker accumulates in durable history. Removing
it is a data-hygiene fix, not a cosmetic one; say so in the commit body.

Supporting changes: cns/core/message.py gains transient_system_scaffold, tool_arguments, turn_id,
partial_response, stop_reason and provider_stop_reason (stop_reason is RENAMED to provider_stop_reason
so stop_reason can mean halt/disconnect); cns/core/continuum.py gains
add_user_message(*, message_id, metadata) and discard_transient_user_message.
UnitOfWork.pending_messages and add_post_commit_callback ALREADY EXIST at
cns/infrastructure/continuum_pool.py:41,86 — do not reimplement them. Message is already a dataclass
with id/created_at/metadata as constructor params, so _build_turn_messages needs no object.__setattr__
hack.

## The three items the frontend patch cannot fix (§6.4.4) — do these deliberately

R5: crm's _server_frame_for_event maps ONLY text->assistant_delta and tool_event->tool, returning None
    for everything else. The orchestrator still emits {"type":"thinking",…}. So thinking and model_error
    are dropped by the transport and MessageFrame.include_thinking becomes a lie. RESTORE forwarding of
    both.
R6: crm's TurnCompleteFrame carries only type, turn_id, segment_id. The retained UI needs
    continuum_id, response, metadata.tools_used, metadata.processing_time_ms and metadata.emotion —
    api-client.js:667-696 reads them and messaging.js:888 extractEmotionEmoji consumes the emotion.
    Plan §6.6 KEEPS <mira:my_emotion> in the system prompt, and orchestrator.py:1102 parses with
    preserve_tags=['my_emotion']. ADD these fields to TurnCompleteFrame.
R8: crm's authenticate() inlines `from billing import get_billing_backend` and raises AccountAccessError
    with billing_url="/settings/#billing"; AccountAccessRequiredFrame and _has_product_access exist for
    it. EXCISE all four (D7 omits billing).

## Auth (§6.4.5)

Port a4df669's dual-protocol authenticate(), NOT crm HEAD's cookie-only version:
    token = auth_data.get("token") or websocket.cookies.get("session")
    session_data = self.session_manager.validate_session(token)
    if not session_data: api_token_data = self.auth_service.validate_api_token(token)
plus a third branch comparing against app.state.api_key -> app.state.single_user_id for
MIRA_AUTH_MODE=single. Also port a4df669's set_current_user_data(session_data.model_dump()) — main calls
only set_current_user_id, so get_current_user() raises on WS-originated work today.
If WP3-B has already landed this, verify rather than redo, and report which.

## Client side (~150 lines)

web/assets/javascript/api-client.js: send message_id as a real UUID instead of
_generateId()'s `msg_${Date.now()}_${rand}` (:775-777); drop the `stream` field; rename the switch cases
at :546-595 to the new frame vocabulary; send {type:'halt', turn_id} instead of {type:'cancel'} (:463),
capturing turn_id from turn_started; keep the auth frame's token field only if the server's single-user
branch still accepts it (it does, per §6.4.5).
web/assets/javascript/messaging.js: the await at :1577 and setGenerating(false) at :1579 must resolve;
tool frames already prefer data.tool_name at :1551 and :1555 so they survive; extractEmotionEmoji at
:888 needs R6's emotion field.
web/assets/javascript/history.js: rewrite from offset to cursor — :75-98, :410-433, :243-245, :461, :695,
:711-712 all use offset/next_offset/has_more.
web/assets/javascript/events.js: the stop button at :133-135.

## Also close plan open item O-23 while you are in the orchestrator

e26d031's circuit-breaker finalization was deferred from WP1 because I wrongly judged it WP4-dependent.
It is portable: main already has acc.invoked_tool_loader, already emits CircuitBreakerEvent, and already
has container_id. Add circuit_breaker_finalization_reason, the CircuitBreakerEvent, one ToolErrorEvent
per unexecuted local call, the explanatory fallback text and the terminal CompleteEvent. Without it a
tripped breaker stops the loop silently.
**Its loader-gating half is no longer blocked — O-7 is closed.** The argument shape was settled by WP2-A:
`clients/llm/lifecycle.py:108` synthesises `{"load": [tool_name]}`, which exactly matches
`tools/implementations/invokeother_tool.py:94-103` (schema declares only `load` and
`load_for_rest_of_session`) and `:110`'s `run()` signature. The orchestrator's detection was settled by
WP2-B: `orchestrator.py:772-777` used to gate `acc.invoked_tool_loader` on
`event.arguments.get("mode", "")` against `["load","fallback","prepare_code_execution"]`, but the tool
has never declared a `mode` property, so the flag was never set. WP2-B replaced that with detection on
the parameters the tool actually declares.

So by the time you run, `acc.invoked_tool_loader` works and
`test_successful_tool_loader_triggers_auto_continuation` should already pass. **Verify that, then
implement the gating half of `e26d031`** — gate auto-continuation on the loader reporting
`success:true`. If the test does not already pass, WP2-B's fix did not land as described; investigate
and report rather than re-fixing the detection yourself.
The other known failure in that file,
`test_circuit_breaker_remains_latched_after_final_no_tools_pass`, **is yours** and closes with the
finalization work above.
Also land e370468's orchestrator hunk (exclude invalid_reason calls from persisted_tool_ids), which
could not apply in WP1 because persisted_tool_ids did not exist yet. Yours does.

## Two mira-OSS tests assert the OLD protocol and must be replaced

tests/api/test_websocket_endpoint.py asserts type in {text, complete, pong} and a ping handler — the
pre-D2 vocabulary. It is superseded by tests/test_web_frontend_protocol.py.
tests/api/test_data_endpoint.py asserts offset/search pagination on ?type=history — exactly what D-2
removes (test_history_respects_offset_parameter, test_history_supports_search_query, and an
"offset" in pagination assertion).
**Delete both** rather than authoring replacements, and say so in your report. Do not leave tests
asserting a protocol you removed — and do not write new tests covering the protocol you added. The
recovered characterization tests (`test_web_frontend_protocol.py`, `test_history_cursor.py`,
`test_ordered_turn_persistence.py`) are already your acceptance gate; per the Test scope convention,
this codebase chooses fail-fast over extensive performative testing.

## Verification

1. py_compile every changed .py file.
2. The three characterization tests are your acceptance gate:
   python3 -m pytest tests/test_ordered_turn_persistence.py tests/test_web_frontend_protocol.py \
       tests/test_history_cursor.py -q -p no:cacheprovider --tb=short
   All three currently fail or error (AssistantStep ImportError, and cursor support absent). Report
   per-test results after your change.
3. tests/test_orchestrator_tool_loop.py's two known failures
   (test_circuit_breaker_remains_latched_after_final_no_tools_pass,
   test_successful_tool_loader_triggers_auto_continuation) should now PASS if you closed O-23 and O-7.
   If either still fails, say which and why.
4. Demonstrate each of the eight §6.4.1 defects is fixed. For defect 5 (the <system-scaffold> prompt
   rendered as a user bubble) show that get_history(message_type="regular") no longer returns it.
5. `git grep -n 'assistant_delta\|turn_complete\|turn_stopped\|turn_error\|protocol_error' -- web/`
   shows the client now speaks the new vocabulary.
6. `git grep -n "type: 'cancel'\|type:'cancel'" -- web/` → 0.
7. No `billing` import anywhere in cns/api/websocket_chat.py.
8. Differential test totals before and after.
9. If you can, start the server and exercise one turn end to end. If you cannot (no Vault/Postgres),
   say so explicitly and list what remains unverified by execution. Do not claim a runtime verification
   you did not perform.

[COMMIT CONVENTION]
[REPORT FORMAT]
```

---

## WP5 — Persona as a second parallel system

```
cd /Users/taylut/Programming/GitHub/mira-OSS
git worktree add .worktrees/wp5 -b 2.0/wp5 2.0/integration
```

Branch after **WP2-B has merged**. WP5 runs **in parallel with WP3-B** — their owned sets are verified
disjoint — and **before WP4**, not after it as an earlier draft of this brief said. WP5 touches
`cns/api/actions.py` and `web/settings/index.html`, both of which WP2-B also owns, so WP2-B must land
first; nothing in WP4 depends on WP5.

### Two large pieces of this package are already done — do not redo them

**All schema work is complete.** §6.5.3's objects were authored by WP-S directly into the greenfield
`deploy/mira_service_schema.sql` and have been **applied to a live PostgreSQL 17.11 server and
behaviourally verified**: `persona_revisions`, `persona_state` and `persona_signals` all exist under
those names; the append-only grant on `persona_revisions` is SELECT+INSERT with no UPDATE or DELETE;
RLS is enabled on both; and the `provision_baseline_persona()` AFTER INSERT ON `users` trigger was
**observed to fire**, creating `persona_state` and `persona_revisions` rows for every inserted user.
So the trigger provisions revision 1 for the `ensure_single_user` bootstrap too, which is §12's WP5
gate. **You write no SQL.** Verify your repository module against the landed schema and report any
mismatch, but change nothing in `deploy/`.

**WP2-C already made the D10 deletions in `segment_collapse_handler.py`.** See the wiring section below
for what that leaves you.

### State you inherit

- The five-route `model_configs` contract is live. `persona_service.py`'s four `model_config="primary"`
  call sites are already correct — **verify, do not change.**
- `effort='none'` now means thinking disabled on the wire. Persona's routes are `primary`, so this does
  not affect you, but `EFFORT_LEVELS` admits `'none'` if you validate any effort input.
- `cns/api/actions.py` has been edited by WP2-B: the model picker is retired, the D13 effort-override
  actions are added, and the `rewriter` purpose is migrated. **Locate every site by symbol, not line
  number** — §6.5.4's line references predate three packages.
- `web/settings/index.html` likewise: WP2-B removed the picker UI. The LoRA panel at §6.5.4's
  `:697-800` must still be there and still working, since D1 keeps `LoraDomainHandler`.

```
You are executing WP5 of the mira-OSS 2.0 backport: add the Persona subsystem ALONGSIDE the existing
user model. This is an addition, not a replacement.

[AUTHORITY BLOCK]

## Working location

    /Users/taylut/Programming/GitHub/mira-OSS/.worktrees/wp5

## Read first — plan §6.5 in full, plus §1 "D1 basis"

§1's D1 basis table is the reason this package exists and must be understood before editing: the two
subsystems model DIFFERENT SUBJECTS. The user model records descriptive observations about the user
(lora_trinket.py: "descriptive (observations about the user), not prescriptive (instructions to Mira)")
and drives the <behavioral_checkin> collaborative debrief. Persona evaluates MIRA against a behavioural
contract (persona_evaluation_system.txt: "Evaluate MIRA, not the user." / "Do not infer user traits,
preferences, or knowledge.") and produces prescriptive directives. Upstream they shared one prompt slot,
which is why they look like two versions of one feature. They are not.

THE USER MODEL IS RETAINED IN FULL. Do not delete or modify assessment_extractor.py,
user_model_synthesizer.py, lora_service.py, system_prompt_parser.py, feedback_repository.py,
feedback_tracker.py, lora_trinket.py, auth/seed_lora.py, the eight user-model prompt files,
_process_checkin_response, or the repulsion rewrite loop. Decision D6 keeps the repulsion loop too.

## Port

  cns/services/persona_service.py            385 lines, verbatim
  cns/infrastructure/persona_repository.py   317 lines, with the table rename below
  working_memory/trinkets/persona_trinket.py 27 lines, with TWO edits below
  config/prompts/persona_{critic,evaluation,refinement}_{system,user}.txt and
  config/prompts/persona_manual_refinement_user.txt   (7 files)

## Three changes from upstream — all required

1. **persona_trinket.py variable_name must be "persona_directives", not "behavioral_directives".**
   Upstream reuses the user model's slot. mira-OSS keeps both, so they need separate slots. Also change
   _invalidate_cache's hdel field name to match.
2. **The Persona signals table is `persona_signals`, not `feedback_signals`.** WP-S has already created
   it under that name, because upstream reuses `feedback_signals` with a DISJOINT column set
   (OSS: signal_type, section_id, strength, synthesized; crm: outcome, behavioral_section, strength,
   evidence, evaluated_at, consumed_by_revision_id). Rename every reference in
   persona_repository.py — table names, FKs, index names, RLS assumptions. Verify against
   deploy/mira_service_schema.sql on your branch and report any mismatch.
3. **Model route names.** persona_service.py calls model_config="primary" at four sites
   (:92, :188, :266, :281). That is already correct for mira-OSS 2.0 — WP2 landed the five-route
   contract. Verify, do not change.

## Wiring

  working_memory/composer.py SECTION_LAYOUT — add "persona_directives" as a NEW entry alongside the
      existing "behavioral_directives". Do not replace it.
  cns/integration/factory.py — register PersonaTrinket IN ADDITION TO LoraTrinket. Upstream swaps them
      at :192 and :212; mira-OSS keeps both.
  cns/services/segment_collapse_handler.py — call _process_persona() IN ADDITION TO the existing
      _process_feedback_loop(), and add _get_persona_service(). Do NOT take this file wholesale.

      **The upstream 216-line delta is now mostly spent.** It came from a single crm commit, `6c055c2`
      ("replace LoRA user model pipeline with immutable Persona revisions"), which bundled four
      concerns. Their current disposition:

        batch removal + force_immediate deletion (D10)   DONE by WP2-C — do not redo
        _cleanup_segment_files / Files-API deletion      DONE by WP2-C — do not redo
        demo-user skip on prefs.conversation_llm         WP2-B's — it dies under D13/D4, not yours
        the Persona swap                                 **YOURS, and it is an ADDITION here, not a
                                                          swap** — crm deleted the user model, D1 keeps it

      So your hand-edit reduces to: add `_process_persona()` (~18 L) and `_get_persona_service()`
      alongside the existing methods, and wire the call. **Do not copy crm's deletions — they are either
      already applied or forbidden.** Verify the three D1-retained methods are still present before and
      after your edit: `_init_feedback_loop`, `_process_feedback_loop`,
      `_invalidate_lora_trinket_cache`. §12's WP5 gate requires all three.

      WP2-C recorded its touch map for this file so you can see what moved: the `force_immediate`
      parameter, docstring, propagation and extraction block, a stale batch comment, the
      `_cleanup_segment_files` call and its definition. The file has shifted — locate by symbol.

      Note D-3: _process_persona does NOT swallow exceptions, unlike _process_feedback_loop (84 L,
      4 components, lazy-init retry, broad except). A persona failure propagates into the
      collapse-attempt counter toward MAX_COLLAPSE_ATTEMPTS = 3 tombstone. That asymmetry is intended;
      leave both behaviours as they are and mention it in the commit body.
  Feature flag: gate Persona behind MIRA_PERSONA_ENABLED, added to the
      SYSTEM_FEATURE_FLAG_ENVIRONMENT_FIELDS registry that WP1-B generalised in
      config/config_manager.py. Strict "0"/"1" parsing, omission at construction in the factory (not
      runtime branching), following the pattern already used for subcortical and peanutgallery.
      Default: enabled.
  cns/api/actions.py — add a PersonaDomainHandler with NEW action names. Do NOT reuse the LoRA panel's
      six (get/refine/accept/decline/update/reset), because LoraDomainHandler still serves
      web/settings/index.html:697-800 and D1 keeps it working.
  cns/api/data.py — add DataType.PERSONA alongside the retained DataType.LORA.

## Verification

1. py_compile every changed file.
2. `python3 -m pytest tests/test_persona_service.py -q -p no:cacheprovider --tb=short` — it currently
   errors with ModuleNotFoundError: cns.infrastructure.persona_repository. Report per-test results. If
   the test hardcodes the table name feedback_signals, it will need the persona_signals rename too;
   that is a legitimate amendment (plan §0.2 — crm contracts are not authoritative), but report it.
3. `git grep -n 'feedback_signals' -- cns/infrastructure/persona_repository.py` → 0.
4. `git grep -n 'behavioral_directives'` → must still show the USER MODEL's trinket and composer entry.
   `git grep -n 'persona_directives'` → must show the new ones. Both present is the success condition.
5. `git grep -n 'MIRA_PERSONA_ENABLED'` → registry, factory, and .env.example documentation.
6. Confirm the user model is untouched: `git diff --name-only <base>..HEAD` must NOT include
   assessment_extractor.py, lora_service.py, feedback_tracker.py, feedback_repository.py,
   user_model_synthesizer.py, system_prompt_parser.py, lora_trinket.py, auth/seed_lora.py, or any
   repulsion_rewriter prompt.
7. Differential test totals before and after.

[COMMIT CONVENTION]
[REPORT FORMAT]
```

---

## WP6 — system prompt, documentation, scrub gate, release identity

```
git worktree add .worktrees/wp6 -b 2.0/wp6 <integration-branch-after-WP5>
```

```
You are executing WP6, the final package of the mira-OSS 2.0 backport: the system-prompt harvest, the
documentation pass, the secrets/PII scrub gate, and release identity.

[AUTHORITY BLOCK]

## Working location

    /Users/taylut/Programming/GitHub/mira-OSS/.worktrees/wp6

## Read first

Plan §6.6 (the prompt harvest — read the whole section, especially which commit to harvest from),
§8.5 (documentation and deploy), §11 (the scrub gate, with every located private value),
§0.1 (the four 2.0 release obligations), §10 (open items O-15, O-16, O-20, O-21, O-24 are yours).

## 1. System prompt — harvest from 61315bb, NEVER from crm HEAD

crm HEAD's config/system_prompt.txt is a raw dump from a deployed CRM box (193977c) and is degraded:
it contains the typos `astutue`, `illedgable`, `toolcals`, one garbled sentence ("an observation,
question, or connection that advances the session provides value" — two verbs welded together), and it
DELETED the <mira:my_emotion> block that mira-OSS's retained frontend depends on.

Read `git show 61315bb:config/system_prompt.txt` and hand-merge SEVEN insertions into mira-OSS's
existing file. `git checkout 61315bb -- config/system_prompt.txt` is INVALID — that copy contains
<role>, <data>, <display> and <operation> wrappers full of CRM content. The seven deltas, with their
exact wording, are enumerated in plan §6.6's table. Work through them one at a time.

Must NOT come: the <role> block ({company_name} is an unresolved placeholder — working_memory/core.py
substitutes only {first_name}, relative time, {model_id}, {model_name}, so it would render literally);
the <display> block (nine lines about viewcard_tool and CRM card types mira-OSS does not have); the
<context> line about a 20lh chat window (binds to crm's CSS); the <data> block except that you MAY
extract its one generic principle ("Use the word {first_name} uses") into <collaboration>; and the
removal of <mira:my_emotion>, the forage paragraph, or the file-links line.

The substrate paragraph is ALREADY deleted by WP2-B (plan R2). Verify it is gone; do not re-delete or
restore it.

O-16: the line "Don't make up file links. Write files to the sandbox." needs REWORDING, not retention
or deletion, because D10 removed clients/files_manager.py. Decide the correct wording for a 2.0 that
extracts documents locally to plain text via pypdf.

O-15: after your insertions, verify cns/services/system_prompt_parser.py's get_assessable_sections()
still resolves. The user model anchors observations to system-prompt <section id=> values, so prompt
edits can silently invalidate the anchor set. Report what you find.

Then run 61315bb's own self-consistency check on your merged result: grep for em dashes and for
contrastive negation ("not X, it's Y"). Its thesis is that the prompt teaches by demonstration, and it
banned em dashes while containing six. Report the counts; fix what the prompt itself forbids.

## 2. Documentation pass

Update the per-directory AGENTS.md maps for what actually landed. Do NOT merge crm's versions — they
describe the CRM product. Add auth/AGENTS.md (minus its "one CRM workspace" invariant clause) and
scripts/AGENTS.md (minus the deploy_remote entry). Skip billing/, web/, web/assets/, workphone/.

Fix these known-stale references (plan §8.5):
  tools/implementations/AGENTS.md documents phoneafriend_tool.py — upstream deleted it, mira-OSS KEEPS
      it, so this one is correct for OSS; verify it describes the post-D14 single-`other`-route contract
  config/prompts/AGENTS.md:31 documents repulsion_rewriter_*.txt with consumer FeedbackDomainHandler —
      mira-OSS KEEPS both (D6), so verify the consumer name is right for OSS rather than deleting the entry
  config/AGENTS.md still names analysis_enabled, which WP1-B removed
  clients/AGENTS.md said "Callers pass exactly one of primary, fast, or batch" — **WP2-A already fixed
      this.** Verify rather than redo.
  deploy/mira_service_schema.sql's COMMENT ON TABLE model_configs — **WP-S already fixed this** (it now
      describes all five routes and names `other` as deliberately a different vendor from `primary`).
      Verify.

**Added since this brief was drafted — the stale-reference sweep is larger than §8.5 knew.** WP2-C
deleted the Batch API and enumerated what it deliberately left for you:

  agents/AGENTS.md:25                 still names max_concurrent_batch_agents
  cns/AGENTS.md:31                    stale batch contract line
  cns/services/AGENTS.md:9,20         stale batch contract lines
  lt_memory/AGENTS.md:7-8             stale batch contract lines
  lt_memory/processing/AGENTS.md:26   stale batch contract line
  clients/llm/dialects/anthropic.py:781   a DOCSTRING still naming build_batch_params and
      agents/batch.py — both deleted. WP2-C left it deliberately because the file belonged to a
      concurrent package. It is a comment-only fix; make it.

Verify each against the current tree rather than trusting the line numbers, and grep for other
batch/files references the enumeration may have missed.

O-24: rule on memory-curation-v2-serf-implementation-brief.md. crm's e26d031 deleted this 799-line root
document as a bundled, unrelated change; WP1-C declined the deletion. Decide whether 2.0 still wants it.

O-20 is **RESOLVED — delete, do not keep as a backup path.** An audit of `deploy/` established the
evidence chain, so this is execution rather than a decision:

- The only live entry is `deploy/deploy.sh:66-71`, which `source`s both `lib/migrate.sh` and
  `migrate.sh` when `MIGRATE_MODE` is true.
- **`deploy/migrate.sh` does not parse.** Line 575 is `if nohup mira >/dev/null 2>&1 &; then` —
  `bash -n` fails with `syntax error near unexpected token ';'`. The same failure reproduces on
  `main`, and blame points to `0254a8c1` (2025-12-29), so it is **pre-existing, not introduced by this
  programme**. Because `source` parses the whole file under `set -e`, `deploy.sh --migrate` dies before
  its first phase. **The entire family, including the rollback path, has been unreachable for months.**
- **Do not fix line 575.** Repairing the syntax would resurrect a half-broken path:
  `capture_database_snapshot` and `verify_database_snapshot` silently record `0` for dropped tables
  (`lib/migrate.sh:684`, `:1196`), so a user would be prompted to "acknowledge data loss" on tables 2.0
  deleted by design.
- `schema_aware_restore.py` has exactly one caller (`lib/migrate.sh:1567`) and its headline feature is
  keyed to `CONFIG_TABLES = {'conversation_llm','internal_llm'}` (`:44`) — both tables are gone, so
  under 2.0 it degrades to a generic table loader.

**Action: delete `deploy/migrate.sh`, `deploy/lib/migrate.sh` and `deploy/schema_aware_restore.py`, plus
the `--migrate`/`--dry-run` branch and its usage text in `deploy/deploy.sh:7-13,44-77`, which becomes
dangling.** `deploy/deploy_database.sh` is **not** part of the family — it is a standalone fresh-install
tool and stays. In the README's reinstall-not-upgrade wording, say that a user salvaging 1.x data needs
only `pg_dump` and manual work; that is the honest replacement for the deleted path.

The stale table and column references the audit found inside those files (`lib/migrate.sh:640,645-646,
684,970,1196`, `schema_aware_restore.py:9,44`, plus `continuums.last_message_position` and a
`user_activity_days` `ORDER BY created_at` against a table with no such column) all die with the
 deletion — **do not fix them individually.**

## 3. Release identity (plan §0.1 obligations)

O-21: VERSION is 2026.06.25 (CalVer, identical in both repos). Set the 2.0 marker and decide whether the
scheme becomes semver (2.0.0) or stays CalVer with a major suffix.
Rewrite README.md for 2.0. It must state explicitly that **1.x -> 2.0 is a reinstall, not an upgrade**,
and that 1.x conversation history, memories and domain knowledge do not carry forward. That is the first
thing a 1.x user needs to know.
Update docs/MANUAL_INSTALL.md for the greenfield schema and the deleted migrations directory.
Retain mira-OSS's NEARFUTURE_FEATURES.md — crm deleted it; that deletion was not adopted.

**R-8 — pin floors for critical dependencies.** `requirements.txt:27` lists `anthropic` with **no
version constraint**, while `clients/llm/dialects/anthropic.py:341` sends `output_config` — a parameter
the older 0.52.2 SDK does not have (verified: no `output_config` in `Messages.create`'s signature,
though `thinking` and `ThinkingConfigDisabledParam` are present). This is **pre-existing, not a 2.0
regression**: the same reference exists at `main` and at crm HEAD, and crm's `requirements.txt:30` is
likewise unpinned. It is not an install blocker either — a fresh `pip install` resolves to the latest
SDK, which is new enough.

It is a **reproducibility defect** for a distributed package whose posture is fresh-install-only: an
install pinned by a distro, a lockfile or a warm cache breaks at *runtime* with a `TypeError` on the
`batch` route, and nothing catches it earlier. Note the `assessment` route is unaffected —
`effort='none'` emits `thinking={"type":"disabled"}` and never reaches `output_config`.

**Action:** establish the minimum `anthropic` version providing `output_config` and pin a floor
(`anthropic>=<version>`). Audit the other unpinned critical entries the same way — `openai`,
`httpx[http2]`, `psycopg`, `valkey` — for any feature the code uses that a floor would protect.
**Pin floors, not exact versions**; exact pins fight the rest of the dependency graph.

**Three loose ends the `deploy/` audit found, all yours:**

- `deploy/docker/scripts/init-mira.sh:249-252` writes `/opt/vault/provider_endpoint.txt` and
  `provider_model.txt` for non-Groq container providers, with a comment saying the rewrite "will be done
  after PostgreSQL is running via s6". **Nothing in the repository reads either file** — a pre-existing
  dead mechanism. Consequence: a non-Groq container install keeps the groq endpoint on the `fast` route,
  and no container flow ever rewrites the `primary` row. **Implement it or delete it**; do not leave a
  mechanism that looks like it configures the install and does not.
- `deploy/oss_ui/chat.html` has **no Python consumer** (`git grep chat.html -- '*.py'` is empty), while
  its siblings `marked.min.js` and `purify.min.js` are read at import time by `oss_ui.py:19-20` and are a
  §0 invariant. Rule on `chat.html`: delete it, or document what serves it. **Do not touch the two
  vendor scripts.**
- `deploy/deploy_database.sh:151` prints `password: new_secure_password_2024` as example text in its
  post-install "Next steps" output. A placeholder, not a credential, but it reads like one. Reword.

Also: the repository's `.git/config` registers a `deploy` remote at
`ssh://admin@192.168.1.9/...`. **That is not in the tree and will not ship**, so it is not a scrub-gate
failure — but it is the private appliance IP, and it is worth removing from the working repository
while you are doing the §11 pass. Report whether you removed it.

**D-15 — the two architecture overviews.** `scratch/MIRA_ARCHITECTURE_OVERVIEW.md` (882 lines) and
`scratch/MIRA_ARCHITECTURE_OVERVIEW copy.md` (599 lines) are gitignored and were never shipped. Both
document `internal_llm`/`conversation_llm` resolution, `batch_result_handlers` and `files_manager` —
**all three are gone from 2.0.** Plan D-15's instruction is to **re-derive from the post-backport tree
rather than update them.** Decide whether 2.0 ships an architecture overview at all; if it does, write
it against the tree you have, not against those files.

## 4. Scrub gate — plan §11. This is the release blocker.

Run every check in §11 against the FULL tree, not just your own changes:

  git grep -nE '192\.168\.1\.9|/opt/crm_mira|mirafor\.biz|Qwopus|k3-256k|kimi_key|llama_server_key|crm-mira|crm_mira|taylorsatula|/Users/taylut|42069|admin@|mira-origin' \
      -- . ':!.pi/plans' ':!scratch'

The plan file itself legitimately quotes these values as things to scrub, so exclude it. Report every
hit with file:line and either fix it or explain why it is acceptable.

Also verify:
  no seed_closeout_test.sql, card-playground.html, workphone.md, DROPLET.md, droplet.env.example,
      deploy_remote.sh, openrouter_opus_chat.sh or extract_imessage_corpus.py anywhere in the tree
  license.txt is still AGPL-3.0 (34523 bytes) — crm replaced it with a 29-byte profanity
  .gitignore retains the _crm_client.py and crm_*_tool.py entries, and did NOT inherit crm's bad-paste
      junk lines (@pytest.fixture, @pytest.mark.parametrize, @router.delete/get/patch/post/put)
  no stripe_, square_, twilio, pywebpush or business_voice reference survives outside the plan file
  requirements.txt contains no CRM dependency

## Verification

Report actual output for: the scrub grep (must be empty or fully explained); `wc -c license.txt`;
the em-dash and contrastive-negation counts in the merged prompt; O-15's answer; a `git diff --stat`
of your whole package; and differential test totals. Confirm `python3 -m compileall -q .` exits 0.

[COMMIT CONVENTION]
[REPORT FORMAT]
```
