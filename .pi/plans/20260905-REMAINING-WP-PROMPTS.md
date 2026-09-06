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

---

## WP2-A — model_configs: the LLM-layer chokepoint

Create the worktree first:

```
cd /Users/taylut/Programming/GitHub/mira-OSS
git worktree add .worktrees/wp2a -b 2.0/wp2a <WP-S-merge-result>
```

where `<WP-S-merge-result>` is the integration branch carrying WP-S (see the handoff for the current
chain). WP2-A must not start before WP-S is merged, because it validates against the `model_configs`
table WP-S authors.

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
  clients/llm/events.py       drop ProviderSwitchEvent — but FIRST run
                              `git grep -n ProviderSwitchEvent` and report every consumer.
                              cns/api/websocket_chat.py may render provider_switch; if it does, note it
                              for WP4 rather than editing that file (WP4 owns it).
  config/config.py            remove ApiConfig.max_tokens and ApiConfig.analysis_enabled's replacement
                              validate_compaction_trigger_tokens; add
                              validate_compaction_budget(primary_max_tokens: int). Take this from
                              6899d07's config/config.py hunk — WP1-B deliberately declined it because
                              it was WP2's. Keep SystemConfig.subcortical_enabled and peanutgallery_enabled
                              (WP1-B added the former).
  utils/power_on_self_test.py _check_llm_configuration asserting the five routes; and FIX the two
                              config.api.max_tokens readers at :709 and :1064, which break when you
                              remove that field. Do NOT touch the RLS expected_tables list — WP-S owns it.
  utils/cost_accumulator.py   re-key from internal_llm row names to model_config_name (D5). Keep the
                              FALLBACK_PRICES mechanism and the usage_pricing lookup; only the keying
                              changes. Its docstring explicitly names OSS as the reason FALLBACK_PRICES
                              exists — preserve that intent.
  cns/api/chat.py             re-attach cost recording (:272-292 start/drain) if WP-S or D10 disturbed it

## Three things that are yours and easy to miss

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

3. **ROUTE_FALLBACKS must be re-derived, not copied.** crm's is
   {"assessment": "primary", "difficult": "fast"}, encoding a local-llama-server vs cloud split that
   does not hold for OSS seeding (plan §6.1.4). WP-S seeded all five routes at cloud providers
   (openrouter / groq / anthropic). Therefore: `other` gets **no** fallback, because falling back would
   silently convert "consult an outside model" into "consult yourself". Make criticality derive from
   what the chat path depends on rather than from route names. If every route is cloud and therefore
   fatal, an empty ROUTE_FALLBACKS is the correct answer — say so explicitly in the commit body rather
   than leaving a stale mapping.

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
5. `git grep -n 'config.api.max_tokens'` → 0 (you removed the field and fixed both readers).
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

I enumerated these myself; treat this as the checklist and report any site you find that is not on it.

**internal_llm= keyword call sites (22), with target route from §6.1.3:**

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
  utils/power_on_self_test.py (WP2-A)
  auth/ (WP3), cns/api/websocket_chat.py (WP4), tests/ (WP0/O-22)

cns/services/orchestrator.py is shared with WP4 and WP5. Touch ONLY the conversation_llm field, the
llm_kwargs construction, and the effort-override read. Leave message persistence, frame emission and
_surface_memories alone.

## Verification

1. py_compile every changed file.
2. `git grep -nE "internal_llm|conversation_llm" -- '*.py'` → must be 0 across the whole tree. This is
   the headline check; report the count before and after.
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

Create after WP3-A and WP-S have both merged. Brief to be written once WP3-A reports, because it depends
on how WP3-A resolved the `app_url` laziness question and the `auth.config` import constraint. Skeleton:

- `auth/database.py` — port; reconcile `create_user`'s INSERT against WP-S's actual `users` columns
  (crm's version omits `conversation_llm`/`balance_usd`, which WP-S also omits, so this may now align);
  port `initialize_mira_account()` and `_prepopulate_welcome_content()` as the replacement for
  `main.py`'s inline seeding; keep the admin-session / user-session split verbatim.
- `auth/service.py` — excise the six CRM touch points (`:26`, `:41`, `:132-140`, `:187-194`, `:243`,
  `:267-289`) via WP3-A's `AccountProvisioner`; generalise the hardcoded dev identity at `:216-231`
  (`dev@crm-mira.local`, `"Taylor"`, `"America/Detroit"`); prefer lazy `get_auth_service()` over the
  module-level singleton at `:676`.
- `auth/api.py` — the region map is in plan §6.3.5. Keep the name `get_current_user`; add the
  single-user union branch from §6.3.4; excise `_require_billing_entitlement` (`:299-336`) and the
  `*_entitled_*` ladder; skip `get_current_member*` (D12); repoint `/dev/session` from `/workspace/` to
  `/chat`; drop or repoint `get_current_user_for_pages` (mira-OSS has no `/login/`).
- `auth/account_gc.py` — port the concept, replace `CRMWorkspaceLifecycleService.delete_account()` with
  `NullProvisioner`/`local_teardown`, drop the `LEFT JOIN crm_workspaces` and the `cleanup_pending`
  branch.
- `auth/email_service.py` — **new design work, no upstream reference.** crm's posts to a private
  HMAC-signed gateway whose server side is an untracked PHP file. D3 requires a pluggable SMTP sender.
- `main.py` — the three-mode bootstrap: keep `ensure_single_user()` running in `single` mode; mount
  `auth.api.router` at `/v0/auth`; mount `cns/api/oss_ui.py` in `single`/`dev` only; gate the scheduler
  registration on mode; retain every existing page route and the `/assets` mount.
- `cns/api/websocket_chat.py` — only the auth path, using `a4df669`'s dual-protocol shape (plan §6.4.5),
  plus `set_current_user_data()`. The protocol rewrite itself is WP4.
- Do not port `cns/api/*.py` wholesale (plan §6.3.3).
- `utils/scheduled_tasks.py` — re-add the `auth.service` registration, mode-gated; do not take
  `get_users_due_for_job`'s entitlements JOIN.
- Deploy: seed `app_url` (and email settings for `multi`) in `deploy/postgresql.sh` and
  `deploy/docker/scripts/init-mira.sh`.
- Excise `tests/test_auth_graft.py`'s ~7 CRM-only tests (plan §6.3.5) — keep session hashing,
  `logout_others`, cleanup ordering, compensation-on-failure, the removed-OSS-auth contracts, the
  explicit-RLS-identity checks and the `conversation_llm` negative assertion.

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

Take from c1297b3, 457a56e, a11d04e, edbbed2(part b) and d138ca4. The complete construct list with line
references is in plan §6.4.2 — ProtocolModel with extra="forbid", the ClientFrame and ServerFrame
discriminated unions, TypeAdapter validation on BOTH directions, ChatConnection's single-reader /
single-writer bounded queues (32 in, 128 out) with ClientDisconnected and StopWriter sentinels,
AssistantStep, TurnAccumulator.append_text returning a stable entry_id, _build_turn_messages emitting
provider-step order with monotonic microsecond offsets from base_time, staged user-message commit with
pending_messages committed on failure, TurnCompletedEvent moved to a post-commit callback and suppressed
when stopped or auto_continuing, transient_system_scaffold plus discard_transient_user_message,
tool_stream_frame(), set_cancel_reason / get_cancel_reason, and check_cancelled() at the five points in
tool_loop.py.

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
Its loader-gating half (gate auto-continuation on the loader reporting success:true) is blocked on O-7:
it rewrites the event.arguments.get("load") form that 97951be introduced, and mira-OSS still detects the
loader via mode in ["load","fallback","prepare_code_execution"]. Resolve O-7 — decide the invokeother_tool
argument shape and make the orchestrator and the tool agree — or defer the gating half and say so.
Also land e370468's orchestrator hunk (exclude invalid_reason calls from persisted_tool_ids), which
could not apply in WP1 because persisted_tool_ids did not exist yet. Yours does.

## Two mira-OSS tests assert the OLD protocol and must be replaced

tests/api/test_websocket_endpoint.py asserts type in {text, complete, pong} and a ping handler — the
pre-D2 vocabulary. It is superseded by tests/test_web_frontend_protocol.py.
tests/api/test_data_endpoint.py asserts offset/search pagination on ?type=history — exactly what D-2
removes (test_history_respects_offset_parameter, test_history_supports_search_query, and an
"offset" in pagination assertion).
Update or delete both, and say which in your report. Do not leave tests asserting a protocol you removed.

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
git worktree add .worktrees/wp5 -b 2.0/wp5 <integration-branch-after-WP4>
```

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
      _process_feedback_loop(). Do NOT take this file wholesale: its upstream 216-line delta bundles the
      Persona swap (take), batch removal and force_immediate deletion (D10 — check whether WP2 already
      did this), _cleanup_segment_files / Files-API deletion (D10), and a demo-user skip deletion that
      depends on prefs.conversation_llm (dead after D13). Hand-edit. Retain _process_feedback_loop,
      _init_feedback_loop and _invalidate_lora_trinket_cache.
      Note D-3: _process_persona does NOT swallow exceptions, unlike _process_feedback_loop. A persona
      failure propagates into the collapse-attempt counter toward MAX_COLLAPSE_ATTEMPTS = 3 tombstone.
      That asymmetry is intended; leave both behaviours as they are and mention it in the commit body.
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
  clients/AGENTS.md says "Callers pass exactly one of primary, fast, or batch" — five routes, and the
      fifth is `other` not `difficult`
  deploy/mira_service_schema.sql's COMMENT ON TABLE model_configs — WP-S should have fixed this; verify

O-24: rule on memory-curation-v2-serf-implementation-brief.md. crm's e26d031 deleted this 799-line root
document as a bundled, unrelated change; WP1-C declined the deletion. Decide whether 2.0 still wants it.

O-20: deploy/deploy.sh --migrate, deploy/migrate.sh, deploy/lib/migrate.sh and
deploy/schema_aware_restore.py implement a backup-then-restore upgrade path that the 2.0 posture makes
dead (WP-S deleted deploy/migrations/ entirely). Decide: delete them, or keep a documented export path
for users salvaging 1.x data. crm kept all three alongside its greenfield schema, which is an
inconsistency not worth inheriting. Check whether WP-S already reported dangling references.

## 3. Release identity (plan §0.1 obligations)

O-21: VERSION is 2026.06.25 (CalVer, identical in both repos). Set the 2.0 marker and decide whether the
scheme becomes semver (2.0.0) or stays CalVer with a major suffix.
Rewrite README.md for 2.0. It must state explicitly that **1.x -> 2.0 is a reinstall, not an upgrade**,
and that 1.x conversation history, memories and domain knowledge do not carry forward. That is the first
thing a 1.x user needs to know.
Update docs/MANUAL_INSTALL.md for the greenfield schema and the deleted migrations directory.
Retain mira-OSS's NEARFUTURE_FEATURES.md — crm deleted it; that deletion was not adopted.

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
