# LLM provider failover — design (2026-10-11)

Status: IMPLEMENTED 2026-10-11 (uncommitted at time of writing). Verified by a
Tier-2 live probe (/tmp/failover_probe.py, 6/6): config validators; diverter
trip/success-reset/flip-back counters; resolver divert against real DB-loaded
routes (route name + max_tokens cap preserved, clean restore); end-to-end real
call diverted to the live gateway (glm-5.3-flash, billed as 'primary'); recovery
probe job flipping back after 2 passes against 3 real primary targets; POST
reachability check passing through the shared probe module (3.4s). Probe-caught
defects fixed: staticmethod/self mismatch in resolver, missing `global` on the
diverter's recovery counter, wrong ModelSelection import in probe.py.

## What existed before

Closed-source MIRA (`taylorsatula/mira_closedsourceoriginal`, commit `b7388897`,
2025-10-04) had `EmergencyFallbackProvider` + class-level failover state +
5-minute recovery timer. Deleted by the continuum/unified-LLM migration
(`2adb5322`, 2025-10-24). Only a thin per-call `fallback_factory` survived in
that repo's `LLMLifecycle`. None of the old code is portable to v2; the design
below is native. OSS v2 history contains no failover code of any kind.

## Principal decisions (2026-10-11)

1. **One global backup target** specified in config.py, used for all five routes.
2. **Trip by live-call counter**: N consecutive `ProviderRetryableError`s
   (includes `ProviderStallError`) from the same primary endpoint trips the
   diverter immediately. No probe-driven tripping.
3. **Embeddings never divert.** The probe reports embedding health as advisory
   logging only; embedding calls keep failing against a dead primary during an
   outage (memory extraction pauses; chat works).
4. **Flip back after 2 consecutive full probe passes** (configurable).
5. **Process-local state** (module-global flag, like the old mechanism). No
   Valkey key. Restart mid-outage re-parks the boot gate on the dead primary
   for up to ~2 min per round until a probe squeaks under the bound, then the
   counter re-trips on the first live calls.

## Confirmed touchpoints (source-verified)

- **Text choke point**: `LLMProvider.generate_response`/`stream_events`
  (`clients/llm_provider.py`) is the single entry for every text call (~25
  call sites, census-verified: orchestrator tool loop `primary`; extraction,
  summaries, persona, user-model, portrait, lora, live compaction on
  `batch`; subcortical, tools, domaindoc on `fast`; assessment extractor;
  agent-selected routes; phoneafriend; injection-screen escalation;
  repulsion rewriter). Only out-of-funnel caller: the POST probe
  (`utils/power_on_self_test.py`), which builds selections from DB rows.
- **Route resolution**: `ModelResolver.resolve` → `get_model_config(name)` →
  `_model_config_cache` (startup-loaded, exactly five names enforced).
- **Errors**: `ProviderRetryableError` (5xx/timeout; `ProviderStallError`
  subclasses it) is the outage signal in both dialect families;
  `ProviderAuthError` and `ProviderContextOverflowError` are separate.
- **Pricing**: `cost_accumulator` keys off `metadata.model_config_name`.
- **Embeddings**: `get_embeddings_provider()` singleton from the one-row
  `embedding_config` (local in-process or remote HTTP). Vectors are not
  cross-model comparable; schema locks the row once vectors exist.
- **POST gate**: probes all five routes before binding; untripped-at-boot is
  guaranteed with process-local state, so the gate stays untouched.
- **Scheduler**: `scheduler_service.register_job(job_id, IntervalTrigger(...))`;
  POST's required-job set is a fixed literal (`power_on_self_test.py` ~1211).

## Design

### 1. Config block — `config/config.py` (ApiConfig)

```
failover_enabled                        bool = False
failover_endpoint_url                   str | None
failover_model                          str | None
failover_dialect_name                   str | None   # openai/anthropic/openrouter/groq
failover_api_key_name                   str | None   # Vault key; None = unauthenticated
failover_max_tokens_cap                 int | None   # None = keep each route's max_tokens
failover_recovery_probe_seconds         int = 120
failover_trip_consecutive_failures      int = 3
failover_flipback_consecutive_passes    int = 2
```

Model-level validator: fields complete when enabled; `failover_dialect_name`
coerces through the dialect registry's names; enabling while the install is
offline-local is allowed (a local backup endpoint is the old mechanism's
default shape).

### 2. Diverter — `clients/llm/diverter.py` (new)

Process-global, thread-safe (one `threading.Lock`):

- `record_success(endpoint)` / `record_failure(endpoint)`: consecutive-failure
  counter keyed by primary endpoint; success resets; N failures trips
  (TOAST log with endpoint + reason). Records for the failover endpoint and
  all records while tripped are ignored — backup failures never count and
  never recurse (one diversion level only).
- `tripped() -> bool` (False unless `failover_enabled`).
- `tripped_since`, `flip()` (clears state, TOAST log).
- No event-bus publication in v1: the bus is built during CNS graph init and
  the diverter is clients-layer; coupling is a layering inversion. TOAST
  journal lines are the operator surface. Publication via the taxonomy can be
  added later if a consumer needs it.

### 3. Diversion — `clients/llm/resolver.py:resolve`

After the normal selection, if `diverter.tripped()`: build a `ModelSelection`
from the config block with `model_config_name` = the **original** route name
(pricing continuity; nothing downstream learns a new route name) and
`max_tokens = min(route_max_tokens, failover_max_tokens_cap or ∞)`.
Capabilities validation then runs against the failover dialect naturally —
a backup that cannot serve tools fails those calls loudly, which is the
truthful behavior. `_model_config_cache` is never mutated.

### 4. Trip detection — `clients/llm_provider.py`

Both call paths already wrap the lifecycle call in try/except:

- `generate_response`: on `ProviderRetryableError` →
  `diverter.record_failure(selection.endpoint_url)`; on success →
  `record_success(selection.endpoint_url)` before returning. AuthError and
  ContextOverflowError pass through uncounted (misconfiguration and
  request-shape problems must not divert around themselves).
- `stream_events`: same hook inside the existing except around the event loop
  (stall surfaces mid-iteration); success recorded after the loop completes.
  The existing `ErrorEvent`-then-reraise behavior is untouched.

### 5. Recovery probe — job + shared probe function

- Factor the first-token reachability probe (landed in POST as `732eaca`)
  into `clients/llm/probe.py` (`probe_first_token(selection, timeout) -> None`
  or raises); `utils/power_on_self_test.py` and the new job both call it —
  one sanctioned probe mechanism.
- New job (registered in `initialize_all_scheduled_tasks` flow **only when
  `failover_enabled`**, so the POST required-job set stays untouched):
  `IntervalTrigger(seconds=failover_recovery_probe_seconds)`.
- While tripped: probe every deduped primary target (dedup key
  (dialect, endpoint, api_key_name, model), same as POST's
  `_llm_probe_targets`); 2 consecutive full passes → `diverter.flip()`.
  While healthy: no-op return (no billed probes when nothing is wrong).
- Advisory: when tripped and the primary embedding provider is remote, also
  embed one `_PROBE_TEXT` through the primary and TOAST its health
  (dimensionality is inherently correct; this is visibility only).

### 6. Boot/restart interplay

Unchanged POST gate. Restart mid-outage sequence: gate probes dead primary →
parks (rounds of ~2 min) → primary answers under the bound eventually →
binds → first live calls fail N times → diverter trips within seconds. The
park window is the accepted cost of process-local state (decision 5).

### 7. Characterization / verification shape (NO MOCKS)

- Diverter unit path: real module, no infrastructure — record N failures →
  tripped; success resets; backup-endpoint records ignored while tripped.
- Resolver divert path: seed `_model_config_cache` directly with test
  `ModelConfig` values (in-memory fixture of a cache the code itself
  maintains — not an infrastructure mock), trip the diverter, assert the
  resolved selection is the failover target carrying the original route name.
- Live Tier-2 probe: scratch or local fixture endpoint for the backup; point
  `failover_endpoint_url` at a local llama-server/httpx echo; assert
  end-to-end diversion and flip-back through the real lifecycle.
- Trip-path probe: point `failover_endpoint_url` at a live target, drive
  failures against an unreachable primary endpoint, observe trip + diversion.

### 8. Map updates (same commit)

- `clients/llm/AGENTS.md`: replace both no-fallback doctrine sentences with
  the diverter contract (tripped → resolver diverts; counter trip semantics;
  one level, no recursion); add `diverter.py`/`probe.py` to Files.
- `config/AGENTS.md`: the failover block.
- `clients/AGENTS.md`: provider/embeddings row citations.
- `utils/AGENTS.md`: POST probe now delegates to `clients/llm/probe.py`;
  note the conditional probe job.

## Deliberately excluded

- Per-route backup targets (principal chose global; revisit only if a
  capability mismatch bites in practice — the loud capabilities failure makes
  any mismatch visible immediately).
- Embedding diversion (dimension-mismatch hazard; principal chose probe-only).
- Valkey-persisted trip state (principal chose process-local).
- Event-bus publication of trip/restore (layering inversion; TOAST suffices).
- `model_configs` failover rows (CHECK constraint would need a schema change;
  config.py needs none).
- Changes to the POST gate (unreachable with process-local state).

## Files expected to change

1. `config/config.py` — failover block + validators
2. `clients/llm/diverter.py` — new
3. `clients/llm/probe.py` — new (factored from POST)
4. `clients/llm/resolver.py` — divert branch
5. `clients/llm_provider.py` — trip detection hooks
6. `utils/power_on_self_test.py` — delegate to `clients/llm/probe.py`
7. `utils/scheduled_tasks.py` or a new `utils/failover_jobs.py` — probe job
8. Maps: `clients/llm/AGENTS.md`, `config/AGENTS.md`, `clients/AGENTS.md`, `utils/AGENTS.md`
