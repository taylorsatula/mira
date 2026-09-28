# MIRA - Python Project Guide

**Complex problems require simple and clear solutions.**

MIRA is a FastAPI application with event-driven architecture coordinating three core systems: CNS (conversation management via immutable Continuum aggregate), Working Memory (trinket-based system prompt composition), and LT_Memory (memory extraction/linking/refinement). Subsystems coordinate by event-bus publication, not direct calls — the whole system hangs off a small set of events owned by `cns/core/events.py`. MIRA also models its own relationship with the user as a designed surface: a user model that describes the user, a Persona that prescribes to MIRA, a portrait injected into the system prompt, and metacognitive observers. PostgreSQL RLS with contextvars provides automatic user isolation - all user-scoped queries, tool access, and repository operations enforce `user_id` filtering at the database level.

The User's name is Taylor.

## 🗺️ Nested AGENTS.md Maps — Shape & Maintenance

Every directory with ≥2 source files, or any invariant not documented in its
parent directory's map, has an `AGENTS.md` orientation map. Smaller
directories are documented in the parent's `## Files` section. Full template
and authoring rules: `docs/AGENTS_MAP_SPEC.md`.

**Map registry** — every directory with a map, and what it owns. A directory not
listed here that meets the coverage gate needs a map created; a listed
directory whose map is missing is a defect.

| Directory | Map owns |
|---|---|
| `agents/` | Sidebar-agent runtime: base-class loop mechanics, dispatcher, spawn paths |
| `agents/implementations/` | Concrete agents (forage, while-the-cats-away, memory curator) and the Mode Contract |
| `agents/triggers/` | Dispatcher work-item discovery triggers |
| `auth/` | Passwordless identity, sessions, CSRF, API tokens, WebAuthn, account lifecycle |
| `clients/` | External infrastructure clients: Vault, Postgres, Valkey, SQLite, embeddings, Lattice |
| `clients/llm/` | Provider-neutral LLM boundary: typed contracts, route resolution, lifecycle policy |
| `clients/llm/dialects/` | Per-provider wire-format dialects, error taxonomy, thinking translation |
| `cns/` | Conversation orchestration: layering contract over the five subsystems below |
| `cns/api/` | FastAPI routing layer, server side of the WebSocket turn protocol |
| `cns/core/` | Immutable domain model: Continuum aggregate, messages, domain events |
| `cns/infrastructure/` | Persistence and caching: repositories, UnitOfWork, segment sentinel lifecycle |
| `cns/integration/` | Event bus and CNS dependency-graph construction |
| `cns/services/` | Turn orchestration, segment collapse, memory surfacing, user-model and Persona pipelines |
| `config/` | Pydantic config schema, the config singleton, the system prompt |
| `config/prompts/` | LLM prompt templates and the loader contract |
| `deploy/` | Host-metal and Docker deployment tooling |
| `deploy/docker/scripts/` | Container bootstrap and supervision scripts |
| `deploy/lib/` | Shared bash helper libraries, Vault state machine |
| `docs/` | Operator documentation, plus two cited authorities: `AGENTS_MAP_SPEC.md` (map shape) and `REGISTER.md` (writing register for any text a model reads) |
| `lt_memory/` | Long-term memory: storage, scoring, retrieval, linking, entity services |
| `lt_memory/processing/` | Extraction pipeline and consolidation |
| `scripts/` | Operational CLI entry points run against a deployed service |
| `tests/` | Probe-artifact homes: disposable exhibit, shared real-infra fixtures, admission-gated batteries — not a suite |
| `tests/fixtures/` | Reusable probe scaffolding: real-infrastructure setup/teardown, claim-free, no simulation |
| `tests/protected/` | Admission-gated permanent verification batteries; the exact-phrase authorization rule and the no-mock, no-drift, cannot-execute requirements |
| `tests/tmp/` | Display exhibit for the disposable-probe pattern: autodeleted-on-sight test policy, one exemplary probe |
| `tools/` | Tool framework: base class, repository, config registry |
| `tools/implementations/` | All concrete LLM-callable tools |
| `tui/` | Minimal terminal chat client: sync-REST REPL brick, retained (unused) WS stack, display-filter mirror of the web client, endpoint store, headless token bootstrap, config-file settings |
| `utils/` | Cross-cutting infrastructure: identity, scheduling, storage, security, observability |
| `working_memory/` | Event-driven system-prompt composition via trinkets |
| `working_memory/trinkets/` | Trinket implementations: one `variable_name` slot each |

**Ancestor maps:** an ancestor map of `<dir>/` is the `AGENTS.md` of a parent
directory, up to repo root (for `cns/api/`: `cns/AGENTS.md` and this root
file). The harness loads the full ancestor chain whenever a file in the
directory is read, so those maps are present in context together with this
one.

**Shape, authoring rules, and audits:** the full map spec — fixed section order,
the four Rule requirements, cross-map reciprocity (each fact has exactly one
owning map; others cite it in one line), `## Files` and anchor conventions, the
complete maintenance trigger table, and the two run-before-committing audits
(path-anchor and reciprocity) — lives in `docs/AGENTS_MAP_SPEC.md`.

The high-frequency obligations, binding in every session: map updates are part
of the same commit as the triggering change — a new/deleted/renamed file updates
its directory's `## Files` bullet; a behavior or contract change greps its map
for the old statement; a wiring edge is stated fully by the owner map and cited
one-line by every counterpart; a directory reaching the coverage gate gets a
map. Every bullet must change agent behavior — delete on sight what doesn't
(the requirement-5 gate and line budgets live in `docs/AGENTS_MAP_SPEC.md`).
Run the audits in `docs/AGENTS_MAP_SPEC.md` before committing any change
that touched source files or maps. When in doubt, update the map — an
unnecessary edit costs a line; a missed one misleads every session.


## Repo root files

- `main.py` — application entry point and wiring hub: router mounts, the
  middleware stack, the global `APIError` → HTTP-status mapping, lifespan
  startup/shutdown ordering (LT_Memory factory → CNS graph → sidebar/heartbeat
  jobs → announcement; `websocket_chat.close_all_connections()` awaited at
  shutdown; Valkey flush preserves `heartbeat:` and `pending_memories:`/`pending_memories_done:`/`pending_memories_attempts:` prefixes — the pending-memory queue is the durable record of user-confirmed memories and must survive restarts), and the pre-server
  POST gate before bind. The per-directory contracts are owned by
  `cns/api/AGENTS.md`, `auth/AGENTS.md`, `cns/integration/AGENTS.md`,
  `agents/AGENTS.md`, `config/AGENTS.md`, `lt_memory/AGENTS.md`,
  `utils/AGENTS.md`.
- `requirements.txt` — dependency pins; the optional block's visibility to
  `Dockerfile.base` is owned by `deploy/AGENTS.md`.
- `VERSION` — release identity string, read by `cns/api/update.py:get_latest_version`
  and reported by `/health` (`cns/api/AGENTS.md`).
- `README.md`, `license.txt`, `NEARFUTURE_FEATURES.md` — static repo documents;
  no runtime consumers.
- `UPGRADE_PATH.md` — unimplemented design guide; its design-only status, the
  no-upgrade-path doctrine, and the dead `--migrate` references are owned by
  `deploy/AGENTS.md` and `docs/AGENTS.md`.

## 🚨 Critical Principles (Non-Negotiable)

# 🛑 NO MOCKS. NO PARALLEL SUITES. VERIFICATION IS LIVE OR IT DOESN'T COUNT.

Prohibited: test files, pytest fixtures, mock objects, stubs, fakes, offline test databases, and any check that runs against simulated infrastructure. A passing mock-based check provides no information about this system.

Every line in this tree is agent-written; no human has traced any path. Type-clean, pyflakes-clean code that has never executed is the standard failure product of this workflow and has shipped here before. Verification is therefore mechanical and live, never assumed. A test is what a probe becomes when its answer must never change again — not a starting artifact.

Required verification, in the POST tradition:
- **Boot gate** — probes Vault, Postgres, and model routes against live infrastructure before the server binds.
- **Path-probes** — invoke critical runtime paths against the live system exactly as production would.

**Probe-surface membership (standing rule):** any path whose failure would report incorrect data to users, lose data, or degrade silently — reads, writes, searches, auth flows, failure paths. If a required probe cannot run against live infrastructure, fix the code until it can; do not simulate.

Quality guarantee: boot-survival plus path-probe coverage. A clean boot verifies the wiring; a passed path-probe verifies the handler. Code covered by neither is unverified — flag it in review. A bug found in unprobed code is fixed together with the probe that covers it. Path-probes are production code: normal review discipline, the same credentials plumbing, production-identical failure behavior.

### Load the `writing-probes` skill before you probe

This section owns the doctrine; the skill owns the craft. Load `writing-probes`, every time, before:
- writing, running, or reviewing a probe;
- reproducing a defect live, or answering "does this actually work?";
- deciding whether a check earns permanence — `CheckSpec` path-probe, `tests/protected/`, or discard;
- diagnosing a probe that failed — code bug or probe bug?

It carries the anatomy and the static scaffold, the shape catalog, the fake rule, the EXECUTED/UNVERIFIED contract, the probe-bug taxonomy, and how to shape code so it is probe-able.

### ⚡ Realtime verification loop (proportionate by behavioral surface)

Models are pretrained to verify by writing tests. When that reflex fires during a change, write a path-probe instead — same verification goal, production-code artifact.

Select the tier by behavioral surface touched, not by change size:

- **Tier 0 — no behavioral surface.** Comments, docstrings, formatting, import regrouping, verified-mechanical renames, documentation. `py_compile` + pyflakes on touched files. No further verification required.
- **Tier 1 — behavioral edits within surface already covered by boot or probes.** Execute the changed path once against live infrastructure; re-read the diff against the invariants (every failure path reports the failure truthfully, nothing silently degraded, no assertion of behavior not executed); end the change report with **EXECUTED** (what ran) or **UNVERIFIED** (why not, and which probe would cover it).
- **Tier 2 — new behavioral surface or elevated stakes.** New feature, endpoint, or path; schema or data-migration change; failure-behavior, security, or auth change; new dependency. Tier 1 plus: persistence round-trip against dev infrastructure (write, read back, verify, clean up — RLS and constraints included); register the path-probe where membership hits; a second agent re-derives the diff (skip it only when one-shot execution covers the change, and say so).

The verification homes are indexed in the map registry above; their admission, cleanup, and realism policies are owned by `tests/tmp/AGENTS.md`, `tests/fixtures/AGENTS.md`, and `tests/protected/AGENTS.md`, and the skill's promote-or-discard moment owns the choice among them.

### Technical Integrity
- **No Shortcuts**: never substitute a cheaper check for the required one — no mock-based verification, no skipped probe, no "improvements" during extraction, no deferred path-probe. Shortcuts ship as silent breakage.
- **Verify Contracts Before Building On Them**: verify an unfamiliar helper's or internal API's contract at the boundary you depend on — inputs, outputs, types, side effects, failure modes — with the smallest direct probe or existing reference usage before building on it. Most preventable slipups come from trusting names or remembered APIs.
- **Evidence-Based Position Integrity**: form assessments from evidence and hold them under pushback. Do not adjust conclusions to match the human's apparent preference; when their proposal contradicts the assessment, push back and say why.
- **Blunt Technical Communication**: reject technically unsound ideas directly — "bad", "infeasible" — and correct wrong assumptions about code or constraints immediately ("That's wrong"). After rejection or correction, provide the working alternative or the accurate facts.
- **Plain Speech by Audience**: explanations written for Taylor lead in plain English — what the thing does, in ordinary words, symbols and metric names attached only if the subject is the code itself or he asks for them. Precision rules still govern artifacts (code, docs, commit messages) and any statement about specific files. Dense terminology in a human-facing explanation is a communication failure even when every word is accurate.
- **Concrete Code Communication**: name exact methods, files, and snippets — "the `extract_topic_changed_tag()` method that calls `tag_parser.extract_topic_changed()`", not "the tag processing logic". No vague referents.
- **Numeric Precision**: no invented numbers. Qualitative language unless the figure comes from measurement, benchmark, requirement, or calculation.
- **No Tech-Bro Evangelism**: describe work accurately — a feature is a feature, a fix is a fix. No "revolutionary"/"fundamental shift" framing or buzzwords.

### Security & Reliability
- **Credential Management**: all sensitive values stored in Vault via `utils.vault_client` functions; per-user credentials via `UserCredentialService` (`utils.user_credentials`). Never env vars or hardcoded values; missing credentials fail with a clear error, never a fallback.
- **Fail-Fast Infrastructure**: required-infrastructure failures (Valkey, database, embeddings, event bus) MUST propagate. Never catch and return None/[]/defaults — that masks outages as normal operation. try/except only for: (1) adding context before re-raising, (2) legitimately optional features (telemetry, cache), (3) async handlers that will retry. A database query returning [] means "no data found", not "query failed".
- **No Optional[X] Hedging**: a function depending on required infrastructure returns the real type or raises. `Optional[str]` for a subcortical result lets generation silently fail; `str` forces the caller to handle the exception. Optional is for genuine "value may not exist" semantics (user preference unset), never "infrastructure might be broken".
- **Timezone Consistency**: use `utils/timezone_utils.py` functions (`utc_now()`, `format_utc_iso()`) for all datetime operations — never `datetime.now()` directly.
- **Backwards Compatibility**: don't depreciate; ablate. Breaking changes preferred — notify the human first. Back-compat is not retained unless directed; MIRA is a greenfield system design.
- **Know Thy Self**: models tend to invent new endpoints or change existing patterns instead of looking at what is there. Always survey the existing code before assuming.

### Core Engineering Practices
- **Thoughtful Component Design**: hide complexity internally, expose simple APIs — automatic user scoping, DI for cross-cutting concerns, middleware for infrastructure. Ask how the design eliminates repetitive work and prevents common mistakes.
- **Integrate Rather Than Invent**: use the platform mechanism (DI, validation, async); deviate only with documented justification.
- **Root Cause Diagnosis**: examine related files and dependencies before changing code; fix problems at their source — never adapt downstream to compensate for an upstream bug.
- **Simple Solutions First**: prefer the small fix, never at the cost of correctness. Implement exactly what is requested; unrequested "safety" features create problems.
- **Handle Pushback Constructively**: "Is this the best solution?" / "Are you sure?" usually means the human thinks it isn't — re-derive the reasoning that led there instead of defending it.
- **Convergent Path Refactoring**: multiple code paths doing the same thing with divergent implementations is an architectural defect. Map the paths, find the real convergence point, remediate there — no refactoring theater.

### Design Discipline Principles
- **Make Strong Choices**: one format/approach unless concrete use cases require alternatives. No "just in case" features, no "if available" fallbacks, no `Any` where the structure is known.
- **Fail-Fast, Fail-Loud**: don't return `[]`/`{}` when parsing fails — it masks errors as "no data found". `warning`/`error` for problems, not `debug`. Validate inputs at entry; raise `ValueError` with diagnostics, not generic `Exception`.
- **Types as Documentation**: avoid `Optional[X]` except genuine domain optionality; `TypedDict` over `Dict[str, Any]`; type what the code expects (`UUID`, not `str`). Replace positional tuples with named structures — `result.query_expansion`, not `result[0]`.
- **Naming Discipline**: `ContinuumRepository` → `continuum_repo`, not `conversation_repo`. One term per concept; method names match action — `get_user()` gets, `validate_user()` validates.
- **Forward-Looking Documentation**: write what code does, not what it replaced; history goes in commit messages.
- **Standardization Over Premature Flexibility**: no flexibility without a concrete second use case — wait for the pattern to emerge from real code.
- **Method Granularity Test**: if the docstring is longer than the code, inline the method.
- **Hardcode Known Constraints**: don't parameterize what won't vary; constants with a comment explaining why.

### Preventive Mechanism Rules

These convert recurring failure modes into mandatory patterns. The banned forms are greppable — check for them during review. Each rule is an instruction, not advice: follow the pattern as written.

- **No Per-Request State on Singletons**: values computed for one request, turn, or user never live on instance attributes of process-global singletons — they flow through parameters or a per-call context object. An event handler that must return a result is the wrong tool: call a function and use its return value; the event bus is notification, not call-and-return. Instance attributes are for construction-time wiring and explicitly-keyed process-wide caches only — anything mutated per turn and read back after a publish is state smuggling through the bus.
- **Bounded Waits at Every Boundary**: any call crossing the asyncio↔thread or queue boundary carries a bound — `future.result(timeout=…)`, a queue put with an overflow policy (timeout, drop, or fail — never a bare unbounded `Queue.put`), `asyncio.wait_for` around every await that can stall. A possibly-unbounded blocking call placed between cancellation checks makes cancellation unreachable, which is a defect by itself.
- **Guard Clauses Live in the Mutation**: for check-then-act on shared state, the precondition belongs in the write itself — `UPDATE/DELETE … WHERE <guard>`, compare-and-set, `GETDEL` — never in a preceding read. Destructive and status-changing writes re-verify their precondition inside the statement; a scan whose transaction commits before its action loop protects nothing.
- **One Sanctioned Path per Hazard**: every recurring hazard has one mandatory mechanism — use it, never hand-roll a parallel one; raw forms behind a sanctioned wrapper are defects. The sanctioned mechanisms are enumerated in Codebase Patterns below (`load_prompt()`, timezone utils, the contextvar/RLS flow, Vault/`UserCredentialService`, event-bus coordination).
- **External Content Crosses One Boundary**: anything sourced outside the system (fetched pages, email bodies, third-party API responses, user-supplied file content) passes through injection screening and untrusted-content wrapping before entering ANY model context — tool results, trinket content, agent work items, system prompts. Text interpolated into XML-like prompt structures is escaped at the interpolation point. A tool that returns external text "for the model to read" unwrapped violates this rule regardless of what any doc claims.
- **Configure Structured Data — Never String-Patch It**: installers and provisioning set values via variables and parameterized statements. Never `sed`/grep-patch structured data (SQL, JSON, YAML) by matching literal strings from a previous revision — a non-matching literal is a silent no-op; a matching one is a fuse for the next edit. Where a literal must exist, generate it from the source that defines it. Installation is code: it gets a smoke run before release, same as any handler.
- **Generated, Not Transcribed, Format Examples**: when a prompt instructs a model what to emit, the examples and format tokens are produced by the same code that parses the format (shared constants/formatters). Hand-written format examples drift from the parser — the model follows the example, the code follows the spec, and the mismatch is silent.
- **Extract by Identity, Not Position or Shape**: data recovered from a probabilistic source (LLM output) or a lossy store is keyed by an identifier emitted alongside it — never "the last one", never "the first match". Persist explicit type tags; never infer structure from content shape (prefix/suffix sniffing). Storage round-trips must be lossless both ways.
- **Derived State Self-Heals or Computes On Read**: counters and denormalizations incrementally maintained across multiple write paths will drift — one path forgets, another double-counts. Either recompute on read, or schedule a recompute job so drift decays on its own. "Every write path updates every derivation" is a checklist, not a mechanism.
- **Degrade Only to Acceptable**: exception handling is positional first — critical request-path code propagates; fire-and-forget consumers log and swallow; background durability paths (collapse, extraction) tolerate-and-log because the model is off the call stack. Know which side you are on before writing a try/except. Then: a catch-and-continue site is legal only when the degraded result is still correct for its consumer. Every such site's log message states the user-visible consequence; if the degraded output would silently corrupt the deliverable (missing required sections, wrong shape, empty-required data), propagate instead. A log line is not a user interface.
- **Retry Counters Never Gate Data on Infrastructure Failure**: attempt counters that trigger destructive or data-losing fallbacks (tombstones, abandons) count only failures the data caused; infrastructure/LLM outages are excluded. Three provider blips must not consume a data-bearing budget.
- **Every Claimed Defense Has a Witness**: any sentence in this tree claiming something is "wrapped", "enforced", "atomic", or "verified" corresponds to a live path-probe. If writing the probe is impractical, delete the claim. Prose that promises an unwitnessed mechanism is where bugs hide longest.

## 🏗️ Architecture & Design

### User Context Management
- **Administrative tasks outside HTTP context** (scheduled jobs, batch operations, cross-user commands): explicitly `set_current_user_id(user_id)` per user, or `AdminSession` to bypass RLS entirely when querying across all users. Request-scoped context flow is covered under Cross-Cutting Patterns.

### Tool Architecture
Use `tools/AGENTS.md` and `tools/implementations/AGENTS.md` as entry points; `tools/HOW_TO_BUILD_A_TOOL.md` is the step-by-step walkthrough (symbol anchors, four-piece registration, `run()` and return-envelope contracts, verification tiers). The sibling walkthroughs are `agents/HOW_TO_BUILD_AN_AGENT.md` and `working_memory/trinkets/HOW_TO_BUILD_A_TRINKET.md` — all three are kept aligned with the code they describe, so a contract change in a base class updates its guide in the same commit. Design for single responsibility (extraction tools extract, persistence tools store). Business logic lives in system prompts/working memory, not tools. Tool data goes in user-specific storage via `self.user_data_path` / `self.db`. Include recovery guidance in error responses. Verification is live probes (see NO MOCKS); a tool touching a critical path ships with its path-probe.

### LLM Caller Interface Design
All model-facing prose — system prompts, tool parameter descriptions, agent directives, trinket content — is an interface contract where imprecise language causes behavioral failures downstream. Every word constrains behavior: "literal string" not "text", "exact substring" not "pattern"; the reader is a language model that infers defaults from word choices. Ground descriptions in implementation behavior, not intent; drop jargon the caller lacks context for; state co-dependencies inline; if current wording would cause misuse, fix it.

### Interface Design
When calling code misuses an interface, fix the caller — never adapt the interface to accommodate misuse.

### Dependency Management
- **Minimal Dependencies**: prefer stdlib. New external deps require documented justification; cross-reference Python imports, deploy scripts, Dockerfiles, and optional feature paths before adding or removing. Remove only when nothing imports, invokes, or operationally installs them.

### Investigation & Mechanical Refactoring
- **Investigations Need Evidence**: answer with specific files, functions, and observed behavior; separate verified facts from inferences; when inconclusive, say what was checked and why it does not prove the point.
- **Mechanical Renames Stay Mechanical**: map old names to new, update definitions and references, verify no old symbol remains. No behavior changes mixed in unless the caller asked for both.

## 🧭 Codebase Patterns

### User ID Resolution
All user-scoped code resolves `user_id` via contextvar from `utils/user_context.py` (module contract owned by `utils/AGENTS.md`). Set once at the API boundary; flows automatically through the request. Never pass `user_id` through event dicts, parameters, or instance fields instead of the contextvar. Explicit `user_id` parameters are acceptable when sourced from the contextvar (e.g. `ManifestQueryService.get_segments(user_id)`). Outside HTTP context, call `set_current_user_id(user_id)` explicitly.

### Cross-Cutting Patterns
Patterns that apply in every directory. Directory maps may restate these with local specifics.

- **Thread spawns copy user context**: any thread/executor spawn uses `contextvars.copy_context().run(fn)`, or RLS loses `app.current_user_id`. Canonical pattern: `cns/services/tool_loop.py`.
- **RLS fails closed**: a query without user context returns zero rows, no error — "empty" is ambiguous. Only admin sessions (`BYPASSRLS`) may omit user context. (`clients/postgres_client.py` canary, `auth/database.py`)
- **Model-supplied timestamps are the user's local wall time**, never UTC: parse with `parse_time_string(value, tz_name=...)` + `ensure_utc`; exact wall times are DST-strict via `normalize_exact_local_wall_time()`. (`utils/timezone_utils.py`)
- **LLM calls route by name**: `model_config='<route>'` on the five fixed routes; never hardcode models, endpoints, or API keys. (`clients/llm/`)
- **Credentials**: system-level from Vault only; per-user via `UserCredentialService`. Missing credentials raise with setup guidance — no env-var or default fallbacks. (`clients/vault_client.py`, `utils/user_credentials.py`)
- **User SQLite encryption**: the `encrypted__` column prefix drives transparent Fernet encryption; declare the columns in DDL and never double-decrypt. (`utils/userdata_manager.py`)
- **Preview-before-save** for user-instructed revisions: candidate held in Valkey under an opaque `preview_id` with TTL, consumed delete-after-read — the client never round-trips stored text. (`cns/services/persona_service.py` et al.)
- **Memory short IDs** (`mem_XXXXXXXX`) are irreversible prefixes of full UUIDs: short form for LLM-facing surfaces, full form for persistence and stamps. (`utils/tag_parser.py`)
- **Use-day intervals** (`*_use_days`) are modular activity-day gates, not calendar cadences. (`utils/scheduled_tasks.py`)
- **Per-user tool config**: `config.<tool>_tool` merges the user's override fresh on every access over the global default; secret fields round-trip via the redaction sentinel. (`config/config_manager.py`, `utils/tool_config_store.py`)
- **Prompt templates load via `load_prompt()`** — never `open()` a prompt file directly. (`config/prompts/loader.py`)
- **Event handlers are synchronous**, registered by event class `__name__`; async work inside a handler spawns a thread with copied context. (`cns/integration/event_bus.py`)
- **The LLM is an untrusted component**: escape, allowlist, and validate at every LLM boundary — tool arguments JSON-Schema-validated, credentials injected server-side (the model names a credential it never sees), thinking signatures round-tripped untampered. The external-content ingestion boundary itself is owned by Preventive Mechanism Rules above. (`clients/llm/`, `utils/untrusted_content.py`)
- **Inter-component coordination goes through the event bus**, not direct service calls: event taxonomy owned by `cns/core/events.py`, bus owned by `cns/integration/event_bus.py`. New features subscribe and publish.
- **Trinket state is per-user**, keyed by the contextvar — trinket instances are process-global singletons shared across users; never store user state on instance attributes. (`working_memory/trinkets/base.py`)

### Activity Days & Use-Day Scheduling
Jobs fire on user activity days, not calendar time (log in Monday, skip Tuesday, return Wednesday → the counter ticks Monday and Wednesday only). Mechanics are owned by `utils/AGENTS.md`; `*_use_days` interval semantics by `config/AGENTS.md` — read both before adding or changing a use-day-gated job.

### Provider Stall Detection
All live LLM transports run through `clients.llm.lifecycle.LLMLifecycle`; there is no fallback route — stalls (`ProviderStallError`) and provider failures propagate. Full policy owned by `clients/llm/AGENTS.md` and `utils/AGENTS.md`; dialects and the add-a-dialect blast radius by `clients/llm/dialects/AGENTS.md`.

### Power-On Self-Test
POST checks live in `utils/power_on_self_test.py`. The pre-server gate runs in a subprocess before Hypercorn binds, is bounded (attempts then parks rather than exits — a restart-on-exit supervisor would loop real, billed LLM probes; `MIRA_POST_GATE_FAILURE_ACTION=exit` opts out), and the post-server probe/CLI are owned by `utils/AGENTS.md` and `scripts/AGENTS.md`. POST checks exercise real infrastructure, never mocks.

## ⚡ Performance & Tool Usage
- **Synchronous Over Async**: prefer synchronous unless there is genuine I/O concurrency. Async overhead hurts without actual concurrency; sync is easier to debug and reason about.
- **Model Dispatch**: route selection is `model_configs`-route-based (`primary`/`fast`/`batch`/`assessment`/`other`) — owned by `clients/llm/AGENTS.md`. Match the route to the task: high-frequency mechanical judgments on `fast`; tasks requiring semantic understanding stay on `primary`. Never route by vendor model name.

## 📝 Implementation Guidelines

### Implementation Approach
When modifying files, write as if the new code was always the plan. Never reference removals. Understand surrounding architecture first.

### Plan Mode
🚨 **Never enter plan mode autonomously** — wait for explicit user activation (`/plan`); autonomous entry is disruptive UX.

Ordinary implementation plans stay concise. ADRs only for durable architecture decisions needing rationale, alternatives, and consequences recorded.

## 🔄 Continuous Improvement
- Convert specific feedback into general principles. Consider multiple approaches before implementing.
- Fix issues at the root — there is no test suite to fall back on, so correctness comes from fail-fast behavior and direct verification.

## 📚 Reference Material

### Commands
- **Database**: On a deployed instance, `psql -U postgres -h localhost -d mira_service` (postgres superuser, `mira_service` the primary DB). A development checkout may have no `mira_service` database: for a schema-backed scratch DB call `tests/fixtures/scratch_db.py:scratch_database(user=<local superuser>, host="localhost")`, which applies the shipped schema and drops the DB on exit.
- **Local dev infrastructure**: gate a probe with `tests/fixtures/live_infra.py:require` — it reports BLOCKED when Postgres (5432), Valkey (6379), or Vault (8200) is not listening. The app's Vault client authenticates only via AppRole (`VAULT_ADDR` + `VAULT_ROLE_ID`/`VAULT_SECRET_ID`); a `~/.vault-token` authenticates the `vault` CLI, not the app. Without AppRole credentials `clients.vault_client` raises, so Vault-backed paths report UNVERIFIED on such a checkout.

### Git Workflow
- Commit only when the user explicitly asks.
- Review `git status --short` before staging.
- Stage explicit paths. Do not use `git add -A` or `git add .` unless the user explicitly asks to stage everything.
- Use a semantic prefix (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`) and a concise subject.
- For non-trivial commits, include body sections for `ROOT CAUSE` and `SOLUTION RATIONALE`.
- After committing, report the commit hash and the high-level file/change summary.

### Pydantic Standards
Pydantic BaseModel for structured data (configs, API models, DTOs): `from pydantic import BaseModel, Field`; `Field()` with descriptions and defaults; complete annotations; docstrings stating purpose. Naming: `*Config` for configs, `*Request`/`*Response` for API models.



---

# Critical Anti-Patterns to Avoid

Recurring mistakes kept as incident records — the examples are historical, the lessons are current.

| Pattern | Example | Lesson |
|---|---|---|
| Git workflow violations | HEREDOC commit messages; `git add -A` without permission; missing ROOT CAUSE / SOLUTION RATIONALE; no post-commit summary | Follow the Git Workflow section before every commit |
| Over-engineering | Severity levels when binary worked/failed suffices | If you can't explain why it's needed, it probably isn't |
| Credential fallbacks | Hardcoded API keys; fallback values for missing credentials | Vault + `UserCredentialService`; fail fast when credentials are missing |
| Cross-user data access | Manual `user_id` filtering in individual queries | Tools get user-scoped access via `self.db`; isolation is architectural |
| "Improving" during extraction | Removing `_previously_enabled_tools` state storage because it "seemed unnecessary" | Extract working code exactly as-is; improve later |
| Premature abstraction | Wrapper classes for single-use utilities; config objects for nonexistent scenarios | Straightforward first; abstractions emerge from repeated real patterns |
| Infrastructure hedging | `try: db.query() except: return []` | Fail-Fast Infrastructure above; silent degradation is diagnostic hell |
| UUID mismatches at boundaries | `TypeError: Object of type UUID is not JSON serializable` | Native types internally; convert only at serialization boundaries; early conversion breaks comparisons |
| Incomplete path replacement | Replacing `_generate_non_streaming()` but missing the buried `_write_firehose()` call | Trace ALL side effects — logging, metrics, state, events; verify by booting |

## 🖥️ Dev-instance deployment — libvirt host (192.168.1.9)

**The toolkit ships IN THIS REPO at `deploy/vm/`** — read `deploy/vm/README.md`
first. Three modes, all exercised 2026-09-16/17 end-to-end: local libvirt (on the
host), `--host admin@192.168.1.9` (orchestrated from a workstation, no local libvirt
needed), and `--ip 192.168.65.2 --vm-user mira_service --vm-pass …` (plain ssh
onto an existing VM — validated on an aarch64 workstation VM: dev build deployed in 182 s, v3 sarcophagus
restored, 9/9 facts PASS, chat-continuity confirmed from restored memories).
`extract.sh` snapshots a live instance into a sealed sarcophagus the same way —
validated against that aarch64 VM too (cross-arch, cross-user remap: units `User=`,
credentials, and ownership follow `--vm-user`).

**Reference host recipe** (run on the host, or from a workstation via `--host`):

```bash
ssh admin@192.168.1.9 \
  /home/admin/mira_instance_snapshots/bin/oneshot.sh \
  /home/admin/mira_instance_snapshots/mlfactory_v4_mira   # or another sarcophagus
```

That spawns a fresh VM from the host's default-state frozen base template, deploys a
dev build from `/home/admin/mira-OSS-worktree` (a snapshot of this worktree), injects
the sarcophagus (Postgres, user data incl. domaindocs, Vault with real keys, units),
and verifies health + row counts against the sarcophagus's SNAPSHOT-FACTS.txt.
`--fresh` rebuilds a running VM (old disk preserved); omit it to reuse a running one.
Deploy-only on a fresh VM (no instance state): `oneshot.sh` mandates a sarcophagus by
design, so run its phases 1–4 without inject — the validated driver is
`bin/deploy-only.sh` on the reference host (2026-09-18: template spawn + dev deploy,
healthy in 152 s). Give the config real `chat_api_key`/`subcortical_api_key` and the
deploy seeds them into Vault (`deploy/postgresql.sh` Step 14), so the instance comes
up healthy with live model routes and no inject. To redeploy onto an already-running
VM, run `deploy/deploy.sh --config <yml> --local` by hand inside it.

**Refresh the host's source snapshot after changing this worktree** (the host deploys
from its copy, not from here):

```bash
cd ~/Programming/GitHub/mira-OSS && tar --exclude=.git --exclude=data --exclude=logs \
  --exclude=scratch --exclude=__pycache__ --exclude='*.pyc' --exclude=.env \
  --exclude=venv --exclude=.claude -czf - . | \
  ssh admin@192.168.1.9 'tar -C /home/admin/mira-OSS-worktree -xzf -'
```

**`deploy/deploy.sh --local` (added 2026-09-16):** installs MIRA from the CURRENT
DIRECTORY (a mira-OSS checkout, typically with uncommitted dev changes) instead of
wget-ing the main-branch tarball from GitHub — same target, ownership, and downstream
steps. Run from the repo root. Excludes (parity with the GitHub tarball): `.git`,
`venv`, `__pycache__`, `*.pyc`, `.env`, `.claude`, `.DS_Store`, `data`, `logs`,
`scratch`. Note the deploy is greenfield-only: it installs the schema into an empty
`mira_service` (drop the DB first on re-deploys — oneshot.sh does this for you).

**Deploy model-route defaults (lunaroute):** `deploy/mira_service_schema.sql` seeds all
five `model_configs` routes at the lunaroute gateway — `primary` on `glm-5.3`
(`provider_key`), the four aux routes on the `glm-5.3-flash` family (`subcortical_key`).
`deploy/python.sh` no longer string-patches the schema; hosted installs apply the seed
rows and then `deploy/postgresql.sh` rewrites `primary` from the chat config and the aux
routes from the subcortical config with UPDATEs after application (same mechanism as
OFFLINE_SQL), so a default (lunaroute) config leaves the seed untouched and any departure
is applied live. `chat_provider_type` takes `openai` (any OpenAI-compatible endpoint) or
`anthropic`; the old value `generic` is gone (breaking, no alias). The mechanism is owned
by `deploy/AGENTS.md`; a staging-copy refresh needs no patch.

**Working with the deployed instance** (minting API tokens, chat endpoint, DB probing,
turn-in-flight rules, memory/schema maps): read
`/home/admin/mira_instance_snapshots/AGENTS.md` on the host — the full quickbook of
validated commands lives there. Sarcophagi lineage (v1/v2/v3), extraction tooling, and
restore contracts are documented there too.
