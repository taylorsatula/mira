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
| `docs/` | Operator-facing documentation |
| `lt_memory/` | Long-term memory: storage, scoring, retrieval, linking, entity services |
| `lt_memory/processing/` | Extraction pipeline and consolidation |
| `scripts/` | Operational CLI entry points run against a deployed service |
| `tests/tmp/` | Display exhibit for the disposable-probe pattern: autodeleted-on-sight test policy, one exemplary probe |
| `tools/` | Tool framework: base class, repository, config registry |
| `tools/implementations/` | All concrete LLM-callable tools |
| `utils/` | Cross-cutting infrastructure: identity, scheduling, storage, security, observability |
| `web/` | Browser UI pages, serving contract, load manifests |
| `web/assets/javascript/` | Client JS modules, client side of the WebSocket turn protocol |
| `working_memory/` | Event-driven system-prompt composition via trinkets |
| `working_memory/trinkets/` | Trinket implementations: one `variable_name` slot each |

**Ancestor maps:** an ancestor map of `<dir>/` is the `AGENTS.md` of a parent
directory, up to repo root (for `cns/api/`: `cns/AGENTS.md` and this root
file). The harness loads the full ancestor chain whenever a file in the
directory is read, so those maps are present in context together with this
one.

**Section order:** `# <dir>/ — <role, one clause>` → `## Rules` → `## Files`
→ `## Wiring` (omit if the directory has no cross-file edges) → at most one
deep-dive titled with a domain term. Bullets everywhere; prose only inside
`###` subsections and the deep-dive; `###` subsections may live inside
`## Rules` and do not consume the deep-dive slot. A reader in another
directory decides relevance from the title alone.

**`## Rules` — each bullet must satisfy all four requirements:**
1. Describes a constraint that is not visible from reading one file in this
   directory.
2. Does not restate global doctrine from this root file. Ancestor maps are
   additional context, not repetition of it.
3. Is understandable using only this map and its ancestor maps.
4. Names the enforcing symbols or files in backticks. State the rule's
   content here; use the anchor to locate the enforcement. Do not summarize
   another map's rule — cite the map or symbol that owns it.

**Cross-map rules:**
- Each fact is documented in exactly one of the maps that load together. A
  constraint that applies at both parent and child level may appear in both;
  the non-owning map states it in one line and cites the owning map.
- Adding a member to a registry, enum, or contract family requires
  coordinated edits in several files. Document the full list once, in order,
  with the file anchor and the consequence of skipping each step.
- `## Wiring` documents edges where this directory is an endpoint, plus
  ordering constraints within this directory. Data flows owned by ancestor
  code are cited by owner, not restated.

**`## Files`:** one bullet per file: what it owns, entry points (names an
agent would search for), non-obvious gotchas. `Consumers:` (frontend maps:
`Calls:`) only for files used across directory boundaries; those fields
belong in `## Files` only — elsewhere, anchors go in prose. `__init__.py`
and doc files get an honest entry; a stale doc gets a staleness Rule.

**Anchors:** paths are repo-root-relative; bare file names in `## Files`
resolve against the map's own directory. Do not cite line numbers; they
change on the next edit of the target file.

**Do not include:** restatements of root doctrine; change history ("we used
to X"); content derivable from the file's own docstring; unspecific
summaries; filler to lengthen a map. (Root itself may briefly restate
cross-cutting patterns that maps also carry — the one-home rule applies
between maps, not between root and a map.)

**Maintenance (required — trigger table, not judgment):** map updates are
part of the same commit as the triggering change. None of the actions below
is optional. When in doubt, update the map — an unnecessary edit costs a
line; a missed one misleads every session.

| Event | Required action |
|---|---|
| New file in a mapped directory | Add its `## Files` bullet |
| Deleted or renamed file | Update or remove its `## Files` bullet; fix any anchor citing the old path |
| Behavior or contract change in a file | Grep that directory's map for the changed symbol; update the Rule, Files gotcha, or Wiring edge stating the old behavior |
| New member of a registry, enum, or contract family | Update the blast-radius Rule in the owning map |
| Directory reaches ≥2 source files, or gains an invariant its parent's map does not own | Create its `AGENTS.md` (shape above) |
| File moved between directories | Update both maps' `## Files` sections |
| New map created or removed | Update the parent map's `## Files` pointer |

**Enforcement:** the audits below are the check that no trigger was missed.
Run them before committing any change that touched source files or maps. A
clean audit means no map action was skipped; a flagged anchor is a skipped
trigger. For changes the audit cannot express, the grep-the-map step in the
table is the check.

**Audit before committing a map change:**

    for m in **/AGENTS.md; do
      d=$(dirname "$m")
      # path anchors (repo-root-relative), taken from everything EXCEPT the
      # ## Files section — Files names resolve against the map's directory,
      # including subdirectory-qualified names like agents/base_system.txt:
      awk '/^## Files/{s=1;next} /^## /{s=0} !s' "$m" \
        | grep -ohE '`[^`]*\.(py|txt|sql|md|js|sh|hcl|html|css|json)(:[A-Za-z_][A-Za-z0-9_]*)?`' \
        | sed 's/`//g; s/:[A-Za-z_][A-Za-z0-9_]*$//' | grep / | grep -v '^/' | grep -v '\\*' | grep -v ' ' | sort -u \
        | while read p; do [ -e "$p" ] || echo "MISSING-PATH ($m): $p"; done
      # (absolute paths like /opt/vault/... refer to the deploy host, not the
      #  repo; the audit skips them)
      # bare file names from ## Files only, sub-bullets included:
      awk '/^## Files/{f=1;next} /^## /{f=0} f' "$m" \
        | grep -ohE '^[[:space:]]*- `[^`]*`' | sed 's/^[[:space:]]*- `//; s/`$//' | sort -u \
        | while read p; do [ -e "$d/$p" ] || echo "MISSING-FILE ($m): $p"; done
    done

## 🚨 Critical Principles (Non-Negotiable)

# 🛑 NO MOCKS. NO PARALLEL SUITES. VERIFICATION IS LIVE OR IT DOESN'T COUNT.

Prohibited: test files, pytest fixtures, mock objects, stubs, fakes, offline test databases, and any check that runs against simulated infrastructure. A passing mock-based check provides no information about this system.

Every line in this tree is agent-written; no human has traced any path. Type-clean, pyflakes-clean code that has never executed is the standard failure product of this workflow and has shipped here before. Verification is therefore mechanical and live, never assumed.

Required verification, in the POST tradition:
- **Boot gate** — probes Vault, Postgres, and model routes against live infrastructure before the server binds.
- **Path-probes** — invoke critical runtime paths against the live system exactly as production would.

Probe-surface membership (standing rule): any path whose failure would report incorrect data to users, lose data, or degrade silently — reads, writes, searches, auth flows, failure paths. If a required probe cannot run against live infrastructure, fix the code until it can; do not simulate.

Quality guarantee: boot-survival plus path-probe coverage. A clean boot verifies the wiring; a passed path-probe verifies the handler. Code covered by neither is unverified — flag it in review. A bug found in unprobed code is fixed together with the probe that covers it.

Probes are production code: normal review discipline, the same credentials plumbing, production-identical failure behavior.

`tests/tmp/` holds one exemplary disposable probe as a display exhibit of this pattern — any test file added there is autodeleted instantly; see `tests/tmp/AGENTS.md`.

### ⚡ Realtime verification loop (proportionate by behavioral surface)

Models are pretrained to verify by writing tests. When that reflex fires during a change, write a path-probe instead — same verification goal, production-code artifact.

Select the tier by behavioral surface touched, not by change size:

**Tier 0 — no behavioral surface.** Comments, docstrings, formatting, import regrouping, verified-mechanical renames, documentation. Run `py_compile` and pyflakes on touched files. No further verification required.

**Tier 1 — behavioral edits within surface already covered by boot or probes.** Bug fixes, small logic changes, contract-preserving refactors, config wiring of existing behavior.
1. Execute the changed path once against live infrastructure: `python3 -c` with the real store, or a request against the running dev server.
2. Re-read the diff against the doctrine invariants: (a) does every failure path report the failure truthfully, (b) is any failure silently degraded, (c) does the diff assert behavior it has not executed.
3. End the change report with a verification state: EXECUTED (what ran) or UNVERIFIED (why not, and which probe would cover it).

**Tier 2 — new behavioral surface or elevated stakes.** New features, endpoints, or paths; schema or data-migration changes; failure-behavior, security, or auth changes; new dependencies. Tier 1 steps, plus where applicable:
- Persistence changes: round-trip against dev infrastructure — write, read back, verify, clean up. RLS and database constraints are part of the check.
- Paths meeting the probe-surface membership rule: add the path-probe.
- Adversarial pass: a second agent independently re-derives the diff. Recommended for new-surface and failure-behavior changes; when one-shot execution covers the change, skip the pass and state the skip.

**Probe surface:** path-probes are production code registered alongside the POST gate — same credentials plumbing, same failure behavior, same review discipline. Membership follows the standing rule above.

### Technical Integrity
- **No Shortcuts**: never substitute a cheaper check for the required one — no mock-based verification, no skipped probe, no "improvements" during extraction, no deferred path-probe. Shortcuts ship as silent breakage.
- **Verify Contracts Before Building On Them**: verify an unfamiliar helper's or internal API's contract at the boundary you depend on — inputs, outputs, types, side effects, failure modes — with the smallest direct probe or existing reference usage before building on it. Most preventable slipups come from trusting names or remembered APIs.
- **Evidence-Based Position Integrity**: form assessments from evidence and hold them under pushback. Do not adjust conclusions to match the human's apparent preference; when their proposal contradicts the assessment, push back and say why.
- **Blunt Technical Communication**: reject technically unsound ideas directly — "bad", "infeasible" — and correct wrong assumptions about code or constraints immediately ("That's wrong"). After rejection or correction, provide the working alternative or the accurate facts.
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
- **Exception policy is positional**: critical request-path code propagates; fire-and-forget consumers log and swallow; background durability paths (collapse, extraction) tolerate-and-log because the model is off the call stack. Know which side you are on before writing a try/except.
- **Per-user tool config**: `config.<tool>_tool` merges the user's override fresh on every access over the global default; secret fields round-trip via the redaction sentinel. (`config/config_manager.py`, `utils/tool_config_store.py`)
- **Prompt templates load via `load_prompt()`** — never `open()` a prompt file directly. (`config/prompts/loader.py`)
- **Event handlers are synchronous**, registered by event class `__name__`; async work inside a handler spawns a thread with copied context. (`cns/integration/event_bus.py`)
- **The LLM is an untrusted component**: escape, allowlist, and validate at every LLM boundary — tool arguments JSON-Schema-validated, untrusted content wrapped (`<untrusted_content>`), credentials injected server-side (the model names a credential it never sees), thinking signatures round-tripped untampered. (`clients/llm/`, `utils/prompt_injection_defense.py`, `tools/implementations/web_tool.py`)
- **Inter-component coordination goes through the event bus**, not direct service calls: event taxonomy owned by `cns/core/events.py`, bus owned by `cns/integration/event_bus.py`. New features subscribe and publish.
- **Trinket state is per-user**, keyed by the contextvar — trinket instances are process-global singletons shared across users; never store user state on instance attributes. (`working_memory/trinkets/base.py`)

### Activity Days & Use-Day Scheduling
MIRA uses **use-day scheduling** — periodic jobs fire based on user activity days, not calendar time. A user who logs in Monday, skips Tuesday, returns Wednesday has their counter tick on Monday and Wednesday only. This prevents wasted work on inactive users and ensures jobs run at consistent engagement intervals.

The mechanics — activity tracking, the `get_users_due_for_job(interval)` gate, job registration, and the current job list — are owned by `utils/AGENTS.md`; the `*_use_days` interval semantics are owned by `config/AGENTS.md`. Read those maps before adding or changing a use-day-gated job.

### Provider Stall Detection
All live LLM transports run through `clients.llm.lifecycle.LLMLifecycle`, which enforces provider response timeouts and raises `ProviderStallError` on stall. There is no fallback route — a stall or provider failure propagates. The full policy (non-streaming vs streaming wrapping, route criticality, cost recording) is owned by `clients/llm/AGENTS.md` and `utils/AGENTS.md`. Provider-specific transports are dialects under `clients/llm/dialects/`; adding one is a multi-file blast radius documented in `clients/llm/dialects/AGENTS.md`.

### Power-On Self-Test
MIRA uses POST checks in `utils/power_on_self_test.py`. The pre-server gate (`run_pre_server_post_gate`) runs before Hypercorn binds and launches checks in a subprocess so probe-side singletons cannot leak into the serving process. The gate is bounded: `PRE_SERVER_GATE_ATTEMPTS` rounds with `PRE_SERVER_GATE_RETRY_SECONDS` between failures, then it parks (sleeps forever, server never binds) rather than exiting — a restart-on-exit supervisor would otherwise turn gate failure into an unbounded loop of real, billed LLM probes. Set `MIRA_POST_GATE_FAILURE_ACTION=exit` to exit instead under supervisors like systemd where restart backoff is already sane. The in-process CLI remains operational (`python -m utils.power_on_self_test pre-server`); the post-server probe and its CLI shim are documented in `scripts/AGENTS.md`. POST checks must exercise real infrastructure and must not use mocks.

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
- **Database**: Always use `psql -U postgres -h localhost -d mira_service` - postgres is the superuser, mira_service is the primary database

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
