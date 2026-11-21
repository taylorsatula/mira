# Peanut Gallery → Tophat: the standing session record

**Status:** implemented on branch `feat/peanut-gallery-tophat`. Implementation follows this runbook; §7's defaults were applied. §6 remains the verification path.
**Session:** redesign of the Peanut Gallery metacognitive observer (`taylorsatula/mira`, `main` @ `531d73e`).
**Purpose:** a runbook. An agent with an empty context window should be able to read this, orient in the tree, and actualize the change. Every behavioral claim in this document is cited to `path:line`.

**Reading note:** §1 describes the pre-change code at `main` @ `531d73e` (the "as-built" system this change replaces); its `path:line` citations name that commit, not the current tree. §3 onward describes the design this branch implements; where the two disagree, §3 and the code win.

---

## 0. TL;DR

Peanut Gallery was built to be a **metacognitive observer that keeps MIRA honest** — a standing, higher-order view of the session, like an internal monologue living at the top of the context window. What it ships today is a periodic **alarm system**: every fifth turn a fast-route observer pass re-reads ~20 messages plus a tool-noise ledger, and if the verdict is non-`noop`, drops a short-lived, loud directive into the HUD. In tool-heavy (coding) turns the observer drowns in tool chatter, its verdicts get noise, and the "guidance" becomes a loud interruption.

The redesign replaces the observer with a **tophat**: a standing, append-only **session record** that the model sees every turn, refreshed by the observer on the same 5-turn cadence (the cadence is the design's efficiency load-bearer, not a defect). The observer's job narrows from *"judge everything, emit an alarm"* to *"extend the record, and only ever say something about behavior when it genuinely must."* Steering (a rare, short, critical-only-fourth-wall directive) is preserved as the single exception; everything else is a record.

---

## 1. What it is today (the as-built system)

### 1.1 The full path

1. **Trigger.** `PeanutGalleryService` subscribes to `TurnCompletedEvent` (`cns/services/peanutgallery_service.py:58`) and fires on every 5th **segment** turn: `if event.segment_turn_number % PG_TRIGGER_INTERVAL != 0: return` (`:72`). `PG_TRIGGER_INTERVAL = 5` (`:29`). The event is published post-commit in `self._publish_events([event])` (`cns/services/orchestrator.py:1387-1395`), after `turn_number`/`segment_turn_number` are computed (`cns/services/orchestrator.py:1382-1386`). A dropped/synthetic turn publishes nothing.
2. **Snapshot + executor.** `_handle_turn_completed` (`:65-86`) snapshots `tuple(event.continuum.messages)`, copies the ambient context, and submits `_run_observation` to a `ThreadPoolExecutor` (`config.worker_pools.peanutgallery_workers`, `:53-56`; default 1 worker, `config/config.py:264`). The copied context carries the user identity, so the executor thread can call `get_current_user_id()` — no context-smuggling through the payload is needed.
3. **Observation.** `_run_observation` (`:88-103`) calls `self.model.evaluate(...)` and swallows any exception (fire-and-forget; a failed observation must not affect a turn).
4. **Model.** `PeanutGalleryModel.evaluate` (`cns/services/peanutgallery_model.py:81-96`) narrows `messages` to a window, builds an execution ledger, and makes one fast-route LLM call (`model_config="fast"`, `:143-147`). The window is sliced as `window_turns * 2` where `_observer_window_turns = analysis_interval_turns * 2` (`:71`), so with the default interval of 5 the slice is **20 conversational messages** (`:98-112`).
5. **Ledger.** `_build_execution_trace` (`:251-314`) extracts latest user request / latest assistant response (`:259-266`), ≤5 commitments (`:304-306`), and ≤8 tool-activity entries (`:308-311`), each rendered to ≤500 chars (`:360-373`) and escaped at interpolation (`:33-39`).
6. **Parse.** `_parse_observer_response` (`:157-197`) accepts `<mira:noop/>` (`:166`) or `<mira:peanutgallery type="concern|coaching|initiative">` (`:170-174`), with optional `severity="critical"` (`:188`). `noop` is the designed outcome; anything malformed degrades to `noop` (`:176-178`). `initiative` is never critical (`:219`).
7. **Injection.** `_apply_result` (`peanutgallery_service.py:105-115`) maps concern/coaching/initiative → `_inject_guidance`, which publishes `UpdateTrinketEvent` (`:135-145`) with `continuum_id="peanutgallery"` (a fake continuum id, `:136`), `target_trinket="PeanutGalleryTrinket"`, and context `{action: "add_guidance", type, text, ttl: PG_GUIDANCE_TTL_TURNS, critical}` (`:139-144`).
8. **Trinket.** `PeanutGalleryTrinket` (`working_memory/trinkets/peanutgallery_trinket.py:55`) implements `variable_name = "peanutgallery_guidance"` (`:55`), holds `_user_guidance` (`:68`), and on `add_guidance` stores an entry with `expires_at_turn = current_turn + ttl` (`:131-142`). `_expire_items` (`:73-88`) drops entries once `current_turn > expires_at_turn`; visibility is `current_turn <= expires_at_turn` (`:109`) — guidance added at turn T is visible at T, T+1, T+2 (≈3 turns, despite docs saying "expires after two turns"). `_clear_all_state` (`:90-94`) runs on segment collapse via the central flush (`working_memory/core.py:234-253`).
9. **Render.** `generate_content` (`:162-213`) renders into the **notification center** — a `<mira:hud>` assistant message after history, frame-wrapped by the composer (`working_memory/composer.py:245-278`; injected at `cns/services/orchestrator.py:610`): a standard section (`:180`) with a "silently repair" instruction, an initiative section (`:189-198`), and a critical section carrying a ~560-char fourth-wall block with `⟦PEANUT_GALLERY_DIRECTIVE⟧` and two scripted examples (`:200-210`). Guidance text is interpolated **unescaped** (`:183-185`, `:194-196`, `:206-208`) — the repo's own map calls this an injection defect (`working_memory/trinkets/AGENTS.md:45`).
10. **Base.** `StatefulTrinket._on_turn_completed` sets `current_turn = event.turn_number` (`working_memory/trinkets/base.py:195-199`) — a **cumulative user-message count** (orchestrator computes `turn_number = user_message_count`, `cns/services/orchestrator.py:1382-1386`), while the trigger uses `segment_turn_number`. The two clocks differ, so "TTL in turns" is not anchored to segment semantics.

### 1.2 The observer's output contract

`noop` | `concern(critical?)` | `coaching(critical?)` | `initiative(never critical)` — exactly one action per pass, and the concert between these four is how the observer is "used". The system prompt (`config/prompts/peanutgallery_system.txt`) frames this as a *rare, high-precision* intervention: "NOOP (most common — 90%+ of observations)" / "COACHING (rare)" / "INITIATIVE (rare)". The trigger fires unconditionally every fifth segment turn (`peanutgallery_service.py:72`) — nothing gates the pass on the conversation actually needing it.

### 1.3 Wiring & config

`cns/integration/factory.py:_initialize_peanutgallery_service` (`:409-450`) builds the model with `analysis_interval_turns=PG_TRIGGER_INTERVAL` (`:431-434`), constructs the trinket with `default_ttl=PG_GUIDANCE_TTL_TURNS` (`:436-440`), subscribes the service to the bus, and is gated by `config.system.peanutgallery_enabled` (`:419-421`). `PG_TRIGGER_INTERVAL`/`PG_GUIDANCE_TTL_TURNS` are imported from `peanutgallery_service.py:29-30` (`factory.py:429`). Config gates: `system.peanutgallery_enabled` (`config/config.py:154`) driven by `MIRA_PEANUTGALLERY_ENABLED` (`config/config_manager.py:77`); `worker_pools.peanutgallery_workers` (`config/config.py:264`).

---

## 2. Why it misses its role

The role (per README/glossary: "a metacognitive observer that emits at most one out-of-band concern or coaching signal", `docs/GLOSSARY.md:21`) — and the user's framing: *a tophat to the context window, like an internal monologue* — is **ambient continuity**: a standing higher-order view of the session, kept quiet and true over many turns. The as-built system misses that in six ways:

1. **The window is mis-sized, and the size is the wrong question.** The trigger fires every 5 turns, but the observer's window is `analysis_interval_turns * 2` for `_observer_window_turns` (model `:71`), then multiplied by 2 again in the slice (`:101`) — 20 conversational messages, each rendered to ≤500 chars. The window carries the *smoothing* the design needs (it blurs single-turn noise into trajectory), but in a tool-heavy coding turn it is mostly `[Used tool: bash]` markers and tool-result chatter. The 5-turn ledger then re-renders the same window a second time (`:251-314`).
2. **A periodic alarm, not a standing layer.** The pass runs every 5 turns and emits an alarm into the HUD; the default rendering either injects a "silently repair" instruction every active turn (`trinket:180`) or a critical block with two scripted examples every active turn (`:200-210`). Nothing is ambient: it is a beep with a short fuse.
3. **Three action types with mostly-identical semantics.** `coaching` and `initiative` are both non-critical, non-fourth-wall, background-steering directives; `concern` is the only one with genuine criticality (`:219`). This trifurcation multiplies the prompt surface and parse surface without adding a distinct behavior.
4. **TTL bookkeeping is wrong for the load.** TTL=2 but visibility is `current_turn <= expires_at_turn`, so guidance lives ~3 turns (visible at T, T+1, T+2) — while docs say "expires after two turns". The rendered `expires_in="0_turns"` at the last turn (`:183`, `:194`, `:206`). The trigger counts segment turns; the expiry counts cumulative user-message turns (`base.py:195-199` + orchestrator `:1382-1386`).
5. **Injection defect.** Guidance text is user-derived (spliced from conversation) and rendered unescaped into the composer (`trinket:183-185`, `:194-196`, `:206-208`) — a known defect (`working_memory/trinkets/AGENTS.md:45`).
6. **Dead code and fake ids.** The reversed `for message in reversed(...)` loop (`service:75-77`) computes nothing. `continuum_id="peanutgallery"` (`service:136`) is a fake continuum token in a payload field readers may probe.

The net effect in a coding session: the observer pays a full fast-route pass over mostly tool chatter, emits whatever fits, and the model spends turn-after-turn on guidance directives instead of the work. This is the failure the redesign targets.

---

## 3. The design: peanutgallery as a standing tophat

### 3.1 The concept

A **tophat** — a cap at the top of the context window — is a standing record, not an event. The unit of continuity is not "a fresh verdict about the last 5 turns" but "the session's shape, kept as a record that compounds." The observer's 5-turn cadence stays (that is the correct cost amortization — a standing per-turn observer would re-pay the full window-read every turn), but what the pass is *for* changes:

- **Standing record (the tophat):** appended once per pass (every 5th segment turn) with 1–3 one-line beats. It is append-only, per session, and flushed on segment collapse by the standard `_flush_stateful_trinkets` path (`working_memory/core.py:234-253`). It never expires. It is the structure the model anchors to — "here is what has happened so far, in sequence, with decisions and pivots" — which is exactly what tool-dense coding sessions lack.
- **Rare steering (the exception):** the standing record mostly *states* what happened; it does not *command* behavior. When the session genuinely degrades, the pass carries a single steering directive — the only piece of the model's "verdict" that survives. Steering is rare (most passes emit none) and short-lived: it renders only until the next pass (5 turns), and it is never a takeover of the record.

### 3.2 Non-goals

Not part of this change: reviving `<mira:internal_monologue>` (dead code — `utils/tag_parser.py` only parses/strips it; nothing instructs it or consumes it; the feature was never made to work). The trigger cadence stays at 5. No changes to the main model, client surface, or HUD delivery mechanism. No new user-facing feature — the HUD is a private anchor by design (`config/system_prompt.txt:43`; `cns/services/orchestrator.py:610` — the notification center is injected as an assistant message in the LLM request, never streamed to the client).

### 3.3 Refinement vs. as-built — the delta

| # | Component | Today | After |
|---|-----------|-------|-------|
| 1 | Trigger | every 5th segment turn | **unchanged** |
| 2 | Window | `_observer_window_turns = 10`, sliced ×2 → **20 conversational messages** (model `:71`, `:101`) | **bounded delta**: last ~5 segment turns of conversational messages (count-based: ≤ 2 × interval conversational messages) — the new work since the last pass, plus the standing record for continuity |
| 3 | Ledger | full-window commitments/tool activity (≤5 / ≤8, 120-150 chars/entry) | **delta-scoped ledger** — same shape and caps, but built only over the delta window |
| 4 | Output | `noop` / `concern` / `coaching` / `initiative` + guidance, with a 4-way criticality split | `session_log` (1–3 lines, each one sentence) + optional `steering` (text + `critical` flag). No fourth "initiative". `coaching` is absorbed into `steering` (it was a non-critical steering directive with the same signature) |
| 5 | Rendering | standard + initiative + critical sections with a ~560-char 4th-wall block + 2 scripted examples, every active turn, ~3-turn TTL | one standing block with an unconditional `<session_log>` section (record, no expiry) + an optional `<steering>` section (rare, expires at next pass). Guidance text escaped at interpolation |
| 6 | State | dict of expiring guidance entries with absolute-expiry turns | per-user: `log: list[str]` (append-only) + `steering: {text, critical} | None` |
| 7 | Injection | `continuum_id="peanutgallery"` fake id | real continuum id from the event |
| 8 | TTL machinery | `PG_GUIDANCE_TTL_TURNS`, `_expire_items`, `expires_at_turn`, `turns_remaining` | removed |
| 9 | Parsing | 4 action types + severity | `session_log` (required-ish) + `steering` (optional); failure → skip append, no steering |
| 10 | Prompts | long standing contract with 4 action types and 90%+ noop | short contract: append 1–3 beats, output steering only when it genuinely matters |
| 11 | Dead code | reversed loop, doubled window math | removed |

### 3.4 Steering semantics (the one behavioral piece kept)

- **Standard steering** — silent repair, one sentence, no fourth-wall authorization.
- **Critical steering** — the one concern-type verdict that authorizes the fourth wall: the model may explicitly acknowledge the slip to the user. This is the only section that may be visible to the user indirectly (only via the model's own reply), and it is rare. The fourth-wall text is the *system prompt's* contract, not a scripted example block — the model output must be plain prose; the examples in the current trinket (`trinket:203`) are deleted.
- **Lifetime:** until the next pass (≈5 turns). If the next pass emits no steering, the previous steering is dropped silently. The record never re-renders because of expiry; it re-renders when the pass appends.
- **Criticality rule:** critical only when the flagged issue is genuinely broken — false claim that the user is acting on, a failed commitment, an inconsistency the user noticed. The standing record's whole point is that everything else is not worth an alarm.

### 3.5 The record's shape (the standing block)

One to three beats per pass, one sentence each, ≤ ~300 chars, append-only, newest last. The record is a *record* — it describes what happened and the direction the session took, not a directive. Rendering (default; see Open questions):

```
<mira:peanutgallery>
  <header>Session record (append-only — describes session shape, not a directive; may lag the conversation):</header>
  <session_log>
    - Session begins. The user wants to talk about making an indian food dish.
    - The conversation pivots to cybersecurity; Mira runs many toolcalls, including nmap on a local Roborock S8 vacuum.
    - User just pivoted to checking the weather.
  </session_log>
  <steering>One directive, or nothing.</steering>
</mira:peanutgallery>
```

Rules: every line is escaped at the interpolation point (`html.escape`, per `working_memory/trinkets/AGENTS.md:45`). Empty record → return `""` (no section). Header may be omitted (prefer a stable, short one) — see Open questions.

### 3.6 The observer contract (parse & failure modes)

New parse shape (`_parse_observer_response`):

```
<mira:session_log>
  one sentence per line   # 1–3 lines, one sentence each, ≤ ~300 chars
</mira:session_log>

<mira:steering critical="true">
  one directive sentence      # OPTIONAL — absent on most passes
</mira:steering>
```

- `session_log` missing / empty → **skip the append** (log unchanged). The record is not rewritten; silence is the default fate.
- `steering` missing → no steering (and any prior steering is dropped by the trinket).
- `steering` present but empty → no steering.
- `steering critical="true"` (case-insensitive) → critical; any other value or a bare `<steering>` → standard. Only `critical="true"` takes the fourth-wall path. A `<mira:steering>` tag appearing inside the `<mira:session_log>` block is treated as log content, never as a directive.
- Neither tag present → no-op pass.
- Failure anywhere (parse error, LLM error, empty response) → no-op pass; log unchanged. A failed pass still clears any standing steering — the pass boundary is what bounds steering's lifetime.

All of this stays inside the same "fire-and-forget, never blocks a turn" pattern as today (`peanutgallery_service.py:88-103`).

### 3.7 Observer input

The pass reads:
1. **Standing record** — the current list of beats (from the trinket; empty on the first pass).
2. **Delta window** — the last ~5 segment turns' conversational messages (count-based: `min(len(conversational_messages), 2 * interval)`; tool-message text is *not* included in the formatted messages).
3. **Delta ledger** — same machinery as today, but scoped to the delta window only: latest user request + latest assistant response + ≤5 commitments + ≤8 tool-activity entries, each ≤500 chars.

`_format_with_ids` is kept (8-char ids give the observer anchor points). The 10→20 window doubling (`model.py:71` + `:101`) is deleted.

### 3.8 The steering prompt's new contract (the important part)

The observer's system prompt is rewritten to a standing-record contract. The load-bearing principles:

- The pass's job is to append the session record. Appending is almost always the right output.
- Steering is the exception — it is issued only when the session genuinely degraded (epistemic overreach, contradiction, trace divergence, looping, drift, capability bluff) and the record alone cannot anchor the model. Most passes emit no steering.
- The record is a background briefing, never a directive; it does not instruct the model to change behavior.
- The record is stale by definition (it lags the turn). Stale record ≠ wrong steering.
- Total output discipline: one or three lines + at most one steering directive; no preamble, no headings, no analysis.

This is the difference from today's `peanutgallery_system.txt` (134 lines built around "NOOP 90%+" / "CONCERN" / "COACHING" / "INITIATIVE"): one new contract that does both halves honestly. The old 90%-noop rule becomes "append, don't steer."

---

## 4. Component-by-component plan

### 4.1 `working_memory/trinkets/peanutgallery_trinket.py` (rewrite)

- `variable_name` stays `"peanutgallery_guidance"` (it is the key the composer section `peanutgallery_guidance` is registered under; changing it would break `working_memory/composer.py:65`).
- State: `_user_records: dict[str, list[str]]` + `_user_steering: dict[str, dict | None]`. No ` GuidanceEntry`, no `ActiveGuidance` TypedDict, no `uuid4`, no `_expire_items` TTL logic, no `expires_at_turn`, no `turns_remaining`.
- `_expire_items()` → always return `False` (documented: the record never expires; the steering section is cleared only by the next pass that emits nothing). `_clear_all_state()` → pop both dicts for the current user (segment collapse path).
- New event handling: `context["action"] == "observe"` → append `context["log_lines"]` to `_user_records[user_id]`; set or clear `_user_steering[user_id]` from `context["steering"]` (`{text, critical} | None`).
- `generate_content` renders the standing block: `<mira:peanutgallery>` → `<session_log>` with escaped lines → optional `<steering critical="...">` → close. Steering renders only when present; escaped at interpolation (`html.escape`, plain text — `working_memory/trinkets/AGENTS.md:45`).
- Keep the class methods `__init__` (dropping the `default_ttl` arg) and `handle_update_request`. Keep the `StatefulTrinket` base (needed for the segment-collapse flush).
- **Do NOT** add any out-of-band state store, Valkey field, or API surface — the standing block is an in-memory, session-scoped slot like every other trinket.

### 4.2 `cns/services/peanutgallery_model.py` (rewrite core)

- `PeanutGalleryResult` becomes:
  ```python
  @dataclass
  class PeanutGalleryResult:
      session_log: list[str] = field(default_factory=list)  # new beats to append
      steering: Optional[str] = None                        # rare directive
      steering_critical: bool = False
  ```
- `PeanutGalleryModel.__init__(llm_provider, analysis_interval_turns)` — keep the constructor arg (factory passes `PG_TRIGGER_INTERVAL`); add turn-keyed windowing for `_delta_slice` (`self._delta_window_turns = analysis_interval_turns`), with a message-count fallback when the history carries no `turn_id` stamps; the old `* 2` window doubling is gone.
- `evaluate(messages, standing_log: list[str])` — new signature. `standing_log` comes from the trinket (the service reads it before submitting the observation). First pass → `[]`.
- `_delta_slice(messages)` — `messages` → conversational subset (`_is_conversational_message`, kept) → keep the conversational messages from the last `interval` turns (keyed on the `turn_id` stamped on user and assistant messages), falling back to a message-count window when no `turn_id` stamps are present → return the raw slice from the earliest kept message.
- `_build_delta_ledger(delta_messages)` — the existing `_build_execution_trace` logic, re-scoped to the delta slice only (latest request/response, ≤5 commitments, ≤8 tool entries, ≤500-char truncation, escaping). The ledger always renders; an empty delta yields `none` placeholders.
- `_run_observer` — build the prompt from a single user message: `<standing_record>` + `<window_messages>` (formatted messages) + `<delta_ledger>` + the task instruction. One fast-route call (`model_config="fast"`), unchanged.
- `_parse_observer_response` — the new contract (`:286-292`-shape below; do not parse old action types). Fail to no-op.
- `_format_with_ids` — kept (8-char ids).
- `_render_message_content` — kept (500-char cap).
- Deleted: `_get_observation_messages` (superseded by `_delta_slice`), `_extract_commitments`, `_extract_tool_uses` keep; `_build_execution_trace` renamed/reused as `_build_delta_ledger`.

### 4.3 `cns/services/peanutgallery_service.py` (rewrite)

- Keeps `PG_TRIGGER_INTERVAL = 5` (`:29`) and `PG_TRIGGER_INTERVAL`'s comment. **Removes** `PG_GUIDANCE_TTL_TURNS` (`:30`).
- `_handle_turn_completed` — same trigger (`:72`); **deletes** the dead `reversed(...)` loop (`:75-77`); snapshots `tuple(event.continuum.messages)`; also reads the current standing record from the trinket handle (`trinket.get_session_log()`), and passes both into the executor thread.
- `_run_observation(messages, standing_log)` — unchanged try/except fire-and-forget (`:88-103`); errors → log-and-swallow, no turn impact.
- `_apply_result(result)` — now publishes a single `UpdateTrinketEvent` with `action: "observe"`, `log_lines: result.session_log`, `steering: {"text": result.steering, "critical": result.steering_critical} | None`. **Real** continuum id from the event.
- Constructor takes `trinket` (new required arg, the object returned by the factory — see 4.4).

### 4.4 `cns/integration/factory.py` (`:409-450`)

- Construct the trinket first and keep the strong reference. Pass it to `PeanutGalleryService` as its third constructor argument; the service reads `trinket.get_session_log()` when it submits the observation. The model receives the record only as the pass's input, never as a handle (§4.2).
- Drop the `default_ttl=PG_GUIDANCE_TTL_TURNS` argument and the `PG_GUIDANCE_TTL_TURNS` import (`:429`).
- The gate (`peanutgallery_enabled`) and the no-op behavior when closed stay (`:419-421`).
- The factory's own composition order is otherwise unchanged (`:431-445`).

### 4.5 Prompts

**`config/prompts/peanutgallery_system.txt`** (rewrite): the standing-record contract. Paragraph-level structure:
1. Role — a background recorder for the session's shape, not an alarm; you are not replying to the user.
2. The record — 1–3 one-sentence beats per pass, appended, never rewritten; capture decisions, pivots, stuck points, milestones.
3. Steering — the exception; one sentence; only when the session genuinely degraded; critical only when a user is acting on a false claim or a commitment was dropped; standard otherwise; absent on most passes.
4. Staleness — the record lags the turn; do not treat it as current state.
5. Output format — exactly the two tags; nothing else.

**`config/prompts/peanutgallery_user.txt`** (rewrite): `<standing_record>` (current beats, escaped) + `<window_messages>` (formatted messages with IDs, escaped) + `<delta_ledger>` (escaped) + the task sentence.

**`config/system_prompt.txt:43`** (amend): update the HUD sentence — "time, reminders, memories, subagent results, and the peanut gallery's session record and rare steering" — and add one sentence naming the record's nature: background, may lag up to 5 turns, not a directive, do not cite it to the user.

### 4.6 Docs & maps (same commit)

- `README.md:29-36` — rewrite the "Subcortical & Peanut Gallery" section for the tophat shape.
- `docs/GLOSSARY.md:21` — one-sentence update: peanut gallery is now a standing session record with rare steering.
- `working_memory/trinkets/AGENTS.md:20` (`peanutgallery_trinket.py` line) — rewrite the line. `:25` — construction line, update for the strong reference and dropped `default_ttl`. `:45` — the escaping note's second half: the peanutgallery citation becomes fixed-by-this-change (remove the "unescaped defects" clause).
- `cns/services/AGENTS.md:25-26` — rewrite service/model lines.
- `config/prompts/AGENTS.md:61-62` — rewrite the two prompt lines.
- `working_memory/trinkets/HOW_TO_BUILD_A_TRINKET.md:47,134,539` — update the stateful-trinket roster entries.
- `cns/integration/AGENTS.md:15` — the peanutgallery-enabled gate line, if the text names the trinket constructor args.

### 4.7 Inline module docstrings

New/rewritten files carry their own docstrings; the docstring of `peanutgallery_service.py` (line 2's "The model's `action_type` is the whole contract") is re-written to describe the new contract. `peanutgallery_trinket.py`'s docstring drops the TTL/expiry language.

---

## 5. Implementation phases (ordered)

1. **Phase 0 — Orientation.** Re-read §1-§3 of this doc, the four code paths (service, model, trinket, factory), the two prompts, the composer/core sections, and `tests/AGENTS.md` (the verification doctrine). Confirm nothing has changed since `531d73e`.
2. **Phase 1 — Observer model + prompts.** Rewrite `peanutgallery_model.py` (result dataclass, `evaluate`, delta slice, delta ledger, new parse) and the two prompt files. No behavior change is wired yet (the old service still drives it) — but the model is constructed by the factory, so keep the constructor signature compatible.
3. **Phase 2 — Trinket.** Rewrite `peanutgallery_trinket.py` (standing record, new event payload, no TTL, escaping).
4. **Phase 3 — Service + factory.** Rewrite `peanutgallery_service.py` (dead loop, new event, real continuum id, trinket handle) and `factory.py:_initialize_peanutgallery_service` (strong reference, dropped `default_ttl`).
5. **Phase 4 — Docs & maps** (§4.6-4.7). `config/system_prompt.txt:43` amendment.
6. **Phase 5 — Verification** (§6).
7. **Change report** — end the report with EXECUTED/UNVERIFIED per Tier 1/2 doctrine.

Each phase is independently verifiable: pyflakes/`py_compile` on touched files at every phase boundary (Tier 0), live check at Phase 5 (Tier 1/2).

---

## 6. Verification

Doctrine (`tests/AGENTS.md`): **no test files, no pytest, no stubs.** Verification is live against the dev infrastructure. This change is Tier 2 (new behavioral surface).

**Tier 1 — live pass:**
- Boot the dev service (uncomment/enable `peanutgallery_enabled`) and drive a session (~10+ turns, including a tool-heavy coding-ish stretch). Confirm from logs/HUD: the standing record appears, it grows exactly on the 5th segment turn, it contains the pivots/decisions (sanity-check 2-3 beats by hand), steering is rare and rare in the right situations, the previous steering is dropped when the pass emits none, and the record is clean-flushed on segment collapse (start a new segment and confirm the record re-initializes).
- Confirm the HUD is model-only (not streamed) and the standing record renders inside `<mira:hud>` on turns where the record exists.

**Tier 2 — probe + boot check (decision point, see §7):**
- A path-probe is a production-code-class thing (`tests/protected/AGENTS.md`; load `writing-probes` before any probe decision). Give the observer a canned conversation and confirm the parse keeps the contract (session_log appended, no steering, escapes). It is probabilistic by nature (LLM output), so the probe should check *plumbing* (service subscribes → executor runs → trinket renders → flush works), not the *verdict*.
- Optionally register a boot-gate `CheckSpec` (`utils/power_on_self_test.py:473-487` roster) named `peanutgallery` that exercises the wiring. **This is optional — see §7.**
- Second agent re-derives the diff (skip only when one-shot execution covers it).

**Static gates (Tier 0):** `python3 -m py_compile` and pyflakes on every touched file.

---

## 7. Open questions (defaults applied if unanswered)

1. **Record rendering shape** — labelled `<session_log>` block with `-` lines (default), or prose header? Default: labelled block with a short stable header.
2. **Steering lifetime** — cleared by the next pass that emits none (default; one-pass lifetime, ≈5 turns). Alternative: a 2–3 turn TTL. Default chosen for simplicity.
3. **Record cap** — unbounded (default; per-session, flushed on collapse, ~1 line per 5 turns). Alternative: cap at ~40 lines with oldest dropped. Default chosen — the growth is trivial per the user's own estimate.
4. **Boot-gate CheckSpec** — optional (default: no new spec; Tier 2 verification through the live path instead). The user did not request it.
5. **Fourth-wall semantics** — kept for critical steering only (default, preserves the one genuinely high-stakes capability). Coaching/initiative types are dropped.
6. **Any out-of-band record** — no. Standing block only (default). No Valkey copy, no API surface.

---

## 8. Appendix — orientation map (for an agent with zero context)

### Files to read first (in this order)
1. `config/system_prompt.txt` — the base prompt + line 43 (HUD contract) — `config/system_prompt.txt:43`.
2. `working_memory/composer.py` — section layout + notification-center builder — `:38-67`, `:245-278`.
3. `working_memory/core.py` — compose round-trip, trinket routing, state flush — `:111-180`, `:184-204`, `:234-253`.
4. `working_memory/trinkets/base.py` — `EventAwareTrinket` / `StatefulTrinket` — `:162-216`.
5. `working_memory/trinkets/peanutgallery_trinket.py` — the trinket (whole file, 213 lines).
6. `cns/services/peanutgallery_service.py` — trigger + injection (147 lines).
7. `cns/services/peanutgallery_model.py` — the observer (387 lines).
8. `cns/integration/factory.py:409-450` — the wiring.
9. `cns/core/events.py:72-116`, `:171-188` — `TurnCompletedEvent` / `UpdateTrinketEvent`.
10. `config/prompts/peanutgallery_system.txt` + `peanutgallery_user.txt` — the observer contract.
11. `tests/AGENTS.md` — the verification doctrine (no test files).
12. `config/config.py:150-160, 261-267`; `config/config_manager.py:71-82` — config gates.

### What to expect in the tree

- `cns/services/peanutgallery_*.py` — the observer and its orchestrator.
- `working_memory/trinkets/peanutgallery_trinket.py` — the HUD slot.
- `config/prompts/peanutgallery_*.txt` — the observer's prompts.
- `.pi/agents/`, `.agents/` — unrelated agent fragments; ignore.
- `notes/` — session-specific reports; ignore (they are not design material).
- No tests exist for peanutgallery (verified by searching `tests/`); the doctrine forbids adding them.

### The one-paragraph mental model

A user turn ends → `TurnCompletedEvent` is published post-commit → the peanutgallery service filters on segment turn % 5 → the observer (one fast-route LLM call) reads the last 5 turns plus the standing record → it returns 1–3 beats to append plus optional steering → the trinket appends the beats to its per-user record and publishes its content → the composer frames the whole thing inside `<mira:hud>` as a sliding assistant message before the model's next reply. On segment collapse the record is flushed with every other stateful trinket.

---

*Prepared as a design/runbook for the redesign of peanutgallery. Implement against `taylorsatula/mira` `main` @ `531d73e`.*
