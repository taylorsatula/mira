# tui/ — MIRA terminal chat client — build plan (user-approved 2026-09-18)

> **REVISION 2026-09-18 (post-build pivot, user-directed):** the first
> implementation shipped a full-screen Textual UI. Review after live use:
> good primitives, but the full-screen app overrides terminal-native
> behavior (virtual scroll, key/mouse capture, screen ownership) and its
> framework machinery concentrated most of the build's defects. UI layer
> rebuilt as **Rich + prompt_toolkit with native scrollback**; streaming
> renders in a bounded preview region above the prompt and finalizes into
> scrollback; settings became config-file-only (no settings screen);
> `client.py`/`protocol.py`/`endpoints.py` survived verbatim. The original
> Textual sections below remain as the historical record of build one.

Greenfield TUI client for a **deployed** MIRA instance. This document is the
session-surviving source of truth for the build; every protocol claim below was
verified against source during the 2026-09-18 investigation session. After
implementation, this file stays as the design record — the `tui/AGENTS.md` map
owns the living contracts.

## 1. Locked decisions (user-confirmed via questionnaire + chat)

| Decision | Choice |
|---|---|
| Framework | **Textual** (6.6.0 installed) — async-first, differential renderer, ModalScreen overlays, docked input. Avoids hand-rolled ANSI state machines. |
| Wire | `websockets` 15.x (asyncio client) for WS; `httpx` for REST history fetch. Pin both in `tui/requirements.txt` with rationale comments. Server `requirements.txt` untouched. |
| Placement | **`tui/` top-level dir**, run via `python -m tui`. Pure client — imports **nothing** from the server tree. Peer of `web/` (another client surface). |
| Endpoint/key store | Local JSON at `~/.config/mira-tui/config.json`, chmod 0600, multiple named endpoints, active pointer. |
| History default | Everything from the active session + the most recent session summary; configurable in /settings. |
| Follow-ups during a turn | **Outbox queue**: input stays live; queued messages visible in a dock widget; undo-until-sent via key command (pull back into editor); auto-send one at a time as each terminal frame arrives. Server enforces one active turn per *user* (`UserRequestLock`, `TURN_BUSY`), so queuing is the only correct client behavior. |
| Pages | Only `/settings` (endpoint CRUD + select, history-fetch mode, thinking toggle). **No** /memory, /toolfeedback, or other secondary pages. |
| Streaming | WS turn protocol (preferred over sync POST /v0/api/chat). |
| Design language | botwithmemory web UI: black background, mono font, cyan active-state / green thinking / red alerts, `You:` / `MIRA:` colored sender labels, tool calls as ASCII trees (`├─`/`└─` param/result lines) with state transitions (running → collapsed → expandable), "Where we left off last session:" header for pre-session context. |

## 2. File structure

```
tui/
  __main__.py      entry: python -m tui — argparse (--config-debug), boots app
  app.py           Textual App: main screen (transcript + dock), bindings, screens
  components.py    One widget class per message type:
                   UserMessage / AssistantStream / ToolCallBlock /
                   SegmentSummaryBanner / QueuedMessageList / StatusBar
  protocol.py      Client-side mirror of the WS frame schemas (strict, extra=forbid)
  client.py        MiraClient: WS connect + auth + frame pump, REST history pager;
                   emits typed events ONLY (delta, tool_start, tool_result,
                   turn_end, proactive, error, auth_ok...) — UI never touches the wire
  endpoints.py     EndpointStore: load/save ~/.config/mira-tui/config.json (0600)
  settings.py      SettingsScreen (ModalScreen): endpoint list/select/add/edit/delete,
                   history_fetch mode, include_thinking default
  requirements.txt textual, websockets, httpx — pinned, commented
  AGENTS.md        map, created with the implementation (registered in root map registry)
  BUILD_PLAN.md    this file
```

## 3. Server protocol (all verified in `cns/api/websocket_chat.py` + `cns/api/data.py`)

### WebSocket — `ws(s)://<base>/v0/ws/chat`
- First frame within 10 s: `{"type":"auth","token":"<api token>"}` → `{"type":"auth_success","user_id":...}`. Auth errors → `protocol_error` (AUTH_TIMEOUT / AUTH_REQUIRED / AUTH_FAILED). Token can be a session token or API token.
- Send: `{"type":"message","message_id":<uuid4>,"content":str(1..100_000),"include_thinking":bool, ("image","image_type") XOR ("document","document_type")}` — attachments mutually exclusive, skipped in v1.
- Send: `{"type":"halt","turn_id":...}` (stop active turn → `turn_stopped` reason "halt"); `{"type":"ping"}` → `pong`.
- Receive per turn: `turn_started {turn_id, message_id, segment_id}` → zero+ of `assistant_delta {turn_id, segment_id, entry_id, content}` (accumulate per entry_id), `thinking {content}` (only if include_thinking), `tool {event: tool_detected|tool_executing|tool_completed|tool_error, tool_name, tool_id, arguments?, result?, is_error?}`, `model_error {message}` (invalid tool call, recovering), `context_reset {turn_id, segment_id}` (**discard all buffered text for the turn, restart the entry buffer**) → exactly ONE terminal: `turn_complete {continuum_id, response, tools_used, processing_time_ms, emotion}` | `turn_stopped {reason: halt|disconnect}` | `turn_error {code, message}`.
- Unsolicited: `proactive_message {message_id, turn_id?, content, created_at}` (server-initiated heartbeat breakout — the web UI never renders it; the TUI is its first consumer, render as an assistant message), `server_shutdown`, `protocol_error {code, message, message_id?}` (TURN_BUSY, NO_MATCHING_ACTIVE_TURN, MALFORMED_FRAME...).
- Reconnect recovery pattern (from web core.js): re-fetch one history page (limit 50), rebuild the interrupted turn from the first user row.

### REST — Bearer token
- `GET /v0/api/data?type=history&limit=N[&before=<cursor>][&message_type=regular|all]` → `{"messages":[...],"meta":{"has_more":bool,"next_before":"<opaque>"}}`. Keyset pagination ONLY — `offset` and `search` are rejected server-side. Pages come oldest→newest; keep fetching with `meta.next_before` while `has_more`.
- Message shape: `{id, role, content, timestamp, metadata, tool_call_id, is_error}`.
- **Sessions = segments.** Sentinels arrive inline among history messages: `metadata.is_segment_boundary='true'`, `status: active|paused|collapsed`. A **collapsed sentinel's `content` IS the session summary** (`metadata.display_title`, `segment_end_time`); the active sentinel has `content='[Segment in progress]'`. `message_type=regular` includes sentinels and excludes system notifications + keepsleeping heartbeats.
- Default load recipe: paginate newest→oldest until crossing the newest **collapsed** sentinel → render it as the SegmentSummaryBanner, then everything after it chronologically = active session. Handle "no collapsed sentinel within fetch window" gracefully (active session only). Modes: `session_plus_summary` (default) | `session_only` | `all` (page until has_more=false, oldest-first rendering).
- Auth: API tokens cannot be self-minted (`POST /v0/auth/api-tokens` needs a browser session + CSRF). /settings instructs: paste the raw token from the web settings page (shown once at mint). Bearer needs no CSRF; no rate limiting on these routes.

## 4. Event-bus architecture (from pi's proven structure)

- `client.py` owns all I/O and emits typed events onto a callback/queue surface; `app.py` subscribes and mutates components. The streaming client never touches widgets.
- Transcript: scroll-follow container (auto-scroll to end while pinned; user scroll-up unpins, "jump to bottom" on new content when unpinned).
- Fixed dock (bottom): QueuedMessageList → Input → StatusBar (endpoint name, turn state, emotion/timing).
- Streaming: mutate the active AssistantStream component on delta; ToolCallBlock transitions running (spinner) → collapsed preview → expandable on keypress.
- `/settings` = ModalScreen with focus capture; switching endpoint tears down the client and replays the startup path (disconnect → fetch history → render → connect).
- Keymap: Enter submits (or queues if a turn is active), Ctrl-C halts active turn, queued items removable/editable via key command (undo-until-sent), `/settings` and `/exit` slash commands.

## 5. Endpoint store schema

```json
{
  "active": "devvm",
  "endpoints": {
    "devvm": {
      "base_url": "http://localhost:1993",
      "api_key": "…",
      "history_fetch": "session_plus_summary"
    }
  }
}
```
0600 perms, fail-fast with clear errors on missing/corrupt file; first run creates it with one empty endpoint and opens /settings.

## 6. Verification environment (live, ready as of 2026-09-18)

- Fresh deploy-only dev VM on the libvirt host (admin@192.168.1.9), all five routes live on lunaroute (glm-5.3 / glm-5.3-flash family), healthy, ~152 s deploy. VM sits on the host's NAT: reach it from this Mac via tunnel:
  `ssh -L 1993:192.168.122.252:1993 admin@192.168.1.9` → base URL `http://localhost:1993`.
- Minted API token for this instance: `<redacted 2026-09-26: instance decommissioned, token dead>`. **Note:** the fresh instance's history is nearly empty (one probe message, no collapsed segments yet). For summary-banner verification either (a) collapse the active segment via `POST /v0/api/actions {"domain":"continuum","action":"collapse_segment"}` with the Bearer token, then chat more to open a new segment, or (b) restore the 1079-message v4 sarcophagus with `oneshot.sh mlfactory_v4_mira --fresh` on the host.
- Verification battery (Tier 2, live, NO MOCKS): boot gate = app connects + auths + renders; probes = history load (all modes), streaming turn with a tool call, context_reset handling, follow-up queue + undo-until-sent, halt, endpoint switching reloads history, auth-failure path (bad token → clear error), proactive_message rendering (if a heartbeat breakout can be coaxed). Adversarial pass: second agent re-derives the diff.

## 7. Repo rules that bind this build

- NO MOCKS / no test files — verification is live probes (running the TUI against the instance); `tests/tmp` autodeletes, `tests/protected` needs the exact authorization phrase.
- `tui/AGENTS.md` map + root map registry row land in the same commit as the code; the frame mirror in `protocol.py` cross-references `cns/api/websocket_chat.py` in both maps (drift risk is owned there).
- Minimal deps, documented justification in `tui/requirements.txt`; the server's `requirements.txt` gains nothing.
- Fail-fast, fail-loud client: infrastructure errors surface in the UI status area with the real message, never silent defaults.
