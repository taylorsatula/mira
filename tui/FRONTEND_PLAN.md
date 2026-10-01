# tui/ — greenfield frontend plan (user-approved 2026-09-30)

Session-surviving source of truth for the frontend rebuild. **Read this file
first after any context compaction**, then the orientation list at the bottom.

## Status

- Plan presented and approved; user decisions recorded below.
- 2026-09-30 (post-compaction): implementation delegated to Sonnet 5.5
  subagents at the user's request. **The "Pinned contracts (v2)" section below
  supersedes the summary contracts in "Modules" and the
  `run_in_terminal` wording in the invariants wherever they differ.**
  Waves: (1) `text.py`+`transcript.py`+probe 1 ‖ `screen.py`+probe 2 →
  (2) `chat.py`+`app.py`+protocol-level local-server probe → (3) maps/docs,
  delete `render.py` → (4) live probe vs VM (user will nudge when it boots) →
  blind second-agent re-derivation → report EXECUTED/UNVERIFIED.
- The user asked: build as far as possible without the VM, then report and
  wait for the nudge.
- Ledger: wave 1A DONE — `tui/text.py` + `tui/transcript.py` written,
  orchestrator-reviewed; probe 1 PASSED (36 cases, 3483 splits, 113740
  asserts; `scratchpad/probe1/probe_text_transcript.py`). Choices accepted:
  one combined think|mira block regex (stream spans == final filter spans);
  timestamp `^` anchor kept (a timestamp after a leading newline is shown,
  as in render.py).
- Ledger: wave 1B DONE — `tui/screen.py` written; probe 2 PASSED 49/49
  (`scratchpad/probe2/harness.py`; pyte venv at `scratchpad/probe2/venv`).
  UNVERIFIED in pyte: rule residue after a terminal HEIGHT shrink (pyte drops
  rows unlike real terminals) — check by hand in iTerm2/Terminal.app.
  Orchestrator review found two startup-ordering defects, sent back to the
  same agent: (1) push-to-bottom must be lazy one-shot on the first write of
  any kind, else a banner emitted before `run()` scrolls off-screen;
  (2) `_running` must flip in `run_async(pre_run=…)`, else an emit can run
  the print step before the app's first frame. FIXED (`_ensure_bottom`,
  `_mark_running` via `pre_run`); probe 2 now 54/54 incl. a startup scenario.
  Review note for the re-derivation pass: `emit` awaits `_finished.wait()`
  unbounded during teardown.
- Wave 2 DONE (pending one fix): `chat.py` + `app.py` rewritten,
  orchestrator-reviewed; probe 3-local PASSED 140/140
  (`scratchpad/probe3/probe3_local.py`, stand-in `standin.py` validates every
  frame with `parse_inbound_frame`; mutation control on the response-fallback
  rule went red as expected). Agent choices accepted: message frame carries
  `raw.strip()`; `in_flight` exists before `send_message` returns
  (`message_id=None`); user halt returns the queue to the box on ANY turn end;
  `AuthOk`/`ConnectFailed` honoured only while `connecting`; frames for an
  inactive turn → red protocol line; quit prints partial reply / held proactive
  / `sent but not confirmed by MIRA:` / `not sent:`.
  Found real `client.py` defects (witness `scratchpad/probe3/client_defects.py`):
  D1 `run()` posts nothing on a clean server close; D2 `connect()` never resets
  `_server_shutdown_seen`. The agent first compensated in chat.py (`PumpEnded`);
  orchestrator ruled root-cause fix: edit `client.py` (both defects), delete
  `PumpEnded`. DONE: client.py D1/D2 fixed, `PumpEnded` deleted;
  `client_defects.py` → CLIENT_DEFECTS_FIXED PASSED; probe 3 rerun 140/140
  (`scratchpad/probe3/run2.log`). **Tell the user**: client.py (the "keep as
  is" backend) got these two defect fixes.
- Wave 3 DONE, orchestrator-reviewed: `tui/AGENTS.md` rewritten (38 lines,
  9 Rules, 13 Files, 4 named flows; `client.fetch_history` marked VESTIGIAL),
  root registry row, `cns/api/AGENTS.md` two citations, `__main__`/`__init__`
  docstrings (exit-code line fixed by orchestrator), `requirements.txt`
  +`prompt_toolkit==3.0.51`, `tui/render.py` DELETED (reference grep clean;
  untracked `UNCHECKED_KATAS.txt:32` still names render.py — historical, left).
  Audits clean for tui/ and cns/api maps (pre-existing root `bin/deploy-only.sh`
  and cns/api `actions.py` LONG-BULLET flags untouched).
- Blind re-derivation DONE (swarm-investigator, Sonnet; plan withheld):
  `scratchpad/review/blind_review.md`. 1 finding + 2 design risks + 4 side
  discoveries; orchestrator triage:
  - ACCEPTED S1: `ReplyStream.flush` showed unclosed tags raw → a reply cut
    off inside the closing `<mira:my_emotion>` tag printed tag text. Contract
    CHANGED (supersedes v2 text.py bullet): flush/finish never reveal held
    text — commit `_filter(raw[:hold])`, discard the rest. FIXED +
    orchestrator-reviewed; probe 1 PASSED (43 cases, 4139 splits, 149097
    asserts, incl. the reviewer's three cases → `["Hi there."]`).
  - ACCEPTED risk 1 as a defect: server sends the terminal frame before
    releasing `UserRequestLock` (`process_turn` finally) → queued auto-send
    races into `TURN_BUSY`. Client fix: bounded `TURN_BUSY` retry
    (0.5/1/2 s, private `RetrySend`), Ctrl+C during the wait → `send
    cancelled`. **Tell the user** the root cause is server-side ordering in
    `cns/api/websocket_chat.py` (not changed by us). Fix → chat agent.
  - ACCEPTED risk 2: websockets 1 MiB `max_size` vs untruncated tool-result
    frames → `MAX_FRAME_BYTES` 64 MiB in client.py. Fix → chat agent.
  - ACCEPTED side: ModelError notice flushes the stream first; server-
    initiated `turn_stopped` gets a notice. Fix → chat agent.
  - DEFERRED side: cancelled `connect()` leaves the socket open until the
    server's 10 s AUTH_TIMEOUT; mismatched `turn_started` leaves in_flight
    set (needs protocol drift). Mention in report.
  - Chat-agent fixes DONE + orchestrator-reviewed (`RetrySend`,
    `TURN_BUSY_RETRY_DELAYS`, `MAX_FRAME_BYTES` with a 1 MiB mutation control
    that went red, ModelError flush, server-stop notice). `tui/AGENTS.md`
    updated by orchestrator for all three contract changes; audits clean.
- Final offline verification: probe 1 PASSED on final code (149097 asserts);
  py_compile + pyflakes clean on all of `tui/`. Probe 2 PASSED 54/54, probe 3
  PASSED 177/177 on final code (no stray processes) →
  `scratchpad/final_probe2.log`, `scratchpad/final_probe3.log`.

### Live VM state (2026-10-01)

- `<libvirt-host>` is the libvirt HOST, not the VM. The
  MIRA VM is a libvirt guest at `<vm-nat-ip>` (NAT, reachable only
  via the host). Port 1993 open there; health is `GET /v0/api/health`
  (healthy, deployed version string `2026.06.25`).
- Orchestrator tunnel: `ssh -fN -o ExitOnForwardFailure=yes -L
  19930:<vm-nat-ip>:1993 <user>@<libvirt-host>` → base URL `http://127.0.0.1:19930`
  (re-open with the same command if it dies; find it with
  `lsof -ti tcp:19930 -sTCP:LISTEN`).
- `--login` EXECUTED live (single-mode, zero prompts) into the scratch store
  `scratchpad/live/store.json` (0600, endpoint `devvm`); the user's real store
  untouched (md5 recorded in `scratchpad/live/real_store.md5`).
- Live probe DONE (`scratchpad/live/live_probe.py`, run1–run4 logs; ~14 real
  turns): L1 startup, L2 bad token (server `AUTH_FAILED`), L3 plain turn
  (live `<mira:my_emotion>` filtered, thinking status seen), L4 tool turn
  (`invokeother_tool` → `punchclock_tool`, footer `3 tools`), L5 queue
  auto-send (no race seen, 0.0 s gap), L6 halt, L7 relay drop + reconnect,
  L8 quit — ALL PASS. No frame drift vs deployed server. No client defect.
- Live finding: first sends often bounced `TURN_BUSY` for tens of seconds
  (≈50–57 s). VM journal: heartbeat turns run for this user ~every minute and
  take minutes each (lunaroute streaming failures, invalid `heartbeat_tool`
  calls, "heartbeat_tick … skipped: maximum number of running instances"),
  holding the per-user lock. → client changed to a PATIENT wait (quick
  retries, then poll every 3 s up to 600 s; Ctrl+C cancels). DONE +
  orchestrator-reviewed; probe 3 177/177 (13b = 8 s busy then accept;
  scenario 6 now witnesses restore-to-box via `INVALID_MESSAGE`); 600 s
  exhaustion UNVERIFIED in the stand-in (bound shown from the code). Live
  recheck saw no bounce (lock free) → forced live check RUNNING: connection A
  (real `MiraClient`) holds a long turn, the TUI sends on connection B →
  cross-connection lock `TURN_BUSY` → patient wait. Map updated.
  Heartbeat bug report delivered to the user in chat (they will investigate).

### Live VM probe plan (step 4 — run when the user says the VM is up)

1. Reachability: `nc -z -G 3 <vm-nat-ip> 1993` from the host, then
   `GET /v0/api/health` through the tunnel.
2. Token into a SCRATCH store (never `~/.config/mira-tui/config.json`):
   `python3 -m tui --config <scratchpad>/live/store.json --login --base-url
   http://127.0.0.1:19930 --save-as devvm` (single-mode → zero prompts;
   never print the token).
3. Live harness = probe 3's pty + pyte harness pointed at the real server
   (copy `probe3/harness.py`; no stand-in). Scenarios, each asserting
   exactly-once tokens and no tag/ESC/rule residue in history:
   plain turn (ask for a unique token echoed back); tool turn (prompt that
   forces a tool, e.g. ask MIRA to search memory or check the time via a
   tool); queue: Enter a second message mid-reply → it auto-sends after the
   footer (this exercises the real `TURN_BUSY` lock-release race — record
   whether a retry fired); Ctrl+C halt mid-reply → `stopped after`, queued
   text back in the box; bad api_key store → exit 1 with `--login` guidance;
   disconnect + reconnect (restart the service over ssh if the user allows,
   else UNVERIFIED); `context_reset` / `INVALID_MESSAGE` / `TURN_SETUP_FAILED`
   only if reproducible (else UNVERIFIED).
4. Ask the user to try it by hand in iTerm2/Terminal.app: resize narrower and
   shorter mid-stream (pyte could not verify reflow / height-shrink residue).
5. Report EXECUTED / UNVERIFIED per path; do not commit unless asked.
  UNVERIFIED pending the live VM: real-server timing/lock/halt latency,
  `TURN_SETUP_FAILED`/`INVALID_MESSAGE` paths, real tool payloads; not driven:
  Ctrl+C during a reconnect, `NO_MATCHING_ACTIVE_TURN` notice; resize while
  streaming in a real reflowing terminal.
- (`tui/BUILD_PLAN.md`, `tui/app.py`, `tui/endpoints.py`, `tui/login.py`,
  `cns/api/websocket_chat.py`, `cns/api/AGENTS.md` already show as modified in
  `git status` from earlier sessions — not ours; leave them alone except
  `app.py`, which we rewrite, and the one `websocket_chat.py` bullet in
  `cns/api/AGENTS.md`.)
- Do NOT commit unless the user explicitly asks (repo git rules; load the
  `git-workflow` skill before any commit).

## The user's ask (verbatim intent)

- Greenfield the **user-facing side** of the TUI; the backend (`client.py`,
  `protocol.py`, `endpoints.py`, `login.py`) "works pretty well" — keep it.
- A **Rich** TUI: chat back and forth with MIRA over the **streaming WebSocket**.
- The only "nicety": write/edit a message while MIRA is replying — a **fixed
  input bar at the bottom** with `───` rules above and below the input field;
  when input grows past one line it **expands upward**.
- Must not have the classic TUI bug: **text double-prints because it isn't
  properly cleared.**
- "Forward-looking design that can be extended … deep modules with clean
  contracts … ship product you're proud of."

## User decisions (2026-09-30)

1. **Enter while MIRA is replying → QUEUE** the message; send automatically as
   soon as the reply completes.
2. **No grey delimiter rows** in scrollback — *as long as it's still visually
   easy to tell when MIRA is done*. Plan: status line flips from spinner to
   idle; plus a dim one-line footer after each finished reply (e.g.
   `· 14s · 2 tools`) — call this out in the final report so the user can veto.
3. **Keep** the dim tool one-liners in scrollback (`· used calendar_tool` /
   `· calendar_tool failed`).
4. **Live verification instance:** the user is libvirt-ing a dev VM at
   **`<libvirt-host>`** (VM reached through an SSH tunnel). Mint a token
   into a SCRATCH store, never the user's real one:
   `python3 -m tui --config <scratchpad>/tui-probe.json --login --base-url http://127.0.0.1:19930 --save-as devvm`
   (single-mode instance → zero prompts). The user's real store
   `~/.config/mira-tui/config.json` (active `default` → `http://localhost:1993`)
   must not be modified.

## Verified facts (checked against source/installed libs this session)

- `tui/protocol.py` matches the server's frame models in
  `cns/api/websocket_chat.py` exactly — no drift. (Server diff in working tree
  only adds `AUTH_UNAVAILABLE` protocol_error during auth; `ProtocolErrorFrame`
  already accepts any code.)
- `assistant_delta.content` is **raw model text** (`orchestrator.py` ~L980–990:
  `TextEvent.content` forwarded verbatim) → contains `<mira:...>` tags (e.g.
  `<mira:my_emotion>`) and possibly think blocks. Client must filter, including
  tags split across deltas and spanning lines.
- `entry_id` = one provider step; a new entry starts after tool calls. Synthetic
  file-artifact text (`📎 **[name](/v0/api/files/…)** (size)`) arrives as text.
- `turn_complete.response` = tag-parsed clean text (my_emotion preserved). It
  can contain text that was **never streamed**: blank-after-tool-error fallback
  (`orchestrator.py` ~L1319–1330). Rule: display `response` only if the stream
  displayed nothing.
- `context_reset` comes from context-overflow retry (`orchestrator.py`
  ~L1266–1295); second overflow streams the literal fallback
  `collapse the segment, please. incremental compaction failed`.
- Server enforces one turn per user: `TURN_BUSY` protocol_error (with our
  `message_id`) on a second message; heartbeat turns can also hold the lock.
  Validation before `turn_started` → `protocol_error` `INVALID_MESSAGE` /
  `TURN_SETUP_FAILED`.
- Rich 14.2.0 (probed): strips only control codes 7,8,11,12,13 — **ESC
  survives**, even inside `Text(...)` (OSC title / `\x1b[2J` pass through).
  Plain-string `console.print` parses markup: `"[/bold]"` raises `MarkupError`;
  `:thumbs_up:` → emoji; highlight on by default. → Always print `Text`, with
  `markup=False, emoji=False, highlight=False`, and sanitize ourselves.
- prompt_toolkit **3.0.51** (installed, not yet in `tui/requirements.txt`):
  - `run_in_terminal(func)` → task; `in_terminal` chains calls in order via
    `app._running_in_terminal_f`; waits for CPRs; `renderer.erase()`; runs func
    in cooked mode; then `renderer.reset()`, `_request_absolute_cursor_position()`,
    `_redraw()`. If the app isn't running, func just runs.
  - Renderer (non-fullscreen) height = `max(_min_available_height, last_height,
    preferred)`; `_min_available_height` = rows from cursor to terminal bottom
    (CPR) → the app fills to the bottom; a flexible empty `Window()` at the top
    of the HSplit keeps the bar pinned at the bottom. Without CPR support it
    degrades to sitting right under the output (still correct).
  - `FormattedTextControl` honors a `('[SetCursorPosition]', '')` fragment →
    use it to tail-follow the in-progress paragraph in a height-capped Window.
  - `TextArea(dont_extend_height=...)`, `Application(refresh_interval=...,
    erase_when_done=...)` exist.
  - To verify during build: `create_output()` unwraps `StdoutProxy` to the real
    stdout when `patch_stdout` is active.
- Environment: Python 3.12.11 at
  `/opt/homebrew/Caskroom/miniconda/base/envs/mira/bin/python3`; installed
  rich 14.2.0, websockets 15.0.1, httpx 0.28.1, pydantic 2.11.9,
  prompt_toolkit 3.0.51. **pyte missing, tmux missing.** No local MIRA
  (port 1993 closed; local 5432/6379/8200 open but no AppRole creds — don't try
  to boot MIRA locally; use the VM).

## Design

### Screen

```
  (normal terminal scrollback: selectable, searchable, survives exit)
  You
  what's on my calendar tomorrow?

  MIRA
  Let me check.
  · used calendar_tool
  You have two things tomorrow:
  - 9:00 dentist
  - 2:30 call with Sam, and I'd sugg        ◄─┐  live region: in-progress paragraph (tail-follow, capped)
  queued: also move the dentist if it…        │  queued/sending messages (dim cyan, 1 line each)
  MIRA is replying… 14s · Ctrl+C to stop      │  status line (dim; red for alerts)
  ──────────────────────────────────────────  │  top rule
  I'll also need to move the dentist if       │  input TextArea: grows upward (filler shrinks),
  it overlaps with▌                           │  capped height, scrolls inside beyond cap
  ──────────────────────────────────────────◄─┘  bottom rule (pinned to terminal bottom)
```

HSplit top→bottom: flexible filler `Window()` · live pending Window · queued
lines Window · status Window · rule · TextArea · rule.

### Modules

```
 __main__.py ─► app.py                  composition root: TTY check, config, connect, run → exit code
                  │ builds
     ┌────────────┼─────────────────┐
     ▼            ▼                 ▼
  chat.py ────► screen.py        client.py ─► protocol.py      (client/protocol/endpoints/login UNCHANGED)
  ChatSession   Screen
     │  │       (prompt_toolkit)
     │  └─────► transcript.py    look of scrollback elements (Rich Text)
     └────────► text.py          ReplyStream + sanitize + tag filter (pure, no I/O)
```

| Module | Contract | Hides |
|---|---|---|
| `text.py` (new, pure) | `sanitize(s) -> str`; `ReplyStream`: `feed(entry_id, delta) -> list[str]` (newly committed display lines), `pending() -> str` (display-safe in-progress tail, no newlines), `finish() -> list[str]`, `discard()` | tag hold-back across deltas, clean-cut newlines, entry boundaries → paragraph break, blank-line collapse, leading `[5:47pm]` strip, escape stripping |
| `transcript.py` (new) | `you(text)`, `mira_label()`, `mira_lines(lines)`, `tool_line(name, ok)`, `reply_footer(...)`, `notice(text)`, `alert(text)` → `rich.text.Text` | colors/spacing in one place |
| `screen.py` (new) | `Screen(post_intent)`; `async emit(*renderables, live=None)` (print to scrollback; apply `live` in the same erase→print→redraw step); `set_live(live)`; `consume_input(text)` (compare-and-clear: removes `text` prefix only); `restore_input(text)`; `async run()`; `close()`. Intents posted: `Submit(text)`, `Interrupt()`, `Quit()` | prompt_toolkit layout, pinning, keys, every redraw/cursor move, Rich→terminal printing |
| `chat.py` (new) | `ChatSession(client, screen, queue).run() -> int` | turn state machine, outgoing message queue, halt, reconnect, deferred proactive, `response` fallback |
| `app.py` (rewritten) | `run(store) -> int`, `setup_guidance(store)` | wiring, first connect with clear errors |

Delete `tui/render.py` (filters move to `text.py`; retired Rich renderers,
`ActiveTurn`, `history_blocks` ablated per repo "don't deprecate; ablate").
REST `_post_chat` + `preflight` go away (WS auth replaces preflight).

### Event flow — one queue, one consumer

```
 keyboard ─► Screen key bindings ─► Intent: Submit | Interrupt | Quit ─┐ put_nowait (QueueFull → status flash, never silent)
                                                                       ▼
 server ──► MiraClient.run() pump ─► ClientEvent ─────────────► [ bounded asyncio.Queue ]  (client uses await put → backpressure)
                                                                       │ single consumer, strict order
                                                                       ▼
                                                             ChatSession.handle(item)
                          ┌──────────────────────────────────────────┼──────────────────────────┐
                          ▼                                          ▼                          ▼
                ReplyStream.feed/finish                Screen.emit(blocks, live=…) / set_live   MiraClient.send_message / send_halt / connect
```

### Turn states (with the outgoing queue)

```
 connecting ─AuthOk─► idle ─Enter(text)─► sending ─turn_started─► replying ─turn_complete|turn_stopped|turn_error─┐
                       ▲   (box cleared; text shown    │ TURN_BUSY / INVALID_MESSAGE / setup fail:                 │
                       │    "sending…" in live area)   │ text back into input box (restore_input), red line        │
                       │◄──────────────────────────────┘                                                           │
                       │◄─── queue empty ◄─────────────── reply end: commit stream tail, footer ◄──────────────────┘
                                         queue non-empty ─► pop oldest ─► sending
 Enter while sending/replying ─► append to outgoing queue (box cleared; shown as "queued: …" in live area)
 any state ─Disconnected / server_shutdown─► disconnected: red line; unsent (sending + queued) texts back into the
            input box; status "disconnected — Enter to reconnect"; Enter → connecting → send box text if any
```

- A user message prints to scrollback as a **You block only at
  `turn_started`** (server accepted it). Until then it lives in the live
  region ("queued"/"sending"). Invariant: a message is in exactly one place —
  input box, live region, or scrollback. Never lost, never doubled.
- `turn_started` for a message: emit You block + MIRA label in one step.
- Queue sends on any reply end (complete / stopped / error).
- Disconnect while `sending` (before `turn_started`): restore to box with note
  "connection lost before MIRA confirmed this message — check history before
  resending".

### Double-print prevention (invariants)

1. prompt_toolkit is the ONLY redrawer; it measures wrapped rows. No hand-counted
   cursor-up / clear-line anywhere.
2. ONE path to scrollback: `Screen.emit` → `run_in_terminal(func)`; `func`
   prints AND applies the new live state; redraw follows. Moving text from live
   → scrollback is one atomic step.
3. Scrollback append-only; printed soft-wrapped (`soft_wrap=True`, no inserted
   newlines) so terminal reflow handles resize; only the bar redraws.
4. Single consumer → no interleaved emits/state updates.
5. Typed text: one copy (box), cleared by compare-and-clear on accept.
6. Final `response` never re-printed when stream displayed text.
7. Stray stdout/stderr writes (library warnings) caught by
   `patch_stdout(raw=True)`; our Rich console writes to the REAL stdout inside
   `emit` (capture the real file before patching).

### ReplyStream algorithm (text.py)

- `sanitize` per delta: `\r\n`→`\n`, drop lone `\r`; remove CSI/OSC/other ESC
  sequences, remaining C0 (except `\n`, `\t`), DEL, C1. Never emits ESC.
- Per entry: raw buffer from `base` (start of uncommitted raw).
- `hold` = earliest of: unclosed think open; unclosed paired `<mira:NAME …>`
  (no `</mira:NAME>` after); trailing partial tag (`<` with no `>` whose
  fragment is a prefix of `<think>`, `</think>`, `<mira:…`, `</mira:…`).
  Safe region = `raw[base:hold]`, monotonic.
- Commit cut = last `\n` in the safe region NOT inside any complete tag-block
  match span. Committed chunk = filter(raw[base:cut+1]) split into lines;
  `base = cut+1`.
- `pending()` = filter(raw[base:hold]) (no newlines possible).
- Filter = remove complete think blocks + `<mira:…>` paired/self-closing (regex
  from current `render.py`; build tag literals by **concatenation** — literal
  tag strings get HTML-entity-mangled when written through agent tool
  payloads; verify on-disk bytes).
- Blank lines: drop leading blanks of reply; collapse runs >1; drop trailing at
  finish. Leading `[h:mmam]` stripped from the reply's first text only.
- Entry change: flush previous entry fully (finish semantics for that entry),
  insert one paragraph break before next entry's first text. Tool events also
  flush the current entry (step ended) before the tool line prints.
- `finish()`: commit everything remaining, applying the non-streaming filter
  (unclosed tags shown as-is, matching the ported web-client semantics).
- `discard()` (context_reset): drop uncommitted state; ChatSession prints dim
  notice that the server discarded the text above and restarted the reply.

### Behavior / keys

| Situation | Behavior |
|---|---|
| Enter, idle | send (→ sending) |
| Enter, sending/replying | queue |
| Alt+Enter (Esc,Enter) / Ctrl+J | newline in box (Shift+Enter indistinguishable in most terminals) |
| Paste | bracketed paste inserts as-is |
| Ctrl+C replying | halt; if no `turn_started` yet, send halt on arrival; 2nd Ctrl+C while stopping → quit |
| Ctrl+C idle | box non-empty → clear; empty → quit. Ctrl+D on empty box / `/exit` → quit |
| tool frames | status "running <tool>…"; scrollback dim one-liner on completed/error |
| model_error | dim notice |
| thinking frames | status "MIRA is thinking…" (content not rendered) |
| proactive_message during a turn | deferred until reply ends; else printed immediately as MIRA block |
| auth failure at startup | exit 1 before UI with `--login` guidance |
| not a TTY | exit 1 with message |
| long paragraph, no newline yet | live region tail-follows (capped rows) |
| input taller than cap | scrolls inside the box |
| status elapsed timer | `refresh_interval` redraw; computed at render from monotonic start |

## Pinned contracts (v2, 2026-09-30) — authoritative

### Design corrections found while pinning

- **No `run_in_terminal` for our output.** Verified in
  `prompt_toolkit/input/vt100.py`: `cooked_mode` turns `ECHO|ICANON` back on
  for the duration of `run_in_terminal`, so a key typed during an emit is
  echoed by the tty into scrollback AND later read into the box — the
  double-print bug. `raw_mode` leaves OFLAG alone (OPOST/ONLCR stay on, `\n`
  still prints as CRLF). Screen therefore owns a **print step that never
  leaves raw mode**: `await app.renderer.wait_for_cpr_responses()` (bounded,
  library default 1 s), then synchronously, with no await in between:
  `app.renderer.erase()` → write the rendered ANSI through `app.output`
  (`write_raw` + `flush`) → `app._request_absolute_cursor_position()` →
  `app._redraw()`. Our own emits are serialized by an `asyncio.Lock`.
  `patch_stdout(raw=True)` stays as the safety net for stray library writes
  (rare path; it uses `run_in_terminal` internally — accepted).
- **Push-to-bottom at startup.** After `renderer.reset()` the next redraw
  happens before the CPR answer arrives, so whenever free rows exist below the
  output the bar would draw mid-screen and then jump down (flicker). Before
  the Application starts, Screen writes `rows - 1` newlines so the cursor sits
  on the bottom row; from then on output only ever scrolls, the bar stays at
  the bottom, and the top filler only absorbs `last_height` slack.
- **Ctrl+C halt returns the queue to the box.** A user-initiated halt means
  "stop"; queued follow-ups go back into the input box instead of
  auto-sending. (Flag to the user in the final report.)
- **Rejected / undeliverable messages:** every unsent message (in-flight +
  queued) goes back to the box together, oldest first, joined by a blank line.

### `tui/text.py` (pure; no I/O; imports only stdlib `re`)

- `sanitize(text: str) -> str` — `\r\n`→`\n`; drop lone `\r`; remove CSI
  (`ESC [` … final byte 0x40–0x7E), OSC (`ESC ]` … BEL or `ESC \`), DCS/SOS/PM/APC
  (`ESC P|X|^|_` … `ESC \`), any other `ESC`+1 char, lone `ESC`; remove
  remaining C0 except `\n` and `\t`, DEL, and C1 (U+0080–U+009F). Output never
  contains `\x1b`.
- `class ReplyStream` — one per reply (per turn, or per proactive message).
  - `feed(entry_id: str, delta: str) -> list[str]` — sanitizes `delta`,
    appends to the current entry; returns newly committed display lines (each
    without `\n`; `""` = blank line). An `entry_id` different from the current
    one first flushes the current entry (see `flush`).
  - `flush() -> list[str]` — end of the current provider step (called on tool
    events and entry change): commits everything remaining in the entry with
    the non-streaming filter (unclosed tags shown as-is, web-client
    semantics); the next non-blank line committed afterwards is preceded by
    one blank line (paragraph break).
  - `pending() -> str` — display-safe in-progress tail: no `\n`, no ESC, never
    a tag fragment, never a partial leading timestamp (`[5:4`).
  - `finish() -> list[str]` — `flush()` and end the stream; `feed` after
    `finish` raises `RuntimeError`.
  - `discard() -> None` — `context_reset`: drop the uncommitted buffer;
    already-committed lines stay committed; the next text starts a fresh
    entry (paragraph break if anything was shown; timestamp rule re-armed).
  - `has_text: bool` (property) — True once any non-blank line has been
    returned by `feed`/`flush`/`finish`.
  - Blank-line rules (implemented by holding back blank lines, since emitted
    lines cannot be retracted): never emit leading blanks of the reply; collapse
    runs of blanks to one; never emit trailing blanks.
  - Leading ephemeral timestamp `^\[\d{1,2}:\d{2}[ap]m\]\s*` (case-insensitive)
    is stripped from the first display text of each entry.
  - Lines are `rstrip()`ed (tag removal leaves trailing spaces); leading
    whitespace is preserved (indented code).
- `display_lines(text: str) -> list[str]` — one-shot: a fresh `ReplyStream`,
  one `feed`, `finish`. Used for the `turn_complete.response` fallback and
  `proactive_message` content. One code path for both.
- Tag literals are built by string concatenation (see tui/AGENTS.md tag-literal
  hazard); verify on-disk bytes by execution.

### `tui/transcript.py` (pure; Rich `Text` only)

Every function returns one `rich.text.Text`; a block's leading blank line is
part of the Text (`"\n"` prefix) so spacing lives here only.
- `banner(endpoint: str, base_url: str) -> Text` — dim:
  `MIRA · <endpoint> (<base_url>) · Enter send · Alt+Enter newline · Ctrl+C stop/quit`.
- `you(text: str) -> Text` — `"\n"` + `You` (bold cyan) + `"\n"` + `sanitize(text)`
  in the default foreground (user text is shown as typed — no tag filtering).
- `mira_label() -> Text` — `"\n"` + `MIRA` (bold green).
- `mira_lines(lines: list[str]) -> Text` — lines joined by `"\n"`, default fg.
- `tool_line(name: str, ok: bool) -> Text` — dim `· used <name>` / dim red
  `· <name> failed`.
- `reply_footer(seconds: float, tools: int, stopped: bool) -> Text` — dim:
  `· 14s · 2 tools` (`· 1 tool`; tools part omitted at 0); stopped:
  `· stopped after 14s`. Seconds rounded to whole seconds (`0s` allowed).
- `notice(text: str) -> Text` — dim. `alert(text: str) -> Text` — red.
- All text passes through `text.sanitize` before entering a Text.

### `tui/screen.py` (owns prompt_toolkit and every byte written to the terminal)

```python
@dataclass(frozen=True)
class Status:
    text: str
    tone: Literal["idle", "busy", "alert"] = "idle"
    since: float | None = None   # time.monotonic() start; busy → spinner + live "· 14s"

@dataclass(frozen=True)
class Live:
    pending: str = ""                 # ReplyStream.pending()
    sending: str | None = None        # message sent, not yet accepted (turn_started)
    queued: tuple[str, ...] = ()      # waiting behind the current reply, oldest first
    status: Status = Status("")

@dataclass(frozen=True)
class Submit:  text: str   # raw box text at Enter time (may be empty/blank)
@dataclass(frozen=True)
class Interrupt: pass      # Ctrl+C
@dataclass(frozen=True)
class Quit: pass           # Ctrl+D on an empty box
Intent = Submit | Interrupt | Quit

class Screen:
    def __init__(self, inbox: asyncio.Queue) -> None   # posts Intents with put_nowait; QueueFull → terminal bell, box untouched
    async def emit(self, *blocks: Text, live: Live | None = None) -> None  # print blocks to scrollback (+ apply live) in ONE print step; returns after written
    def set_live(self, live: Live) -> None             # live region only; invalidate
    def consume_input(self, text: str) -> bool         # compare-and-clear: box starts with text → remove that prefix, True; else False, box untouched
    def restore_input(self, text: str) -> None         # box = text (+ "\n\n" + old box if non-empty); cursor at end
    def clear_input(self) -> bool                      # box non-empty → clear, True; empty → False
    async def run(self) -> None                        # push-to-bottom, run the Application until close(); patch_stdout(raw=True) inside
    def close(self) -> None                            # exit the Application (erase_when_done → bar removed, scrollback ends clean)
```
- Layout (HSplit top→bottom): flexible filler `Window()` · pending Window
  (wrap, `[SetCursorPosition]` tail-follow, height ≤ cap, hidden when empty) ·
  outbox Window (`sending: …` / `queued: …`, dim cyan, one row each, first line
  + `…`) · status Window (1 row; dim; red when `alert`; busy → spinner +
  elapsed from `since`) · rule `─` · TextArea (wrap, grows with content up to a
  cap, scrolls inside beyond it) · rule `─`.
- Keys: Enter → `Submit(box text)` (box NOT cleared — ChatSession calls
  `consume_input`); Alt+Enter (Esc Enter) and Ctrl+J → newline; Ctrl+C →
  `Interrupt`; Ctrl+D → `Quit` when box empty (else default delete-char);
  bracketed paste inserts as-is.
- Printing: Rich `Console(file=StringIO, force_terminal=True,
  color_system="standard", markup=False, emoji=False, highlight=False,
  soft_wrap=True)` renders each block to ANSI (capture); the print step writes
  it via `app.output`. Before `run()` starts / after it ends, `emit` writes the
  same bytes to the real stdout directly (no app → nothing to erase).
- `Application(full_screen=False, mouse_support=False, refresh_interval≈0.5,
  erase_when_done=True)`.

### `tui/chat.py` — `ChatSession(client, screen, inbox, endpoint_label, base_url).run() -> int`

- Precondition: `client.connect()` already succeeded (its `AuthOk` is on the
  inbox). `run()` emits the banner, starts `screen.run()` and `client.run()`
  as tasks, and is the single consumer of the inbox. Returns 0 on user quit.
- State: `outbox: deque[str]`; `in_flight: (message_id, text) | None`;
  `turn: (turn_id, ReplyStream, started, tools, halt_requested) | None`;
  `halt_on_start: bool`; `deferred_proactive: list[str]`; connection state
  `connected | connecting | disconnected`.
- `connect()` reports each failure twice (AuthFailed event AND ClientError,
  and some paths raise without the event). The reconnect task wrapper catches
  `ClientError` and posts a private `ConnectFailed(error)` item; `AuthFailed`
  events are ignored (documented in code). On success the wrapper starts the
  pump task; state flips on the `AuthOk` event.
- Event handling, in brief (full table in the agent brief): Submit → `/exit`
  quits; blank → reconnect when disconnected, else ignored; else
  `consume_input` (False → ignore) then send now (idle) / queue (busy) /
  queue + reconnect (disconnected). TurnStarted(matching in_flight) → emit You
  block + MIRA label in one step. Deltas → `feed` → emit lines / set_live
  pending. Tool frames → `flush` + status `running <tool>…`; completed/error →
  tool line. ContextReset → `discard` + notice. TurnComplete/Stopped/Error →
  `finish`, response fallback (only if `not has_text`), footer, then
  deferred proactive, then next queued (except after a user halt: queued →
  box). ProtocolError matching in_flight (by `message_id`, or
  `INVALID_MESSAGE`/`TURN_SETUP_FAILED` while in_flight and no turn) → all
  unsent back to box + red line. Disconnected/ServerShutdown → finish any
  turn, red line, all unsent back to box, status `disconnected — Enter to
  reconnect`. Interrupt → halt (turn) / halt-on-start (in_flight) / cancel
  connect / clear box / quit; second Interrupt while stopping → quit. On quit
  with unsent messages, print them as a `not sent:` notice after the bar is
  gone.

#### ChatSession event table (authoritative)

`Live` is **derived on read** from state by one `_live()` method — never
incrementally maintained: `pending = turn.stream.pending()` (or `""`),
`sending = in_flight.text`, `queued = tuple(outbox)`, `status` per this list:
idle → `ready` (idle); in_flight → `sending…` (busy, since send);
turn → `MIRA is replying… · Ctrl+C to stop` / `MIRA is thinking… · Ctrl+C to stop`
(after a Thinking frame, until the next delta) / `running <tool>… · Ctrl+C to stop`
(busy, since turn start); halt requested → `stopping…` (busy); connecting →
`connecting…` (busy); disconnected → `disconnected — press Enter to reconnect`
(alert). "restore unsent" = `screen.restore_input("\n\n".join([in_flight.text] + outbox))`,
then clear both. Every `ClientError` from `send_message`/`send_halt` → red line
with its message (+ restore unsent for sends) — never swallowed.

| Item | Condition | Action |
|---|---|---|
| `AuthOk` | — | connected; if outbox and not busy → send next |
| `AuthFailed` | — | ignored (`ConnectFailed` covers every connect failure) |
| `ConnectFailed(error)` (private) | — | disconnected; red `could not connect: <message> [<code>]` (+ `python3 -m tui --login` hint on `AUTH_FAILED`); restore unsent |
| `Submit(raw)` | `raw.strip() == "/exit"` | quit (only command; any other `/text` is a message) |
| | blank | disconnected → reconnect; else ignore |
| | `consume_input(raw)` is False | ignore (duplicate Enter) |
| | disconnected | outbox.append; reconnect |
| | connecting / busy | outbox.append |
| | idle | send |
| `TurnStarted` | `message_id == in_flight.message_id` | emit You block + MIRA label in ONE emit; turn starts; in_flight cleared; if `halt_on_start` → `send_halt(turn_id)` |
| | otherwise | red protocol line |
| `AssistantDelta` | current turn | `feed` → lines → `emit(mira_lines)` else `set_live` |
| `Thinking` | current turn | status thinking |
| `ToolUpdate` | `tool_detected`/`tool_executing` | `flush` → emit lines; status running tool |
| | `tool_completed` / `tool_error` | emit `tool_line(name, ok = event == completed and not is_error)`; count unique `tool_id` |
| `ModelError` | — | notice `model error: <message> — MIRA is retrying` |
| `ContextReset` | current turn | `discard`; notice (says the text above was discarded only if anything had been shown) |
| `TurnComplete` | current turn | `finish` → lines; if `not has_text` → `display_lines(response)`; nothing at all → notice `(empty reply)`; footer(`processing_time_ms`/1000, tools); after-reply |
| `TurnStopped` | current turn | `finish`; footer(local elapsed, tools, stopped=True); after-reply(user_halt = halt requested) |
| `TurnError` | current turn | `finish`; red `MIRA hit an error: <message> [<code>]`; after-reply |
| after-reply | — | emit deferred proactive; user_halt and outbox → restore unsent + notice `queued messages returned to the input box`; else outbox → send next |
| `Proactive` | busy | defer |
| | idle | emit MIRA label + `display_lines(content)` |
| `ProtocolError` | `message_id == in_flight.message_id`, or code ∈ {`INVALID_MESSAGE`, `TURN_SETUP_FAILED`} while in_flight and no turn | red `not sent — <message> [<code>]` (`TURN_BUSY`: say MIRA is busy with another turn); restore unsent; `halt_on_start = False` |
| | `NO_MATCHING_ACTIVE_TURN` | notice `nothing to stop — the reply had already ended` |
| | other | red `protocol error: <message> [<code>]` |
| `ServerShutdown` | — | red line; `client.close()`; lost-connection |
| `Disconnected` | — | red line; lost-connection |
| lost-connection | — | turn → `finish`, emit lines + notice `reply cut off`; in_flight → note `connection lost before MIRA confirmed this message — check history before resending`; restore unsent; disconnected; emit deferred proactive |
| `Interrupt` | turn, not halting | `send_halt(turn_id)`; halting |
| | turn halting, or in_flight with `halt_on_start` | quit |
| | in_flight | `halt_on_start = True` |
| | connecting | cancel the connect task; disconnected; restore unsent |
| | else | `screen.clear_input()` False → quit |
| `Quit` | — | quit |
| `ScreenExited(exc)` (private) | — | the screen task ended without `close()` → re-raise (fail loud) |

- Reconnect: `client.close()` + await the previous pump task (bounded) before
  `connect()` — the old pump's `finally` sets `client._conn = None` and must
  not race a new connection. The connect wrapper task posts `ConnectFailed`
  on `ClientError`; on success it starts the new pump task.
- Quit: `screen.close()` → await the screen task (bar erased) →
  `client.close()` → await the pump (bounded) → if anything is unsent, emit
  `notice("not sent: …")` per message (Screen no longer running → direct
  write) → return 0.

### `tui/app.py` — `run(store) -> int`, `setup_guidance(store) -> str`

- No usable endpoint → setup guidance (carry `_SETUP_GUIDE` over, minus the
  REST-chat wording), exit 1. stdin/stdout not a TTY → exit 1 with message.
- `asyncio.run`: inbox `asyncio.Queue(maxsize=1024)`; `MiraClient`;
  `await client.connect()` BEFORE the UI — `ClientError` → red error + guidance
  keyed on code (`AUTH_FAILED` → `python3 -m tui --login`;
  `AUTH_CONNECTION_FAILED`/`AUTH_TIMEOUT` → check instance/base_url; else the
  message) → exit 1; Ctrl+C during connect → 130. Then `Screen`,
  `ChatSession(...).run()`.

## Deliverables

- New: `tui/text.py`, `tui/screen.py`, `tui/transcript.py`, `tui/chat.py`.
- Rewritten: `tui/app.py`.
- Deleted: `tui/render.py`.
- Updated: `tui/__main__.py` (docstring/description), `tui/__init__.py`
  (docstring), `tui/requirements.txt` (+`prompt_toolkit==3.0.51` with rationale
  comment), `tui/AGENTS.md` (rewrite Rules/Files/Wiring; list this plan file or
  delete it at the end), root `AGENTS.md` map-registry row for `tui/`,
  `cns/api/AGENTS.md` (`websocket_chat.py` bullet says "The browser client that
  consumed these frames was removed; no in-tree client documents the client side
  today." → replace with one-line citation of `tui/AGENTS.md` as client-side
  owner).
- Unchanged: `client.py`, `protocol.py`, `endpoints.py`, `login.py`.

## Verification (Tier 2 — new behavioral surface)

Reload the `writing-probes` skill before writing/running probes. Probes are
disposable: live in the scratchpad, never in the repo tree.

1. **text.py, pure execution:** replay replies split at every boundary (through
   tags, multi-line tags, escapes, entries, newlines, `[5:47pm]`); assert
   committed+pending == whole-text filter result; no tag fragment / ESC ever
   emitted; controls (ordinary `<` text like `a < b` must display).
2. **screen.py in a real PTY:** scratch venv `--system-site-packages` + `pip
   install pyte` (scratch only, not a project dep). Spawn the real app/Screen
   in a pty, answer CPR (`\x1b[6n` → `ESC[row;colR`) via pyte's
   `write_process_input`, feed keystrokes; read pyte `HistoryScreen` (screen +
   scrollback). Assert every message appears exactly once across: streamed
   lines, long wrapped lines, input grow/shrink, resize mid-stream, queue
   flush. Driving `Screen` directly is legal (it's the subject's own contract,
   no server fake).
3. **Live against the VM** (through the SSH tunnel, scratch store): send +
   stream, tool-call reply, halt, Enter-during-reply queue flush, disconnect +
   reconnect (break network/stop service), bad key at startup, TURN_BUSY if
   reproducible.
4. Second agent re-derives the diff (root AGENTS.md Tier 2 rule).
5. Report EXECUTED / UNVERIFIED per path. `py_compile` + pyflakes on all touched
   files.
6. Before any commit: read `docs/AGENTS_MAP_SPEC.md`, run its path-anchor and
   reciprocity audits.

## Orientation — read these (in this order) after compaction

1. This file.
2. `tui/AGENTS.md` — current (pre-rebuild) map; rules about tag-literal hazard,
   bounded waits, sentinel detection still apply to retained modules.
3. `tui/client.py` — `MiraClient` API the frontend drives: `connect()` (puts
   `AuthOk`/`AuthFailed` on the queue AND raises `ClientError` on failure — handle
   once, don't double-report), `run()` (pump; `Disconnected` only on unexpected
   close; quiet after `close()`/`server_shutdown`), `send_message(content) ->
   message_id`, `send_halt(turn_id=None)` (raises `NO_ACTIVE_TURN` if no turn
   started), `close()`; event dataclasses `AuthOk … Disconnected`.
4. `tui/protocol.py` — frame shapes (reference only).
5. `tui/render.py` — source of the filter regexes to port into `text.py`
   (`_THINK_BLOCK_RE`, `_MIRA_TAG_RE`, `_EPHEMERAL_TIMESTAMP_RE`,
   `_INCOMPLETE_TAG_RES`), then delete.
6. `tui/app.py` — `_SETUP_GUIDE` text + exit-code conventions to carry over;
   then rewrite.
7. `tui/__main__.py`, `tui/endpoints.py` (`store.load()` must be explicit;
   `EndpointConfig.include_thinking`).
8. `cns/api/websocket_chat.py` — server side of the protocol (frames L100–325,
   dispatch L703–752, turn lifecycle L761–939).
9. `cns/services/orchestrator.py` L978–1030 (text/tool stream events) and
   L1266–1330 (context_reset + post-stream response fallback).
10. prompt_toolkit source at
    `/opt/homebrew/Caskroom/miniconda/base/envs/mira/lib/python3.12/site-packages/prompt_toolkit/`
    — `application/run_in_terminal.py`, `renderer.py` (~L526–545, L630–652),
    `layout/controls.py` (`[SetCursorPosition]` ~L269/L389), `widgets/base.py`
    (`TextArea`), `patch_stdout.py`.
11. Root `AGENTS.md` (verification tiers, fail-loud, bounded waits, git rules)
    and `docs/AGENTS_MAP_SPEC.md` (map shape + audits).
