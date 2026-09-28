# tui/ — Minimal terminal chat client over the deployed REST/WS API

Standalone client (`python -m tui`) for a deployed MIRA instance, rebuilt
brick-by-brick from 2026-09-19. Current brick: a bare **synchronous REPL** —
alternating cyan/green You/MIRA text blocks, a dark-grey delimiter row
between turns, a plain `input()` line. No prompt_toolkit, no Rich panels, no
streaming, no reconnect machinery. The turn wire is sync REST
(`POST /v0/api/chat`, owned by `cns/api/chat.py`). Imports NOTHING from the
server tree; a pure client, the only in-tree client surface.

The WebSocket stack (`client.py` + `protocol.py`) is **retained, currently
unused by the REPL** — it is the verified foundation for the later streaming
brick. Do not delete it and do not half-wire it into `app.py` until that
brick lands.

## Rules

- The active wire is sync REST: `app.py:_post_chat` mirrors the
  `ChatRequest`/envelope shape of `cns/api/chat.py` (`{success, data|error}`,
  response text under `data.response`). When that handler changes,
  `_post_chat` changes in the same commit.
- `protocol.py` remains the strict (`extra="forbid"`) mirror of the WS frames
  of `cns/api/websocket_chat.py` and the history envelope of
  `cns/api/data.py`, and `client.py` remains its event-pump consumer — the
  pre-rebuild drift rule (mirror + pump + history models change together
  with the server handler) still binds them for the streaming brick.
- Screen output is ONLY `app.py`'s ANSI helpers (`_color`, `_delimiter`,
  `_block`, `_clear_last_line`) — no rich bridge, no hand-rolled codes
  anywhere else. The clear-line trick assumes single-visual-line rows: a
  wrapped/pasted multi-line input leaves debris above the cleared row
  (accepted MVP limit).
- Message text — both directions — passes through
  `render.py:filter_system_tags` before display; display filters stay owned
  by `render.py` (ported from the removed web client's filters).
- Tag-literal hazard: literal think/mira tag strings written through agent
  tool payloads get HTML-entity-mangled on disk.
  `render.py:_THINK_BLOCK_RE` builds its pattern by concatenation — any new
  code needing those literals must do the same and verify on-disk bytes by
  execution.
- Settings have NO UI — the JSON store (`endpoints.py`) plus
  `--config-debug` is the interface. `/exit` is the only command the REPL
  knows. Do not add interactive configuration.
- Segment sentinels in history: detection goes through
  `client.py:_is_sentinel` (server may send `is_segment_boundary` as JSON
  bool or string). A collapsed sentinel's `content` IS the session summary.
- Every await in the retained WS stack is bounded (constants at the top of
  `client.py`); the REPL's one HTTP call carries `_CHAT_TIMEOUT`. No
  unbounded waits anywhere.

## Files

- `__main__.py` — CLI entry `main()`: `--config PATH` store override, `--config-debug` (prints store layout, NEVER the api_key), `--login [--endpoint NAME | --base-url URL [--save-as NAME]]` (headless token mint via `login.py`, exits before the chat app); exit 0 on `/exit`, 1 on fatal store errors and first-run-no-endpoint.
- `app.py` — the minimal REPL (`run()`): prompt → clear echoed line → colored block → grey delimiter → dim `thinking…` indicator → sync `POST /v0/api/chat` → clear indicator → MIRA block → delimiter. Errors print red with the real server message; Ctrl+C at the prompt exits 0, Ctrl+C during a request drops it client-side with a notice (the server turn may still complete and land in history). `setup_guidance` prints the config template when no usable endpoint exists.
- `client.py` — RETAINED, unused by the REPL: `MiraClient` (WS auth + frame pump, REST history pager), the `ClientEvent` dataclass union, `ClientError` (sole exception type). Pure asyncio, bounded waits. Foundation of the streaming brick.
- `protocol.py` — RETAINED, unused by the REPL: strict pydantic v2 mirror of all WS frames + `parse_inbound_frame` / `dump_outbound_frame` + REST history models. The WS drift anchor cited above.
- `endpoints.py` — `EndpointConfig`, `EndpointStore` (0600 JSON at `~/.config/mira-tui/config.json`), `HISTORY_FETCH_MODES`. Gotcha: the constructor does NOT auto-read — callers must `store.load()` explicitly.
- `login.py` — headless token bootstrap: `mint_api_token()` chains `GET /v0/auth/local/session` (single-mode: zero-credential; 404 → magic-link flow with email + pasted link token) → `POST /v0/auth/csrf` → `POST /v0/auth/api-tokens` (`x-csrf-token` header), mirroring `auth/api.py`. Minted token goes straight into the 0600 store and is never printed (only the can't-write-store last resort prints it). Token names retry with `-2`/`-3` suffixes on the server's `duplicate_token_name`. Live multi-mode flow is code-verified only.
- `render.py` — display filters (`filter_system_tags` / `filter_streaming_text` / `summarize_tool_result` / `format_content_blocks` / `user_content_text`, ported from the removed web client's filters) — used by the REPL via `filter_system_tags` — plus the retired Rich block renderers, `ActiveTurn` preview state, and `history_blocks` mapping, retained for the history/streaming bricks. No prompt_toolkit imports.
- `requirements.txt` — the client's own pin set (websockets, httpx, pydantic, rich); prompt_toolkit was removed with the old UI. The server's `requirements.txt` gains nothing from this package.
- `BUILD_PLAN.md` — design record of the 2026-09-18 Textual build and the post-build pivot to terminal-native rendering; historical, not a living contract.

## Wiring

- Startup: `__main__.main()` → `store.load()` → (no history fetch in this brick) → `app.run()` prints the dim status line and enters the REPL loop.
- Per turn: `input()` → `_clear_last_line()` (the typed echo must not duplicate the user block — the emitted block is the ONLY copy) → user block → delimiter → `thinking…` → `_post_chat` → MIRA block → delimiter. The server enforces one active turn per user (`UserRequestLock` in `cns/api/chat.py`); a second concurrent request answers 400 — surfaced red by the REPL.
- Later bricks, in dependency order: history rendering on startup (`client.fetch_history` + `render.history_blocks`), streaming preview over WS (`client.py` + `protocol.py` + an input-preserving rendering strategy), reconnect.
