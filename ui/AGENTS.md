# ui/ — MIRA's two chat clients: the terminal client (`ui/tui/`) and the browser client (`ui/browser/`)

## Rules

- Both clients speak the WebSocket turn protocol whose server side is owned by `cns/api/AGENTS.md` (`## WebSocket turn protocol (server side)`). A server frame change in `cns/api/websocket_chat.py` (a new frame type, a renamed or added field, a new `protocol_error` code a client must act on) updates `ui/tui/protocol.py` (+ `ui/tui/client.py:_frame_to_event`) AND `ui/browser/assets/javascript/api-client.js` (`_handleFrame`, `TURN_STREAM_FRAMES`) in the same commit. Drift symptoms differ: the TUI's strict models raise `ProtocolError`/`UNPARSEABLE_FRAME`; the browser logs `Unknown server frame type` and drops the frame — a dropped terminal frame leaves the send promise unsettled and the page stuck in its generating state.
- Twin code across the two clients, same-commit change required (each map owns its file; the pairing is owned here):
  - Reply display filter: `ui/browser/assets/javascript/messaging.js:ReplyFilter` is a port of `ui/tui/text.py:ReplyStream` (think blocks, `mira:` paired/self-closing tags split across deltas, hold-back of possible tag fragments, interior-only blank lines, the leading `[5:47pm]` stamp, the per-`entry_id` paragraph break). Drift shows as one client printing tag text or stamps the other hides.
  - TURN_BUSY and reconnect backoff: `api-client.js` `TURN_BUSY_*`/`RECONNECT_*` constants equal `ui/tui/chat.py` `TURN_BUSY_RETRY_DELAYS`/`TURN_BUSY_POLL_SECONDS`/`TURN_BUSY_MAX_WAIT_SECONDS` and `AUTO_RECONNECT_DELAYS`/`AUTO_RECONNECT_POLL_SECONDS`.
  - History row presentation: `ui/browser/assets/javascript/history.js:renderHistoryRows` groups rows the way `ui/tui/history.py:replay_blocks` does (assistant + tool rows sharing `metadata.turn_id` form one reply; collapsed sentinels render as summary blocks; active/paused boundaries are skipped).
- `__init__.py` must stay import-free: `python3 -m ui.tui` imports the `ui` package first, inside the TUI venv (`/opt/mira/tui-venv`), which carries none of the server's dependencies — a server import here breaks the `mira` command on every install.
- Neither client imports server code, with one exception: `ui/tui/update.py` imports the stdlib-only `utils/release_identity.py`. The browser client is static files; `main.py` serves them (contract owned by `ui/browser/AGENTS.md`).

## Files

- `__init__.py` — docstring only, no re-exports; makes `ui` the package `-m ui.tui` resolves through (the deployed `mira` wrapper runs `python3 -P -m ui.tui`, owned by `deploy/AGENTS.md`).
- `tui/` — streaming WebSocket terminal client; map `ui/tui/AGENTS.md`.
- `browser/` — browser client served by `main.py`; map `ui/browser/AGENTS.md`.
