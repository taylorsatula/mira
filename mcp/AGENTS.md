# mcp/ — agent-facing check-in clients for a deployed MIRA instance

## Rules

- Twin contract with `tui/endpoints.py`: `talkto_mira.ts` reads the TUI endpoint store (`~/.config/mira-tui/config.json`, `active` endpoint's `base_url` + `api_key`). Any store-format change updates `endpoints.py` and `talkto_mira.ts` in the same commit — a format drift surfaces only when the extension throws at call time.
- `talkto_mira.ts` is the pi-native face of the same behavioral contract the server-side endpoint serves: one tool, one complete chat turn over `POST /v0/api/chat`, no turn logic client-side. The server-side twin of every mapping row lives in `cns/api/mcp.py:_check_in` — change a mapping row in both places in the same commit, and keep the busy-anchor ("already in progress", from `cns/api/chat.py`'s ValidationError message) identical in both. The tool's model-facing `DESCRIPTION` is byte-identical to `cns/api/mcp.py:_CHECK_IN_DESCRIPTION` — edit both in one commit.
- The extension needs no `mcp_enabled` flag (it talks the always-on REST surface); the `/v0/mcp` endpoint is for non-pi MCP clients. Supersession is deliberate: for pi, `talkto_mira` replaces the MCP path.

## Files

- `talkto_mira.ts` — pi extension registering the model-callable `talkto_mira` tool (`pi.registerTool`, TypeBox schema: `message` required; `image`/`image_type`, `document`/`document_type` optional pairs). Agent-to-MIRA, not a human frontend: the model drives it; `details` carries the turn metadata (`baseUrl`, `continuumId`, `metadata`) for the human's result inspector. Failure rows (busy, timeout, auth, unreachable) are thrown so pi renders a failed tool result carrying the message. Installed by symlink into pi's global extension dir (`~/.pi/agent/extensions/`); credentials from the TUI endpoint store (twin contract above). `READ_TIMEOUT_MS` (600 s) mirrors `cns/api/mcp.py:_CHAT_READ_TIMEOUT_S` — a bound that covers every realistic turn while the server's lock TTL covers the structural worst case.

## Wiring

- Server-side endpoint (the thing this directory's tool shadows): `cns/api/mcp.py`, mounted at `/v0/mcp` by `main.py` only when `config.system.mcp_enabled` (`MIRA_MCP_ENABLED=1`); its contract is owned by `cns/api/AGENTS.md` and `mcp_enabled` by `config/AGENTS.md`.
