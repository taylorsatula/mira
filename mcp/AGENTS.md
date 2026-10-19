# mcp/ — the pi `talkto_mira` extension: one model-callable check-in turn against a deployed MIRA instance

## Files

- `talkto_mira.ts` — the pi extension that registers the `talkto_mira` tool. Open it before changing what the tool sends or reports back, or before touching the twin contract its header states.

## Wiring

- Check-in turn: pi's model → `execute` → `resolveEndpoint` → `POST {base_url}/v1/api/chat` → `ChatEndpoint.process_request` → the `{success, data}` envelope → `execute`'s error rows. `resolveEndpoint` runs before any request exists, so a missing or unreadable store ends the call with nothing sent and nothing persisted; a request-body key or envelope-shape change lands in this file in the same commit as `cns/api/chat.py`. Counterpart: `cns/api/AGENTS.md`.
- Server-side twin: `cns/api/mcp.py:_check_in` serves the same turn to non-pi MCP clients. That mount is gated on `config.system.mcp_enabled`; this path needs no flag. What crosses is the tool's parameter set, the twin `DESCRIPTION`/`_CHECK_IN_DESCRIPTION` text, and the error rows — see the twin and busy-row Rules — while the gate, mount, and session-manager lifespan stay on the server side. Counterpart: `cns/api/AGENTS.md`.
- Endpoint store: `resolveEndpoint` reads `active`, `endpoints[active].base_url`, and `endpoints[active].api_key` from the store `ui/tui/endpoints.py` writes, and ignores the store's other keys. Two further literals are duplicated across the pair: the store path (`_DEFAULT_PATH` there, a `join(homedir(), ...)` here) and the `python3 -m ui.tui --login` remedy string all three of `resolveEndpoint`'s error rows name. A format change, a path move, or a `--login` command rename updates this file in the same commit, or the TUI keeps working while every check-in dies at `resolveEndpoint` — on an unreadable store, or on a missing file whose message points at the old location. Counterpart: `ui/tui/AGENTS.md`.

## Rules

- A check-in client is a rider, never an implementation: `talkto_mira.ts:execute` and `cns/api/mcp.py:_check_in` authenticate, validate, lock, and commit nothing themselves — they hand a `POST /v1/api/chat` to the real endpoint, which does all of that and then runs the post-commit memory extraction. Turn behavior added inside a check-in client (retries, prompt preparation, memory handling) forks the one turn path, and a check-in stops being the turn the WS and TUI clients get.
- The model-facing text is a byte-identical twin: `DESCRIPTION` ≡ `cns/api/mcp.py:_CHECK_IN_DESCRIPTION`, with `READ_TIMEOUT_MS` paired against `_CHAT_READ_TIMEOUT_S`. Neither timeout bounds a legal turn — the server's per-user lock TTL derives from the per-turn tool-call budget in `cns/api/chat.py`, which no fixed client timeout can cover — so both state the same realistic-turn ceiling. `execute`'s error rows pair with `_check_in`'s `ToolError` rows too, except the abort row, which is pi-only. Any of these changes in both files in one commit, or the two surfaces teach different tools to the models reaching them.
- The description's attachment literals are calibrated against server validators: the image set and size cap in `cns/api/chat.py:SUPPORTED_IMAGE_FORMATS`/`MAX_IMAGE_SIZE_MB`, the document set and cap in `utils/document_processing.py:SUPPORTED_DOCUMENT_FORMATS`/`MAX_DOCUMENT_SIZE_MB`, and the one-image-or-one-document rule in `ChatRequest.validate_attachment_exclusion`. A limit change re-derives the description in both twins, or the tool teaches attachments the server rejects with a 400 after the caller has already assembled the payload.
- The busy row is the one transcription of a server *constant*: `cns/api/mcp.py` imports `_BUSY_REJECTION_MESSAGE`, TypeScript cannot, so `execute` matches the literal substring `"already in progress"`. An edit to that constant keeps the substring — `rg 'already in progress'` before merging — or the busy row dies silently and the model gets the generic HTTP 400 row where wait-and-retry guidance belongs.
- MIRA's reply reaches the calling model untouched: it is MIRA's own composed turn output, not external content MIRA ingested, so the root's untrusted-content boundary governs what MIRA takes in and not what it emits. Screening or wrapping the reply double-treats MIRA's voice as hostile data.
- Nothing in this repo imports it (`rg 'talkto_mira'` hits only `mcp/talkto_mira.ts` and maps). pi reads `~/.pi/agent/extensions/` at startup, so editing the repo copy changes a running pi only once the file is placed or linked there and pi restarts — test against the deployed pi, not the repo.

Elsewhere: this map owns the client half of a check-in — how the tool finds its instance, what it sends, and what it reports back; it does not own the turn it calls. Where the description's "may be extracted into long-term memory" promise resolves — at segment collapse, after commit, never inside the call — belongs to `lt_memory/processing/`; pi's own extension API is out of tree, and this file's use of it is the documentation; how any of this was built is `git log`. Every other destination this directory reaches is an edge — see `## Wiring` above.

## What an abandoned check-in leaves behind

The case that decides the error rows and the absence of retry logic:

1. `execute` composes the abort signal from pi's signal and `READ_TIMEOUT_MS`, then sends the POST. By the time the server starts model work it has already acquired the per-user lock and staged the accepted message.
2. pi gives up — the user aborts, or the timeout fires. Only the client-side wait ends. Nothing on the server cancels the handler, so the turn keeps running.
3. The turn finishes on its own, commits the message and the reply into the continuum, and releases the lock. Everything a normal turn triggers post-commit still runs.
4. A retry landing inside that window reaches the endpoint while the lock is held and gets the busy rejection. A retry landing after it appends a second copy of the message to MIRA's permanent history.
5. That is why the timeout row says the turn may still be running and why the tool has no retry: the message is in MIRA's history either way, and a blind retry can make it appear twice.
