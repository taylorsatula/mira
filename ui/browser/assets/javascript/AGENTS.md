# ui/browser/assets/javascript/ — browser client modules: the network layer, the chat page's turn rendering, and shared page helpers

## Rules

- Classic `<script>` files, no modules: every top-level `const`/`let`/`function`/`class` of every script on a page shares one global lexical scope, and modules call each other through those names or explicit `window.*` exports. A top-level name declared in two scripts on the same page is a SyntaxError that kills the later script whole. New files wrap their body in an IIFE and export through `window`, as `api-client.js`, `sidebar.js`, `location.js`, `theme.js` and `repulsion-feedback.js` do.
- `api-client.js:MiraAPIClient._httpRequest` is the one HTTP path for every page and module; a raw `fetch` to `/v0/...` elsewhere is a defect. It sends `credentials:'include'`, adds `X-CSRF-Token` to non-GET/HEAD requests under `/v0/api` and `/v0/auth` (except `POST /v0/auth/csrf` itself), resolves to the envelope's `data` (paged types carry their paging block inside it: `data.meta`), and rejects `MiraHttpError` (`.status`, `.code` = envelope `error.code`). The server holds one CSRF token per session and every `POST /v0/auth/csrf` replaces it, so a second tab's fetch invalidates this tab's copy: `_ensureCsrfToken` keeps one fetch in flight per tab, and a 403 `CSRF_INVALID`/`CSRF_REQUIRED` refetches and retries exactly once (the retry-site comment names the server alternative, get-or-create in `auth/session.py:generate_csrf_token`). A 401 runs `auth.clearToken()` → `onAuthChange({type:'logout'})` → the page navigates to `/`.
- One send in flight: `chat.sendMessage` rejects `SEND_IN_FLIGHT` while `_send` is set, and every turn frame is matched to it by `message_id` (`turn_started`) then `turn_id`; a frame for any other turn is logged and dropped. The promise settles exactly once: `{status:'complete'}`, `{status:'stopped', reason: halt|disconnect|stall}`, or a `MiraTurnError` whose `.code` is the server code or a client code (`SEND_IN_FLIGHT`, `EMPTY_MESSAGE`, `MESSAGE_TOO_LONG`, `NOT_AUTHENTICATED`, `CONNECTION_LOST`, `TURN_BUSY` after 600 s). `MAX_CONTENT_LENGTH` equals `cns/api/websocket_chat.py:MAX_CONTENT_LENGTH` (100 000): the server answers an over-length frame with an unattributed `MALFORMED_FRAME`, which would leave the send unsettled, so the client must reject first — change both together.
- `onMessage` listener ownership: `core.js:initializeAPIClient` owns the page-wide listeners for `tool` (→ `ui.js:updateToolIndicator`) and `proactive_message` (→ `messaging.js:handleProactiveMessage`); `messaging.js:runTurn` pushes `handleTurnFrame` for one send and splices it out in `finally`. A second page-wide proactive listener renders each proactive message twice.
- Every page constructs its own `MiraAPIClient`, and the constructor calls `connect()`: settings, memories and domaindocs tabs each hold an authenticated chat WebSocket with a 30 s ping, though only the chat page listens for frames.
- Cross-client twins (`messaging.js:ReplyFilter` ↔ `ui/tui/text.py:ReplyStream`, backoff constants ↔ `ui/tui/chat.py`, `history.js` row grouping ↔ `ui/tui/history.py`) are owned by `ui/AGENTS.md`; the frame same-commit rule too.
- Tool badge icons — add one: (1) put `<stem>.png` in `ui/browser/assets/images/icons/toolicons/`, where the stem is the tool name minus `_tool` (`reminder_tool` maps to `reminders` in `ui.js:toolIconSrc`); (2) add the stem to `ui.js:TOOL_ICONS`. Skip (2) and the tool shows `default.png`; list a stem with no file and the badge is a broken image.
- `localStorage` keys in use (per-browser conveniences only, never server state): `mira-theme` (`theme.js` + the inline page bootstraps), `mira-queue` (Retry Send queue: `core.js`, `messaging.js`), `mira-sidebar-collapsed` (`sidebar.js`), `mira-location-consent` (`location.js`), `dismissed-announcement` (`messaging.js:showAnnouncement`). Pick a new `mira-` key for new state.

## Files

- `api-client.js` — `MiraAPIClient` (`auth`, `chat.sendMessage`/`chat.halt`, `history.getHistory({before, limit})` → `{messages, hasMore, nextBefore}` oldest→newest, `actions.executeAction`, `data.getData`, `eventHandlers`), `MiraHttpError`, `MiraTurnError`; the only `window` exports. Cookie auth: the first frame is `{type:'auth'}` with no token. Message ids come from `crypto.getRandomValues` (`randomUUID` is absent on plain-http LAN origins). Frame handling: `## Frame handling`.
- `core.js` — chat-page root: `AppState`, the `elements` DOM cache (read at script load — the script tags sit after the markup), `initializeAPIClient`, `initializeMira`, `Functions.collapseSegment` (`continuum` `collapse_segment`; `collapsed:false` shows the server's deferral message). `restoreInterruptedResponse` re-reads the last turn from history 2 s after a reconnect that follows a `disconnect` stop.
- `messaging.js` — the chat turn: Retry Send queue, attachments (images downscaled to 512 px unless "Preserve resolution"), `sendMessage` → `runTurn` (`beginTurn`, `handleTurnFrame`, `completeTurn`/`stopTurn`/`failTurn`, `endTurn`), `ReplyFilter`/`filterReplyText`, streaming block renderer (`marked` + `DOMPurify` with `MARKDOWN_SANITIZE_CONFIG`), thinking stream and drawer, `loadLastSessionResponse` (newest exchange from history), `showAnnouncement` (`data?type=user` `preferences.announcement`), proactive rendering. `streamingState.displayedExchange` is the exchange on screen; `history.js` reads it.
- `history.js` — `window.InlineHistory`: inline history above the reply, paging back with `before = nextBefore`. Leading reply rows whose user row is on the next older page are carried to it; the first page omits the exchange already on screen (`withoutDisplayedExchange`) and waits on `messaging.js:sessionRestored`.
- `ui.js` — `ASSET_VERSION`, `TOOL_ICONS`/`toolIconSrc`, tool-indicator state machine, send animations, loading-screen wave, status badges, attachment/queue popover state.
- `events.js` — `setupEventListeners`/`setupDelegatedEvents` (called by `core.js` after startup): send/stop, Enter-to-send on desktop only, image paste (downscaled to 540 px), popovers, queue actions.
- `sidebar.js` — every page: collapse/expand, mobile drawer, History button (`?history=1` opens inline history on `/chat`), and on the chat page Pause/Unpause and Wrap Up. Runs at load and returns silently when `#app-sidebar` is not yet in the DOM, so every page's `<script>` tags go after the sidebar markup.
- `inactivity-warning.js` — `window.inactivityWarningManager`: session status bar above the compose bar, polling `continuum` `get_segment_status` every 30 s; `pause_session`/`resume_session`; `onStatusChange` feeds the sidebar Pause button.
- `domain-knowledge.js` — `DomainKnowledgeManager` (the `domain_knowledge` action domain plus `data?type=domaindocs`; section actions take an optional `parent` header) and `initDomainKnowledge` (chat-page indicator and popover; sets `window.domainManager`). Also loaded by `ui/browser/domaindocs/index.html`.
- `tool-config.js` — `ToolConfigManager` for the settings Tools tab: `/v0/api/actions/tools` list/get/schema/PUT/validate/DELETE, forms generated from the JSON Schema, `email_tool` folder discovery from the validate response's `discovered`.
- `location.js` — consent modal, then `POST /v0/api/location` with browser coordinates; later page loads query only when the permission is already granted. Needs the `geolocation=(self)` Permissions-Policy (`auth/security_middleware.py`).
- `repulsion-feedback.js` — `window.repulsionFeedback` (`noteUserMessage`, `scanResponse`): the `TELLS` regex bank over each completed reply; a hit is captured at once and again with the user's reason via `feedback` `capture_repulsion`.
- `collapse-animation.js` — `window.collapseAnimation` (font swap, then gravity fall) used by Wrap Up.
- `theme.js` — `window.miraSetTheme`/`miraGetTheme`; follows the OS preference while in Auto.
- `marked.min.js`, `purify.min.js` — vendored marked v12.0.0 and DOMPurify 3.0.9; chat page only; `messaging.js` calls `marked.setOptions` at load.

## Wiring

- Chat page load order (`ui/browser/chat/index.html`, a contract): inline theme bootstrap → `theme.js` → `marked.min.js`, `purify.min.js` → `api-client.js` → `core.js` → `ui.js`, `sidebar.js`, `messaging.js`, `history.js`, `events.js` → `domain-knowledge.js`, `collapse-animation.js`, `inactivity-warning.js`, `location.js`, `repulsion-feedback.js`. Load-time dependencies: `messaging.js` needs `marked`; `history.js` reads `core.js:elements`; `sidebar.js` subscribes to `inactivityWarningManager` on `DOMContentLoaded`, after the later script defined it. Other pages load `theme.js`, `api-client.js` and `sidebar.js` plus `tool-config.js` (settings) or `domain-knowledge.js` (domaindocs).
- Chat startup (`DOMContentLoaded` → `core.js:initializeMira`): `new MiraAPIClient()` (connects) → `auth.validateToken()` (`GET /v0/auth/session`; false = logout already navigating) → `initDomainKnowledge` → `updateQueueIndicator` → `loadLastSessionResponse` (resolves `sessionRestored`) → `setupEventListeners`/`setupDelegatedEvents`. Input handlers are wired last, so nothing can send before the restore decides what the response box shows.

## Frame handling

| Server frame | `api-client.js` | Chat page effect |
|---|---|---|
| `auth_success` | reset reconnect backoff, start ping keepalive, emit `onConnect`, transmit a waiting send | `core.js` refreshes the queue indicator; restores an interrupted reply |
| `pong` | clears the pending-ping flag; a ping unanswered for 30 s closes the socket (→ reconnect) | — |
| `server_shutdown` | logged; the server's close triggers reconnect | — |
| `protocol_error` `AUTH_FAILED`/`AUTH_REQUIRED`/`AUTH_TIMEOUT` | `auth.clearToken()`, logout event, no reconnect | navigate to `/` |
| `protocol_error` `AUTH_UNAVAILABLE` | logged; reconnect with backoff; never `clearToken` | — |
| `protocol_error` `TURN_BUSY` for the in-flight `message_id` | resend the same frame after 0.5/1/2 s, then every 3 s, up to 600 s from the first bounce; a halt meanwhile resolves `stopped`/`halt` | send button stays in stop mode |
| other `protocol_error` attributed to the send (its `message_id`, or `INVALID_MESSAGE`/`TURN_SETUP_FAILED` after transmit) | reject with its code | `failTurn` notice |
| `protocol_error` unattributed | `NO_MATCHING_ACTIVE_TURN` warned (halt raced the end); others logged | — |
| `turn_started` | binds `turn_id`; sends a halt requested before it arrived | `handleTurnFrame` binds the turn |
| `assistant_delta` | appends to `streamedText` | `ReplyFilter.feed(entry_id, content)` → block render |
| `thinking` | forwarded | live thinking text, folded into a drawer when text starts |
| `tool` | forwarded | `core.js` → tool badge; `messaging.js` flushes the filter and records a tool pill on completion/error |
| `context_reset` | clears `streamedText` | new `ReplyFilter`, response box emptied |
| `model_error` | forwarded | alert notice, removed when the turn settles |
| `turn_complete` | resolves `complete` | `response` rendered only when the filter committed no text; `domaindoc_tool` use refreshes the domain list |
| `turn_stopped` | resolves `stopped` with `reason` | `stall` → error notice; `disconnect` → restore from history after reconnect |
| `turn_error` | rejects with its code | `TURN_BUSY`/`CONNECTION_LOST`/`NOT_AUTHENTICATED` → message to Retry Send; else error notice |
| `proactive_message` | forwarded outside any send | shown below the reply; held until the turn settles while generating |
| socket close | after `turn_started` → resolves `stopped`/`disconnect`; transmitted but unstarted → rejects `CONNECTION_LOST`; untransmitted → sent after reconnect (≤600 s) | per the rows above |
