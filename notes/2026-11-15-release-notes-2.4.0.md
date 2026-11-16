# MIRA v2026.11.15-2.4.0

A feature release: an optional shared-secret gate for anyone reaching MIRA
from outside the machine, and a first-run experience that configures the
local install by itself. No database steps, no updater changes.

## Remote access gate (new, opt-in)

Single-user MIRA auto-provisions a browser session for any peer that can
reach port 1993. That is the right default on loopback and the wrong one on
a LAN or tailnet. The gate closes that gap: when Vault holds
`secret/mira/gate` (field `token`), every non-loopback HTTP and WebSocket
request must carry a `mira_gate` cookie derived from that token, or it is
refused — a 401 with code `REMOTE_GATE_REQUIRED` for API clients, a redirect
to `/remote-gate` for browser navigations, and a WebSocket close (4401)
before accept. Loopback peers are never gated, so local tooling, the
installer's health checks, and the Docker healthcheck are unaffected.

- The cookie carries an HMAC-SHA256 of the token, not the token; it is
  HttpOnly, SameSite=Lax, and Secure when the request is HTTPS. Comparison is
  constant-time. The gate page and its submit endpoint are the only exempt
  paths; the submit endpoint is rate-limited per IP (10 attempts / 5 min)
  and logs `auth.remote_gate_*` security events.
- A Vault read failure fails closed (the gate engages with an unguessable
  token) rather than open.
- The installer asks once (question 11, default no — local installs see zero
  behavior change) and says plainly that anyone who might open MIRA from a
  phone should press yes. `--config` installs set `remote_gate: yes|no`.
  When enabled, the installer generates the secret, writes it to
  `secret/mira/gate`, and copies it into the credential dump; an existing
  secret is kept, never rotated.
- Enable or disable at any time: `vault kv put secret/mira/gate
  token="$(openssl rand -hex 24)"` / `vault kv delete secret/mira/gate`,
  then restart MIRA. See `docs/REMOTE_GATE.md`.

Verified end to end on a disposable Ubuntu 26.04 VM installed from this
tree: gate on, 401 without a cookie, gate page served, 200 with the cookie,
browser flow lands on `/v1/auth/local/session`, WebSocket refused without
the cookie. Also verified over Tailscale from a second machine.

## TUI

- First run now configures the local install automatically: with an empty
  endpoint store, `mira` mints a token against `http://localhost:1993` via
  the same chain as `--login`, stores it as the `local` endpoint, and drops
  you into chat — no manual `--login` step. A configured-but-inactive remote
  endpoint is never shadowed; the old guidance path remains for that case
  and for non-TTY invocations.
- Turn boxes now draw with double-line glyphs (`║`, `═`, `╚`/`╝`) and each
  turn opens with a blank line.

## For upgraders

`mira update`, as always, from v2026.10.16-2.1 or later. No database steps,
no `requirements.txt` changes (the server venv is not rebuilt), no
configuration migration. The gate is off unless you enable it.

## Verification

- Gate rows a–c verified at build time against the tagged asset:
  `build_release_artifact v2026.11.15-2.4.0` passed `scripts.migrate lint`
  (3 versioned and 1 repeatable step follow the rules), and
  `verify_release_identity` confirmed tag = v + VERSION, the asset's VERSION
  matches the repo, and the asset's `RELEASE_COMMIT` equals the tagged
  commit (562a342).
- Row c (convergence) ran on the build host against a local PostgreSQL 17
  superuser; `scripts.migrate generate` rewrote `deploy/migrations/BASELINE`
  and `HEAD_FINGERPRINT.json` byte-identically (no diff).
- Gate rows d (skip-chain) and e (restore round trip): not run. This
  release ships no database steps and no changes under `deploy/migrations/`
  (verified: no diff against v2026.11.15-2.3.2) and no updater changes.
  The rows carry their v2026.10.21-2.1.2 results unchanged.
