# MIRA v2026.11.15-2.4.0

A feature release: an optional shared-secret gate for anyone reaching MIRA
from outside the machine, a guided phone-access setup in the installer, and a
first-run experience that configures the local install by itself. No database
steps, no updater changes.

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
- The gate secret is four EFF large-wordlist words separated by hyphens
  (about 52 bits), chosen to be typable on a phone keyboard. Anyone wanting
  more entropy can set their own token — any string works.
- Behind a loopback reverse proxy such as `tailscale serve`, the gate no
  longer stands aside: a loopback peer that sets `X-Forwarded-For` is treated
  as proxied and the gate engages. The header is never parsed for the gate
  decision — its presence alone disqualifies the loopback exemption — so no
  header content can make a proxied request look local. The header is
  ignored from non-loopback peers.
- Enable or disable at any time: `vault kv put secret/mira/gate
  token="$(diceware -n 4 --no-caps -d '-')"` / `vault kv delete secret/mira/gate`,
  then restart MIRA. See `docs/REMOTE_GATE.md`.

## Phone access from the installer (new)

Question 12 (`tailscale: yes` in a `--config` file) walks a fresh install the
rest of the way to MIRA on your phone: it installs Tailscale on the machine,
signs it into your tailnet (you open a login link during the install), and
serves MIRA at `https://<machine>.<tailnet>.ts.net` with `tailscale serve`.
It implies the gate — without it, anyone on the tailnet would get full
single-user access. Your phone needs the free Tailscale app signed into the
same account; opening the URL and entering the gate secret once is all the
phone-side setup there is. The installer refuses to expose MIRA via
`tailscale serve` when the gate is not enabled, verifies after serving that
an unauthenticated request is refused (tearing the serve config down if it
is not), and leaves an existing non-MIRA `tailscale serve` configuration
untouched.

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

Re-released 2026-11-16: the tag was moved from 562a342 to the commit
carrying the phone-access installer flow and the gate fix, and the asset was
rebuilt from the moved tag.

- Gate rows a–b verified at build time against the tagged asset:
  `build_release_artifact v2026.11.15-2.4.0` passed `scripts.migrate lint`
  (3 versioned and 1 repeatable step follow the rules), and
  `verify_release_identity` confirmed tag = v + VERSION, the asset's VERSION
  matches the repo, and the asset's `RELEASE_COMMIT` equals the tagged
  commit.
- Row c (convergence) re-ran clean on a throwaway local PostgreSQL 17
  cluster (initdb, trust auth, no production data): `scripts.migrate
  generate` rewrote `deploy/migrations/BASELINE` and
  `HEAD_FINGERPRINT.json` byte-identically (git diff empty).
- Gate rows d (skip-chain) and e (restore round trip): not run. This
  release ships no database steps and no changes under `deploy/migrations/`
  or the schema files (verified: empty diff against v2026.11.15-2.3.2 and
  against the previous tag) and no updater changes. The rows carry their
  v2026.10.21-2.1.2 results unchanged.
- Gate behavior verified on a live install: 401 without a cookie, gate page
  served, 307 with a valid cookie, 401 with an invalid one, over both a
  direct tailnet connection and `tailscale serve`; spoofed `X-Forwarded-For`
  (including `127.0.0.1`, `garbage, 127.0.0.1`, and duplicate header lines)
  does not bypass the gate, and a loopback peer sending XFF is gated.
- The installer's phone-access flow was not run end to end on a fresh
  machine; it is covered by review and shell-level checks, not a live
  install. The macOS path is written but untested.
