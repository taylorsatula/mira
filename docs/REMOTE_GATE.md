# Remote Gate

MIRA's single-user mode auto-provisions a browser session for anyone who can
reach port 1993. That is the right default on loopback and the wrong one on a
tailnet or LAN. The remote gate closes that gap with one shared secret.

## What it does

When the gate is enabled, every request from a non-loopback peer — pages,
assets, the API, and the WebSocket — must carry a `mira_gate` cookie derived
from the shared secret. Browser navigations without it are redirected to
`/remote-gate`, a self-contained page that asks for the secret; API clients
get a `401` with code `REMOTE_GATE_REQUIRED`. Loopback peers are never gated,
so local tooling, the installer's health checks, and the Docker healthcheck
are unaffected.

The gate is an outer layer, not a replacement for auth: after it admits a
request, the normal credential ladder (single-user auto-login, sessions, API
tokens) applies unchanged.

## Enabling

The gate is enabled by the presence of a secret in Vault, and disabled by its
absence. There is no config file and no environment variable.

The installer asks about this (question 11, default no; `remote_gate:` in a
`--config` file) and recommends it for anyone who might open MIRA from a phone
or another device. When enabled, it generates the secret — four EFF
large-wordlist words separated by hyphens, about 52 bits, chosen to be
typable on a phone keyboard — writes it to `secret/mira/gate`, and copies it
into the credential dump (`~/MIRA_credentials.txt`). Re-running the installer
keeps an existing secret rather than rotating it. If you want more entropy,
set your own token instead: any string works, e.g. a longer diceware run or a
hashed random key — `vault kv put secret/mira/gate token="$(diceware -n 7 --no-caps -d '-')"`.

Question 12 (`tailscale:` in a `--config` file) goes one step further and
sets up phone access end to end: it installs Tailscale on the machine, signs
it into your tailnet (you open a login link during the install), and serves
MIRA at `https://<machine>.<tailnet>.ts.net` with `tailscale serve`. It
implies the gate. Your phone needs the free Tailscale app, signed into the
same account; opening the URL and entering the gate secret once is all the
phone-side setup there is. The installer refuses to expose MIRA via
`tailscale serve` when the gate is not enabled, and after serving it checks
that an unauthenticated request is refused before reporting success.

To enable it by hand on an existing install:

```bash
vault kv put secret/mira/gate token="$(diceware -n 4 --no-caps -d '-')"
```

Then restart MIRA. The token is read once per process; enabling or disabling
takes effect on restart. Store the token somewhere safe — it is the only
credential the gate checks, and it is shown nowhere in the UI.

## Disabling

```bash
vault kv delete secret/mira/gate
```

Restart MIRA.

## Notes

- The cookie is `HttpOnly`, `SameSite=Lax`, `Secure` only when the request is
  HTTPS (e.g. behind `tailscale serve`), and lives 45 days (the session
  lifetime). It carries an HMAC of the token, not the token itself.
- The submit endpoint is rate-limited per client IP (10 attempts per 5
  minutes) and both success and failure are logged as `auth.remote_gate_*`
  security events.
- If Vault cannot be read at request time, the gate fails closed: remote
  access is denied rather than opened.
- The token is compared in constant time. Rotating it invalidates all
  outstanding gate cookies (they are HMACs of the old token).
- WebSocket handshakes without a valid gate cookie are closed with code 4401
  before acceptance.
- A loopback peer that sets `X-Forwarded-For` is treated as a reverse proxy
  (e.g. `tailscale serve`) and is no longer exempt from the gate. The header
  is never parsed for the gate decision — its presence alone disqualifies the
  loopback exemption — so no header content can make a proxied request look
  local. The header is ignored from non-loopback peers.
- `/remote-gate` and `/v1/auth/remote-gate` are the only exempt paths.
- For HTTPS (recommended for a PWA over cellular), put `tailscale serve` in
  front: `tailscale serve --bg 1993` gives `https://<host>.<tailnet>.ts.net`
  with a valid certificate. The gate works over plain HTTP too — the tunnel
  provides the transport encryption — but browsers will not offer PWA install
  without a secure context.
