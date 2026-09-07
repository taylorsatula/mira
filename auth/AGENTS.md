# auth/ — Passwordless identity, sessions, and account lifecycle

## Rules

- Auth behavior is owned locally. Treat this directory as the only authority for auth modes, magic links, sessions, CSRF, API tokens, WebAuthn, rate limits, security logging, and headers. Session cookies are `SameSite=Lax` so cross-site returns and magic-link clicks carry the session; browser writes rely on explicit CSRF tokens.
- `MIRA_AUTH_MODE` (`single` default, `dev`, `multi`) is parsed strictly in `mode.py` — an invalid value raises rather than coercing, because a typo would silently change the process's security model. The mode is read from the environment on every call; never capture it at import time.
- In `single` mode the public account-creation surface (`/signup`, `/magic-link`, `/verify`, `/dev/session`) 404s. Those routes are writes elsewhere: one unauthenticated request could add a second `users` row and permanently brick `main.py`'s single-user boot guard. Do not re-open any of them under `single`.
- All auth configuration comes from Vault through `AuthConfig` (`config.py`), resolved lazily on first attribute access; importing the module performs no I/O. Non-Vault constants stay plain module values so `auth.session`/`auth.rate_limiter` can import the module in installs with no email transport.
- Raw auth tokens are returned only at issuance. PostgreSQL and Valkey store hashes or hashed key names.
- `get_current_user()` establishes the context used by PostgreSQL RLS. API tokens are accepted only from bearer headers; browser writes use cookie sessions plus CSRF.
- `AccountProvisioner` (`provisioning.py`) is the minimal excision seam between account lifecycle inside `auth` and sidecar state outside it. `NullProvisioner` is the default and only implementation — do not extend the seam with hooks, priorities, ordering, events or registration.
- Email delivery goes through the `MailSender` seam (`email_service.py`); the only backend is stdlib SMTP. Nothing in `auth` may import an HTTP email gateway client.
- The CSP header is gated behind `MIRA_CSP` and off by default: `script-src 'self'` is not satisfiable by the retained `web/` UI (inline handlers remain), and hard-enabling would serve a blank page. The other five security headers are always sent.

## Files

- `api.py` — `/v0/auth` routes plus the `get_current_user` dependency (session → API-token ladder, with the mode-gated single-user bearer branch) and the page-redirect variants.
- `service.py` — Magic-link, session, signup, account deletion, and the development-session bootstrap (`create_development_session`).
- `database.py` — PostgreSQL auth records and user creation. Session discipline is the security contract here: pre-authentication reads run on the BYPASSRLS admin session (no user context exists yet); post-authentication per-user token operations run RLS-scoped.
- `session.py` — Hashed Valkey sessions and CSRF tokens, including revoke-all-except-current traversal.
- `mode.py` — `auth_mode()` / `single_user_mode_enabled()`: the strict `MIRA_AUTH_MODE` parser described in Rules.
- `provisioning.py` — `AccountProvisioner` seam and `NullProvisioner` (see Rules).
- `seed_lora.py` — `seed_lora_postgres(user_id)`: initializes `feedback_synthesis_tracking` during account creation so the user model pipeline has its state row.
- `account_gc.py` — Scheduled cleanup of unactivated accounts (abandoned signups) and expired demo subjects (the demo branch stays inert while nothing can mint a `subject_kind='demo'` row); deletion goes through the provisioner's `local_teardown`.
- `config.py` — Lazy Vault-backed `AuthConfig`; import-time I/O-free by design.
- `dev_mode.py` — Explicit `MIRA_DEV` gate used only for local session bootstrap and HTTP cookie security; production authentication still uses magic links.
- `email_service.py` — `MailSender` seam + `SmtpMailSender` (stdlib `smtplib`), configured from Vault `mira/database`-adjacent service fields; used by magic-link and notification flows in `multi`/`dev` modes.
- `rate_limiter.py` — Email/IP rate limiting in Valkey.
- `webauthn_service.py` — Platform passkey (Touch ID / Face ID) challenges and credential management. Registration requests resident (discoverable) credentials; login supports both an email-keyed ceremony and a discoverable ceremony (no email typed, user resolved from the credential's userHandle via an opaque server-side challenge ID). Credentials are stored keyed by their base64url credential ID — the exact encoding browsers return in ceremonies.
- `security_logger.py` / `security_middleware.py` — Auth audit events and global security headers (see the CSP rule above).
- `types.py` / `exceptions.py` — Shared Pydantic contracts (`SessionData`, `APITokenContext`, `CookieSettings`) and `AuthError`.

## Wiring

`main.py`'s bootstrap branches on `mode.py`: `single` resolves one bearer key to the hardcoded local user (`ensure_single_user`, boot-guarded); `dev`/`multi` mount this router and bootstrap accounts through `service.py` signup, each new account running `seed_lora_postgres` and `provisioning` teardown symmetry via `account_gc`. `cns/api/websocket_chat.py` reuses the same ladder for the first auth frame.
