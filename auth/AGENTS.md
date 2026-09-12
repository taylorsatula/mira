# auth/ — Passwordless identity, sessions, and account lifecycle

## Rules

- Auth behavior is owned locally. Treat this directory as the only authority for auth modes, magic links, sessions, CSRF, API tokens, WebAuthn, rate limits, security logging, and headers. Session cookies are `SameSite=Lax` so cross-site returns and magic-link clicks carry the session; browser writes rely on explicit CSRF tokens.
- `MIRA_AUTH_MODE` (`single` default, `multi`) is parsed strictly in `mode.py` — an invalid value raises rather than coercing, because a typo would silently change the process's security model. The mode is read from the environment on every call; never capture it at import time.
- Under `single` mode identity is fixed to the auto-provisioned local account (`user@localhost`), created lazily by `GET /v0/auth/local/session` on first visit. The public account-creation surface (`/signup`, `/magic-link`, `/verify`) 404s there: a stray second row would split data across identities. Do not re-open any of them under `single`.
- All auth configuration comes from Vault through `AuthConfig` (`config.py`), resolved lazily on first attribute access; importing the module performs no I/O. Non-Vault constants stay plain module values so `auth.session`/`auth.rate_limiter` can import the module in installs with no email transport.
- Raw auth tokens are returned only at issuance. PostgreSQL and Valkey store hashes or hashed key names.
- `get_current_user()` establishes the context used by PostgreSQL RLS. API tokens are accepted only from bearer headers; browser writes use cookie sessions plus CSRF.
- `AccountProvisioner` (`provisioning.py`) is the minimal excision seam between account lifecycle inside `auth` and sidecar state outside it. `NullProvisioner` is the default and only implementation — do not extend the seam with hooks, priorities, ordering, events or registration.
- Email delivery goes through the `MailSender` seam (`email_service.py`); the only backend is stdlib SMTP. Nothing in `auth` may import an HTTP email gateway client.
- The CSP header is gated behind `MIRA_CSP` and off by default: `script-src 'self'` is not satisfiable by the retained `web/` UI (inline handlers remain), and hard-enabling would serve a blank page. The other five security headers are always sent.

## Files

- `api.py` — `/v0/auth` routes plus the `get_current_user` dependency (session → API-token ladder, uniform across modes) and the page-redirect variants; `GET /local/session` bootstraps the single-mode local account.
- `service.py` — Magic-link, session, signup, account deletion, and the single-mode local-account bootstrap (`create_local_session`).
- `database.py` — PostgreSQL auth records and user creation. Session discipline is the security contract here: pre-authentication reads run on the BYPASSRLS admin session (no user context exists yet); post-authentication per-user token operations run RLS-scoped.
- `session.py` — Hashed Valkey sessions and CSRF tokens, including revoke-all-except-current traversal.
- `mode.py` — `auth_mode()`: the strict `MIRA_AUTH_MODE` parser described in Rules.
- `provisioning.py` — `AccountProvisioner` seam and `NullProvisioner` (see Rules).
- `seed_lora.py` — `seed_lora_postgres(user_id)`: initializes `feedback_synthesis_tracking` during account creation so the user model pipeline has its state row.
- `account_gc.py` — Scheduled cleanup of unactivated accounts (abandoned signups) and expired demo subjects (the demo branch stays inert while nothing can mint a `subject_kind='demo'` row); deletion goes through the provisioner's `local_teardown`.
- `config.py` — Lazy Vault-backed `AuthConfig`; import-time I/O-free by design.
- `email_service.py` — `MailSender` seam + `SmtpMailSender` (stdlib `smtplib`), configured from Vault `mira/database`-adjacent service fields; used by magic-link and notification flows in `multi` mode.
- `rate_limiter.py` — Email/IP rate limiting in Valkey.
- `webauthn_service.py` — Platform passkey (Touch ID / Face ID) challenges and credential management. Registration requests resident (discoverable) credentials; login supports both an email-keyed ceremony and a discoverable ceremony (no email typed, user resolved from the credential's userHandle via an opaque server-side challenge ID). Credentials are stored keyed by their base64url credential ID — the exact encoding browsers return in ceremonies.
- `security_logger.py` / `security_middleware.py` — Auth audit events and global security headers (see the CSP rule above).
- `types.py` / `exceptions.py` — Shared Pydantic contracts (`SessionData`, `APITokenContext`, `CookieSettings`) and `AuthError`.

## Wiring

`main.py` mounts this router in both modes; under `single` accounts bootstrap lazily through `create_local_session` (first visit provisions `user@localhost`), under `multi` through `service.py` signup, every provisioning path seeding feedback tracking via `seed_lora_postgres` inside `_initialize_account`, with `account_gc` providing teardown symmetry. `cns/api/websocket_chat.py` reuses the same credential ladder for the first auth frame.
