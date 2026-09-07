# scripts — Operational utilities

Small, runnable helpers that operate outside the normal request lifecycle.
They run against a deployed environment (bare-metal systemd or the container),
where Vault and the `mira_service` database are reachable — not against the
unit-test harness.

## Files

- `post_server_post.py` — Runs the post-server power-on self-test probe against a live MIRA service via HTTP diagnostics.
- `__init__.py` — Package marker only; it exists so operational scripts run with `python -m scripts.<name>`.

## Patterns

- Scripts that touch the database should respect RLS when exercising app code and use `PostgresClient(..., admin=True)` only for cross-user lookups or administrative fixes.
- Vault-required scripts must set `VAULT_ROLE_ID` and `VAULT_SECRET_ID` from `/opt/vault/role-id.txt` and `/opt/vault/secret-id.txt` when run outside the service environment.
