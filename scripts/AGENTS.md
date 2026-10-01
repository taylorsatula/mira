# scripts/ — operational entry points run against a live, deployed service

## Rules

- Every script here is a thin CLI shim: argument parsing and output formatting live in the owning module (e.g. `utils/power_on_self_test.py:post_server_cli`), not in this directory. Fix behavior there; never fork logic into a script.
- When run outside the service environment, Vault-required scripts must export `VAULT_ROLE_ID` and `VAULT_SECRET_ID` from `/opt/vault/role-id.txt` and `/opt/vault/secret-id.txt` (populated by `deploy/lib/vault.sh`); `clients/vault_client.py` reads exactly those env vars and raises if missing.
- Bare-process invocation mechanics (learned the hard way, 2026-09-15): run from `/opt/mira/app` with `PYTHONPATH=/opt/mira/app` — a probe launched from an arbitrary cwd or without the path gets `ModuleNotFoundError` for `utils`/`clients`. And never let `/tmp` sit on `sys.path`: stale deploy-staging files there (e.g. a scp'd `config.py`) shadow the real modules and produce baffling import errors. Stage deploys under unique names and keep probes out of `/tmp`.
- Scripts touching the database must respect RLS for app-level paths; `PostgresClient(..., admin=True)` is only for cross-user lookups or administrative fixes (root AGENTS.md user-context doctrine applies here too).

## Files

- `post_server_post.py` — CLI shim that runs the post-server power-on self-test against the already-bound live MIRA service via HTTP diagnostics. Delegates entirely to `post_server_cli()` in `utils/power_on_self_test.py` (flags: `--base-url`, `--deadline-seconds`, `--json`); exits 0 only if `report.required_passed`. Run as `python -m scripts.post_server_post`. Gate mechanics and probe ownership belong to `utils/power_on_self_test.py` — see root AGENTS.md POST section, do not restate here. Consumers: invoked by operators only (the pre-server boot gate in `main.py` is the deploy-time release check; nothing in `deploy/` or runtime invokes this probe); anchor `utils/power_on_self_test.py:post_server_cli` documents it as this script's CLI.
- `__init__.py` — Docstring only, no re-exports; exists so operational scripts run with `python -m scripts.<name>`.
