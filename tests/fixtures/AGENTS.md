# tests/fixtures/ — reusable probe scaffolding: real-infrastructure setup and teardown, no claims

These modules lower the cost of a live probe without authoring any claim. They
provision or destroy real infrastructure, or report a result; none asserts
behavior, simulates a subject, or reads a production table. Import from an inline
probe: `sys.path.insert(0, "tests/fixtures")`, then import the module directly —
no package `__init__.py`, matching `tests/tmp/` and `tests/protected/`.

## Rules

- A fixture here may provision or destroy real infrastructure and report PASS/FAIL, and nothing else. `harness.check` records what a callable did; the callable is the thing that touches reality. A simulated subject is a defect here, not a shortcut — shared scaffolding that faked a subject would industrialize the false pass the root no-mocks doctrine exists to prevent.
- Every fixture calls the sanctioned production surface — `utils.user_context.set_current_user_id`, the shipped `deploy/mira_service_schema.sql` — never a reimplementation. A helper duplicating a production surface is itself the defect: fix the production seam (probe-able-code craft: the `writing-probes` skill, routed through the root `AGENTS.md` NO MOCKS pointer) or the probe that needs it, not this directory.
- Teardown is checkable, not self-enforcing: `probe_user()`'s exit path clears user context and calls `cleanup(user_id)` best-effort (`rmtree` with `ignore_errors=True` — a failed removal is NOT raised); the hard checks live in `residue.assert_user_removed` and `assert_no_keys`, which a probe must call to verify its own blast radius. A fixture helper that cannot perform its primary action raises rather than returning a default.
- A fixture that cannot reach its dependency reports BLOCKED and exits nonzero (`live_infra.require`), never a default or an empty result. "Cannot run" and "ran and failed" stay distinguishable — the probes this scaffolding serves depend on the distinction.
- `scratch_database()` opens its admin connection for `CREATE`/`DROP DATABASE` against a generated `mira_probe_*` name, plus role provisioning and the pgvector availability check (`CREATE ROLE`/`SELECT` on `pg_roles`/`pg_available_extensions`); it never reads or writes an application table, and the caller reaches the yielded database through its own connection.

## Files

- `harness.py` — claim-free results collector. `check(label, fn)` runs `fn`, records and prints `PASS`/`FAIL` with the raw outcome (a raised exception is recorded, never swallowed); `finish()` prints `SUMMARY` + `PROBE PASSED`/`PROBE FAILED` and returns the exit code (nonzero on any failure or zero checks). Holds no infrastructure and no assertion. Consumers: disposable `/tmp` probes wanting a uniform marker.
- `live_infra.py` — reachability gate. `reachable(host, port)` is a real TCP connect; `require(service, host=…, port=…)` exits 3 with `<SERVICE>-PROBE-BLOCKED` when nothing listens. Default ports: postgres 5432, valkey 6379, vault 8200. Call before a live probe so a dead dependency reports BLOCKED, not FAIL.
- `probe_user.py` — throwaway-user lifecycle. `probe_user()` yields a fresh UUID with `set_current_user_id` set and clears context + removes `data/users/<id>` on exit; `cleanup(user_id)` is the explicit form; `user_dir(user_id)` is repo-root-anchored (cwd-independent). Creates no DB row — user-scoped filesystem and contextvar paths only.
- `residue.py` — residue checks. `assert_user_removed(user_id)` and `assert_no_keys(prefix, client)` raise on leftover state; call after teardown so a probe verifies its own blast radius is empty.
- `scratch_db.py` — throwaway Postgres database. `scratch_database(**conn_kwargs)` creates a `mira_probe_*` DB, applies the shipped schema **via a `psql` subprocess** (the schema's `\if`/`\set` guards are psql meta-commands psycopg cannot send; the five `embedding_*` variables are passed with `-v`, sourced from `describe_for_installer` the same way `deploy/lib/embedding_config.sh` resolves them), provisions the `mira_admin`/`mira_dbuser` roles and checks pgvector availability itself, yields the DB name, drops the roles it created and the DB `WITH (FORCE)`. `conn_kwargs` are local admin libpq kwargs (`user=`, `host=`); documented caller pattern is `psycopg.connect(dbname=name, **conn_kwargs)`.
