# deploy/migrations/ — the database step chain that carries an installed `mira_service` forward from release to release, and its authoring contract

Elsewhere: the fresh-install schema and the ownership bootstrap — `deploy/AGENTS.md` (`mira_service_schema.sql`, `bootstrap_adopt.sql`) · the runner (discovery, validation, lint, plan) — `utils/AGENTS.md` (`utils/schema_migrations.py`) · the CLI and its exit codes — `scripts/AGENTS.md` (`scripts/migrate.py`) · plain-English maintainer overview — `docs/MAINTAINER_UPGRADE_GUIDE.md` · how a step came to exist — `git log`.

## Rules

- Step numbers are their own dense 4-digit counter (`0001`, `0002`, …), unrelated to release versions; a gap or a duplicate number refuses the whole chain. A versioned step runs once, in ascending order; a step numbered below the database's highest applied step is refused as out of order — renumber it above head.
- A shipped step is immutable: installs that already ran it record its sha256, and a changed file is refused on every one of them (`CHECKSUM_MISMATCH`). Fix forward with a new step — nightly installs apply a step the moment it merges, so "shipped" means merged.
- `BASELINE` and `HEAD_FINGERPRINT.json` are generated, never hand-edited (`python3 -m scripts.migrate generate`, a release-cut step in `deploy/RELEASE.md`); `BASELINE` rows are append-only across releases — the generator refuses a result that does not extend the checked-in file. Origins are the closed set `utils/schema_migrations.py:SCHEMA_ORIGINS` (the releases before the chain existed; every later release carries its ledger, so none is ever added). The generator builds each origin from its tagged schema + today's `deploy/bootstrap_adopt.sql`, walks every postgres-store step, and refuses (`DO NOT PUBLISH`, nothing written) unless each origin + chain equals the current schema + bootstrap — so a schema-file edit without its step, or a step without its schema-file edit, cannot ship. Consequence: rows hash structure only; a vault/env/valkey step's row repeats the previous fingerprint.
- Discovery reads only `V<NNNN>__<slug>.sql|.py` and `R__<slug>.sql|.py`; a `.sql`/`.py` file starting with `V` or `R` that fails that grammar is refused, and `BASELINE`, `HEAD_FINGERPRINT.json`, `AGENTS.md`, `__pycache__/` are skipped.
- Every step declares its header; nothing is inferred from the body:

  | Field | `.sql` (leading `--` block) | `.py` (module-level literal) |
  |---|---|---|
  | store (required) | `-- store: postgres` | `STORE = "postgres"` |
  | notice | `-- notice: <text>` + `-- ` continuation lines | `NOTICE = "<text>"` |
  | typed operator value | `-- REQUIRES_OPERATOR_INPUT: <prompt>` | `REQUIRES_OPERATOR_INPUT = "<prompt>"` |
  | dialect twins | `-- dialect-twins: DialectName, DIALECT_NAMES` | `DIALECT_TWINS = ("DialectName", "DIALECT_NAMES")` |
  | contract marker | `-- expand-contract: contract-for V<NNNN>` | `EXPAND_CONTRACT = "contract-for V<NNNN>"` |

  `store` is one of `postgres`, `vault`, `env`, `valkey`. A `.py` step is read with `ast`, never imported, at discovery.
- The notice is the one line an operator reads before consenting: plain English, what changes and what is protected, none of the banned words (ledger, revision, checksum, migration, schema, fingerprint, adoption, baseline). A step without one shows the generic `S-STEP-DEFAULT`; ship that only when the step truly changes nothing the operator would notice.
- Lint (`utils/schema_migrations.py:lint_steps`, run by `mira migrate lint` and refused at artifact build) bans: transaction control (`BEGIN`/`COMMIT`/`ROLLBACK`/`SAVEPOINT` outside dollar-quoted bodies — the runner owns the step's transaction); reading the clock (`NOW()` etc., except as a column `DEFAULT`; `time.time()`/`datetime.now()` in `.py`) or the release `VERSION`; any `embedding_config` reference (model and dimensions are fixed at install); `UPDATE model_configs SET` of a route field (`model`, `dialect_name`, `endpoint_url`, `api_key_name`, `effort`, `max_tokens` — the operator's choices); a literal `vector(<digits>)`; a `dialect_name` CHECK change without the dialect-twins marker (`clients/llm/types.py:DialectName` must move with it); `DROP TABLE`/`DROP COLUMN`/`RENAME` without the expand-contract marker (the expand step shipped a release earlier). A step must do the same thing on every install, whenever it runs.
- A `.py` step defines `up(ctx)` and optionally `verify(ctx)` (same transaction); it never commits. `ctx` (`utils/schema_migrations.py:StepContext`): `ctx.pg` the run's connection as `mira_admin`; `ctx.vault` — `read`, `patch` (writes only changed fields), `delete_fields(…, confirmed_new=…)` (refuses until the replacement fields exist); `ctx.env` — `set_var`/`unset_var` on the settings file (atomic, 0600); `ctx.invalidate_valkey()`; `ctx.operator_input(prompt)`; `ctx.embedding_dimensions()` (interpolate with `psycopg.sql` only). Non-postgres work is idempotent and runs before the ledger row, which the runner writes last in the step's transaction — a crash between them re-runs the work, never skips it.
- A step needing a typed value declares `REQUIRES_OPERATOR_INPUT`; with no terminal the run stops before any backup or change (exit 3) and the prompt is shown in the consent plan. A change spanning stores (a Vault field and the code reading it) is split expand → repoint → contract across steps or releases, never one step.
- Lock key `MIGRATION_ADVISORY_KEY` (1296650817, `b"MIRA"`) belongs to the runner; nothing else in MIRA may take that advisory lock.
- A repeatable step's ledger key is `R__<slug>`; it re-runs whenever its bytes change, so it is written converge-to-state, never as a delta — and converges only what MIRA owns: a row an operator may have edited (a `usage_pricing` price is the operator's override) is inserted when missing (`ON CONFLICT DO NOTHING`), never deleted or overwritten. Repeatables carry reference data, never structure — `BASELINE` hashes the chain without them.
- Adoption of a database whose structure matches no known release or step is refused (exit 2) with a full difference report; only `mira update --adopt-drifted` (or `mira migrate --adopt-drifted` once the code is current), typed by a human for that one run, passes it — adoption drift surfaces only during an update, when the installed code may predate `mira migrate`, and every ledger row that run writes records `applied_by = "<version>+adopt-drifted"`. No config or environment variable disables the refusal — an unknown structure never silently converges.

## Files

- `V0001__widen_segment_sentinel_uniqueness.sql` — one live segment sentinel per continuum covers `paused` too: replaces `idx_messages_active_segment_unique` with `idx_messages_live_segment_unique`. Precondition DO block refuses (names the continuum id and count, changes nothing) when a continuum already holds two live sentinels.
- `V0002__drop_keepsleeping_turn_index.sql` — drops `idx_messages_keepsleeping_turn_id` (v2026.10.06 only; a no-op from v2026.10.03).
- `V0003__create_heartbeat_armed_users.sql` — the `heartbeat_armed_users` presence table + `idx_heartbeat_armed_wake`, byte-for-byte the schema file's DDL; no grant of its own — `mira_admin` owns it (creator), which the fingerprint counts the same as the fresh install's blanket grant + bootstrap transfer.
- `R__usage_pricing.sql` — the shipped `usage_pricing` keys (`__default__` + the five routes) inserted when missing; operator prices kept.
- `BASELINE` — generated: one `origin=<tag>` line per `SCHEMA_ORIGINS` entry, then `revision=<NNNN>` after each step on the newest origin.
- `HEAD_FINGERPRINT.json` — generated: the full structure at head, what drift reports diff against; its hash must equal `BASELINE`'s last row (`read_head_fingerprint` refuses otherwise).

## Verification

Nothing here is a suite; these are the checks a change to the chain, the runner, or `deploy/update.sh` re-derives, each against real infrastructure, never a mock (root NO MOCKS doctrine):

- Convergence — `python3 -m scripts.migrate generate` on a scratch server: every origin + chain equals the fresh install, or `DO NOT PUBLISH`.
- Skip-chain — one `migrate` from 0000/0001/0002 lands on the same structure and ledger (minus timestamps) as walking release by release.
- Adoption — an origin database: plan shows the one-time sentence + each NOTICE; Y stamps nothing above the match and applies the rest; a hand-added column refuses with a report naming it; `--adopt-drifted` proceeds and records `+adopt-drifted`.
- Failure — a step that raises: exit 1, the backup restored (structure with ledger, row counts, settings file equal to before), and through `deploy/update.sh` code, git HEAD, `.git/index`, venvs, `mira` wrapper, unit back; a damaged dump parks with the printed `dropdb`/`createdb`/`pg_restore` commands and MIRA stays stopped.
- Consent — N changes nothing and never stops the service; no terminal → exit 3; no superuser seam → exit 3; a `REQUIRES_OPERATOR_INPUT` step with no terminal → exit 3 before any backup.
- Concurrency and interruption — two runners: each step applied once; SIGKILL or power loss mid-step: no ledger row, no partial DDL, the lock released within `CLIENT_CHECK_INTERVAL_SECONDS`, a rerun completes.
- Boot guard — `schema_revision` parks on no ledger, behind, ahead, and a gap in the tree, and passes at head.
- Fresh install — the installer stamps 0001..head; `mira status` reports head with nothing waiting.
- Container — the `mira-migrate` oneshot applies a new step on restart with no `docker exec`; a failing step restores and keeps `mira` down.
- Per-user SQLite — a legacy-shaped `userdata.db` reaches `USERDATA_HEAD` with encrypted fields decrypting; a new file is stamped without running steps.
