# Persistent agents — implementation state (2026-11-19)

Branch `agents/persistent-agents`, base `531d73e`. P0 (heartbeat fixed-cadence
failure fix) is commit `74bca02`. P1–P6 are implemented and scratch-verified;
nothing is merged, no PR is open, and the live install is untouched.

## What exists now

- `agent_definitions` + `agent_results` tables (migration step 6; fresh files
  get them lazily via `ensure_*_table`). Briefs gained `agent_id` and
  `last_outcome`.
- `agents/definitions.py` — registry CRUD, validation, tool blacklist
  (`heartbeat_tool`, `agent_tool`), auto-pause at 3 consecutive failures.
- `agents/results_store.py` — the ledger. Encrypted via the manager's
  encrypting accessors; 14-day archive sweep, never deletes; `recall` reads
  archived rows forever.
- `agents/implementations/templated_agent.py` — one class, three templates
  (investigator/implementer/watcher); prompt files in `config/prompts/agents/`.
  Memory namespace `agent:<name>`.
- `agents/dispatch.py` — deterministic bridge: a brief reported `fired` with an
  `agent_id` spawns that agent at the gate's `on_completion`, guarded by
  definition state, `min_interval_sec`, and the in-flight registry.
- `agent_tool` — create/list/inspect/pause/resume/retire/recall/run_now.
  Essential tool. `subagent_tool` unchanged.
- Trinket renders the ledger: 5 newest per agent, `<more>` expander with
  hidden/archived counts; in-flight entries stay in memory.

## Verified (scratch DBs only, never the live userdata)

Migration on a simulated pre-migration file (user_version 5 → 6) and on fresh
files; ledger round-trip; trinket render/cap/expander/dismiss; dispatch
decisions (dispatch, min_interval skip, paused skip, missing-agent skip,
in-flight skip); full trigger → report → on_completion → spawn path with the
spawn seam stubbed; agent_tool end-to-end. `import main` clean; boot gate
`_check_tools` passes with agent_tool essential.

NOT verified: a real LLM run of a templated agent (no LLM credentials in the
build shell); the gate's actual fired/false judgment; anything against the
live install. The P5 gate ran with the spawn seam stubbed and the LLM absent.

## Bugs found and fixed along the way

- heartbeat briefs were unreadable end-to-end: writes went through
  `db.insert` (encrypting) but every read path used raw `execute`/`fetchall`
  (no decryption), and `_report`/`apply_run_failure` wrote plaintext into
  `encrypted__last_result` via raw SQL. The gate was seeing ciphertext.
  Reads now decrypt (legacy plaintext tolerated), writes encrypt.
- `_ensure_briefs_columns` read `row[1]` off dict rows from `db.execute`
  (KeyError on every fresh briefs table) — now `row["name"]`.
- `results_store.maybe_archive_sweep` compared ISO-with-T timestamps against
  SQLite `datetime('now')` (space format) — comparisons were meaningless;
  cutoff is now computed in Python with the same format.

## Open items for review

- `agent_tool` added to `ESSENTIAL_TOOLS` (repo.py) — every user gets it.
- `agent_tool` added to `NEVER_GRANTABLE_TOOLS` (agents may not mint agents) —
  a judgment call beyond the original heartbeat-only blacklist; veto if wrong.
- `run_now` bypasses `min_interval_sec` but not the in-flight guard or the
  state check.
- `seen_at` is set on render, informational only; the 14-day archive clock
  runs off `created_at` for both seen and unseen rows.
- Renaming an agent orphans its `agent:<name>` memory namespace and ledger
  grouping (the `agent_id` agt_… stays stable).
- P7 (PR) not started, per plan — Taylor reviews first.
