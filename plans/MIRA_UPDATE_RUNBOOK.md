# MIRA `mira update` overhaul — implementation runbook

Binding constraints, stated once:

- Verification is live path-probes only. No test files, no mocks, no simulated
  infrastructure (`tests/AGENTS.md`). Disposable probes live in `/tmp`
  (`tests/tmp/AGENTS.md`); shared setup/teardown is imported from
  `tests/fixtures/` (`sys.path.insert(0, "tests/fixtures")`), never
  reimplemented.
- Every behavior change updates its directory's `AGENTS.md` map **in the same
  commit** (`docs/AGENTS_MAP_SPEC.md` maintenance trigger table). Map edits are
  enumerated as numbered steps inside each phase, not deferred to Phase 8.
- No new runtime dependencies (CROSS-4): `psycopg` 3, `hvac`, stdlib. All three
  are already in `requirements.txt`.
- Line references are to the working tree at the time of writing; they are
  anchors for reading, not edit coordinates.
- **This overhaul is a breaking change in its own right.** Backward
  compatibility with the existing release process is not required. A changed
  preserve set on the reinstall path, a changed release-payload shape, and a
  corrected PostgreSQL floor are permitted by ruling; none needs a compatibility
  shim.
- **The mechanism is permanent** — it exists for the project's lifetime in some
  form. Every conflict below is therefore resolved to the durable answer, not
  the minimally disruptive one, and every witness of a load-bearing invariant is
  destined for a permanent home (Phase 8 step 8).
- **Ablate, don't depreciate.** A retired path is deleted in the same commit.
  A path kept only so existing installs can reach the mechanism is named, marked
  transitional in the code and the map, and given a removal that is an explicit
  operator decision — never a silently permanent second path (root `AGENTS.md`:
  Backwards Compatibility).

## Requirement → phase matrix

| ID | Phase(s) | Witness |
|---|---|---|
| PROV-1, PROV-2, PROV-3, PROV-4 | 0 | `probe_prov_commit.sh`, `probe_prov_status.sh`, `probe_release_artifact.sh` (PROV-2) |
| CUST-1 … CUST-7 | 0 | `probe_custom_export.sh`, `probe_venv_drift.sh` (CUST-3) |
| VENV-1 … VENV-5 | 0 | `probe_venv_stamp.sh` |
| VER-6 (deploy-script probes on `/tmp` scratch trees) | 0 | `probe_venv_stamp.sh`, `probe_custom_export.sh`, `probe_venv_drift.sh`, `probe_prov_commit.sh`, `probe_prov_status.sh`, `probe_update_sh_scratch.sh`, `probe_release_artifact.sh`, `probe_state_preserve.sh` |
| REL-1 (release-time integrity gate) | 0 (condition a: the artifact builder + `verify_release_identity`), 2 (b: ledger-integrity checks), 3 (c/d/e witnesses), 5 (the gate text and its cut-time execution) | `probe_release_checklist.sh` (extended to all five conditions), `probe_release_artifact.sh` |
| CROSS-1 | 0 ships alone; 1–8 depend on 0 only for the stamp/export call sites | phase-0 probe set green before Phase 1 opens |
| MIG-8, MIG-2 (table), MIG-17 | 1 | `probe_bootstrap_adopt.py`, `probe_ownership_shape.py`, `probe_secrets_manifest.py` |
| MIG-1, MIG-13, MIG-14 (authoring rules as checks) | 2 | `probe_migrate_discovery.py` |
| MIG-3 (status/validate/dry-run), MIG-4 | 2 | `probe_migrate_validate.py`, `probe_migrate_status.py` |
| MIG-3 (migrate), MIG-5, MIG-6, MIG-7, MIG-9, MIG-15, MIG-20 | 3 | `probe_migrate_apply.py`, `probe_migrate_privileges.py`, `probe_restore_roundtrip.py`, `probe_backup_retention.py`, `probe_sigkill_resume.py`, `probe_adoption_drift.py` (MIG-9, as amended by R2) |
| MIG-10, MIG-11, MIG-12, MIG-3 (`mira migrate` dispatch) | 4 | `probe_post_schema_revision.py`, `probe_update_phase65.sh`, `probe_superuser_seam.sh`, `probe_container_migrate.sh` |
| MIG-19 | 5 | `probe_v0001.py` |
| MIG-18 | 6 | `probe_env_reconcile.py`, `probe_repeatable_step.py` |
| MIG-16 | 7 | `probe_userdata_migrations.py` |
| CROSS-2 | every phase (own steps) + 8 (audit sweep) | `docs/AGENTS_MAP_SPEC.md` anchor/reciprocity/budget audits |
| CROSS-3 | 2, 3 (runner signature) | every phase-2/3 probe runs with no sudo |
| CROSS-4 | 2, 3 | `probe_no_new_deps.py` |
| VER-1 … VER-7 | distributed; index in Phase 8 | see Probe index |
| D1 … D6 | ruled — every tag applied in place | DECISIONS table |

## DECISIONS (ruled)

R10: the former D1–D6 parameters table is collapsed into this one. Each row is a
ruling, not an open parameter. `Lands at` names the steps and
probe-index rows that carry it; a step whose text contradicts its row is a
defect. The `Rn` tags name the operator ruling that closed the row. A `[Dn]` tag
inside step text points at this table's row — it marks where a ruling is applied,
not an open question.

| ID | Decision | Ruling | Lands at |
|---|---|---|---|
| D1 | Failure path after a failed migrate/health-poll | **Auto-restore stands.** Restore the backup, roll back code/venvs, restart on the old version; park when the restore itself fails. Reverses the recorded "v1 never auto-rolls back" decision — see the reversal flag below. | 3.10, 4.1, 4.3; P3-5, P4-3 |
| D2 | Adoption fingerprint drift | **R2 — refuse + explicit recorded override. No silent convergence.** Three outcomes at adopt (3.4): fingerprint match ⇒ stamp `BASELINE` and proceed; mismatch ⇒ exit 2, print the drift report, name the exact override invocation; `mira migrate --adopt-drifted` ⇒ the override is **recorded in the ledger**, then the full idempotent chain runs. The known case is the maintainer's own hand-patched install — the pre-applied partial-unique-index change V0001 covers (5.1): review the drift report, override once, done. Unknown drift stays refused, permanently: the flag is per-invocation, and no config value disables the check. | 2.4, 2.7, 2.8, 3.4, 3.11, 1.5, 5.1; P3-9 |
| D3 | Container schema step | **R3 — the container runs migrate at every start.** The boot-guard-refusal alternative (`docker exec … mira migrate`) is deleted, not parked. | 4.7; P4-6 |
| D4 | `CREATEDB` for `mira_admin` | **R6 — `mira_admin` stays narrow: no `CREATEDB` grant.** Restore = schema-reset `public` + `pg_restore --no-owner`, probe-proven. The restore round-trip probe is promoted from phase verification to a **shipping-gate condition** (REL-1(e)). | 1.8, 3.9; P3-4, P3-5, REL-1(e) |
| D5 | Step needing operator-only values | **Stands.** Prompt on a TTY; park + exit 3 otherwise. No value is ever defaulted. | 3.6; P3-11 |
| D6 | Major-version crossings | **Stands.** Out of scope; they stay one-off scripts. | 5.4, 8.3 |
| REL-1 | Release-time integrity gate | **R7 — new requirement.** A release is **not published** unless all five cut-time conditions hold (gate table at 5.4). Owned by `deploy/RELEASE.md`; any failure = do not publish. Authoring-time enforcement is the defense; install-time validation (`_check_schema_revision`, 4.5) is the backstop. | 0.10, 0.21 (a), 2.6 (b), 3.9 (e witness), 5.4 (gate text + execution); P5-6, P0-7 |

**D1 reversal flag.** `deploy/update.sh:30-33` states "on failure they are kept
and the rollback commands are printed (v1 never auto-rolls back)", and
`deploy/AGENTS.md`'s `update.sh` bullet repeats "v1 never auto-rolls back". D1
reverses both. Every step that implements it (3.10, 4.1, 4.3) carries the
reversal in its own text, and both the script header and the map bullet are
rewritten in the same commit — the trigger table's rule for a removed behavior is
*delete the statement*, never annotate it.

**R8 transition symmetry.** The first release carrying this mechanism **ships
`BREAKING.md`**: the one-time superuser-seam bootstrap is operator action, which
is exactly the narrowed meaning MIG-19 leaves the marker ("the operator must act
or consent"). Consequence, binding: `mira update`'s presence gate is unchanged,
so every existing install **refuses** the transition release and takes it through
`install.sh` or the manual path the file prints. Subsequent releases do not carry
it. Plan stated at 5.3 and 5.4.

**R9 probe permanence.** A permanent mechanism with disposable witnesses is a
contradiction. The load-bearing witnesses — convergence, skip-chain,
interrupt/SIGKILL, the guard set, privilege + restore round-trip — are destined
for `tests/protected/` as a permanent battery. Admission there requires the
operator to type the exact phrase `AUTHORIZE PROTECTED TEST SAVE`
(`tests/AGENTS.md:45`, `tests/protected/AGENTS.md:5`), **which has not been
granted**. Until it is: they remain disposable `/tmp` probes per
`tests/tmp/AGENTS.md`, their index rows carry the `→ protected (pending
authorization)` marker, and Phase 8 step 8 is the explicit decision step that
asks. The marker's scope and the offline-rule obstacle are stated at 8.6 and 8.8.

## Conventions

### Probe contract

Every probe below is a disposable live probe in `/tmp`, written to the anatomy
in the `writing-probes` skill: docstring naming the question, the real surface,
and the bound; setup → exercise → assert → terminal marker → nonzero exit →
teardown → residue assertion. Each prints exactly one of:

| State | Meaning | Exit |
|---|---|---|
| `PROBE PASSED: <name>` | ran against live infrastructure, every assertion held | 0 |
| `PROBE FAILED: <name> — <assertion>` | ran, an assertion did not hold | 1 |
| `<SERVICE>-PROBE-BLOCKED: <reason>` | required infrastructure unreachable | 3 |

Gate every live probe on `tests/fixtures/live_infra.py:require` before touching
anything, so a dead dependency reports BLOCKED and not FAIL. Postgres 5432,
Valkey 6379, Vault 8200 are the defaults there; `require` exits 3 with
`<SERVICE>-PROBE-BLOCKED`.

A change report ends with **EXECUTED** (what ran, what the input was) or
**UNVERIFIED** (why not, and which probe above would cover it). Never both
silently.

### Venues

| Venue | Used for | Notes |
|---|---|---|
| Repo checkout, local Postgres superuser | Phases 1–3, 5–7 DB probes | `tests/fixtures/scratch_db.py:scratch_database(user=<local superuser>, host="localhost")`. No sudo (CROSS-3). |
| Repo checkout + live Vault AppRole | MIG-15 `ctx.vault`, MIG-17 | Blocked without `VAULT_ADDR` + `VAULT_ROLE_ID`/`VAULT_SECRET_ID`; a `~/.vault-token` authenticates the CLI, not the app. Report UNVERIFIED, not FAIL. |
| `/tmp` scratch trees | Phase 0 deploy-script probes | `MIRA_APP_DIR`/`MIRA_TUI_VENV` overrides exist for exactly this (`deploy/update.sh:42-50`, `ui/tui/update.py:29-30`). Non-canonical root ⇒ no service stop/start. |
| Deployed host over SSH | Phase 4 wiring probes, MIG-11 seam | `sudo -n -u postgres` on Linux; the invoking user is the superuser on macOS Homebrew (`deploy/postgresql.sh:110` vs `:119`). |
| Container | MIG-12 | Interpreter is `/opt/mira/venv/bin/python3`, NOT `/opt/mira/app/venv` (`deploy/docker/Dockerfile.base:31`). |

### Prior art, cited where its pattern is used

| Design | Pattern used | Where |
|---|---|---|
| Flyway | `V`/`R` step split; `schema_history` ledger; checksum validation; baseline-on-adopt; `outOfOrder=false`; advisory lock; undo explicitly optional | 1.2, 2.1–2.5, 3.1, 3.4 |
| Django migrations | adopting an unversioned existing DB (`--fake-initial` equivalent); linear dependency; no required down | 1.5, 3.4 |
| Rails ActiveRecord | schema-dump staleness guard → the convergence probe | VER-1, 5.4 |
| Alembic | version table + one transaction per step | 3.5 |
| dbmate / goose | SQL-file-first authoring, `V####__slug.sql` discovery | 2.1 |
| Sqitch | per-change `verify(ctx)` on Python steps | 2.2, 3.5 |
| dpkg conffiles | change-detection + preserve + report | 0.6–0.9 (CUST) |
| Gentoo Portage `CONFIG_PROTECT` | the export-directory concept | 0.7 |
| Android A/B seamless updates | build-new-slot, atomic switch, keep-old-slot | 0.3–0.5 (VENV), generalizing `deploy/update.sh`'s existing `venv.new`/aside pattern |
| Expand/contract (ParallelChange) | the multi-release authoring rule | 2.4 (MIG-5, MIG-14) |
| Chrome release channels | ledger-ahead refusal, no downgrades | 2.5 (MIG-4) |
| Postgres transactional DDL | atomic step + ledger row | 3.5 (MIG-5) |
| `pg_dump`/`pg_restore` custom format | backup/restore as a tested primitive | 3.8–3.10 (MIG-6/7) |
| Existing in-repo precedent | `sudo -n -u postgres pg_dump -Fc` / `pg_restore --no-owner` | `deploy/vm/extract-stage-vm.sh:16`, `deploy/vm/inject-restore-vm.sh:32` |

### New and touched files

| Path | Status | First phase |
|---|---|---|
| `deploy/lib/venv_stamp.sh` | new | 0 |
| `deploy/lib/custom_export.sh` | new | 0 |
| `deploy/lib/provenance.sh` | new — R1: owns `build_release_artifact`, `read_release_commit`, and the transitional `resolve_release_sha` | 0 |
| `deploy/lib/state_preserve.sh` | new — R4: the one shared code-clear function, both callers | 0 |
| `deploy/update.sh` | edit | 0 |
| `deploy/python.sh` | edit — R1: install-payload fetch switches to the release-asset URL (`:120-130`); R4: reinstall clear calls the shared function (`:87-90`) | 0 |
| `utils/release_identity.py` | edit — R1: `RELEASE_ASSET_URL` replaces `TAG_TARBALL_URL` at the release-fetch site; REL-1(a): `verify_release_identity()` | 0 |
| `ui/tui/update.py` | edit — R1: update-payload fetch switches to the release-asset URL (`:86`, `_download_release`) | 0, 4 |
| `deploy/RELEASE.md` | edit — R1: the tag-time artifact-build step; R7: the REL-1 gate; R8: the transition-release plan | 0, 5 |
| `deploy/bootstrap_adopt.sql` | new | 1 |
| `deploy/mira_service_schema.sql` | edit | 1 |
| `deploy/migrations/BASELINE` | new | 1 |
| `deploy/migrations/AGENTS.md` | new | 1 |
| `deploy/migrations/V####__slug.sql\|.py` | new | 5 |
| `deploy/migrations/R__slug.sql\|.py` | new | 6 |
| `utils/schema_migrations.py` | new | 2 |
| `utils/userdata_migrations.py` | new | 7 |
| `scripts/migrate.py` | new — R2: `--adopt-drifted` | 2 |
| `clients/vault_client.py` | edit | 1 |
| `utils/power_on_self_test.py` | edit | 1, 4 |
| `ui/tui/__main__.py` | edit | 4 |
| `ui/tui/migrate.py` | new | 4 |
| `deploy/docker/scripts/init-mira.sh` | edit — R3: migrate at every container start | 4 |
| `deploy/AGENTS.md` | edit — R5: the PostgreSQL floor statement corrected to 17 | 1 |
| `tests/fixtures/scratch_db.py` | edit | 1 |
| `BREAKING.md` | rewrite — R8: ships with the transition release carrying its narrowed meaning; removed in the next release | 5 |

Not touched by any ruling: `deploy/deploy.sh`'s bootstrap `git clone --branch
<tag>` (`deploy/deploy.sh:64-77`) and `install.sh`'s clone (`install.sh:135`). A
clone of the tag is its own provenance — `git rev-parse HEAD` inside it is the
tagged commit — so R1 leaves both alone. Only the two **tarball** fetch sites
move to the release asset.

Runner-library placement rationale (binding): `scripts/AGENTS.md` Rule 1 makes
every file there a thin CLI shim, so the migration logic cannot live in
`scripts/migrate.py`; `deploy/` is not an importable package, so the logic
cannot live beside the steps. `utils/` is the only importable home. Steps stay
flat in `deploy/migrations/` per MIG-1; the runner loads them with
`importlib.util.spec_from_file_location`.

---

## Phase 0 — independent trio: venv stamps, custom-build export, provenance

Goal:        ship VENV + CUST + PROV with no migration mechanism present (CROSS-1).
Preconditions: none beyond a clean working tree. Touches only `deploy/update.sh`, `deploy/python.sh`, `deploy/lib/`, `deploy/RELEASE.md`, `deploy/docker/scripts/init-mira.sh`, `ui/tui/update.py`, `ui/tui/__main__.py`, `utils/release_identity.py`.

Steps:

1. `deploy/lib/venv_stamp.sh` (new). Two functions, sourced after `output.sh`
   per `deploy/lib/AGENTS.md` source-order rule.
   - `write_venv_stamp <venv-dir> <requirements-file> <base-python> <extras-csv>`
     writes `<venv-dir>/.mira-build-stamp` as `key=value` lines:
     `base_python`, `base_python_version`, `requirements_sha256`,
     `requirements_path`, `extras` (sorted CSV of `torch`,
     `sentence-transformers`, `playwright`), `built_at`. Mode 0644.
   - `venv_stamp_matches <venv-dir> <requirements-file> <base-python> <extras-csv>`
     returns 0 on an exact match of `base_python_version` +
     `requirements_sha256` + `extras`, 1 on mismatch, 2 on an absent or
     unreadable stamp. It additionally runs `"<venv-dir>/bin/python3" -c 'import sys'`
     and returns 1 when the interpreter fails to start (VENV-2 force-build).
   - Stamp records **build inputs**, not live package state — the escape hatch
     is step 5's `--rebuild-venvs` (VENV-4).
2. `deploy/python.sh` — write the server-venv stamp at its creation site. The
   `Creating virtual environment` block (`deploy/python.sh:205-217`) and the
   `Reusing existing virtual environment` branch both fall through to Step 5;
   add `write_venv_stamp venv requirements.txt "$PYTHON_CMD" "$EXTRAS_CSV"`
   immediately after the Step 5 `pip install -r requirements.txt` completes
   (`deploy/python.sh:246-268`), where `EXTRAS_CSV` is assembled from
   `CONFIG_EMBEDDING_PROVIDER=local` (torch, sentence-transformers) and
   `CONFIG_INSTALL_PLAYWRIGHT=yes` (playwright).
3. `deploy/python.sh` Step 5b (`deploy/python.sh:294-313`) — write the TUI-venv
   stamp: `write_venv_stamp /opt/mira/tui-venv /opt/mira/app/ui/tui/requirements.txt "$PYTHON_CMD" ""`.
   **See CONFLICTS C-1**: the requirement names `deploy/finalize.sh` as this
   creation site; source puts it in `deploy/python.sh`.
4. `deploy/update.sh:build_venv` (`deploy/update.sh:154-172`) — append a
   `write_venv_stamp "$dest" "$reqfile" "$PY" "$extras"` call as the function's
   last statement, `extras` passed in by the caller. Source
   `deploy/lib/venv_stamp.sh` beside the existing `deploy/lib/output.sh` and
   `deploy/lib/services.sh` sources at `deploy/update.sh:64-65`.
5. `deploy/update.sh` — VENV-2 conditional build. Replace the unconditional
   block at `deploy/update.sh:174-221` with:
   - Parse a `--rebuild-venvs` flag from `$3..` (VENV-4); the existing
     positional contract is `update.sh <release-tree> <tag>`, so the flag is
     optional-third-onward and `ui/tui/update.py:_gate_and_hand_off`
     (`ui/tui/update.py:122-149`) passes it through when the operator supplied it.
   - `SERVER_EXTRAS` = the CSV derived from `HAD_TORCH`/`HAD_ST`/`HAD_PLAYWRIGHT`
     (already detected at `deploy/update.sh:118-131`).
   - `venv_stamp_matches "${APP_DIR}/venv" "${TREE}/requirements.txt" "$PY" "$SERVER_EXTRAS"`
     → rc 0 sets `SKIP_SERVER_VENV=yes`; rc 1 or 2 builds. Same for
     `"${TUI_VENV}"` against `${TREE}/ui/tui/requirements.txt` with empty extras.
   - Print one line per venv: `server venv: reusing (build inputs unchanged)` or
     `server venv: rebuilding (<reason>)`, reason ∈ {`stamp absent`,
     `requirements changed`, `extras changed`, `base python changed`,
     `interpreter dead`, `--rebuild-venvs`}.
   - A skipped venv is never `mv`'d: the user's `pip install`ed packages survive
     (VENV-2).
6. `deploy/update.sh` — VENV-3. Move the TOAST shim copy
   (`deploy/update.sh:210-216`) out of the build branch to run
   **unconditionally** against `${APP_DIR}/venv` after the swap, so a skipped
   rebuild still refreshes `deploy/_mira_log_levels.py` and
   `mira-log-levels.pth` in the live site-packages.
7. `deploy/update.sh` phase 5 (`deploy/update.sh:261-272`) — VENV-5. Guard each
   `mv` on the corresponding `SKIP_*_VENV`: no aside copy, no swap, and no
   `rm -rf` of a nonexistent aside at `deploy/update.sh:353-359`. Rewrite both
   rollback `print_info` lines (`deploy/update.sh:326`, `:346`) to emit only the
   `mv` commands for venvs that were actually swapped, assembled into a
   `ROLLBACK_VENV_CMDS` string during phase 5.
8. `deploy/lib/custom_export.sh` (new). CUST-1/CUST-2/CUST-6. One
   implementation, called by both `deploy/update.sh` phase 6 and
   `deploy/python.sh`'s reinstall path.
   - `official_baseline_commit <tree>` → the newest commit whose subject matches
     `^(install|update) ` (`git -C <tree> log -1 --format=%H --grep='^\(install\|update\) '`).
     Empty output ⇒ no official baseline ⇒ return 1.
   - `detect_custom_delta <tree> <baseline>` → three lists on stdout, NUL-safe:
     `git diff --name-status <baseline>` for tracked adds/modifies/**deletes**
     (deletions first-class, CUST-1), plus
     `git ls-files --others --exclude-standard` for untracked files. The exclude
     file is `deploy/python.sh:159-171`'s, so `data/`, `venv/`, `.env`, `logs/`,
     `scratch/` never appear.
   - `export_custom_build <tree> <baseline> <tag> <dest-root>` → writes
     `<dest-root>/mira-custom-<STAMP>/` containing (a) full copies of every
     added/modified file preserving relative paths, (b) `custom.patch` =
     `git diff <baseline>` (unified), (c) `RESTORE_MY_CUSTOM_BUILD.txt`.
     `<dest-root>` is `$HOME/Desktop` when it is a writable directory, else
     `$HOME`. Catalog contents, in this order: per-file status;
     `sha256sum` of each exported file; the deletion list with re-delete
     instructions (the overlay reinstates deleted files — the catalog names each
     one and the `rm` that removes it again); `git log --format='%H %s' <baseline>..HEAD`
     (the custom commit list); the baseline tag + SHA the mods were written
     against; and restore semantics written for a coding agent to execute
     (CUST-7: no blind-restore subcommand exists, by design).
   - `print_custom_alert <export-dir> <file-count>` → CUST-4 banner, printed
     only when the delta is nonempty.
   - `record_venv_drift <old-venv> <requirements-file> <catalog>` → CUST-3.
     Called only when the corresponding `SKIP_*_VENV` is `no` (a rebuild will
     actually occur). Diffs `<old-venv>/bin/python3 -m pip freeze` against the
     new requirements file and appends a `## Venv package drift` section to the
     catalog: packages present in the old venv that the new requirements do not
     name (each with its frozen version), and requirement lines whose frozen
     version will change. Hand-installed extras (`torch`,
     `sentence-transformers`, `playwright`, plus anything the operator
     `pip install`ed) are what this section exists to name. A skipped venv
     writes nothing — its packages are preserved in place (VENV-2).
9. `deploy/update.sh` phase 6 (`deploy/update.sh:274-302`) — CUST-5. **Delete**
   the `git stash push --include-untracked` block at `deploy/update.sh:277-281`
   and the same block in `deploy/python.sh:74-79`; call
   `export_custom_build` before the code clear — which is step 19's
   `clear_code_preserving_state`, replacing the inline `find … -maxdepth 1` at
   `deploy/update.sh:286-290` and `deploy/python.sh:87-90` (R4).
   Verification that retiring the stash is safe (CUST-5, stated here as
   required): `tools/implementations/bash_tool.py:_load_app_tree`
   (`tools/implementations/bash_tool.py:659-684`) reads only
   `<tree>/.git/info/exclude` and derives the protected top-level name set from
   it. It calls `git status` nowhere and reads no stash. Its consumer
   `_classify_app_tree_path` (`tools/implementations/bash_tool.py:716-737`)
   classifies a path as editable when its top-level segment is **not** in the
   exclude set. Retiring the stash therefore changes nothing `_load_app_tree`
   observes: the exclude file is untouched, the protected set is identical, and
   a dirty tree stays editable exactly as a clean one is. Witness:
   `probe_bashtool_no_stash_dependency` below.
10. `deploy/lib/provenance.sh` (new). PROV-1/PROV-2 + REL-1(a). **R1: releases
    are self-describing artifacts.** Provenance travels *in* the payload, not
    beside it, and the GitHub auto-archive path is retired.
    - `build_release_artifact <tag> <dest-dir>` — the tag-time artifact builder
      `deploy/RELEASE.md` invokes (step 18). Sequence, every step fatal:
      1. `git archive --format=tar.gz --prefix="mira-${tag#v}/" -o <tmp>.tar.gz "<tag>"`
      2. extract into a scratch directory
      3. `git rev-parse "<tag>^{commit}"` → write the 40-hex SHA as the single
         line of `RELEASE_COMMIT` at the extracted tree's root
      4. re-tar to `<dest-dir>/mira-${tag#v}.tar.gz`, preserving the
         single-top-level-directory shape GitHub's auto-archive had, so
         `ui/tui/update.py:_extract_release`'s one-root assertion
         (`ui/tui/update.py:89-105`) and `deploy/python.sh:121`'s
         `SRC_DIR=/tmp/mira-${RELEASE_TAG#v}` derivation both hold unchanged.
      5. **REL-1(a) gate**: call `verify_release_identity(repo_root, tag,
         <extracted root>)` (step 21) and **produce no asset** when it returns
         `ok=False` — the scratch directory is removed and the exit is nonzero,
         naming the failed check and both observed values. From Phase 5 step 4
         onward this same call site also enforces REL-1(b) by invoking
         `python3 -m scripts.migrate lint` (Phase 2 step 7 — DB-free, so the bash
         builder needs neither a database nor Vault) and refusing the asset on a
         nonzero exit.

    `RELEASE_COMMIT` is injected **at artifact build time and never committed
    to the repo** — a committed file cannot contain its own tag's SHA. That is
    the C-2 chicken-and-egg, and it is why the artifact, not the repo, carries
    provenance. `build_release_artifact` refuses to run when
    `git ls-files --error-unmatch RELEASE_COMMIT` succeeds.
    - `read_release_commit <tree>` → the 40-hex contents of
      `<tree>/RELEASE_COMMIT`. Absent, unreadable, or not 40 hex ⇒ return 1 and
      name the state: `pre-provenance install`.
    - `resolve_release_sha <tag>` → `git ls-remote https://github.com/taylorsatula/mira-OSS.git "refs/tags/<tag>^{}"`,
      first field. Bounded: `timeout 30`. Empty ⇒ return 1.
      **TRANSITIONAL — pre-provenance installs only.** This is the adoption path
      for an install whose payload predates the self-describing artifact and so
      carries no `RELEASE_COMMIT`. It is not a fallback for a provenance-bearing
      tree: `read_release_commit` wins unconditionally and a tree carrying
      `RELEASE_COMMIT` never consults the remote. Its removal is a deliberate
      operator decision, recorded as such in `deploy/RELEASE.md` (step 18) and
      `deploy/lib/AGENTS.md` (step 17) — never a silently permanent second path.
    - `resolve_local_sha <tree>` → `git -C <tree> rev-parse HEAD`, plus
      `local-dirty` when `git -C <tree> status --porcelain` is nonempty.
    - `official_commit_message <kind> <label> <sha>` → `"<kind> <label> @ <sha>"`,
      `kind` ∈ {`install`, `update`}. For a nightly, `label` is
      `nightly-<shortsha>` and `<sha>` is `NIGHTLY_SHA` from the environment
      (already resolved by `ui/tui/update.py:_run_nightly_update`,
      `ui/tui/update.py:151-190`) — never re-derived.
11. `deploy/python.sh:173-181` — replace the `INSTALL_COMMIT_MSG` assembly with
    `official_commit_message install "$RELEASE_TAG" "$INSTALL_SHA"`, where
    `INSTALL_SHA` resolves in this order (R1):
    1. `--local`: `resolve_local_sha "$LOCAL_SOURCE_DIR"` (suffix
       ` (local tree)` retained, plus `-dirty` when the source tree is dirty —
       PROV-2: a `--local` baseline is non-pristine by construction).
    2. otherwise: `read_release_commit "$SRC_DIR"` — the artifact's own
       `RELEASE_COMMIT`, present on every release cut from Phase 0 onward.
    3. otherwise, and only when step 2 named the state `pre-provenance install`:
       `resolve_release_sha "$RELEASE_TAG"` (transitional, step 10). The branch
       prints `pre-provenance install: resolving the tag over the network`.
    A failure of all three is a hard `exit 1`, not an empty SHA.
12. `deploy/update.sh:296-302` — PROV-1. Replace the `else` branch
    (`deploy/update.sh:301`, "Install tree is not a git repo — no update record
    written.") with `print_error` + `exit 1`. The install tree being a git repo
    is an invariant (`deploy/python.sh:152-155` `git init`s it); the
    proceed-without-record path is deleted, not annotated. Move this check into
    phase 1 preflight (`deploy/update.sh:83-110`) so it fires **before** the
    venvs are built.
13. `deploy/update.sh:296-300` — commit with
    `official_commit_message update "$TAG" "$NEW_SHA"`, `NEW_SHA` resolved by
    `read_release_commit "$TREE"` (R1: the payload names its own commit), falling
    back to `resolve_release_sha "$TAG"` only when that names a
    `pre-provenance install` — the transitional path, step 10 — and to
    `NIGHTLY_SHA` when `NIGHTLY=1` (already resolved by `ui/tui/update.py`,
    never re-derived). The existing `--allow-empty` and the non-fatal
    `|| print_warning` stay: the commit is a record, and a failed record must not
    abort a healthy swap.
14. `deploy/docker/scripts/init-mira.sh` — PROV-3. The container has no repo.
    Write `/opt/mira/app/.mira-image-stamp` in `main()`
    (`deploy/docker/scripts/init-mira.sh:358`) with `image_id`, `built_at`,
    `version` (from `/opt/mira/app/VERSION`), and `release_tag` when the image
    carries one. Add a matching `LABEL org.opencontainers.image.revision` to
    `deploy/docker/Dockerfile`.
15. `ui/tui/update.py` — PROV-4. Add `run_status() -> int` beside `run_update`
    (`ui/tui/update.py:193`), reading only git and the ledger (absent ledger ⇒
    print `ledger: not adopted`). Reports: last official commit + SHA; the
    uncommitted diff against it (`git diff --stat` + `git ls-files --others --exclude-standard`);
    custom commits since it (`git log --oneline <baseline>..HEAD`); labeled
    stashes (`git stash list`); ledger head + pending count. Stdlib only — no
    psycopg, no hvac (see CONFLICTS C-3): the ledger read is
    `git`-free and the pending count is deferred to Phase 2's `mira status`,
    which runs on the server venv.
16. `ui/tui/__main__.py:64-69` — add a `status` subcommand to the pre-argparse
    dispatch, same shape as `update`.
17. Map edits, same commit (`docs/AGENTS_MAP_SPEC.md` trigger table):
    - `deploy/lib/AGENTS.md` — four `## Files` bullets (`venv_stamp.sh`,
      `custom_export.sh`, `provenance.sh`, `state_preserve.sh`) and a `## Rules`
      bullet stating the stamp's build-inputs-not-live-state semantics and the
      `--rebuild-venvs` escape hatch, plus a `## Rules` bullet stating that
      `RELEASE_COMMIT` is injected at artifact build time and never committed,
      and that `resolve_release_sha` is **transitional** with its removal an
      operator decision (R1).
    - `deploy/AGENTS.md` — rewrite the `update.sh` bullet: conditional venv
      rebuild, custom-build export replacing the stash, the shared
      `clear_code_preserving_state` replacing the inline `find` (R4), hard
      refusal on a non-repo tree, `@ <sha>` in every official commit resolved
      from the payload's own `RELEASE_COMMIT`. Rewrite the `python.sh` bullet's
      stash clause (behavior removed ⇒ delete the statement), its
      narrower-preserve-set clause (R4 supersedes it), and its
      "wget of the pinned tag tarball" clause (R1: the release **asset**).
      Rewrite the `RELEASE.md` bullet for step 18.
    - `utils/AGENTS.md:28` — the `release_identity.py` bullet: `TAG_TARBALL_URL`
      → `RELEASE_ASSET_URL`, and `verify_release_identity()` added. **Line-count
      check (C-5)** — this is a seventh edit to a 76-line map.
    - `deploy/docker/scripts/AGENTS.md` — `init-mira.sh` bullet gains the
      image-stamp file.
    - `ui/tui/AGENTS.md` — `__main__.py` and `update.py` bullets gain `status`;
      the `update.py` bullet's "downloads the release tarball" clause names the
      release **asset** (R1).
    - `ui/AGENTS.md:11` — the "one exception" clause stays true (`update.py`
      still imports only stdlib `utils/release_identity`); verify, do not edit
      unless step 15 or step 21 imported something else.
    - Root `AGENTS.md` map registry — no new directory yet (`deploy/lib/` is
      already registered).
    - `deploy/lib/AGENTS.md` is 38 lines; four bullets plus two Rules keep it
      under 60.
18. `deploy/RELEASE.md` — R1 + REL-1 wiring. Four edits:
    - **New "Release artifact" section, executed at tag time.** The published
      payload is built by `deploy/lib/provenance.sh:build_release_artifact
      <tag> <dest>` (step 10), attached as the release's asset named
      `mira-<tag-without-v>.tar.gz`, and is the only tarball any install or
      update fetches. GitHub's auto-generated `archive/refs/tags/` tarball is
      **retired as a payload source** — it carries no `RELEASE_COMMIT` and
      cannot. State the reason in one line: a committed file cannot contain its
      own tag's SHA, so the artifact carries provenance, not the repo.
    - **"Provenance" section**: every official commit carries `@ <sha>`; the SHA
      comes from the payload's own `RELEASE_COMMIT`; `--local` installs record
      source HEAD + a dirty flag; `git ls-remote` resolution is the
      **transitional** path for a `pre-provenance install` and its removal is a
      recorded operator decision.
    - **REL-1 cross-reference**: the release-cut checklist gains the integrity
      gate. The gate's five conditions and their owning phases are tabulated at
      Phase 5 step 4, which rewrites this file's checklist; step 18 adds only
      the pointer, so the gate has one home. State the incremental landing:
      condition (a) is enforced **from this phase** — `build_release_artifact`
      produces no asset when `verify_release_identity` fails — and (b)–(e) join
      the gate at Phase 5 step 4, when the runner and their witnesses exist.
      Authoring-time enforcement is the defense; install-time validation
      (`_check_schema_revision`, Phase 4 step 5) is the backstop.
    - **The manual-recovery block** (`deploy/RELEASE.md:53-60`) wgets
      `archive/refs/tags/X.XX.tar.gz`. It is a tarball fetch site and switches
      to the release-asset URL with the rest (R1); the note about GitHub
      stripping a leading `v` from the extracted directory name stays true
      because `build_release_artifact` reproduces that `--prefix`.
19. `deploy/lib/state_preserve.sh` (new). **R4 — one preserve set, one clear
    function, two callers.** Sourced after `output.sh` per `deploy/lib/AGENTS.md`'s
    source-order rule.
    - `clear_code_preserving_state <app-dir> [--sudo]` runs the
      `find <app-dir> -mindepth 1 -maxdepth 1 … -exec rm -rf {} +` clear with the
      **`deploy/update.sh:286-290` superset** preserve set:
      `venv`, `.env`, `data`, `logs`, `.git`, `run.sh`, `venv*`, `tui-venv*`.
      The set is a single module-level array in this file; neither caller
      restates it.
    - `--sudo` prefixes `find` with `sudo`. `deploy/python.sh` passes it (its
      tree is root-owned until the chown at `deploy/python.sh:137-138`);
      `deploy/update.sh` does not (`/opt/mira` is user-owned).
    - `--print` writes the assembled `find` argv to stdout, NUL-separated, and
      executes nothing. It exists so the `--sudo` composition and the preserve
      set are checkable without a destructive run and without a fake `sudo` on
      `PATH` — probe-ability is a design constraint on the function, not a
      verification-time discovery (root `AGENTS.md`).
    - Call sites, both replacing an inline `find`: `deploy/update.sh:286-290`
      (phase 6) and `deploy/python.sh:87-90` (the reinstall path).
    - **The reinstall path's narrower set is broken-allowed.** A reinstall now
      keeps `run.sh` (previously deleted and regenerated later by
      `deploy/finalize.sh:189`) and keeps any `venv.pre-update-<STAMP>` /
      `tui-venv.pre-update-<STAMP>` aside. The second is what unblocks the D1
      failure path: aside venvs survive a mid-recovery reinstall, so
      `ROLLBACK_VENV_CMDS` (step 7) still has something to `mv` back.
    - `/opt/mira/backups` is outside `<app-dir>` and therefore outside the
      clear's scope on both call sites (C-9's verified fact; re-asserted by
      `probe_backup_retention.py`).
20. `deploy/python.sh:120-130` — **R1: the install payload fetch switches to the
    release asset.** Replace the `wget` at `deploy/python.sh:123-124`
    (`…/archive/refs/tags/${RELEASE_TAG}.tar.gz`) with the release-asset URL
    (`…/releases/download/${RELEASE_TAG}/mira-${RELEASE_TAG#v}.tar.gz`).
    `TARBALL` (`:120`), `SRC_DIR` (`:121`), the extract (`:126-127`), the copy
    (`:129-130`) and the cleanup (`:132-134`) are unchanged: the asset is the
    same single-top-level-directory tarball `build_release_artifact` produces.
    A 404 on the asset is a hard failure naming the tag and the missing asset —
    **never** a silent fall-through to `archive/refs/tags/`. The tarball branch
    has no post-copy file check today (only the `--local` branch does, at
    `deploy/python.sh:97-103`); add one asserting `/opt/mira/app/RELEASE_COMMIT`
    exists and is 40 hex, so an artifact that lost its provenance file fails here
    rather than at step 11's resolution.
    The repo component of the URL stays exactly what each fetch site uses today
    (`deploy/python.sh:124`'s literal `taylorsatula/mira-OSS`); the pre-existing
    divergence from `utils/release_identity.py:21`'s `taylorsatula/mira` is
    out of scope and is not silently reconciled here.
21. `ui/tui/update.py` + `utils/release_identity.py` — **R1: the update payload
    fetch switches to the release asset.**
    - `utils/release_identity.py:22` — `TAG_TARBALL_URL` is **deleted** and
      replaced by
      `RELEASE_ASSET_URL = f"https://github.com/{GITHUB_REPO}/releases/download/v{{tag}}/mira-{{tag}}.tar.gz"`.
      The module stays stdlib-only, so `ui/AGENTS.md:11`'s one-exception
      invariant and the TUI venv's dependency set are both unaffected.
      `BRANCH_TARBALL_URL` (`:24`) is untouched: a nightly is a branch archive,
      not a release, and carries no `RELEASE_COMMIT` — its identity is
      `NIGHTLY_SHA` (step 10).
    - `utils/release_identity.py` — add
      `verify_release_identity(repo_root, tag, artifact_root) -> IdentityReport`,
      the mechanical form of REL-1(a). A frozen dataclass / Pydantic model with
      `ok: bool` and one field per check, each carrying its two observed values:
      `tag_matches_version` (`tag == "v" + (repo_root / "VERSION").read_text().strip()`,
      the relationship `ui/tui/update.py:237`'s `tag = f"v{latest}"` already
      relies on); `artifact_matches_repo` (the artifact tree's `VERSION` equals
      the repo's); `release_commit_matches_tag`
      (`read_release_commit`'s value equals `git rev-parse "<tag>^{commit}"`).
      No check defaults to true; an unreadable input is a failed check naming
      the path.
    - `ui/tui/update.py:86` — `_download_release` formats `RELEASE_ASSET_URL`
      instead of `TAG_TARBALL_URL`. `_extract_release` (`:89-105`) is unchanged
      and its one-top-level-directory assertion is what step 10's re-tar must
      satisfy. A 404 names the tag and the missing asset; there is no
      `archive/refs/tags/` fallback.

Verification:

| Probe | Executes | Asserts |
|---|---|---|
| `/tmp/probe_venv_stamp.sh` (VER-6) | Builds two throwaway venvs in `/tmp` from `ui/tui/requirements.txt` (5 pinned packages — the cheap requirements file), writes stamps via the real `deploy/lib/venv_stamp.sh`, then calls `venv_stamp_matches` | rc 0 on an identical re-check; rc 1 after appending a comment-free line to a copy of the requirements file; rc 1 after changing the extras CSV; rc 2 after `rm .mira-build-stamp`; rc 1 after `chmod -x <venv>/bin/python3` (interpreter-dead force-build). Control: a venv whose stamp matches is NOT rebuilt — assert `venv.new` was never created by a dry run of the phase-3 block. `PROBE PASSED` |
| `/tmp/probe_custom_export.sh` (CUST-1/2/4/7, VER-6) | `git init` a scratch tree in `/tmp`, commit `install vTEST @ abc123`, then create all four delta cases: a new untracked file, a modification to a tracked file, a **deletion** of a tracked file, and a committed-then-modified file; source `deploy/lib/custom_export.sh` and run `export_custom_build` | The export directory exists at the resolved `<dest-root>`; every added/modified file is byte-identical to the tree copy (`sha256sum` compare against the catalog); the catalog lists the deleted file with a re-delete `rm` line; the custom-commit list names the intermediate commit; `git apply --check custom.patch` succeeds against a fresh checkout of the baseline commit; `print_custom_alert` produced output. Control: a tree with an empty delta produces **no** directory and **no** banner (CUST-4). `PROBE PASSED` |
| `/tmp/probe_venv_drift.sh` (CUST-3, VER-6) | Builds a throwaway venv in `/tmp` carrying one package the new requirements do not name (`rich`) and one they do at a different pin; runs the real `record_venv_drift` against a real catalog file; then runs it again with the skip flag set | Rebuild arm: the catalog gains a `## Venv package drift` section naming `rich` with its frozen version as present-but-unnamed, and naming the pin change for the other package. Skip arm: the catalog is byte-identical before and after. Control: a venv whose freeze exactly matches the requirements produces an empty drift list, and the section is still written saying so — an omitted section and an empty one must be distinguishable. `PROBE PASSED` |
| `/tmp/probe_prov_status.sh` (PROV-4, VER-6) | A `/tmp` install tree with a real `install vTEST @ <sha>` baseline commit, one custom commit on top, one uncommitted modification, one untracked file, and one labeled stash from the pre-Phase-0 mechanism; runs `mira status` via `/opt/mira/tui-venv`-shaped invocation (`python3 -m ui.tui status`) | Output names: the baseline commit subject and its 40-hex SHA; the uncommitted diff against it (both the modified and the untracked file); the one custom commit since it; the labeled stash by its message; and `ledger: not adopted` (Phase 0 has no ledger yet). Control: a tree with no delta since the baseline reports an empty custom section rather than omitting it. Stdlib only — assert the module imports without `psycopg` and without `hvac` by running it under an interpreter where neither is installed. `PROBE PASSED` |
| `/tmp/probe_prov_commit.sh` (PROV-2, VER-6) | Runs `official_commit_message` for all four cases (tag install, tag update, nightly, `--local` dirty); `read_release_commit` against a `/tmp` tree carrying a real `RELEASE_COMMIT` and against one without; and the **transitional** `resolve_release_sha v2026.10.06-2.0` against the real remote, bounded `timeout 30` | Messages are exactly `install <tag> @ <40-hex>`, `update <tag> @ <40-hex>`, `update nightly-<12hex> @ <40-hex>`, `install <tag> @ <40-hex> (local tree) local-dirty`. `read_release_commit` returns the 40-hex SHA when the file is present, and returns 1 with the state named `pre-provenance install` when it is absent, empty, or 39 hex — the three malformed arms each witnessed. **Precedence (R1)**: a tree carrying `RELEASE_COMMIT` resolves without any network call — assert no `git ls-remote` subprocess is spawned by wrapping `git` on `PATH` with a witness script that fails the probe if invoked. `resolve_release_sha` returns a 40-hex SHA for a tag that exists and returns 1 for `refs/tags/vdoes-not-exist`. Reports BLOCKED (exit 3) when the network is unreachable, distinguished from a failed lookup. `PROBE PASSED` |
| `/tmp/probe_release_artifact.sh` (PROV-2, **REL-1(a)**, VER-6) | A `/tmp` bare clone of the checkout with one real annotated tag created on a real commit whose `VERSION` matches the tag; runs the real `build_release_artifact <tag> <dest>`, extracts the produced asset into a second `/tmp` root, and runs the real `verify_release_identity` over all three | The asset exists at `<dest>/mira-<tag-without-v>.tar.gz`; it extracts to **exactly one** top-level directory (the shape `ui/tui/update.py:_extract_release` requires); `RELEASE_COMMIT` is present at that root and equals `git rev-parse <tag>^{commit}` byte for byte; every other file is byte-identical to `git archive <tag>`'s output (diff the two extractions with `RELEASE_COMMIT` excluded — the injection adds one file and changes nothing else). **Never committed**: `git ls-files --error-unmatch RELEASE_COMMIT` fails in the source repo, and `build_release_artifact` refuses to run when it is made to succeed (create the file, commit it, assert the refusal names it). `verify_release_identity` returns `ok=True` with all three checks true; three negative controls each flip exactly one field — a `VERSION` edited to a different value flips `tag_matches_version`, an artifact built from a different commit flips `release_commit_matches_tag`, an artifact whose `VERSION` was overwritten flips `artifact_matches_repo`. A 404-shaped control: an asset name that does not exist is reported as a missing asset, never silently re-fetched from `archive/refs/tags/`. Teardown: both `/tmp` roots removed; assert no residue. `PROBE PASSED` |
| `/tmp/probe_state_preserve.sh` (CUST-6, **R4**, VER-6) | Two user-owned `/tmp` fake install roots populated with every name in the preserve set plus `main.py`, `README.md`, and a `scratch/` directory; sources the real `deploy/lib/state_preserve.sh`, runs `clear_code_preserving_state <root>` on each (no `--sudo`: a `/tmp` root needs none, and nothing is faked), then runs `--print` with and without `--sudo` | Both roots end in **exactly** the same surviving set: `venv`, `.env`, `data`, `logs`, `.git`, `run.sh`, and every `venv*`/`tui-venv*` aside (`venv.pre-update-<STAMP>`, `tui-venv.pre-update-<STAMP>`, `venv.new`) — and nothing else. The narrower reinstall-path set is gone: `run.sh` and both asides survive the reinstall-shaped clear, which is what the D1 rollback depends on. `--print` shows the argv: identical preserve set in both, `sudo` as argv0 only in the `--sudo` arm, `-mindepth 1 -maxdepth 1` and the target root present in both, so the sudo composition is witnessed without a fake `sudo` on `PATH`. **One set, not two**: grep both call sites in the edited `deploy/update.sh` and `deploy/python.sh` and assert neither contains an inline `find … -maxdepth 1` clear any more, and that the eight preserve names appear in `deploy/lib/state_preserve.sh` only. `/opt/mira/backups` scope control: assert the printed target argument is the app dir, so a sibling of it is unreachable. Teardown: both roots removed; assert no residue. `PROBE PASSED` |
| `/tmp/probe_update_sh_scratch.sh` (PROV-1, VENV-2/5, VER-6) | End-to-end: a `/tmp` release tree (copy of the checkout with a bumped `VERSION`) against a `/tmp` fake install root with `MIRA_APP_DIR`/`MIRA_TUI_VENV` overrides, `CANONICAL=no` so no service is touched | Non-repo install root ⇒ hard refusal before any venv build (assert `venv.new` absent, exit 1). Repo root with matching stamps ⇒ `Skipping` lines for both venvs, no aside directories created, and the rollback print contains no `mv` for a venv that was not swapped (VENV-5). Repo root with a modified `requirements.txt` ⇒ that venv rebuilds and the other does not. `PROBE PASSED` |
| `/tmp/probe_bashtool_no_stash_dependency.py` | Imports the real `tools.implementations.bash_tool`, reads `_APP_TREE`, and calls `_classify_app_tree_path` on (a) a tracked file, (b) `data/users/x`, (c) the tree root — with the tree dirty and with it clean | `_load_app_tree()` returns the same `(root, frozenset)` in both states; `_classify_app_tree_path` returns `""` for (a), `"app-untracked-delete"` for (b), `"app-root-delete"` for (c) in both states. This is the CUST-5 witness that retiring the stash changes nothing `bash_tool` observes. `PROBE PASSED` |
| `/tmp/probe_no_new_deps.py` | `git diff --stat` of the phase against `requirements.txt` and `ui/tui/requirements.txt` | Both files unmodified. Every new `import` in a new `.py` file resolves to stdlib, `psycopg`, or `hvac`; `utils/release_identity.py` remains stdlib-only after step 21 (assert its import set is unchanged in kind — no new third-party name), which is what keeps `ui/AGENTS.md:11`'s one-exception invariant and the TUI venv's dependency set intact. `PROBE PASSED` |

Done when:

- [ ] All ten Phase-0 probes report `PROBE PASSED` (or a named UNVERIFIED with its covering probe).
- [ ] `deploy/update.sh` header (`:1-58`) rewritten: phases renumbered, the "Privileges: none" claim re-checked against step 12 (still true in Phase 0), and the v1-never-auto-rolls-back sentence left intact until D1 lands in Phase 3.
- [ ] `grep -n "stash push" deploy/update.sh deploy/python.sh` returns nothing.
- [ ] `grep -n "not a git repo" deploy/update.sh` returns nothing.
- [ ] `grep -rn "archive/refs/tags" deploy/ utils/ ui/ install.sh README.md docs/` returns nothing (R1: the auto-archive payload path is retired at all three sites — `deploy/python.sh:124`, `utils/release_identity.py:22`, `deploy/RELEASE.md:55`).
- [ ] `grep -rn "TAG_TARBALL_URL" .` returns nothing (R1: ablated, not aliased).
- [ ] `grep -c "clear_code_preserving_state" deploy/update.sh deploy/python.sh deploy/lib/state_preserve.sh` ≥ 1 in each, and the eight preserve names appear in `deploy/lib/state_preserve.sh` only (R4).
- [ ] `grep -rn "find .*-maxdepth 1" deploy/update.sh deploy/python.sh` returns nothing (R4: both inline clears are gone, replaced by the shared call).
- [ ] `grep -n "pre-provenance install" deploy/lib/provenance.sh deploy/python.sh deploy/update.sh deploy/RELEASE.md deploy/lib/AGENTS.md` ≥ 1 in each — the transitional state is named at every site that can reach it and marked transitional in both the code and the map (R1).
- [ ] `bash -n` clean on `deploy/update.sh`, `deploy/python.sh`, and all four new `deploy/lib/*.sh`.
- [ ] `deploy/lib/AGENTS.md`, `deploy/AGENTS.md`, `deploy/docker/scripts/AGENTS.md`, `ui/tui/AGENTS.md`, `utils/AGENTS.md` edited in the same commit; `wc -l` under 60 on each except `utils/AGENTS.md`, whose count is recorded per C-5.
- [ ] The `docs/AGENTS_MAP_SPEC.md` anchor audit and citation audit run clean over the five edited maps.
- [ ] Phases 1–8 are not started (CROSS-1: this trio ships independently).

---

## Phase 1 — prerequisites: ownership, ledger table, adoption bootstrap, secrets manifest

Goal:        make `mira_admin` able to own and alter the schema, create the ledger, and enumerate required Vault secrets.
Preconditions: Phase 0 shipped. Local Postgres reachable with a superuser; `psql` on PATH; pgvector installed (`tests/fixtures/scratch_db.py:_require_pgvector`).

Steps:

1. `deploy/bootstrap_adopt.sql` (new, MIG-8). Runs **only** through the
   superuser seam (Linux `sudo -u postgres psql -d mira_service -v ON_ERROR_STOP=1 -f`;
   macOS plain `psql`, matching `deploy/postgresql.sh:110` vs `:119`). Contents,
   in order, every statement idempotent:
   - `DO` block transferring ownership to `mira_admin` of every table, index,
     sequence, and function in `public`, driven off `pg_class`/`pg_proc` — a
     generated `ALTER … OWNER TO mira_admin` per row via
     `format()` + `EXECUTE`, never a transcribed object list.
   - `GRANT CREATE ON SCHEMA public TO mira_admin;` (PG15+ removed the implicit
     grant; the schema file's `GRANT USAGE ON SCHEMA public TO mira_dbuser` at
     `deploy/mira_service_schema.sql:53` stays as-is).
   - `CREATE TABLE IF NOT EXISTS schema_migrations (…)` — the step-2 DDL,
     duplicated here because an existing install has no ledger and the schema
     file's `$schema_precondition$` block (`deploy/mira_service_schema.sql:36-47`)
     refuses to run against a non-empty database. **The duplication is
     generated, not transcribed**: `bootstrap_adopt.sql` `\i`-includes nothing,
     so the DDL text is emitted by `utils/schema_migrations.py:ledger_ddl()`
     into both files at release-cut time (Phase 5 step 5.5 wires the generator).
     Until then, the two copies are a named twin contract and both are listed in
     `deploy/migrations/AGENTS.md`.
   - `CREATE SCHEMA IF NOT EXISTS` nothing else; no data touched.
   - Stamping is **not** in this file — the runner owns it (step 3.4), so there
     is one stamping path.
2. `deploy/mira_service_schema.sql` — add the ledger table (MIG-2) after the
   grants block (`:838-863`), plus `GRANT SELECT ON schema_migrations TO mira_dbuser;`.
   DDL:
   ```sql
   CREATE TABLE schema_migrations (
       revision    TEXT PRIMARY KEY,
       name        TEXT NOT NULL,
       kind        TEXT NOT NULL CHECK (kind IN ('versioned', 'repeatable')),
       store       TEXT NOT NULL CHECK (store IN ('postgres', 'vault', 'env', 'valkey')),
       checksum    TEXT NOT NULL,
       applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
       applied_by  TEXT NOT NULL,
       duration_ms INTEGER NOT NULL DEFAULT 0
   );
   GRANT SELECT ON schema_migrations TO mira_dbuser;
   ```
   No `ENABLE ROW LEVEL SECURITY`: the ledger is not user-scoped, and it must
   NOT be added to `_check_postgres_rls`'s `expected_tables` list
   (`utils/power_on_self_test.py:628-645`).
3. `deploy/mira_service_schema.sql` — amend the header comment block (`:1-14`).
   The line "This is deliberately not a migration" stays true for the file's
   DDL; add one sentence naming `deploy/migrations/` as the post-install chain
   and `schema_migrations` as its ledger. `deploy/AGENTS.md`'s Rule "Do not add
   migration DDL to the schema file" is amended in step 12 — **see CONFLICTS C-4**.
4. `deploy/migrations/BASELINE` (new, MIG-1). Two `key=value` lines:
   `revision=0001` and `fingerprint_sha256=<hash>`. The hash is **generated** by
   `utils/schema_migrations.py:baseline_fingerprint()` over a fresh
   scratch-database apply of `deploy/mira_service_schema.sql` — never
   hand-typed. `revision` is the head revision the schema file is equivalent to
   (VER-1 is the witness that they agree).
5. `deploy/migrations/AGENTS.md` (new; the directory has ≥2 files once Phase 5
   lands steps, and it owns an invariant no ancestor map carries — the coverage
   gate in `docs/AGENTS_MAP_SPEC.md`). Fixed section order. `## Rules` carries:
   the revision sequence is its own monotonic 4-digit counter, never derived
   from `VERSION`, tags, or dates (MIG-13); merged steps are immutable; the
   `BASELINE` fingerprint is generated not transcribed; discovery excludes
   `BASELINE`, `AGENTS.md`, and `__pycache__`. `## Rules` also carries the
   adoption contract (R2): adoption drift is **refused** with exit 2 and a drift
   report, never converged silently; the only way past it is the explicit
   per-invocation `mira migrate --adopt-drifted`; and the override is **recorded
   in the ledger** in the form Phase 3 step 4 defines, so an install that was
   adopted through an override says so forever. `## Files` carries one bullet per
   file including `BASELINE`.
6. Root `AGENTS.md` map registry — add the row
   `| deploy/migrations/ | Versioned and repeatable migration steps, the BASELINE marker, and the authoring contract |`.
7. `deploy/postgresql.sh` — call `bootstrap_adopt.sql` immediately after the
   schema apply (`:221-231`). It is a no-op-shaped call on a fresh install
   (ownership transfer over objects the superuser just created, ledger already
   present from the schema file), and it is the same file an existing install
   runs. One path, two callers (root `AGENTS.md`: Convergent Path Refactoring).
   Add the matching line to `deploy/deploy_database.sh`, which
   `deploy/AGENTS.md` requires be kept consistent with `postgresql.sh`.
8. `tests/fixtures/scratch_db.py` — reproduce the production ownership shape
   (MIG-8's last clause). Two edits:
   - `CREATE DATABASE "{name}"` (`tests/fixtures/scratch_db.py:124`) becomes
     `CREATE DATABASE "{name}" OWNER mira_admin`, matching
     `deploy/postgresql.sh:116`'s `createdb -O mira_admin`.
   - After `_apply_schema`, run `deploy/bootstrap_adopt.sql` through the same
     `psql` subprocess seam `_apply_schema` already uses
     (`tests/fixtures/scratch_db.py:88-107`), so a probe sees production form:
     `mira_admin`-owned objects, `CREATE` on `public`, ledger present.
   - `_ROLES` (`tests/fixtures/scratch_db.py:31-34`) gains **no** `CREATEDB`
     (R6: `mira_admin` stays narrow; `deploy/postgresql.sh:108`'s `ROLE_SQL` has
     none either). Restore therefore resets the `public` schema rather than
     dropping the database (3.9), and `probe_migrate_privileges.py` witnesses the
     absence as a refused `CREATE DATABASE`.
9. `clients/vault_client.py` — MIG-17. Add beside `preload_secrets()`
   (`clients/vault_client.py:204-238`):
   ```python
   REQUIRED_SECRET_MANIFEST: tuple[tuple[str, tuple[str, ...], str], ...] = (
       ("mira/database", ("admin_url", "service_url", "username", "password"),
        "vault kv patch secret/mira/database <field>=<value>"),
       ("mira/services", ("valkey_url", "app_url", "userdata_encryption_key"),
        "vault kv patch secret/mira/services <field>=<value>"),
       ("mira/api_keys", ("provider_key",),
        "vault kv patch secret/mira/api_keys <field>=<value>"),
   )
   ```
   Field lists are copied from the writers, not invented:
   `deploy/postgresql.sh` Step 14 (`secret/mira/database` admin_url/service_url/username/password;
   `secret/mira/services` app_url/valkey_url/userdata_encryption_key/diagnostics_token),
   `deploy/docker/scripts/init-mira.sh:311-312`, and the fields
   `_check_vault` already treats as required
   (`utils/power_on_self_test.py:524-528`). Provider-conditional fields
   (`kagi_api_key`, `embeddings_key`, `systemone_key`, `MIRA_SMTP_*`) are NOT in
   the manifest — `deploy/AGENTS.md`'s `postgresql.sh` bullet documents them as
   conditional, and a manifest entry for a conditional field parks every install
   that opted out. Add `check_required_secrets() -> list[SecretGap]` returning a
   typed `SecretGap(path, field, remediation)` per missing field, and a pure
   `format_secret_gap(gap) -> str` that renders the exact
   `vault kv patch secret/mira/<path> <field>=<value>` command.
10. `utils/power_on_self_test.py` — new required `CheckSpec("vault_secrets_manifest", True, _check_vault_secrets_manifest)`
    registered in `run_pre_server_post()`'s spec list
    (`utils/power_on_self_test.py:283-295`), after `"vault"`. The probe function
    calls `check_required_secrets()` and raises `RuntimeError` whose message is
    the `format_secret_gap()` rendering of every gap — the operator sees the
    exact command, not "secrets missing". Returns
    `{"paths_checked": n, "fields_checked": m}`.
11. `deploy/docker/scripts/init-mira.sh` — the container provisions Vault in
    `init_vault` (`:234`) and never runs `postgresql.sh`. Add a
    `bootstrap_adopt.sql` call in `init_postgresql` (`:132`) right after the
    schema apply at `:203`, through the container's superuser seam
    (`sudo -u postgres psql`, as `:190`/`:194`/`:203` already do).
12. Map edits, same commit:
    - `deploy/AGENTS.md` — **R5: correct the PostgreSQL floor.** The
      `dependencies.sh` bullet (`deploy/AGENTS.md:25`) states "the distro's
      CURRENT postgresql major is used — 18.6 verified live, schema floor is 14
      — not a pinned 17". The floor is corrected to **17**, which is what the
      project actually ships (`deploy/docker/Dockerfile.base:102-104` installs
      `postgresql-17`; `deploy/postgresql.sh` uses `brew --prefix
      postgresql@17`). Keep the two facts distinct in the rewritten clause: the
      **supported floor** is 17, while the Arch *package* remains the distro's
      current major rather than a pinned one. Binding consequence, stated in the
      same bullet: 17 ≥ 15, so `public` is owned by `pg_database_owner` on every
      supported install and `mira_admin` owns the database
      (`deploy/postgresql.sh:116`) — which is what makes the schema-reset restore
      path's `DROP SCHEMA public CASCADE` (3.9 step 4) unambiguous rather than
      version-dependent (C-6, closed by this correction). Breaking-allowed doc
      correction; lands in this phase's commit because this phase is what makes
      the restore path real.
    - `deploy/AGENTS.md` — amend the Rule "Do not add migration DDL to the
      schema file" to name the two permitted non-DDL-migration contents
      (`schema_migrations` ledger + its `GRANT SELECT`); add a `## Files` bullet
      for `bootstrap_adopt.sql`; extend the `mira_service_schema.sql` bullet's
      consumer list with `deploy/bootstrap_adopt.sql` and note the ledger;
      extend the `postgresql.sh` and `deploy_database.sh` bullets with the
      bootstrap call.
    - `deploy/migrations/AGENTS.md` — created (step 5), carrying the R2
      adoption-refusal Rule.
    - Root `AGENTS.md` — map registry row (step 6); the root's
      `## Repo root files` `BREAKING.md` bullet is untouched this phase.
    - `clients/AGENTS.md:5` and `:15` — the `vault_client.py` bullet gains
      `REQUIRED_SECRET_MANIFEST` / `check_required_secrets()` and the Rule gains
      the manifest as the fifth permitted access point's data source.
    - `utils/AGENTS.md:24` — the `power_on_self_test.py` bullet's check
      enumeration gains `vault_secrets_manifest`. **`utils/AGENTS.md` is 76
      lines**; this edit plus Phase 0's `release_identity.py` edit, Phase 7's
      `userdata_migrations.py` bullet and Phase 2/3's `schema_migrations.py`
      bullet cross 80, where `docs/AGENTS_MAP_SPEC.md` requires a split — and
      `utils/` has no subdirectories to split into. **See CONFLICTS C-5.** Run
      the consolidation pass in the commit that crosses 60 and record the line
      count.
    - `tests/fixtures/AGENTS.md` — the `scratch_db.py` bullet gains
      `OWNER mira_admin` and the `bootstrap_adopt.sql` apply; the `## Rules`
      bullet describing what `scratch_database()` opens its admin connection for
      gains the bootstrap apply.
    - `deploy/docker/scripts/AGENTS.md` — the `init-mira.sh` bullet and the
      `init_postgresql` restart-safety Rule gain the bootstrap call.

Verification:

| Probe | Executes | Asserts |
|---|---|---|
| `/tmp/probe_bootstrap_adopt.py` | `live_infra.require("postgres")`; `scratch_database(user=<superuser>, host="localhost")`; then, over a second scratch DB created **without** the fixture's bootstrap step, runs `deploy/bootstrap_adopt.sql` through a real `psql -v ON_ERROR_STOP=1` subprocess as the superuser | Idempotence: running the file twice in a row exits 0 both times. Ownership: `SELECT DISTINCT tableowner FROM pg_tables WHERE schemaname='public'` is exactly `{'mira_admin'}`; same for `pg_indexes` (via `pg_class.relowner`), `pg_sequences`, and `pg_proc` in `public`. Privilege: `has_schema_privilege('mira_admin','public','CREATE')` is `True`. Ledger: `schema_migrations` exists with the eight columns and both CHECK constraints, and `has_table_privilege('mira_dbuser','schema_migrations','SELECT')` is `True`. Structural mutation: the pre/post structural fingerprint over everything **except** `schema_migrations` is byte-identical — adoption stamps, it does not reshape. Teardown: the fixture drops both DBs; assert no `mira_probe_*` survives. `PROBE PASSED` |
| `/tmp/probe_ledger_grants.py` | Connects to the scratch DB as `mira_dbuser` over TCP and runs `SELECT count(*) FROM schema_migrations`, then `INSERT` | SELECT succeeds (returns 0). INSERT raises `psycopg.errors.InsufficientPrivilege`. Control pair, both directions. `PROBE PASSED` |
| `/tmp/probe_secrets_manifest.py` | `live_infra.require("vault")`; exports `VAULT_ADDR`/`VAULT_ROLE_ID`/`VAULT_SECRET_ID` from `/opt/vault/*.txt` (or reports UNVERIFIED on a dev checkout without AppRole creds); calls the real `clients.vault_client.check_required_secrets()` against live Vault, then calls the pure `format_secret_gap()` with three real `SecretGap` values | Live call returns `[]` on a provisioned install. `format_secret_gap` output is exactly `vault kv patch secret/mira/database admin_url=<value>`-shaped for each of the three. The manifest's field sets equal the sets written by `deploy/postgresql.sh` Step 14 and `deploy/docker/scripts/init-mira.sh:init_vault` — derived by grepping those two files for `vault kv put`/`vault_put_*` field names and set-comparing, so a silently dropped field fails the probe. No secret value is printed; assert `value not in captured_stdout`. `PROBE PASSED` |
| `/tmp/probe_post_manifest_check.py` | Runs `python -m utils.power_on_self_test pre-server --json --report-marker` against live infrastructure and parses between `MIRA_POST_REPORT_BEGIN`/`END` | `vault_secrets_manifest` appears in `checks` with `required=true` and `passed=true`. Control: monkeypatching nothing — instead call `_check_vault_secrets_manifest()` directly with a manifest whose first entry names a field that does not exist (`mira/database/nonexistent_field`), and assert the raised `RuntimeError` message contains the literal `vault kv patch secret/mira/database nonexistent_field=`. The bound: this exercises the real check function with a real Vault read; it does not simulate Vault. `PROBE PASSED` |
| `/tmp/probe_baseline_fingerprint.py` | Two independent scratch databases from `scratch_database()`; computes `baseline_fingerprint()` on each and `sha256`s the result | The two hashes are equal (determinism). The hash equals the `fingerprint_sha256` line in `deploy/migrations/BASELINE` — the checked-in value is the generated one. Normalization witnessed: the two DBs have different OIDs and different `embedding_dimensions` only if the fixture varied them; assert the fingerprint excludes OIDs and normalizes the dimensions value by re-running with an explicitly different `-v embedding_dimensions` and confirming the hash is unchanged. `PROBE PASSED` |

Done when:

- [ ] All five probes `PROBE PASSED`, or UNVERIFIED with the named blocker (Vault AppRole absence is the expected one on a dev checkout).
- [ ] `grep -c schema_migrations deploy/mira_service_schema.sql deploy/bootstrap_adopt.sql` ≥ 1 in each.
- [ ] `deploy/bootstrap_adopt.sql` applied twice in a row exits 0 twice.
- [ ] `tests/fixtures/scratch_db.py` yields a DB where `pg_tables.tableowner` is uniformly `mira_admin`.
- [ ] `utils/power_on_self_test.py` spec list contains `vault_secrets_manifest` and the boot gate still passes on a real install (`probe_post_manifest_check.py`).
- [ ] `psql -X -v ON_ERROR_STOP=1 … -f deploy/mira_service_schema.sql` still refuses a non-empty database (the `$schema_precondition$` block is untouched).
- [ ] Seven maps edited in the same commit; `docs/AGENTS_MAP_SPEC.md` anchor, citation, and budget audits clean; `utils/AGENTS.md` consolidation recorded.
- [ ] `grep -n "schema floor is 14" deploy/AGENTS.md` returns nothing and the corrected bullet names 17 with its two shipped witnesses (R5).

---

## Phase 2 — runner, read paths only

Goal:        discovery, ordering, checksums, validation, `--dry-run`, `status`. No mutation of any store.
Preconditions: Phase 1 shipped. `utils/schema_migrations.py` importable from the server venv.

Steps:

1. `utils/schema_migrations.py` (new) — discovery (MIG-1, dbmate/goose pattern).
   - `MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "deploy" / "migrations"`,
     overridable by an explicit `migrations_dir` parameter on every entry
     function (CROSS-3, and the seam VER-2 uses to author throwaway steps).
   - `discover_steps(migrations_dir) -> list[Step]`. `Step` is a frozen
     dataclass: `revision: str` (4 digits), `slug: str`, `kind: Literal["versioned","repeatable"]`,
     `store: Literal["postgres","vault","env","valkey"]`, `path: Path`,
     `checksum: str` (sha256 of the file body, computed over the exact bytes),
     `requires_operator_input: bool`, `dialect_twins: tuple[str, ...]`.
   - Filename grammar, enforced: `V(\d{4})__([a-z0-9_]+)\.(sql|py)` and
     `R__([a-z0-9_]+)\.(sql|py)`. Anything else in the directory is skipped
     silently **except** a file matching `^[VR]` that fails the grammar, which
     raises `ValueError` naming the file and the grammar — a typo'd step must not
     disappear (root `AGENTS.md`: Fail-Fast, Fail-Loud).
   - `store` is declared, never inferred: the SQL/Python header must carry
     `-- store: postgres` (SQL) or `STORE = "postgres"` (Python); absent ⇒
     `ValueError`. Never infer from content shape (root `AGENTS.md`: Extract by
     Identity, Not Position or Shape).
   - `requires_operator_input` from the literal token `REQUIRES_OPERATOR_INPUT`
     in the header; `dialect_twins` from `-- dialect-twins: DialectName, DIALECT_NAMES`.
   - `BASELINE` parsed by `read_baseline(migrations_dir) -> Baseline(revision, fingerprint_sha256)`.
2. `utils/schema_migrations.py` — ordering + duplicate/gap detection (MIG-4).
   `order_steps(steps) -> OrderedChain` raises `StepOrderError` (a
   `ValueError` subclass carrying diagnostics) on: a duplicate revision; a gap
   in the versioned sequence (revisions are dense from `0001`); a repeatable
   step whose slug appears twice. Sorting is by integer revision, ascending —
   never lexicographic over a variable-width field, and never by mtime.
3. `utils/schema_migrations.py` — ledger read.
   `read_ledger(conninfo) -> list[LedgerRow]` over `psycopg.connect(conninfo)`,
   `LedgerRow` a `TypedDict` mirroring the eight columns.
   `ledger_head(rows) -> str | None` is the max versioned revision.
   `repeatable_due(rows, steps) -> list[Step]` compares each `R__` step's
   checksum against its ledger row's — a changed body re-runs (Flyway's
   repeatable semantics).
4. `utils/schema_migrations.py` — validation (MIG-4, Chrome-channel
   ledger-ahead semantics). `validate(chain, ledger, baseline) -> Validation`
   returns a typed result; `Validation.refusals` is a list of
   `Refusal(code, message)`. Codes and messages, exact:
   | Code | Condition | Message |
   |---|---|---|
   | `DOWNGRADE` | a ledger revision sorts above the tree head | `database is at revision <N>; this tree's newest step is <M> — refusing to downgrade` |
   | `CHECKSUM_MISMATCH` | an applied revision's checksum ≠ the tree file's | `revision <N> (<slug>) was edited after it was applied — fix forward with a new revision` |
   | `OUT_OF_ORDER` | a pending versioned revision sorts below the applied head | `revision <N> sorts below the applied head <M>; out-of-order application is refused` |
   | `DUPLICATE_REVISION` / `GAP` | from step 2 | as raised |
   | `REVERTED_MIGRATION` | an applied revision has no file in the tree | `revision <N> is recorded as applied but this tree has no such step — a reverted or partial checkout` |
   | `ADOPTION_DRIFT` | adoption fingerprint ≠ `BASELINE`'s and `--adopt-drifted` is absent (raised by 3.4's `bootstrap_check`, same `Refusal` vocabulary) | `install schema does not match the baseline fingerprint (observed <12hex>, expected <12hex>) — the drift report above names every difference. Review it, then re-run exactly: mira migrate --adopt-drifted` |
   `pending(chain, ledger, baseline) -> list[Step]` is a pure function over the
   same inputs, returning versioned steps above the head plus due repeatables.
   `render_drift_report(observed_fp, expected_fp) -> str` renders the two
   fingerprints as added / removed / changed object lists — the report
   `ADOPTION_DRIFT` prints above its own message (R2: the operator reviews the
   report before overriding, so the report is part of the refusal, not a
   debugging extra).
5. `utils/schema_migrations.py` — structural fingerprint (VER-1's instrument,
   Rails staleness-guard idea). `structural_fingerprint(conninfo) -> dict` reads,
   in a fixed key order, from `pg_catalog`/`information_schema` only:
   tables + columns (name, type, nullability, default), index definitions
   (`pg_get_indexdef`), constraints (`pg_get_constraintdef`), RLS policies
   (`pg_policy` + `pg_get_expr(polqual, polrelid)`), triggers
   (`pg_get_triggerdef`), function signatures
   (`pg_get_function_identity_arguments` + return type + `prosecdef`), and
   grants (`information_schema.role_table_grants` + `has_schema_privilege`).
   Normalizations, all required: OIDs excluded; `embedding_dimensions`-derived
   `vector(N)` types rewritten to `vector(<DIM>)`; `pg_class.relpersistence`
   excluded; the `schema_migrations` table itself excluded from the comparison
   used by adoption (step 3.4) but included in VER-1's.
   `fingerprint_sha256(fp) -> str` is a canonical-JSON sha256
   (`json.dumps(fp, sort_keys=True, separators=(",",":"))`).
6. `utils/schema_migrations.py` — MIG-14 authoring rules as mechanical checks.
   `lint_steps(chain) -> list[LintFinding]`, each finding naming the file and the
   rule:
   - a `.sql` step containing more than one statement-terminating transaction
     control (`BEGIN`/`COMMIT`/`ROLLBACK`) ⇒ finding (MIG-5: one transaction per
     step, supplied by the runner).
   - a step body reading `VERSION`, importing `utils.release_identity`, or
     calling `time.time()`/`datetime.now()`/`CURRENT_DATE`/`NOW()` in a
     **branch** position ⇒ finding (MIG-14 last clause). `NOW()` as a column
     default is fine; the lint is an AST/regex check for a conditional on it.
   - a step touching `embedding_config` ⇒ finding (MIG-14: never touched).
   - a step issuing `UPDATE model_configs SET` on any of `model`/`dialect_name`/
     `endpoint_url`/`api_key_name`/`effort`/`max_tokens` ⇒ finding
     (MIG-14: operator-owned; `INSERT … ON CONFLICT DO NOTHING` and a CHECK
     widening are permitted).
   - a step containing a hardcoded `vector(<N>)` literal ⇒ finding (MIG-14:
     dimensions read from the install's `embedding_config` row at run time).
   - a step whose body alters a CHECK value list on `model_configs.dialect_name`
     without a `-- dialect-twins:` header naming `DialectName` and
     `DIALECT_NAMES` ⇒ finding (MIG-14; the twins live at
     `clients/llm/types.py:11` and `:26`).
   - a step containing `DROP TABLE`/`DROP COLUMN`/`ALTER … RENAME` in the same
     revision that stops using it ⇒ **not** mechanically checkable from one file;
     the lint instead requires every such statement to carry a
     `-- expand-contract: contract-for V<N>` header naming the expand revision.
     Absent header ⇒ finding.
   `lint_steps` runs in `validate` and any finding is a refusal (exit 2).
   `discover_steps` + `order_steps` + `lint_steps` over `deploy/migrations/` is
   REL-1(b), the ledger-integrity half of the release gate (5.4); it is exposed
   as `scripts/migrate.py lint` (step 7) so the release builder can run it
   **without a database connection** — one implementation, reached through the
   shim, never a second copy in bash.
7. `scripts/migrate.py` (new) — thin CLI shim per `scripts/AGENTS.md` Rule 1:
   argparse and output formatting only. Subcommands `migrate`, `restore`,
   `status`; flags `--dry-run`, `--migrations-dir`, `--backups-dir`,
   `--adopt-drifted` (R2: the only way past an `ADOPTION_DRIFT` refusal;
   accepted by `migrate`, rejected by `status`, `restore` and `lint`),
   `--rebuild-venvs` is NOT here (it is `update.sh`'s). Subcommand `lint` runs
   `discover_steps` + `order_steps` + `lint_steps` over `--migrations-dir`
   (default `deploy/migrations/`) and opens **no connection and no Vault read**
   — it is REL-1(b)'s cut-time form, callable from `deploy/lib/provenance.sh`'s
   bash builder as `python3 -m scripts.migrate lint`; exit 0 clean, 2 on any
   finding. Phase 2 implements `status`, `lint` and `migrate --dry-run`; `migrate`
   without `--dry-run` and `restore` print `not implemented until Phase 3` and
   exit 1.
   Exit codes (MIG-3): 0 ok/nothing-to-do; 1 step failure; 2 validation failure;
   3 blocked.
   **Connection resolution**: the shim calls
   `clients.vault_client.get_database_url("mira_service", admin=True)`
   (`clients/vault_client.py:121-146`) and passes the resulting string as
   `conninfo` to `utils.schema_migrations`. `conninfo` is a **required
   parameter** of every runner entry function — there is no env-var credential
   path and no default, so a probe supplies
   `psycopg.conninfo.make_conninfo(dbname=<scratch>, user=<superuser>)` and the
   production code never grows a fallback (root `AGENTS.md`: Credential
   Management; CROSS-3). Requires `VAULT_ADDR` + `VAULT_ROLE_ID`/`VAULT_SECRET_ID`
   exported per `scripts/AGENTS.md` Rule 2.
8. `utils/schema_migrations.py` — `status(conninfo, migrations_dir) -> StatusReport`,
   a Pydantic `BaseModel` per the repo's Pydantic standards: `ledger_head`,
   `tree_head`, `baseline`, `pending: list[str]`, `repeatable_due: list[str]`,
   `refusals: list[Refusal]`, `fingerprint_drift: bool`,
   `adopted_with_drift_override: bool` (R2: derived on read as "any ledger row
   whose `applied_by` ends with `+adopt-drifted`", per 3.4 — computed, never
   maintained as a separate counter). `scripts/migrate.py` renders it.
9. Map edits, same commit:
   - `utils/AGENTS.md` — a `## Files` bullet for `schema_migrations.py` naming
     `discover_steps`, `order_steps`, `validate`, `pending`,
     `structural_fingerprint`, `lint_steps`, `render_drift_report`, `status`,
     the five refusal codes including `ADOPTION_DRIFT`, and the
     required-`conninfo`-parameter contract. Consolidation pass if this crosses
     60 (CONFLICTS C-5).
   - `scripts/AGENTS.md` — a `## Files` bullet for `migrate.py` stating it is a
     shim over `utils.schema_migrations`, the four subcommands including the
     DB-free `lint`, the four exit codes, `--adopt-drifted` and what it is the
     only way past, and the Vault-env precondition (which `lint` does not have —
     it opens nothing). **See CONFLICTS C-3** for the `mira migrate`
     dispatch shape this implies.
   - `deploy/migrations/AGENTS.md` — `## Files` gains `BASELINE`'s generated
     fingerprint clause and the `## Rules` bullet enumerating the lint rules
     (the add-one recipe shape from `docs/AGENTS_MAP_SPEC.md`: every file to
     touch, each anchored, with the concrete failure each skipped step produces).
   - `deploy/AGENTS.md` — the `mira_service_schema.sql` bullet's consumer list
     gains `utils/schema_migrations.py` (fingerprint + `BASELINE` generation).

Verification:

| Probe | Executes | Asserts |
|---|---|---|
| `/tmp/probe_migrate_discovery.py` | Writes a real throwaway step set into `/tmp/mig_disc/` (`V0001__a.sql`, `V0002__b.sql`, `V0004__d.sql`, `V0002__dup.sql`, `R__pricing.sql`, `BASELINE`, `Vx__bad.sql`, `README.md`), then calls the real `discover_steps` and `order_steps` | Discovery finds exactly the five well-formed steps and skips `README.md`; `Vx__bad.sql` raises `ValueError` naming the file and the grammar; the gap set `{0001,0002,0004}` and the duplicate `0002` each raise `StepOrderError` with the right code. Control: a dense set `0001,0002,0003` orders to exactly that list. Checksums: editing one byte of `V0001__a.sql` changes its `checksum`; the same file's checksum is stable across two `discover_steps` calls. `store` absent ⇒ `ValueError`. Teardown: `rm -rf /tmp/mig_disc`. `PROBE PASSED` |
| `/tmp/probe_migrate_validate.py` | `live_infra.require("postgres")`; four scratch databases via `scratch_database()`; real ledger rows inserted through the real `record_ledger_row`-shaped SQL; then the real `validate()` | **All four ledger refusals (MIG-4, VER-4)**, each with a positive and negative control: (1) ledger at `0005`, tree head `0003` ⇒ `DOWNGRADE`, message names both revisions; tree head `0005` ⇒ no refusal. (2) ledger checksum for `0002` differs from the tree file's ⇒ `CHECKSUM_MISMATCH` naming the slug; matching checksum ⇒ none. (3) pending `0002` with head `0003` ⇒ `OUT_OF_ORDER`; pending `0004` ⇒ none. (4) ledger row `0002` with no tree file ⇒ `REVERTED_MIGRATION`. The fifth code, `ADOPTION_DRIFT`, is `bootstrap_check`'s and is witnessed by `probe_adoption_drift.py`, not here — assert only that the `Refusal` code vocabulary has exactly the five members the table at step 4 names, so a code added without a witness fails. **Nothing mutated**: `structural_fingerprint()` and `read_ledger()` before and after each `validate()` call are identical. `render_drift_report` over two real differing fingerprints names the added, removed, and changed objects and prints no connection string. `lint_steps` findings: a step with `UPDATE model_configs SET model=` is flagged; the same step with `INSERT … ON CONFLICT DO NOTHING` is not; a step with `vector(768)` hardcoded is flagged; a `dialect_name` CHECK change without `-- dialect-twins:` is flagged and with it is not; a step importing `utils.release_identity` is flagged. `PROBE PASSED` |
| `/tmp/probe_migrate_status.py` | `scripts/migrate.py status` as a real subprocess against a scratch DB, with `--migrations-dir` pointed at `/tmp/mig_status/`; three ledger states (empty, at head, ahead of head) | Exit 0 and the rendered report names `ledger_head`, `tree_head`, `pending` count for the first two; exit 2 and the `DOWNGRADE` message for the third. stdout contains no secret and no connection string with a password (assert the password substring is absent). `PROBE PASSED` |
| `/tmp/probe_migrate_dryrun.py` | `scripts/migrate.py migrate --dry-run` against a scratch DB with two pending steps | Exit 0; stdout names both pending revisions in order; `structural_fingerprint()` before == after; `read_ledger()` before == after. The dry run really connects and really reads — assert the report's `baseline` equals `deploy/migrations/BASELINE`'s. `PROBE PASSED` |
| `/tmp/probe_fingerprint_determinism.py` | Two independent scratch DBs; `structural_fingerprint()` on each, twice per DB | Four hashes, all equal. Then `CREATE TABLE probe_extra(i int)` in one and re-fingerprint: that hash differs and the diff names `probe_extra` — the fingerprint is sensitive, not constant (no-op-assertion guard). Drop it, re-fingerprint, hash returns to the original. `PROBE PASSED` |
| `/tmp/probe_exit_codes.py` | `scripts/migrate.py` as a subprocess for each of the exit conditions | 0 on nothing-to-do; 2 on a validation refusal with the ledger unmutated; 3 when `PGHOST` points at a closed port (real connection-refused, no fake); 1 reserved — assert `migrate` without `--dry-run` exits 1 with the Phase-2 `not implemented` text. **`lint` arm (REL-1(b))**: `scripts/migrate.py lint` exits 0 over the real `deploy/migrations/` and 2 over a `/tmp` copy corrupted once each way (a duplicate revision, then a lint-violating step), naming the file and the rule each time; assert it opened no connection by running it with `PGHOST` pointed at a closed port and `VAULT_ADDR` unset, and that `--adopt-drifted` is rejected on it. `PROBE PASSED` |
| `/tmp/probe_no_new_deps.py` | Re-run of Phase 0's, over this phase's diff | `requirements.txt` and `ui/tui/requirements.txt` unmodified; every import in `utils/schema_migrations.py` and `scripts/migrate.py` is stdlib, `psycopg`, `hvac`, `pydantic`, or an in-repo module. `PROBE PASSED` |

Done when:

- [ ] All seven probes `PROBE PASSED`.
- [ ] No probe used sudo, and no probe connected to `mira_service` — scratch DBs only (CROSS-3).
- [ ] `scripts/migrate.py` contains no SQL string and no `psycopg` import (shim discipline, `scripts/AGENTS.md` Rule 1).
- [ ] Every runner entry function takes `conninfo` as a required parameter; `grep -n "getenv\|environ" utils/schema_migrations.py` returns only `REQUIRES_OPERATOR_INPUT`-related TTY detection, never a credential read.
- [ ] `migrate` without `--dry-run` mutates nothing (exit 1, Phase-2 stub).
- [ ] Four maps edited in the same commit; audits clean.

---

## Phase 3 — mutation: apply loop, adapters, backup, restore, failure path, adoption

Goal:        make `migrate` and `restore` real, backed up, transactional per step, and reversible.
Preconditions: Phase 2 green. `pg_dump`/`pg_restore` on PATH. `/opt/mira/backups` writable (host venue) or a `--backups-dir` override (probe venue).

Steps:

1. `utils/schema_migrations.py` — advisory lock (MIG-3, Flyway pattern).
   `migration_lock(conn)` is a context manager issuing
   `SELECT pg_advisory_lock(%s)` with a module constant
   `MIGRATION_ADVISORY_KEY = <int literal>` and `pg_advisory_unlock` on exit.
   The key is a hardcoded constant with a comment explaining why (root
   `AGENTS.md`: Hardcode Known Constraints) and is registered in
   `deploy/migrations/AGENTS.md` so a second advisory-lock user does not
   collide. Held for the **whole run**, acquired before the bootstrap check and
   released after the last ledger write. A second concurrent runner blocks on
   the lock and does not interleave.
2. `utils/schema_migrations.py` — step context adapters (MIG-15). No new client
   code; each adapter wraps something that already exists.
   | Adapter | Implementation | Bound |
   |---|---|---|
   | `ctx.pg` | the run's own `psycopg.Connection` on `conninfo` (`mira_admin`, `BYPASSRLS` — `deploy/postgresql.sh:108`); the step's transaction is the runner's, so `ctx.pg` never calls `commit()` | one transaction per step (MIG-5) |
   | `ctx.vault` | `hvac.Client` built from `clients.vault_client.VaultClient().client` (`clients/vault_client.py:33-92`) — the AppRole-authenticated client, no new auth code. Methods: `read(path) -> dict`, `patch(path, fields)`, `delete_fields(path, fields)`. `patch` is read→compare→write; `delete_fields` refuses unless the caller passed `confirmed_new=<fields>` and every one of them is present in the post-write read (MIG-15: delete-old only after new confirmed) | `mira-policy` grants `create/read/update/delete/list` on `secret/*` (`deploy/lib/vault.sh:112-119`), so the app's own AppRole can patch. KV v2 native version history is the undo. |
   | `ctx.env` | a declarative editor for `/opt/mira/systemone.env`: `set_var(name, value)`, `unset_var(name)`; writes the whole file atomically (`tmp` + `os.replace`), mode `0600`, preserving no comments (the file is generated, `deploy/finalize.sh:35-71`) | the path is a parameter (`env_file`) so probes point at `/tmp` |
   | `ctx.invalidate_valkey` | calls the existing `ValkeyClient.flush_except_whitelist` (`clients/valkey_client.py:424`) with the **same** preserve-prefix list `main.py:161-166` passes. The list is imported from one module-level constant that `main.py` is refactored to use too, so the two cannot drift (root `AGENTS.md`: Convergent Path Refactoring) | the constant moves to `clients/valkey_client.py` as `BOOT_FLUSH_PRESERVE_PREFIXES`; `main.py:161` reads it |
   `StepContext` is a frozen dataclass built per run, never a singleton
   attribute (root `AGENTS.md`: No Per-Request State on Singletons).
3. `utils/schema_migrations.py` — step loading. A `.sql` step is executed with
   `conn.execute(sql_text)` inside the runner's transaction. A `.py` step is
   loaded with `importlib.util.spec_from_file_location` and must expose
   `up(ctx) -> None` and optionally `verify(ctx) -> None` (Sqitch pattern);
   `verify` runs after `up` inside the same transaction and a raise rolls the
   step back. A `.py` step missing `up` ⇒ `ValueError` naming the file.
4. `utils/schema_migrations.py` — adoption (MIG-8/MIG-9, Flyway
   baseline-on-adopt + Django `--fake-initial`). **R2: drift is refused, never
   converged.** `bootstrap_check(conninfo, migrations_dir, adopt_drifted: bool) -> Adoption`:
   - ledger table absent ⇒ run `deploy/bootstrap_adopt.sql` **only** when a
     superuser seam is available; otherwise exit 3 with
     `ADOPT-PROBE-BLOCKED: ledger absent and no superuser seam — run
     deploy/bootstrap_adopt.sql through sudo -u postgres psql`. On a probe
     venue, `scratch_database()` already applied it (Phase 1 step 8), so this
     branch does not fire.
   - ledger present with 0 rows and tables present ⇒ adoption. Compute
     `structural_fingerprint()` and compare to `read_baseline().fingerprint_sha256`.
     **Three outcomes, and only three:**
     | # | Condition | Action | Exit |
     |---|---|---|---|
     | 1 | fingerprint **matches** | stamp revisions `0001..BASELINE.revision` with `applied_by='baseline'`, `kind='versioned'`, `checksum` from the tree file, `duration_ms=0`; then the normal path runs whatever is above `BASELINE.revision` | 0 |
     | 2 | fingerprint **differs**, `adopt_drifted` is `False` | stamp **nothing**, mutate nothing; print `render_drift_report(observed, expected)` — added / removed / changed objects, with both fingerprints' first 12 hex — then the `ADOPTION_DRIFT` refusal message, which names the exact override invocation `mira migrate --adopt-drifted` | **2** |
     | 3 | fingerprint **differs**, `adopt_drifted` is `True` | stamp nothing, run the **full idempotent chain `0001..head`**, and record the override in the rows that run writes (form below) — every step is idempotent to its own commit point (MIG-14), which is what makes the override safe on a schema that already carries some steps' end states | 0 |
     **The override is recorded, not merely permitted.** Every ledger row an
     outcome-3 run writes carries `applied_by = "<version>+adopt-drifted"`, and
     revision `0001`'s row additionally carries the overridden fingerprint in its
     `name` field as `"<slug> adopt-drifted=<first 12 hex of the observed
     fingerprint>"` — the same `name`-suffix mechanism Phase 6 step 5 uses for
     reconciliation reports, so there is no separate override row, no new column,
     and no DDL change. `status` derives `adopted_with_drift_override` on read as
     "any ledger row whose `applied_by` ends with `+adopt-drifted`" (2.8) — a
     permanent historical fact about the install, never a maintained counter.
     **The flag is per-invocation and non-sticky.** It authorizes the one run it
     is passed to; it writes no state that a later run reads, and no config value
     or env var disables the check. Unknown drift stays refused, permanently.
     The known case is the maintainer's own hand-patched install — the
     pre-applied partial-unique-index change `V0001` covers (5.1): review the
     drift report, override once, done.
   - ledger present with rows ⇒ normal path, no adoption. `adopt_drifted` is
     ignored here and the runner says so on stdout — an override passed to an
     already-adopted install is a no-op, never a license to skip validation.
   - ledger absent and **no** tables ⇒ a fresh empty database; the schema file
     has not been applied. Exit 2 with `no schema and no ledger — apply
     deploy/mira_service_schema.sql first`. Migrations never create the baseline
     schema.
   **MIG-9 as briefed asked for warn-and-converge; R2 supersedes it.** The
   requirement is satisfied by outcomes 2 and 3 together — drift is surfaced
   loudly and the chain still converges an install that is behind — but only on
   an explicit recorded act, never silently. No step, log line, or map bullet may
   describe adoption drift as a warning.
5. `utils/schema_migrations.py` — apply loop (MIG-5, Alembic/Postgres
   transactional-DDL pattern).
   ```
   with migration_lock(conn):
       adoption = bootstrap_check(..., adopt_drifted)
       validation = validate(...);  if validation.refusals: return 2
       backup = take_backup(...)          # before the FIRST mutation
       for step in pending:
           with conn.transaction():        # psycopg3 nested-transaction block
               run_step(ctx, step)
               insert_ledger_row(conn, step, duration_ms)   # same transaction
       for step in repeatable_due:
           with conn.transaction():
               run_step(ctx, step); upsert_ledger_row(conn, step, duration_ms)
   ```
   Ordering inside a step, by store (MIG-5): `postgres` steps put all their SQL
   in the one transaction with the ledger row. Non-postgres steps do the
   idempotent non-SQL work FIRST and the ledger row LAST, so a crash between
   them re-runs the work rather than recording it done. Cross-store changes are
   split into expand/contract revision sequences by authoring rule (step 6,
   MIG-14) — never one step.
   A step failure rolls back its own transaction, leaves the ledger at the last
   committed revision, releases the lock, and returns exit 1 with the failing
   revision, the slug, and the raw exception. `applied_by` is
   `utils.release_identity.get_current_version()` (`utils/release_identity.py:53-63`)
   or `nightly-<shortsha>` when `data/nightly_stamp` exists — read from the file,
   never re-derived.
6. `utils/schema_migrations.py` — `REQUIRES_OPERATOR_INPUT` [D5]. A step with
   the token gets `ctx.operator_input(prompt) -> str`, which reads from
   `sys.stdin` when `sys.stdin.isatty()` and otherwise raises
   `OperatorInputRequired(step, prompt)`; the runner catches it, rolls back,
   and returns exit 3 with the prompt text and `parked: re-run
   mira migrate from an interactive terminal`. No value is ever defaulted.
7. `utils/schema_migrations.py` — vector dimensions (MIG-14). `ctx.embedding_dimensions() -> int`
   reads `SELECT embedding_dimensions FROM embedding_config LIMIT 1` at run
   time. Steps needing a `vector(N)` column call it and interpolate with
   `psycopg.sql.SQL`/`Identifier` composition — never an f-string into DDL and
   never a literal (root `AGENTS.md`: Configure Structured Data — Never
   String-Patch It).
8. `utils/schema_migrations.py` — backup (MIG-6, `pg_dump`/`pg_restore` custom
   format). `take_backup(conninfo, backups_dir, stamp) -> BackupManifest`:
   - destination `<backups_dir>/<STAMP>/`, `STAMP` = `date -u +%Y%m%dT%H%M%SZ`
     matching `deploy/update.sh:225`. Host default `/opt/mira/backups` — a
     sibling of `/opt/mira/app`, therefore outside the code clear's scope on
     both call sites (the shared `clear_code_preserving_state` of Phase 0 step
     19, which replaces `deploy/update.sh:286-290` and `deploy/python.sh:87-90`
     and is rooted at the app dir with `-mindepth 1 -maxdepth 1`). Verified in
     source; assert it again in `probe_backup_retention.py`.
   - disk precheck BEFORE the dump: `SELECT pg_database_size('mira_service')`,
     compare against `shutil.disk_usage(backups_dir).free`; require
     `free >= 2 * size`; else exit 3 with the two numbers.
   - `pg_dump -Fc` over TCP as `mira_admin` using the same `conninfo`
     (`PGDATABASE`/`PGHOST`/`PGUSER`/`PGPASSWORD` derived from it via
     `psycopg.conninfo.make_conninfo` round-trip — no new credential path).
     Bounded: `subprocess.run(..., timeout=<seconds>)` with the timeout a module
     constant sized from the measured dump rate; a timeout is a failure, not a
     warning (root `AGENTS.md`: Bounded Waits at Every Boundary).
   - Vault KV JSON export: `ctx.vault.read()` for each of `mira/api_keys`,
     `mira/database`, `mira/services` → `<STAMP>/vault.json`, mode `0600`.
     Never printed; never logged.
   - Copies of `/opt/mira/systemone.env` (`deploy/finalize.sh:35`), and the
     supervisor definition — `/etc/systemd/system/mira.service`
     (`deploy/finalize.sh:104`) on Linux,
     `$HOME/Library/LaunchAgents/com.mira.app.plist` (`deploy/finalize.sh:235`)
     on macOS. Each copy is best-effort with a **manifest entry recording its
     absence and why** — the manifest is the record, not a silent skip.
   - `manifest.json`: stamp, pg dump filename + `pg_restore --list` TABLE DATA
     count (the integrity check `deploy/vm/extract-stage-vm.sh:17` already
     uses), vault export field-count per path (never values), env/unit paths
     copied or absent, `pg_database_size`, `VERSION`, ledger head.
   - Retention: after a successful run, list `<backups_dir>` newest-first, keep
     3, delete the rest, and print a deletion report naming every removed
     directory. Deletion happens only on success — a failed run keeps its own
     backup.
9. `utils/schema_migrations.py` — `restore(conninfo, backups_dir, stamp) -> RestoreResult`
   (MIG-7). Sequence, each step bounded and each failure reported with the exact
   manual command that continues it:
   1. read `<backups_dir>/<stamp>/manifest.json`; absent ⇒ exit 1 naming the path.
   2. `pg_restore --list` the dump; a nonzero exit ⇒ exit 1 (corrupt backup, do
      not proceed to destroy the live DB).
   3. terminate other backends on `mira_service`
      (`SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='mira_service' AND pid <> pg_backend_pid()`,
      the shape `deploy/vm/inject-restore-vm.sh:30` uses).
   4. **R6: `mira_admin` stays narrow — no `CREATEDB` grant.** Reset the
      `public` schema as `mira_admin`:
      `DROP SCHEMA public CASCADE; CREATE SCHEMA public; GRANT USAGE ON SCHEMA public TO mira_dbuser;`
      — NOT `DROP DATABASE`, because `mira_admin` has no `CREATEDB`
      (`deploy/postgresql.sh:108`) and R6 declines to grant it. This requires
      `mira_admin` to own the `public` schema, which holds because `public` is
      owned by `pg_database_owner` from PG15 and `mira_admin` owns the database
      (`createdb -O mira_admin`, `deploy/postgresql.sh:116`). **C-6 is closed by
      R5**: the supported floor is corrected to 17 in `deploy/AGENTS.md` in
      Phase 1's commit, so the `pg_database_owner` semantics hold on every
      supported install and this path is unambiguous rather than
      version-dependent. Step 5's `pg_restore --no-owner` is the other half of
      the ruling; together they are the probe-proven restore, and
      `probe_restore_roundtrip.py` is a REL-1(e) shipping-gate condition (5.4),
      not merely a phase verification.
   5. `pg_restore --no-owner -d mira_service` over TCP as `mira_admin`
      (`deploy/vm/inject-restore-vm.sh:32` precedent). `--no-owner` so the
      restored objects belong to `mira_admin`, then re-run
      `deploy/bootstrap_adopt.sql` when a superuser seam is available to restore
      grants the dump did not carry.
   6. trusted extensions: assert `pgcrypto`, `vector`, `pg_trgm` are present
      (`SELECT extname FROM pg_extension`); a missing one is a hard failure
      naming the `CREATE EXTENSION` that needs a superuser (CROSS-3's
      installer-declared exception).
   7. restore `systemone.env` and the supervisor definition from the backup
      copies when the manifest records them present; regenerate the unit per
      Phase 6 step 6.4 when it does not.
   8. Vault: restore only fields the manifest records as changed by the run —
      `ctx.vault.read()` current vs the backup JSON, patch back the differences.
      KV v2 version history is the second line of defense; state it in the
      report.
   9. fingerprint check: `structural_fingerprint()` post-restore must equal the
      pre-dump fingerprint recorded in the manifest. Mismatch ⇒ exit 1 with both
      rendered.
10. `deploy/update.sh` — new phase 6.5 will call this in Phase 4; here, add the
    **failure path** [D1] to the runner only:
    `recover_from_failed_run(conninfo, backups_dir, stamp, app_dir, tui_venv, old_version) -> Recovery`:
    - call `restore(...)` for the store changes;
    - print the code/venv rollback commands `deploy/update.sh` already assembles
      (`ROLLBACK_VENV_CMDS` from Phase 0 step 7, plus
      `git -C <app_dir> reset --hard HEAD~1`);
    - restart on the old version through the same
      `launchd_reload_agent`/`sudo -n systemctl start mira` branch
      (`deploy/update.sh:307-323`);
    - health-poll the OLD `VERSION` for 600 s using the existing poll loop shape
      (`deploy/update.sh:329-348`).
    - If the restore itself fails ⇒ **park**: leave the service down, print the
      backup path and the exact manual commands (the `pg_restore`, the schema
      reset, the venv `mv`s, the restart), exit 1. Never retry silently and
      never report success.
11. `scripts/migrate.py` — wire `migrate` (no `--dry-run`) and `restore <STAMP>`
    to the runner. `restore` takes the stamp positionally and lists available
    stamps when it is absent or unknown. `migrate --adopt-drifted` passes
    `adopt_drifted=True` into `bootstrap_check` (step 4) and into nothing else:
    it does not relax `validate`, does not skip a checksum, and does not survive
    the invocation. `ui/tui/migrate.py` (Phase 4 step 6) forwards `*args_list`
    unchanged, so `mira migrate --adopt-drifted` reaches the runner with no
    extra wiring.
12. `clients/valkey_client.py` — add `BOOT_FLUSH_PRESERVE_PREFIXES` as a module
    constant holding the six prefixes currently inlined at `main.py:162-165`, and
    edit `main.py:161-166` to pass it. One home, two readers.
13. Map edits, same commit:
    - `utils/AGENTS.md` — extend the `schema_migrations.py` bullet: apply loop,
      advisory-lock constant, adapters, backup/restore, exit codes, the
      three-outcome adoption rule with the recorded-override form
      (`applied_by` suffix + `name` suffix), and the `conninfo`-is-a-parameter
      rule. Consolidation if over 60 (C-5).
    - `clients/AGENTS.md` — `valkey_client.py` bullet gains
      `BOOT_FLUSH_PRESERVE_PREFIXES` and its two readers (`main.py`,
      `utils/schema_migrations.py`); the `vault_client.py` bullet gains the
      fact that `ctx.vault` reuses `VaultClient().client` rather than
      constructing a second hvac client, and that `mira-policy` is what makes
      the patch path work.
    - `clients/valkey_client.py`'s own directory map is `clients/AGENTS.md`;
      `main.py` is a root file — the root `AGENTS.md` `main.py` bullet's
      Valkey-flush clause cites the new constant's owner.
    - `deploy/migrations/AGENTS.md` — the advisory-lock key, the expand/contract
      authoring rule, the `REQUIRES_OPERATOR_INPUT` contract, the
      `ctx.embedding_dimensions()` rule, and the R2 adoption contract: drift is
      refused with exit 2 and a drift report, `--adopt-drifted` is the only way
      past it, and the override is recorded in the ledger rows it writes.
    - `scripts/AGENTS.md` — `migrate.py` bullet gains `migrate`/`restore` and
      the four exit codes' meanings.

Verification:

| Probe | Executes | Asserts |
|---|---|---|
| `/tmp/probe_migrate_apply.py` (**VER-1** convergence + **VER-2** skip-chain) | `live_infra.require("postgres")`. Builds a `/tmp/mig_chain/` step set of five real `.sql` steps (`V9001` create table, `V9002` add column + backfill, `V9003` add NOT NULL, `V9004` create index, `V9005` drop the `V9002` temp column — a real expand/contract pair) plus one `R__probe_pricing.sql` wholesale-replace. Scratch DBs via `scratch_database()`. **VER-1**: fingerprint of a fresh scratch DB (baseline schema, chain not run) vs a second scratch DB where the chain ran to head; then compare the first against `BASELINE`'s recorded hash. **VER-2**: for K ∈ {9001, 9003, 9004}, run `migrate` from K to head in ONE invocation on a DB already at K, and separately run each revision in its own invocation on a twin DB | **VER-1**: the fresh-DB fingerprint equals `BASELINE`'s hash (schema ≡ baseline). After applying the full chain to a third DB built by applying the *baseline schema then nothing*, the fingerprint equals the fingerprint of a DB built by *applying the chain from 0000* — modulo the normalizations. **VER-2**: for every K, `structural_fingerprint(one_shot) == structural_fingerprint(walked)` and `read_ledger(one_shot) == read_ledger(walked)` after dropping `applied_at` and `duration_ms` from each row. The `R__` step re-runs when its body changes (edit one byte, re-run, assert a second ledger row with the new checksum) and does not when it does not. `PROBE PASSED` |
| `/tmp/probe_migrate_interrupt.py` (**VER-3** part 1) | A `/tmp/mig_fail/` chain where `V9003__boom.sql` raises (`SELECT 1/0`). Runs `migrate` | Exit 1; the error names `V9003` and its slug and carries the raw Postgres message. `ledger_head == 9002`; the ledger has exactly the rows `0001..9002` (plus baseline stamps) and no `9003` row. `V9001`'s and `V9002`'s objects exist; `V9003`'s do not. Re-running after fixing the step file to `SELECT 1` completes to head and the ledger ends at `9005`. Control: `structural_fingerprint()` mid-failure equals the fingerprint of a DB migrated only to `9002`. `PROBE PASSED` |
| `/tmp/probe_sigkill_resume.py` (**VER-3** part 2) | Launches `scripts/migrate.py migrate` as a real subprocess against a scratch DB with a step that blocks on `pg_sleep(30)`; waits for the ledger to show the prior revision, then `SIGKILL`s the process; re-runs | The killed run leaves no ledger row for the in-flight revision and no partial DDL from it (transaction rollback witnessed: the `pg_sleep` step's `CREATE TABLE` is absent). The advisory lock is released by the backend's death: the re-run acquires it within 5 s (bounded) rather than hanging. The re-run completes to head. Assert `pg_advisory_lock` is not held afterward (`SELECT count(*) FROM pg_locks WHERE objid = <KEY>` is 0). `PROBE PASSED` |
| `/tmp/probe_migrate_privileges.py` (**VER-5**) | Scratch DB post-`bootstrap_adopt.sql`; connects as `mira_admin` **over TCP** (`host=localhost`, password from the fixture's role provisioning or peer auth per `deploy/AGENTS.md`'s pg_hba note) and executes each statement | `ALTER TABLE users ADD COLUMN probe_col int` then `DROP COLUMN` — both succeed. `CREATE TABLE probe_t(i int)` succeeds. `CREATE INDEX ON users(email)` succeeds. `ALTER TABLE … DROP CONSTRAINT` / `ADD CONSTRAINT` on a real constraint succeeds. All four `SECURITY DEFINER` functions callable: `embedding_config_refuse_change_with_vectors()` (via its trigger — insert a vector row then attempt an `embedding_config` UPDATE and assert the trigger's refusal message), `resolve_active_user_identity(text)`, `active_user_identity(uuid)`, `can_read_global_memories()`. `CREATE DATABASE probe_x` **fails** with `InsufficientPrivilege` — R6 witnessed, `mira_admin` has no `CREATEDB`. Teardown drops every probe object; assert the fingerprint returns to its pre-probe value. `PROBE PASSED` |
| `/tmp/probe_restore_roundtrip.py` (**VER-5** restore + MIG-7 + **REL-1(e) shipping-gate condition**) | Scratch DB; seed real rows through `tests/fixtures/probe_user.py` and the production repositories; `take_backup`; then `DROP SCHEMA public CASCADE`; `CREATE SCHEMA public`; `restore` — the whole round trip executed as `mira_admin` **over TCP**, never as the superuser, so the R6 narrow-privilege claim is the thing under test | Post-restore: the three trusted extensions are present in `pg_extension`; `structural_fingerprint()` equals the fingerprint recorded in the manifest; the seeded row counts are identical; RLS still fails closed — reconnect as `mira_dbuser` with no `app.current_user_id` set and assert a `SELECT` on `continuums` returns **zero rows and no error** (root `AGENTS.md`: RLS fails closed). Object ownership is uniformly `mira_admin`. **R6-specific**: the `DROP SCHEMA public CASCADE` succeeds as `mira_admin` with no `CREATEDB` and no superuser — the `pg_database_owner` semantics the corrected PG-17 floor guarantees (R5); assert `pg_database_owner` resolves to `mira_admin` for this database before the drop, so the probe names the mechanism it depends on rather than passing by luck. Teardown: the fixture drops the DB; `residue.assert_user_removed` for the probe user. **This probe's recorded state is a release gate**: a cut whose change report does not show it `PROBE PASSED` is not published (5.4, REL-1(e)). `PROBE PASSED` |
| `/tmp/probe_backup_retention.py` (MIG-6) | `--backups-dir /tmp/probe_backups`; four successive `take_backup` calls with distinct stamps; plus a run with `--backups-dir` on a filesystem whose free space is reported (real `shutil.disk_usage`) | After the fourth, exactly 3 directories remain and the deletion report names the removed one. `manifest.json` carries the `pg_restore --list` TABLE DATA count > 0, the vault field counts, and the env/unit presence entries. `vault.json` is mode `0600`. The disk precheck fires when `pg_database_size * 2` exceeds free space — witness it by asserting the precheck function's arithmetic against the two real numbers rather than by filling a disk. `/opt/mira/backups` is not inside `/opt/mira/app`: assert `Path("/opt/mira/backups").is_relative_to("/opt/mira/app")` is `False`, and that `clear_code_preserving_state`'s target argument on both call sites (`deploy/update.sh` phase 6, `deploy/python.sh` reinstall) is the app dir — so the clear's `-mindepth 1 -maxdepth 1` scope cannot reach a sibling of it (R4 moved the clear into `deploy/lib/state_preserve.sh`; the assertion follows it there). `PROBE PASSED` |
| `/tmp/probe_vault_adapter.py` (MIG-15 `ctx.vault`) | `live_infra.require("vault")`; a scratch KV path under `secret/mira/probe_adapter` (inside `mira-policy`'s `secret/*` scope, `deploy/lib/vault.sh:113-115`) | `read` → `patch` → `read` round-trips the exact field set. `delete_fields` **refuses** when `confirmed_new` names a field absent from the post-write read, and succeeds when it is present. KV v2 version history: `read_latest_versions` shows ≥2 versions after the patch. Teardown: `delete_metadata` the scratch path and assert `list` no longer names it. Reports UNVERIFIED (not FAILED) when AppRole env is absent, naming the export commands from `scripts/AGENTS.md` Rule 2. No secret value printed — assert against captured stdout. `PROBE PASSED` |
| `/tmp/probe_valkey_invalidate.py` (MIG-15 `ctx.invalidate_valkey`) | `live_infra.require("valkey")`; seeds real keys under each of the six preserved prefixes and one under `probe:doomed`; calls `ctx.invalidate_valkey()` | Every preserved-prefix key survives (exact value compare); `probe:doomed` is gone. The preserve list used is `clients.valkey_client.BOOT_FLUSH_PRESERVE_PREFIXES` — assert identity with the list `main.py:161` passes, so the two cannot drift. Teardown: `residue.assert_no_keys("probe:", client)`. `PROBE PASSED` |
| `/tmp/probe_adoption_drift.py` (**VER-4** adoption arms, MIG-9, **R2**) | Four scratch DBs: (a) unledged baseline DB with `schema_migrations` dropped; (b) a DB where `V0001`'s end state is **already** present by construction (the shipped schema, which carries `idx_messages_live_segment_unique` at `deploy/mira_service_schema.sql:394-396`) but the ledger says `0000` — the maintainer's hand-patched install, the known case; (c) a DB with a hand-added extra column — unknown drift; (d) an already-adopted DB with ledger rows. Each run through the real `scripts/migrate.py migrate` subprocess, once without and once with `--adopt-drifted` | **Outcome 1 (match)**: (a) stamps `0001..BASELINE.revision` with `applied_by='baseline'`, exit 0, and the fingerprint over everything except `schema_migrations` is unchanged — adoption stamps, it does not reshape. **Outcome 2 (refuse)**: (b) and (c) each exit **2**, stamp nothing, mutate nothing (`structural_fingerprint()` and `read_ledger()` identical before and after), print a drift report naming the added/removed/changed objects — for (b) the already-present index, for (c) the extra column — and print the literal override invocation `mira migrate --adopt-drifted`. **No silent convergence anywhere**: assert neither run wrote a ledger row and neither reached head. **Outcome 3 (recorded override)**: re-running (b) with `--adopt-drifted` exits 0, runs the full chain `0001..head`, converges — the fingerprint equals (a)'s and the ledger head is the tree head — and **records the override**: every row that run wrote carries `applied_by` ending in `+adopt-drifted`, revision `0001`'s `name` carries `adopt-drifted=<12hex>`, and `status` reports `adopted_with_drift_override=True` with `fingerprint_drift` still naming the observed difference. Same three assertions on (c): unknown drift is overridable only by the same explicit act, and the record says so. **Non-sticky**: a second `migrate` on (b) without the flag exits 0 with nothing to do — it does not re-refuse (already adopted) and it does not silently re-authorize anything. **Ignored on an adopted install**: (d) with `--adopt-drifted` applies only what is genuinely pending and says on stdout that the flag was ignored. `PROBE PASSED` |
| `/tmp/probe_advisory_lock.py` (MIG-3) | Two concurrent `scripts/migrate.py migrate` subprocesses against the same scratch DB with a slow step | Exactly one wins the lock and applies; the second blocks then applies nothing (its `pending` is empty by the time it runs) and exits 0 with a nothing-to-do report. No interleaved ledger rows: `read_ledger` shows each revision exactly once. `PROBE PASSED` |
| `/tmp/probe_operator_input.py` (D5) | A `/tmp/mig_op/` chain with one `REQUIRES_OPERATOR_INPUT` step; run with stdin closed (not a TTY), then run under a real pty | Non-TTY: exit 3, the prompt text printed, the word `parked`, no ledger row for that revision, no mutation. PTY: the step receives the typed value and records it; the ledger row exists. `PROBE PASSED` |

Done when:

- [ ] All eleven probes `PROBE PASSED` (Vault-dependent ones may report UNVERIFIED with the AppRole blocker named).
- [ ] `mira restore <STAMP>` is a real command with a passing round-trip probe (MIG-7's explicit requirement) — and that probe's recorded state is carried forward as REL-1(e) (5.4).
- [ ] Adoption drift exits 2 with a drift report and the named override; the override is recorded in the ledger; nothing converges silently (R2).
- [ ] No probe used sudo; the only superuser touch is `scratch_database()`'s own admin connection and `bootstrap_adopt.sql` where the fixture applies it (CROSS-3).
- [ ] `pg_advisory_lock` is held for the whole run and released on every exit path, including SIGKILL (witnessed).
- [ ] Every non-postgres step's ledger row is written last (assert by reading the apply loop and by the interrupt probe).
- [ ] Retention keeps exactly 3 and prints what it deleted.
- [ ] Five maps edited in the same commit; audits clean; `utils/AGENTS.md` line count recorded.

---

## Phase 4 — wiring: update.sh phase 6.5, superuser-seam preflight, boot guard, container, dispatch

Goal:        connect the runner to the three real invocation paths and make a stale schema refuse to boot.
Preconditions: Phases 2 and 3 green. A deployed host reachable over SSH for the seam probe.

Steps:

1. `deploy/update.sh` — MIG-11 superuser-seam preflight, added to phase 1
   (`deploy/update.sh:83-110`) **before anything is touched**:
   - Linux: `sudo -n -u postgres -c 'psql --version'`; a nonzero exit ⇒
     `print_error` + `exit 1` naming `sudo systemctl stop mira && mira update`
     as the recovery, exactly the shape the existing systemd-stop guard uses
     (`deploy/update.sh:244-250`).
   - macOS: the invoking user is the superuser (`deploy/postgresql.sh:119`'s
     plain `psql` branch). Verify with `psql -d postgres -tAc 'SELECT rolsuper FROM pg_roles WHERE rolname = current_user'`
     and require `t`.
   - Container (`MIRA_APP_DIR` under a container root) — skipped; `init-mira.sh`
     owns its own seam.
   - The preflight is advisory-only when the install's ledger head already
     equals the new tree's head (no migration will run) — determined by a
     read-only `scripts/migrate.py status` call, so an install with nothing to
     migrate never demands sudo. **D1 reversal note**: this is the first sudo
     use in `update.sh` beyond `systemctl`; the header's "Privileges: none"
     paragraph (`deploy/update.sh:35-42`) is rewritten in step 8.
2. `deploy/update.sh` — new phase 6.5 between the code swap (`:274-302`) and the
   restart (`:305`):
   ```
   # 6.5 Migrate the database (MIG-11). Runs AFTER the code swap so the tree's
   #     own steps and its own BASELINE are what apply, and BEFORE the restart so
   #     a failed migration never leaves the new code running against the old schema.
   "${APP_DIR}/venv/bin/python3" -m scripts.migrate migrate
   ```
   `PYTHONPATH="${APP_DIR}"` exported for the call (the `mira` wrapper's
   convention, `deploy/finalize.sh:317`), and `VAULT_ADDR`/`VAULT_ROLE_ID`/
   `VAULT_SECRET_ID` exported from `/opt/vault/*.txt` when the systemd unit is
   not the start path (`scripts/AGENTS.md` Rule 2). Exit codes: 0 continues;
   1/2/3 invoke step 3's failure path.
   Runs on the canonical root only when a service exists; on a non-canonical
   root it still runs (a probe's throwaway tree needs it) but skips the restart.
3. `deploy/update.sh` — [D1] failure path, replacing the two printed-rollback
   `print_info` blocks (`:334`, `:347`) and the `STARTED != yes` branch
   (`:330-335`):
   - migrate exit 1/2/3 ⇒ call
     `python3 -m scripts.migrate restore <STAMP>` (the backup phase 6.5's runner
     took), then execute `ROLLBACK_VENV_CMDS`, then
     `git -C "${APP_DIR}" reset --hard HEAD~1`, then restart on the old version,
     then health-poll the OLD `VERSION` for 600 s.
   - restore failure ⇒ **park**: print the backup path, the exact manual
     `pg_restore` / schema-reset / venv-`mv` / restart commands, and `exit 1`.
     The service stays down by design; the report says so in those words.
   - Health-poll failure on the new version ⇒ the same auto-restore path.
   - Header rewrite: delete "v1 never auto-rolls back" (`deploy/update.sh:30-33`)
     and state the new contract. **This is the D1 reversal; `deploy/AGENTS.md`'s
     `update.sh` bullet carries the same sentence and is rewritten in step 8.**
4. `deploy/update.sh` — the health-poll (`:323-352`) already asserts
   `status==healthy` AND `version==<new>`; add a third assertion that the
   response's ledger head equals the tree head by calling
   `scripts/migrate.py status --json` after the poll passes, so a migrated-but-
   unrecorded run cannot pass as healthy.
5. `utils/power_on_self_test.py` — MIG-10. New
   `CheckSpec("schema_revision", True, _check_schema_revision)` in
   `run_pre_server_post()`'s list (`utils/power_on_self_test.py:283-295`),
   placed after `"postgres_rls"`. `_check_schema_revision()` is **read-only over
   the app's ordinary connection** — `PostgresClient("mira_service")` as
   `mira_dbuser` (`clients/postgres_client.py:90`), using the
   `GRANT SELECT` from Phase 1 step 2. Four park conditions, each raising
   `RuntimeError` with the named message:
   | Condition | Query | Message |
   |---|---|---|
   | ledger table absent | `to_regclass('public.schema_migrations') IS NULL` | `schema_migrations is absent — this install was never adopted; run deploy/bootstrap_adopt.sql then mira migrate` |
   | ledger behind tree head | `max(revision)` < the tree head read from `deploy/migrations/` | `database is at revision <N>, this tree needs <M> — run: mira migrate` |
   | ledger ahead of tree head | `max(revision)` > tree head | `database is at revision <N> but this tree's newest step is <M> — a downgrade or a reverted checkout; refusing to boot` |
   | an applied revision absent from the tree | ledger revisions minus tree revisions, nonempty | `revisions <list> are recorded as applied but this tree has no such step — a reverted or partial checkout` |
   Migrations NEVER run at boot: the check contains no write and no import of the
   runner's apply path — only `discover_steps` and `read_ledger`. Return
   `{"ledger_head": …, "tree_head": …, "pending": n}`.
6. `ui/tui/__main__.py` — MIG-3's `mira migrate` dispatch, same pre-argparse
   style as `update` (`ui/tui/__main__.py:64-69`). **See CONFLICTS C-3**: the TUI
   venv (`/opt/mira/tui-venv`, `ui/tui/requirements.txt` = websockets, httpx,
   pydantic, rich, prompt_toolkit) has no `psycopg` and no `hvac`, so the
   dispatch subprocess-execs the **server** venv rather than importing the
   runner:
   ```python
   if args_list and args_list[0] in ("migrate", "restore", "status"):
       from .migrate import run_migrate_command
       return run_migrate_command(args_list)
   ```
   `ui/tui/migrate.py` (new) builds
   `[<app_dir>/venv/bin/python3, "-m", "scripts.migrate", *args_list]` with
   `PYTHONPATH=<app_dir>` and `exec`s it via `subprocess.run`, forwarding the
   exit code unchanged. `*args_list` is forwarded verbatim and there is **no
   flag whitelist**, so `mira migrate --adopt-drifted` (3.11) reaches the runner
   with no dispatch-side change — a whitelisted flag set would be a second place
   to edit every time the runner grows a flag, and the override invocation the
   `ADOPTION_DRIFT` refusal names must be the one that actually works.
   This preserves `ui/AGENTS.md:11`'s invariant that the
   clients import no server code beyond stdlib `utils/release_identity`.
   `run_status` from Phase 0 step 15 moves here and gains the ledger head +
   pending count by reading the runner's `status --json` through the same
   subprocess (PROV-4).
7. `deploy/docker/scripts/init-mira.sh` — MIG-12, **R3: the container runs
   migrate at start.** The alternative — a boot guard that refuses and leaves the
   operator to run `docker exec … mira migrate` — is **deleted**, not parked: no
   step, no map bullet, and no probe arm implements it. Replace the
   schema-skip-when-DB-exists gate at `:194-203`:
   ```
   if ! <db exists>; then
       createdb -O mira_admin mira_service
       <apply mira_service_schema.sql>          # first boot only
   fi
   <always> /opt/mira/venv/bin/python3 -m scripts.migrate migrate
   ```
   placed after `init_vault` (`:234`) in `main()` (`:358`), because the runner
   needs Vault for `get_database_url`. That is a **reordering** of
   `init_postgresql` → `init_vault` → migrate, and
   `deploy/docker/scripts/AGENTS.md`'s Wiring section states the order is
   load-bearing — amend it in step 9. Closes the drift hole where a rebuilt
   image with a persisted volume got new code and no schema step.
8. Map edits, same commit:
   - `deploy/AGENTS.md` — rewrite the `update.sh` bullet end to end: phase 6.5,
     the superuser-seam preflight, the D1 auto-restore failure path (deleting
     "v1 never auto-rolls back"), the third health assertion, and the fact that
     `update.sh` now uses sudo beyond `systemctl`. Rewrite the
     "In-place update vs. the greenfield database" Rule: the database is no
     longer greenfield across an update; `mira_service_schema.sql` still installs
     into an empty database only, and the chain in `deploy/migrations/` carries
     an existing one forward. Delete the Rule's sentence "The updater never
     migrates data and never touches the DB." Add a `## Wiring` line for the
     phase-6.5 ordering (code swap → migrate → restart) and what breaks if
     reordered.
   - `deploy/docker/scripts/AGENTS.md` — the `init-mira.sh` bullet and the
     `init_postgresql` restart-safety Rule: the migrate step now runs on every
     start; the Wiring flow order becomes `init_postgresql` → `init_vault` →
     `migrate` → `exec /init`, with the reason (the runner reads Vault).
   - `utils/AGENTS.md:24` — the `power_on_self_test.py` bullet's enumeration
     gains `schema_revision` with its four park conditions, and the bullet gains
     the "migrations never run at boot" clause. Line-count check (C-5).
   - `ui/tui/AGENTS.md` — `__main__.py` bullet gains the three subcommands;
     a new `migrate.py` bullet stating the subprocess-exec-onto-the-server-venv
     shape and why (no psycopg/hvac in the TUI venv).
   - `ui/AGENTS.md:11` — the "one exception" clause stays accurate; add
     `migrate.py` as a second stdlib-only module that imports no server code
     (it subprocess-execs instead), or leave unchanged if the sentence already
     covers it. Verify by re-reading, do not assume.
   - `scripts/AGENTS.md` — `migrate.py` bullet gains the `mira migrate` caller
     and the `PYTHONPATH` requirement.
   - Root `AGENTS.md` — the `main.py` bullet's lifespan clause is unchanged; the
     `BREAKING.md` bullet is rewritten in Phase 5.

Verification:

| Probe | Executes | Asserts |
|---|---|---|
| `/tmp/probe_post_schema_revision.py` (MIG-10) | `live_infra.require("postgres")`; four scratch DBs, each put into one of the four states by real SQL; then calls `_check_schema_revision()` directly with `PostgresClient` pointed at the scratch DB | (a) `DROP TABLE schema_migrations` ⇒ raises with the `never adopted` message naming `deploy/bootstrap_adopt.sql`. (b) ledger at `9001`, tree head `9003` (via `--migrations-dir /tmp/mig_chain/`) ⇒ raises with `run: mira migrate`. (c) ledger at `9005`, tree head `9003` ⇒ raises with the downgrade message. (d) ledger row `9004` present, file deleted from the tree copy ⇒ raises naming `9004`. Control: ledger at head ⇒ returns the dict with `pending == 0` and raises nothing. Read-only witnessed: `structural_fingerprint()` and `read_ledger()` identical before and after every call, including the four failing ones. `PROBE PASSED` |
| `/tmp/probe_superuser_seam.sh` (MIG-11) | On the deployed host over SSH, quoted-heredoc delivery (`cat <<'REMOTE' \| ssh host 'bash -s'`): runs the preflight block extracted from the edited `deploy/update.sh` | Linux host: `sudo -n -u postgres -c 'psql --version'` exits 0 and prints a version. macOS host: the `rolsuper` query returns `t`. Negative control: with `sudo -k` and no NOPASSWD, the Linux preflight exits 1 **before** `venv.new` exists — assert `[ ! -e "${APP_DIR}/venv.new" ]`. `PROBE PASSED` |
| `/tmp/probe_update_phase65.sh` (MIG-11) | A `/tmp` release tree carrying a two-step `/tmp/mig_chain/` and a `/tmp` install root whose scratch DB is one revision behind; `MIRA_APP_DIR`/`MIRA_TUI_VENV` overrides, `CANONICAL=no` | Phase 6.5 runs after the code swap (assert the tree's `VERSION` is already the new one when the runner starts, by having the step write `cat $APP_DIR/VERSION` into a witness table) and before the restart. Ledger head == tree head afterwards. Negative control: a chain whose second step raises ⇒ exit 1, the restore runs, the aside venvs are put back, `git log -1` on the install repo is the pre-update commit, and the printed report names the backup path. Second negative: `restore` made to fail (point `--backups-dir` at an empty dir) ⇒ exit 1, the word `park`, the manual command block present, and no `mv` executed. `PROBE PASSED` |
| `/tmp/probe_boot_gate_end_to_end.sh` | On a deployed host: stop the service, `DROP TABLE schema_migrations` in a **copy** of the DB (never the live one) — or, on a scratch root, start `main.py` with `PostgresClient` pointed at a scratch DB whose ledger is behind | The server never binds: `port_probe_status 1993` stays free for the whole bounded window, and the journal/stderr carries `schema_revision` in the required-failures list plus the `run: mira migrate` message. `MIRA_POST_GATE_FAILURE_ACTION=park` witnessed by the process still being alive and the port still closed after the bounded wait. Teardown restores the table. `PROBE PASSED` |
| `/tmp/probe_mira_migrate_dispatch.sh` (MIG-3, C-3) | `/usr/local/bin/mira migrate --dry-run`, `mira status`, `mira restore` (no stamp) on a deployed host, and `/opt/mira/tui-venv/bin/python3 -c 'import psycopg'` | The last command **fails** with `ModuleNotFoundError` — the bound that makes the subprocess-exec necessary, witnessed rather than assumed. The first three exit 0/0/1 respectively and their stdout is the server venv's output (assert a marker only `utils.schema_migrations` prints). `mira` from inside a checkout resolves the installed tree, not the cwd (the `-P` flag's contract, `deploy/finalize.sh:318-322`). `PROBE PASSED` |
| `/tmp/probe_container_migrate.sh` (MIG-12, **R3**) | `docker run` a rebuilt image against a **persisted** `mira-pgdata` volume from the previous image, with a one-step chain added | Container start runs the migrate step: the ledger head inside the container equals the new image's tree head. The old behavior's hole is closed — assert the step's witness row exists. **R3-specific negative control**: no boot-guard refusal path exists — assert the container reaches `exec /init` and serves `/v0/api/health` on the same start, with no operator intervention and no `docker exec`, so a regression to the deleted alternative fails here. Second start with no new steps ⇒ exit 0, nothing applied, no schema mutation (fingerprint stable across the two starts). Interpreter witnessed: `ps` inside the container shows `/opt/mira/venv/bin/python3 -m scripts.migrate`, not `/opt/mira/app/venv`. Reports BLOCKED when docker is unavailable. `PROBE PASSED` |
| `/tmp/probe_health_third_assertion.sh` | `curl -sf http://127.0.0.1:1993/v0/api/health` and `scripts/migrate.py status --json` on a deployed host | The health body's `status` is `healthy`, its `version` equals `VERSION`, and the status JSON's `ledger_head` equals the tree head with `pending == []`. Control: on a host deliberately left one revision behind (a scratch install), the third assertion fails and phase 7 reports it. `PROBE PASSED` |

Done when:

- [ ] All seven probes `PROBE PASSED`, or UNVERIFIED/BLOCKED with the venue named.
- [ ] `grep -n "never auto-rolls back" deploy/update.sh deploy/AGENTS.md` returns nothing (D1 reversal complete in both places).
- [ ] `grep -n "The updater never migrates data" deploy/AGENTS.md` returns nothing.
- [ ] The boot gate parks on all four ledger states and boots clean at head.
- [ ] `update.sh`'s header "Privileges:" paragraph matches the actual sudo uses.
- [ ] Seven maps edited in the same commit; audits clean.

---

## Phase 5 — first migration: V0001, BREAKING.md narrowed to the transition release, RELEASE.md rewrite and the REL-1 gate

Goal:        ship the one real migration, narrow `BREAKING.md`'s meaning and ship it as the transition release, and make the release procedure carry the chain and the REL-1 gate.
Preconditions: Phases 1–4 green. `deploy/mira_service_schema.sql` already contains `idx_messages_live_segment_unique` (`:394-396`) and no `idx_messages_active_segment_unique` anywhere in the tree.

Steps:

1. `deploy/migrations/V0001__widen_segment_sentinel_uniqueness.sql` (new,
   MIG-19). Header carries `-- store: postgres` and the expand/contract note.
   Body, in order:
   ```sql
   -- store: postgres
   -- Origin: BREAKING.md (2026-10-10 duplicate-sentinel lockout). The shipped
   -- baseline schema already carries the widened index; this step converges an
   -- install that predates it.
   DO $precondition$
   DECLARE
       bad RECORD;
   BEGIN
       SELECT continuum_id, count(*) AS n
         INTO bad
         FROM messages
        WHERE metadata->>'is_segment_boundary' = 'true'
          AND metadata->>'status' IN ('active', 'paused')
        GROUP BY continuum_id
       HAVING count(*) > 1
        LIMIT 1;
       IF FOUND THEN
           RAISE EXCEPTION
             'V0001 refuses: continuum % carries % live segment sentinels; one live sentinel per continuum must remain. Resolve the duplicate rows first (reassign or remove the stray segment''s messages, then delete the stray sentinel row), then re-run mira migrate.',
             bad.continuum_id, bad.n;
       END IF;
   END
   $precondition$;

   DROP INDEX IF EXISTS idx_messages_active_segment_unique;
   CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_live_segment_unique
       ON messages(continuum_id)
       WHERE metadata->>'is_segment_boundary' = 'true'
         AND metadata->>'status' IN ('active', 'paused');
   ```
   The precondition **REFUSES and names the continuum** — it never repairs
   (MIG-19). `IF NOT EXISTS`/`IF EXISTS` make it idempotent to its own commit
   point (MIG-14), which is what lets R2's override arm run the full chain
   against a DB that already has the index — the maintainer's own hand-patched
   install is exactly that DB (`BREAKING.md` in the tree today carries the
   manual SQL that pre-applied it), and it adopts by reviewing the drift report
   and passing `--adopt-drifted` once.
2. `deploy/migrations/BASELINE` — regenerate: `revision=0001`,
   `fingerprint_sha256` from `baseline_fingerprint()` over a fresh scratch apply
   of the (unchanged) `deploy/mira_service_schema.sql`. Generated, never typed.
3. `BREAKING.md` — **rewrite, do not delete** (MIG-19 + **R8**). The old
   content — the manual `DROP INDEX`/`CREATE UNIQUE INDEX` block for the widened
   segment-sentinel uniqueness — is deleted: V0001 (step 1) now carries that
   change through the chain, so the manual SQL is superseded and leaving it
   would instruct the operator to hand-apply a step the runner is about to run.
   The file **ships with this release**, which is the first release carrying the
   mechanism, and carries the transition content instead: what the operator must
   do or consent to for the one-time superuser-seam bootstrap
   (`deploy/bootstrap_adopt.sql` through `sudo -u postgres psql`, Phase 1 step
   1 / Phase 4 step 1's preflight), why an existing install's first adoption
   needs it, and the manual path the gate prints. That is precisely MIG-19's
   narrowed meaning — **the operator must act or consent** — so the marker is
   not a leftover, it is the correct signal for this one release.
   Binding consequence: the presence gate in `ui/tui/update.py:133-135` and
   `deploy/update.sh:97-102` is **unchanged**, so every existing install
   **refuses** the transition release and takes it through `install.sh` or the
   manual path the file prints. Step 4 records the removal in the next release.
4. `deploy/RELEASE.md` — rewrite the "Breaking releases and `mira update`"
   section:
   - `BREAKING.md`'s only remaining meaning: **the operator must act or
     consent** (D6). A schema change is no longer breaking by definition; the
     chain carries it.
   - **R8 transition-release plan.** The first release carrying the mechanism
     ships `BREAKING.md` (step 3) because the one-time superuser-seam bootstrap
     is operator action. Existing installs therefore refuse it and adopt through
     `install.sh` or the file's manual path; a `pre-provenance install` reaching
     the mechanism this way resolves its SHA through the transitional
     `resolve_release_sha` (Phase 0 step 10). **The next release removes
     `BREAKING.md`** — state the removal as a named step of the following cut,
     not as an intention, so the marker does not linger.
   - **REL-1 release-time integrity gate (R7).** A release is **not published**
     unless, at cut time, all five conditions hold. Any failure = do not
     publish. Each names its owning phase:
     | # | Condition | Owning phase | Mechanical form at cut time |
     |---|---|---|---|
     | a | `VERSION` ↔ tag ↔ commit identity aligned | 0 | `verify_release_identity(repo_root, tag, artifact_root)` (Phase 0 step 21), called by `build_release_artifact` — the artifact is **not produced** when it returns `ok=False` |
     | b | ledger-integrity checks pass over `deploy/migrations/` | 2 | `scripts/migrate.py lint` (2.7): discovery grammar, `order_steps`, and every `lint_steps` finding over the real `deploy/migrations/`, **with no database connection**. Exit 0 required. Called by `build_release_artifact` from this phase onward |
     | c | convergence probe green | 3, 5 | `probe_ver1_convergence.py` (P5-2) recorded `PROBE PASSED` at this cut |
     | d | skip-chain probe green | 3, 5 | `probe_ver2_skipchain.py` (P5-3) recorded `PROBE PASSED` at this cut |
     | e | restore round-trip probe green | 3 | `probe_restore_roundtrip.py` (P3-5) recorded `PROBE PASSED` at this cut |
     (a) and (b) are enforced by the builder: no artifact, no release. (c), (d)
     and (e) are recorded-state conditions — the cut's change report must carry
     each probe's `PROBE PASSED` line, and the checklist names them as
     publish-blocking, not advisory. State the doctrine in the section:
     **authoring-time enforcement is the defense; install-time validation
     (`_check_schema_revision`, Phase 4 step 5) is the backstop.** REL-1 lands
     incrementally: (a) from Phase 0, the full five from this phase — the first
     release carrying the mechanism is the first one gated.
   - The immutability rule: a merged step is never edited; fix forward with a
     new revision. State the enforcement (`lint_steps` + the `CHECKSUM_MISMATCH`
     refusal) and the consequence of violating it (every already-updated install
     refuses the next update with exit 2).
   - The nightly contract: `update nightly-<shortsha>` commits, `applied_by`
     records `nightly-<shortsha>`, `data/nightly_stamp` still advances only after
     the health poll (`deploy/update.sh:361-369`). A nightly is a branch
     archive, carries no `RELEASE_COMMIT`, and is therefore outside REL-1 —
     state that explicitly so the gate's scope is not guessed at.
   - The release-cut checklist gains: build the artifact with
     `build_release_artifact` (Phase 0 step 10) and attach it as the release
     asset; regenerate `deploy/migrations/BASELINE`'s fingerprint; regenerate
     `deploy/mira_service_schema.sql` so it equals baseline + chain (VER-1 is
     the witness); regenerate `bootstrap_adopt.sql`'s ledger DDL from
     `ledger_ddl()`; run the five REL-1 conditions above.
   - The artifact and provenance sections from Phase 0 step 18 stay.
5. `utils/schema_migrations.py` — add `ledger_ddl() -> str` and
   `write_generated_artifacts(repo_root)` emitting the ledger DDL into both
   `deploy/mira_service_schema.sql` and `deploy/bootstrap_adopt.sql` between
   sentinel comment lines. Wire it into the release-cut checklist (step 4). This
   retires the Phase 1 step 1 twin-copy hazard by making one copy generated from
   the other's source (root `AGENTS.md`: Generated, Not Transcribed).
6. Root `AGENTS.md` — rewrite the `BREAKING.md` bullet under
   `## Repo root files`: its presence still makes `mira update` refuse the tree,
   but the meaning is now "the operator must act or consent", and a schema change
   alone no longer requires it. Name the transition-release fact (R8): the first
   release carrying the migration mechanism ships the marker for the one-time
   superuser-seam bootstrap, and the next release removes it. Rewrite the
   `UPGRADE_PATH.md` bullet's "no-upgrade-path doctrine" clause to cite
   `deploy/migrations/`.
7. `README.md:144` and `docs/MANUAL_INSTALL.md:143` — the two identical
   sentences claiming "the database stays greenfield either way … never rely on
   the updater to carry data" are now false. Rewrite both to state that
   `mira update` runs the migration chain, takes a backup first, and that
   `mira restore <STAMP>` is the recovery path.
8. `docs/AGENTS.md:7` — the `UPGRADE_PATH.md` co-ownership Rule states
   "releases never migrate the database in place — a schema change is breaking,
   `mira update` refuses it". Rewrite to the narrowed meaning and cite
   `deploy/migrations/AGENTS.md` as the chain's owner.
9. Map edits, same commit: `deploy/migrations/AGENTS.md` (the `V0001` bullet and
   the `BASELINE` revision), `deploy/AGENTS.md` (the `RELEASE.md` bullet —
   including REL-1's five conditions and their owning phases, and the Rule that
   called any schema change breaking), `docs/AGENTS.md`,
   root `AGENTS.md`. `docs/AGENTS.md` is 18 lines; `deploy/AGENTS.md` is 49 —
   both stay under 60 with these edits.

Verification:

| Probe | Executes | Asserts |
|---|---|---|
| `/tmp/probe_v0001.py` (**VER-7** per-migration data-level) | `live_infra.require("postgres")`; scratch DB; seeds real users via `tests/fixtures/probe_user.py` and real continuum/message rows through the production repositories (`cns/infrastructure/continuum_repository.py`), including one deliberately corrupted continuum carrying **two** live sentinels; then runs the real `migrate` | **Refusal arm**: exit 1, the exception message names the corrupted `continuum_id` and the sentinel count, the ledger is unchanged, and the fingerprint is unchanged (nothing mutated). **Clean arm**: after deleting the stray sentinel through the production path, `migrate` exits 0; `idx_messages_active_segment_unique` is absent from `pg_indexes`; `idx_messages_live_segment_unique` is present and `pg_get_indexdef` contains `status' IN ('active''::text, ''paused''::text)`; the ledger has a `0001` row with `kind='versioned'`, `store='postgres'`, and a nonempty `applied_by`. **Data-level post-conditions**: every message row seeded before the migration is still present with identical `continuum_id`/`metadata`/`created_at` (row-count and content compare, not just structure); a fresh message insert on a paused segment now raises the unique violation the index exists to produce (the 2026-10-10 lockout's guard, witnessed live). **RLS still fails closed**: reconnect as `mira_dbuser` with no user context and assert zero rows and no error on `messages`. **Idempotence**: re-running `V0001`'s body directly exits 0 and changes nothing. Teardown: `residue.assert_user_removed` for every probe user; the fixture drops the DB. `PROBE PASSED` |
| `/tmp/probe_ver1_convergence.py` (**VER-1**) | Two scratch DBs: one from `scratch_database()` (baseline schema), one from `scratch_database()` on a copy of the tree with `deploy/mira_service_schema.sql` reduced to its pre-V0001 form (`git show <pre-V0001-commit>:deploy/mira_service_schema.sql`) then migrated through the chain | `structural_fingerprint(baseline) == structural_fingerprint(reduced + chain)` after normalization. The `BASELINE` fingerprint equals both. This is the Rails staleness guard: the schema file and the chain are two sources of truth and the probe is the witness they agree. `PROBE PASSED` |
| `/tmp/probe_ver2_skipchain.py` (**VER-2**) | Re-run of Phase 3's `probe_migrate_apply.py` skip-chain arm with the real `V0001` in the chain instead of only the `V9xxx` probe steps, plus K = `0000` and K = `0001` | One-invocation and walked chains produce identical fingerprints and identical ledgers modulo `applied_at`/`duration_ms` (MIG-20). `PROBE PASSED` |
| `/tmp/probe_baseline_generated.py` | Runs `write_generated_artifacts` into a `/tmp` copy of the repo and diffs against the working tree | Zero diff — the checked-in ledger DDL in both files is the generated text. Then edits `ledger_ddl()`'s output in the copy and re-runs: both files change together (the twin cannot drift). `PROBE PASSED` |
| `/tmp/probe_breaking_gate_unchanged.py` (MIG-19, **R8**) | `ui/tui/update.py:_gate_and_hand_off` against three real trees: a `/tmp` release tree with a `BREAKING.md`, one without, and **the working tree's own transition-release `BREAKING.md`** from step 3 | With the marker: returns 1, `_print_breaking_block` output contains the file's text verbatim and the `install.sh` command, and `deploy/update.sh` was never invoked (assert by making the tree's `update.sh` a witness script that touches a file). Without: `update.sh` is invoked. **R8 arm**: the transition release this phase produces is itself refused by the unchanged gate — assert the working tree's `BREAKING.md` reaches `_print_breaking_block` and names the superuser-seam bootstrap, so the plan at 5.4 is witnessed rather than asserted. The mechanism is unchanged by MIG-19 — this probe is the witness that narrowing the meaning did not narrow the gate. `PROBE PASSED` |
| `/tmp/probe_release_checklist.sh` (**REL-1**, MIG-1, PROV-2) | Executes the Phase 5 step 4 checklist against the working tree, including all five REL-1 gate conditions | `deploy/migrations/BASELINE`'s `revision` equals the highest `V####` in `deploy/migrations/`; its `fingerprint_sha256` equals a freshly computed `baseline_fingerprint()`; `git status --porcelain` shows no generated artifact left dirty and **no `RELEASE_COMMIT`** (it is injected at artifact build time, never committed — R1). **REL-1(a)**: `verify_release_identity` returns `ok=True` for the tag this cut names, and each of its three checks flips to `False` under its own negative control (5.4's table). **REL-1(b)**: `scripts/migrate.py lint` exits 0 over the real `deploy/migrations/`, and a deliberately corrupted `/tmp` copy (a duplicate revision, then a lint-violating step) exits 2 naming the file and the rule each time. **REL-1(c)/(d)/(e)**: the probe reads this cut's change reports and requires a recorded `PROBE PASSED` line for `probe_ver1_convergence.py`, `probe_ver2_skipchain.py`, and `probe_restore_roundtrip.py`; a missing line is a gate failure, not a warning. **Builder enforcement**: `build_release_artifact` produces no asset when (a) or (b) fails — assert the destination directory is empty and the exit is nonzero. Terminal marker on any failed condition is `DO NOT PUBLISH: <condition>`, distinct from `PROBE FAILED`. **R8 control**: `BREAKING.md` is **present** in the tree at this cut and carries the transition content, and `deploy/RELEASE.md` names its removal in the next cut. `PROBE PASSED` |

Done when:

- [ ] All six probes `PROBE PASSED`.
- [ ] `BREAKING.md` **rewritten to the transition content and still present** (R8); the old manual `DROP INDEX`/`CREATE UNIQUE INDEX` block is gone; `deploy/RELEASE.md` names its removal in the next cut. `grep -rn "BREAKING.md" deploy/ ui/ README.md docs/ AGENTS.md` shows only the gate mechanism, the narrowed-meaning text, and the transition-release plan.
- [ ] REL-1's five conditions all hold for this cut, and `build_release_artifact` refuses to produce an asset when (a) or (b) is made to fail (R7).
- [ ] `deploy/migrations/V0001__widen_segment_sentinel_uniqueness.sql` exists, lints clean, and is idempotent.
- [ ] `deploy/migrations/BASELINE` regenerated; `revision=0001`.
- [ ] VER-1 green: schema file ≡ baseline + chain.
- [ ] Four maps + `README.md` + `docs/MANUAL_INSTALL.md` + `deploy/RELEASE.md` edited in the same commit; audits clean.

---

## Phase 6 — repeatable steps and env/unit reconciliation

Goal:        make reference data converge-on-run and stop `systemone.env` and the systemd unit from silently drifting.
Preconditions: Phases 1–5 green.

Steps:

1. `deploy/migrations/R__usage_pricing.sql` (new, MIG-14). Header
   `-- store: postgres`. Body is a **wholesale replace** written as
   converge-to-declared-state, never apply-delta (MIG-20):
   `DELETE FROM usage_pricing;` followed by a full `INSERT` of the declared rows.
   `usage_pricing` is reference data with no operator columns; `GRANT SELECT` to
   `mira_dbuser` already exists (`deploy/mira_service_schema.sql:844`). Re-runs
   whenever the file's checksum changes (Phase 2 step 3's `repeatable_due`).
2. `utils/schema_migrations.py` — `declared_environment_variables(repo_root) -> dict[str, str]`
   (MIG-18). **Generated, not transcribed**: walk the tree's `.py` files with
   `ast`, collect every `os.getenv("MIRA_…")` / `os.environ["MIRA_…"]` /
   `os.environ.get("MIRA_…")` literal and every module-level constant whose
   value is a `MIRA_*` string (this is how `auth/mode.py:41`'s
   `AUTH_MODE_ENVIRONMENT_FIELD` and
   `utils/power_on_self_test.py:49`'s `POST_GATE_FAILURE_ACTION_ENV` are found —
   neither is in a `config_manager` registry; **see CONFLICTS C-7**). Union with
   the four `config/config_manager.py` registries
   (`SYSTEM_FEATURE_FLAG_ENVIRONMENT_FIELDS:34`,
   `SYSTEM_STRING_ENVIRONMENT_FIELDS:43`,
   `SYSTEMONE_STRING_ENVIRONMENT_FIELDS:48` +
   `SYSTEMONE_PROVIDER_ENVIRONMENT_FIELD:52`,
   `API_INT_ENVIRONMENT_FIELDS:88`) mapped to their Pydantic field names. Return
   `{env_name: owning_symbol}`.
3. `utils/schema_migrations.py` — `reconcile_env_file(env_file, declared, current) -> EnvReconciliation`
   (MIG-18). Pure function over three inputs, returning a Pydantic model with
   `carried_forward: list[str]`, `dropped: list[DroppedVar]` (present in the file,
   no longer declared — each with the file line it came from), `newly_defaulted: list[NewVar]`
   (declared, absent from the file — each with the Pydantic default that will now
   apply), and `rendered: str` (the regenerated file text). The regenerated file
   carries only still-declared variables with their carried-forward values, one
   `NAME=value` per line, sorted, no comments.
4. `utils/schema_migrations.py` — `reconcile_supervisor_unit(...) -> UnitReconciliation`
   (MIG-18). Regenerates `/etc/systemd/system/mira.service` from the current
   `/opt/vault/role-id.txt` and `/opt/vault/secret-id.txt` so an AppRole
   rotation stops being permanent — the unit bakes both values inline today
   (`deploy/finalize.sh:82-83` reads them, `:119-120` writes them). The template
   is `deploy/finalize.sh:104-135`'s, extracted into one shared generator both
   `finalize.sh` and the reconciler call, so the two cannot drift. macOS needs no
   reconciliation: `com.mira.app.plist` (`deploy/finalize.sh:235-261`) execs
   `run.sh`, which reads `/opt/vault/*.txt` at every start
   (`deploy/finalize.sh:208-209`) — state this in the map, do not generate a
   plist. Requires sudo on Linux; runs only when the unit exists and its baked
   `VAULT_SECRET_ID` differs from the file's.
5. `utils/schema_migrations.py` — wire 2/3/4 as `R__reconcile_env.py`
   (`-- store: env`, `up(ctx)` calls `reconcile_env_file` and writes via
   `ctx.env`, `verify(ctx)` re-reads and asserts the declared set) and
   `R__reconcile_unit.py` (`-- store: env`). Both **REPORT** what they changed:
   `dropped` and `newly_defaulted` lists printed to stdout and recorded in the
   ledger row's `name` field suffix. A renamed variable silently taking a
   Pydantic default is the failure this kills, so a nonempty `newly_defaulted`
   list is printed at WARNING, not swallowed.
6. `deploy/update.sh` phase 6.5 — the runner already runs both repeatables; no
   new wiring. Add one line to the phase-6.5 output summarizing
   `dropped`/`newly_defaulted` counts so the operator sees the reconciliation
   happened.
7. Map edits, same commit:
   - `deploy/migrations/AGENTS.md` — `## Files` bullets for the three step files;
     `## Rules` gains the repeatable-step contract (wholesale replace,
     converge-to-declared-state, re-runs on checksum change) and the env/unit
     reconciliation's REPORT obligation.
   - `utils/AGENTS.md` — the `schema_migrations.py` bullet gains
     `declared_environment_variables` (AST-generated, not transcribed, and why:
     two live `MIRA_*` readers sit outside every registry) and the reconcilers.
   - `config/AGENTS.md` — a one-line citation that the four registries in
     `config/config_manager.py` are one of two sources for
     `declared_environment_variables`, owned by `utils/AGENTS.md`.
   - `deploy/AGENTS.md` — the `finalize.sh` bullet gains the shared unit
     generator and its second caller.
   - `deploy/lib/AGENTS.md` — untouched unless the generator lands in a lib file;
     if it does, add the bullet.

Verification:

| Probe | Executes | Asserts |
|---|---|---|
| `/tmp/probe_repeatable_step.py` | `live_infra.require("postgres")`; scratch DB; `migrate` twice; then edit one byte of `R__usage_pricing.sql` in a `/tmp` tree copy and `migrate` again | First run applies it and writes one ledger row with `kind='repeatable'`. Second run applies nothing (`repeatable_due` empty, no new row). Third run (edited body) applies it again with the new checksum. Data-level: `SELECT count(*) FROM usage_pricing` equals the declared row count after every run, and a hand-inserted extra row is **gone** after the third (wholesale replace witnessed, not delta). `PROBE PASSED` |
| `/tmp/probe_env_reconcile.py` (MIG-18) | Calls the real `declared_environment_variables()` over the working tree, then the pure `reconcile_env_file()` against four hand-written `/tmp` env files | The declared set contains all four registries' variables **and** `MIRA_AUTH_MODE` and `MIRA_POST_GATE_FAILURE_ACTION` — the two out-of-registry readers, named by their owning symbol. A file carrying a variable no longer declared ⇒ it appears in `dropped` with its line number and is absent from `rendered`. A declared variable absent from the file ⇒ it appears in `newly_defaulted` with the Pydantic default named. A renamed variable (`MIRA_TIMEZONE` → `MIRA_TZ`) produces **both** a `dropped` and a `newly_defaulted` entry — the exact failure MIG-18 exists to surface. `rendered` is `0600`-bound when written via `ctx.env` and round-trips: re-parsing `rendered` yields the same carried-forward values. No secret appears in `rendered` (the file is non-secret by contract, `deploy/finalize.sh:30-33`). `PROBE PASSED` |
| `/tmp/probe_unit_reconcile.py` (MIG-18) | On a Linux host with a real `mira.service`: reads the unit, compares its baked `VAULT_SECRET_ID` to `/opt/vault/secret-id.txt`, runs the reconciler with sudo, re-reads | When they differ, the regenerated unit carries the file's current value and every other line is byte-identical to `deploy/finalize.sh:104-135`'s template (diff the two generators' output — the shared-template claim witnessed). `systemctl daemon-reload` succeeds and `systemctl cat mira` shows the new value. When they match, nothing is written (mtime unchanged). macOS: the reconciler is a no-op and says why. Reports BLOCKED without sudo. `PROBE PASSED` |
| `/tmp/probe_env_reconcile_live.py` | End-to-end on a deployed host: `cp /opt/mira/systemone.env /tmp/env.before`; append `MIRA_RETIRED_VAR=1`; run `mira migrate`; diff | `MIRA_RETIRED_VAR` is gone from the file; every still-declared variable kept its value; stdout names `MIRA_RETIRED_VAR` under dropped. The service still boots healthy afterwards (health poll, bounded). Teardown: `/tmp/env.before` compared, then removed. `PROBE PASSED` |

Done when:

- [ ] All four probes `PROBE PASSED` (unit probe may be BLOCKED without a Linux host + sudo).
- [ ] `declared_environment_variables()` finds the two out-of-registry readers — the mechanical check that the reconciler cannot silently drop them.
- [ ] The systemd unit template has exactly one generator, called from both `deploy/finalize.sh` and the reconciler.
- [ ] Five maps edited in the same commit; audits clean.

---

## Phase 7 — per-user SQLite chain

Goal:        give the per-user encrypted SQLite an ordered ledger without abandoning converge-on-open.
Preconditions: Phases 1–3 green (the ledger/step vocabulary is reused, not reinvented).

Steps:

1. `utils/userdata_migrations.py` (new, MIG-16). An ordered step list, each
   entry `(version: int, name: str, apply: Callable[[sqlite3.Cursor], None])`.
   The ledger is `PRAGMA user_version` — no new table.
   `run_userdata_migrations(conn) -> int` reads `PRAGMA user_version`, applies
   every step with `version > current` in ascending order inside one
   `sqlite3` transaction per step, sets `PRAGMA user_version` to the step's
   version in the same transaction, and returns the new version. A step raising
   rolls back that step only; the DB stays at the last committed version and the
   exception propagates (fail-fast — the caller is `_ensure_database`, on the
   user's own open path).
2. `utils/userdata_manager.py` — convert the four existing helpers **as-is**,
   introspection guards retained (MIG-16):
   | Source | Becomes | Version |
   |---|---|---|
   | `_migrate_pager_schema` (`utils/userdata_manager.py:300-349`) | `utils/userdata_migrations.py` step 1 | 1 |
   | `_migrate_domaindoc_schema` (`:376-414`) | step 2 | 2 |
   | `_migrate_domaindoc_versions_unique` (`:416-451`) | step 3 | 3 |
   | `_migrate_contacts_schema` (`:475-488`) | step 4 | 4 |
   Bodies are moved verbatim; the `PRAGMA table_info` / `PRAGMA index_list`
   guards stay, because converge-on-open must still hold for a database whose
   `user_version` is 0 but whose tables already exist (every install today).
3. `utils/userdata_manager.py:_initialize_tool_schemas` (`:189-208`) — keep the
   `CREATE TABLE IF NOT EXISTS` calls (converge-on-open is retained, MIG-16),
   replace the four direct `_migrate_*` calls with one
   `run_userdata_migrations(self.connection)` call, and **delete** the
   boundary comment at `:192-194` ("If user DB migrations become order-sensitive
   again, introduce versioned migrations") — the behavior it defers now exists,
   and `docs/AGENTS_MAP_SPEC.md` requires deleting a statement of a behavior
   that is gone.
4. `utils/userdata_manager.py` — a database opened for the first time gets
   `PRAGMA user_version` set to the head version **after** the `CREATE TABLE IF
   NOT EXISTS` statements and **without** running the four steps, because there
   is nothing to converge. Distinguish by whether the file existed before
   `_ensure_database` touched it (`utils/userdata_manager.py:184-187` already
   branches on `self.db_path.exists()`). A pre-existing file at
   `user_version == 0` runs the full chain (its guards make that safe).
5. Tables created ad hoc by tools outside the manager (MIG-16's last clause)
   stay where they are. Enumerate them in the map bullet so a future step does
   not assume the manager owns them.
6. Map edits, same commit:
   - `utils/AGENTS.md:48` — the `userdata_manager.py` bullet: "Owns inline
     schema init/migration" becomes "Owns inline schema init; ordered migrations
     live in `userdata_migrations.py`". Add a `userdata_migrations.py` bullet
     naming `run_userdata_migrations`, the `PRAGMA user_version` ledger, the
     four converted steps and their versions, the new-file shortcut, and the
     ad-hoc-tool-tables clause. Line-count check (C-5) — this is the edit most
     likely to cross 80.
   - `tools/implementations/AGENTS.md` — a one-line citation that per-user
     SQLite schema versioning is owned by `utils/AGENTS.md`
     (`utils/userdata_migrations.py`), for the tools that create their own
     tables.

Verification:

| Probe | Executes | Asserts |
|---|---|---|
| `/tmp/probe_userdata_migrations.py` (MIG-16) | `live_infra.require("vault")` (the Fernet key chain needs `userdata_encryption_key`); a real throwaway user from `tests/fixtures/probe_user.py`; four database shapes built for real: (a) brand-new empty file, (b) a file with the pre-migration `pager_devices` shape and `user_version=0`, (c) a file with the legacy `domaindoc_versions` unique index `(domaindoc_id, version_num)` and real rows, (d) a fully current file | (a) `user_version == 4` and every table exists, with **no** step body executed — witness by having each step append to a list the probe reads. (b) all four steps run; the introspection guards skip the DDL that already matches; `user_version == 4`. (c) `_migrate_domaindoc_versions_unique`'s rebuild fires: the new `(domaindoc_id, section_id, version_num)` index exists, the old one is gone, and **every row survived with its values byte-identical** (the map's "copying rows verbatim — no renumbering" contract, `utils/AGENTS.md:48`, witnessed at data level). (d) nothing runs. **Interrupt arm**: a step patched to raise on (c) leaves `user_version` at 2, the legacy table intact, and the exception propagated (not swallowed); re-running after unpatching completes to 4. **Encryption arm**: `encrypted__` columns written before the migration decrypt correctly after it — the Fernet path (`utils/userdata_manager.py:522-557`) still round-trips. Teardown: `residue.assert_user_removed(user_id)`. `PROBE PASSED` |
| `/tmp/probe_userdata_no_manager_regression.py` | Opens a real user DB through `get_user_data_manager()` (`utils/userdata_manager.py:667`) before and after the phase's edit, on a copy of a real `data/users/<uid>/userdata.db` | `select`/`insert`/`update` on `pager_devices`, `domaindocs`, `contacts`, `trigger_rules`, `sidebar_activity` all behave identically; `clear_manager_cache()` closes cleanly. The control arm: the same operations against an unmodified checkout's manager produce the same results — no behavior change beyond the ledger. `PROBE PASSED` |

Done when:

- [ ] Both probes `PROBE PASSED`.
- [ ] `grep -n "_migrate_pager_schema\|_migrate_domaindoc_schema\|_migrate_domaindoc_versions_unique\|_migrate_contacts_schema" utils/userdata_manager.py` returns only the deleted-call sites' absence — the four helpers live in `utils/userdata_migrations.py`.
- [ ] The `:192-194` boundary comment is deleted, not annotated.
- [ ] Converge-on-open still holds: a `user_version=0` database with current tables opens without error and without running a step body.
- [ ] Two maps edited in the same commit; `utils/AGENTS.md` line count recorded and under 80 (or the consolidation documented).

---

## Phase 8 — docs, maps, probe index

Goal:        close the documentation surface and freeze the probe index.
Preconditions: Phases 0–7 shipped.

Steps:

1. Run both `docs/AGENTS_MAP_SPEC.md` audits over every map touched by Phases
   0–7 (path-anchor, `## Files` bare-name, citation, budget, density). Fix every
   `MISSING-PATH`, `MISSING-FILE`, `MISSING-MAP`, `OVER-TARGET`,
   `OVER-CEILING`, `LONG-BULLET`.
2. Run the reciprocity audit: every wiring edge added by this overhaul is stated
   fully by exactly one owning map and cited in one line by every counterpart.
   The edges to check, in both directions:
   | Edge | Owner | Counterparts citing it |
   |---|---|---|
   | `update.sh` phase 6.5 → `scripts/migrate.py` → `utils/schema_migrations.py` | `deploy/AGENTS.md` | `scripts/AGENTS.md`, `utils/AGENTS.md` |
   | `mira migrate` → server-venv subprocess | `ui/tui/AGENTS.md` | `scripts/AGENTS.md` |
   | `bootstrap_adopt.sql` callers (`postgresql.sh`, `deploy_database.sh`, `init-mira.sh`, `scratch_db.py`) | `deploy/AGENTS.md` | `deploy/docker/scripts/AGENTS.md`, `tests/fixtures/AGENTS.md` |
   | ledger `GRANT SELECT` → `_check_schema_revision` | `utils/AGENTS.md` | `deploy/AGENTS.md` |
   | `BOOT_FLUSH_PRESERVE_PREFIXES` two readers | `clients/AGENTS.md` | root `AGENTS.md` (`main.py` bullet) |
   | unit-template shared generator | `deploy/AGENTS.md` | `utils/AGENTS.md` |
   | `deploy/migrations/BASELINE` fingerprint generator | `deploy/migrations/AGENTS.md` | `deploy/AGENTS.md`, `utils/AGENTS.md` |
   | release artifact → `RELEASE_COMMIT` → the two tarball fetch sites (R1) | `deploy/AGENTS.md` | `deploy/lib/AGENTS.md`, `ui/tui/AGENTS.md`, `utils/AGENTS.md` |
   | `clear_code_preserving_state` and its two callers (R4) | `deploy/lib/AGENTS.md` | `deploy/AGENTS.md` |
   | REL-1's five gate conditions → their owning phases (R7) | `deploy/AGENTS.md` | `deploy/migrations/AGENTS.md`, `utils/AGENTS.md` |
3. `deploy/HOW_TO_MIGRATE_OLD_INSTALLS.txt` — re-read against the new mechanism.
   It describes agent-run salvage across a greenfield reinstall; the chain now
   carries an existing database forward, so its opening framing is stale. Rewrite
   the framing to scope it to the two cases that still reinstall: a major
   crossing (D6), and the **R8 transition release** — which ships `BREAKING.md`,
   is therefore refused by the unchanged `mira update` gate, and sends the
   operator down exactly this file's reinstall + agent-run history-import path
   (5.3, 5.4). Leave the runbook body intact.
4. `NEARFUTURE_FEATURES.md` — remove any migration-mechanism item now shipped.
5. Write the probe index below into `deploy/migrations/AGENTS.md` as a
   `## Verification` deep-dive (the spec permits one deep-dive titled with a
   domain term), or into `deploy/AGENTS.md` if the line budget forces it. A
   claimed defense without a witness is a defect (root `AGENTS.md`: Every
   Claimed Defense Has a Witness); this index is the witness list.
6. Promote-or-discard pass (`writing-probes` skill, moment six), under **R9**.
   Two probes are already permanent as production `CheckSpec`s:
   - `probe_post_schema_revision.py` → `CheckSpec("schema_revision")` (Phase 4 step 5).
   - `probe_secrets_manifest.py` → `CheckSpec("vault_secrets_manifest")` (Phase 1 step 10).
   **The load-bearing witnesses are destined for `tests/protected/`** as a
   permanent battery — a permanent mechanism with disposable witnesses is a
   contradiction. Their index rows carry the marker
   `→ protected (pending authorization)`, and they **remain disposable `/tmp`
   probes** under `tests/tmp/AGENTS.md` until step 8's phrase is typed. The set,
   by invariant (twelve index rows):
   | Invariant | Index rows |
   |---|---|
   | convergence (VER-1) | P3-1, P5-2 |
   | skip-chain (VER-2, MIG-20) | P3-1, P5-3 |
   | interrupt / SIGKILL (VER-3) | P3-2, P3-3 |
   | the guard set — every refusal the mechanism can issue | P2-2 (the four ledger refusals), P3-9 (`ADOPTION_DRIFT` + the recorded override), P4-1 (the boot gate's four park conditions — its **live** half is already permanent as `CheckSpec("schema_revision")`, so what is protected-destined is the offline-checkable half: the four condition→message derivations), P5-1 (V0001's precondition refusal), P5-5 (the `BREAKING.md` gate) |
   | privilege + restore round-trip (VER-5, R6, REL-1(e)) | P3-4, P3-5 |
   Everything else is disposable: delete from `/tmp`, do not commit. No file is
   added to `tests/tmp/` (autodeleted on sight) and none to `tests/protected/` in
   this phase — step 8 is the only route in.
7. `docs/AGENTS.md:5` — the "path printed in a doc must exist" Rule names
   `BREAKING.md` as the one sanctioned exception (a release-time artifact).
   `BREAKING.md` still qualifies under MIG-19's narrowed meaning and is in fact
   present at this cut (R8, 5.3); verify the sentence still reads true and leave
   it, or amend the parenthetical to the narrowed meaning.
8. **Protected-battery authorization decision (R9).** Explicit, recorded, not
   skippable. Ask the operator to type the exact phrase
   `AUTHORIZE PROTECTED TEST SAVE` for the twelve index rows in step 6's table
   (`tests/AGENTS.md:45`, `tests/protected/AGENTS.md:5`). Authorization is
   per-file and per-save: not agent judgment, not a paraphrase, not a prior
   authorization on a different file, not inferred intent.
   - **Not granted** — the state this runbook is written in: every probe in the
     table stays a disposable `/tmp` probe, the index marker stays
     `→ protected (pending authorization)`, nothing is written under `tests/`,
     and the request is recorded in this phase's change report as an open
     decision with the phrase quoted verbatim, so granting it needs no
     re-derivation of what was asked.
   - **Granted**: save each named file, one commit per authorization, and add
     its `## Files` bullet to `tests/protected/AGENTS.md` in the same commit.
   - **Obstacle to resolve before saving, binding.**
     `tests/protected/AGENTS.md:37-40` requires a battery there to be runnable
     offline and to need no Postgres, Vault, or Valkey, and routes a check that
     needs live infrastructure to a POST path-probe under
     `utils/power_on_self_test.py` instead. Every probe in step 6's table is a
     live-infrastructure probe — scratch databases, a real `SIGKILL`ed
     subprocess, TCP as `mira_admin`. Admission therefore needs one of two
     things, and the request must name which:
     (i) an operator waiver of the offline rule recorded in
     `tests/protected/AGENTS.md` beside the authorization — the folder already
     carries one such recorded exception at `tests/protected/AGENTS.md:47`
     (`injection_defense_probe.py`'s live half); or
     (ii) reshaping each witness into the two homes the doctrine already
     provides — the offline-checkable half as a `tests/protected/` battery over
     the pure functions (`validate`, `order_steps`, `lint_steps`,
     `render_drift_report`, `verify_release_identity`), the live half as
     `CheckSpec` path-probes.
     The phrase waives disposability, not the offline rule: do not save a live
     probe into `tests/protected/` on the strength of it alone.

Verification:

| Probe | Executes | Asserts |
|---|---|---|
| `/tmp/probe_map_anchor_audit.sh` (P8-1) | The four `docs/AGENTS_MAP_SPEC.md` audit loops (path-anchor, `## Files` bare-name, citation, budget, density) run verbatim over the fourteen maps this overhaul touched | Zero `MISSING-PATH`, `MISSING-FILE`, `MISSING-MAP` lines. Zero `OVER-CEILING`. Any `OVER-TARGET` or `LONG-BULLET` is either resolved by the consolidation pass or recorded as named debt in the change report. Budget line counts printed per map. `PROBE PASSED` |
| `/tmp/probe_map_reciprocity.sh` (P8-2) | For each of the ten edges in step 2's table: grep the owning map for the full statement and every counterpart map for a one-line citation, then resolve each cited path | Every edge has exactly one owning statement and a citation in each counterpart; every cited map path resolves to an existing file; no edge is stated fully in two maps (the one-home rule). The two permanent `CheckSpec`s are named in `utils/AGENTS.md` and cited from `deploy/AGENTS.md` and `deploy/migrations/AGENTS.md`. `PROBE PASSED` |
| `/tmp/probe_requirement_coverage.sh` | Parses this runbook's requirement → phase matrix and the probe index; for each requirement ID, resolves the named probes and reads their recorded exit state from the change reports | Every one of MIG-1…20, PROV-1…4, CUST-1…7, VENV-1…5, VER-1…7, CROSS-1…4, and REL-1 maps to at least one probe whose recorded state is `PROBE PASSED` or an explicit `UNVERIFIED` naming its blocker. For REL-1 the coverage is per-condition: all five of (a)–(e) resolve to a named probe, and a condition with no recorded green state fails the probe and prints `DO NOT PUBLISH`. A requirement with no probe, or a probe with no recorded run, fails the probe and prints the ID. No requirement resolves to a mock-based or file-based check. `PROBE PASSED` |

### Probe index

Marker in the Requirement column (R9):
`→ protected (pending authorization)` = a load-bearing witness whose permanent
home is `tests/protected/`, held as a disposable `/tmp` probe until the operator
types `AUTHORIZE PROTECTED TEST SAVE` (Phase 8 steps 6 and 8). The twelve marked
rows are the set named there; nothing lands in `tests/protected/` before then.

| ID | Probe | Requirement | Venue | Tier |
|---|---|---|---|---|
| P0-1 | `probe_venv_stamp.sh` | VENV-1/2/4, VER-6 | `/tmp` venvs | 2 |
| P0-2 | `probe_custom_export.sh` | CUST-1/2/4/7, VER-6 | `/tmp` git tree | 2 |
| P0-3 | `probe_prov_commit.sh` | PROV-2, VER-6, REL-1(a) | `/tmp` + real remote | 2 |
| P0-3b | `probe_prov_status.sh` | PROV-4, VER-6 | `/tmp` install tree | 1 |
| P0-3c | `probe_venv_drift.sh` | CUST-3, VER-6 | `/tmp` venv | 2 |
| P0-4 | `probe_update_sh_scratch.sh` | PROV-1, VENV-2/5, VER-6 | `MIRA_APP_DIR` override | 2 |
| P0-5 | `probe_bashtool_no_stash_dependency.py` | CUST-5 | in-process | 1 |
| P0-6 | `probe_no_new_deps.py` | CROSS-4 | diff | 0 |
| P0-7 | `probe_release_artifact.sh` | PROV-2, REL-1(a), VER-6 | `/tmp` clone + real tag | 2 |
| P0-8 | `probe_state_preserve.sh` | CUST-6, R4, VER-6 | two `/tmp` install roots | 2 |
| P1-1 | `probe_bootstrap_adopt.py` | MIG-8 | scratch DB + superuser | 2 |
| P1-2 | `probe_ledger_grants.py` | MIG-2 | scratch DB, TCP | 2 |
| P1-3 | `probe_secrets_manifest.py` | MIG-17 | live Vault | 2 |
| P1-4 | `probe_post_manifest_check.py` | MIG-17 | live infra | 2 |
| P1-5 | `probe_baseline_fingerprint.py` | MIG-1, VER-1 | two scratch DBs | 2 |
| P2-1 | `probe_migrate_discovery.py` | MIG-1/13/14 | `/tmp` step dir | 2 |
| P2-2 | `probe_migrate_validate.py` | MIG-4/14, VER-4, REL-1(b) → protected (pending authorization) | four scratch DBs | 2 |
| P2-3 | `probe_migrate_status.py` | MIG-3, PROV-4 | scratch DB | 1 |
| P2-4 | `probe_migrate_dryrun.py` | MIG-3 | scratch DB | 2 |
| P2-5 | `probe_fingerprint_determinism.py` | VER-1 | two scratch DBs | 2 |
| P2-6 | `probe_exit_codes.py` | MIG-3 | scratch DB + closed port | 2 |
| P3-1 | `probe_migrate_apply.py` | VER-1, VER-2, MIG-20 → protected (pending authorization) | scratch DBs | 2 |
| P3-2 | `probe_migrate_interrupt.py` | VER-3 → protected (pending authorization) | scratch DB | 2 |
| P3-3 | `probe_sigkill_resume.py` | VER-3 → protected (pending authorization) | subprocess + SIGKILL | 2 |
| P3-4 | `probe_migrate_privileges.py` | VER-5, R6 → protected (pending authorization) | scratch DB, TCP as `mira_admin` | 2 |
| P3-5 | `probe_restore_roundtrip.py` | MIG-7, VER-5, D1, R6, **REL-1(e)** → protected (pending authorization) | scratch DB | 2 |
| P3-6 | `probe_backup_retention.py` | MIG-6 | `/tmp` backups dir | 2 |
| P3-7 | `probe_vault_adapter.py` | MIG-15 | live Vault, scratch KV path | 2 |
| P3-8 | `probe_valkey_invalidate.py` | MIG-15 | live Valkey | 2 |
| P3-9 | `probe_adoption_drift.py` | MIG-9, VER-4, R2 → protected (pending authorization) | four scratch DBs | 2 |
| P3-10 | `probe_advisory_lock.py` | MIG-3 | two subprocesses | 2 |
| P3-11 | `probe_operator_input.py` | D5 | pty + closed stdin | 2 |
| P4-1 | `probe_post_schema_revision.py` | MIG-10 → protected (pending authorization); live half already permanent as `CheckSpec("schema_revision")` | four scratch DBs | 2 |
| P4-2 | `probe_superuser_seam.sh` | MIG-11 | deployed host, SSH | 2 |
| P4-3 | `probe_update_phase65.sh` | MIG-11, D1 | scratch tree + scratch DB | 2 |
| P4-4 | `probe_boot_gate_end_to_end.sh` | MIG-10 | deployed host | 2 |
| P4-5 | `probe_mira_migrate_dispatch.sh` | MIG-3, C-3 | deployed host | 1 |
| P4-6 | `probe_container_migrate.sh` | MIG-12, R3 | docker | 2 |
| P4-7 | `probe_health_third_assertion.sh` | MIG-11 | deployed host | 1 |
| P5-1 | `probe_v0001.py` | MIG-19, VER-7 → protected (pending authorization) | scratch DB + probe users | 2 |
| P5-2 | `probe_ver1_convergence.py` | VER-1, REL-1(c) → protected (pending authorization) | two scratch DBs | 2 |
| P5-3 | `probe_ver2_skipchain.py` | VER-2, MIG-20, REL-1(d) → protected (pending authorization) | scratch DBs | 2 |
| P5-4 | `probe_baseline_generated.py` | MIG-2 | `/tmp` repo copy | 1 |
| P5-5 | `probe_breaking_gate_unchanged.py` | MIG-19, R8 → protected (pending authorization) | three trees, one the working tree | 2 |
| P5-6 | `probe_release_checklist.sh` | MIG-1, PROV-2, **REL-1(a–e)** | working tree | 1 |
| P6-1 | `probe_repeatable_step.py` | MIG-14, MIG-20 | scratch DB | 2 |
| P6-2 | `probe_env_reconcile.py` | MIG-18 | AST over the tree | 2 |
| P6-3 | `probe_unit_reconcile.py` | MIG-18 | Linux host + sudo | 2 |
| P6-4 | `probe_env_reconcile_live.py` | MIG-18 | deployed host | 2 |
| P7-1 | `probe_userdata_migrations.py` | MIG-16, VER-7 | live Vault + probe user | 2 |
| P7-2 | `probe_userdata_no_manager_regression.py` | MIG-16 | real DB copy | 1 |
| P8-1 | `probe_map_anchor_audit.sh` | CROSS-2 | repo | 0 |
| P8-2 | `probe_map_reciprocity.sh` | CROSS-2 | repo | 0 |
| P8-3 | `probe_requirement_coverage.sh` | all | this runbook + change reports | 0 |

Done when:

- [ ] Both map audits clean over all fourteen touched maps, and `probe_requirement_coverage.sh` reports no uncovered requirement ID.
- [ ] Every requirement row in the matrix names at least one probe that reported `PROBE PASSED`, or an explicit UNVERIFIED with its blocker.
- [ ] The fourteen touched maps are: `AGENTS.md` (root), `deploy/AGENTS.md`, `deploy/lib/AGENTS.md`, `deploy/migrations/AGENTS.md`, `deploy/docker/scripts/AGENTS.md`, `scripts/AGENTS.md`, `ui/AGENTS.md`, `ui/tui/AGENTS.md`, `utils/AGENTS.md`, `clients/AGENTS.md`, `config/AGENTS.md`, `tests/fixtures/AGENTS.md`, `docs/AGENTS.md`, `tools/implementations/AGENTS.md`.
- [ ] No probe file remains in the repository tree.
- [ ] `deploy/HOW_TO_MIGRATE_OLD_INSTALLS.txt` framing re-scoped to major crossings and the R8 transition release.
- [ ] The two promoted `CheckSpec`s are the only permanent additions made in this phase; nothing landed in `tests/tmp/` or `tests/protected/`.
- [ ] Step 8's authorization request is recorded in the change report with the phrase `AUTHORIZE PROTECTED TEST SAVE` quoted verbatim, the twelve named index rows listed, and the offline-rule obstacle (`tests/protected/AGENTS.md:37-40`) stated with its two admissible resolutions — an open decision, not a silently deferred one (R9).

---

## CONFLICTS

Requirements that do not match source, or that source makes unsafe as written.
Each row records the ruling that closed it, or stays marked open. No row is
deleted: a closed conflict is the record of why the phases read the way they do.

| ID | Requirement | Source fact | Nature | Resolution |
|---|---|---|---|---|
| C-1 | VENV-1 names `deploy/finalize.sh` as the TUI-venv creation site | The TUI venv is created in `deploy/python.sh:294-313` (Step 5b, `python -m venv /opt/mira/tui-venv`). `deploy/finalize.sh` never creates a venv; it writes the `/usr/local/bin/mira` wrapper at `:310-323` and `run.sh` at `:189`. | Wrong file named | **Open — no ruling touches this row.** Phase 0 step 3 stamps in `deploy/python.sh`. Confirm, or name a real `finalize.sh` venv site. |
| C-2 | PROV-2: the release process embeds a `RELEASE_COMMIT` file in the release tree at tag time | The payload tree is GitHub's auto-generated tag archive (`deploy/python.sh:124` wgets `archive/refs/tags/${RELEASE_TAG}.tar.gz`; `ui/tui/update.py:79-87` downloads the same). An auto-generated archive contains only the tagged tree, so a `RELEASE_COMMIT` committed into that tree cannot contain the SHA of the commit that adds it. `install.sh` and `deploy/deploy.sh:65-73` do `git clone --branch <tag>`, where the SHA *is* available — but `python.sh` discards the clone and re-fetches the tarball. | Mechanically impossible as written | **CLOSED by R1 — releases become self-describing artifacts; the GitHub auto-archive path is retired.** `deploy/RELEASE.md` gains a tag-time artifact-build step (Phase 0 step 18) executed by `deploy/lib/provenance.sh:build_release_artifact` (Phase 0 step 10): `git archive <tag>` → extract → write `RELEASE_COMMIT` (the tagged commit's full 40-hex SHA) into the tree → re-tar → attach as the release asset. The file is injected at artifact build time and **never committed** — a committed file cannot contain its own tag's SHA, which is exactly the chicken-and-egg named in the source fact, and is why the artifact, not the repo, carries provenance. Both tarball fetch sites switch from `archive/refs/tags/…` to the release-asset URL: `deploy/python.sh:120-130` (Phase 0 step 20) and `ui/tui/update.py:86` via `utils/release_identity.py:22`'s new `RELEASE_ASSET_URL` (Phase 0 step 21), plus the manual-recovery block at `deploy/RELEASE.md:55`. The bootstrap `git clone --branch <tag>` at `deploy/deploy.sh:64-77` and `install.sh:135` is **unaffected** — a clone of the tag is its own provenance. The `git ls-remote` tag→SHA resolution is **demoted to a transitional adoption path** for the named state `pre-provenance install`; it is marked transitional in code and in both maps, and its removal is a deliberate operator decision recorded in `deploy/RELEASE.md`, not a silently permanent fallback (ablate, don't depreciate). The second alternative above — installing from `install.sh`'s clone — is rejected: it would keep two payload shapes. Witnesses: P0-3, P0-7. |
| C-3 | MIG-3: `scripts/migrate.py` dispatched as `mira migrate` from `ui/tui/__main__.py`, same dispatch style as `mira update` | `mira` runs on `/opt/mira/tui-venv` (`deploy/finalize.sh:322`), whose `ui/tui/requirements.txt` is exactly websockets, httpx, pydantic, rich, prompt_toolkit — no `psycopg`, no `hvac`. `ui/AGENTS.md:11` states the clients import no server code beyond stdlib `utils/release_identity.py`; `scripts/AGENTS.md` Rule 1 forbids logic in `scripts/`. | The named dispatch cannot import the runner | **Open — no ruling touches this row.** Phase 4 step 6 subprocess-execs `/opt/mira/app/venv/bin/python3 -m scripts.migrate`, mirroring how `mira update` subprocess-execs `deploy/update.sh`. Alternative: add psycopg + hvac to `ui/tui/requirements.txt` (weakens the separate-venv rationale at `deploy/python.sh:296-299`). R2 rides the chosen shape: Phase 4 step 6 forwards `*args_list` verbatim with no flag whitelist, so `mira migrate --adopt-drifted` needs no dispatch-side change. |
| C-4 | MIG-2: the ledger table is created by the baseline schema | `deploy/AGENTS.md` Rule: "Do not add migration DDL to the schema file." | Rule text contradicts the requirement | **Open — no ruling touches this row.** Phase 1 steps 2–3 add the table and amend the Rule in the same commit. Confirm the amendment's wording (permitted contents: the ledger and its `GRANT SELECT`). |
| C-5 | CROSS-2: every behavior change updates its map in the same commit | `utils/AGENTS.md` is already 76 lines. `docs/AGENTS_MAP_SPEC.md` requires a consolidation pass at 60 and a **split into subdirectory maps** at 80. `utils/` has no subdirectories (38 `.py` files, flat). Phases 1, 2, 3, 4, 6, and 7 each add or extend a bullet there. | The spec's 80-line remedy is unavailable for `utils/` | **Open as to remedy; scope widened by R1.** Either consolidate `utils/AGENTS.md` hard enough to absorb the edits under 80, or create subdirectories under `utils/` (a structural change well beyond this overhaul's scope), or accept an over-ceiling map as recorded debt. R1 adds a seventh edit to the same 76-line map — Phase 0 step 17's `release_identity.py` bullet (`TAG_TARBALL_URL` → `RELEASE_ASSET_URL`, plus `verify_release_identity`) — so the consolidation pass fires in **Phase 0's** commit rather than Phase 1's, and Phase 1 step 12 records the count from there. |
| C-6 | D4 default: restore resets the `public` schema instead of dropping the database | `DROP SCHEMA public CASCADE` requires ownership of `public`. On PG15+ `public` is owned by `pg_database_owner` and `mira_admin` owns the database (`deploy/postgresql.sh:116`, `createdb -O mira_admin`), so it works. On PG14 and earlier `public` is owned by the bootstrap superuser, so it fails. `deploy/AGENTS.md` states the schema floor is 14 (Arch uses the distro's current major; 18.6 verified live). Also, `pg_restore` of a dump containing `CREATE EXTENSION vector` needs a superuser unless pgvector's control file marks it trusted. | The D4 default is version- and extension-dependent | **CLOSED by R5, with R6 holding the privilege half.** The floor statement at `deploy/AGENTS.md:25` is corrected to **17** — what the project actually ships (`deploy/docker/Dockerfile.base:102-104` installs `postgresql-17`; `deploy/postgresql.sh` uses `brew --prefix postgresql@17`) — in the same commit as the prerequisites phase (Phase 1 step 12). 17 ≥ 15, so `public` is owned by `pg_database_owner` on every supported install and the schema-reset restore path's `DROP SCHEMA public CASCADE` is unambiguous rather than version-dependent; the Arch *package* stays the distro's current major, and the corrected clause keeps those two facts distinct. Breaking-allowed doc correction. R6 keeps `mira_admin` narrow — **no `CREATEDB` grant** — so restore is schema-reset `public` + `pg_restore --no-owner` (3.9 steps 4–5), probe-proven: `probe_migrate_privileges.py` witnesses the refused `CREATE DATABASE` and `probe_restore_roundtrip.py` witnesses the whole round trip as `mira_admin` over TCP, the latter **promoted from phase verification to a REL-1(e) shipping-gate condition** (5.4). The extension half is unchanged and stays with the witnesses: `pg_restore` of a dump carrying `CREATE EXTENSION vector` needs a superuser unless pgvector's control file marks it trusted, which is why 3.9 step 6 asserts the three extensions post-restore and names the `CREATE EXTENSION` a superuser would need. Raising the floor in `deploy/dependencies.sh` and routing `restore` through the superuser seam are both rejected: the first would pin the Arch package, the second would make recovery depend on an elevation the runner otherwise never needs. |
| C-7 | MIG-18: reconcile `systemone.env` from `config/config_manager.py`'s four declared-variable registries | Two live `MIRA_*` readers sit outside every registry: `auth/mode.py:41` `AUTH_MODE_ENVIRONMENT_FIELD = "MIRA_AUTH_MODE"` and `utils/power_on_self_test.py:49` `POST_GATE_FAILURE_ACTION_ENV = "MIRA_POST_GATE_FAILURE_ACTION"`. `MIRA_LOG_DIR` is read by the launcher/unit contract, not by Python. Regenerating from the four registries alone would drop them — the exact silent-defaulting failure MIG-18 exists to kill. | The named source set is incomplete | **Open — no ruling touches this row.** Phase 6 step 2 unions the four registries with an AST sweep of every `MIRA_*` env read in the tree (generated, not transcribed). Confirm the AST sweep is acceptable, or enumerate the out-of-registry names explicitly and own the drift risk. |
| C-8 | CUST-6: one shared implementation called by both `deploy/update.sh` phase 6 and `deploy/python.sh`'s reinstall path | The two clear steps have **different** preserve sets: `deploy/update.sh:286-290` preserves `venv .env data logs .git run.sh venv* tui-venv*`; `deploy/python.sh:87-90` preserves only `venv .env data logs .git`. A reinstall therefore deletes `run.sh` (regenerated later by `deploy/finalize.sh:189`) and deletes any `venv.pre-update-<STAMP>` aside left by a failed update — the rollback artifact D1 depends on. | Unifying the shared function changes `python.sh` behavior | **CLOSED by R4 — preserve sets unified on the `update.sh` superset.** One shared function, `clear_code_preserving_state` in the new `deploy/lib/state_preserve.sh` (Phase 0 step 19), called by both `deploy/update.sh` phase 6 and `deploy/python.sh`'s reinstall path, with the eight-name set (`venv`, `.env`, `data`, `logs`, `.git`, `run.sh`, `venv*`, `tui-venv*`) declared once in that file and restated by neither caller. `--sudo` is a flag on the function, not a second preserve set: `deploy/python.sh` passes it (its tree is root-owned until the chown at `:137-138`), `deploy/update.sh` does not. The reinstall path's current narrower set is **broken-allowed** — this overhaul is a breaking change by ruling, so a reinstall now keeping `run.sh` (which `deploy/finalize.sh:189` regenerates anyway) and keeping every `venv.pre-update-<STAMP>` / `tui-venv.pre-update-<STAMP>` aside needs no compatibility shim. That second half is what unblocks the D1 failure-path rollback: aside venvs survive a mid-recovery reinstall, so `ROLLBACK_VENV_CMDS` (Phase 0 step 7) still has something to `mv` back. Parameterizing the set per caller is rejected — two behaviors is two preserve sets, which is the defect this row names. Witness: P0-8. |
| C-9 | MIG-6: backup destination `/opt/mira/backups/<STAMP>/` | Verified outside both code-clear scopes (`deploy/update.sh:286` and `deploy/python.sh:87` are both `-mindepth 1` under `/opt/mira/app`). But `/opt/mira` is chowned to `$MIRA_USER:$MIRA_GROUP` at install (`deploy/python.sh:138`) while `/opt/mira/backups` does not exist until first use, and creating it needs no sudo only because `/opt/mira` is already user-owned. On the container, `/opt/mira` is image-owned and `/opt/mira/backups` is **not** in the `Dockerfile` `VOLUME` list (`deploy/AGENTS.md` Docker data-persistence bullet) — a container recreate loses every backup. | Container backups are not persisted | **Open — no ruling touches this row.** Add `/opt/mira/backups` to the `Dockerfile` `VOLUME` list and to the documented `docker run` mounts, or disable `take_backup` in the container and say so. Not decided here. Note the interaction with R7: REL-1(e) makes the restore round-trip a publish-blocking condition, so a container install whose backups do not survive a recreate has a green gate and no usable backup — resolve this row before the first gated cut. |
| C-10 | MIG-1 / MIG-8: `BASELINE` declares the revision the schema file corresponds to, and adoption stamps the ledger at `BASELINE` | `deploy/mira_service_schema.sql` already contains V0001's end state (`idx_messages_live_segment_unique` at `:394-396`), so `BASELINE` = head at every cut (VER-1 forces this). An existing install adopted **after** a later release's swap is at an older revision than the new tree's `BASELINE`; stamping it at `BASELINE` would skip every intermediate step. | Literal MIG-8 is unsafe for an install that is behind | **CLOSED by R2 — refuse plus an explicit recorded override; no silent convergence.** Phase 3 step 4 is a three-outcome rule, and MIG-8's literal "stamp at `BASELINE`" survives only as the **fingerprint-match** outcome, which is the one case where stamping is truthful. Mismatch ⇒ **exit 2**, stamp nothing, mutate nothing, print `render_drift_report(observed, expected)` as added / removed / changed objects, then the `ADOPTION_DRIFT` refusal naming the exact override invocation `mira migrate --adopt-drifted` (2.4). The override ⇒ the override is **recorded in the ledger** — `applied_by = "<version>+adopt-drifted"` on every row that run writes, plus the overridden fingerprint's first 12 hex in revision `0001`'s `name`, the same `name`-suffix mechanism Phase 6 step 5 uses, so no column and no DDL change — and the full idempotent chain `0001..head` then runs. The flag is per-invocation and non-sticky, writes no state a later run reads, and no config value or env var disables the check: **unknown drift stays refused, permanently.** The known case is the maintainer's own hand-patched install — the pre-applied partial-unique-index change V0001 covers, whose manual SQL is the `BREAKING.md` currently in the tree: review the drift report, override once, done. `status` derives `adopted_with_drift_override` from the `applied_by` suffix on read (2.8), never from a maintained counter. The former warn-and-converge default is deleted, not annotated. The alternative — adopting against the **old** tree before the code swap — is rejected: it moves the bootstrap out of phase 6.5 and contradicts MIG-11. Witness: P3-9. |
