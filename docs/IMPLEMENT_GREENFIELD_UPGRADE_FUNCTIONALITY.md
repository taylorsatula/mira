# Implement greenfield upgrade functionality

Implementation runbook for the `mira update` mechanism: release artifacts with
provenance, stamp-gated venvs, custom-build preservation, a versioned
database step chain with a ledger, consent preflight, automatic backup and
restore, and a boot guard. Phases ship in order; each phase's commit carries
its code, its map edits, and its probe results.

## Before starting

Read, in this order:

| Source | Why |
|---|---|
| `AGENTS.md` (root) | Critical principles, preventive mechanism rules, comment discipline, git workflow |
| `tests/AGENTS.md` | NO MOCKS doctrine, probe surfaces, the verification loop |
| `writing-probes` skill | Probe anatomy, the probe-able-code craft |
| `maintaining-agentsmd-maps` skill | Map law, trigger table, anchor audits (run before every commit) |
| `deploy/AGENTS.md`, `deploy/lib/AGENTS.md` | Installer, updater, lib source-order rule, elevation helpers |
| `scripts/AGENTS.md` | Thin-shim rule, Vault env precondition |
| `utils/AGENTS.md`, `clients/AGENTS.md` | Where runner logic lives; Vault/Valkey clients |
| `tests/fixtures/AGENTS.md`, `tests/tmp/AGENTS.md` | Scratch DB fixture, disposable-probe policy |

## Binding constraints

- Verification is live probes only, written per the `writing-probes` skill, in `/tmp`, gated on `tests/fixtures/live_infra.py:require`. No test files, no mocks, no fake binaries on `PATH`.
- Every phase-2/3 probe runs against `tests/fixtures/scratch_db.py:scratch_database` with no sudo — the runner must stay probe-able on a dev checkout.
- No new dependencies. Runner imports: stdlib, `psycopg`, `hvac`, `pydantic`, in-repo modules — all already in `requirements.txt`.
- Runner logic lives in `utils/schema_migrations.py`; `scripts/migrate.py` is argparse + rendering only — `scripts/AGENTS.md` makes every script a thin shim, and `deploy/` is not importable.
- Runner entry functions take `conninfo` as a required parameter. Only the shim resolves it, via `clients.vault_client.get_database_url("mira_service", admin=True)` — probes pass a scratch conninfo, and production grows no credential fallback.
- Normal migration runs as `mira_admin` over TCP, sudo-free. The superuser is used only by: `deploy/bootstrap_adopt.sql`, `restore`, and the systemd unit write. These go through the superuser seam (Phase 3 step 9).
- `mira_admin` never receives `CREATEDB` — restore recreates the database through the seam, so the app's admin role stays narrow.
- The positional contract `deploy/update.sh <release-tree> <tag> [flags]` and the env vars `NIGHTLY`, `NIGHTLY_SHA`, `MIRA_APP_DIR`, `MIRA_TUI_VENV` are fixed: an installed pre-mechanism `ui/tui/update.py` invokes the new tree's `update.sh` with them.
- Migrations never run at server boot — the boot guard only refuses; mutation happens where a backup and a rollback exist (`update.sh` phase 6.5, the container start before `exec /init`, the installer).
- A shipped step file is immutable. Fix forward with a new step — installs that already ran it refuse a changed checksum.
- Map edits ship in the same commit as the code they describe — the map skill's trigger table.
- No line-number anchors in code comments or maps — they change on the next edit of the target file.

## Operator language

- Every operator-facing string is plain English: what happens, what is protected, the exact next command.
- Banned on operator surfaces (allowed in logs): `ledger`, `revision`, `checksum`, `migration`, `schema`, `fingerprint`, `adoption`, `baseline`. The command `mira migrate` is allowed.
- Every refusal, park, exit-3, and failure string contains a runnable command (`mira update`, `mira migrate`, `mira restore`, or an explicit `sudo`/`psql`/`pg_restore`/`systemctl` line).
- Machine markers (`MIRA_PLAN_BEGIN`/`END`, `ADOPT-PROBE-BLOCKED`, refusal codes) are printed under a plain-English sentence, never instead of one.
- Technical detail (step numbers, fingerprint prefixes, table names) goes to the log line beside each operator string.
- Exact strings are in the [Operator strings](#operator-strings) table; steps cite them by ID.

## Exit codes

| Program | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `scripts/migrate.py` | ok / nothing to do | a step failed after the backup was taken | validation, lint, or drift refusal; nothing mutated | blocked: no seam, no TTY for operator input, backup failed or disk short, DB unreachable |
| `deploy/update.sh` | up to date, updated, or cancelled at the prompt | failure (rolled back, or parked) | — | needs a terminal / blocked at the consent gate; nothing live touched |
| `mira update` (`ui/tui/update.py`) | forwards `update.sh` unchanged | | | |

## Terms

| Term | Meaning |
|---|---|
| step | a file in `deploy/migrations/`: versioned `V####__slug.sql\|.py` (runs once, ascending) or repeatable `R__slug.sql\|.py` (re-runs when its bytes change) |
| ledger | table `schema_migrations`, one row per applied step |
| head | the highest versioned step number in the tree |
| origin | a supported pre-mechanism release whose schema adopts at step `0000`: `v2026.10.03-2.0`, `v2026.10.06-2.0` (closed set) |
| `BASELINE` | generated file of fingerprints: one per origin (step `0000`) and one per step `0001..head` |
| `HEAD_FINGERPRINT.json` | generated full structure at head; what drift reports diff against |
| seam | the tiered superuser invocation (Phase 3 step 9) |
| `STAMP` | `date -u +%Y%m%dT%H%M%SZ`, one per update run, shared by venv asides, backup dir, snapshot ref |

## Files

| Path | Status | Phase |
|---|---|---|
| `deploy/GITHUB_REPO` | new | 0 |
| `deploy/lib/repo_identity.sh` | new | 0 |
| `deploy/lib/provenance.sh` | new | 0 |
| `deploy/lib/venv_stamp.sh` | new | 0 |
| `deploy/lib/state_preserve.sh` | new | 0 |
| `deploy/lib/custom_export.sh` | new | 0 |
| `deploy/update.sh` | edit | 0, 4 |
| `deploy/python.sh` | edit | 0 |
| `deploy/deploy.sh`, `install.sh`, `README.md`, `deploy/docker/Dockerfile` | edit | 0, 3, 5 |
| `utils/release_identity.py` | edit | 0 |
| `ui/tui/update.py`, `ui/tui/__main__.py` | edit | 0, 4, 5 |
| `ui/tui/migrate.py` | new | 4 |
| `deploy/RELEASE.md` | edit | 0, 5 |
| `deploy/docker/scripts/init-mira.sh` | edit | 0, 1, 3, 4 |
| `deploy/lib/services.sh`, `deploy/dependencies.sh` | edit | 1, 4 |
| `deploy/bootstrap_adopt.sql` | new | 1 |
| `deploy/mira_service_schema.sql` | edit | 1 |
| `deploy/postgresql.sh`, `deploy/deploy_database.sh` | edit | 1, 4 |
| `tests/fixtures/scratch_db.py` | edit | 1 |
| `clients/vault_client.py` | edit | 1 |
| `utils/power_on_self_test.py` | edit | 1, 4 |
| `utils/schema_migrations.py` | new | 2, 3, 5, 6 |
| `scripts/migrate.py` | new | 2, 3, 6 |
| `clients/valkey_client.py`, `main.py` | edit | 3 |
| `deploy/migrations/` (`AGENTS.md`, `BASELINE`, steps) | new | 1, 5, 6 |
| `deploy/finalize.sh` | edit | 6 |
| `utils/userdata_migrations.py` | new | 7 |
| `utils/userdata_manager.py` | edit | 7 |

---

## Phase 0 — release artifact, provenance, venv stamps, custom-build preservation

**Read first:** `deploy/update.sh` (whole file), `deploy/python.sh` (tarball fetch, `--local` branch, git init, exclude file, stash block, clear, `INSTALL_COMMIT_MSG`, Step 5 / 5b venvs), `ui/tui/update.py` (`_download_release`, `_extract_release`, `_gate_and_hand_off`, `_run_nightly_update`, `run_update`), `ui/tui/__main__.py` (pre-argparse dispatch), `utils/release_identity.py`, `deploy/RELEASE.md`, `deploy/deploy.sh` (clone block), `install.sh` (`REPO=`), `tools/implementations/bash_tool.py` (`_load_app_tree`, `_classify_app_tree_path`), `deploy/docker/scripts/init-mira.sh` (`main`).

### Steps

1. **Repo name constant.**
   - `deploy/GITHUB_REPO`: one line, `taylorsatula/mira`.
   - `deploy/lib/repo_identity.sh:mira_github_repo` prints the stripped file contents; the root is resolved from `${BASH_SOURCE[0]}`. If the file is missing or empty: `return 1` naming the path. No fallback literal.
   - `utils/release_identity.py`: `GITHUB_REPO` is read from `deploy/GITHUB_REPO` at import. If missing or empty, raise `OSError` naming the path.
   - Replace each literal with the constant: `deploy/deploy.sh` clone URL, `deploy/docker/Dockerfile` `image.source` label, the `README.md` doc link, `deploy/python.sh` fetch URL.
   - `install.sh` `REPO=` stays a literal (it bootstraps the fetch). Its equality with the constant is a release-gate check.
   - Source `repo_identity.sh` in `deploy/deploy.sh` and `deploy/update.sh` lib blocks, after `output.sh`.
2. **Release artifact** — `deploy/lib/provenance.sh:build_release_artifact <tag> <dest-dir>`. Every step is fatal:
   1. Refuse if `git ls-files --error-unmatch RELEASE_COMMIT` succeeds.
   2. `git archive --format=tar.gz --prefix="mira-${tag#v}/" <tag>`, then extract to a scratch dir.
   3. Write `git rev-parse "<tag>^{commit}"` (40 hex) as the single line of `<root>/RELEASE_COMMIT`.
   4. Run `python3 -m scripts.migrate lint`. Nonzero means no asset (wired once Phase 2 exists).
   5. Run `verify_release_identity`. `ok=False` means no asset, exit nonzero naming the check and both observed values.
   6. Re-tar to `<dest-dir>/mira-${tag#v}.tar.gz` with a single top-level directory.
3. **`utils/release_identity.py:verify_release_identity(repo_root, tag, artifact_root) -> IdentityReport`** — a frozen model with `ok` and three checks, each carrying its two observed values:
   - `tag_matches_version`: `tag == "v" + VERSION`.
   - `artifact_matches_repo`: the artifact's `VERSION` equals the repo's.
   - `release_commit_matches_tag`.
   An unreadable input fails its check, naming the path. The module stays stdlib-only.
4. **Fetch sites switch to the release asset** (`/releases/download/<tag>/mira-<tag-without-v>.tar.gz`):
   - `utils/release_identity.py`: delete `TAG_TARBALL_URL` and add `RELEASE_ASSET_URL`. `BRANCH_TARBALL_URL` is unchanged (nightly).
   - `ui/tui/update.py:_download_release`: format `RELEASE_ASSET_URL`.
   - `deploy/python.sh` tarball branch: use the asset URL from `$(mira_github_repo)`. After the copy, assert `/opt/mira/app/RELEASE_COMMIT` is 40 hex.
   - `deploy/RELEASE.md` manual-recovery `wget`: the asset URL.
   - A 404 is a hard failure naming the tag and the asset. No `archive/refs/tags/` fallback anywhere.
5. **Provenance functions** (`deploy/lib/provenance.sh`):
   - `read_release_commit <tree>` prints 40 hex. Absent or malformed: return 1, state `pre-provenance tree`.
   - `resolve_release_sha <tag>` runs `timeout 30 git ls-remote https://github.com/$(mira_github_repo).git "refs/tags/<tag>^{}"` and prints the first field. Empty: return 1. Used only when `read_release_commit` returns 1, which happens only for a tree fetched by a pre-mechanism `ui/tui/update.py` (auto-archive).
   - `resolve_local_sha <tree>` prints `rev-parse HEAD`, plus ` local-dirty` when `status --porcelain` is nonempty.
   - `official_commit_message <kind> <label> <sha>` prints `"<kind> <label> @ <sha>"`, `kind` ∈ {`install`, `update`}. Nightly label: `nightly-<12hex>`, sha from `NIGHTLY_SHA`.
6. **Install commit** (`deploy/python.sh`): `INSTALL_SHA` comes from, in order: `resolve_local_sha` (`--local`, suffix ` (local tree)`); then `read_release_commit "$SRC_DIR"`; then `resolve_release_sha`. All three failing is `exit 1`.
7. **Update provenance** (`deploy/update.sh`):
   - Phase 1: if the install tree is not a git repo, `print_error` + `exit 1` (delete the proceed-without-record branch).
   - Then record `PRE_UPDATE_SHA="$(git -C "$APP_DIR" rev-parse HEAD)"`.
   - Phase 6 commit: `official_commit_message update "$TAG" "$NEW_SHA"`. `NEW_SHA` comes from `read_release_commit "$TREE"`, else `resolve_release_sha "$TAG"`, else `NIGHTLY_SHA` when `NIGHTLY=1`.
   - The commit keeps `--allow-empty` and stays non-fatal.
8. **Venv stamps** — `deploy/lib/venv_stamp.sh`:
   - `write_venv_stamp <venv> <reqfile> <base-python> <extras-csv>` writes `<venv>/.mira-build-stamp`, mode 0644. Keys: `base_python`, `base_python_version`, `requirements_sha256`, `requirements_path`, `extras` (sorted CSV of `torch`, `sentence-transformers`, `playwright`), `built_at`.
   - `venv_stamp_matches …` returns 0 on a match of `base_python_version` + `requirements_sha256` + `extras`, 1 on mismatch or when `<venv>/bin/python3 -c 'import sys'` fails, and 2 when the stamp is absent.
9. **Stamp call sites:**
   - `deploy/python.sh` Step 5: write the server-venv stamp after `pip install -r requirements.txt`. The extras come from `CONFIG_EMBEDDING_PROVIDER=local` and `CONFIG_INSTALL_PLAYWRIGHT=yes`.
   - `deploy/python.sh` Step 5b: write the TUI-venv stamp (`ui/tui/requirements.txt`, empty extras).
   - `deploy/update.sh:build_venv`: write the stamp as its last statement.
10. **Conditional venv build** (`deploy/update.sh` phase 3):
    - Parse `--rebuild-venvs` from `$3..`; `ui/tui/update.py:_gate_and_hand_off` passes it through.
    - Server extras come from the existing `HAD_TORCH`/`HAD_ST`/`HAD_PLAYWRIGHT` detection.
    - Per venv, stamp rc 0 sets `SKIP_*_VENV=yes`. Otherwise build `venv.new` / `tui-venv.new`.
    - Print `<label> venv: reusing (build inputs unchanged)` or `<label> venv: rebuilding (<reason>)`. Reason ∈ {`stamp absent`, `requirements changed`, `extras changed`, `base python changed`, `interpreter dead`, `--rebuild-venvs`}.
11. **Swap guarded** (phase 5):
    - Move a venv aside (`*.pre-update-<STAMP>`) and swap it only when it was rebuilt.
    - Assemble `ROLLBACK_VENV_CMDS` from the swaps that actually happened.
    - Refresh the TOAST shim (`deploy/_mira_log_levels.py`, `mira-log-levels.pth`) into the live venv's site-packages unconditionally.
    - On success, delete only the asides that exist.
12. **Shared code clear** — `deploy/lib/state_preserve.sh`:
    - The preserve set is one array in this file: `venv .env data logs scratch .git run.sh venv* tui-venv*`. It must contain every slash-anchored entry of the `.git/info/exclude` heredoc in `deploy/python.sh` — those are the state names `bash_tool` refuses to delete, so the clear must not delete them either.
    - `clear_code_preserving_state <app-dir> [--sudo] [--print]` runs `find <app-dir> -mindepth 1 -maxdepth 1 ! -name … -exec rm -rf {} +`. `--sudo` prefixes `sudo`. `--print` writes the argv NUL-separated and executes nothing.
    - `list_cleared_files <app-dir>` prints, NUL-separated and sorted, every regular file and symlink the clear would delete (everything under the non-preserved top-level names), git-ignored files included. It omits only regenerable output: `__pycache__/` directories and `*.pyc` files. It is the single definition of "what the clear destroys", consumed by the export and the snapshot.
    - Replace the inline clears in `deploy/update.sh` phase 6 and `deploy/python.sh` reinstall (the latter passes `--sudo`).
13. **Custom-build export** — `deploy/lib/custom_export.sh`:
    - `official_baseline_commit <tree>`: the newest commit matching `^(install|update) `. None: return 1.
    - `detect_custom_delta <tree> <baseline>`: NUL-safe `<status>\t<path>` lines, built from `list_cleared_files` and the baseline tree (`git ls-tree -r <baseline>`), never from git's tracked/untracked view — `.gitignore` must not hide a file from the save:
      - `A`: a listed file absent from the baseline (untracked and git-ignored files alike);
      - `M`: a listed file whose blob (`git hash-object`) or mode differs from the baseline's;
      - `D`: a baseline path under a non-preserved top-level name that no longer exists.
    - `export_custom_build <tree> <baseline> <tag> <dest-root>` writes `<dest-root>/mira-custom-<STAMP>/`. `dest-root` is `$HOME/Desktop` if writable, else `$HOME`. Contents:
      - full copies of every `A`/`M` file, relative paths and modes preserved; symlinks copied as links (`cp -P`);
      - `custom.patch`: with a throwaway `GIT_INDEX_FILE` seeded by `git read-tree <baseline>`, `git add -f` every `A`/`M` path and `git rm --cached` every `D` path, then `git diff --cached --binary <baseline>` — ignored and binary files included;
      - `RESTORE_MY_CUSTOM_BUILD.txt`, opening with a plain-English lead, then: per-file status; sha256 per exported file; the deletion list with the `rm` that re-deletes each; `git log --format='%H %s' <baseline>..HEAD`; the baseline tag and SHA; restore instructions written for a coding agent (re-apply onto the new release; no built-in restore command).
    - Space precheck before writing: the total byte size of the `A`/`M` files must fit in the free space of `dest-root`'s volume and, separately, of the install's `.git` volume (the snapshot stores the same bytes). Short ⇒ S-PRESERVE-SPACE, return nonzero.
    - Verification after writing: every copy's sha256 equals its source's, and the copy count equals the `A`/`M` count. Any mismatch ⇒ return nonzero.
    - Every failure returns nonzero; the caller stops the update before the clear (step 15).
    - `print_custom_alert <export-dir> <count>` prints S-CUSTOM, only when the delta is nonempty.
    - `record_venv_drift <live-venv> <reqfile> <catalog>` runs only for a venv that will be swapped, before the swap. It appends `## Venv package drift`: packages present but unnamed (with frozen versions) and changed pins. An empty list still writes the section, saying so.
14. **Snapshot ref** — `deploy/lib/custom_export.sh:snapshot_working_tree <tree> <stamp>`:
    - Use a throwaway `GIT_INDEX_FILE`, removed on every exit path.
    - Seed it with `git read-tree HEAD`, then `git add -A` (tracked and untracked changes, deletions), then `git add -f` every path from `list_cleared_files` (git-ignored files).
    - Run `git write-tree`, `git commit-tree -p HEAD -m "pre-update snapshot <stamp>"`, then `git update-ref refs/mira/pre-update/<stamp> <commit>`.
    - Verify: `git ls-tree -r <ref>` contains every path from `list_cleared_files` with the file's current blob. Any missing path ⇒ delete the ref, return nonzero.
    - Every step is fatal. It leaves HEAD, the real index, and the worktree untouched.
15. **Save, then clear** (`deploy/update.sh`):
    - **Phase 4.5 — save** (service stopped, so nothing edits the tree; venvs not yet swapped):
      1. `export_custom_build` + `print_custom_alert` when the delta is nonempty.
      2. `record_venv_drift` for each venv that phase 5 will swap.
      3. `snapshot_working_tree`.
      4. Any nonzero return ⇒ delete the partial export dir and the ref, delete `venv.new`/`tui-venv.new`, restart the old service, print S-PRESERVE-FAIL, `exit 1`. The install tree, the venvs, and the database are untouched.
    - **Phase 6 — code swap:** `clear_code_preserving_state`, overlay, `git add -A`, commit. The clear runs only after phase 4.5 returned 0.
    - Delete both `git stash push --include-untracked` blocks (`update.sh`, `python.sh`).
    - The phase-8 success path runs `git update-ref -d refs/mira/pre-update/<STAMP>`. The Desktop export is never deleted by MIRA.
    - `deploy/python.sh` reinstall over an existing tree runs the same export before its clear; failure aborts the reinstall before the clear.
16. **Container image stamp** — `init-mira.sh:main` writes `/opt/mira/app/.mira-image-stamp` (`image_id`, `built_at`, `version`, `release_tag`). `deploy/docker/Dockerfile` gains `LABEL org.opencontainers.image.revision`.
17. **`mira status`, git half** — `ui/tui/update.py:run_status() -> int`, stdlib only. Reports: the last official commit + SHA; `git diff --stat` + untracked files against it; `git log --oneline <baseline>..HEAD`; `git stash list`. It prints no database line. Dispatch `status` in `ui/tui/__main__.py`, shaped like `update`.
18. **`deploy/RELEASE.md`:**
    - New "Release artifact" section: `build_release_artifact`, the asset name, and that the asset is the only payload.
    - "Provenance" section: the `@ <sha>` commits, `--local` dirty marking, and `resolve_release_sha` as the path for a tree fetched by a pre-mechanism updater.
19. **Maps:**
    - `deploy/lib/AGENTS.md`: cards for the five new libs. Rules: stamp = build inputs, not live state, with the `--rebuild-venvs` escape hatch; `RELEASE_COMMIT` is injected at build time and never committed; `deploy/GITHUB_REPO` is the one repo spelling (`install.sh` is the named exception).
    - `deploy/AGENTS.md`: `update.sh`, `python.sh`, `RELEASE.md` cards.
    - `utils/AGENTS.md`: the `release_identity.py` card.
    - `ui/tui/AGENTS.md`: `update.py` and `__main__.py` (`status`, asset).
    - `deploy/docker/scripts/AGENTS.md`: `init-mira.sh` (image stamp).

### Probes

| Probe | Asserts |
|---|---|
| `probe_repo_name_single.sh` | `mira_github_repo` equals the file and returns 1 when it is removed; Python `GITHUB_REPO` equals it; each URL constant contains it once; `grep -rn "mira-OSS"` over `deploy/ utils/ ui/ install.sh README.md docs/` is empty; `install.sh` `REPO=` equals the constant; the origin remote's redirect target equals the constant |
| `probe_release_artifact.sh` | A real annotated tag in a `/tmp` clone produces the asset; it has exactly one top-level dir; `RELEASE_COMMIT` equals `rev-parse <tag>^{commit}`; every other file is byte-identical to `git archive`; the builder refuses when `RELEASE_COMMIT` is committed; each `verify_release_identity` check flips under its own negative control; a missing asset is reported, never re-fetched from `archive/` |
| `probe_prov_commit.sh` | The four exact message shapes; `read_release_commit` handles present / absent / empty / 39-hex; a tree with `RELEASE_COMMIT` spawns no `git ls-remote` (witness wrapper fails the probe if invoked); `resolve_release_sha` resolves a real tag and returns 1 for a missing one; BLOCKED when offline |
| `probe_venv_stamp.sh` | Throwaway venvs from `ui/tui/requirements.txt`: rc 0 when identical; rc 1 after a requirements change, an extras change, or `chmod -x python3`; rc 2 when the stamp is removed; a matching stamp creates no `venv.new` |
| `probe_state_preserve.sh` | Every slash-anchored entry of `deploy/python.sh`'s exclude heredoc is in the preserve array (parsed from both files); `list_cleared_files` on a `/tmp` root lists ignored files (`scratch2/x`, `a.bak`) and omits `__pycache__/`, `*.pyc`, and every preserved name; two `/tmp` roots end with exactly the preserve set plus asides; `--print` argv has `sudo` as argv0 only with `--sudo`; no inline `find … -maxdepth 1` remains in `update.sh`/`python.sh`; the preserve names appear only in `state_preserve.sh` |
| `probe_custom_export.sh` | A `/tmp` install-shaped tree carrying the real `.gitignore` and exclude file. Delta cases: untracked, modified, deleted, committed-then-modified, a mode change, a symlink, a binary file, and **git-ignored** files the clear would delete (`junk_drawer/tool.py`, `notes.bak`, `tools/implementations/crm_jobs_tool.py`, `training_data/x.jsonl`). Every case is in the catalog and copied byte-identical with its mode; `custom.patch` applied with `git apply --binary` to a fresh baseline checkout reproduces every file; `__pycache__/` and `*.pyc` are absent; the custom commit is listed; an empty delta produces no dir and no banner; an unwritable `dest-root` and an undersized volume (quota-limited `/tmp` mount, or BLOCKED) each return nonzero with no partial dir left |
| `probe_venv_drift.sh` | Rebuild arm names the unnamed package and the pin change; skip arm leaves the catalog byte-identical; an exact match still writes an empty section |
| `probe_snapshot_rollback.sh` | Modified, untracked, deleted, mode-change, and **git-ignored** cases (`junk_drawer/tool.py`, `notes.bak`) plus a preserved `data/` file: the snapshot ref contains every `list_cleared_files` path; the snapshot leaves HEAD, index, and worktree unchanged; after overlay + rollback (Phase 4 step 5 sequence) every file sha256 and mode matches, HEAD equals `PRE_UPDATE_SHA`, untracked is untracked again, `data/` is untouched, and a staged change returns unstaged; read-only `.git/objects` makes the snapshot exit nonzero with no ref |
| `probe_update_sh_scratch.sh` | `MIRA_APP_DIR` override, non-canonical root: a non-repo root is refused before any `venv.new`; matching stamps produce reuse lines, no asides, and no `mv` in the rollback commands; a changed `requirements.txt` rebuilds only that venv; an export that fails at phase 4.5 (unwritable `HOME`) exits 1 with S-PRESERVE-FAIL, every file in the tree unchanged (sha256 compare), no `venv.new`, no ref |
| `probe_prov_status.sh` | Names the baseline subject + SHA, the uncommitted and untracked changes, the custom commit, and the stash; no database line; imports with neither `psycopg` nor `hvac` installed |
| `probe_bashtool_no_stash_dependency.py` | `_load_app_tree` and `_classify_app_tree_path` give identical results on a dirty and a clean tree |
| `probe_no_new_deps.py` | Both requirements files unchanged; new imports are stdlib / psycopg / hvac / pydantic / in-repo; `utils/release_identity.py` stays stdlib-only |

### Done when

- [ ] All Phase-0 probes `PROBE PASSED`, or UNVERIFIED with the covering probe named.
- [ ] Each of these greps returns nothing:
  - `grep -rn "stash push" deploy/`
  - `grep -n "not a git repo" deploy/update.sh`
  - `grep -rn "archive/refs/tags\|TAG_TARBALL_URL" deploy/ utils/ ui/ install.sh README.md docs/`
  - `grep -rn "find .*-maxdepth 1" deploy/update.sh deploy/python.sh`
- [ ] `bash -n` passes on every edited or new `.sh` file.
- [ ] The map audits are clean.

---

## Phase 1 — PostgreSQL floor, ledger, ownership bootstrap, secrets manifest

**Read first:** `deploy/mira_service_schema.sql` (header, `$schema_precondition$`, extensions, grants block), `deploy/postgresql.sh` (role SQL, `createdb -O mira_admin`, schema apply, Step 14), `deploy/deploy_database.sh`, `deploy/dependencies.sh` (branch chain end), `deploy/lib/services.sh`, `tests/fixtures/scratch_db.py`, `clients/vault_client.py` (`get_database_url`, `preload_secrets`), `utils/power_on_self_test.py` (`run_pre_server_post`, `_check_vault`, `_check_postgres_rls`), `deploy/docker/scripts/init-mira.sh` (`init_postgresql`, `init_vault`), `deploy/lib/vault.sh` (policy).

### Steps

1. **PostgreSQL floor** (`deploy/lib/services.sh`):
   - `pg_major_installed` prints the installed server major, with no other output, resolved per branch:
     - Debian: the `pg_lsclusters -h` row on port 5432, else the newest.
     - Fedora/RHEL: `/usr/pgsql-17/bin/postgres`, else the first `/usr/pgsql-*`.
     - Arch: `postgres --version`.
     - macOS: `$(brew --prefix postgresql@17)/bin/postgres`.
     - Unresolvable: return 1.
   - `require_pg_floor <floor>`: below the floor prints S-PGFLOOR + `exit 1`; unresolvable prints S-PGNONE + `exit 1`.
   - Call `require_pg_floor 17` in `deploy/dependencies.sh` after the branch chain's `fi`, before `print_success "System dependencies installed"`. The comment names why 17: it is the one major the container image and the macOS keg ship.
2. **Ledger table** — append to the grants block of `deploy/mira_service_schema.sql`, between `-- BEGIN GENERATED LEDGER DDL` / `-- END GENERATED LEDGER DDL` sentinels:
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
   - The ledger has no RLS. It is not added to `_check_postgres_rls`'s expected tables.
   - The header comment gains one sentence: `deploy/migrations/` carries an installed database forward; `schema_migrations` is its ledger.
3. **`deploy/bootstrap_adopt.sql`** — runs only through the seam (`-v ON_ERROR_STOP=1`). Every statement is idempotent:
   - A `DO` block that transfers ownership to `mira_admin` of every table, index, sequence, and function in `public`, generated from `pg_class`/`pg_proc` via `format()` + `EXECUTE`.
   - `GRANT CREATE ON SCHEMA public TO mira_admin;`
   - The ledger DDL as `CREATE TABLE IF NOT EXISTS` + its `GRANT`, inside the same generated sentinels.
   - Nothing is stamped.
4. **Installer calls** — run `bootstrap_adopt.sql` immediately after the schema apply in `deploy/postgresql.sh`, `deploy/deploy_database.sh`, and `init-mira.sh:init_postgresql` (`sudo -u postgres psql`).
5. **`tests/fixtures/scratch_db.py`** — `CREATE DATABASE … OWNER mira_admin`; apply `bootstrap_adopt.sql` after `_apply_schema` through the same `psql` seam. `_ROLES` gains no `CREATEDB`.
6. **`deploy/migrations/AGENTS.md`** (new):
   - Section order per the map skill.
   - Rules: step numbers are their own dense 4-digit counter; shipped steps are immutable; `BASELINE` is generated; discovery skips `BASELINE`, `HEAD_FINGERPRINT.json`, `AGENTS.md`, `__pycache__`; adoption drift is refused (exit 2) and only `mira migrate --adopt-drifted` passes it, recorded in the ledger.
   - Root `AGENTS.md` registry row: `| deploy/migrations/ | Versioned and repeatable database steps, the generated BASELINE fingerprint table, and the step authoring contract |`.
7. **Secrets manifest** (`clients/vault_client.py`):
   - `REQUIRED_SECRET_MANIFEST`:
     - `mira/database`: `admin_url`, `service_url`, `username`, `password`.
     - `mira/services`: `valkey_url`, `app_url`, `userdata_encryption_key`.
     - `mira/api_keys`: `provider_key`.
   - Provider-conditional fields are excluded.
   - `check_required_secrets() -> list[SecretGap]`; `format_secret_gap(gap) -> str` returns `vault kv patch secret/mira/<path> <field>=<value>`.
8. **POST check** — `CheckSpec("vault_secrets_manifest", True, _check_vault_secrets_manifest)` after `"vault"` in `run_pre_server_post`. It raises `RuntimeError` carrying every gap's command and returns `{"paths_checked", "fields_checked"}`.
9. **Maps:**
   - `deploy/AGENTS.md`: floor 17, mechanically enforced by `require_pg_floor`; `bootstrap_adopt.sql` card and its callers; the schema-file Rule permits only the ledger table + its grant as non-baseline DDL.
   - `deploy/lib/AGENTS.md`: `services.sh` floor functions.
   - `clients/AGENTS.md`: the manifest.
   - `utils/AGENTS.md`: the `vault_secrets_manifest` check.
   - `tests/fixtures/AGENTS.md`: owner + bootstrap.
   - `deploy/docker/scripts/AGENTS.md`: the bootstrap call.

### Probes

| Probe | Asserts |
|---|---|
| `probe_pg_floor.sh` | `pg_major_installed` matches the host's independently derived major; `require_pg_floor 99` exits 1 naming both numbers + the command; a stripped `PATH` gives S-PGNONE; the Debian port-5432 arm is UNVERIFIED on other hosts |
| `probe_bootstrap_adopt.py` | Run twice, exits 0 both times; owners of tables, indexes, sequences, and functions are exactly `{mira_admin}`; `mira_admin` has CREATE on `public`; ledger shape and both CHECKs; `mira_dbuser` has SELECT; fingerprint excluding the ledger is unchanged |
| `probe_ledger_grants.py` | As `mira_dbuser` over TCP: SELECT ok, INSERT raises `InsufficientPrivilege` |
| `probe_secrets_manifest.py` | Live Vault returns `[]`; `format_secret_gap` shape is right; the manifest field sets equal the fields written by `postgresql.sh` Step 14 and `init_vault` (grep-derived); no secret printed. UNVERIFIED without AppRole |
| `probe_post_manifest_check.py` | `--json --report-marker` shows `vault_secrets_manifest` required + passed; a nonexistent field raises with the literal `vault kv patch secret/mira/database nonexistent_field=` |

### Done when

- [ ] Probes pass, or UNVERIFIED naming the AppRole blocker.
- [ ] The schema file still refuses a non-empty database.
- [ ] `scratch_database()` yields uniformly `mira_admin`-owned objects.
- [ ] The map audits are clean.

---

## Phase 2 — runner read paths

**Read first:** `scripts/AGENTS.md`, an existing `scripts/*.py` shim, `utils/power_on_self_test.py` (`--report-marker` convention), `clients/llm/types.py` (`DialectName`, `DIALECT_NAMES`), `deploy/mira_service_schema.sql` (`embedding_config`, `model_configs`, `vector(` sites).

### Steps

1. **Discovery** — `utils/schema_migrations.py:discover_steps(migrations_dir) -> list[Step]`:
   - `MIGRATIONS_DIR` is the default; every entry function accepts `migrations_dir`.
   - `Step` is a frozen dataclass: `revision` (4 digits, `""` for repeatables), `slug`, `kind`, `store`, `path`, `checksum` (sha256 of the bytes), `requires_operator_input`, `dialect_twins`, `notice`.
   - Grammar: `V(\d{4})__([a-z0-9_]+)\.(sql|py)` and `R__([a-z0-9_]+)\.(sql|py)`. A file starting with `V`/`R` that fails the grammar raises `ValueError`. Other files are skipped.
   - Header declarations:

     | Field | `.sql` | `.py` |
     |---|---|---|
     | store (required) | `-- store: postgres` | `STORE = "postgres"` |
     | notice (optional) | `-- notice: <text>`, continuation lines `-- ` joined with one space | `NOTICE = "<text>"` |
     | operator input | `-- REQUIRES_OPERATOR_INPUT: <prompt>` | `REQUIRES_OPERATOR_INPUT = "<prompt>"` |
     | dialect twins | `-- dialect-twins: DialectName, DIALECT_NAMES` | same, as a tuple |
     | contract marker | `-- expand-contract: contract-for V<N>` | same |
   - A missing store raises `ValueError`. Nothing is inferred from body content.
2. **Ordering** — `order_steps(steps) -> OrderedChain`: integer sort; raises `StepOrderError(ValueError)` on a duplicate revision, a gap (dense from `0001`), or a duplicate repeatable slug.
3. **BASELINE read** — `read_baseline(migrations_dir) -> Baseline`:
   - `origins: tuple[Origin(tag, fingerprint_sha256), ...]` and `rows: tuple[BaselineRow(revision, fingerprint_sha256), ...]` (`0001..head`).
   - `match(fp) -> str | None`: `"0000"` for any origin, the revision for a row, else `None`.
   - File grammar: lines `origin=<tag> fingerprint_sha256=<hex>`, then `revision=<NNNN> fingerprint_sha256=<hex>`.
   - Raises `ValueError` naming the line for: a malformed line, non-dense rows, or a last row that is not the chain head.
4. **Ledger read** — `read_ledger(conninfo) -> list[LedgerRow]` (TypedDict of the 8 columns); `ledger_head(rows)`; `repeatable_due(rows, steps)` (checksum differs or no row).
5. **Fingerprint** — `structural_fingerprint(conninfo, *, include_ledger: bool) -> dict`, read from `pg_catalog`/`information_schema` in fixed key order:
   - Coverage: tables + columns (name, type, nullability, default), `pg_get_indexdef`, `pg_get_constraintdef`, policies (`pg_get_expr(polqual, polrelid)`), `pg_get_triggerdef`, function identity args + return type + `prosecdef`, table grants + schema privileges, extensions.
   - Normalized: OIDs excluded; `vector(N)` becomes `vector(<DIM>)`; `relpersistence` excluded.
   - `fingerprint_sha256(fp)` = sha256 of `json.dumps(fp, sort_keys=True, separators=(",", ":"))`.
6. **Validation** — `validate(chain, ledger, baseline) -> Validation` with `refusals: list[Refusal(code, message)]`. Each refusal logs its technical detail.

   | Code | Condition | Message |
   |---|---|---|
   | `DOWNGRADE` | a ledger revision above tree head | S-DOWNGRADE |
   | `REVERTED_MIGRATION` | a ledger revision ≤ tree head with no tree file | S-REVERTED |
   | `CHECKSUM_MISMATCH` | an applied versioned step's checksum ≠ the file's | S-CHECKSUM |
   | `OUT_OF_ORDER` | a pending versioned step below the ledger head | S-OUTOFORDER |
   | `DUPLICATE_REVISION` / `GAP` | from `order_steps` | S-DUPLICATE / S-GAP |
   | `LINT` | any `lint_steps` finding | S-LINT |
   | `ADOPTION_DRIFT` | raised by `bootstrap_check` (Phase 3) | S-DRIFT |

   - `pending(chain, ledger) -> list[Step]`: versioned steps above head + due repeatables.
   - `render_drift_report(observed_fp, expected_fp)`: lead line S-DRIFT-LEAD, then added / removed / changed objects. `expected_fp` is `read_head_fingerprint(migrations_dir)` — the full head structure from `deploy/migrations/HEAD_FINGERPRINT.json` (`BASELINE` holds only hashes, which cannot name a difference).
   - `read_head_fingerprint(migrations_dir) -> dict`: its `fingerprint_sha256` must equal `BASELINE`'s head row, else `ValueError`.
7. **Lint** — `lint_steps(chain) -> list[LintFinding(file, rule)]`:
   - more than one `BEGIN`/`COMMIT`/`ROLLBACK` in a `.sql` step;
   - a body reading `VERSION`, importing `utils.release_identity`, or branching on `time.time()`/`datetime.now()`/`NOW()`/`CURRENT_DATE`;
   - any reference to `embedding_config`;
   - `UPDATE model_configs SET` touching `model`/`dialect_name`/`endpoint_url`/`api_key_name`/`effort`/`max_tokens`;
   - a literal `vector(<digits>)`;
   - a `dialect_name` CHECK change without `-- dialect-twins: DialectName, DIALECT_NAMES`;
   - `DROP TABLE`/`DROP COLUMN`/`ALTER … RENAME` without `-- expand-contract: contract-for V<N>`.
8. **Status** — `status(conninfo, migrations_dir) -> StatusReport` (Pydantic): `ledger_head`, `tree_head`, `pending`, `repeatable_due`, `refusals`, `fingerprint_drift` (when the ledger is at head: the live structure, ledger excluded, ≠ `HEAD_FINGERPRINT.json`), `adopted_with_drift_override` (derived on read: any `applied_by` ending `+adopt-drifted`).
9. **Plan** — `plan(conninfo, migrations_dir) -> PlanReport` (Pydantic), read-only:
   - Fields: `adoption_needed` (ledger absent, or present and empty, with tables present); `pending`; `repeatable_due`; `operator_input` (prompts); `notices` (number → NOTICE or the default); `refusals`; `needs_consent` (any of adoption / pending / repeatable / operator input).
   - It calls `validate` and the read-only branch of `bootstrap_check`.
10. **Plan text** — `render_plan(report, elevation) -> str`, in order:
    1. one line per pending versioned step (NOTICE or S-STEP-DEFAULT);
    2. one line per due repeatable;
    3. S-PLAN-FIRST when `adoption_needed`;
    4. S-ELEV-* by `elevation` when `needs_consent`;
    5. each operator-input prompt verbatim;
    6. S-PROTECT.

    `needs_consent=False` renders S-NOTHING only.
11. **CLI** — `scripts/migrate.py`:

    | Subcommand | Flags | Phase-2 behavior |
    |---|---|---|
    | `migrate` | `--dry-run`, `--plan`, `--report-marker`, `--elevation {passwordless,interactive,none}`, `--stamp <STAMP>`, `--adopt-drifted`, `--migrations-dir`, `--backups-dir` | `--dry-run` and `--plan` implemented; otherwise exits 1 `not implemented` |
    | `restore <STAMP>` | `--backups-dir` | exits 1 `not implemented` |
    | `status` | `--json`, `--migrations-dir` | implemented |
    | `lint` | `--migrations-dir` | implemented; opens no connection and reads no Vault |
    | `reconcile` | `--env-file` | Phase 6 |

    - `--plan --report-marker` wraps JSON (`needs_consent`, `adoption_needed`, `pending_notices`, `repeatable_due`, `operator_input`, `refusals`) between `MIRA_PLAN_BEGIN`/`MIRA_PLAN_END`.
    - A flag passed to a subcommand that does not accept it is an argparse error.
    - The shim contains no SQL and no `psycopg` import.
12. **Maps:**
    - `utils/AGENTS.md`: the `schema_migrations.py` card, including the required-`conninfo` rule and the operator/log language split.
    - `scripts/AGENTS.md`: the `migrate.py` card (subcommands, exit codes, `--plan` marker, Vault env precondition; `lint` has none).
    - `deploy/migrations/AGENTS.md`: the header-declaration table, lint rules, NOTICE authoring rule.

### Probes

| Probe | Asserts |
|---|---|
| `probe_migrate_discovery.py` | A `/tmp` step set: the five well-formed steps are found; `README.md` is skipped; `Vx__bad.sql` raises; a gap and a duplicate each raise `StepOrderError`; checksums are stable and change on a 1-byte edit; a missing store raises; both NOTICE spellings parse and join |
| `probe_migrate_validate.py` | Every code has positive and negative controls; `DOWNGRADE` vs `REVERTED_MIGRATION` precedence; nothing mutated (fingerprint + ledger identical); messages hold no banned word and contain a command; the code vocabulary equals the table; each lint rule flags its violation and passes its permitted form |
| `probe_fingerprint_determinism.py` | Four hashes over two DBs are equal; an added table changes the hash and the diff names it; dropping it restores the hash |
| `probe_migrate_status.py` | Empty / at-head / ahead ledgers give exit 0 / 0 / 2; `--json` parses as one object; no password substring in the output |
| `probe_migrate_plan.py` | Arms: ledger absent / one behind with NOTICE + operator-input steps / at head. JSON flags are correct; NOTICE text is verbatim; the three elevation sentences differ; at-head prints S-NOTHING with `needs_consent=false`; fingerprint + ledger unchanged in every arm; no banned words; no password |
| `probe_exit_codes.py` | 0 nothing to do; 2 refusal; 3 on a closed `PGHOST` port; `lint` gives 0 on the real dir and 2 on corrupted copies with no connection opened (closed port, `VAULT_ADDR` unset); `--adopt-drifted` is rejected by `lint` |

### Done when

- [ ] All Phase-2 probes `PROBE PASSED`; none used sudo or `mira_service`.
- [ ] The map audits are clean.

---

## Phase 3 — mutation: lock, adapters, adoption, apply, backup, restore

**Read first:** `clients/vault_client.py` (`VaultClient`), `clients/valkey_client.py` (`flush_except_whitelist`), `main.py` (lifespan Valkey flush + preserve prefixes), `deploy/finalize.sh` (`systemone.env` writer, unit + plist paths), `deploy/lib/services.sh` (`acquire_sudo` / `SUDO_ASKPASS`), `deploy/postgresql.sh` (macOS plain-`psql` branch), `deploy/vm/extract-stage-vm.sh` + `deploy/vm/inject-restore-vm.sh` (dump/restore precedent), `deploy/docker/Dockerfile` (`VOLUME`), `README.md` (`docker run` lines), `utils/release_identity.py:get_current_version`.

### Steps

1. **Advisory lock** — `migration_lock(conn)`: `pg_advisory_lock(MIGRATION_ADVISORY_KEY)`, a hardcoded int constant (registered in `deploy/migrations/AGENTS.md`). It is held from before `bootstrap_check` until the last ledger write.
2. **Step context** — `StepContext`, a frozen dataclass built per run:
   - `ctx.pg`: the run's connection. Steps never commit.
   - `ctx.vault`: `VaultClient().client` with `read(path)`, `patch(path, fields)` (read → compare → write), and `delete_fields(path, fields, confirmed_new=…)`, which refuses unless every `confirmed_new` field is present after the write.
   - `ctx.env`: `set_var` / `unset_var` on `env_file`, written atomically (tmp + `os.replace`), mode 0600.
   - `ctx.invalidate_valkey()`: `flush_except_whitelist(BOOT_FLUSH_PRESERVE_PREFIXES)`.
   - `ctx.operator_input(prompt) -> str`: reads stdin when `isatty()`, else raises `OperatorInputRequired`.
   - `ctx.embedding_dimensions() -> int`: `SELECT embedding_dimensions FROM embedding_config LIMIT 1`. Interpolate with `psycopg.sql` only.
3. **`BOOT_FLUSH_PRESERVE_PREFIXES`** — a module constant in `clients/valkey_client.py` holding the prefixes `main.py` passes today; `main.py` reads it.
4. **Step execution:**
   - `.sql`: `conn.execute(text)` inside the step transaction.
   - `.py`: loaded with `importlib.util.spec_from_file_location`; requires `up(ctx)`, optional `verify(ctx)` (run in the same transaction). A missing `up` raises `ValueError`.
5. **Adoption** — `bootstrap_check(conninfo, migrations_dir, adopt_drifted) -> Adoption`:

   | State | Action | Exit |
   |---|---|---|
   | ledger table absent, tables present | run `bootstrap_adopt.sql` via seam (step 9); no seam ⇒ S-NOSEAM + `ADOPT-PROBE-BLOCKED` on stderr | 3 if no seam |
   | ledger empty, fingerprint (ledger excluded) matches an origin | stamp nothing; run `0001..head` | 0 |
   | ledger empty, matches row `N` | stamp `0001..N` (`applied_by='baseline'`, file checksum, `duration_ms=0`); run `N+1..head` | 0 |
   | ledger empty, no match, no flag | stamp nothing; print `render_drift_report` then S-DRIFT | 2 |
   | ledger empty, no match, `--adopt-drifted` | run `0001..head`; every written row `applied_by="<version>+adopt-drifted"`; row `0001` `name` suffixed ` adopt-drifted=<12hex>` | 0 |
   | ledger has rows | normal path; a passed `--adopt-drifted` is ignored and stdout says so | — |
   | no ledger, no tables | `apply deploy/mira_service_schema.sql first` | 2 |

   - The flag is per-invocation and writes no state a later run reads. No config or env var disables the check.
6. **Apply loop** — `migrate(conninfo, migrations_dir, backups_dir, stamp, adopt_drifted) -> int`:
   ```
   with migration_lock(conn):
       bootstrap_check → validate (refusals ⇒ 2)
       if pending: take_backup(stamp)          # before the first mutation
       for step in versioned pending: with conn.transaction(): run_step; insert ledger row
       for step in repeatable_due:    with conn.transaction(): run_step; upsert ledger row
   ```
   - Non-postgres stores: idempotent work first, ledger row last.
   - A failure rolls back that step and exits 1 with the step number, slug, and raw error.
   - `OperatorInputRequired`: roll back, print the prompt + S-OPINPUT, exit 3.
   - `applied_by`: `get_current_version()`, or `nightly-<12hex>` read from `data/nightly_stamp`.
   - On success, apply retention.
   - **Structure check**, after every successful run (pending or not): the live fingerprint (ledger excluded) vs `read_head_fingerprint`. Equal ⇒ print nothing. Different ⇒ print S-POSTDRIFT, then `render_drift_report`; log both hashes. Exit stays 0 — this reports hand-made database changes; it never refuses or reverts them.
7. **Backup** — `take_backup(conninfo, backups_dir, stamp) -> BackupManifest`:
   - Destination `<backups_dir>/<stamp>/`; host default `/opt/mira/backups` (outside the code clear).
   - Disk precheck: `free >= 2 * pg_database_size`, else S-DISK, exit 3.
   - `pg_dump -Fc` as `mira_admin` over TCP (`PG*` from `conninfo`), with `timeout=` a module constant; a timeout fails.
   - `vault.json` (mode 0600) holds `mira/api_keys`, `mira/database`, `mira/services`; never printed or logged.
   - Copies of `systemone.env` and the supervisor file (`/etc/systemd/system/mira.service` or `~/Library/LaunchAgents/com.mira.app.plist`). An absent one is recorded in the manifest.
   - `manifest.json` is **written last**, after `pg_restore --list` passes: stamp, dump file, TABLE DATA count, Vault field counts, env/unit presence, `pg_database_size`, `VERSION`, ledger head, pre-dump fingerprint sha256.
   - Any failure: exit 3, nothing mutated, no manifest.
   - Retention, on success only: keep the newest 3 dirs and print each deleted path.
8. **Container backups persist:**
   - `deploy/docker/Dockerfile` `VOLUME` gains `/opt/mira/backups`. The comment states that a bare `VOLUME` is anonymous, so persistence comes from the named mount.
   - `README.md` `docker run` lines gain `-v mira-backups:/opt/mira/backups`; the persistence sentence names upgrade backups.
   - `init-mira.sh:main`: `mkdir -p` + `chown mira:mira /opt/mira/backups`.
9. **Superuser seam** — `run_as_superuser(argv, timeout) -> CompletedProcess`:
   - Linux with `SUDO_ASKPASS` set: `sudo -A -u postgres <argv>`.
   - Linux otherwise: `sudo -n -u postgres <argv>`.
   - macOS (current user `rolsuper`): `<argv>` directly.
   - Container root: `sudo -u postgres <argv>`.
   - None available: raise `SeamUnavailable`. The caller exits 3 with S-NOSEAM.
10. **Restore** — `restore(backups_dir, stamp) -> RestoreResult`, each step bounded:
    1. Read `manifest.json`; absent ⇒ exit 1 naming the path.
    2. `pg_restore --list` the dump; nonzero ⇒ exit 1, nothing destroyed.
    3. Via seam: `dropdb --force mira_service`; `createdb -O mira_admin mira_service`; `pg_restore -d mira_service <dump>` (owners, grants, extensions as dumped).
    4. Assert `pgcrypto`, `vector`, `pg_trgm` are present.
    5. Restore the `systemone.env` and unit copies recorded present.
    6. Vault: patch back the fields whose current value differs from `vault.json`.
    7. Fingerprint equals the manifest's; mismatch ⇒ exit 1.
    8. On any failure: print S-RESTORE-PARK, exit 1. Never retry, never report success.
11. **CLI wiring:**
    - `migrate` (no `--dry-run`) → `migrate()`; `--adopt-drifted` reaches `bootstrap_check` only.
    - `restore <STAMP>` → `restore()`; an absent or unknown stamp lists the available stamps.
12. **Maps:**
    - `utils/AGENTS.md`: apply loop, lock key, adapters, seam, backup/restore, adoption table.
    - `clients/AGENTS.md`: `BOOT_FLUSH_PRESERVE_PREFIXES` and its two readers; `ctx.vault` reuses `VaultClient`.
    - Root `AGENTS.md` `main.py` card: cite the constant's owner.
    - `deploy/migrations/AGENTS.md`: lock key, expand/contract rule, operator-input contract, `ctx.embedding_dimensions`.
    - `scripts/AGENTS.md`: `migrate`/`restore`.
    - `deploy/AGENTS.md`: Docker persistence + backups volume.
    - `deploy/docker/scripts/AGENTS.md`: backups dir.

### Probes

| Probe | Asserts |
|---|---|
| `probe_migrate_apply.py` | Chain `V9001..V9005` + `R__probe`: skip-chain, where one invocation from K equals walking one step per invocation, K ∈ {9001, 9003, 9004} (fingerprint + ledger minus `applied_at`/`duration_ms`); the repeatable re-runs only on a byte change |
| `probe_migrate_interrupt.py` | `V9003` = `SELECT 1/0` ⇒ exit 1 naming it; ledger ends at 9002 with no 9003 row; 9003 objects are absent; fixing the file then re-running reaches head |
| `probe_sigkill_resume.py` | SIGKILL during a `pg_sleep` step leaves no row and no partial DDL; the re-run acquires the lock within a bound; nothing in `pg_locks` afterwards |
| `probe_advisory_lock.py` | Two concurrent runs: one applies, the other applies nothing and exits 0; each revision appears exactly once |
| `probe_migrate_privileges.py` | As `mira_admin` over TCP: add/drop column, create table/index, drop/add constraint all succeed; the four `SECURITY DEFINER` functions are callable; `CREATE DATABASE` raises `InsufficientPrivilege` |
| `probe_adoption_drift.py` | DBs from each origin's schema (`git show <tag>:…`), the current schema, `v2026.10.06-2.0` + hand-applied `V0001` index, current + an extra column, and an adopted DB (runnable once Phase 5 ships the chain; until then over a `/tmp` chain + `/tmp` `BASELINE`): origins stamp nothing and converge to head; current stamps `0001..head`; hand-patched matches `0001`; the extra column exits 2 with an unchanged fingerprint + ledger, a report naming the column, and the literal override command; the override converges and records `+adopt-drifted` on every written row plus the `name` suffix, and `status` shows the override; a second run without the flag finds nothing to do; the flag is ignored on the adopted DB |
| `probe_post_migrate_drift.py` | An adopted scratch DB at head: a clean run prints nothing extra; after a hand-added column and a hand-changed index definition, `migrate` exits 0, prints S-POSTDRIFT and a report naming both objects, leaves both in place, and `status --json` shows `fingerprint_drift: true`; no banned words |
| `probe_operator_input.py` | Closed stdin ⇒ exit 3, prompt + S-OPINPUT, no row for that step; under a pty the typed value is used and the row exists |
| `probe_backup_retention.py` | After 4 backups, 3 remain and the report names the removed one; manifest fields present, written last (kill before the manifest leaves none); `vault.json` 0600; disk-precheck arithmetic checked on real numbers; `/opt/mira/backups` is not under `/opt/mira/app`; Dockerfile `VOLUME` (parsed JSON), README mounts, `init-mira.sh` mkdir/chown present |
| `probe_restore_roundtrip.py` | Real seeded rows (`tests/fixtures/probe_user.py`, production repositories) → `take_backup` → damage → `restore` via `run_as_superuser`: extensions present including `vector`; fingerprint equals the manifest; row counts identical; owners `mira_admin`; `mira_dbuser` with no user context gets zero rows and no error; `mira_admin.rolcreatedb = false`; probe user removed |
| `probe_vault_adapter.py` | Scratch path `secret/mira/probe_adapter`: read/patch round trip; `delete_fields` refuses without confirmation; ≥2 KV versions; metadata deleted. UNVERIFIED without AppRole |
| `probe_valkey_invalidate.py` | Keys under every preserved prefix survive; `probe:doomed` is gone; the constant is the same object `main.py` passes |

### Done when

- [ ] All Phase-3 probes pass (Vault probes may be UNVERIFIED with the blocker named).
- [ ] The lock is released on every exit path, SIGKILL included.
- [ ] The only superuser use is `run_as_superuser`.
- [ ] The map audits are clean.

---

## Phase 4 — wiring: consent gate, phase 6.5, failure path, boot guard, dispatch, installers

**Read first:** `deploy/update.sh` (whole file, as edited in Phase 0), `deploy/lib/services.sh` (`acquire_sudo`, `ensure_sudo`), `ui/tui/update.py` (exit-code docstring, `_gate_and_hand_off`), `ui/tui/__main__.py`, `deploy/finalize.sh` (`mira` wrapper `PYTHONPATH`), `utils/power_on_self_test.py` (`run_pre_server_post`), `clients/postgres_client.py`, `deploy/docker/scripts/init-mira.sh` (`init_postgresql`, `init_vault`, `main` order), `deploy/postgresql.sh` (Step 14).

### Steps

1. **`seam_mode`** (`deploy/lib/services.sh`) is read-only: no prompt, no `sudo -k`.

   | Verdict | Condition |
   |---|---|
   | `none` | macOS with `psql -d postgres -tAc "SELECT rolsuper FROM pg_roles WHERE rolname=current_user"` = `t`; or the container root |
   | `passwordless` | Linux and `sudo -n -u postgres psql --version` exits 0 |
   | `interactive` | Linux, not passwordless, `[ -t 0 ]` |
   | `unavailable` | Linux, not passwordless, no TTY |
2. **`deploy/update.sh` phase order:**

   | Phase | Action |
   |---|---|
   | 1 preflight | trees present; versions differ (nightly equality ⇒ exit 0); install tree is a git repo; `PRE_UPDATE_SHA`; `SEAM_MODE="$(seam_mode)"` |
   | 2 | extras detection on the live venv |
   | 3 | stamp-gated build of `venv.new` / `tui-venv.new` |
   | 3.5 consent gate | step 3 |
   | 4 | stop the service: `launchctl` on macOS; Linux `ensure_sudo` + `sudo systemctl stop mira` |
   | 4.5 | save: export, drift record, snapshot — any failure restarts the old service and exits 1 with nothing changed (Phase 0 step 15) |
   | 5 | guarded swap + TOAST shim refresh |
   | 6 | clear, overlay, commit |
   | 6.5 | step 4 |
   | 7 | restart (macOS `launchd_reload_agent`; Linux `ensure_sudo` + `sudo systemctl start mira`); health poll 600 s for `status=healthy` AND `version=<new>`; then `scripts.migrate status --json` must show `ledger_head == tree_head` and `pending == []` |
   | 8 | success: delete swapped asides; `git update-ref -d refs/mira/pre-update/<STAMP>`; advance `data/nightly_stamp` (nightly only); print the result. Failure: step 5 |

   - Delete every `sudo -n systemctl` site and its "Cannot stop mira.service without a sudo password" refusal block.
3. **Consent gate** (phase 3.5). Every exit from the gate first deletes `venv.new` and `tui-venv.new`:
   1. `PLAN_PY` = `venv.new/bin/python3` if built, else `venv/bin/python3`.
   2. Export `VAULT_ADDR`, `VAULT_ROLE_ID`, `VAULT_SECRET_ID` from `/opt/vault/*.txt`.
   3. Run `PYTHONPATH="$TREE" "$PLAN_PY" -m scripts.migrate migrate --plan --report-marker --elevation "$SEAM_MODE"`.
   4. Nonzero exit, missing marker, or bad JSON ⇒ print the runner output + S-PLAN-FAIL, exit 3.
   5. `needs_consent=false` ⇒ no output, no prompt; continue.
   6. `needs_consent=true` and `SEAM_MODE=unavailable` ⇒ print the notices + S-NOTTY-SEAM, exit 3.
   7. `needs_consent=true` and no TTY ⇒ print the notices + S-NOTTY-CONSENT, exit 3.
   8. Otherwise print the plan text, then S-PROMPT, then `read -r`. `Y`/`y` continues; anything else ⇒ S-CANCEL, exit 0.
   9. After Y with `SEAM_MODE=interactive`: print S-ELEV, then `acquire_sudo`; failure ⇒ exit 1.
4. **Phase 6.5:**
   - Run `PYTHONPATH="$APP_DIR" "$APP_DIR/venv/bin/python3" -m scripts.migrate migrate --stamp "$STAMP"`, with the Vault env exported and `SUDO_ASKPASS` inherited.
   - Then run `scripts.migrate reconcile` (Phase 6) and print S-RECONCILE.
   - Any nonzero exit ⇒ step 5.
   - On a non-canonical root it still runs; only service control is skipped.
5. **Failure path.** Triggers: phase 6.5 nonzero, or a failed phase-7 check.
   1. If the new version is running, stop it.
   2. If `<backups>/<STAMP>/manifest.json` exists ⇒ `scripts.migrate restore "$STAMP"`. On failure: S-RESTORE-PARK is already printed; print S-UPDATE-PARK, leave the service stopped, exit 1.
   3. Execute `ROLLBACK_VENV_CMDS`.
   4. `git -C "$APP_DIR" reset --hard "refs/mira/pre-update/$STAMP"`, then `git -C "$APP_DIR" reset --mixed "$PRE_UPDATE_SHA"`.
   5. Restart the old version; health-poll the old `VERSION` for 600 s.
   6. Print S-ROLLED-BACK, exit 1. Keep the snapshot ref and the backup.
   - Rewrite the header: phases, the sudo uses (seam + `systemctl`, at most one password prompt after consent), and automatic rollback.
6. **Boot guard** — `CheckSpec("schema_revision", True, _check_schema_revision)` after `"postgres_rls"`:
   - Read-only, as `mira_dbuser` through `PostgresClient("mira_service")`.
   - Uses only `discover_steps` and a ledger read; no import of the apply path.
   - Returns `{ledger_head, tree_head, pending}`.

   | Condition | Message |
   |---|---|
   | `to_regclass('public.schema_migrations') IS NULL` | S-BOOT-UNPREPARED |
   | ledger head < tree head | S-BOOT-BEHIND |
   | ledger head > tree head | S-BOOT-AHEAD |
   | a ledger revision ≤ tree head has no file | S-BOOT-MISSING |
7. **Dispatch** (`ui/tui/__main__.py` + new `ui/tui/migrate.py`):
   - `migrate` and `restore` run `[<app_dir>/venv/bin/python3, "-m", "scripts.migrate", *args]` with `PYTHONPATH=<app_dir>`, forwarding args verbatim (no flag whitelist) and the exit code unchanged.
   - `run_status` moves to `ui/tui/migrate.py` and appends the database half by parsing `scripts.migrate status --json` from the same subprocess builder.
8. **`ui/tui/update.py` exit-code docstring:** 0 = up to date / updated / cancelled; 1 = failure; 3 = needs a terminal.
9. **Container:**
   - `init-mira.sh:main` order: `init_postgresql` (create DB + schema + bootstrap on first boot only) → `init_vault` → `/opt/mira/venv/bin/python3 -m scripts.migrate migrate` (every start) → `exec /init`.
   - Delete the schema-skip gate's assumption that an existing DB needs nothing.
10. **Host installer:**
    - `deploy/postgresql.sh`, after Step 14: run `PYTHONPATH=/opt/mira/app /opt/mira/app/venv/bin/python3 -m scripts.migrate migrate` with the Vault env exported. Nonzero is fatal. This stamps the fresh install at head.
    - `deploy/deploy_database.sh`: print that exact command as the last manual next-step.
11. **Maps:**
    - `deploy/AGENTS.md`: the `update.sh` card end to end (phase table, consent gate, failure path); the in-place-update Rule (the chain carries a database forward; the schema file installs empty databases only); Wiring line for code swap → migrate → restart; `postgresql.sh` / `deploy_database.sh` migrate call.
    - `deploy/lib/AGENTS.md`: `seam_mode`; `update.sh` as a caller of `acquire_sudo`/`ensure_sudo`.
    - `deploy/docker/scripts/AGENTS.md`: start order and why (the runner reads Vault).
    - `utils/AGENTS.md`: `schema_revision` + migrations never run at boot.
    - `ui/tui/AGENTS.md`: dispatch, `migrate.py` card, exit codes.
    - `ui/AGENTS.md`: verify the import invariant still holds.
    - `scripts/AGENTS.md`: callers (`mira`, `update.sh` preflight and 6.5, installers, container).

### Probes

| Probe | Asserts |
|---|---|
| `probe_superuser_seam.sh` | Real states, no fake sudo: valid ticket ⇒ `passwordless`; after `sudo -k` under a pty ⇒ `interactive` with no prompt; `/dev/null` stdin ⇒ `unavailable`; macOS ⇒ `none` (`rolsuper=t`); no `passwordless` prerequisite in any `print_*` |
| `probe_preflight_consent.sh` | `/tmp` tree + root, `CANONICAL=no`, a scratch DB one step behind. Y arm: plan before prompt, NOTICE verbatim, exact prompt, proceeds to phase 4. N arm: exit 0, S-CANCEL, `venv.new` removed, `VERSION`/`git log -1`/ledger/fingerprint unchanged, no service action. Non-TTY: exit 3, S-NOTTY-CONSENT. Empty plan: no prompt. Unavailable seam: exit 3 before any prompt text. `PLAN_PY` is `venv.new` when built |
| `probe_update_phase65.sh` | Under a pty answering Y: the step sees the new `VERSION`; ledger at head after. Failing step ⇒ restore runs, asides put back, files equal the pre-update tree, HEAD equals `PRE_UPDATE_SHA`. Truncated dump ⇒ park block, no `mv`, exit 1. Validation refusal (no manifest) ⇒ no restore, code + venvs rolled back |
| `probe_post_schema_revision.py` | Four states raise their exact messages (no banned words, command present); at head returns `pending == 0`; read-only |
| `probe_boot_gate_end_to_end.sh` | A server pointed at a behind scratch DB never binds the port and logs `schema_revision` + `Run: mira update` |
| `probe_mira_migrate_dispatch.sh` | `tui-venv` cannot import `psycopg`; `mira migrate --dry-run` / `mira status` / `mira restore` give 0 / 0 / 1 with server-venv output; `mira` from a checkout resolves the installed tree |
| `probe_container_migrate.sh` | A rebuilt image on a persisted volume applies the new step at start; serves `/v0/api/health` with no `docker exec`; a second start applies nothing; interpreter `/opt/mira/venv/bin/python3`. BLOCKED without docker |
| `probe_fresh_install_stamp.py` | Scratch DB from the current schema + bootstrap, then `migrate` ⇒ ledger stamped `0001..head`, `schema_revision` passes |
| `probe_health_third_assertion.sh` | At head, the third check passes; one behind, it fails and phase 7 reports it |

### Done when

- [ ] All Phase-4 probes pass, or BLOCKED/UNVERIFIED with the venue named.
- [ ] `grep -n "sudo -n systemctl\|never auto-rolls back" deploy/update.sh deploy/AGENTS.md` is empty.
- [ ] The map audits are clean.

---

## Phase 5 — first steps, BASELINE, release gate, BREAKING gate removal

**Read first:** `deploy/mira_service_schema.sql` (`idx_messages_live_segment_unique`, `heartbeat_armed_users`), `git diff v2026.10.06-2.0 HEAD -- deploy/mira_service_schema.sql`, `git diff v2026.10.03-2.0 HEAD -- deploy/mira_service_schema.sql`, `ui/tui/update.py` (`_print_breaking_block`, `_gate_and_hand_off`, module docstring), `deploy/update.sh` (BREAKING re-check), `deploy/RELEASE.md`, `deploy/deploy.sh` header, `README.md` + `docs/MANUAL_INSTALL.md` update sentence, root `AGENTS.md` (`BREAKING.md` card), `deploy/AGENTS.md`, `ui/tui/AGENTS.md`, `docs/AGENTS.md`.

### Steps

1. **`deploy/migrations/V0001__widen_segment_sentinel_uniqueness.sql`:**
   ```sql
   -- store: postgres
   -- notice: Fixes a rule that could lock a conversation after a message was
   -- saved while it was paused. Nothing is deleted. If a conversation is
   -- already in that state, MIRA stops and names it before changing anything.
   DO $precondition$
   DECLARE
       bad RECORD;
   BEGIN
       SELECT continuum_id, count(*) AS n INTO bad
         FROM messages
        WHERE metadata->>'is_segment_boundary' = 'true'
          AND metadata->>'status' IN ('active', 'paused')
        GROUP BY continuum_id
       HAVING count(*) > 1
        LIMIT 1;
       IF FOUND THEN
           RAISE EXCEPTION 'This upgrade stopped before changing anything: one conversation has % live segment boundaries (conversation %), and only one is allowed. Ask MIRA to remove the duplicate — it needs that conversation id — then run: mira update', bad.n, bad.continuum_id;
       END IF;
   END
   $precondition$;
   DROP INDEX IF EXISTS idx_messages_active_segment_unique;
   CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_live_segment_unique
       ON messages(continuum_id)
       WHERE metadata->>'is_segment_boundary' = 'true'
         AND metadata->>'status' IN ('active', 'paused');
   ```
2. **`V0002__drop_keepsleeping_turn_index.sql`:** `-- store: postgres`; `-- notice: Removes a database index this version no longer uses. Nothing in your data changes.`; `DROP INDEX IF EXISTS idx_messages_keepsleeping_turn_id;`
3. **`V0003__create_heartbeat_armed_users.sql`:** `-- store: postgres`; `-- notice: Adds the table MIRA uses to know which users have scheduled briefs. Nothing in your data changes.`
   - Body: the schema file's `heartbeat_armed_users` `CREATE TABLE IF NOT EXISTS` + `idx_heartbeat_armed_wake` `CREATE INDEX IF NOT EXISTS`, plus every grant/policy the schema file applies to that table, copied from the file.
   - The convergence probe fails on any difference.
4. **BASELINE generator** — `utils/schema_migrations.py`:
   - `SCHEMA_ORIGINS = ("v2026.10.03-2.0", "v2026.10.06-2.0")`.
   - `baseline_fingerprints(repo_root, origins) -> str`:
     1. For each origin, apply `git show <tag>:deploy/mira_service_schema.sql` to its own scratch DB and fingerprint it (ledger excluded).
     2. On the newest origin's DB, apply `V0001..head` in order, fingerprinting after each.
   - Write `deploy/migrations/BASELINE` (rows append-only across releases) and `deploy/migrations/HEAD_FINGERPRINT.json` (the head DB's canonical fingerprint JSON, rewritten every cut).
5. **Ledger DDL generator** — `ledger_ddl() -> str` and `write_generated_artifacts(repo_root)`: rewrite the sentinel blocks in `deploy/mira_service_schema.sql` and `deploy/bootstrap_adopt.sql`.
6. **Delete the BREAKING gate** (the file is already absent):
   - `ui/tui/update.py`: delete `_print_breaking_block`, the `BREAKING.md` check in `_gate_and_hand_off`, and the docstring paragraph. The `deploy/update.sh` presence check stays.
   - `deploy/update.sh`: delete the `BREAKING.md` re-check and its header clauses.
   - `deploy/RELEASE.md`: delete the breaking-release section.
   - `deploy/deploy.sh` header: delete the breaking-release + "database stays greenfield" sentences.
   - Root `AGENTS.md`: delete the `BREAKING.md` card; the `UPGRADE_PATH.md` card cites `deploy/migrations/`.
   - `deploy/AGENTS.md`, `ui/tui/AGENTS.md`, `docs/AGENTS.md`: delete every BREAKING statement and the "schema change is breaking" rule.
   - `README.md` + `docs/MANUAL_INSTALL.md`: the update sentence becomes: `mira update` shows what it will do and asks once before changing the database, takes a backup first, and puts the install back if anything fails; `mira restore <STAMP>` restores a backup.
7. **`deploy/RELEASE.md` — "Updating an installed MIRA" + release gate.** A release is not published unless all of these hold:

   | # | Condition | Mechanism |
   |---|---|---|
   | a | `VERSION` ↔ tag ↔ commit | `verify_release_identity` inside `build_release_artifact` (no asset on failure) |
   | b | step-chain integrity | `scripts.migrate lint` inside `build_release_artifact` (no asset on failure) |
   | c | convergence | `probe_convergence.py` `PROBE PASSED` in the cut's change report |
   | d | skip-chain | `probe_skipchain.py` `PROBE PASSED` |
   | e | restore round trip | `probe_restore_roundtrip.py` `PROBE PASSED` |

   Checklist lines:
   - build and attach the artifact;
   - regenerate `BASELINE` (append-only) and `write_generated_artifacts`;
   - `install.sh` `REPO=` equals `deploy/GITHUB_REPO`;
   - every new step carries a NOTICE or deliberately ships without one;
   - record the convergence probe's released-tag report;
   - `grep -rn BREAKING` over the shipped tree is empty;
   - shipped steps are never edited (enforced by `CHECKSUM_MISMATCH`);
   - nightly: `update nightly-<12hex>` commits, `applied_by` records the nightly, the gate does not apply.
8. **Maps:**
   - `deploy/migrations/AGENTS.md`: cards for `V0001`–`V0003` and `BASELINE`; the origins rule (closed set).
   - `deploy/AGENTS.md`: the `RELEASE.md` card with the gate.
   - `utils/AGENTS.md`: the generators.

### Probes

| Probe | Asserts |
|---|---|
| `probe_v0001.py` | Two live sentinels on a real seeded continuum ⇒ exit 1, message names the id + count + command, nothing mutated. After the stray is removed through the production path: old index gone, new index def contains `IN ('active'…'paused')`, ledger row present; seeded rows unchanged by content; a paused-segment duplicate insert raises; RLS fails closed; a re-run of the body changes nothing |
| `probe_convergence.py` | For each origin, `git show <tag>` schema + chain equals the current schema file's fingerprint equals the `BASELINE` head row; prints every release tag's matched row or `UNMATCHED` |
| `probe_skipchain.py` | Over the real chain, one invocation from K equals walking, for K ∈ {0000, 0001, 0002} |
| `probe_baseline_generated.py` | `baseline_fingerprints` twice gives identical output equal to the checked-in file; `write_generated_artifacts` into a `/tmp` copy gives zero diff, and an edited `ledger_ddl()` changes both files |
| `probe_breaking_absent.sh` | Release trees with and without a stray `BREAKING.md` behave identically (a witness `update.sh` is invoked in both, no `BREAKING` output); `_print_breaking_block` does not exist; a tree without `deploy/update.sh` still returns 1 |
| `probe_release_checklist.sh` | Runs the checklist on the working tree: gate a/b each refuse when broken (no asset); c/d/e need recorded `PROBE PASSED` lines; a failure prints `DO NOT PUBLISH: <condition>` |

### Done when

- [ ] All Phase-5 probes pass.
- [ ] `grep -rn "BREAKING" ui/ deploy/ README.md docs/ AGENTS.md` is empty.
- [ ] The map audits are clean.

---

## Phase 6 — reference data and settings reconciliation

**Read first:** `deploy/mira_service_schema.sql` (`usage_pricing`), `config/config_manager.py` (the four env registries), `auth/mode.py` (`AUTH_MODE_ENVIRONMENT_FIELD`), `utils/power_on_self_test.py` (`POST_GATE_FAILURE_ACTION_ENV`), `deploy/finalize.sh` (`systemone.env` writer, systemd unit template, `run.sh`, plist).

### Steps

1. **`deploy/migrations/R__usage_pricing.sql`:** `-- store: postgres`; `DELETE FROM usage_pricing;` + a full `INSERT` of the declared rows.
2. **Declared env vars** — `declared_environment_variables(repo_root) -> dict[str, str]`: an AST sweep of every `MIRA_*` literal in `os.getenv`/`os.environ[...]`/`os.environ.get` and every module-level `MIRA_*` string constant, unioned with the four `config_manager` registries. Maps name → owning symbol.
3. **Env reconciliation** — `reconcile_env_file(env_file, declared, current) -> EnvReconciliation`, pure: `carried_forward`, `dropped` (with source line), `newly_defaulted` (with the Pydantic default), and `rendered` (sorted `NAME=value`, no comments).
4. **Unit generator:**
   - Extract the systemd unit template from `deploy/finalize.sh` into one generator, called by `finalize.sh` and by the reconciler.
   - `reconcile_supervisor_unit()`: Linux only, and only when the unit's baked `VAULT_SECRET_ID` ≠ `/opt/vault/secret-id.txt`. Then write via `sudo -A`/`sudo -n` + `systemctl daemon-reload`.
   - No elevation available ⇒ print the exact `sudo` command and continue.
   - macOS: no-op, because the plist execs `run.sh`, which reads the AppRole files at start.
5. **`scripts.migrate reconcile`:**
   - Runs both reconcilers on every update (phase 6.5, after `migrate`).
   - Writes the env file through `ctx.env`.
   - Prints S-RECONCILE; the variable names go to the log.
   - A nonempty `newly_defaulted` logs at WARNING.
6. **Maps:**
   - `deploy/migrations/AGENTS.md`: repeatable-step contract (wholesale replace, converge-to-state).
   - `utils/AGENTS.md`: declared-variable sweep + reconcilers.
   - `config/AGENTS.md`: one-line citation that the registries feed the sweep.
   - `deploy/AGENTS.md`: the `finalize.sh` shared unit generator.

### Probes

| Probe | Asserts |
|---|---|
| `probe_repeatable_step.py` | First run applies with a `kind='repeatable'` row; a second run applies nothing; a 1-byte edit re-applies; a hand-inserted extra row is gone |
| `probe_env_reconcile.py` | The declared set includes all four registries + `MIRA_AUTH_MODE` + `MIRA_POST_GATE_FAILURE_ACTION`; a retired var lands in `dropped` with its line; a missing var lands in `newly_defaulted` with its default; a rename produces both; `rendered` round-trips |
| `probe_unit_reconcile.py` | Linux + real unit: a differing secret is regenerated, every other line equals the `finalize.sh` generator's output, `systemctl cat` shows the new value; a match writes nothing (mtime unchanged); macOS says why it is a no-op. BLOCKED without sudo |
| `probe_env_reconcile_live.py` | Deployed host: an appended `MIRA_RETIRED_VAR` is gone after `mira migrate` + `reconcile`, other values kept, health stays green |

### Done when

- [ ] All Phase-6 probes pass (the unit probe may be BLOCKED).
- [ ] The map audits are clean.

---

## Phase 7 — per-user SQLite chain

**Read first:** `utils/userdata_manager.py` (`_ensure_database`, `_initialize_tool_schemas`, the four `_migrate_*` helpers, the encryption path), `tests/fixtures/probe_user.py`, `tests/fixtures/residue.py`.

### Steps

1. **`utils/userdata_migrations.py`:**
   - `STEPS: tuple[tuple[int, str, Callable[[sqlite3.Cursor], None]], ...]`.
   - `run_userdata_migrations(conn) -> int` applies each step above `PRAGMA user_version`, one transaction per step, setting `user_version` in the same transaction. A raise rolls back that step and propagates.
2. **Move the helpers verbatim, introspection guards kept:** `_migrate_pager_schema` = 1, `_migrate_domaindoc_schema` = 2, `_migrate_domaindoc_versions_unique` = 3, `_migrate_contacts_schema` = 4.
3. **`_initialize_tool_schemas`:**
   - Keep `CREATE TABLE IF NOT EXISTS`.
   - Replace the four calls with `run_userdata_migrations`.
   - Delete the "introduce versioned migrations" comment.
   - A file new at `_ensure_database` gets `user_version = head` without running steps.
4. **Maps:**
   - `utils/AGENTS.md`: `userdata_manager.py` card (schema init) and `userdata_migrations.py` card (ledger = `user_version`, the four steps, new-file shortcut, tool-created tables not owned).
   - `tools/implementations/AGENTS.md`: one-line citation.

### Probes

| Probe | Asserts |
|---|---|
| `probe_userdata_migrations.py` | Live Vault + probe user. New file: version 4, no step body ran. Pre-migration `pager_devices`: all four run, guards skip. Legacy `domaindoc_versions` index: rebuilt with rows byte-identical. Current file: nothing runs. A raise at step 3 leaves version 2 and propagates. `encrypted__` values decrypt after |
| `probe_userdata_no_manager_regression.py` | On a copy of a real `userdata.db`: CRUD behaves identically to an unmodified checkout's manager |

### Done when

- [ ] Both probes pass.
- [ ] The four helper names are absent from `utils/userdata_manager.py`.
- [ ] The map audits are clean.

---

## Phase 8 — closeout

**Read first:** every map touched in Phases 0–7, `deploy/HOW_TO_MIGRATE_OLD_INSTALLS.txt`, `NEARFUTURE_FEATURES.md`, `tests/protected/AGENTS.md`.

### Steps

1. Run the map skill's anchor audits and the reciprocity check over every touched map.

   | Edge | Owner | Cited by |
   |---|---|---|
   | `update.sh` → `scripts/migrate.py` → `utils/schema_migrations.py` | `deploy/AGENTS.md` | `scripts/`, `utils/` |
   | consent gate → `migrate --plan` → `render_plan` | `deploy/AGENTS.md` | `scripts/`, `utils/` |
   | `mira migrate`/`restore`/`status` → server venv | `ui/tui/AGENTS.md` | `scripts/` |
   | `bootstrap_adopt.sql` callers | `deploy/AGENTS.md` | `deploy/docker/scripts/`, `tests/fixtures/` |
   | ledger `GRANT SELECT` → `schema_revision` | `utils/AGENTS.md` | `deploy/` |
   | `BOOT_FLUSH_PRESERVE_PREFIXES` | `clients/AGENTS.md` | root (`main.py`) |
   | `BASELINE` generator + origins | `deploy/migrations/AGENTS.md` | `deploy/`, `utils/` |
   | release artifact → `RELEASE_COMMIT` → fetch sites | `deploy/AGENTS.md` | `deploy/lib/`, `ui/tui/`, `utils/` |
   | `clear_code_preserving_state`, `snapshot_working_tree` | `deploy/lib/AGENTS.md` | `deploy/` |
   | `seam_mode`/`acquire_sudo`/`ensure_sudo` → `update.sh` | `deploy/lib/AGENTS.md` | `deploy/` |
   | `deploy/GITHUB_REPO` readers | `deploy/lib/AGENTS.md` | `deploy/`, `utils/`, root (`install.sh`) |
   | `require_pg_floor` | `deploy/lib/AGENTS.md` | `deploy/` |
   | backups volume | `deploy/AGENTS.md` | `deploy/docker/scripts/`, `utils/` |
   | unit template generator | `deploy/AGENTS.md` | `utils/` |
2. Scope `deploy/HOW_TO_MIGRATE_OLD_INSTALLS.txt`'s framing to pre-2.0 installs (major crossings). `mira update` never prints it.
3. Remove any shipped migration item from `NEARFUTURE_FEATURES.md`.
4. Write the probe list into `deploy/migrations/AGENTS.md` as a `## Verification` deep-dive.
4a. Verify `docs/MAINTAINER_UPGRADE_GUIDE.md` against the shipped code: every path, command, flag, and quoted message exists exactly as written (messages against the Operator strings table as implemented). Delete its "Status" paragraph's not-yet-implemented sentence. Its checklist must match `deploy/RELEASE.md`'s; `deploy/RELEASE.md` and `deploy/migrations/AGENTS.md` each gain a one-line pointer to the guide.
5. **Language sweep** — `probe_normie_language.sh`:
   - Covers every operator surface: `print_*` args in the edited `deploy/` scripts, `render_plan`, `render_drift_report`, every `Refusal.message`, the boot-guard messages, every NOTICE and `RAISE EXCEPTION` under `deploy/migrations/`, and the surviving `ui/tui/update.py` messages.
   - No banned word; every refusal/park/exit-3 string contains a command; the same words are present at the matching log sites.
   - Positive control: an injected jargon string is reported with its file and line.
6. **Permanent battery** — ask the operator to type `AUTHORIZE PROTECTED TEST SAVE` for `tests/upgrades/`.
   - Candidates: `probe_migrate_validate`, `probe_migrate_apply`, `probe_migrate_interrupt`, `probe_sigkill_resume`, `probe_migrate_privileges`, `probe_restore_roundtrip`, `probe_adoption_drift`, `probe_post_schema_revision`, `probe_v0001`, `probe_convergence`, `probe_skipchain`, `probe_breaking_absent`.
   - Not granted: all stay disposable, and the request is recorded verbatim in the change report.
   - Granted: create `tests/upgrades/AGENTS.md` first (exact-phrase admission per file per save; live infrastructure is the contract; NO MOCKS; BLOCKED exit 3 when infrastructure is down; run as `python3 tests/upgrades/<name>.py`). Add the root registry row, then one commit per file with its card.

### Done when

- [ ] The audits and reciprocity check are clean.
- [ ] `probe_normie_language.sh` passes.
- [ ] No probe file remains in the repo (unless the battery was granted).
- [ ] The battery decision is recorded.
- [ ] This file and its `docs/AGENTS.md` card are deleted in the closing commit — the implementation and the maps are the standing record.

---

## Operator strings

| ID | Text |
|---|---|
| S-NOTHING | `Nothing to do — your database is already ready for this version.` |
| S-STEP-DEFAULT | `This step changes MIRA's database structure. Nothing in your data is removed by it.` |
| S-PLAN-FIRST | `This update includes a one-time upgrade to how MIRA stores its data. Your messages, memories, and settings are untouched.` |
| S-ELEV-INTERACTIVE | `Your admin password may be asked for, once.` |
| S-ELEV-PASSWORDLESS | `No password is needed.` |
| S-ELEV-NONE | `No password is needed on this machine.` |
| S-PROTECT | `MIRA backs up your data before it starts and puts your install back exactly as it was if anything goes wrong.` |
| S-PROMPT | `Press Y to continue, N to cancel (nothing changes): ` |
| S-CANCEL | `Update cancelled — nothing changed.` |
| S-PLAN-FAIL | `MIRA could not read your database to work out what this upgrade needs to do. Nothing was changed. Make sure MIRA is running, then run: mira update` |
| S-NOTTY-CONSENT | `This upgrade changes your database and needs your OK first, and this session has no terminal to ask you in. Nothing was changed. Run: mira update again from a terminal where you can type.` |
| S-NOTTY-SEAM | `This upgrade needs to reach your database as its administrator, and this session has no terminal to ask for your password. Nothing was changed. Run: mira update again from a terminal where you can type.` |
| S-ELEV | `MIRA needs your admin password once to upgrade its database.` |
| S-NOSEAM | `MIRA could not reach the database as its administrator, so it cannot upgrade your database's structure. Nothing was changed. Run: mira update from a terminal, and type your admin password when asked.` |
| S-OPINPUT | `MIRA needs you to type a value to finish this upgrade, and this session has no terminal. The upgrade stopped before that step. Run: mira migrate from a terminal.` |
| S-DISK | `There is not enough free disk space to back up your data before upgrading (needs <X>, has <Y>). Nothing was changed. Free up space, then run: mira update` |
| S-DOWNGRADE | `Your database has already been upgraded past this version of MIRA (it is at step <N>; this version knows steps up to <M>). MIRA will not move your data backwards, and nothing was changed. Install the newer version again: mira update` |
| S-REVERTED | `Your database records step <N>, but this copy of MIRA has no such step — the copy is incomplete. Nothing was changed. Re-run from a complete release: mira update` |
| S-CHECKSUM | `Step <N> (<slug>) was changed after MIRA ran it on your database, so MIRA cannot tell what state that part of your data is in. Nothing was changed. This is a fault in the release, not in your data — report it, then re-run: mira update` |
| S-OUTOFORDER | `Step <N> would have to run before steps MIRA has already run (your database is at step <M>). MIRA never runs steps out of order, and nothing was changed. Re-run: mira update` |
| S-DUPLICATE | `Two steps in this release claim the same number (<N>). This is a fault in the release; nothing was changed. Report it, then run: mira update` |
| S-GAP | `This release has no step numbered <N>. This is a fault in the release; nothing was changed. Report it, then run: mira update` |
| S-LINT | `A step in this release breaks MIRA's rules for safe upgrades (<file>). This is a fault in the release; nothing was changed. Report it, then run: mira update` |
| S-DRIFT-LEAD | `Here is everything that is different about your database's structure:` |
| S-DRIFT | `Your database's structure is not one this version of MIRA recognizes — every difference is listed above. Nothing was changed. If you made that change yourself and want to continue anyway, run exactly: mira migrate --adopt-drifted` |
| S-BOOT-UNPREPARED | `MIRA's database has not been prepared for this version yet. Run: mira update` |
| S-BOOT-BEHIND | `MIRA's database is at step <N> and this version needs step <M>. Run: mira update` |
| S-BOOT-AHEAD | `MIRA's database is at step <N>, which is newer than this version of MIRA knows about (step <M>). MIRA will not start on a database it would have to move backwards. Install the newer version: mira update` |
| S-BOOT-MISSING | `This copy of MIRA is missing step <list>, which your database has already run — the copy is incomplete. MIRA will not start on it. Reinstall the complete version: mira update` |
| S-RESTORE-PARK | Block, one item per line: `The upgrade failed and MIRA could not put your data back automatically.` / `Your database is partway through the change, so your install is not as it was.` / `Your backup is safe at: <path>` / `To put it back by hand, run:` then the exact superuser `dropdb --force mira_service`, `createdb -O mira_admin mira_service`, `pg_restore -d mira_service <dump>` lines |
| S-UPDATE-PARK | Block: `MIRA is stopped and will stay stopped until your data is put back.` / the `ROLLBACK_VENV_CMDS` lines / `git -C <app> reset --hard refs/mira/pre-update/<STAMP>` / `git -C <app> reset --mixed <PRE_UPDATE_SHA>` / the restart command / `Then run: mira update` |
| S-ROLLED-BACK | `The update could not finish, so MIRA put your install back exactly as it was and restarted the previous version (<old>). Details are in <log>. Run: mira update to try again.` |
| S-PRESERVE-SPACE | `There is not enough free disk space to save a copy of the files you changed (needs <X>, has <Y> at <path>). Nothing was changed. Free up space, then run: mira update` |
| S-PRESERVE-FAIL | `MIRA could not save a safe copy of the files you changed, so it stopped before touching anything. Your install is exactly as it was and is running again. Details are in <log>. Run: mira update to try again.` |
| S-POSTDRIFT | `Your database has changes MIRA did not make. MIRA left them exactly as they are and will keep working. They are listed below; if you did not make them, report them. Run: mira status to see this list again.` |
| S-CUSTOM | `MIRA found <n> files you changed in your install. A copy of every one is safe at <dir>. This update replaces MIRA's program files and keeps your data. To put your changes back, open RESTORE_MY_CUSTOM_BUILD.txt in that folder, or ask MIRA to put them back.` |
| S-RECONCILE | `MIRA updated its own settings file: <n> settings removed, <m> added with their defaults.` (printed only when either count is nonzero) |
| S-PGFLOOR | `MIRA needs PostgreSQL 17 or newer; this machine has <major>.` then, on Debian, the cluster on port 5432 named, then the exact install/upgrade command for the branch (and on Debian the `pg_dropcluster`/`pg_ctlcluster` command), then `Then re-run the installer.` |
| S-PGNONE | `No PostgreSQL server was found on this machine. Install PostgreSQL 17, then re-run the installer.` |
