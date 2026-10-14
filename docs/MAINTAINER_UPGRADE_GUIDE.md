# Maintainer guide: shipping releases that upgrade cleanly

What you, the maintainer, have to do so that `mira update` keeps working for
users. Users do nothing manual on a normal release; you have a few habits per
database change and a checklist per release.

**Status:** this describes the update mechanism specified in
`docs/IMPLEMENT_GREENFIELD_UPGRADE_FUNCTIONALITY.md`. Until that runbook has
been implemented, none of the files, commands, or checks below exist yet.
After it lands, `deploy/RELEASE.md` holds the authoritative release checklist,
`deploy/migrations/AGENTS.md` the step-authoring rules; this guide is the
plain-English overview that points at them.

---

## How an update works (so user reports make sense)

When a user runs `mira update`:

1. MIRA downloads the release file you attached to the GitHub release.
2. It builds the new Python environments (or reuses the old ones when nothing
   changed), while MIRA keeps running.
3. It works out what the update will do to the database and, if anything, shows
   the user a plain-English list and asks **Y/N**. Nothing has changed yet. On
   Linux it may ask for their admin password once.
4. It stops MIRA and saves a copy of every file the user changed — to
   `~/Desktop/mira-custom-<date>/` — plus an exact snapshot inside the install's
   git history. If saving fails, it stops and puts everything back.
5. It replaces the program files, backs up the database, runs your database
   steps, restarts MIRA, and checks that it's healthy.
6. If anything fails after the swap, it restores the database backup, puts the
   old code and files back exactly, and restarts the old version.

At startup, MIRA refuses to run on a database that doesn't match its code
(behind, ahead, or missing a step), and tells the user to run `mira update`.

---

## Every time you change the database structure

A "database structure change" is anything that alters tables, columns,
indexes, constraints, functions, policies, or grants.

**1. Make the change in two places.**

- `deploy/mira_service_schema.sql` — what brand-new installs get.
- A new step file in `deploy/migrations/` — what existing installs run.

The release checks fail if these two disagree, so you can't forget one
silently.

**2. Name the step with the next number.** Numbers are dense and never reused:
if the last step is `V0007__…`, yours is `V0008__short_description.sql`.
Lowercase, underscores.

**3. Write it so it can run twice safely.** Use `IF NOT EXISTS` / `IF EXISTS`.
The user's database may already have part of the change.

**4. Write the user-facing notice.** One to three plain sentences: what changes
for them, and that their data is safe. No jargon — the words "schema",
"migration", "ledger", "revision", "checksum", "fingerprint", "adoption",
"baseline" are rejected by the language check.

Example:

```sql
-- store: postgres
-- notice: Speeds up searching your past conversations. Nothing in your data
-- changes.
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
```

**5. Never edit a step after it's merged to `main`.** Nightly users run it the
moment it merges. If it's wrong, add a new step that fixes it. Editing a
shipped step makes every install that already ran it refuse its next update.

**6. Removing or renaming something takes two releases.** First release: add
the new thing and start using it. A later release: remove the old thing, in a
step whose header says `-- expand-contract: contract-for V<the earlier step>`.
The lint check enforces the header.

**Things a step must never do** (the lint check catches them):

- Change the user's model settings (`model_configs` model, endpoint, key name,
  effort, token limits) or touch `embedding_config`.
- Hardcode a `vector(<number>)` size — read it from the install at run time.
- Depend on the date, the clock, or MIRA's version number.
- Change the allowed `dialect_name` values without the
  `-- dialect-twins: DialectName, DIALECT_NAMES` header (the Python lists in
  `clients/llm/types.py` must change in the same commit).

**Reference data** that should simply equal what the release says (like
`usage_pricing`) lives in a repeatable step, `R__name.sql`, which replaces the
whole table and re-runs whenever you edit the file. Users' hand edits to those
rows are overwritten — by design.

---

## Every release

1. Bump `VERSION`, commit, and create the tag (`v` + the `VERSION` value).
2. Build the release file:
   `deploy/lib/provenance.sh` → `build_release_artifact <tag> <dest-dir>`.
   It refuses to build if the version, tag, and commit don't line up, or if any
   database step breaks the authoring rules.
3. **Attach `mira-<version>.tar.gz` to the GitHub release.** Installs and
   updates fetch only this file; without it, `mira update` fails for everyone.
4. Regenerate the generated files and commit them if they changed:
   `deploy/migrations/BASELINE`, `deploy/migrations/HEAD_FINGERPRINT.json`, and
   the generated step-table blocks in `deploy/mira_service_schema.sql` and
   `deploy/bootstrap_adopt.sql`.
5. Run the three release probes and keep their `PROBE PASSED` lines:
   - convergence (the schema file equals "old releases + all steps"),
   - skip-chain (jumping several releases at once ends the same as one at a
     time),
   - restore round-trip (a backup really restores).
6. Read the convergence probe's **release-tag list**. Any tag marked
   `UNMATCHED` is a release whose users would be refused as "modified by hand".
   Expected for pre-2.0 tags; unexpected for anything newer.
7. Check `install.sh`'s `REPO=` line still equals `deploy/GITHUB_REPO`.
8. Any failed check means **do not publish**.

Nightly builds skip this gate — which is why "never edit a merged step"
matters most there.

---

## When a user reports a problem

| User sees | What it means | What to do |
|---|---|---|
| "Your database's structure is not one this version of MIRA recognizes" + a list | Their database was changed by hand (or is from a pre-2.0 install) | Review the list with them. If the differences are theirs and harmless: `mira migrate --adopt-drifted` (recorded permanently). Pre-2.0: reinstall per `deploy/HOW_TO_MIGRATE_OLD_INSTALLS.txt` |
| "Your database has changes MIRA did not make" (after a successful update) | Hand-made database changes exist; MIRA left them alone | Usually nothing. If one collides with a future step, write a step that handles both shapes |
| "Step N (...) was changed after MIRA ran it" | A shipped step was edited | Your release bug. Restore the original step file and fix forward with a new step; re-release |
| "MIRA's database is at step N, newer than this version" | They installed a nightly, then a stable release older than it | They should stay on nightly or wait for a stable release past step N |
| "one conversation has N live segment boundaries" | Data that a step refuses to change automatically | Help them remove the duplicate; re-run `mira update` |
| "The update could not finish, so MIRA put your install back" | Something failed; fully rolled back | Get the log path from the message; fix; re-release |
| "MIRA could not put your data back automatically" + commands | Restore itself failed; MIRA is stopped | Walk them through the printed commands; backup is under `/opt/mira/backups/<date>/` |
| "MIRA found N files you changed" | Their custom files are saved on the Desktop | Nothing. `RESTORE_MY_CUSTOM_BUILD.txt` in that folder tells an agent how to reapply them |
| "needs a terminal" / exit code 3 | The update needs a Y/N or a password and ran without a terminal | Run `mira update` from an interactive terminal |

User-side recovery commands: `mira status` (what's installed and changed),
`mira restore <date>` (put a database backup back), `mira update` (retry).

---

## Things that stay manual by design

- **Major-version jumps** (like pre-2.0 → 2.0) are not handled by the step
  chain; each needs its own plan when it comes.
- **Writing steps and their notices** — the judgment part.
- **Reviewing the unmatched-release list** at each cut.
