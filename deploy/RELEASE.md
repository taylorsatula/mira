# Stable Release Procedure

Plain-English overview for maintainers: `docs/MAINTAINER_UPGRADE_GUIDE.md`.

The installer is wired to the release asset: `deploy/deploy.sh` defines the
constant `RELEASE_TAG` (currently `v2026.10.21-2.1.2`) and exports it, and
`deploy/python.sh` downloads that release's asset instead of the main branch.
`RELEASE_TAG` is the switch: cut a stable release by moving that one constant
in `deploy/deploy.sh`, tagging the repo, building the asset, and publishing.

The stable entrypoint is the root `install.sh`. It resolves the newest
published release from GitHub's `releases/latest` redirect, clones that tag,
and hands off to its `deploy/deploy.sh`. The README pipes `install.sh` from
`main`; only cut releases point the installer at a real tag, and a
`RELEASE_TAG` on `main` whose release has no asset fails the installer's
download until the asset is attached.

## Release artifact

The release asset is the only payload installs and updates fetch:
`mira-<version>.tar.gz` (the tag without its leading `v`), attached to the
GitHub release, served at
`https://github.com/<repo>/releases/download/<tag>/mira-<version>.tar.gz`
(`<repo>` is `deploy/GITHUB_REPO`). GitHub's auto-generated source archive is
never fetched — it carries no `RELEASE_COMMIT`.

Build it from a checkout that has the tag:

```sh
LOUD_MODE=false
source deploy/lib/output.sh
source deploy/lib/services.sh
source deploy/lib/repo_identity.sh
source deploy/lib/provenance.sh
build_release_artifact v<version> /tmp/release
```

`build_release_artifact` `git archive`s the tag into a single top-level
`mira-<version>/` directory, writes the tagged commit's SHA into
`RELEASE_COMMIT`, and refuses to produce an asset when:

- the tag itself tracks a `RELEASE_COMMIT` file (it is injected at build
  time, never committed);
- `install.sh`'s `REPO=` or `deploy/deploy.sh`'s bootstrap clone URL differs
  from `deploy/GITHUB_REPO` (the two places the repository is spelled
  literally, because each runs before any checkout exists);
- the database step chain fails `python3 -m scripts.migrate lint`;
- any `utils/release_identity.py:verify_release_identity` check fails: the
  tag equals `v` + `VERSION`, the asset's `VERSION` equals the repo's, and the
  asset's `RELEASE_COMMIT` equals the commit the tag points at.

Attach the printed file to the GitHub release. Without it, `mira update` and
fresh installs of that release fail for everyone.

## Provenance

Every install tree is a git repository whose official commits name their
source exactly:

- `install <tag> @ <sha>` — written by `deploy/python.sh`. The SHA comes from
  the asset's `RELEASE_COMMIT`; for `--local` it is the checkout's `HEAD`,
  suffixed ` local-dirty` when the checkout had uncommitted changes, and the
  subject ends ` (local tree)`.
- `update <tag> @ <sha>` — written by `deploy/update.sh`. A tree carrying no
  `RELEASE_COMMIT` (one a pre-mechanism `mira update` fetched from GitHub's
  auto-archive) is resolved with `resolve_release_sha <tag>` — the tag as
  GitHub lists it.
- `update nightly-<12hex> @ <sha>` — `mira update --nightly`, the SHA being
  the resolved main head.

The newest such commit is the baseline `mira update` diffs against to find
the operator's own changes (`deploy/lib/custom_export.sh`).

## Updating an installed MIRA

Deployed installs update themselves with `mira update`: the TUI subcommand
(`ui/tui/update.py`) resolves the newest published release, fetches its asset,
and hands the machine work to that release's own `deploy/update.sh`. It shows
what it will do and asks once before changing the database, takes a backup
first, runs the release's database steps (`deploy/migrations/`), and puts the
install back — code, venvs, and the database — if anything fails. `mira
restore <STAMP>` restores a backup by hand.

A schema change ships as two edits in one commit: the change in
`deploy/mira_service_schema.sql` (fresh installs) and a numbered step in
`deploy/migrations/` that brings an installed database to the same structure
(authoring rules: `deploy/migrations/AGENTS.md`). A shipped step is never
edited — the runner refuses a changed checksum.

## Release gate

A release is not published unless every condition holds:

| # | Condition | Mechanism |
|---|---|---|
| a | `VERSION` ↔ tag ↔ commit | `verify_release_identity` inside `build_release_artifact` (no asset on failure) |
| b | step-chain integrity | `scripts.migrate lint` inside `build_release_artifact` (no asset on failure) |
| c | convergence | `python3 -m scripts.migrate generate` succeeds: every origin in `utils/schema_migrations.py:SCHEMA_ORIGINS` plus the chain reaches exactly the fresh-install structure (it prints `DO NOT PUBLISH` and writes nothing otherwise) |
| d | skip-chain | one update from the oldest supported release lands on the same structure and ledger as walking release by release (VM check) |
| e | restore round trip | an update whose step fails restores the database and the install exactly (VM check) |

Checklist, in order:

1. Move `RELEASE_TAG`, set `VERSION`, commit.
2. Regenerate: `PATH="$(source deploy/lib/output.sh; source deploy/lib/services.sh; pg_bindir):$PATH" PYTHONPATH=. python3 -m scripts.migrate generate --superuser <a local superuser>`. It rewrites the generated ledger DDL and `deploy/migrations/BASELINE` (rows are append-only — it refuses a result that does not extend the checked-in file) plus `HEAD_FINGERPRINT.json`. Commit any change.
3. `install.sh`'s `REPO=` equals `deploy/GITHUB_REPO` (the builder refuses otherwise).
4. Every new step carries a NOTICE, or ships without one deliberately (the plan then shows the generic sentence).
5. Tag, build and attach the artifact (above).
6. Run gate rows d and e on a VM and record the result in the release notes.
7. Shipped steps are untouched since their release (`git diff <previous-tag> -- deploy/migrations/V*` shows additions only).

Nightly (`mira update --nightly`) records `update nightly-<12hex> @ <sha>`
commits and `applied_by` = the nightly label in the database ledger; the gate
does not apply.

## Manual recovery

Installing a release's code by hand (e.g. recovering an install):

```sh
cd /tmp
wget -q https://github.com/taylorsatula/mira/releases/download/vX.XX/mira-X.XX.tar.gz
tar -xzf mira-X.XX.tar.gz -C /tmp
sudo tar -C /tmp/mira-X.XX -cf - . | sudo tar -C /opt/mira/app -xf -
rm -rf /tmp/mira-X.XX.tar.gz /tmp/mira-X.XX
```

Replace `X.XX` with the release version (the tag without its leading `v`;
the asset extracts to `mira-<version>/`).
