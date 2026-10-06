# Stable Release Procedure

The installer is wired to the tagged tarball: `deploy/deploy.sh` defines the
constant `RELEASE_TAG` (currently `v2026.10.06-2.0`) and exports it, and
`deploy/python.sh` downloads `refs/tags/${RELEASE_TAG}.tar.gz` instead of the
main branch. `RELEASE_TAG` is the switch: cut a stable release by moving that
one constant in `deploy/deploy.sh`, then tagging the repo.

The stable entrypoint is the root `install.sh`. It resolves the newest
published release from GitHub's `releases/latest` redirect, fetches that tag's
`deploy/deploy.sh`, and hands off — so cutting a release is just moving
`RELEASE_TAG`, tagging, and publishing. The README pipes `install.sh` from
`main`; only cut releases point the installer at a real tag, and an unreleased
`RELEASE_TAG` on `main` will 404 until its tag exists.

The tarball carries its own `VERSION` file, which is what gets copied into
`/opt/mira/app` — the installed code and the reported version always come
from the same tagged tree.

## Breaking releases and `mira update`

Deployed installs update themselves with `mira update`: the TUI subcommand
(`tui/update.py`) resolves the newest published release, fetches its tarball,
and hands the machine work to that release's own `deploy/update.sh` (rebuild
venvs, stop service, swap code with the state-preserve set, restart,
health-poll on the new `VERSION`).

A release is **breaking** when its tree carries `BREAKING.md` at the repo
root. Presence of the file is the entire contract — `mira update` refuses the
release, prints the file verbatim, and points at the manual path; absence
means the release is safe to install in place. `deploy/update.sh` re-checks
the marker, so a direct invocation cannot bypass the gate.

Cutting a breaking release therefore means:

1. Make the change (typically a `mira_service_schema.sql` change — the
   database stays greenfield, so any schema change is breaking by definition;
   also anything else that makes an in-place swap unsafe).
2. Add `BREAKING.md` at the repo root in the same commit. Its content is
   operator-facing prose, printed verbatim by `mira update`: what broke, why
   in-place update is refused, and the manual path (reinstall via
   `install.sh` — the old database is renamed aside automatically and Vault
   credentials are preserved — then ask MIRA to bring its history forward
   itself with its bash tool; `deploy/HOW_TO_MIGRATE_OLD_INSTALLS.txt` and
   the `hosted-recovery` skill carry that runbook).
3. Remove `BREAKING.md` in the first subsequent release that is safe again —
   the marker describes a release's relationship to what precedes it, so it
   must not linger into non-breaking releases.

The manual procedure below is kept for reference (e.g. recovering an
install by hand):

```sh
cd /tmp
wget -q -O mira-X.XX.tar.gz https://github.com/taylorsatula/mira-OSS/archive/refs/tags/X.XX.tar.gz
tar -xzf mira-X.XX.tar.gz -C /tmp
sudo cp -r /tmp/mira-X.XX/* /opt/mira/app/
rm -f /tmp/mira-X.XX.tar.gz
rm -rf /tmp/mira-X.XX
```

Replace `X.XX` with the release tag (this procedure was previously preserved
as a comment block in `deploy/python.sh`; it moved here so the runbook lives
with the rest of the deployment documentation). Note that GitHub's tag
archive strips a leading `v` from the tag in the extracted directory name
(`v2026.10.05` extracts to `mira-2026.10.05/`), which is why the wired
installer derives the path as `mira-${RELEASE_TAG#v}`.
