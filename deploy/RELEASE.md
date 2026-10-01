# Stable Release Procedure

The installer is wired to the tagged tarball: `deploy/deploy.sh` defines the
constant `RELEASE_TAG` (currently `v2026.06.25`) and exports it, and every
fetch site pins to it — the README's curl entrypoint fetches
`deploy/deploy.sh` from that tag, the bootstrap clone passes
`--branch "$RELEASE_TAG"`, and `deploy/python.sh` downloads
`refs/tags/${RELEASE_TAG}.tar.gz` instead of the main branch. `RELEASE_TAG`
is the switch: cut a stable release by moving that one constant in
`deploy/deploy.sh` and updating the tag in the README install command, then
tagging the repo.

The tarball carries its own `VERSION` file, which is what gets copied into
`/opt/mira/app` — the installed code and the reported version always come
from the same tagged tree.

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
(`v2026.06.25` extracts to `mira-2026.06.25/`), which is why the wired
installer derives the path as `mira-${RELEASE_TAG#v}`.
