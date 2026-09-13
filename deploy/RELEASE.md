# Stable Release Procedure

During active development, `deploy/python.sh` downloads MIRA from the main
branch. When cutting a stable release, switch the installer to the tagged
tarball instead:

```sh
cd /tmp
wget -q -O mira-X.XX.tar.gz https://github.com/taylorsatula/mira-OSS/archive/refs/tags/X.XX.tar.gz
tar -xzf mira-X.XX.tar.gz -C /tmp
sudo cp -r /tmp/mira-OSS-X.XX/* /opt/mira/app/
rm -f /tmp/mira-X.XX.tar.gz
rm -rf /tmp/mira-OSS-X.XX
```

Replace `X.XX` with the release tag (this procedure was previously preserved
as a comment block in `deploy/python.sh`; it moved here so the runbook lives
with the rest of the deployment documentation).
