#!/bin/bash
# extract-stage-vm.sh — stage all sarcophagus components under /tmp/sarc-v2 (in the VM).
VM_USER="${VM_USER:-ubuntu}"
# Run as user `$VM_USER` over ssh. Uses sudo -n only for postgres operations.
# Live services are NOT touched: pg_dump is transactional, sqlite uses the
# consistent .backup() API, vault storage has been quiescent since provision.
set -euo pipefail
STAGE="${SARC_STAGE:-/tmp/sarc-v2}"
APP=/opt/mira/app
VENV=$APP/venv

rm -rf "$STAGE"
mkdir -p "$STAGE"/{postgres,systemd,data-users,system-info}

echo "== 1. postgres dump (pg_dump -Fc, transactional) =="
sudo -n -u postgres pg_dump -Fc mira_service > "$STAGE/postgres/mira_service.dump"
TOC=$(sudo -n -u postgres pg_restore --list "$STAGE/postgres/mira_service.dump" | grep -c "TABLE DATA")
echo "pg_restore --list OK — $TOC TABLE DATA entries"
[ "$TOC" -ge 20 ] || { echo "FATAL: dump looks truncated (<20 TABLE DATA)" >&2; exit 1; }

echo "== 2. data-users: copy tree, then consistent sqlite snapshots =="
cp -a "$APP/data" "$STAGE/data-users/data"
mkdir -p "$STAGE/sqlsnap"
$VENV/bin/python3 - <<'PYEOF'
import os, sqlite3
src_root = "/opt/mira/app/data"
dst_root = "/tmp/sarc-v2/sqlsnap"
n = 0
for dirpath, dirs, files in os.walk(src_root):
    for fn in files:
        if not fn.endswith(".db") or fn.endswith("-wal") or fn.endswith("-shm"):
            continue
        src = os.path.join(dirpath, fn)
        rel = os.path.relpath(src, src_root)
        dst = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        s = sqlite3.connect("file:%s?mode=ro" % src, uri=True)
        d = sqlite3.connect(dst)
        s.backup(d)
        d.close(); s.close()
        print("  snapshot:", rel)
        n += 1
print("  db snapshots taken:", n)
assert n > 0, "no sqlite dbs found — data layout changed, investigate"
PYEOF
# replace live dbs in the staged copy with consistent snapshots; drop wal/shm
find "$STAGE/sqlsnap" -name '*.db' | while read -r f; do
  rel=${f#"$STAGE/sqlsnap/"}
  cp "$f" "$STAGE/data-users/data/$rel"
  rm -f "$STAGE/data-users/data/$rel-wal" "$STAGE/data-users/data/$rel-shm"
done
rm -rf "$STAGE/sqlsnap"
tar -C "$STAGE/data-users" -czf "$STAGE/data-users.tar.gz" data
rm -rf "$STAGE/data-users"

echo "== 3. vault (file storage + init keys; quiescent since provision) =="
tar -C /opt -czf "$STAGE/vault.tar.gz" vault

echo "== 4. app code (source + llm logs; no venv/data/pycache) =="
tar -C /opt/mira -czf "$STAGE/app-code.tar.gz" \
  --exclude='app/venv' --exclude='app/data' \
  --exclude='__pycache__' --exclude='*.pyc' \
  app logs

echo "== 5. systemd units + configs =="
for u in mira vault vault-unseal valkey; do
  cp "/etc/systemd/system/$u.service" "$STAGE/systemd/"
done
# /etc/valkey is root:valkey 0750 — $VM_USER cannot read it; go through sudo (passwordless in-VM)
sudo -n cat /etc/valkey/valkey.conf > "$STAGE/systemd/valkey.conf"
# loud completeness check — a silent gap here broke the first production inject
for f in mira vault vault-unseal valkey; do
  [ -s "$STAGE/systemd/$f.service" ] || { echo "FATAL: unit $f.service missing from stage" >&2; exit 1; }
done
[ -s "$STAGE/systemd/valkey.conf" ] || { echo "FATAL: valkey.conf missing from stage" >&2; exit 1; }
ls /etc/systemd/system/ | grep -E 'mira|vault|valkey' > "$STAGE/systemd/unit-list.txt"

echo "== 6. home-ubuntu (ssh + instance credentials) =="
tar -C /home -czf "$STAGE/home-ubuntu.tar.gz" \
  --exclude="$VM_USER/.cache" \
  $VM_USER/.ssh $VM_USER/MIRA_credentials.txt $VM_USER/.vault-token

echo "== 7. system-info (OS + package inventory for reprovision) =="
{ head -2 /etc/os-release; uname -a; } > "$STAGE/system-info/os-kernel.txt"
dpkg --get-selections | grep -w install$ > "$STAGE/system-info/dpkg-selections.txt"
$VENV/bin/python3 -m pip freeze > "$STAGE/system-info/pip-freeze.txt"
{
  sudo -n -u postgres psql --version
  $VENV/bin/python3 --version
  valkey-server --version 2>/dev/null || true
  vault --version 2>&1 || true
  echo "-- databases --"
  sudo -n -u postgres psql -tAc "SELECT datname FROM pg_database WHERE NOT datistemplate"
} > "$STAGE/system-info/versions.txt"
sudo -n -u postgres psql -d mira_service -c '\dt' > "$STAGE/system-info/pg-tables.txt"
sudo -n -u postgres pg_dump -s mira_service > "$STAGE/system-info/pg-schema.sql"

echo "== 8. checksums =="
cd "$STAGE"
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS
du -sh "$STAGE"
echo "== staging complete =="
