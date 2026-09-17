#!/bin/bash
# deploy/vm/inject-restore-vm.sh — restore a sarcophagus into THIS VM.
# Run as a normal user with passwordless sudo over ssh (the drivers do this);
# payload staged at $1 (default /tmp/inject-sarc). VM_USER (default ubuntu)
# remaps ownership + unit User= lines when the sarcophagus was captured on a
# VM with a different user (e.g. ubuntu → mira_service).
set -euo pipefail
SRC="${1:-/tmp/inject-sarc}"
VM_USER="${VM_USER:-ubuntu}"
APP=/opt/mira/app
VENV=$APP/venv

[ -f "$SRC/postgres/mira_service.dump" ] || { echo "FATAL: no dump in $SRC" >&2; exit 1; }

echo "== 0. turn-in-flight gate =="
LOCKS=$(valkey-cli --scan --pattern 'user_lock:*' 2>/dev/null || sudo -n valkey-cli --scan --pattern 'user_lock:*' || true)
if [ -n "$LOCKS" ]; then
  echo "FATAL: MIRA turn in flight ($LOCKS) — do not restore now." >&2; exit 1
fi

echo "== 1. stop mira =="
sudo -n systemctl stop mira 2>/dev/null || true

echo "== 2. app code (overlay live source; deletions are not removed by untar) =="
sudo -n tar -C /opt/mira -xzf "$SRC/app-code.tar.gz"
sudo -n chown -R "$VM_USER:$VM_USER" /opt/mira/app /opt/mira/logs

echo "== 3. postgres (drop + recreate + restore; --no-owner) =="
sudo -n -u postgres psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='mira_service' AND pid <> pg_backend_pid()" >/dev/null
sudo -n -u postgres psql -qc "DROP DATABASE IF EXISTS mira_service"
sudo -n -u postgres psql -qc "CREATE DATABASE mira_service OWNER postgres"
sudo -n -u postgres pg_restore -U postgres -d mira_service --no-owner "$SRC/postgres/mira_service.dump"

echo "== 4. user data (wholesale replace) =="
rm -rf "$APP/data"
tar -C "$APP" -xzf "$SRC/data-users.tar.gz"
sudo -n chown -R "$VM_USER:$VM_USER" "$APP/data"

echo "== 5. vault (wholesale replace; auto-unseal rides in vault-unseal.service) =="
sudo -n systemctl stop vault-unseal vault 2>/dev/null || true
sudo -n rm -rf /opt/vault
sudo -n tar -C /opt -xzf "$SRC/vault.tar.gz"
sudo -n chown -R "$VM_USER:$VM_USER" /opt/vault
sudo -n systemctl start vault vault-unseal

echo "== 6. home (instance credentials; remapped to $VM_USER) =="
TMP=$(mktemp -d)
tar -C "$TMP" -xzf "$SRC/home-ubuntu.tar.gz"
for f in MIRA_credentials.txt .vault-token; do
  [ -f "$TMP"/*/$f ] && sudo -n cp "$TMP"/*/$f "/home/$VM_USER/" && sudo -n chown "$VM_USER:$VM_USER" "/home/$VM_USER/$f"
done
rm -rf "$TMP"
# NOTE: the captured authorized_keys is deliberately NOT restored — the
# operator's own key access is already bootstrapped; credentials are the point.

echo "== 7. systemd units (User= remapped; valkey.conf; idempotent) =="
for u in "$SRC"/systemd/*.service; do
  sudo -n sed -i -e "s/^User=ubuntu\$/User=$VM_USER/" -e "s/^Group=ubuntu\$/Group=$VM_USER/" "$u"
  sudo -n cp "$u" /etc/systemd/system/
done
[ -f "$SRC/systemd/valkey.conf" ] && sudo -n cp "$SRC/systemd/valkey.conf" /etc/valkey/valkey.conf
sudo -n systemctl daemon-reload

echo "== 8. venv drift guard =="
$VENV/bin/python3 -m pip install -q -r "$APP/requirements.txt" 2>/dev/null \
  || echo "WARN: pip install failed — venv may still be fine if unchanged" >&2

echo "== 9. start mira =="
sudo -n systemctl start mira

echo "== 10. health poll (no fixed sleeps) =="
for i in $(seq 1 150); do
  H=$(curl -s --max-time 5 http://127.0.0.1:1993/v0/api/health || true)
  echo "$H" | grep -q '"status":"healthy"' && { echo "HEALTHY after ~$((i * 2))s: $H"; exit 0; }
  sleep 2
done
echo "FATAL: mira not healthy in 300 s — journal (WARNING+):" >&2
sudo -n journalctl -u mira -n 40 --no-pager >&2 || true
exit 1
