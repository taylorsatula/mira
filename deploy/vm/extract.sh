#!/bin/bash
# deploy/vm/extract.sh — sarcophagus the live MIRA instance on a deployed VM
# into a new sealed snapshot dir (append-only; refuses to overwrite).
#
# Usage: extract.sh [flags] <new-sarcophagus-name>
#   Modes as oneshot.sh (local libvirt default / --host remote / --ip plain ssh).
#   The sarcophagus materializes on the CALLER under --snap-dir (remote mode:
#   on the machine running this script).
#
# Gates: mira healthy, no turn in flight (valkey user_lock), dump verified with
# pg_restore --list, sqlite via consistent .backup() snapshots — then
# VM↔caller sha256 byte parity, local sqlite integrity verify, SNAPSHOT-FACTS,
# MANIFEST. The README prose is copied from the newest existing sarcophagus or
# the toolkit README pointer — review + edit it, then regen the manifest.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/lib.sh"

NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --*) f="${1#--}"; case "$f" in
      host|domain|vm-user|snap-dir|vmimg-dir|template|xml|pubkey|ip|vm-pass) shift; set_common "$f" "${1:?--$f needs a value}" ;;
      *) echo "unknown flag --$f" >&2; exit 2 ;; esac ;;
    *) NAME="$1" ;;
  esac
  shift
done
finish_flags
[ -n "$NAME" ] || { echo "usage: extract.sh [flags] <new-name>" >&2; exit 2; }
DEST="$SNAP_DIR/$NAME"
[ -e "$DEST" ] && { echo "FATAL: $DEST exists — never overwrite a sarcophagus" >&2; exit 1; }

echo "== preflight: health + turn gate =="
IP=$(vmip)
H=$(vmssh "$IP" 'curl -s --max-time 8 http://127.0.0.1:1993/v0/api/health || true')
echo "$H" | grep -q '"status":"healthy"' || { echo "FATAL: mira not healthy: $H" >&2; exit 1; }
turn_lock_gate || exit 1

echo "== ssh bootstrap =="
bootstrap_ssh

echo "== stage inside VM =="
vmssh "$IP" 'rm -f /tmp/extract-stage-vm.sh'
vmcp_to "$IP" "$HERE/extract-stage-vm.sh" /tmp/
vmssh "$IP" "VM_USER=$VM_USER bash /tmp/extract-stage-vm.sh"

echo "== pull =="
mkdir -p "$DEST"
ssh_dir_copy() {  # recursive pull through the appropriate transport
  local ip="$1"
  if [ -n "$JUMP" ] && [ "$IS_LIBVIRT" = 0 -o -n "$REMOTE_HOST" ]; then
    :  # scp handles -J via ProxyJump below
  fi
  scp -q -r -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
      ${JUMP:+-o ProxyJump="$JUMP"} "$VM_USER@$ip:/tmp/sarc-v2/*" "$DEST/"
}
ssh_dir_copy "$IP"

echo "== byte-parity VM↔caller =="
(cd "$DEST" && sha256sum -c <(vmstream "$IP" 'cat /tmp/sarc-v2/SHA256SUMS') >/dev/null) \
  || { echo "FATAL: VM↔caller byte-parity failed" >&2; exit 1; }
echo "parity OK"

echo "== caller-side sqlite verification =="
VT=$(mktemp -d)
tar -xzf "$DEST/data-users.tar.gz" -C "$VT"
python3 - "$VT" <<'PYEOF'
import sqlite3, glob, sys
root = sys.argv[1]
dbs = glob.glob(root + '/**/*.db', recursive=True)
assert dbs, "no sqlite dbs in extracted data-users.tar.gz"
for db in dbs:
    c = sqlite3.connect(db)
    r = c.execute("PRAGMA integrity_check").fetchone()[0]
    assert r == "ok", f"integrity_check failed for {db}: {r}"
    print(f"  {db.replace(root + '/', '')}: integrity ok")
    c.close()
print("sqlite verification OK")
PYEOF
rm -rf "$VT"

echo "== SNAPSHOT-FACTS.txt =="
{
  echo "# SNAPSHOT-FACTS — generated $(date -u '+%Y-%m-%d %H:%M UTC') by extract.sh"
  echo "# domain: $DOMAIN   vm-ip: $IP   vm-user: $VM_USER   mode: $([ -n "$REMOTE_HOST" ] && echo "remote($REMOTE_HOST)" || { [ -n "$VMIP" ] && echo "plain-ssh" || echo "local"; })"
  echo "mira_version: $(vmssh "$IP" 'cat /opt/mira/app/VERSION 2>/dev/null || echo unknown')"
  for t in users messages memories api_tokens model_configs; do
    echo "pg_$t: $(vmssh "$IP" "sudo -n -u postgres psql -d mira_service -tAc 'SELECT count(*) FROM $t'" | tr -d '[:space:]')"
  done
  echo "continuum_id: $(vmssh "$IP" "sudo -n -u postgres psql -d mira_service -tAc 'SELECT id FROM continuums'" | tr -d '[:space:]')"
  echo "user_id: $(vmssh "$IP" 'ls /opt/mira/app/data/users/' | head -1)"
  echo "pg_tables: $(vmssh "$IP" "sudo -n -u postgres psql -d mira_service -tAc \"SELECT count(*) FROM pg_tables WHERE schemaname = current_schema\"" | tr -d '[:space:]')"
  echo "sqlite_domaindocs: $(vmssh "$IP" "sudo -n -u $VM_USER /opt/mira/app/venv/bin/python3 -c \"import sqlite3,glob; db=glob.glob('/opt/mira/app/data/users/*/userdata.db')[0]; print(sqlite3.connect(db).execute('SELECT count(*) FROM domaindocs').fetchone()[0])\"" | tr -d '[:space:]')"
} > "$DEST/SNAPSHOT-FACTS.txt"
cat "$DEST/SNAPSHOT-FACTS.txt"

echo "== manifest =="
(cd "$DEST" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 \
  && sha256sum -c MANIFEST.sha256 >/dev/null) || { echo "FATAL: manifest self-check failed" >&2; exit 1; }
echo "manifest OK"

echo "== README =="
if [ -f "$HERE/README.md" ]; then
  { echo "# $NAME"
    echo
    echo "> **Sarcophagus extracted $(date '+%Y-%m-%d %H:%M %Z') by deploy/vm/extract.sh."
    echo "> Facts: SNAPSHOT-FACTS.txt (authoritative). Contents contract + restore"
    echo "> instructions: deploy/vm/README.md in the mira-OSS repo. EDIT THIS HEADER"
    echo "> into a real orientation (what changed since the previous sarcophagus).**"
  } > "$DEST/README-RESTORE.md"
fi

echo "== VM staging cleanup =="
vmssh "$IP" 'rm -rf /tmp/sarc-v2 /tmp/extract-stage-vm.sh'

echo "== DONE: $DEST =="
du -sh "$DEST"
