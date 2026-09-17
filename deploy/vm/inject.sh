#!/bin/bash
# deploy/vm/inject.sh — restore a sealed instance sarcophagus onto a MIRA VM
# that has a deployed stack (post-deploy, or any running instance), then verify
# against the sarcophagus's SNAPSHOT-FACTS.txt.
#
# Usage: inject.sh [flags] <sarcophagus-name-or-path>
#   Runs standalone; also phase 6 of oneshot.sh. Same modes as oneshot
#   (--host remote libvirt / --ip plain ssh / local libvirt default).
#
# Restores: app code, Postgres (drop+recreate+pg_restore --no-owner), user data
# (sqlite snapshots), Vault (wholesale — real API keys ride in it), instance
# credentials, systemd units (User= remapped to the actual VM user), venv drift
# guard, mira restart, health poll, fact verification.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/lib.sh"

SARC=""
while [ $# -gt 0 ]; do
  case "$1" in
    --*) f="${1#--}"; case "$f" in
      host|domain|vm-user|snap-dir|vmimg-dir|template|xml|pubkey|ip|vm-pass) shift; set_common "$f" "${1:?--$f needs a value}" ;;
      *) echo "unknown flag --$f" >&2; exit 2 ;; esac ;;
    *) SARC="$1" ;;
  esac
  shift
done
finish_flags
[ -n "$SARC" ] || { echo "usage: inject.sh [flags] <sarcophagus>" >&2; exit 2; }

for f in postgres/mira_service.dump data-users.tar.gz app-code.tar.gz \
         vault.tar.gz home-ubuntu.tar.gz systemd/mira.service SNAPSHOT-FACTS.txt; do
  [ -f "$SARC/$f" ] && continue
  if [ -d "$SARC" ]; then echo "FATAL: $SARC/$f missing" >&2
  else SARC=$(resolve_sarc "$SARC") && continue; fi
  exit 1
done
echo "sarcophagus: $SARC"

IP=$(vmip)
echo "VM: $DOMAIN at $IP (user $VM_USER)"

echo "== quiesce mira early (gate, then stop before payload push) =="
turn_lock_gate || exit 1
vmssh "$IP" 'sudo -n systemctl stop mira' 2>/dev/null || true

echo "== push payload =="
STAGE=/tmp/inject-sarc
vmssh "$IP" "rm -rf $STAGE && mkdir -p $STAGE"
vmcp_to "$IP" "$SARC/postgres" "$SARC/data-users.tar.gz" "$SARC/app-code.tar.gz" \
               "$SARC/vault.tar.gz" "$SARC/home-ubuntu.tar.gz" "$SARC/systemd" "$STAGE/"

echo "== run restore inside VM =="
vmcp_to "$IP" "$HERE/inject-restore-vm.sh" /tmp/
vmssh "$IP" "VM_USER=$VM_USER bash /tmp/inject-restore-vm.sh $STAGE"

echo "== verify against SNAPSHOT-FACTS.txt =="
FAIL=0
while IFS=': ' read -r k v; do
  case "$k" in
    pg_*) t=${k#pg_}
      if [ "$t" = "tables" ]; then Q="SELECT count(*) FROM pg_tables WHERE schemaname = current_schema"
      else Q="SELECT count(*) FROM $t"; fi
      GOT=$(vmssh "$IP" "sudo -n -u postgres psql -d mira_service -tAc \"$Q\"" | tr -d '[:space:]')
      [ "$GOT" = "$v" ] && echo "  PASS $k: $GOT" || { echo "  FAIL $k: got $GOT want $v"; FAIL=1; } ;;
    sqlite_*) what=${k#sqlite_}
      GOT=$(vmssh "$IP" "sudo -n -u $VM_USER /opt/mira/app/venv/bin/python3 -c \"import sqlite3,glob; db=glob.glob('/opt/mira/app/data/users/*/userdata.db')[0]; print(sqlite3.connect(db).execute('SELECT count(*) FROM $what').fetchone()[0])\"" | tr -d '[:space:]')
      [ "$GOT" = "$v" ] && echo "  PASS $k: $GOT" || { echo "  FAIL $k: got $GOT want $v"; FAIL=1; } ;;
    mira_version)
      GOT=$(vmssh "$IP" 'cat /opt/mira/app/VERSION')
      [ "$GOT" = "$v" ] && echo "  PASS $k: $GOT" || { echo "  FAIL $k: got $GOT want $v"; FAIL=1; } ;;
  esac
done < "$SARC/SNAPSHOT-FACTS.txt"

echo "== cleanup staging =="
vmssh "$IP" "rm -rf $STAGE /tmp/inject-restore-vm.sh"

H=$(vmssh "$IP" 'curl -s --max-time 10 http://127.0.0.1:1993/v0/api/health || true')
echo "$H" | grep -q '"status":"healthy"' && echo "  PASS health: $H" || { echo "  FAIL health: $H"; FAIL=1; }
if [ "$FAIL" = 0 ]; then
  echo "== DONE: MIRA restored from $SARC — healthy, facts verified =="
else
  echo "== DONE WITH FAILURES — journal: vmssh journalctl -u mira -n 50 ==" >&2
  exit 1
fi
