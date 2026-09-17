#!/bin/bash
# deploy/vm/oneshot.sh — one command: a VM with a dev-build MIRA and a chosen
# instance sarcophagus restored onto it, fully verified.
#
# Modes (auto-selected by flags):
#   libvirt local   (default)          — spawn/reuse domain from the frozen base
#   libvirt remote  --host user@host   — same, orchestrated over ssh from anywhere
#                                        (virsh executes on the host; no local
#                                        libvirt needed — macOS-safe)
#   plain ssh       --ip <addr>        — deploy + restore onto an EXISTING VM
#                                        (any hypervisor/cloud; use --vm-pass for
#                                        the one-time password bootstrap)
#
# Usage:
#   oneshot.sh [flags] <sarcophagus-name-or-path> [--fresh]
#   flags: --host U@H | --ip ADR | --vm-pass PW | --domain NAME | --vm-user U
#          --snap-dir DIR | --vmimg-dir DIR | --template FILE | --xml FILE
#          --source DIR (mira-OSS checkout; default: the repo containing this
#          script) | --config FILE (deploy config yml)
#
# Sarcophagus: a name under --snap-dir (remote mode: the HOST's snap dir, it is
# synced to the caller first) or any local path to a sealed sarcophagus dir.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/lib.sh"

SARC=""
FRESH=0
SRC="${SOURCE:-$(cd "$HERE/../.." && pwd)}"
CONFIG="${CONFIG:-$HERE/deploy-config-dev-vm.yml}"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --fresh) FRESH=1 ;;
    --source) shift; SRC="${1:?}" ;;
    --config) shift; CONFIG="${1:?}" ;;
    --loud|--quiet) ;;      # accepted, reserved
    --*) f="${1#--}"; case "$f" in
      host|domain|vm-user|snap-dir|vmimg-dir|template|xml|pubkey|ip|vm-pass) shift; set_common "$f" "${1:?--$f needs a value}" ;;
      *) echo "unknown flag --$f" >&2; exit 2 ;; esac ;;
    *) SARC="$1" ;;
  esac
  shift
done
finish_flags
[ -n "$SARC" ] || { echo "usage: oneshot.sh [flags] <sarcophagus> [--fresh]" >&2; exit 2; }
DISK="$VMIMG_DIR/$DOMAIN.qcow2"

echo "== 0. preflight =="
[ -f "$SRC/main.py" ] && [ -f "$SRC/deploy/deploy.sh" ] \
  || { echo "FATAL: --source '$SRC' is not a mira-OSS checkout" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "FATAL: deploy config missing: $CONFIG" >&2; exit 1; }
SARC=$(resolve_sarc "$SARC")
echo "mode: $([ "$IS_LIBVIRT" = 1 ] && echo "libvirt $([ -n "$REMOTE_HOST" ] && echo remote || echo local)" || echo "plain-ssh $VMIP")  domain/user: $DOMAIN/$VM_USER"
echo "sarcophagus: $SARC   source: $SRC"

echo "== 1. spawn =="
if [ "$IS_LIBVIRT" = 1 ]; then
  STATE=$(VIRSH domstate "$DOMAIN" 2>/dev/null) || STATE=""
  [ -n "$STATE" ] || STATE=undefined
  if [ "$STATE" = "running" ] && [ "$FRESH" = 1 ]; then
    turn_lock_gate || exit 1
    [ "$(readlink -f "$DISK" 2>/dev/null)" = "$(readlink -f "$TEMPLATE")" ] \
      && { echo "FATAL: live disk IS the template — refusing to clobber the base" >&2; exit 1; }
    echo "--fresh: shutting down $DOMAIN"
    VIRSH shutdown "$DOMAIN"
    for _ in $(seq 1 90); do [ "$(VIRSH domstate "$DOMAIN")" = "shut off" ] && break; sleep 2; done
    [ "$(VIRSH domstate "$DOMAIN")" = "shut off" ] || { echo "FATAL: shutdown timed out" >&2; exit 1; }
    [ -f "$DISK" ] && mv "$DISK" "$DISK.pre-oneshot-$(date +%Y%m%d-%H%M%S)"
    STATE=shutoff
  fi
  if [ "$STATE" != "running" ]; then
    [ -f "$TEMPLATE" ] || { echo "FATAL: base template missing: $TEMPLATE (see make-base-template.sh)" >&2; exit 1; }
    [ -f "$DISK" ] || { echo "fresh disk: copying base template"; cp --reflink=auto "$TEMPLATE" "$DISK"; }
    if [ "$STATE" = "undefined" ]; then
      echo "defining domain $DOMAIN from $VM_XML"
      sed -e "s|__DOMAIN__|$DOMAIN|g" -e "s|__DISK__|$DISK|g" "$VM_XML" > "/tmp/oneshot-$DOMAIN.xml"
      VIRSH define "/tmp/oneshot-$DOMAIN.xml"; rm -f "/tmp/oneshot-$DOMAIN.xml"
    fi
    VIRSH start "$DOMAIN"
  else
    echo "domain $DOMAIN running — reusing (--fresh to rebuild)"
  fi
  wait_guest_agent
else
  echo "plain-ssh mode: VM $VMIP must already exist and be reachable"
fi

IP=$(vmip)
echo "== 2. ssh bootstrap (VM at $IP) =="
bootstrap_ssh

echo "== 3. push worktree + config =="
vmssh "$IP" 'rm -rf ~/mira-OSS && mkdir ~/mira-OSS'
tar -C "$SRC" --exclude=.git -czf - . | vmstream "$IP" 'tar -C ~/mira-OSS -xzf -'
vmcp_to "$IP" "$CONFIG" 'deploy-config.yml'
echo "pushed $(du -sh "$SRC" | cut -f1) + config"

echo "== 4. deploy (dev build; log: VM /tmp/deploy.log) =="
# Rebuild semantics: gate on in-flight turns, stop mira, terminate backends,
# drop the DB (deploy is greenfield-only). Active connections silently block
# DROP DATABASE otherwise — that collision is what the 2>/dev/null would hide.
turn_lock_gate || exit 1
vmssh "$IP" 'sudo -n systemctl stop mira 2>/dev/null; sudo -n -u postgres psql -qc "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='mira_service' AND pid <> pg_backend_pid()" >/dev/null 2>&1; sudo -n -u postgres psql -qc "DROP DATABASE IF EXISTS mira_service" >/dev/null 2>&1; true' >/dev/null
vmssh "$IP" 'rm -f /tmp/deploy.exit /tmp/deploy.log; nohup bash -c "cd ~/mira-OSS && ./deploy/deploy.sh --config ~/deploy-config.yml --local --loud > /tmp/deploy.log 2>&1; echo \$? > /tmp/deploy.exit" >/dev/null 2>&1 & echo started'
T0=$(date +%s)
while [ "$(vmssh "$IP" 'test -f /tmp/deploy.exit && cat /tmp/deploy.exit || echo running')" = "running" ]; do
  [ $(( $(date +%s) - T0 )) -gt 3600 ] && { echo "FATAL: deploy exceeded 60 min" >&2; vmssh "$IP" 'tail -20 /tmp/deploy.log'; exit 1; }
  sleep 15
done
RC=$(vmssh "$IP" 'cat /tmp/deploy.exit')
[ "$RC" = 0 ] || { echo "FATAL: deploy failed (exit $RC) after $(( $(date +%s) - T0 ))s:" >&2; vmssh "$IP" 'tail -40 /tmp/deploy.log'; exit 1; }
echo "deploy finished OK in $(( $(date +%s) - T0 ))s"

echo "== 5. mira.service poll (placeholder-key deploys park the POST gate by
       design — inject installs real Vault/routes and restarts) =="
for _ in $(seq 1 60); do
  [ "$(vmssh "$IP" 'systemctl is-active mira 2>/dev/null')" = "active" ] && break
  sleep 2
done
[ "$(vmssh "$IP" 'systemctl is-active mira 2>/dev/null')" = "active" ] \
  || { echo "FATAL: mira.service not active after deploy" >&2; vmssh "$IP" 'journalctl -u mira -n 20 --no-pager; tail -20 /tmp/deploy.log'; exit 1; }
echo "mira.service active"

echo "== 6. inject sarcophagus + verify =="
"$HERE/inject.sh" $( [ -n "$REMOTE_HOST" ] && echo --host "$REMOTE_HOST" ) \
                   $( [ "$IS_LIBVIRT" = 0 ] && echo --ip "$VMIP" ${VMPASS:+--vm-pass "$VMPASS"} ) \
                   --domain "$DOMAIN" --vm-user "$VM_USER" "$SARC"

echo "== DONE: $DOMAIN at $IP — dev build + $SARC restored and verified =="
