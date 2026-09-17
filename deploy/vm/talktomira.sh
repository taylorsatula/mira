#!/usr/bin/env bash
# talktomira.sh — one-liner chat with the MIRA instance running in the libvirt VM.
#
#   ./talktomira.sh --message "Howdy whats up"
#
# Handles the whole boilerplate: resolves the VM's current DHCP address,
# opens a local session, mints an API token (CSRF dance included), sends the
# message to /v0/api/chat, and prints the reply text. Long turns block their
# HTTP call — we wait up to 15 min and stream nothing back until it lands.
#
# Environment overrides:
#   MIRA_HOST   ssh target of the libvirt host (default admin@192.168.1.9)
#   MIRA_DOMAIN libvirt domain to resolve (default ubuntu_vm)
#   MIRA_VM_IP  skip domifaddr resolution if you already know the IP
#
# Options:
#   -m, --message TEXT   the message to send (required)
#       --host USER@HOST  override MIRA_HOST for this call
#       --raw             print the full JSON reply instead of just the text
#   -h, --help           this help
#
# Gotchas encoded here (see deploy/vm/README.md): VM DHCP addresses change
# with every spawned MAC — never hardcode; cookie-auth POSTs need
# X-CSRF-Token while Bearer requests don't; never shrink --max-time.

set -euo pipefail

HOST="${MIRA_HOST:-admin@192.168.1.9}"
DOMAIN="${MIRA_DOMAIN:-ubuntu_vm}"
VM_IP="${MIRA_VM_IP:-}"
MSG=""
RAW="no"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--message) MSG="${2:?--message requires text}"; shift 2 ;;
        --host)       HOST="$2"; shift 2 ;;
        --raw)        RAW="yes"; shift ;;
        -h|--help)    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

if [[ -z "$MSG" ]]; then
    echo "error: --message is required (see --help)" >&2
    exit 2
fi

# Resolve the VM's current IP on the host (DHCP changes every spawn).
if [[ -z "$VM_IP" ]]; then
    VM_IP=$(ssh -n "$HOST" "virsh domifaddr $DOMAIN" \
        | grep -o '192[0-9.]*' | head -1)
    if [[ -z "$VM_IP" ]]; then
        echo "error: could not resolve an IP for domain '$DOMAIN' on $HOST — is the VM running?" >&2
        exit 1
    fi
fi

# Base64 carries the message through both ssh and shell quoting unharmed.
MSG_B64=$(printf '%s' "$MSG" | base64 | tr -d '\n')

# Everything below runs on the libvirt host, against the VM's API.
# NOTE: no ssh -n here — the heredoc IS stdin (gotcha 1: -n is for command
# execution only, never pipe/stream transfers).
ssh "$HOST" VM_IP="$VM_IP" MSG_B64="$MSG_B64" RAW="$RAW" 'bash -s' <<'REMOTE'
set -euo pipefail
BASE="http://${VM_IP}:1993"
JAR=$(mktemp)
trap 'rm -f "$JAR"' EXIT

curl -s --max-time 10 -c "$JAR" "$BASE/v0/auth/local/session" >/dev/null

CSRF=$(curl -s --max-time 10 -b "$JAR" -c "$JAR" -X POST "$BASE/v0/auth/csrf" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"]["csrf_token"])')

TOKEN=$(curl -s --max-time 10 -b "$JAR" -H "X-CSRF-Token: $CSRF" \
    -H 'Content-Type: application/json' \
    -d "{\"name\":\"talktomira-cli-$(date +%s)\"}" -X POST "$BASE/v0/auth/api-tokens" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"]["token"])')

PAYLOAD=$(MSG_B64="$MSG_B64" python3 -c '
import base64, json, os
print(json.dumps({"message": base64.b64decode(os.environ["MSG_B64"]).decode()}))')

if [[ "$RAW" = "yes" ]]; then
    curl -s --max-time 900 -H "Authorization: Bearer $TOKEN" \
        -H 'Content-Type: application/json' -d "$PAYLOAD" \
        -X POST "$BASE/v0/api/chat"
    echo
else
    echo "waiting for reply (long turns block; up to 15 min)..." >&2
    curl -s --max-time 900 -H "Authorization: Bearer $TOKEN" \
        -H 'Content-Type: application/json' -d "$PAYLOAD" \
        -X POST "$BASE/v0/api/chat" \
    | python3 -c '
import sys, json
r = json.load(sys.stdin)
if not r.get("success"):
    sys.exit(f"error: {r}")
d = r["data"]
print(d["response"])
m = d.get("metadata") or {}
bits = []
if m.get("tools_used"): bits.append("tools: " + ", ".join(m["tools_used"]))
if m.get("surfaced_memories"): bits.append("memories: " + str(len(m["surfaced_memories"])))
if m.get("processing_time_ms"): bits.append(str(m["processing_time_ms"]) + " ms")
if bits: print("  [" + " | ".join(bits) + "]", file=sys.stderr)
'
fi
REMOTE
