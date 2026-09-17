#!/bin/bash
# deploy/vm/vm-exec.sh — run a shell command inside the VM as ROOT via the
# qemu-guest-agent (works on a keyless fresh boot; libvirt modes only).
#
# Usage: vm-exec.sh [flags] 'command'
#   flags as oneshot.sh (--host/--domain/...). Plain-ssh mode has no
#   guest-agent — use: ssh $VM_USER@<ip> 'sudo -n <command>'.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/lib.sh"
CMD=""
while [ $# -gt 0 ]; do
  case "$1" in
    --*) f="${1#--}"; case "$f" in
      host|domain|vm-user|snap-dir|vmimg-dir|template|xml|pubkey|ip|vm-pass) shift; set_common "$f" "${1:?--$f needs a value}" ;;
      *) echo "unknown flag --$f" >&2; exit 2 ;; esac ;;
    *) CMD="$1" ;;
  esac
  shift
done
finish_flags
[ -n "$CMD" ] || { echo "usage: vm-exec.sh [flags] 'command'" >&2; exit 2; }
[ "$IS_LIBVIRT" = 1 ] || { echo "FATAL: vm-exec needs libvirt; use plain ssh sudo in --ip mode" >&2; exit 1; }
vmexec "$CMD"
