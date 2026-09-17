# deploy/vm/lib.sh — shared configuration + transport wrappers for the MIRA
# VM toolkit. Source this file; do not execute directly.
#
# Two operating modes:
#   LOCAL  (default)  — this machine runs libvirt; virsh is local; VM ssh is direct.
#   REMOTE (--host U@H or VM_HOST=U@H) — orchestrate from anywhere with ssh:
#     all virsh commands execute ON the host (macOS has no virsh — do not use
#     qemu+ssh URIs); VM ssh/scp jump through the host (-J).
#
# Caller-overridable environment: VM_HOST, VM_DOMAIN, VM_USER, SNAP_DIR,
# VMIMG_DIR, TEMPLATE, VM_XML, SSH_PUB, VM_SSH_JUMP.
#
# Hard-won rules encoded here (do not "simplify" them away — each comment is a
# scar from a real failure; see deploy/vm/README.md § Gotchas):
#   - `ssh -n` for command execution (or ssh eats a caller's read-loop stdin),
#     plain ssh for pipe/stream transfers (or the producer SIGPIPEs).
#   - virsh domstate on a missing domain can return empty without failing the
#     || fallback — normalize explicitly.
#   - long MIRA turns block their HTTP call — always big --max-time or poll.

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
REMOTE_HOST="${VM_HOST:-}"          # user@host for --host mode; empty = local
JUMP="${VM_SSH_JUMP:-$REMOTE_HOST}" # ssh jump to reach the VM; usually the host
VM_USER="${VM_USER:-ubuntu}"
DOMAIN="${VM_DOMAIN:-ubuntu_vm}"
SNAP_DIR="${SNAP_DIR:-$HOME/mira-snapshots}"
VMIMG_DIR="${VMIMG_DIR:-$HOME/virtual_machine}"
TEMPLATE="${TEMPLATE:-$VMIMG_DIR/ubuntu_vm-template.qcow2}"
VM_XML="${VM_XML:-}"               # domain skeleton; default: base-vm.xml next to lib.sh
[ -n "$VM_XML" ] || VM_XML="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/base-vm.xml"

here() { cd "$(dirname "${BASH_SOURCE[0]}")" && pwd; }

# caller's public key for the VM bootstrap (guest-exec appends it)
pubkey() {
  if [ -n "${SSH_PUB:-}" ]; then cat "$SSH_PUB"
  elif [ -f "$HOME/.ssh/id_ed25519.pub" ]; then cat "$HOME/.ssh/id_ed25519.pub"
  elif [ -f "$HOME/.ssh/id_rsa.pub" ]; then cat "$HOME/.ssh/id_rsa.pub"
  elif command -v ssh-add >/dev/null && ssh-add -L 2>/dev/null | head -1; then :
  else echo "FATAL: no public key found (set SSH_PUB=<file>)" >&2; return 1
  fi
}

# ----------------------------------------------------------------------------
# Transports
# ----------------------------------------------------------------------------
# virsh — executes on the libvirt machine (host in remote mode). The remote
# path passes through an EXTRA shell hop (ssh joins args, the remote shell
# re-parses) — %q-escape every argument or inner quotes get eaten and virsh
# receives broken JSON ({execute:guest-ping} — a real failure this caused).
VIRSH() {
  if [ -n "$REMOTE_HOST" ]; then
    local _q _a=()
    for _q in "$@"; do printf -v _q '%q' "$_q"; _a+=("$_q"); done
    ssh -n "$REMOTE_HOST" virsh "${_a[@]}"
  else
    virsh "$@"
  fi
}

# run a command in the VM as $VM_USER over ssh (arg 1 = VM IP)
vmssh() {  # vmssh <ip> <command...>  — command execution (stdin-safe)
  local ip="$1"; shift
  ssh -n -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
      ${JUMP:+-J "$JUMP"} "$VM_USER@$ip" "$@"
}
vmstream() {  # vmstream <ip> <command...> — pipe/stream variant (NO -n)
  local ip="$1"; shift
  ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
      ${JUMP:+-J "$JUMP"} "$VM_USER@$ip" "$@"
}
vmcp_from() {  # vmcp_from <ip> <remote-path> <local-dir>
  scp -q -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
      ${JUMP:+-o ProxyJump="$JUMP"} "$1:$2" "$3"
}
vmcp_to() {  # vmcp_to <ip> <local-path...> <remote-dir>
  local ip="$1"; shift
  local last="${@: -1}"; local rest=("${@:1:$#-1}")
  scp -q -r -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
      ${JUMP:+-o ProxyJump="$JUMP"} "${rest[@]}" "$VM_USER@$ip:$last"
}

# qemu-guest-agent exec (runs as ROOT in the VM; works on keyless fresh boots —
# this is the bootstrap primitive). Usage: vmexec '<command>'
vmexec() {
  local payload pid out rc so se
  payload=$(jq -nc --arg c "$1" \
    '{"execute":"guest-exec","arguments":{"path":"/bin/sh","arg":["-c",$c],"capture-output":true}}')
  pid=$(VIRSH qemu-agent-command "$DOMAIN" "$payload" | jq -r '.return.pid')
  [ -n "$pid" ] && [ "$pid" != "null" ] || { echo "guest-exec failed to start" >&2; return 2; }
  for _ in $(seq 1 300); do
    out=$(VIRSH qemu-agent-command "$DOMAIN" \
      "{\"execute\":\"guest-exec-status\",\"arguments\":{\"pid\":$pid}}")
    [ "$(echo "$out" | jq -r '.return.exited')" = "true" ] && break
    sleep 0.2
  done
  rc=$(echo "$out" | jq -r '.return.exitcode')
  so=$(echo "$out" | jq -r '.return."out-data" // empty' | base64 -d)
  se=$(echo "$out" | jq -r '.return."err-data" // empty' | base64 -d)
  [ -n "$so" ] && printf '%s\n' "$so"
  [ -n "$se" ] && printf '%s' "$se" >&2
  return "$rc"
}

# resolve the VM's current IP (libvirt NAT DHCP — changes on every spawned MAC;
# NEVER hardcode). Prints the IP.
vmip() {
  if [ "$IS_LIBVIRT" = 0 ]; then printf '%s' "$VMIP"; return 0; fi
  local ip
  ip=$(VIRSH domifaddr "$DOMAIN" 2>/dev/null | awk '/ipv4/{print $4}' | cut -d/ -f1 | head -1)
  [ -n "$ip" ] || ip=$(vmexec 'ip -4 addr show scope global | grep -o "inet [0-9.]*"' | awk '{print $2}' | head -1)
  [ -n "$ip" ] || { echo "FATAL: cannot resolve VM IP" >&2; return 1; }
  printf '%s' "$ip"
}

# wait until the guest-agent answers (VM freshly booted)
wait_guest_agent() {
  for _ in $(seq 1 150); do
    VIRSH qemu-agent-command "$DOMAIN" '{"execute":"guest-ping"}' >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "FATAL: guest-agent not answering after 300 s" >&2; return 1
}

# append the caller's pubkey to the VM's authorized_keys (idempotent) — the
# template's image only trusts whoever built it; every spawned VM needs the
# operator's key for the ssh phases.
bootstrap_ssh() {
  if [ "$IS_LIBVIRT" = 0 ]; then
    if [ -n "$VMPASS" ]; then bootstrap_ip; else vmssh "$VMIP" 'id' >/dev/null; fi
    return
  fi
  local pub; pub=$(pubkey) || return 1
  vmexec "grep -qF '$pub' /home/$VM_USER/.ssh/authorized_keys 2>/dev/null || echo '$pub' >> /home/$VM_USER/.ssh/authorized_keys
mkdir -p /home/$VM_USER/.ssh; chown -R $VM_USER:$VM_USER /home/$VM_USER/.ssh; chmod 700 /home/$VM_USER/.ssh; chmod 600 /home/$VM_USER/.ssh/authorized_keys"
}

# MIRA turn-in-flight gate: refuse extract/flush/stop when valkey holds a
# user_lock. Empty output = safe.
turn_lock_gate() {
  local l
  if [ "$IS_LIBVIRT" = 1 ]; then
    l=$(vmexec 'valkey-cli --scan --pattern "user_lock:*" 2>/dev/null || true')
  else
    l=$(vmssh "$VMIP" 'sudo -n valkey-cli --scan --pattern "user_lock:*" 2>/dev/null || true')
  fi
  [ -z "$l" ] || { echo "FATAL: MIRA turn in flight ($l) — do not disturb" >&2; return 1; }
}

# ---------------------------------------------------------------------------
# Plain-ssh mode (--ip): no libvirt on the calling machine (e.g. macOS) and
# the VM is managed by another hypervisor or lives in the cloud. Set VMIP
# (--ip) and optionally VMPASS (--vm-pass) for the one-time password bootstrap.
VMIP="${VMIP:-}"
VMPASS="${VMPASS:-}"
IS_LIBVIRT=1
[ -n "$VMIP" ] && IS_LIBVIRT=0   # env-based default; re-derived after flag parsing:

# finish_flags — call once AFTER argument parsing in every driver: mode can be
# selected by flags (--ip), so IS_LIBVIRT cannot be finalized at source time.
finish_flags() { IS_LIBVIRT=1; [ -n "$VMIP" ] && IS_LIBVIRT=0; return 0; }

# One-time password bootstrap for ip-mode: installs the caller's pubkey and
# passwordless sudo (the deploy is headless; its preflight cannot answer a sudo
# password without a tty). sshpass when available, else expect (ships with macOS).
bootstrap_ip() {
  local pub; pub=$(pubkey) || return 1
  local script
  script=$(mktemp)
  { echo "mkdir -p ~/.ssh && chmod 700 ~/.ssh"
    echo "grep -qF '$pub' ~/.ssh/authorized_keys 2>/dev/null || echo '$pub' >> ~/.ssh/authorized_keys"
    echo "chmod 600 ~/.ssh/authorized_keys"
    # single-stdin discipline: the password PIPE must be sudo's only stdin — a
    # heredoc on the same command would override it (sudo then reads the sudoers
    # line as the password; a real two-hour lesson in POSIX redirection order).
    echo "echo '$VMPASS' | sudo -S -p '' sh -c 'printf \"%s ALL=(ALL) NOPASSWD: ALL\\n\" \"$VM_USER\" > /etc/sudoers.d/90-mira-dev-nopasswd && chmod 440 /etc/sudoers.d/90-mira-dev-nopasswd'"
    echo "sudo -n true 2>/dev/null && echo BOOTSTRAP-OK"
  } > "$script"
  local out
  if command -v sshpass >/dev/null; then
    out=$(sshpass -p "$VMPASS" ssh -o StrictHostKeyChecking=accept-new \
      -o PreferredAuthentications=password -o PubkeyAuthentication=no \
      "$VM_USER@$VMIP" /bin/sh < "$script" 2>&1)
  else
    out=$(expect <<EXP
set timeout 30
spawn scp -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no $script $VM_USER@$VMIP:/tmp/.mira-bootstrap.sh
expect "assword:" { send "$VMPASS\r" }
expect {
  "100%" { }
  eof { }
}
spawn ssh -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no $VM_USER@$VMIP /bin/sh /tmp/.mira-bootstrap.sh
expect "assword:" { send "$VMPASS\r" }
expect eof
EXP
)
  fi
  rm -f "$script"
  vmssh "$VMIP" 'id' >/dev/null 2>&1 \
    || { echo "FATAL: password bootstrap installed no usable key ssh" >&2; return 1; }
  vmssh "$VMIP" 'sudo -n true 2>/dev/null' \
    || { echo "FATAL: key ssh works but passwordless sudo does not — check /tmp/.mira-bootstrap.sh leftovers in the VM" >&2; return 1; }
  echo "bootstrap ok: key + passwordless sudo"
}

# normalize a sarcophagus argument: existing local dir wins; bare name expands
resolve_sarc() {
  local s="$1" local_dir=""
  if [ -d "$s" ]; then
    local_dir="$s"
  elif [ -n "$REMOTE_HOST" ] && ssh -n "$REMOTE_HOST" "test -d '$SNAP_DIR/$s'" 2>/dev/null; then
    local_dir=$(mktemp -d)/sarc
    mkdir -p "$local_dir"
    ssh -n "$REMOTE_HOST" "tar -C '$SNAP_DIR' -czf - '$s'" | tar -C "$local_dir" -xzf -
    local_dir="$local_dir/$s"
  elif [ -d "$SNAP_DIR/$s" ]; then
    local_dir="$SNAP_DIR/$s"
  else
    echo "FATAL: no sarcophagus at '$s' (nor under \$SNAP_DIR)" >&2; return 1
  fi
  (cd "$local_dir" && sha256sum -c MANIFEST.sha256 >/dev/null) \
    || { echo "FATAL: sarcophagus manifest mismatch: $s" >&2; return 1; }
  [ -f "$local_dir/SNAPSHOT-FACTS.txt" ] \
    || { echo "FATAL: no SNAPSHOT-FACTS.txt in $local_dir" >&2; return 1; }
  printf '%s' "$local_dir"
}

# set_common <flag> <value> — assign one common flag (no leading dashes).
# NOTE: a shared parser that "consumes" the caller's positional args cannot
# work — a function gets a COPY of "$@" and its shifts never affect the
# caller. Each driver therefore owns its argument loop and calls this per flag:
#   --*) f="${1#--}"; case "$f" in
#     host|domain|vm-user|snap-dir|vmimg-dir|template|xml|pubkey|ip|vm-pass)
#       shift; set_common "$f" "${1:?--$f needs a value}" ;;
#     *) echo "unknown flag --$f" >&2; exit 2 ;; esac ;;
set_common() {
  case "$1" in
    host) REMOTE_HOST="$2"; JUMP="${VM_SSH_JUMP:-$REMOTE_HOST}" ;;
    domain) DOMAIN="$2" ;;
    vm-user) VM_USER="$2" ;;
    snap-dir) SNAP_DIR="$2" ;;
    vmimg-dir) VMIMG_DIR="$2" ;;
    template) TEMPLATE="$2" ;;
    xml) VM_XML="$2" ;;
    pubkey) SSH_PUB="$2" ;;
    ip) VMIP="$2" ;;
    vm-pass) VMPASS="$2" ;;
    *) return 1 ;;
  esac
}
