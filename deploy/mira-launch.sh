#!/bin/bash
# MIRA launcher — the one start path every supervisor runs (systemd unit,
# launchd agent, or a manual start). Installed by deploy/finalize.sh to
# /opt/mira/bin/mira-launch, outside the code tree it manages.
#
# 1. Environment: PATH for launchd's minimal environment, Vault address and
#    AppRole credentials (kept when the supervisor already set them), the log
#    dir, and /opt/mira/systemone.env exported with set -a.
# 2. Self-edit rollback, only when MIRA_SELF_EDIT_STATE_DIR is set (the
#    supervisor configs set it; a manual start does not) and the tree is a git
#    repository. The contract is owned by utils/self_edit.py:
#      clean tree, no marker      -> start
#      clean tree, marker         -> the trial commit landed but its bookkeeping
#                                    did not: record HEAD as applied, start
#      edited tree, no marker     -> create the marker, start (trial boot)
#      edited tree, marker        -> the trial never reached startup complete:
#                                    stash the change (kept, never deleted),
#                                    record it as failed with the boot log tail,
#                                    start on the clean tree
#    A git failure here parks the launcher (sleeps, never starts MIRA) instead
#    of exiting: an exit would make the supervisor restart into the same
#    failure, and every start runs billed provider probes in the POST gate.
# 3. Boot log: stdout and stderr pass through unchanged and the first
#    BOOT_LOG_CAP bytes of each start are also kept in $STATE/last_boot.log,
#    which is what a failed trial's detail is read from.
#
# Written for bash 3.2 (stock macOS): no associative arrays, no mapfile.

APP=/opt/mira/app
BOOT_LOG_CAP=1000000
RESULT_DETAIL_BYTES=4000

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
cd "$APP" || { echo "mira-launch: cannot cd to $APP" >&2; exit 1; }

export VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
# launchd has no unit ordering: wait (bounded) for Vault to be up AND unsealed.
# The POST gate treats a sealed or unreachable Vault as infrastructure failure.
for _ in $(seq 1 120); do
    S=$(curl -sf "$VAULT_ADDR/v1/sys/seal-status" 2>/dev/null || true)
    case "$S" in *'"sealed":false'*) break;; esac
    sleep 1
done
if [ -z "${VAULT_ROLE_ID:-}" ]; then
    VAULT_ROLE_ID=$(cat /opt/vault/role-id.txt) || exit 1
fi
if [ -z "${VAULT_SECRET_ID:-}" ]; then
    VAULT_SECRET_ID=$(cat /opt/vault/secret-id.txt) || exit 1
fi
export VAULT_ROLE_ID VAULT_SECRET_ID
export MIRA_LOG_DIR="${MIRA_LOG_DIR:-/opt/mira/logs}"
# set -a: sourced assignments must be EXPORTED to reach the server process.
if [ -f /opt/mira/systemone.env ]; then
    set -a
    . /opt/mira/systemone.env
    set +a
fi

if [ -z "${MIRA_SELF_EDIT_STATE_DIR:-}" ] || [ ! -d "$APP/.git" ]; then
    exec venv/bin/python3 main.py "$@"
fi

STATE="$MIRA_SELF_EDIT_STATE_DIR"
BOOT_LOG="$STATE/last_boot.log"

park() {
    echo "mira-launch: $1 — MIRA is NOT started. Fix the tree by hand, then restart the service." >&2
    while true; do sleep 300; done
}

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# write_result <status> <ref> [detail-file]: same format utils/self_edit.py reads.
write_result() {
    {
        printf '%s %s\n' "$1" "$2"
        if [ -n "${3:-}" ] && [ -f "$3" ]; then
            tail -c "$RESULT_DETAIL_BYTES" "$3"
        fi
    } > "$STATE/.result.tmp" || park "cannot write $STATE/.result.tmp"
    mv -f "$STATE/.result.tmp" "$STATE/result" || park "cannot write $STATE/result"
    rm -f "$STATE/offered"
}

[ -d "$STATE" ] || park "self-edit state directory $STATE does not exist"

CHANGES=$(git -C "$APP" status --porcelain --untracked-files=all) \
    || park "git status failed in $APP"

if [ -z "$CHANGES" ]; then
    if [ -e "$STATE/booting" ]; then
        HEAD=$(git -C "$APP" rev-parse HEAD) || park "git rev-parse HEAD failed"
        write_result applied "$HEAD"
        rm -f "$STATE/booting"
        echo "mira-launch: previous trial boot was committed as $HEAD" >&2
    fi
elif [ -e "$STATE/booting" ]; then
    git -C "$APP" stash push --include-untracked --quiet -m "self-edit failed $(utc_now)" \
        || park "the previous trial boot failed and stashing the change failed"
    REF=$(git -C "$APP" rev-parse 'stash@{0}') || park "git rev-parse stash@{0} failed"
    write_result failed "$REF" "$BOOT_LOG"
    rm -f "$STATE/booting"
    echo "mira-launch: previous trial boot failed; change stashed as $REF, starting on the last good code" >&2
else
    touch "$STATE/booting" || park "cannot create $STATE/booting"
    echo "mira-launch: code tree has uncommitted changes; starting a trial boot" >&2
fi

: > "$BOOT_LOG" || park "cannot write $BOOT_LOG"
export PYTHONUNBUFFERED=1
# awk passes every line through to the supervisor's log and copies the first
# BOOT_LOG_CAP bytes into the boot log; it outlives the exec as the reader of
# the server's merged stdout/stderr.
exec > >(awk -v boot_log="$BOOT_LOG" -v cap="$BOOT_LOG_CAP" '
    { print; fflush() }
    n < cap { print > boot_log; fflush(boot_log); n += length($0) + 1 }
') 2>&1
exec venv/bin/python3 main.py "$@"
