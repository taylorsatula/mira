#!/bin/bash
# deploy/update.sh — the machine half of `mira update`: update a deployed
# MIRA install in place. Executed (not sourced) by the TUI's `mira update`
# subcommand (ui/tui/update.py), which resolved and downloaded the release tree
# this script receives. It can also be run by hand:
#
#   bash <release-tree>/deploy/update.sh <release-tree> <tag> [--rebuild-venvs] [--adopt-drifted] [--rebuild-venvs-only]
#
# The positional contract (<release-tree> <tag>) and the environment below are
# fixed: an installed pre-mechanism updater invokes the NEW tree's update.sh
# with exactly these.
#
# Phases (the in-place-update Rule in deploy/AGENTS.md is the contract):
#   1    Preflight — nothing is touched before the service stops.
#   2    Detect the live venv's extras.
#   3    Build only the venvs whose build stamps changed (server still running).
#   3.5  Consent: the release's own step runner computes, read-only, what the
#        database needs; anything that changes the database is shown in plain
#        English and needs Y. No terminal, or no way to reach the database as
#        its administrator, stops here (exit 3) with nothing changed.
#   4    Elevate once (Linux; macOS only to rewrite /usr/local/bin/mira), stop.
#   4.5  Save the operator's own changes; failure restarts the old service.
#   5    Swap rebuilt venvs in; refresh the TOAST shim.
#   6    Move the old code aside, overlay the release, commit, rewrite the
#        `mira` command and (Linux) the systemd unit when their bytes differ.
#   6.5  Run the database steps (backup first, inside the runner), then drop
#        settings-file variables nothing in the release reads.
#   7    Restart; health-poll for the new VERSION; the database must be at
#        the release's head.
#   8    Success: drop the asides. Any failure from phase 4 on: stop the new
#        version, restore the database backup (with the new code — it alone
#        knows how), put code, git, venvs, and the `mira` command back, and
#        restart the old version. If the database cannot be restored, the
#        install is put back but left stopped, with the exact commands.
#
#   --rebuild-venvs-only is the dedicated venv-rebuild fast path (the TUI's
#   `mira update --rebuild-venvs` routes here): it rebuilds both venvs from
#   the INSTALLED tree and touches nothing else — no release resolution, no
#   code swap, no consent gate, no database steps — so the escape hatch
#   works exactly when the version is current and a venv is wrong.
#
# Privileges: the install's owner, plus at most one sudo password prompt
# after consent (Linux service control and the PostgreSQL superuser seam for
# a one-time ownership handover or a restore).
#
# Environment:
#   MIRA_APP_DIR   install root        (default /opt/mira/app); backups go to
#                  <its parent>/backups and the settings file is
#                  <its parent>/systemone.env
#   MIRA_TUI_VENV  TUI client venv      (default /opt/mira/tui-venv)
#   NIGHTLY        set to 1 by `mira update --nightly`: the version-equality
#                  no-op is skipped and, on success, NIGHTLY_SHA is recorded in
#                  <APP_DIR>/data/nightly_stamp.
#   NIGHTLY_SHA    the resolved main-head SHA (passed by ui/tui/update.py)
#   VAULT_ADDR / VAULT_ROLE_ID / VAULT_SECRET_ID
#                  the AppRole the step runner reads Vault with; read from
#                  /opt/vault/{role-id,secret-id}.txt when not already set
#   LOUD_MODE      true for verbose output (deploy lib convention)
#
# Service management (stop, restart, health-poll, the `mira` command)
# applies ONLY to the canonical root /opt/mira/app — a throwaway root
# (verification probes, staged installs) has no service, and this script must
# never stop or restart the real one on its behalf.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOUD_MODE="${LOUD_MODE:-false}"
source "${SCRIPT_DIR}/lib/output.sh"
source "${SCRIPT_DIR}/lib/services.sh"
source "${SCRIPT_DIR}/lib/repo_identity.sh"
source "${SCRIPT_DIR}/lib/provenance.sh"
source "${SCRIPT_DIR}/lib/venv_stamp.sh"
source "${SCRIPT_DIR}/lib/state_preserve.sh"
source "${SCRIPT_DIR}/lib/custom_export.sh"
source "${SCRIPT_DIR}/lib/host_artifacts.sh"

APP_DIR="${MIRA_APP_DIR:-/opt/mira/app}"
TUI_VENV="${MIRA_TUI_VENV:-/opt/mira/tui-venv}"
MIRA_ROOT="$(dirname "$APP_DIR")"
BACKUPS_DIR="${MIRA_ROOT}/backups"
ENV_FILE="${MIRA_ROOT}/systemone.env"
HEALTH_URL="http://127.0.0.1:1993/v1/api/health"
# 600 s for status=healthy AND the new VERSION: 5x the container image's 120 s
# HEALTHCHECK start-period (deploy/docker/Dockerfile), headroom for a
# bare-metal boot that loads models from disk and runs the POST gate; a
# boot that cannot report healthy in 10 minutes is a failure, not a slow
# start, and rolls back. After the first 30 s the poll prints a
# "still waiting (waiting N more seconds)" countdown every 60 s, so a
# parked boot reads as progress toward the deadline, not a silent hang.
HEALTH_POLL_ROUNDS=120
HEALTH_POLL_INTERVAL_SECONDS=5
CANONICAL="no"
[ "$APP_DIR" = "/opt/mira/app" ] && CANONICAL="yes"

usage() {
    echo "usage: $0 <release-tree> <tag> [--rebuild-venvs] [--adopt-drifted] [--rebuild-venvs-only]" >&2
    exit 1
}
[ $# -ge 2 ] || usage
TREE="$1"
TAG="$2"
shift 2
REBUILD_VENVS="no"
ADOPT_DRIFTED="no"
REBUILD_VENVS_ONLY="no"
while [ $# -gt 0 ]; do
    case "$1" in
        --rebuild-venvs) REBUILD_VENVS="yes" ;;
        --rebuild-venvs-only) REBUILD_VENVS_ONLY="yes"; REBUILD_VENVS="yes" ;;
        --adopt-drifted) ADOPT_DRIFTED="yes" ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
    shift
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
UPDATE_LOG="(not yet opened)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mira-update.XXXXXX")"

# One update at a time on this install. Everything upstream of the database
# phase — the venv build into a shared venv.new, the service stop, the venv
# and code swaps — is destructive and unserialized; two updaters running at
# once destroyed each other's venv and left the install unbootable while
# both claimed a clean put-back (campaign finding KVM-4). The database's own
# advisory lock only guards the migrate phase, far too late.
UPDATE_LOCK="${MIRA_ROOT:-/opt/mira}/.update-in-progress"
acquire_update_lock() {
    local waited=0 holder_pid holder_uid stale_dir
    while true; do
        if mkdir "$UPDATE_LOCK" 2>/dev/null; then
            # World-readable on purpose: the pid and uid are not secrets, and a
            # cross-account holder must stay VISIBLE to other accounts' updaters
            # — an unreadable lock dir reads as empty here, which would treat a
            # live root-run update as take-over-able (third-round verifier F1).
            chmod 755 "$UPDATE_LOCK" 2>/dev/null || true
            if echo "$$" > "${UPDATE_LOCK}/pid" 2>/dev/null; then
                echo "$(id -u)" > "${UPDATE_LOCK}/uid" 2>/dev/null || true
                return 0
            fi
            # A lock whose pid file could not be written can never be released
            # by its holder or taken over by the next run; remove it and refuse
            # rather than wedge every future update on it — but only while the
            # pid file is still empty: if a successor already owns the path,
            # it is not ours to delete.
            if [ ! -s "${UPDATE_LOCK}/pid" ]; then
                rm -rf "$UPDATE_LOCK"
            fi
            return 1
        elif [ ! -e "$UPDATE_LOCK" ] && [ ! -w "${UPDATE_LOCK%/*}" ]; then
            # mkdir failed with NO lock present: this account cannot write the
            # lock's parent at all — a second-account update on an
            # install-owner-owned /opt/mira (the a26c11a probe's finding: it
            # waited 120 s and was told a nonexistent update was running).
            return 2
        fi
        holder_pid="$(cat "${UPDATE_LOCK}/pid" 2>/dev/null)"
        holder_uid="$(cat "${UPDATE_LOCK}/uid" 2>/dev/null)"
        if [ -n "$holder_uid" ] && [ "$holder_uid" != "$(id -u)" ]; then
            # Another account holds the lock (e.g. a root-run update). kill -0
            # cannot prove a cross-user holder dead (EPERM reads as gone), so a
            # takeover here could delete a LIVE holder's lock — never steal
            # across accounts; the wait below refuses honestly instead.
            :
        elif { [ -n "$holder_pid" ] && ! kill -0 "$holder_pid" 2>/dev/null; } || { [ -z "$holder_pid" ] && [ "$waited" -ge 4 ]; }; then
            # The holder is gone — a killed or crashed update whose own EXIT
            # trap failed to clean up — or the pid file is still empty after
            # two wait cycles, which covers a holder that died between its
            # mkdir and its pid write (no live holder takes seconds to reach
            # the write; a SIGSTOP'd one can, and being stolen from yields a
            # second holder — accepted: no alternative release exists for a
            # holder that died mid-acquire). Take the lock over with an
            # atomic mv: only one waiter's mv can succeed. Then re-verify the
            # pid inside the moved dir is still the dead holder — a waiter
            # that lost the race moved the winner's LIVE lock and must put it
            # back, never delete it: the moved dir's content belongs to a
            # live process, and deleting it strands that holder unlocked
            # (verifier finding).
            stale_dir="${UPDATE_LOCK}.stale.$$"
            rm -rf "$stale_dir" 2>/dev/null
            if mv "$UPDATE_LOCK" "$stale_dir" 2>/dev/null; then
                if [ "$(cat "${stale_dir}/pid" 2>/dev/null)" = "$holder_pid" ]; then
                    rm -rf "$stale_dir"
                elif [ ! -e "$UPDATE_LOCK" ]; then
                    mv "$stale_dir" "$UPDATE_LOCK" 2>/dev/null || true
                fi
                # else: a third holder owns UPDATE_LOCK and the moved dir names
                # a live pid — leave it aside as residue; never delete a live
                # holder's lock.
            fi
        fi
        # Every path flows through the bound: a steal whose cleanup keeps
        # failing must still refuse after 120 s, never spin.
        [ "$waited" -ge 120 ] && return 1
        sleep 2
        waited=$((waited + 2))
    done
}
LOCK_RC=0
acquire_update_lock || LOCK_RC=$?
if [ "$LOCK_RC" = 2 ]; then
    print_error "This account cannot write ${UPDATE_LOCK%/*} (where MIRA records its update lock), so the update never started. Nothing was changed. Run the update as the account that owns the install."
    echo "update-lock: no permission to create ${UPDATE_LOCK}" >&2
    exit 1
elif [ "$LOCK_RC" != 0 ]; then
    print_error "Another MIRA update is already running on this install, so this one did nothing. Nothing was changed. Wait for it to finish, then run: mira update"
    echo "update-lock: held at ${UPDATE_LOCK}" >&2
    echo "update-lock: if no update is running, that directory is leftover — remove it (with sudo if it is not yours), then run: mira update" >&2
    exit 3
fi
LOCK_HELD="yes"

OLD_VENV="${APP_DIR}/venv"
NEW_VENV="${APP_DIR}/venv.new"
NEW_TUI_VENV="${TUI_VENV}.new"
CODE_ASIDE="${APP_DIR}.pre-update-${STAMP}"

# Progress flags the EXIT trap reads to undo exactly what happened.
SERVICE_STOPPED="no"
SERVER_VENV_SWAPPED="no"
TUI_VENV_SWAPPED="no"
TOAST_REFRESHED="no"
CODE_MOVED="no"
OVERLAY_STARTED="no"
INDEX_SAVED="no"
UPDATE_COMMITTED="no"
CLI_REWRITTEN="no"
UNIT_REWRITTEN="no"
MIGRATE_STARTED="no"
NEW_SERVICE_STARTED="no"
SUCCEEDED="no"
FAILURE_KIND="rollback"

print_header "MIRA in-place update → ${TAG}"

LAUNCHD_PLIST="$HOME/Library/LaunchAgents/com.mira.app.plist"

stop_mira_service() {
    [ "$CANONICAL" = "yes" ] || return 0
    if [ "$(uname)" = "Darwin" ]; then
        if [ -f "$LAUNCHD_PLIST" ]; then
            # bootout, not kill: unloading the agent takes KeepAlive out of
            # the picture entirely, so a nonzero exit cannot restart the
            # server onto a half-swapped tree mid-update.
            launchctl bootout "gui/$(id -u)/com.mira.app" 2>/dev/null || true
            local i
            for i in $(seq 1 30); do
                if port_probe_status 1993; then sleep 1; else break; fi
            done
        else
            stop_service mira port 1993
        fi
    elif systemctl cat mira > /dev/null 2>&1; then
        sudo systemctl stop mira
    else
        stop_service mira port 1993
    fi
}

start_mira_service() {
    [ "$CANONICAL" = "yes" ] || return 0
    if [ "$(uname)" = "Darwin" ]; then
        if [ -f "$LAUNCHD_PLIST" ]; then
            launchd_reload_agent "$LAUNCHD_PLIST"
        else
            # Supervisor-less install (no LaunchAgent plist): run.sh is the
            # tree's designated start path for this class. Detached, PID
            # recorded in this log — it keeps listening on 1993, so
            # stop_mira_service's port-kill still finds it.
            print_info "No LaunchAgent plist — starting MIRA via ${APP_DIR}/run.sh"
            nohup "${APP_DIR}/run.sh" >/dev/null 2>&1 &
            print_info "MIRA starting under pid $! (no supervisor: it will not be restarted if it exits)"
        fi
    elif systemctl cat mira > /dev/null 2>&1; then
        sudo systemctl start mira
    else
        # Supervisor-less install (declined or failed systemd): same run.sh
        # arm, so phase 7's poll and on_exit's old-version restart both
        # have a server to poll.
        print_info "No systemd unit — starting MIRA via ${APP_DIR}/run.sh"
        nohup "${APP_DIR}/run.sh" >/dev/null 2>&1 &
        print_info "MIRA starting under pid $! (no supervisor: it will not be restarted if it exits)"
    fi
}

start_command() {
    if [ "$(uname)" = "Darwin" ]; then
        if [ -f "$LAUNCHD_PLIST" ]; then
            echo "launchctl bootstrap gui/$(id -u) ${LAUNCHD_PLIST}"
        else
            echo "nohup ${APP_DIR}/run.sh >/dev/null 2>&1 &"
        fi
    elif systemctl cat mira > /dev/null 2>&1; then
        echo "sudo systemctl start mira"
    else
        echo "nohup ${APP_DIR}/run.sh >/dev/null 2>&1 &"
    fi
}

# Polls the health endpoint until it reports status=healthy AND <version>.
wait_healthy() {
    local version="$1" i body
    [ "$CANONICAL" = "yes" ] || return 0
    echo -ne "${DIM}${ARROW}${RESET} Waiting for MIRA ${version} to become healthy... "
    for i in $(seq 1 "$HEALTH_POLL_ROUNDS"); do
        body="$(curl -sf --max-time 5 "$HEALTH_URL" 2>/dev/null || true)"
        if printf '%s' "$body" | grep -q '"status":"healthy"' \
            && printf '%s' "$body" | grep -q "\"version\":\"${version}\""; then
            echo -e "${CHECKMARK} ${DIM}(after ~$((i * HEALTH_POLL_INTERVAL_SECONDS))s)${RESET}"
            return 0
        fi
        # 30 s silent, then a countdown notice every 60 s until the deadline.
        if [ $((i % 12)) -eq 6 ]; then
            echo -ne "${DIM}still waiting (waiting $(((HEALTH_POLL_ROUNDS - i) * HEALTH_POLL_INTERVAL_SECONDS)) more seconds)${RESET} "
        fi
        sleep "$HEALTH_POLL_INTERVAL_SECONDS"
    done
    echo -e "${ERROR}"
    print_error "MIRA did not report healthy on ${version} within $((HEALTH_POLL_ROUNDS * HEALTH_POLL_INTERVAL_SECONDS)) s."
    return 1
}

# Runs the step runner CLI with the given interpreter and code root. -P keeps
# the caller's working directory (perhaps a checkout) off sys.path.
runner() {
    local python="$1" root="$2"
    shift 2
    PYTHONPATH="$root" "$python" -P -m scripts.migrate "$@"
}

# Sets SERVER_EXTRAS to this install's extras CSV (canonical order): torch
# and sentence-transformers are deliberately NOT in requirements.txt (they
# ride the CONFIG_EMBEDDING_PROVIDER=local install choice), and neither is
# the playwright package. The live venv is the only record of that choice —
# read it before anything replaces the venv.
# Usage: detect_server_extras
detect_server_extras() {
    local had_torch="no" had_st="no" had_playwright="no"
    if [ -f "${OLD_VENV}/bin/python3" ]; then
        "${OLD_VENV}/bin/python3" -c "import torch" 2>/dev/null && had_torch="yes"
        "${OLD_VENV}/bin/python3" -c "import sentence_transformers" 2>/dev/null && had_st="yes"
        "${OLD_VENV}/bin/python3" -c "import playwright" 2>/dev/null && had_playwright="yes"
    fi
    SERVER_EXTRAS=""
    [ "$had_torch" = "yes" ] && SERVER_EXTRAS="${SERVER_EXTRAS},torch"
    [ "$had_st" = "yes" ] && SERVER_EXTRAS="${SERVER_EXTRAS},sentence-transformers"
    [ "$had_playwright" = "yes" ] && SERVER_EXTRAS="${SERVER_EXTRAS},playwright"
    SERVER_EXTRAS="$(canonical_extras "$SERVER_EXTRAS")"
}

# Sets PY to the interpreter for new venvs: the base interpreter the live
# venv runs on, so an install on a keg-only Homebrew Python does not
# silently jump to a different minor version. Fall back to PATH python3
# with the 3.12 floor; exits the script when none (or one below the floor)
# is found.
# Usage: resolve_build_python
resolve_build_python() {
    PY=""
    if [ -f "${OLD_VENV}/bin/python3" ]; then
        PY="$(venv_base_python "$OLD_VENV" 2>/dev/null || true)"
    fi
    if [ ! -x "${PY:-}" ]; then
        PY="$(command -v python3 || true)"
        [ -n "$PY" ] || { print_error "No python3 found on PATH."; exit 1; }
    fi
    PY_VERSION="$(python_version_of "$PY")"
    PY_MAJOR="${PY_VERSION%%.*}"
    PY_MINOR="$(echo "$PY_VERSION" | cut -d. -f2)"
    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 12 ]; }; then
        print_error "MIRA requires Python 3.12 or higher; found ${PY_VERSION} (${PY})."
        exit 1
    fi
}

# build_venv <dest> <requirements-file> <label> <extras-csv> <toast-yes|no>
build_venv() {
    local dest="$1" reqfile="$2" label="$3" extras="$4" toast="$5" site
    rm -rf "$dest"
    run_with_status "Creating ${label} venv" "$PY" -m venv "$dest"
    run_with_status "Initializing ${label} pip" "$dest/bin/python3" -m ensurepip
    "$dest/bin/python3" -m pip install --quiet --upgrade pip \
        || print_warning "Could not upgrade ${label} pip; continuing with the bundled version"
    if [ "$LOUD_MODE" = true ]; then
        print_step "Installing ${label} dependencies..."
        "$dest/bin/python3" -m pip install -r "$reqfile"
    else
        ("$dest/bin/python3" -m pip install -q -r "$reqfile") &
        if ! show_progress $! "Installing ${label} packages"; then
            print_error "Failed to install ${label} packages from ${reqfile}"
            print_info "Re-run with LOUD_MODE=true for the detailed error."
            return 1
        fi
    fi
    case ",${extras}," in *,torch,*)
        ("$dest/bin/python3" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu) &
        show_progress $! "Reinstalling PyTorch (CPU)" \
            || { print_error "Failed to reinstall torch for this local-embedding install."; return 1; }
    esac
    case ",${extras}," in *,sentence-transformers,*)
        ("$dest/bin/python3" -m pip install -q sentence-transformers) &
        show_progress $! "Reinstalling sentence-transformers" \
            || { print_error "Failed to reinstall sentence-transformers for this local-embedding install."; return 1; }
    esac
    case ",${extras}," in *,playwright,*)
        ("$dest/bin/python3" -m pip install -q playwright) &
        show_progress $! "Reinstalling playwright" \
            || { print_error "Failed to reinstall playwright for this install."; return 1; }
        # Idempotent and build-specific; the browser cache (~/.cache/ms-playwright)
        # survives the venv rebuild, so this is a fast no-op unless the pinned
        # playwright moved to a new build.
        ("$dest/bin/playwright" install chromium > /dev/null 2>&1) &
        show_progress $! "Ensuring Chromium is cached" \
            || { print_error "playwright install chromium failed."; return 1; }
    esac
    if [ "$toast" = "yes" ]; then
        # The TOAST log level must exist at interpreter startup, before any
        # application import (python.sh Step 5 does the same at install time).
        site="$("$dest/bin/python3" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
        cp "${TREE}/deploy/_mira_log_levels.py" "${site}/"
        echo "import _mira_log_levels" > "${site}/mira-log-levels.pth"
    fi
    write_venv_stamp "$dest" "$reqfile" "$PY" "$extras"
}

# Sets DECIDED_ACTION to reuse|rebuild and prints the decision.
decide_venv() {
    local label="$1" venv="$2" reqfile="$3" extras="$4" rc=0 reason
    if [ "$REBUILD_VENVS" = "yes" ]; then
        reason="--rebuild-venvs"
        rc=1
    else
        venv_stamp_matches "$venv" "$reqfile" "$PY" "$extras" || rc=$?
        reason="$VENV_STAMP_REASON"
    fi
    if [ "$rc" -eq 0 ]; then
        print_info "${label} venv: reusing (build inputs unchanged)"
        DECIDED_ACTION="reuse"
    else
        print_info "${label} venv: rebuilding (${reason})"
        DECIDED_ACTION="rebuild"
    fi
}

# The --rebuild-venvs-only fast path (`mira update --rebuild-venvs` routes
# here through the TUI): rebuild both venvs from the INSTALLED tree and
# touch nothing else — no release resolution, no code swap, no consent gate,
# no database steps — so the venv-rebuild escape hatch works exactly when
# the version is already current and a venv is wrong. The venv machinery is
# the full update's: extras re-detected from the old venvs
# (detect_server_extras), decide_venv/build_venv with the build stamp written last,
# hand-installed-package drift recorded before the swap, the aside/undo
# discipline on the swap (on_exit rolls a failed swap back), and a
# stop/start around it. The service IS stopped for the swap, exactly as a
# full update stops: a live venv cannot be renamed out from under a running
# interpreter (its later lazy imports would fail), so an in-place rebuild
# without a stop is not safe.
rebuild_venvs_only() {
    TREE="$APP_DIR"
    for f in main.py requirements.txt VERSION deploy/_mira_log_levels.py ui/tui/requirements.txt; do
        if [ ! -f "${TREE}/${f}" ]; then
            print_error "Install root ${APP_DIR} is missing ${f} — not a deployed MIRA install."
            exit 1
        fi
    done
    CUR_VERSION="$(cat "${APP_DIR}/VERSION")"
    NEW_VERSION="$CUR_VERSION"
    mkdir -p "${APP_DIR}/logs"
    UPDATE_LOG="${APP_DIR}/logs/update-${STAMP}.log"
    exec > >(tee -a "$UPDATE_LOG") 2>&1
    print_info "Rebuilding the MIRA venvs from the installed tree (${CUR_VERSION}) — nothing else on the install is touched"
    detect_server_extras
    resolve_build_python
    print_header "Python environments"
    decide_venv "server" "$OLD_VENV" "${TREE}/requirements.txt" "$SERVER_EXTRAS"
    SERVER_VENV_ACTION="$DECIDED_ACTION"
    decide_venv "TUI client" "$TUI_VENV" "${TREE}/ui/tui/requirements.txt" ""
    TUI_VENV_ACTION="$DECIDED_ACTION"
    [ -n "$SERVER_EXTRAS" ] && print_info "Reinstalling this install's extras: ${SERVER_EXTRAS}"
    build_venv "$NEW_VENV" "${TREE}/requirements.txt" "server" "$SERVER_EXTRAS" yes
    build_venv "$NEW_TUI_VENV" "${TREE}/ui/tui/requirements.txt" "TUI client" "" no
    if [ "$CANONICAL" = "yes" ] && [ "$(uname)" != "Darwin" ]; then
        acquire_sudo
        # acquire_sudo installs its own EXIT trap for the askpass helper; on_exit
        # removes the helper itself, so the rollback trap is reinstated here.
        trap on_exit EXIT
    fi
    if [ "$CANONICAL" = "yes" ]; then
        print_header "Stopping MIRA"
        SERVICE_STOPPED="yes"
        stop_mira_service
        print_success "MIRA stopped"
    else
        print_info "Non-canonical root ${APP_DIR}: no service stop (the real install's service is never touched)."
        SERVICE_STOPPED="yes"
    fi
    DRIFT_CATALOG="${APP_DIR}/logs/update-${STAMP}-venv-drift.txt"
    if [ -x "${OLD_VENV}/bin/python3" ]; then
        record_venv_drift "server" "$OLD_VENV" "$NEW_VENV" "$DRIFT_CATALOG"
        if [ "$VENV_DRIFT_LOST" -gt 0 ]; then
            print_warning "${VENV_DRIFT_LOST} Python packages installed by hand in the server venv are not part of this release; the list is in ${DRIFT_CATALOG}."
        fi
    fi
    if [ -x "${TUI_VENV}/bin/python3" ]; then
        record_venv_drift "TUI client" "$TUI_VENV" "$NEW_TUI_VENV" "$DRIFT_CATALOG"
    fi
    print_header "Installing the rebuilt venvs"
    SERVER_VENV_SWAPPED="yes"
    [ -e "$OLD_VENV" ] && mv "$OLD_VENV" "${OLD_VENV}.pre-update-${STAMP}"
    mv "$NEW_VENV" "$OLD_VENV"
    print_success "Server venv replaced"
    TUI_VENV_SWAPPED="yes"
    [ -e "$TUI_VENV" ] && mv "$TUI_VENV" "${TUI_VENV}.pre-update-${STAMP}"
    mv "$NEW_TUI_VENV" "$TUI_VENV"
    print_success "TUI client venv replaced"
    if [ "$CANONICAL" = "yes" ]; then
        print_header "Starting MIRA"
        NEW_SERVICE_STARTED="yes"
        start_mira_service || true
        if ! wait_healthy "$NEW_VERSION"; then
            print_info "Logs: tail -100 /opt/mira/logs/mira-launchd.log (macOS) or journalctl -u mira -n 100 (Linux)"
            exit 1
        fi
    else
        print_info "Non-canonical root: restart manually and verify ${HEALTH_URL} reports ${NEW_VERSION}."
    fi
    SUCCEEDED="yes"
    rm -rf "${OLD_VENV}.pre-update-${STAMP}" "${TUI_VENV}.pre-update-${STAMP}" \
        || print_warning "Could not remove the pre-update venv copies (${OLD_VENV}.pre-update-${STAMP}, ${TUI_VENV}.pre-update-${STAMP}) — they are preserved; remove them by hand once you have checked them."
    print_success "MIRA venvs rebuilt (${CUR_VERSION})"
    exit 0
}

# undo_venv_swap <live> <new>: <new> still present means it never moved in,
# so <live> is the original and stays.
undo_venv_swap() {
    local live="$1" new="$2" aside="$1.pre-update-${STAMP}"
    if [ -e "$aside" ]; then
        rm -rf "$live" && mv "$aside" "$live"
    elif [ ! -e "$new" ]; then
        rm -rf "$live"
    fi
}

rollback_install() {
    local problems=0
    if [ "$CODE_MOVED" = "yes" ]; then
        if [ "$OVERLAY_STARTED" = "yes" ]; then
            restore_code_aside "$APP_DIR" "$CODE_ASIDE" --clear-first || problems=$((problems + 1))
        else
            restore_code_aside "$APP_DIR" "$CODE_ASIDE" || problems=$((problems + 1))
        fi
    fi
    if [ "$UPDATE_COMMITTED" = "yes" ]; then
        git -C "$APP_DIR" update-ref -m "mira update rollback" HEAD "$PRE_UPDATE_SHA" \
            || problems=$((problems + 1))
    fi
    if [ "$INDEX_SAVED" = "yes" ]; then
        cp -p "${WORK_DIR}/index" "${APP_DIR}/.git/index" || problems=$((problems + 1))
    fi
    if [ "$CLI_REWRITTEN" = "yes" ]; then
        install_host_file "${WORK_DIR}/mira-cli.old" "$MIRA_CLI_PATH" || problems=$((problems + 1))
    fi
    if [ "$UNIT_REWRITTEN" = "yes" ]; then
        { install_host_file "${WORK_DIR}/mira-unit.old" "$MIRA_UNIT_PATH" 644 && sudo systemctl daemon-reload; } \
            || problems=$((problems + 1))
    fi
    if [ "$MIGRATE_STARTED" = "yes" ] && [ -f "${WORK_DIR}/systemone.env.pre-update" ]; then
        cp -p "${WORK_DIR}/systemone.env.pre-update" "$ENV_FILE" || problems=$((problems + 1))
    fi
    if [ "$TOAST_REFRESHED" = "yes" ]; then
        local shim
        for shim in _mira_log_levels.py mira-log-levels.pth; do
            if [ -f "${WORK_DIR}/toast/${shim}" ]; then
                cp -p "${WORK_DIR}/toast/${shim}" "${LIVE_SITE_PACKAGES}/${shim}" || problems=$((problems + 1))
            else
                rm -f "${LIVE_SITE_PACKAGES}/${shim}" || problems=$((problems + 1))
            fi
        done
    fi
    # Filesystem state, not flags, decides the venv undo: an aside named for
    # this run exists exactly when the live path no longer holds the original.
    if [ "$SERVER_VENV_SWAPPED" = "yes" ]; then
        undo_venv_swap "$OLD_VENV" "$NEW_VENV" || problems=$((problems + 1))
    fi
    if [ "$TUI_VENV_SWAPPED" = "yes" ]; then
        undo_venv_swap "$TUI_VENV" "$NEW_TUI_VENV" || problems=$((problems + 1))
    fi
    return "$problems"
}

# Puts the database back from this run's backup, using the code now in
# APP_DIR (the new release). Returns 0 restored or nothing to restore, 1 failed.
restore_database() {
    [ "$CANONICAL" = "yes" ] || return 0
    [ "$MIGRATE_STARTED" = "yes" ] || return 0
    [ -f "${BACKUPS_DIR}/${STAMP}/manifest.json" ] || return 0
    print_info "Restoring your database from the backup taken before this update..."
    runner "$OLD_VENV/bin/python3" "$APP_DIR" restore "$STAMP" --backups-dir "$BACKUPS_DIR"
}

on_exit() {
    local rc=$?
    set +eu
    trap - EXIT INT TERM
    if [ "$SUCCEEDED" = "yes" ]; then
        rm -rf "$WORK_DIR" "${MIRA_SUDO_DIR:-}"
    elif [ "$SERVICE_STOPPED" = "yes" ]; then
        echo ""
        print_header "Putting MIRA back as it was"
        if [ "$NEW_SERVICE_STARTED" = "yes" ]; then
            stop_mira_service
        fi
        local data_restored="yes"
        restore_database || data_restored="no"
        local restored="yes"
        rollback_install || restored="no"
        if [ "$data_restored" != "yes" ]; then
            print_error "MIRA is stopped and will stay stopped until your data is sorted out."
            if [ "$restored" = "yes" ]; then
                print_info "Your program files were put back to ${CUR_VERSION}."
            else
                print_info "Your program files could not all be put back automatically; the saved previous files are at: ${CODE_ASIDE}."
            fi
            print_info "Do what the message above says about your data, then start MIRA:"
            print_info "  $(start_command)"
            print_info "Then run: mira update"
            rm -rf "$NEW_VENV" "$NEW_TUI_VENV" "$WORK_DIR" "${MIRA_SUDO_DIR:-}"
            release_update_lock
            exit 1
        fi
        local started="no"
        if start_mira_service && wait_healthy "$CUR_VERSION"; then
            started="yes"
        fi
        if [ "$FAILURE_KIND" = "preserve" ]; then
            print_error "MIRA could not save a safe copy of the files you changed, so it stopped before touching anything. Your install is exactly as it was and is running again. Details are in ${UPDATE_LOG}. Run: mira update to try again."
        elif [ "$restored" = "yes" ]; then
            print_error "The update could not finish, so MIRA put your install back exactly as it was and restarted the previous version (${CUR_VERSION}). Details are in ${UPDATE_LOG}. Run: mira update to try again."
        else
            print_error "The update could not finish and MIRA could not put every program file back automatically. Your data is untouched. Details are in ${UPDATE_LOG}. The saved program files are at: ${CODE_ASIDE} — then run: mira update"
        fi
        if [ "$started" != "yes" ] && [ "$CANONICAL" = "yes" ]; then
            print_warning "The previous version did not report healthy. Start it by hand: $(start_command) — then check: curl ${HEALTH_URL}"
        fi
        rc=1
    fi
    rm -rf "$NEW_VENV" "$NEW_TUI_VENV" "$WORK_DIR" "${MIRA_SUDO_DIR:-}"
    release_update_lock
    exit "$rc"
}
release_update_lock() {
    # Called last on every exit path: the rollback it follows can run for
    # minutes (dropdb/createdb/pg_restore, venv moves, restart), and a second
    # updater that acquired the lock inside that window re-entered the exact
    # race the lock exists to prevent. Released only by the process whose pid
    # is inside — a stolen-over lock is not ours to remove.
    if [ "${LOCK_HELD:-no}" = "yes" ] && [ "$(cat "${UPDATE_LOCK}/pid" 2>/dev/null)" = "$$" ]; then
        rm -rf "$UPDATE_LOCK"
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 0. --rebuild-venvs-only: the dedicated venv-rebuild fast path. Dispatched
# before preflight — it skips the release resolution, the code swap, the
# consent gate, and the database entirely, so none of the machinery below
# the dispatch runs for it.
if [ "$REBUILD_VENVS_ONLY" = "yes" ]; then
    rebuild_venvs_only
fi

# 1. Preflight
for f in main.py requirements.txt VERSION deploy/_mira_log_levels.py ui/tui/requirements.txt scripts/migrate.py; do
    if [ ! -f "${TREE}/${f}" ]; then
        print_error "Release tree ${TREE} is missing ${f} — not a MIRA release tree."
        exit 1
    fi
done
for f in main.py requirements.txt VERSION; do
    if [ ! -f "${APP_DIR}/${f}" ]; then
        print_error "Install root ${APP_DIR} is missing ${f} — not a deployed MIRA install."
        exit 1
    fi
done
NEW_VERSION="$(cat "${TREE}/VERSION")"
CUR_VERSION="$(cat "${APP_DIR}/VERSION")"
if [ "$NEW_VERSION" = "$CUR_VERSION" ] && [ "${NIGHTLY:-0}" != "1" ] && [ "$ADOPT_DRIFTED" != "yes" ]; then
    print_info "Already at ${CUR_VERSION} — nothing to do."
    SUCCEEDED="yes"
    exit 0
fi

nogit_refusal() {
    print_error "This MIRA install has no record of its own files, so it cannot be updated in place safely. Nothing was changed. Reinstall it: curl -fsSL https://raw.githubusercontent.com/$(mira_github_repo)/main/install.sh | bash"
    echo "preflight: ${APP_DIR} $1" >&2
    exit 1
}
if ! git -C "$APP_DIR" rev-parse --git-dir > /dev/null 2>&1; then
    # Distinguish git's dubious-ownership refusal from a really missing .git:
    # retrying with this session allowed to trust the directory succeeds only
    # in the ownership case (witnessed: Fedora Q15 — a second-user update was
    # told to reinstall when the .git was fine and merely not trusted).
    if git -C "$APP_DIR" -c "safe.directory=$APP_DIR" rev-parse --git-dir > /dev/null 2>&1; then
        print_error "This MIRA install's history is present, but git will not read it for this user (the install belongs to another account). Nothing was changed. If you really want to update from this account, first tell git to trust the install: git config --global --add safe.directory '$APP_DIR' — then run: mira update"
        echo "preflight: ${APP_DIR} git refused this user (dubious ownership); .git present" >&2
        exit 1
    fi
    nogit_refusal "has no install history (no .git)"
fi
BASELINE="$(official_baseline_commit "$APP_DIR")" || nogit_refusal "has no official install or update commit"
PRE_UPDATE_SHA="$(git -C "$APP_DIR" rev-parse HEAD)"

if [ "${NIGHTLY:-0}" = "1" ]; then
    NEW_SHA="${NIGHTLY_SHA:?NIGHTLY=1 requires NIGHTLY_SHA}"
elif ! NEW_SHA="$(read_release_commit "$TREE" 2>/dev/null)"; then
    if ! NEW_SHA="$(resolve_release_sha "$TAG")"; then
        print_error "MIRA could not confirm which release ${TAG} is (no record in the download, and GitHub did not answer). Nothing was changed. Check your network, then run: mira update"
        exit 1
    fi
fi

# The step runner needs the PostgreSQL client tools of the server's major
# (keg-only on macOS) and the AppRole the service itself uses.
if PG_BIN="$(pg_bindir 2>/dev/null)"; then
    export PATH="${PG_BIN}:${PATH}"
fi
if [ -z "${VAULT_ROLE_ID:-}" ] || [ -z "${VAULT_SECRET_ID:-}" ]; then
    if [ -r /opt/vault/role-id.txt ] && [ -r /opt/vault/secret-id.txt ]; then
        VAULT_ROLE_ID="$(cat /opt/vault/role-id.txt)"
        VAULT_SECRET_ID="$(cat /opt/vault/secret-id.txt)"
    fi
fi
export VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}" VAULT_ROLE_ID="${VAULT_ROLE_ID:-}" VAULT_SECRET_ID="${VAULT_SECRET_ID:-}"
SEAM_MODE="$(seam_mode)"

CLI_REWRITE="no"
CLI_NEEDS_SUDO="no"
if [ "$CANONICAL" = "yes" ] && [ -f "$MIRA_CLI_PATH" ]; then
    render_mira_cli > "${WORK_DIR}/mira-cli.new"
    if ! cmp -s "${WORK_DIR}/mira-cli.new" "$MIRA_CLI_PATH"; then
        CLI_REWRITE="yes"
        [ -w "$MIRA_CLI_PATH" ] || CLI_NEEDS_SUDO="yes"
    fi
fi
# The systemd unit is regenerated from the release's generator with this
# host's own inputs; a unit that already matches is never touched.
UNIT_REWRITE="no"
if [ "$CANONICAL" = "yes" ] && [ "$(uname)" != "Darwin" ] && [ -f "$MIRA_UNIT_PATH" ]; then
    if render_current_mira_unit > "${WORK_DIR}/mira-unit.new" 2>/dev/null; then
        cmp -s "${WORK_DIR}/mira-unit.new" "$MIRA_UNIT_PATH" || UNIT_REWRITE="yes"
    else
        print_warning "MIRA could not read what its service file needs (/opt/vault/*.txt or ${MIRA_UNIT_PATH}), so it will leave the service file as it is."
    fi
fi

mkdir -p "${APP_DIR}/logs"
UPDATE_LOG="${APP_DIR}/logs/update-${STAMP}.log"
exec > >(tee -a "$UPDATE_LOG") 2>&1
print_info "Updating ${CUR_VERSION} → ${NEW_VERSION} in ${APP_DIR} (release commit ${NEW_SHA})"
write_install_exclude "$APP_DIR"

# 2. Live-venv extras: reproduce a local-embedding / browser install
detect_server_extras
resolve_build_python

# 3. Build changed venvs (server still running)
print_header "Python environments"
decide_venv "server" "$OLD_VENV" "${TREE}/requirements.txt" "$SERVER_EXTRAS"
SERVER_VENV_ACTION="$DECIDED_ACTION"
decide_venv "TUI client" "$TUI_VENV" "${TREE}/ui/tui/requirements.txt" ""
TUI_VENV_ACTION="$DECIDED_ACTION"
if [ "$SERVER_VENV_ACTION" = "rebuild" ]; then
    [ -n "$SERVER_EXTRAS" ] && print_info "Reinstalling this install's extras: ${SERVER_EXTRAS}"
    build_venv "$NEW_VENV" "${TREE}/requirements.txt" "server" "$SERVER_EXTRAS" yes
fi
if [ "$TUI_VENV_ACTION" = "rebuild" ]; then
    build_venv "$NEW_TUI_VENV" "${TREE}/ui/tui/requirements.txt" "TUI client" "" no
fi

# 3.5 Consent: what the database needs, read-only, before anything changes
# MIRA_APP_DIR scopes files, never the database: the connection target is
# the host's Vault AppRole, so a non-canonical root's database steps would
# read and change the host install's database. Refuse before the first
# runner exec — a venv-less throwaway must not traceback — with nothing
# changed (the EXIT trap cleans up this run's venv builds and work dir).
if [ "$CANONICAL" != "yes" ]; then
    print_error "Non-canonical root ${APP_DIR}: this update's database steps would read and change the host install's database — MIRA_APP_DIR scopes files, never the database. Nothing was changed."
    exit 3
fi
PLAN_PY="${OLD_VENV}/bin/python3"
[ -x "${NEW_VENV}/bin/python3" ] && PLAN_PY="${NEW_VENV}/bin/python3"
# The plan's elevation sentence must name every password the update will ask
# for. seam_mode covers only the database seam; on macOS the one other
# elevation is the root-owned mira wrapper rewrite (CLI_NEEDS_SUDO, computed
# above), so a Darwin update that will prompt for it must not print "No
# password is needed" (M-1, live-witnessed on the Mac gauntlet).
PLAN_ELEVATION="$SEAM_MODE"
[ "$(uname)" = "Darwin" ] && [ "$CLI_NEEDS_SUDO" = "yes" ] && PLAN_ELEVATION="interactive"
PLAN_ARGS=(migrate --plan --report-marker --elevation "$PLAN_ELEVATION")
[ "$ADOPT_DRIFTED" = "yes" ] && PLAN_ARGS+=(--adopt-drifted)
PLAN_RC=0
runner "$PLAN_PY" "$TREE" "${PLAN_ARGS[@]}" > "${WORK_DIR}/plan.out" 2> "${WORK_DIR}/plan.err" || PLAN_RC=$?
cat "${WORK_DIR}/plan.err" >> "$UPDATE_LOG"
PLAN_TEXT="$(sed '/^MIRA_PLAN_BEGIN$/,$d' "${WORK_DIR}/plan.out")"
PLAN_JSON="$(sed -n '/^MIRA_PLAN_BEGIN$/,/^MIRA_PLAN_END$/p' "${WORK_DIR}/plan.out" | sed '1d;$d')"
NEEDS_CONSENT="$("$PLAN_PY" -c 'import json, sys; print("yes" if json.loads(sys.stdin.read())["needs_consent"] else "no")' <<< "$PLAN_JSON" 2>/dev/null || echo invalid)"
if [ "$PLAN_RC" -eq 2 ] && [ "$NEEDS_CONSENT" != "invalid" ]; then
    echo ""
    printf '%s\n' "$PLAN_TEXT"
    exit 3
fi
if [ "$PLAN_RC" -ne 0 ] || [ "$NEEDS_CONSENT" = "invalid" ]; then
    [ -n "$PLAN_TEXT" ] && printf '%s\n' "$PLAN_TEXT"
    print_error "MIRA could not read your database to work out what this upgrade needs to do. Nothing was changed. Make sure MIRA is running, then run: mira update"
    exit 3
fi
if [ "$NEEDS_CONSENT" = "yes" ]; then
    echo ""
    if [ "$SEAM_MODE" = "unavailable" ]; then
        printf '%s\n' "$PLAN_TEXT"
        if [ "$(uname)" = "Darwin" ]; then
            # seam_mode's Darwin arm prints unavailable only when the account
            # running MIRA answered that it is not the database superuser —
            # no terminal or password changes that, so name the real cause
            # and the only remedy that can work.
            print_error "This upgrade needs to reach your database as its administrator, but the account running MIRA is not this database's superuser. Nothing was changed. Run: mira update again from the account that administers this database (its superuser)."
        else
            print_error "This upgrade needs to reach your database as its administrator, and this session has no terminal to ask for your password. Nothing was changed. Run: mira update again from a terminal where you can type."
        fi
        exit 3
    fi
    if [ ! -t 0 ]; then
        printf '%s\n' "$PLAN_TEXT"
        print_error "This upgrade changes your database and needs your OK first, and this session has no terminal to ask you in. Nothing was changed. Run: mira update again from a terminal where you can type."
        exit 3
    fi
    print_header "MIRA update  ${CUR_VERSION} → ${NEW_VERSION}"
    echo ""
    printf '%s\n' "$PLAN_TEXT"
    echo ""
    ANSWER=""
    read -r -p "Press Y to continue, N to cancel (nothing changes): " ANSWER || ANSWER=""
    case "$ANSWER" in
        Y|y) ;;
        *)
            print_info "Update cancelled — nothing changed."
            exit 0 ;;
    esac
fi

# 4. Elevate once, then stop the service
if [ "$CANONICAL" = "yes" ] && { [ "$(uname)" != "Darwin" ] || [ "$CLI_NEEDS_SUDO" = "yes" ]; }; then
    if [ "$SEAM_MODE" = "interactive" ]; then
        print_info "MIRA needs your admin password once to upgrade."
    fi
    if [ "$(uname)" = "Darwin" ]; then
        # The only sudo work on macOS is the root-owned mira wrapper rewrite
        # (CLI_NEEDS_SUDO=yes is this arm's entry condition); the service is
        # driven as the user, no sudo. Name it, not the installer's packages.
        acquire_sudo \
            "MIRA needs sudo to rewrite the ${MIRA_CLI_PATH} command." \
            "Enter your password once — the rest of the update runs unattended."
    else
        acquire_sudo \
            "MIRA needs sudo to stop and start the MIRA service and rewrite the ${MIRA_CLI_PATH} and service files when they change." \
            "Enter your password once — the rest of the update runs unattended."
    fi
    # acquire_sudo installs its own EXIT trap for the askpass helper; on_exit
    # removes the helper itself, so the rollback trap is reinstated here.
    trap on_exit EXIT
fi
if [ "$CANONICAL" = "yes" ]; then
    print_header "Stopping MIRA"
    SERVICE_STOPPED="yes"
    stop_mira_service
    print_success "MIRA stopped"
else
    print_info "Non-canonical root ${APP_DIR}: no service stop (the real install's service is never touched)."
    SERVICE_STOPPED="yes"
fi

# 4.5 Save the operator's changes (nothing is replaced until this succeeds)
print_header "Checking for your own changes"
EXPORT_ROOT="$HOME"
[ -d "$HOME/Desktop" ] && [ -w "$HOME/Desktop" ] && EXPORT_ROOT="$HOME/Desktop"
FAILURE_KIND="preserve"
# The lib's free-space refusal prints this caller's remedy.
CUSTOM_EXPORT_REMEDY="run: mira update"
export_custom_build "$APP_DIR" "$BASELINE" "$TAG" "$STAMP" "$EXPORT_ROOT"
DRIFT_CATALOG="${APP_DIR}/logs/update-${STAMP}-venv-drift.txt"
if [ -n "$CUSTOM_EXPORT_DIR" ]; then
    DRIFT_CATALOG="${CUSTOM_EXPORT_DIR}/RESTORE_MY_CUSTOM_BUILD.txt"
    print_custom_alert "$CUSTOM_EXPORT_DIR" "$CUSTOM_EXPORT_COUNT"
else
    print_info "No changes of your own to save."
fi
if [ "$SERVER_VENV_ACTION" = "rebuild" ] && [ -x "${OLD_VENV}/bin/python3" ]; then
    record_venv_drift "server" "$OLD_VENV" "$NEW_VENV" "$DRIFT_CATALOG"
    if [ "$VENV_DRIFT_LOST" -gt 0 ]; then
        print_warning "${VENV_DRIFT_LOST} Python packages installed by hand in the server venv are not part of this release; the list is in ${DRIFT_CATALOG}."
    fi
fi
if [ "$TUI_VENV_ACTION" = "rebuild" ] && [ -x "${TUI_VENV}/bin/python3" ]; then
    record_venv_drift "TUI client" "$TUI_VENV" "$NEW_TUI_VENV" "$DRIFT_CATALOG"
fi
FAILURE_KIND="rollback"

# 5. Swap rebuilt venvs; refresh the TOAST shim in the live server venv
print_header "Installing ${TAG}"
if [ "$SERVER_VENV_ACTION" = "rebuild" ]; then
    SERVER_VENV_SWAPPED="yes"
    [ -e "$OLD_VENV" ] && mv "$OLD_VENV" "${OLD_VENV}.pre-update-${STAMP}"
    mv "$NEW_VENV" "$OLD_VENV"
    print_success "Server venv replaced"
fi
if [ "$TUI_VENV_ACTION" = "rebuild" ]; then
    TUI_VENV_SWAPPED="yes"
    [ -e "$TUI_VENV" ] && mv "$TUI_VENV" "${TUI_VENV}.pre-update-${STAMP}"
    mv "$NEW_TUI_VENV" "$TUI_VENV"
    print_success "TUI client venv replaced"
fi
LIVE_SITE_PACKAGES="$("$OLD_VENV/bin/python3" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
mkdir -p "${WORK_DIR}/toast"
for f in _mira_log_levels.py mira-log-levels.pth; do
    [ -f "${LIVE_SITE_PACKAGES}/${f}" ] && cp -p "${LIVE_SITE_PACKAGES}/${f}" "${WORK_DIR}/toast/"
done
TOAST_REFRESHED="yes"
cp "${TREE}/deploy/_mira_log_levels.py" "${LIVE_SITE_PACKAGES}/"
echo "import _mira_log_levels" > "${LIVE_SITE_PACKAGES}/mira-log-levels.pth"

# 6. Swap the code; rewrite the `mira` command
if [ -f "${APP_DIR}/.git/index" ]; then
    cp -p "${APP_DIR}/.git/index" "${WORK_DIR}/index"
    INDEX_SAVED="yes"
fi
CODE_MOVED="yes"
run_with_status "Setting the previous program files aside" move_code_aside "$APP_DIR" "$CODE_ASIDE"
OVERLAY_STARTED="yes"
run_with_status "Overlaying ${TAG}" \
    bash -c "tar -C '${TREE}' --exclude=.git --exclude=venv --exclude=__pycache__ \
        --exclude='*.pyc' --exclude=.env --exclude=.claude --exclude=.DS_Store \
        --exclude=data --exclude=logs --exclude=scratch \
        -cf - . | tar -C '${APP_DIR}' -xf -"
git -C "$APP_DIR" add -A
UPDATE_COMMITTED="yes"
git -C "$APP_DIR" commit --quiet --allow-empty \
    -m "$(official_commit_message update "$TAG" "$NEW_SHA")"
print_success "Recorded $(git -C "$APP_DIR" log -1 --format=%s) in the install's history"
if [ "$CLI_REWRITE" = "yes" ]; then
    cp -p "$MIRA_CLI_PATH" "${WORK_DIR}/mira-cli.old"
    CLI_REWRITTEN="yes"
    run_with_status "Updating the mira command" install_host_file "${WORK_DIR}/mira-cli.new" "$MIRA_CLI_PATH"
fi
if [ "$UNIT_REWRITE" = "yes" ]; then
    cp -p "$MIRA_UNIT_PATH" "${WORK_DIR}/mira-unit.old"
    UNIT_REWRITTEN="yes"
    run_with_status "Updating the MIRA service file" install_host_file "${WORK_DIR}/mira-unit.new" "$MIRA_UNIT_PATH" 644
    sudo systemctl daemon-reload
fi

# Durability barrier: everything above (venvs, code tree, git history, mira
# command, unit) is still page-cache-only, and the database phase below can
# run for minutes — a power cut inside that window left zero-byte program
# files and destroyed every recovery path with them (campaign finding KVM-3).
run_with_status "Making the new program files durable on disk" sync

# 6.5 Database steps (the runner backs up before its first change)
[ "$CANONICAL" = "yes" ] || {
    print_error "Non-canonical root ${APP_DIR}: no database steps — MIRA_APP_DIR scopes files, never the database, and the host install's database is never touched on a throwaway root's behalf."
    exit 3
}
print_header "Upgrading the database"
MIGRATE_ARGS=(migrate --stamp "$STAMP" --backups-dir "$BACKUPS_DIR" --env-file "$ENV_FILE")
[ "$ADOPT_DRIFTED" = "yes" ] && MIGRATE_ARGS+=(--adopt-drifted)
[ "${NIGHTLY:-0}" = "1" ] && MIGRATE_ARGS+=(--applied-by "$TAG")
[ -f "$ENV_FILE" ] && cp -p "$ENV_FILE" "${WORK_DIR}/systemone.env.pre-update"
MIGRATE_STARTED="yes"
runner "$OLD_VENV/bin/python3" "$APP_DIR" "${MIGRATE_ARGS[@]}"
runner "$OLD_VENV/bin/python3" "$APP_DIR" reconcile --env-file "$ENV_FILE"

# 7. Restart, health-poll, and confirm the database is at this release's head
if [ "$CANONICAL" = "yes" ]; then
    print_header "Starting MIRA"
    NEW_SERVICE_STARTED="yes"
    # Guarded, not bare: a start failure must reach the health poll below
    # (which fails, rolls back, and prints the logs pointer) instead of
    # aborting here under set -e with the poll dead.
    start_mira_service || true
    if ! wait_healthy "$NEW_VERSION"; then
        print_info "Logs: tail -100 /opt/mira/logs/mira-launchd.log (macOS) or journalctl -u mira -n 100 (Linux)"
        exit 1
    fi
else
    print_info "Non-canonical root: restart manually and verify ${HEALTH_URL} reports ${NEW_VERSION}."
fi
[ "$CANONICAL" = "yes" ] || {
    print_error "Non-canonical root ${APP_DIR}: no database head-check — the host install's database is never touched on a throwaway root's behalf."
    exit 3
}
STATUS_JSON="$(runner "$OLD_VENV/bin/python3" "$APP_DIR" status --json 2>> "$UPDATE_LOG" | tail -1)"
AT_HEAD="$("$OLD_VENV/bin/python3" -c 'import json, sys; r = json.loads(sys.stdin.read()); blocking = [x for x in r["refusals"] if x.get("code") != "PAYLOAD"]; print("yes" if r["ledger_head"] == r["tree_head"] and not r["pending"] and not blocking else "no")' <<< "$STATUS_JSON" 2>/dev/null || echo no)"
if [ "$AT_HEAD" != "yes" ]; then
    print_error "MIRA started, but its database is not at this release's last step."
    echo "status: ${STATUS_JSON}" >> "$UPDATE_LOG"
    exit 1
fi

# 8. Success
SUCCEEDED="yes"
# Non-fatal: the update is complete — a cleanup failure must not report it
# as failed. The aside survives, holding the operator's pre-update files.
rm -rf "$CODE_ASIDE" "${OLD_VENV}.pre-update-${STAMP}" "${TUI_VENV}.pre-update-${STAMP}" \
    || print_warning "Could not remove the pre-update copies (${CODE_ASIDE}, ${OLD_VENV}.pre-update-${STAMP}, ${TUI_VENV}.pre-update-${STAMP}) — they are preserved; remove them by hand once you have checked them."

# Nightly mode: record the installed branch-head SHA. Written only now —
# after the health poll passed — so a failed update does not advance the
# stamp and `mira update --nightly` retries the same SHA.
if [ "${NIGHTLY:-0}" = "1" ]; then
    mkdir -p "${APP_DIR}/data"
    printf '%s\n' "${NIGHTLY_SHA}" > "${APP_DIR}/data/nightly_stamp"
fi

print_success "MIRA updated: ${CUR_VERSION} → ${NEW_VERSION}"
exit 0
