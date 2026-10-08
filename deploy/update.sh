#!/bin/bash
# deploy/update.sh — the machine half of `mira update`: update a deployed
# MIRA install in place. Executed (not sourced) by the TUI's `mira update`
# subcommand (tui/update.py), which resolved and downloaded the release tree
# this script receives. It can also be run by hand:
#
#   bash <release-tree>/deploy/update.sh <release-tree> <tag>
#
# Never run against a tree carrying BREAKING.md — a breaking release is
# outside this script's contract (reinstall + agent-run history import;
# deploy/RELEASE.md owns the marker). The TUI gates on it; this script
# re-checks, so a direct invocation cannot bypass the gate.
#
# What it does, in order:
#   1. Preflight: both trees present, versions differ, no BREAKING.md.
#   2. Detect venv extras of the OLD install (torch, sentence-transformers,
#      playwright) so a rebuild reproduces a local-embedding or browser
#      install instead of silently stripping it.
#   3. Build a fresh server venv (venv.new) and TUI venv (tui-venv.new) from
#      the new release's requirements — while the server still runs, so the
#      service is down only for the swap itself.
#   4. Stop the service (macOS: launchctl bootout — unloads the agent, so
#      KeepAlive cannot restart it mid-swap; Linux: systemctl stop).
#   5. Move the old venvs aside (instant, same volume), swap the new ones in.
#   6. Swap the code: stash uncommitted edits if the install tree is a git
#      repo, clear everything except the host-local state (venv, .env, data,
#      logs, .git — plus run.sh, which finalize.sh generates into the tree at
#      install time), overlay the new release, record `update <tag>` in the
#      install repo.
#   7. Restart the service and health-poll: /v0/api/health must report the
#      new VERSION within 600 s. On success the aside venvs are removed; on
#      failure they are kept and the rollback commands are printed (v1 never
#      auto-rolls back).
#
# Privileges: none. /opt/mira is owned by the installing user (python.sh
# chowns it at install), and `mira update` runs as that user — a permission
# error means the wrong account and aborts loudly. The only platform needing
# elevation is Linux systemd service control, which is attempted
# non-interactively (`sudo -n`) and aborts before anything is touched when
# that fails.
#
# Environment:
#   MIRA_APP_DIR   install root        (default /opt/mira/app)
#   MIRA_TUI_VENV  TUI client venv      (default /opt/mira/tui-venv)
#   NIGHTLY        set to 1 by `mira update --nightly`: the version-equality
#                  no-op is skipped (a nightly tree can carry the same VERSION
#                  as the installed tree; only the code differs) and, on a
#                  successful update, NIGHTLY_SHA is recorded in
#                  <APP_DIR>/data/nightly_stamp.
#   NIGHTLY_SHA    the resolved OSS main-head SHA (passed by tui/update.py;
#                  written to the stamp after the health poll passes)
#   LOUD_MODE      true for verbose output (deploy lib convention)
#
# Service management (stop, restart, health-poll) applies ONLY to the
# canonical root /opt/mira/app — a throwaway root (verification probes,
# staged installs) has no service, and this script must never stop or
# restart the real one on its behalf.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOUD_MODE="${LOUD_MODE:-false}"
source "${SCRIPT_DIR}/lib/output.sh"
source "${SCRIPT_DIR}/lib/services.sh"

APP_DIR="${MIRA_APP_DIR:-/opt/mira/app}"
TUI_VENV="${MIRA_TUI_VENV:-/opt/mira/tui-venv}"
HEALTH_URL="http://127.0.0.1:1993/v0/api/health"
CANONICAL="no"
[ "$APP_DIR" = "/opt/mira/app" ] && CANONICAL="yes"

if [ $# -ne 2 ]; then
    echo "usage: $0 <release-tree> <tag>" >&2
    exit 1
fi
TREE="$1"
TAG="$2"

print_header "MIRA in-place update → ${TAG}"

# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------
for f in main.py requirements.txt VERSION deploy/_mira_log_levels.py tui/requirements.txt; do
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
if [ -f "${TREE}/BREAKING.md" ]; then
    print_error "Release ${TAG} carries BREAKING.md — breaking releases are never auto-updated."
    print_info "Reinstall via install.sh, then ask MIRA to import its history:"
    print_info "  deploy/HOW_TO_MIGRATE_OLD_INSTALLS.txt (in the new install)"
    exit 1
fi
NEW_VERSION="$(cat "${TREE}/VERSION")"
CUR_VERSION="$(cat "${APP_DIR}/VERSION")"
if [ "$NEW_VERSION" = "$CUR_VERSION" ] && [ "${NIGHTLY:-0}" != "1" ]; then
    print_info "Already at ${CUR_VERSION} — nothing to do."
    exit 0
fi
print_info "Updating ${CUR_VERSION} → ${NEW_VERSION} in ${APP_DIR}"

# ---------------------------------------------------------------------------
# 2. Old-venv extras: reproduce a local-embedding / browser install
# ---------------------------------------------------------------------------
# torch and sentence-transformers are deliberately NOT in requirements.txt
# (they ride the CONFIG_EMBEDDING_PROVIDER=local install choice), and neither
# is the playwright package. The old venv is the only record of that choice —
# read it before anything replaces the venv.
OLD_VENV="${APP_DIR}/venv"
HAD_TORCH="no"; HAD_ST="no"; HAD_PLAYWRIGHT="no"
if [ -f "${OLD_VENV}/bin/python3" ]; then
    "${OLD_VENV}/bin/python3" -c "import torch" 2>/dev/null && HAD_TORCH="yes"
    "${OLD_VENV}/bin/python3" -c "import sentence_transformers" 2>/dev/null && HAD_ST="yes"
    "${OLD_VENV}/bin/python3" -c "import playwright" 2>/dev/null && HAD_PLAYWRIGHT="yes"
fi
if [ "$HAD_TORCH" = "yes" ] || [ "$HAD_ST" = "yes" ]; then
    print_info "Old install has local-embedding extras (torch=${HAD_TORCH}, sentence-transformers=${HAD_ST}) — the rebuild will reinstall them."
fi
if [ "$HAD_PLAYWRIGHT" = "yes" ]; then
    print_info "Old install has the playwright package — the rebuild will reinstall it (the browser cache lives outside the venv and survives)."
fi

# The interpreter for the new venvs: the base interpreter the old venv runs
# on, so an install on a keg-only Homebrew Python does not silently jump to a
# different minor version. Fall back to PATH python3 with the 3.12 floor.
if [ -f "${OLD_VENV}/bin/python3" ]; then
    OLD_BASE="$("${OLD_VENV}/bin/python3" -c 'import sys; print(sys.base_prefix)')"
    PY="${OLD_BASE}/bin/python3"
fi
if [ ! -x "${PY:-}" ]; then
    PY="$(command -v python3 || true)"
    [ -n "$PY" ] || { print_error "No python3 found on PATH."; exit 1; }
fi
PY_VERSION="$("$PY" --version 2>&1 | awk '{print $2}')"
PY_MAJOR="$(echo "$PY_VERSION" | cut -d. -f1)"
PY_MINOR="$(echo "$PY_VERSION" | cut -d. -f2)"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 12 ]; }; then
    print_error "MIRA requires Python 3.12 or higher; found ${PY_VERSION} (${PY})."
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Build the new venvs (server still running)
# ---------------------------------------------------------------------------
build_venv() {
    # build_venv <dest> <requirements-file> <label>
    local dest="$1" reqfile="$2" label="$3"
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
            exit 1
        fi
    fi
}

print_header "Building new venvs"

NEW_VENV="${APP_DIR}/venv.new"
rm -rf "$NEW_VENV"
build_venv "$NEW_VENV" "${TREE}/requirements.txt" "server"

if [ "$HAD_TORCH" = "yes" ]; then
    ("$NEW_VENV/bin/python3" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu) &
    if ! show_progress $! "Reinstalling PyTorch (CPU)"; then
        print_error "Failed to reinstall torch for this local-embedding install."
        exit 1
    fi
fi
if [ "$HAD_ST" = "yes" ]; then
    ("$NEW_VENV/bin/python3" -m pip install -q sentence-transformers) &
    if ! show_progress $! "Reinstalling sentence-transformers"; then
        print_error "Failed to reinstall sentence-transformers for this local-embedding install."
        exit 1
    fi
fi
if [ "$HAD_PLAYWRIGHT" = "yes" ]; then
    ("$NEW_VENV/bin/python3" -m pip install -q playwright) &
    if ! show_progress $! "Reinstalling playwright"; then
        print_error "Failed to reinstall playwright for this install."
        exit 1
    fi
    # Idempotent and build-specific; the browser cache (~/.cache/ms-playwright)
    # survived the venv rebuild, so this is a fast no-op unless the pinned
    # playwright moved to a new build.
    ("$NEW_VENV/bin/playwright" install chromium > /dev/null 2>&1) &
    if ! show_progress $! "Ensuring Chromium is cached"; then
        print_error "playwright install chromium failed."
        exit 1
    fi
fi

# The TOAST log level must exist at interpreter startup, before any
# application import — reinstall the .pth hook into the new venv
# (python.sh Step 5 does this at install time).
SITE_PACKAGES="$($NEW_VENV/bin/python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
run_with_status "Installing TOAST log level into the new venv" \
    cp "${TREE}/deploy/_mira_log_levels.py" "${SITE_PACKAGES}/"
echo "import _mira_log_levels" > "${SITE_PACKAGES}/mira-log-levels.pth"

NEW_TUI_VENV="${TUI_VENV}.new"
rm -rf "$NEW_TUI_VENV"
build_venv "$NEW_TUI_VENV" "${TREE}/tui/requirements.txt" "TUI client"

# ---------------------------------------------------------------------------
# 4. Stop the service (canonical root only)
# ---------------------------------------------------------------------------
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [ "$CANONICAL" = "yes" ]; then
    print_header "Stopping MIRA"
    if [ "$(uname)" = "Darwin" ]; then
        PLIST="$HOME/Library/LaunchAgents/com.mira.app.plist"
        if [ -f "$PLIST" ]; then
            # bootout, not kill: unloading the agent takes KeepAlive out of
            # the picture entirely, so a nonzero exit cannot restart the
            # server onto a half-swapped tree mid-update.
            launchctl bootout "gui/$(id -u)/com.mira.app" 2>/dev/null || true
            for i in $(seq 1 30); do
                if port_probe_status 1993; then sleep 1; else break; fi
            done
            print_success "MIRA stopped (launchd agent unloaded)"
        else
            stop_service mira port 1993
            print_info "No com.mira.app LaunchAgent — stopped by port."
        fi
    else
        if systemctl cat mira > /dev/null 2>&1; then
            if ! sudo -n systemctl stop mira; then
                print_error "Cannot stop mira.service without a sudo password."
                print_info "Run: sudo systemctl stop mira && mira update"
                exit 1
            fi
            print_success "mira.service stopped"
        else
            stop_service mira port 1993
            print_info "No systemd unit — stopped by port."
        fi
    fi
else
    print_info "Non-canonical root ${APP_DIR}: no service stop (the real install's service is never touched)."
fi

# ---------------------------------------------------------------------------
# 5. Swap the venvs
# ---------------------------------------------------------------------------
print_header "Swapping venvs"
if [ -d "${OLD_VENV}" ]; then
    run_with_status "Moving the old server venv aside" mv "${OLD_VENV}" "${OLD_VENV}.pre-update-${STAMP}"
fi
run_with_status "Activating the new server venv" mv "$NEW_VENV" "${OLD_VENV}"
if [ -e "$TUI_VENV" ]; then
    run_with_status "Moving the old TUI venv aside" mv "$TUI_VENV" "${TUI_VENV}.pre-update-${STAMP}"
fi
run_with_status "Activating the new TUI venv" mv "$NEW_TUI_VENV" "$TUI_VENV"

# ---------------------------------------------------------------------------
# 6. Swap the code
# ---------------------------------------------------------------------------
print_header "Installing ${TAG} code"
if [ -d "${APP_DIR}/.git" ] && [ -n "$(git -C "${APP_DIR}" status --porcelain --untracked-files=all 2>/dev/null)" ]; then
    run_with_status "Stashing uncommitted code changes" \
        git -C "${APP_DIR}" stash push --include-untracked --quiet \
            -m "uncommitted changes before update ${TAG}"
fi
# Preserve set = host-local state (python.sh Step 3 contract) plus run.sh
# (finalize.sh Step 15b generates it into the tree at install time; a code
# clear without it would delete the very launcher com.mira.app runs) and the
# aside venvs from this run.
run_with_status "Clearing previous code (state preserved)" \
    find "${APP_DIR}" -mindepth 1 -maxdepth 1 \
        ! -name venv ! -name .env ! -name data ! -name logs ! -name .git \
        ! -name run.sh ! -name 'venv*' ! -name 'tui-venv*' \
        -exec rm -rf {} +
run_with_status "Overlaying ${TAG}" \
    bash -c "tar -C '${TREE}' --exclude=.git --exclude=venv --exclude=__pycache__ \
        --exclude='*.pyc' --exclude=.env --exclude=.claude --exclude=.DS_Store \
        --exclude=data --exclude=logs --exclude=scratch \
        -cf - . | tar -C '${APP_DIR}' -xf -"
if [ -d "${APP_DIR}/.git" ]; then
    git -C "${APP_DIR}" add -A > /dev/null 2>&1
    git -C "${APP_DIR}" commit --quiet --allow-empty -m "update ${TAG}" \
        || print_warning "Could not record the update in the install repo's history (non-fatal)."
else
    print_info "Install tree is not a git repo — no update record written."
fi

# ---------------------------------------------------------------------------
# 7. Restart and health-poll (canonical root only)
# ---------------------------------------------------------------------------
if [ "$CANONICAL" = "yes" ]; then
    print_header "Starting MIRA"
    STARTED="no"
    if [ "$(uname)" = "Darwin" ]; then
        PLIST="$HOME/Library/LaunchAgents/com.mira.app.plist"
        if [ -f "$PLIST" ]; then
            if launchd_reload_agent "$PLIST"; then STARTED="yes"; fi
        else
            print_warning "No LaunchAgent plist — start manually: ${APP_DIR}/run.sh"
        fi
    else
        if systemctl cat mira > /dev/null 2>&1; then
            if sudo -n systemctl start mira; then STARTED="yes"; fi
        else
            print_warning "No systemd unit — start manually: ${APP_DIR}/run.sh"
        fi
    fi
    if [ "$STARTED" != "yes" ]; then
        print_error "Could not restart the service automatically."
        print_info "Rollback if needed: mv ${OLD_VENV} ${OLD_VENV}.failed; mv ${OLD_VENV}.pre-update-${STAMP} ${OLD_VENV}; git -C ${APP_DIR} reset --hard HEAD~1"
        exit 1
    fi
    echo -ne "${DIM}${ARROW}${RESET} Waiting for MIRA to become healthy... "
    HEALTHY=0
    for i in $(seq 1 120); do
        BODY="$(curl -sf "$HEALTH_URL" 2>/dev/null || true)"
        if printf '%s' "$BODY" | grep -q '"status":"healthy"' \
            && printf '%s' "$BODY" | grep -q "\"version\":\"${NEW_VERSION}\""; then
            HEALTHY=1
            break
        fi
        sleep 5
    done
    if [ "$HEALTHY" = 1 ]; then
        echo -e "${CHECKMARK} ${DIM}(healthy on ${NEW_VERSION} after ~$((i * 5))s)${RESET}"
    else
        echo -e "${ERROR}"
        print_error "MIRA did not report healthy on ${NEW_VERSION} within 600 s."
        print_info "Logs: tail -100 /opt/mira/logs/mira-launchd.log"
        print_info "Rollback: mv ${OLD_VENV} ${OLD_VENV}.failed; mv ${OLD_VENV}.pre-update-${STAMP} ${OLD_VENV}; git -C ${APP_DIR} reset --hard HEAD~1; then restart the service"
        exit 1
    fi
else
    print_info "Non-canonical root: restart manually and verify ${HEALTH_URL} reports ${NEW_VERSION}."
fi

# Success: drop the aside venvs.
if [ -d "${OLD_VENV}.pre-update-${STAMP}" ]; then
    rm -rf "${OLD_VENV}.pre-update-${STAMP}"
fi
if [ -e "${TUI_VENV}.pre-update-${STAMP}" ]; then
    rm -rf "${TUI_VENV}.pre-update-${STAMP}"
fi

# Nightly mode: record the installed branch-head SHA. Written only now —
# after the health poll passed — so a failed update does not advance the
# stamp and `mira update --nightly` retries the same SHA. The SHA comes
# from the environment (tui/update.py resolved it); it is never re-derived
# here.
if [ "${NIGHTLY:-0}" = "1" ] && [ -n "${NIGHTLY_SHA:-}" ]; then
    mkdir -p "${APP_DIR}/data"
    printf '%s\n' "${NIGHTLY_SHA}" > "${APP_DIR}/data/nightly_stamp"
fi

print_success "MIRA updated: ${CUR_VERSION} → ${NEW_VERSION}"
exit 0
