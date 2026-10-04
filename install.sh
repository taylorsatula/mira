#!/usr/bin/env bash
# =============================================================================
#  MIRA — installer entrypoint
# =============================================================================
#
#  Quick start:
#    curl -fsSL https://raw.githubusercontent.com/taylorsatula/mira/main/install.sh | bash
#
#  This is the stable front door to the deploy pipeline. It resolves the newest
#  published release, fetches that release's deploy/deploy.sh, and hands off to
#  it — all install logic lives in deploy/, so there is exactly one bootstrap
#  path to maintain.
#
#  Pass installer options through (they go to deploy/deploy.sh):
#    curl -fsSL .../install.sh | bash -s -- --config deploy-config.yml --loud
#
#  Pin a specific release:
#    curl -fsSL .../install.sh | MIRA_RELEASE_TAG=v2026.10.03-2.0 bash
#
#  Environment:
#    MIRA_RELEASE_TAG   install a specific tag instead of the newest release
#    MIRA_NO_COLOR=1    disable ANSI color
# =============================================================================

set -euo pipefail

REPO="taylorsatula/mira"
REPO_URL="https://github.com/${REPO}"
RAW_URL="https://raw.githubusercontent.com/${REPO}"

# -----------------------------------------------------------------------------
# Presentation
# -----------------------------------------------------------------------------
if [ -t 1 ] && [ -z "${MIRA_NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
    ESC=$'\033'
    RESET="${ESC}[0m"; DIM="${ESC}[2m"
    BLUE="${ESC}[38;5;75m"; CYAN="${ESC}[38;5;80m"
    GREEN="${ESC}[38;5;77m"; YELLOW="${ESC}[38;5;186m"; RED="${ESC}[38;5;203m"
else
    RESET=""; DIM=""; BLUE=""; CYAN=""; GREEN=""; YELLOW=""; RED=""
fi

CHECK="${GREEN}✓${RESET}"
ARROW="${CYAN}→${RESET}"
CROSS="${RED}✗${RESET}"
WARN="${YELLOW}⚠${RESET}"

banner() {
    printf '\n'
    printf '%s\n' "${BLUE}      ███╗   ███╗██╗██████╗  █████╗${RESET}"
    printf '%s\n' "${BLUE}      ████╗ ████║██║██╔══██╗██╔══██╗${RESET}"
    printf '%s\n' "${BLUE}      ██╔████╔██║██║██████╔╝███████║${RESET}"
    printf '%s\n' "${BLUE}      ██║╚██╔╝██║██║██╔══██╗██╔══██║${RESET}"
    printf '%s\n' "${BLUE}      ██║ ╚═╝ ██║██║██║  ██║██║  ██║${RESET}"
    printf '%s\n' "${BLUE}      ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝${RESET}"
    printf '%s\n' "${DIM}                installer${RESET}"
    printf '\n'
}

info() { printf '%s%s%s\n' "${DIM}" "$1" "${RESET}"; }
ok()   { printf '%s %s\n' "${CHECK}" "${GREEN}$1${RESET}"; }
warn() { printf '%s %s\n' "${WARN}" "${YELLOW}$1${RESET}" >&2; }
fail() { printf '%s %s\n' "${CROSS}" "${RED}$1${RESET}" >&2; }
die()  { fail "$1"; exit 1; }

# Run a command with a labelled outcome. Extra output is hidden; failures are
# reported by the caller so it can print recovery guidance.
usage() {
    printf 'MIRA installer\n\n'
    printf 'Usage:\n'
    printf '  curl -fsSL %s/main/install.sh | bash\n\n' "${RAW_URL}"
    printf 'Options are forwarded to deploy/deploy.sh:\n'
    printf '  --loud                 verbose installer output\n'
    printf '  --config <file>        non-interactive install from a config file\n'
    printf '  --local                install from the current directory\n\n'
    printf 'Environment:\n'
    printf '  MIRA_RELEASE_TAG       install a specific tag instead of the newest release\n'
}

# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

banner
step_no=1
step_total=3

[ -n "${BASH_VERSION:-}" ] || die "This installer must run under bash (pipe to 'bash', not 'sh')."

printf '%s %s Checking prerequisites... ' "${ARROW}" "[${step_no}/${step_total}]"
for cmd in curl sed; do
    command -v "$cmd" > /dev/null 2>&1 || { printf '%s\n' "${CROSS}"; die "Required command not found: ${cmd}"; }
done
printf '%s\n' "${CHECK}"
step_no=$((step_no + 1))

# -----------------------------------------------------------------------------
# Resolve the newest published release
# -----------------------------------------------------------------------------
printf '%s %s Resolving newest release... ' "${ARROW}" "[${step_no}/${step_total}]"
if [ -n "${MIRA_RELEASE_TAG:-}" ]; then
    TAG="${MIRA_RELEASE_TAG}"
    printf '%s %s\n' "${CHECK}" "${GREEN}${TAG}${RESET} ${DIM}(pinned)${RESET}"
else
    LATEST_URL="$(curl -fsSLo /dev/null -w '%{url_effective}' "${REPO_URL}/releases/latest" 2>/dev/null || true)"
    TAG="$(printf '%s' "${LATEST_URL}" | sed 's#.*/tag/##')"
    if [ -z "${TAG}" ] || [ "${TAG}" = "${LATEST_URL}" ]; then
        printf '%s\n' "${CROSS}"
        fail "Could not resolve the newest release from ${REPO_URL}/releases"
        info "Set MIRA_RELEASE_TAG to install a specific tag, e.g."
        info "  curl -fsSL ${RAW_URL}/main/install.sh | MIRA_RELEASE_TAG=v2026.10.03-2.0 bash"
        exit 1
    fi
    printf '%s %s\n' "${CHECK}" "${GREEN}${TAG}${RESET}"
fi
step_no=$((step_no + 1))

# -----------------------------------------------------------------------------
# Fetch that release's deploy.sh
# -----------------------------------------------------------------------------
printf '%s %s Fetching installer... ' "${ARROW}" "[${step_no}/${step_total}]"
TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t mira-install)"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

DEPLOY_URL="${RAW_URL}/refs/tags/${TAG}/deploy/deploy.sh"
if ! curl -fsSL "${DEPLOY_URL}" -o "${TMP_DIR}/deploy.sh" 2>/dev/null; then
    printf '%s\n' "${CROSS}"
    fail "Could not fetch ${DEPLOY_URL}"
    info "The release may not have finished publishing, or the tag name is wrong."
    exit 1
fi
if [ ! -s "${TMP_DIR}/deploy.sh" ] || [ "$(head -c 2 "${TMP_DIR}/deploy.sh")" != "#!" ]; then
    printf '%s\n' "${CROSS}"
    die "The downloaded installer is not a shell script."
fi
printf '%s\n' "${CHECK}"
chmod +x "${TMP_DIR}/deploy.sh"

# -----------------------------------------------------------------------------
# Hand off to deploy/deploy.sh
# -----------------------------------------------------------------------------
printf '\n'
ok "Installing MIRA ${TAG}"
info "The installer will guide you through configuration."
printf '\n'

# `curl | bash` leaves the script's stdin at EOF, so an interactive installer
# would read nothing. Reattach the handoff to the controlling terminal when one
# exists; a genuinely headless install (--config) still runs with plain stdin.
set +e
if [ -r /dev/tty ]; then
    bash "${TMP_DIR}/deploy.sh" "$@" < /dev/tty
    RC=$?
else
    bash "${TMP_DIR}/deploy.sh" "$@"
    RC=$?
fi
set -e

exit "${RC}"
