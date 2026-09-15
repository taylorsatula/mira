# deploy/lib/output.sh
# Visual output functions for MIRA deployment
# Source this file - do not execute directly
#
# Requires: LOUD_MODE variable to be set (default: false)

# ANSI color codes (muted/professional palette)
RESET='\033[0m'
DIM='\033[2m'
BOLD='\033[1m'
GRAY='\033[38;5;240m'
BLUE='\033[38;5;75m'
GREEN='\033[38;5;77m'
YELLOW='\033[38;5;186m'
RED='\033[38;5;203m'
CYAN='\033[38;5;80m'

# Visual elements
CHECKMARK="${GREEN}✓${RESET}"
ARROW="${CYAN}→${RESET}"
WARNING="${YELLOW}⚠${RESET}"
ERROR="${RED}✗${RESET}"

# Print colored output
print_header() {
    echo -e "\n${BOLD}${BLUE}$1${RESET}"
}

print_step() {
    echo -e "${DIM}${ARROW}${RESET} $1"
}

print_success() {
    echo -e "${CHECKMARK} ${GREEN}$1${RESET}"
}

print_warning() {
    echo -e "${WARNING} ${YELLOW}$1${RESET}"
}

print_error() {
    echo -e "${ERROR} ${RED}$1${RESET}"
}

print_info() {
    echo -e "${DIM}  $1${RESET}"
}

# Execute command with optional output suppression
run_quiet() {
    if [ "$LOUD_MODE" = true ]; then
        "$@"
    else
        "$@" > /dev/null 2>&1
    fi
}

run_with_status() {
    local msg="$1"
    shift

    if [ "$LOUD_MODE" = true ]; then
        print_step "$msg"
        # Loud mode shows command output as it runs, but a command that
        # fails without printing anything (e.g. `test -s file`) would
        # otherwise die silently under set -e. Mark the failure.
        if ! "$@"; then
            echo -e "${ERROR} Command failed: $msg"
            return 1
        fi
    else
        echo -ne "${DIM}${ARROW}${RESET} $msg... "
        if "$@" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            return 1
        fi
    fi
}

# Progress spinner for long operations
show_progress() {
    local pid=$1
    local msg=$2
    local spin='-\|/'
    local i=0

    if [ "$LOUD_MODE" = true ]; then
        wait $pid
        return $?
    fi

    echo -ne "${DIM}${ARROW}${RESET} $msg... "
    while kill -0 $pid 2>/dev/null; do
        i=$(( (i+1) %4 ))
        echo -ne "\r${DIM}${ARROW}${RESET} $msg... ${spin:$i:1}"
        sleep 0.1
    done

    wait $pid
    local status=$?
    if [ $status -eq 0 ]; then
        echo -e "\r${DIM}${ARROW}${RESET} $msg... ${CHECKMARK}"
    else
        echo -e "\r${DIM}${ARROW}${RESET} $msg... ${ERROR}"
    fi
    return $status
}
