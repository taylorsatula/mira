# deploy/preflight.sh
# Final pre-flight validation and sudo elevation
# Source this file - do not execute directly
#
# Requires: lib/output.sh sourced first
# Requires: OS, DISTRO, LOUD_MODE variables set (from config.sh)

# Validate required variables
: "${OS:?Error: OS must be set (run config.sh first)}"

print_header "System Detection"

# Display detected operating system
echo -ne "${DIM}${ARROW}${RESET} Detecting operating system... "
case "$OS" in
    linux)
        case "$DISTRO" in
            debian)
                echo -e "${CHECKMARK} ${DIM}Linux (Debian/Ubuntu)${RESET}"
                ;;
            fedora)
                echo -e "${CHECKMARK} ${DIM}Linux (Fedora/RHEL)${RESET}"
                ;;
            *)
                echo -e "${ERROR}"
                print_error "Unsupported Linux distribution"
                print_info "Detected: $([ -f /etc/os-release ] && . /etc/os-release && echo "$PRETTY_NAME" || echo "Unknown")"
                print_info "Supported: Debian/Ubuntu, Fedora/RHEL/CentOS/Rocky/Alma"
                print_info "For other distros, see manual installation: docs/MANUAL_INSTALL.md"
                exit 1
                ;;
        esac
        ;;
    macos)
        echo -e "${CHECKMARK} ${DIM}macOS${RESET}"
        ;;
esac

# Check if running as root
echo -ne "${DIM}${ARROW}${RESET} Checking user privileges... "
if [ "$EUID" -eq 0 ]; then
   echo -e "${ERROR}"
   print_error "Please do not run this script as root."
   exit 1
fi
echo -e "${CHECKMARK}"

print_header "Beginning Installation"

print_info "This script requires sudo privileges for system package installation."
print_info "Please enter your password - the installation will then run unattended."
echo ""
# Headless installs (nohup/CI) have no tty. `sudo -v` is a timestamp
# operation that demands a terminal even when NOPASSWD satisfies
# elevation (sudo 1.9.x, timestamp_type=tty): it aborts headless runs
# with "a terminal is required to authenticate". Probe elevation with
# a real command instead; prompt for a password only when the probe
# shows one is needed (interactive tty runs).
if ! sudo -n true 2>/dev/null; then
    sudo -v
fi

# Keep sudo alive (Linux only)
if [ "$OS" = "linux" ]; then
    while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &
fi

echo ""
print_success "All configuration collected"
print_info "Installation will now proceed unattended (estimated 10-15 minutes)"
print_info "Progress will be displayed as each step completes"
[ "$LOUD_MODE" = false ] && print_info "Use --loud flag to see detailed output"
echo ""
sleep 1

echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${DIM}Some of these steps will take a long time. If the spinner is still going, it hasn't${RESET}"
echo -e "${DIM}error'd or timed out—everything is okay. It could take 15 minutes or more to complete.${RESET}"
echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
