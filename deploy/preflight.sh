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
            arch)
                echo -e "${CHECKMARK} ${DIM}Linux (Arch)${RESET}"
                ;;
            *)
                echo -e "${ERROR}"
                print_error "Unsupported Linux distribution"
                print_info "Detected: $([ -f /etc/os-release ] && . /etc/os-release && echo "$PRETTY_NAME" || echo "Unknown")"
                print_info "Supported: Debian/Ubuntu, Fedora/RHEL/CentOS/Rocky/Alma, Arch"
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

# Elevation is captured once at install start (acquire_sudo, lib/services.sh):
# the password is held for silent re-priming, so the Homebrew phase cannot make
# sudo prompt again. This call is a no-op when deploy.sh already acquired it;
# it self-acquires when preflight.sh is run standalone.
acquire_sudo

# Keep the ticket fresh on every platform (Linux-only before, part of why macOS
# re-prompted after brew). A cleared ticket is re-primed silently.
while true; do
    sudo -n true 2>/dev/null || sudo -v > /dev/null 2>&1 || true
    sleep 60
    kill -0 "$$" || exit
done 2>/dev/null &

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
