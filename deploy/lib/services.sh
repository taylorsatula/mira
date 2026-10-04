# deploy/lib/services.sh
# Service management and filesystem helper functions
# Source this file - do not execute directly
#
# Requires: lib/output.sh sourced first
# Requires: OS variable set for db/db_user checks

# Port occupancy probe — the single sanctioned mechanism for every caller.
# Tool ladder: lsof → ss → netstat. lsof is guaranteed on macOS; ss (iproute2)
# is guaranteed on every supported Linux target; netstat is the legacy rung.
# A clean minimal system may have only ss — probing with lsof alone made the
# pre-package-install port check fail on stock Debian/Fedora images.
# Returns: 0 = occupied, 1 = free, 2 = indeterminate (no probe tool at all;
# callers must fail loud, never treat 2 as free).
# Usage: port_probe_status PORT
port_probe_status() {
    local port="$1"
    if command -v lsof &> /dev/null; then
        lsof -Pi ":$port" -sTCP:LISTEN -t &> /dev/null && return 0 || return 1
    elif command -v ss &> /dev/null; then
        ss -ltn "sport = :$port" | tail -n +2 | grep -q . && return 0 || return 1
    elif command -v netstat &> /dev/null; then
        # Portable form: GNU renders :PORT, BSD/macOS renders .PORT; -ltn is
        # GNU-only, so parse the generic listing and require a LISTEN row.
        netstat -an | grep -i listen | grep -qE "[:.]$port( |\$)" && return 0 || return 1
    fi
    return 2
}

# Capture sudo elevation ONCE, at install start, and make every later sudo call
# silent. Homebrew clears the sudo credential timestamp on every touchpoint
# (observed live on brew 7.0.7: `sudo -n true` fails immediately after any brew
# command), so a deploy that interleaves brew and sudo — every macOS install —
# re-prompts at its next sudo unless the password is held for re-priming.
#
# Mechanism: the password is read once and handed to sudo through a private
# SUDO_ASKPASS helper (a 0700 temp dir, 0600 file, removed on exit — never an
# env var, never re-read from the tty), and the `sudo()` shell function wraps
# every later call with -A so a cleared ticket is re-primed with no prompt.
# `sudo -k` first drops any cached ticket so the probe answers "is this account
# NOPASSWD?" rather than "is a ticket still valid?". Idempotent: MIRA_SUDO_READY
# short-circuits a second call.
acquire_sudo() {
    [ "${MIRA_SUDO_READY:-}" = "yes" ] && return 0

    sudo -k 2>/dev/null || true
    if sudo -n true 2>/dev/null; then
        MIRA_SUDO_READY="yes"
        return 0
    fi

    if [ ! -t 0 ]; then
        print_error "sudo requires a password but this session has no terminal."
        print_info "Run the installer from a terminal, or configure passwordless sudo"
        print_info "for this account for unattended/CI installs."
        exit 1
    fi

    echo ""
    print_info "This installer needs sudo for system packages."
    print_info "Enter your password once — the rest of the install runs unattended."
    echo ""

    local pass="" attempts=0 primed="no"
    while [ "$attempts" -lt 3 ]; do
        pass=""
        read -r -s -p "$(echo -e ${CYAN}Password${RESET}) (sudo): " pass || { echo ""; break; }
        echo ""
        if printf '%s\n' "$pass" | sudo -S -v > /dev/null 2>&1; then
            primed="yes"
            break
        fi
        attempts=$((attempts + 1))
        [ "$attempts" -lt 3 ] && print_warning "Incorrect password, try again."
    done
    if [ "$primed" != "yes" ]; then
        print_error "sudo authentication failed."
        exit 1
    fi

    MIRA_SUDO_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mira-sudo.XXXXXX")"
    chmod 700 "$MIRA_SUDO_DIR"
    printf '%s\n' "$pass" > "$MIRA_SUDO_DIR/pass"
    chmod 600 "$MIRA_SUDO_DIR/pass"
    printf '#!/bin/sh\ncat %s\n' "$MIRA_SUDO_DIR/pass" > "$MIRA_SUDO_DIR/askpass"
    chmod 700 "$MIRA_SUDO_DIR/askpass"
    export SUDO_ASKPASS="$MIRA_SUDO_DIR/askpass"
    pass=""

    # -A routes a cleared ticket through askpass; -n calls keep their "never
    # prompt" meaning (sudo does not consult askpass under -n).
    sudo() { command sudo -A "$@"; }
    export -f sudo

    trap 'rm -rf "${MIRA_SUDO_DIR:-}"' EXIT
    MIRA_SUDO_READY="yes"
}

# Re-establish the sudo credential timestamp after a Homebrew touchpoint
# cleared it. Silent when acquire_sudo captured the password (the -A wrapper
# re-primes through askpass); falls back to a terminal prompt when it did not.
ensure_sudo() {
    if sudo -n true 2>/dev/null; then
        return 0
    fi
    if sudo -v > /dev/null 2>&1; then
        return 0
    fi
    if [ -t 0 ]; then
        print_warning "Re-authenticating sudo (Homebrew clears the credential timestamp)..."
        if sudo -v; then
            return 0
        fi
    fi
    print_error "sudo credentials were lost (Homebrew clears the ticket) and could not be re-established."
    print_info "Re-run from a terminal, or configure passwordless sudo for unattended installs."
    exit 1
}

# (Re)load a per-user LaunchAgent idempotently — the one sanctioned launchd
# path for every MIRA agent. bootout is asynchronous: a bootstrap issued
# while the old instance is still tearing down fails with "Bootstrap failed:
# 5: Input/output error" (observed live on macOS 15). Wait for the removal
# to settle, then bootstrap + enable. Returns non-zero when the load fails.
# Usage: launchd_reload_agent /path/to/<label>.plist
launchd_reload_agent() {
    local plist="$1"
    local label
    label=$(basename "$plist" .plist)
    launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
    local i
    for i in $(seq 1 20); do
        launchctl print "gui/$(id -u)/$label" > /dev/null 2>&1 || break
        sleep 0.5
    done
    launchctl bootstrap "gui/$(id -u)" "$plist" && launchctl enable "gui/$(id -u)/$label"
}

# CPU count for parallel builds: nproc on GNU/Linux, sysctl on macOS/BSD
# (stock macOS has no nproc). No silent fallback — if neither exists the
# build fails loudly, which is the honest signal.
cpu_count() {
    nproc 2>/dev/null || sysctl -n hw.ncpu
}

# Resolve the PostgreSQL systemd unit this host actually ships.
# PGDG installs (Fedora) provide postgresql-17.service; Debian/Ubuntu and
# Arch ship postgresql.service (postgresql-common wrapper / native unit).
# Existence-based (systemctl cat), not distro-name-based: a wrong name in a
# Requires= dependency makes systemd refuse to start mira.service.
# Echoes the unit name; returns 1 when neither unit exists.
resolve_pg_unit() {
    if systemctl cat postgresql-17.service > /dev/null 2>&1; then
        echo "postgresql-17.service"
    elif systemctl cat postgresql.service > /dev/null 2>&1; then
        echo "postgresql.service"
    else
        return 1
    fi
}

# Resolve the Valkey systemd unit this host actually ships.
# Ubuntu/Debian's valkey-server package installs valkey-server.service
# (redis-style naming); Fedora's and Arch's valkey package installs
# valkey.service. Echoes the unit name; returns 1 when neither exists.
resolve_valkey_unit() {
    if systemctl cat valkey-server.service > /dev/null 2>&1; then
        echo "valkey-server.service"
    elif systemctl cat valkey.service > /dev/null 2>&1; then
        echo "valkey.service"
    else
        return 1
    fi
}

# Check if something exists with consistent pattern
# Usage: check_exists TYPE TARGET [EXTRA]
# Types: file, dir, command, package, db, db_user, service_systemctl, service_brew
check_exists() {
    local type="$1"
    local target="$2"
    local extra="$3"

    case "$type" in
        file)
            [ -f "$target" ]
            ;;
        dir)
            [ -d "$target" ]
            ;;
        command)
            command -v "$target" &> /dev/null
            ;;
        package)
            venv/bin/pip3 show "$target" &> /dev/null
            ;;
        db)
            if [ "$OS" = "linux" ]; then
                sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw "$target"
            else
                psql -lqt | cut -d \| -f 1 | grep -qw "$target"
            fi
            ;;
        db_user)
            if [ "$OS" = "linux" ]; then
                sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$target'" | grep -q 1
            else
                psql postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='$target'" 2>/dev/null | grep -q 1
            fi
            ;;
        service_systemctl)
            systemctl is-active --quiet "$target" 2>/dev/null
            ;;
        service_brew)
            brew services list 2>/dev/null | grep -q "${target}.*started"
            ;;
    esac
}

# Start service with idempotency check
# Usage: start_service SERVICE_NAME SERVICE_TYPE
# Types: systemctl, brew, background (for custom processes)
start_service() {
    local service_name="$1"
    local service_type="$2"

    case "$service_type" in
        systemctl)
            if check_exists service_systemctl "$service_name"; then
                print_info "$service_name already running"
                return 0
            fi
            run_with_status "Starting $service_name" \
                sudo systemctl start "$service_name"
            ;;
        brew)
            if check_exists service_brew "$service_name"; then
                print_info "$service_name already running"
                return 0
            fi
            run_with_status "Starting $service_name" \
                brew services start "$service_name"
            ;;
        background)
            print_error "Background service type requires custom implementation"
            return 1
            ;;
    esac
}

# Stop service with consistent pattern
# Usage: stop_service SERVICE_NAME SERVICE_TYPE [EXTRA]
# Types: systemctl, brew, pid_file (EXTRA=pid_file_path), port (EXTRA=port_number)
stop_service() {
    local service_name="$1"
    local service_type="$2"
    local extra="$3"

    case "$service_type" in
        systemctl)
            if ! check_exists service_systemctl "$service_name"; then
                return 0  # Already stopped
            fi
            run_with_status "Stopping $service_name" \
                sudo systemctl stop "$service_name"
            ;;
        brew)
            if ! check_exists service_brew "$service_name"; then
                return 0  # Already stopped
            fi
            run_with_status "Stopping $service_name" \
                brew services stop "$service_name"
            ;;
        pid_file)
            local pid_file="$extra"
            if [ ! -f "$pid_file" ]; then
                return 0  # PID file doesn't exist
            fi
            local pid=$(cat "$pid_file")
            if ! kill -0 "$pid" 2>/dev/null; then
                rm -f "$pid_file"  # Clean up stale PID file
                return 0
            fi
            kill "$pid" 2>/dev/null && rm -f "$pid_file"
            ;;
        port)
            local port="$extra"
            local pids=""
            # `|| true`: an empty result makes lsof/grep exit non-zero, which
            # must read as "nothing on port", never as a failed command.
            if command -v lsof &> /dev/null; then
                pids=$(lsof -ti ":$port" 2>/dev/null || true)
            elif command -v ss &> /dev/null; then
                # Non-root sees only own-user pids here — same visibility as lsof
                pids=$(ss -ltnp "sport = :$port" 2>/dev/null | tail -n +2 \
                    | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)
            fi
            if [ -z "$pids" ]; then
                return 0  # Nothing on port
            fi
            kill $pids 2>/dev/null || true
            ;;
    esac
}

# Write file only if content has changed
# Usage: write_file_if_changed FILEPATH CONTENT
write_file_if_changed() {
    local target_file="$1"
    local content="$2"

    if [ -f "$target_file" ]; then
        local existing_content=$(cat "$target_file")
        if [ "$existing_content" = "$content" ]; then
            return 1  # File unchanged
        fi
    fi

    echo "$content" > "$target_file"
    return 0
}

# Install Python package if not already installed
# Usage: install_python_package PACKAGE_NAME
install_python_package() {
    local package="$1"

    if check_exists package "$package"; then
        local version=$(venv/bin/pip3 show "$package" | grep Version | awk '{print $2}')
        echo -e "${CHECKMARK} ${DIM}$version (already installed)${RESET}"
        return 0
    fi

    if [ "$LOUD_MODE" = true ]; then
        print_step "Installing $package..."
        venv/bin/pip3 install "$package"
    else
        (venv/bin/pip3 install -q "$package") &
        show_progress $! "Installing $package"
    fi
}

# Live models-list check for the OpenAI-compatible provider being configured:
# prefill CONFIG_PROVIDER_MODEL with the suggested model only when the
# provider's models endpoint actually lists it. On a missing model or a failed
# request the variable is left unset — the caller's empty-model guard is the
# fail-fast, never a silent fallback to an unverified default.
# Requires: output.sh sourced; CONFIG_PROVIDER_ENDPOINT/KEY/NAME set.
prefill_provider_model() {
    local suggested_model="$1"

    if [ -n "$CONFIG_PROVIDER_MODEL" ]; then
        return 0
    fi

    local models_url="${CONFIG_PROVIDER_ENDPOINT%/chat/completions}/models"
    local auth_header=()
    if [ -n "$CONFIG_PROVIDER_KEY" ]; then
        auth_header=(-H "Authorization: Bearer $CONFIG_PROVIDER_KEY")
    fi

    local response
    if response=$(curl -fsS --max-time 15 "${auth_header[@]}" "$models_url" 2>/dev/null) \
        && printf '%s' "$response" | grep -q "\"$suggested_model\""; then
        export CONFIG_PROVIDER_MODEL="$suggested_model"
        print_success "Prefilled model '$suggested_model' (verified against ${CONFIG_PROVIDER_NAME}'s model list)"
    else
        print_warning "Could not confirm '$suggested_model' on ${CONFIG_PROVIDER_NAME}'s model list."
        print_info "Visit your provider's website, pick a model it serves, and enter it when prompted (or set MIRA_PROVIDER_MODEL)."
    fi
}
