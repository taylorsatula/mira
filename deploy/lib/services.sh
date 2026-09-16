# deploy/lib/services.sh
# Service management and filesystem helper functions
# Source this file - do not execute directly
#
# Requires: lib/output.sh sourced first
# Requires: OS variable set for db/db_user checks

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
            if command -v lsof &> /dev/null; then
                local pids=$(lsof -ti ":$port" 2>/dev/null)
                if [ -z "$pids" ]; then
                    return 0  # Nothing on port
                fi
                kill $pids 2>/dev/null
            fi
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
