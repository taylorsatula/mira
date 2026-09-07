#!/bin/bash
# MIRA Docker Container Entrypoint
# Handles first-boot configuration, then hands off to s6-overlay
#
# First-boot detection: checks for /opt/vault/init-keys.txt
# - If absent: runs setup wizard (interactive or env-var based)
# - If present: proceeds directly to s6-overlay startup

set -e

# Source helper libraries for consistent formatting and utility functions
source /opt/mira/app/deploy/lib/output.sh
source /opt/mira/app/deploy/lib/services.sh
LOUD_MODE=false

# =============================================================================
# First-Boot Detection
# =============================================================================

is_first_boot() {
    # Vault initialization creates init-keys.txt - its presence means we're configured
    [ ! -f /opt/vault/init-keys.txt ]
}

has_tty() {
    # Check if both stdin and stdout are TTYs (docker run -it)
    [ -t 0 ] && [ -t 1 ]
}

# =============================================================================
# Environment Variable Configuration (non-interactive mode)
# =============================================================================

check_env_vars() {
    local missing=""

    if [ -z "$MIRA_ANTHROPIC_KEY" ]; then
        missing="$missing MIRA_ANTHROPIC_KEY"
    fi

    if [ -z "$MIRA_PROVIDER_KEY" ]; then
        missing="$missing MIRA_PROVIDER_KEY"
    fi

    if [ -n "$missing" ]; then
        print_error "Missing required environment variables:$missing"
        print_info ""
        print_info "Required environment variables for non-interactive setup:"
        print_info "  MIRA_ANTHROPIC_KEY    - Your Anthropic API key (sk-ant-...)"
        print_info "  MIRA_PROVIDER_KEY     - OpenAI-compatible provider API key (e.g., Groq)"
        print_info ""
        print_info "Optional environment variables:"
        print_info "  MIRA_ANTHROPIC_BATCH_KEY  - Separate batch API key (defaults to main key)"
        print_info "  MIRA_PROVIDER_NAME        - Provider name (default: Groq)"
        print_info "  MIRA_PROVIDER_ENDPOINT    - Custom endpoint URL"
        print_info "  MIRA_PROVIDER_MODEL       - Model name for non-Groq providers"
        print_info "  MIRA_KAGI_KEY             - Kagi search API key"
        print_info "  MIRA_DB_PASSWORD          - Database password"
        print_info ""
        print_info "Or run with -it flag for interactive setup:"
        print_info "  docker run -it -p 1993:1993 mira:latest"
        exit 1
    fi
}

setup_from_env_vars() {
    print_header "Configuring MIRA from Environment Variables"

    # Set defaults
    export CONFIG_ANTHROPIC_KEY="$MIRA_ANTHROPIC_KEY"
    export CONFIG_ANTHROPIC_BATCH_KEY="${MIRA_ANTHROPIC_BATCH_KEY:-$MIRA_ANTHROPIC_KEY}"
    export CONFIG_PROVIDER_KEY="$MIRA_PROVIDER_KEY"
    export CONFIG_PROVIDER_NAME="${MIRA_PROVIDER_NAME:-Groq}"
    export CONFIG_KAGI_KEY="${MIRA_KAGI_KEY:-}"
    export CONFIG_DB_PASSWORD="${MIRA_DB_PASSWORD:-changethisifdeployingpwd}"
    export CONFIG_OFFLINE_MODE="no"

    # Set provider endpoint based on name if not explicitly provided
    if [ -z "$MIRA_PROVIDER_ENDPOINT" ]; then
        case "$CONFIG_PROVIDER_NAME" in
            Groq)
                export CONFIG_PROVIDER_ENDPOINT="https://api.groq.com/openai/v1/chat/completions"
                ;;
            OpenRouter)
                export CONFIG_PROVIDER_ENDPOINT="https://openrouter.ai/api/v1/chat/completions"
                ;;
            "Together AI")
                export CONFIG_PROVIDER_ENDPOINT="https://api.together.xyz/v1/chat/completions"
                ;;
            "Fireworks AI")
                export CONFIG_PROVIDER_ENDPOINT="https://api.fireworks.ai/inference/v1/chat/completions"
                ;;
            Cerebras)
                export CONFIG_PROVIDER_ENDPOINT="https://api.cerebras.ai/v1/chat/completions"
                ;;
            SambaNova)
                export CONFIG_PROVIDER_ENDPOINT="https://api.sambanova.ai/v1/chat/completions"
                ;;
            *)
                export CONFIG_PROVIDER_ENDPOINT="${MIRA_PROVIDER_ENDPOINT:-https://api.groq.com/openai/v1/chat/completions}"
                ;;
        esac
    else
        export CONFIG_PROVIDER_ENDPOINT="$MIRA_PROVIDER_ENDPOINT"
    fi

    export CONFIG_PROVIDER_MODEL="${MIRA_PROVIDER_MODEL:-}"

    print_success "Configuration loaded from environment"
    print_info "  Anthropic Key: ****${CONFIG_ANTHROPIC_KEY: -4}"
    print_info "  Provider: $CONFIG_PROVIDER_NAME"
    print_info "  Provider Key: ****${CONFIG_PROVIDER_KEY: -4}"
}

# =============================================================================
# Service Initialization
# =============================================================================

init_postgresql() {
    print_header "Initializing PostgreSQL"

    # Ensure data directory has correct ownership
    chown -R postgres:postgres /var/lib/postgresql
    chmod 700 /var/lib/postgresql/17/main

    # Initialize PostgreSQL data directory if empty or missing config
    if [ ! -f /var/lib/postgresql/17/main/postgresql.conf ]; then
        print_step "Initializing PostgreSQL data directory..."

        # Clear any partial state from previous failed attempts
        rm -rf /var/lib/postgresql/17/main/*

        sudo -u postgres /usr/lib/postgresql/17/bin/initdb -D /var/lib/postgresql/17/main

        if [ $? -ne 0 ]; then
            print_error "initdb failed"
            exit 1
        fi

        # Configure pg_hba.conf for local connections
        cat > /var/lib/postgresql/17/main/pg_hba.conf <<'EOF'
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             postgres                                peer
local   all             all                                     scram-sha-256
host    all             all             127.0.0.1/32            scram-sha-256
host    all             all             ::1/128                 scram-sha-256
EOF
        chown postgres:postgres /var/lib/postgresql/17/main/pg_hba.conf
    fi

    # Start PostgreSQL temporarily for schema setup (use -w to wait for startup)
    print_step "Starting PostgreSQL for schema setup..."
    if ! sudo -u postgres /usr/lib/postgresql/17/bin/pg_ctl -D /var/lib/postgresql/17/main -l /tmp/pg_setup.log -w start; then
        print_error "PostgreSQL failed to start. Log output:"
        cat /tmp/pg_setup.log 2>/dev/null || echo "No log file found"
        exit 1
    fi

    # Double-check PostgreSQL is ready
    local attempts=0
    while ! sudo -u postgres pg_isready -q; do
        attempts=$((attempts + 1))
        if [ $attempts -gt 30 ]; then
            print_error "PostgreSQL not responding"
            cat /tmp/pg_setup.log 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done

    # Provision roles and database. The schema is a pure DDL contract that
    # assumes both already exist and targets an empty database. The roles are
    # created with the default sentinel password because pg_hba below uses
    # scram-sha-256 for TCP: a passwordless role could never authenticate the
    # postgresql:// URLs that init_vault stores in Vault.
    print_step "Provisioning database roles and database..."
    sudo -u postgres psql -d postgres -v ON_ERROR_STOP=1 -c "DO \$roles\$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mira_admin') THEN CREATE ROLE mira_admin LOGIN PASSWORD 'changethisifdeployingpwd' BYPASSRLS; END IF; IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mira_dbuser') THEN CREATE ROLE mira_dbuser LOGIN PASSWORD 'changethisifdeployingpwd'; END IF; END \$roles\$;"
    # Guard so a first boot that crashed before the schema ran resumes instead
    # of dying on "database already exists" (matches deploy/postgresql.sh). A
    # crash mid-schema still fails fast on the schema's emptiness guard.
    if ! sudo -u postgres psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = 'mira_service'" | grep -q 1; then
        sudo -u postgres createdb -O mira_admin mira_service
    fi

    # Apply schema
    print_step "Applying database schema..."
    sudo -u postgres psql -d mira_service -v ON_ERROR_STOP=1 -f /opt/mira/app/deploy/mira_service_schema.sql

    # Non-Groq provider: repoint the generic-key routes at the endpoint the
    # wizard collected. Postgres is running here, so the rewrite happens at
    # once — there is no deferred s6 step. This mirrors what the bare-metal
    # installer does (python.sh rewrites the seeded rows pre-schema;
    # postgresql.sh rewrites all five rows for offline installs). A container
    # install collects ONE generic provider key and model, so 'other' shares
    # the provider: like an air-gapped install, it has no genuinely different
    # outside vendor to route to. The anthropic routes ('batch',
    # 'assessment') keep the separately collected Anthropic key.
    if [ "$CONFIG_PROVIDER_NAME" != "Groq" ] && [ -n "$CONFIG_PROVIDER_ENDPOINT" ]; then
        if [ -z "$CONFIG_PROVIDER_MODEL" ]; then
            print_error "Provider '$CONFIG_PROVIDER_NAME' needs a model name (MIRA_PROVIDER_MODEL, or answer the wizard prompt); refusing to install routes that cannot resolve."
            exit 1
        fi
        print_step "Routing primary/fast/other model_configs at $CONFIG_PROVIDER_NAME..."
        sudo -u postgres psql -d mira_service -v ON_ERROR_STOP=1 -c "UPDATE model_configs SET dialect_name = 'openai', endpoint_url = '$CONFIG_PROVIDER_ENDPOINT', model = '$CONFIG_PROVIDER_MODEL' WHERE name IN ('primary', 'fast', 'other');"
    fi

    # Update passwords if custom password provided
    if [ "$CONFIG_DB_PASSWORD" != "changethisifdeployingpwd" ]; then
        print_step "Setting custom database password..."
        sudo -u postgres psql -c "ALTER ROLE mira_admin WITH PASSWORD '$CONFIG_DB_PASSWORD';"
        sudo -u postgres psql -c "ALTER ROLE mira_dbuser WITH PASSWORD '$CONFIG_DB_PASSWORD';"
    fi

    # Stop PostgreSQL (s6 will manage it from here)
    print_step "Stopping temporary PostgreSQL instance..."
    sudo -u postgres /usr/lib/postgresql/17/bin/pg_ctl -D /var/lib/postgresql/17/main stop

    print_success "PostgreSQL initialized"
}

init_vault() {
    print_header "Initializing HashiCorp Vault"

    # Source vault helper library
    source /opt/mira/app/deploy/lib/vault.sh

    # Ensure vault data directory exists
    mkdir -p /opt/vault/data
    chown -R mira:mira /opt/vault

    # Start Vault temporarily for initialization
    print_step "Starting Vault for initialization..."
    vault server -config=/opt/vault/config/vault.hcl &
    VAULT_PID=$!

    # Wait for Vault to be ready
    local attempts=0
    while ! curl -s http://127.0.0.1:8200/v1/sys/health > /dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [ $attempts -gt 30 ]; then
            print_error "Vault failed to start"
            exit 1
        fi
        sleep 1
    done

    # Initialize Vault (uses lib/vault.sh functions)
    vault_initialize

    # Store secrets in Vault
    print_step "Storing API credentials in Vault..."
    # model_configs seeds 'fast' with api_key_name = subcortical_key and
    # 'other'/'primary' with provider_key. The container collects one generic
    # fast-inference key, so seed it under both names or the fast route
    # resolves to a missing Vault field at runtime.
    vault kv put secret/mira/api_keys \
        anthropic_key="$CONFIG_ANTHROPIC_KEY" \
        anthropic_batch_key="$CONFIG_ANTHROPIC_BATCH_KEY" \
        provider_key="$CONFIG_PROVIDER_KEY" \
        subcortical_key="$CONFIG_PROVIDER_KEY" \
        kagi_api_key="$CONFIG_KAGI_KEY"

    vault kv put secret/mira/database \
        admin_url="postgresql://mira_admin:${CONFIG_DB_PASSWORD}@localhost:5432/mira_service" \
        service_url="postgresql://mira_dbuser:${CONFIG_DB_PASSWORD}@localhost:5432/mira_service" \
        username="mira_dbuser" \
        password="$CONFIG_DB_PASSWORD"

    CONFIG_USERDATA_ENCRYPTION_KEY=$(openssl rand -base64 32)
    CONFIG_DIAGNOSTICS_TOKEN=$(openssl rand -base64 32)

    # Optional SMTP relay for multi-user mode: MIRA_SMTP_* set on the
    # container at init time is persisted to Vault, which the mail sender
    # reads after the environment (auth/email_service.py). Unset everywhere
    # is fine outside MIRA_AUTH_MODE=multi.
    SMTP_ARGS=""
    if [ -n "${MIRA_SMTP_HOST:-}" ]; then
        SMTP_ARGS="smtp_host=\"${MIRA_SMTP_HOST}\""
        [ -n "${MIRA_SMTP_PORT:-}" ] && SMTP_ARGS="$SMTP_ARGS smtp_port=\"${MIRA_SMTP_PORT}\""
        [ -n "${MIRA_SMTP_FROM:-}" ] && SMTP_ARGS="$SMTP_ARGS smtp_from=\"${MIRA_SMTP_FROM}\""
        [ -n "${MIRA_SMTP_USER:-}" ] && SMTP_ARGS="$SMTP_ARGS smtp_user=\"${MIRA_SMTP_USER}\""
        [ -n "${MIRA_SMTP_PASSWORD:-}" ] && SMTP_ARGS="$SMTP_ARGS smtp_password=\"${MIRA_SMTP_PASSWORD}\""
        [ -n "${MIRA_SMTP_STARTTLS:-}" ] && SMTP_ARGS="$SMTP_ARGS smtp_starttls=\"${MIRA_SMTP_STARTTLS}\""
    fi

    eval vault kv put secret/mira/services \
        app_url=\"http://localhost:1993\" \
        valkey_url=\"valkey://localhost:6379\" \
        userdata_encryption_key=\"\$CONFIG_USERDATA_ENCRYPTION_KEY\" \
        diagnostics_token=\"\$CONFIG_DIAGNOSTICS_TOKEN\" \
        \$SMTP_ARGS

    # Stop Vault (s6 will manage it from here)
    print_step "Stopping temporary Vault instance..."
    kill $VAULT_PID 2>/dev/null || true
    wait $VAULT_PID 2>/dev/null || true

    print_success "Vault initialized and secrets stored"
}

# =============================================================================
# Main Entrypoint Logic
# =============================================================================

main() {
    echo ""
    echo -e "${BOLD}${CYAN}"
    echo "╔════════════════════════════════════════╗"
    echo "║   MIRA Docker Container                ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${RESET}"

    if is_first_boot; then
        print_header "First Boot Detected - Running Setup"

        if has_tty; then
            # Interactive mode - run the full setup wizard
            print_info "Interactive mode detected. Running setup wizard..."
            echo ""
            source /opt/mira/container-setup.sh
        else
            # Non-interactive mode - use environment variables
            print_info "Non-interactive mode. Checking environment variables..."
            check_env_vars
            setup_from_env_vars
        fi

        # Initialize services
        init_postgresql
        init_vault

        print_success "First-boot setup complete!"
        echo ""
    else
        print_info "Existing configuration found. Starting services..."
    fi

    # Hand off to s6-overlay
    print_header "Starting Services via s6-overlay"
    exec /init
}

main "$@"
