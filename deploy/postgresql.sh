# deploy/postgresql.sh
# PostgreSQL service startup, schema deployment, and Vault credential storage
# Source this file - do not execute directly
#
# Requires: lib/output.sh, lib/services.sh, lib/vault.sh sourced first
# Requires: OS, DISTRO, CONFIG_*, LOUD_MODE variables set

# Validate required variables
: "${OS:?Error: OS must be set}"
: "${CONFIG_DB_PASSWORD:?Error: CONFIG_DB_PASSWORD must be set}"
: "${CONFIG_ANTHROPIC_KEY:?Error: CONFIG_ANTHROPIC_KEY must be set}"

if [ "$OS" = "macos" ]; then
    print_header "Step 12: Starting Services"

    start_service valkey brew
    start_service postgresql@17 brew

    sleep 2
fi

# Wait for PostgreSQL to be ready to accept connections. On Linux, start the
# service first if it is not running: a re-run deploy may have stopped it via
# config.sh's occupied-port handling, and the package auto-start only happens
# once, at apt/dnf install time. Fedora PGDG names the unit
# postgresql-17.service; Debian names it postgresql.service.
if [ "$OS" = "linux" ]; then
    if ! sudo -u postgres pg_isready > /dev/null 2>&1 && \
       ! sudo -u postgres /usr/pgsql-17/bin/pg_isready > /dev/null 2>&1; then
        if [ "$DISTRO" = "fedora" ]; then
            run_quiet sudo systemctl start postgresql-17.service
        else
            run_quiet sudo systemctl start postgresql.service
        fi
    fi
fi
echo -ne "${DIM}${ARROW}${RESET} Waiting for PostgreSQL to be ready... "
PG_READY=0
for i in {1..30}; do
    if [ "$OS" = "linux" ]; then
        # On Linux, check with pg_isready (Fedora PGDG uses /usr/pgsql-17/bin/)
        if sudo -u postgres pg_isready > /dev/null 2>&1 || \
           sudo -u postgres /usr/pgsql-17/bin/pg_isready > /dev/null 2>&1; then
            PG_READY=1
            break
        fi
    elif [ "$OS" = "macos" ]; then
        # On macOS, check with pg_isready as current user
        # Homebrew PostgreSQL 17 uses versioned command name
        if pg_isready-17 > /dev/null 2>&1; then
            PG_READY=1
            break
        fi
    fi
    sleep 1
done

if [ $PG_READY -eq 0 ]; then
    echo -e "${ERROR}"
    print_error "PostgreSQL did not become ready within 30 seconds"
    if [ "$OS" = "linux" ]; then
        if [ "$DISTRO" = "fedora" ]; then
            print_info "Check status: systemctl status postgresql-17"
            print_info "Check logs: journalctl -u postgresql-17 -n 50"
        else
            print_info "Check status: systemctl status postgresql"
            print_info "Check logs: journalctl -u postgresql -n 50"
        fi
    elif [ "$OS" = "macos" ]; then
        print_info "Check status: brew services list | grep postgresql"
        print_info "Check logs: brew services info postgresql@17"
    fi
    exit 1
fi
echo -e "${CHECKMARK} ${DIM}(ready after ${i}s)${RESET}"

print_header "Step 13: PostgreSQL Configuration"

# Roles, database ownership, and credentials are deployment concerns. The SQL
# schema deliberately assumes these contracts already exist and targets an
# empty database, so provision them before applying it.
echo -ne "${DIM}${ARROW}${RESET} Provisioning database roles and database... "
# Roles are created with the default sentinel password because pg_hba uses
# scram-sha-256 for TCP connections on every supported platform: a role with
# no password can never authenticate the postgresql://mira_dbuser:...
# (and admin) URLs that Step 14 stores in Vault. The custom-password block
# further down replaces the sentinel when CONFIG_DB_PASSWORD differs.
ROLE_SQL="DO \$roles\$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mira_admin') THEN CREATE ROLE mira_admin LOGIN PASSWORD 'changethisifdeployingpwd' BYPASSRLS; END IF; IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mira_dbuser') THEN CREATE ROLE mira_dbuser LOGIN PASSWORD 'changethisifdeployingpwd'; END IF; END \$roles\$;"
if [ "$OS" = "linux" ]; then
    if ! sudo -u postgres psql -d postgres -v ON_ERROR_STOP=1 -c "$ROLE_SQL" > /dev/null 2>&1; then
        echo -e "${ERROR}"
        print_error "Failed to provision database roles"
        exit 1
    fi
    if ! sudo -u postgres psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = 'mira_service'" | grep -q 1; then
        sudo -u postgres createdb -O mira_admin mira_service || exit 1
    fi
elif [ "$OS" = "macos" ]; then
    if ! psql -d postgres -v ON_ERROR_STOP=1 -c "$ROLE_SQL" > /dev/null 2>&1; then
        echo -e "${ERROR}"
        print_error "Failed to provision database roles"
        exit 1
    fi
    if ! psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = 'mira_service'" | grep -q 1; then
        createdb -O mira_admin mira_service || exit 1
    fi
fi
echo -e "${CHECKMARK}"

# Run the fresh-install schema as the database superuser so extension and
# least-privilege grant setup can complete. The schema is a pure DDL contract:
# it creates no roles, no database, and refuses to run against a non-empty
# database, so it must target mira_service directly.
echo -ne "${DIM}${ARROW}${RESET} Running fresh database schema (tables, indexes, RLS)... "
SCHEMA_FILE="/opt/mira/app/deploy/mira_service_schema.sql"
if [ -f "$SCHEMA_FILE" ]; then
    if [ "$OS" = "linux" ]; then
        if sudo -u postgres psql -d mira_service -v ON_ERROR_STOP=1 -f "$SCHEMA_FILE" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to run schema file"
            exit 1
        fi
    elif [ "$OS" = "macos" ]; then
        if psql -d mira_service -v ON_ERROR_STOP=1 -f "$SCHEMA_FILE" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to run schema file"
            exit 1
        fi
    fi
else
    echo -e "${ERROR}"
    print_error "Schema file not found: $SCHEMA_FILE"
    exit 1
fi

# Route every model_configs entry at local llama-server instances.
# An offline install has no cloud provider, so all five routes are rewritten to
# OpenAI-compatible local endpoints with api_key_name cleared to '' (the
# resolver reads an empty key name as "no credential required").
#
# The main instance takes the capability-sensitive routes; the small,
# speed-critical instance takes 'fast' (subcortical analysis) and 'other'.
# Routing 'other' to a different served model keeps the "outside voice is not
# the chat model" property true when no outside vendor is reachable; a fully
# air-gapped install has no genuinely external perspective to consult.
#
# Set MIRA_LLAMA_MAIN_MODEL / MIRA_LLAMA_SMALL_MODEL (or CONFIG_LLAMA_MAIN_MODEL
# and CONFIG_LLAMA_SMALL_MODEL from the installer) to the model names your
# llama-server instances actually serve; the fallbacks are placeholders.
if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Configuring LLM endpoints for offline mode (llama-server)... "
    LLAMA_MAIN_URL="${MIRA_LLAMA_MAIN_URL:-http://localhost:3090/v1/chat/completions}"
    LLAMA_SMALL_URL="${MIRA_LLAMA_SMALL_URL:-http://localhost:3092/v1/chat/completions}"
    LLAMA_MAIN_MODEL="${MIRA_LLAMA_MAIN_MODEL:-${CONFIG_LLAMA_MAIN_MODEL:-local-main}}"
    LLAMA_SMALL_MODEL="${MIRA_LLAMA_SMALL_MODEL:-${CONFIG_LLAMA_SMALL_MODEL:-local-small}}"
    OFFLINE_SQL="UPDATE model_configs SET dialect_name = 'openai', endpoint_url = '$LLAMA_MAIN_URL', model = '$LLAMA_MAIN_MODEL', api_key_name = '' WHERE name IN ('primary', 'batch', 'assessment'); UPDATE model_configs SET dialect_name = 'openai', endpoint_url = '$LLAMA_SMALL_URL', model = '$LLAMA_SMALL_MODEL', api_key_name = '' WHERE name IN ('fast', 'other');"
    if [ "$OS" = "linux" ]; then
        if sudo -u postgres psql -d mira_service -v ON_ERROR_STOP=1 -c "$OFFLINE_SQL" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_warning "Failed to configure offline mode - you may need to run manually"
        fi
    elif [ "$OS" = "macos" ]; then
        if psql -d mira_service -v ON_ERROR_STOP=1 -c "$OFFLINE_SQL" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_warning "Failed to configure offline mode - you may need to run manually"
        fi
    fi
fi

# Update PostgreSQL passwords if custom password was set
# NOTE: the ALTER pairs run as `if` conditions so a failure reaches the warning
# branch instead of aborting under set -e.
if [ "$CONFIG_DB_PASSWORD" != "changethisifdeployingpwd" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Updating database passwords... "
    if [ "$OS" = "linux" ]; then
        if sudo -u postgres psql -c "ALTER USER mira_admin WITH PASSWORD '${CONFIG_DB_PASSWORD}';" > /dev/null 2>&1 && \
           sudo -u postgres psql -c "ALTER USER mira_dbuser WITH PASSWORD '${CONFIG_DB_PASSWORD}';" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_warning "Failed to update passwords - you may need to update manually"
        fi
    elif [ "$OS" = "macos" ]; then
        if psql postgres -c "ALTER USER mira_admin WITH PASSWORD '${CONFIG_DB_PASSWORD}';" > /dev/null 2>&1 && \
           psql postgres -c "ALTER USER mira_dbuser WITH PASSWORD '${CONFIG_DB_PASSWORD}';" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_warning "Failed to update passwords - you may need to update manually"
        fi
    fi
fi

print_success "PostgreSQL configured"

print_header "Step 14: Vault Credential Storage"

# Build api_keys arguments based on chat provider type
if [ "$CONFIG_CHAT_PROVIDER_TYPE" = "generic" ]; then
    # Generic chat: provider_key = chat key, subcortical_key = subcortical key
    API_KEYS_ARGS="anthropic_key=\"${CONFIG_ANTHROPIC_KEY}\" anthropic_batch_key=\"${CONFIG_ANTHROPIC_BATCH_KEY}\" provider_key=\"${CONFIG_CHAT_API_KEY}\" subcortical_key=\"${CONFIG_SUBCORTICAL_API_KEY}\""
else
    # Anthropic chat: no provider_key needed
    API_KEYS_ARGS="anthropic_key=\"${CONFIG_ANTHROPIC_KEY}\" anthropic_batch_key=\"${CONFIG_ANTHROPIC_BATCH_KEY}\" subcortical_key=\"${CONFIG_SUBCORTICAL_API_KEY}\""
fi
if [ -n "$CONFIG_KAGI_KEY" ]; then
    API_KEYS_ARGS="$API_KEYS_ARGS kagi_api_key=\"${CONFIG_KAGI_KEY}\""
fi
eval vault_put_if_not_exists secret/mira/api_keys $API_KEYS_ARGS

vault_put_if_not_exists secret/mira/database \
    admin_url="postgresql://mira_admin:${CONFIG_DB_PASSWORD}@localhost:5432/mira_service" \
    password="${CONFIG_DB_PASSWORD}" \
    username="mira_dbuser" \
    service_url="postgresql://mira_dbuser:${CONFIG_DB_PASSWORD}@localhost:5432/mira_service"

CONFIG_USERDATA_ENCRYPTION_KEY=$(openssl rand -base64 32)
CONFIG_DIAGNOSTICS_TOKEN=$(openssl rand -base64 32)

# Optional SMTP relay for multi-user mode (MIRA_AUTH_MODE=multi). The mail
# sender reads MIRA_SMTP_* from the environment first and these Vault
# fields second (auth/email_service.py), so exporting MIRA_SMTP_* while
# running the deployer persists a relay for the systemd service, which only
# receives VAULT_* Environment lines. Nothing here is required in the
# default single-user mode: no send happens, and no boot check demands it.
SMTP_ARGS=""
if [ -n "${MIRA_SMTP_HOST:-}" ]; then
    SMTP_ARGS="smtp_host=\"${MIRA_SMTP_HOST}\""
    [ -n "${MIRA_SMTP_PORT:-}" ] && SMTP_ARGS="$SMTP_ARGS smtp_port=\"${MIRA_SMTP_PORT}\""
    [ -n "${MIRA_SMTP_FROM:-}" ] && SMTP_ARGS="$SMTP_ARGS smtp_from=\"${MIRA_SMTP_FROM}\""
    [ -n "${MIRA_SMTP_USER:-}" ] && SMTP_ARGS="$SMTP_ARGS smtp_user=\"${MIRA_SMTP_USER}\""
    [ -n "${MIRA_SMTP_PASSWORD:-}" ] && SMTP_ARGS="$SMTP_ARGS smtp_password=\"${MIRA_SMTP_PASSWORD}\""
    [ -n "${MIRA_SMTP_STARTTLS:-}" ] && SMTP_ARGS="$SMTP_ARGS smtp_starttls=\"${MIRA_SMTP_STARTTLS}\""
fi

eval vault_put_if_not_exists secret/mira/services \
    app_url=\"http://localhost:1993\" \
    valkey_url=\"valkey://localhost:6379\" \
    userdata_encryption_key=\"\${CONFIG_USERDATA_ENCRYPTION_KEY}\" \
    diagnostics_token=\"\${CONFIG_DIAGNOSTICS_TOKEN}\" \
    $SMTP_ARGS

if ! vault kv get -field=userdata_encryption_key secret/mira/services > /dev/null 2>&1; then
    vault kv patch secret/mira/services userdata_encryption_key="${CONFIG_USERDATA_ENCRYPTION_KEY}" > /dev/null
fi
if ! vault kv get -field=diagnostics_token secret/mira/services > /dev/null 2>&1; then
    vault kv patch secret/mira/services diagnostics_token="${CONFIG_DIAGNOSTICS_TOKEN}" > /dev/null
fi

print_success "All credentials configured in Vault"
