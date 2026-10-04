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

    # Homebrew postgresql@17 is keg-only: psql/createdb are NOT symlinked
    # into /opt/homebrew/bin. One resolver, prepended once — every bare
    # psql/createdb/pg_isready call in this file and in later phases of this
    # same shell then resolves against the keg.
    PG_BINDIR="$(brew --prefix postgresql@17)/bin"
    if [ ! -x "$PG_BINDIR/psql" ]; then
        print_error "PostgreSQL 17 client not found at $PG_BINDIR"
        print_info "Expected brew postgresql@17 (installed by dependencies.sh Step 1)."
        exit 1
    fi
    PATH="$PG_BINDIR:$PATH"
    export PATH

    start_service valkey brew
    start_service postgresql@17 brew

    # brew services clears the sudo timestamp as well (ensure_sudo docs) —
    # re-establish elevation for any later sudo step.
    ensure_sudo

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
# Roles are created with the default sentinel password because the Vault URLs
# stored in Step 14 are postgresql://mira_dbuser:<password>@localhost:5432/… —
# libpq always presents a password. On Linux, pg_hba uses scram-sha-256 for
# TCP connections, so the password is verified; a role without one could
# never authenticate. Homebrew's macOS pg_hba defaults to trust for
# localhost, so the password is carried but not challenged there. The
# custom-password block further down replaces the sentinel when
# CONFIG_DB_PASSWORD differs.
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

# Greenfield contract: the schema REFUSES a non-empty database, so a prior MIRA
# install leaves mira_service populated and the apply fails. The operator's data
# is preserved, never dropped: the old database is renamed aside automatically
# and a fresh empty mira_service is created, and finalize.sh reports the kept
# database with a pointer to the migration guide. The rename is non-destructive,
# so it needs no consent and runs unattended too. One stamp covers everything
# this install preserves — the renamed database and every Vault entry copied
# aside carry it — so the end-of-install note names one moment in time.
MIRA_LEGACY_STAMP="$(date +%Y%m%d_%H%M%S)"
MIRA_PREVIOUS_DB=""
MIRA_VAULT_BACKUPS=()
echo -ne "${DIM}${ARROW}${RESET} Verifying mira_service is empty... "
if [ "$OS" = "linux" ]; then
    DB_TABLES="$(sudo -u postgres psql -d mira_service -tAc "SELECT count(*) FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'" 2>/dev/null || true)"
else
    DB_TABLES="$(psql -d mira_service -tAc "SELECT count(*) FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'" 2>/dev/null || true)"
fi
DB_TABLES="${DB_TABLES//[[:space:]]/}"
if [ "$DB_TABLES" = "0" ]; then
    echo -e "${CHECKMARK}"
else
    echo -e "${WARNING}"
    print_warning "mira_service already holds ${DB_TABLES:-an unknown number of} table(s)."
    print_info "MIRA 2.0 installs into a fresh database. The existing one is renamed"
    print_info "aside automatically — never deleted — so its data stays available for"
    print_info "migration (see the note at the end of this install)."
    MIRA_PREVIOUS_DB="mira_service_old_${MIRA_LEGACY_STAMP}"
    DETACH_SQL="SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'mira_service' AND pid <> pg_backend_pid();"
    RENAME_SQL="ALTER DATABASE mira_service RENAME TO \"$MIRA_PREVIOUS_DB\";"
    if [ "$OS" = "linux" ]; then
        sudo -u postgres psql -d postgres -v ON_ERROR_STOP=1 -c "$DETACH_SQL" > /dev/null 2>&1 || true
        if ! sudo -u postgres psql -d postgres -v ON_ERROR_STOP=1 -c "$RENAME_SQL" > /dev/null 2>&1; then
            echo -e "${ERROR}"; print_error "Could not rename mira_service"; exit 1
        fi
        sudo -u postgres createdb -O mira_admin mira_service || exit 1
    else
        psql -d postgres -v ON_ERROR_STOP=1 -c "$DETACH_SQL" > /dev/null 2>&1 || true
        if ! psql -d postgres -v ON_ERROR_STOP=1 -c "$RENAME_SQL" > /dev/null 2>&1; then
            echo -e "${ERROR}"; print_error "Could not rename mira_service"; exit 1
        fi
        createdb -O mira_admin mira_service || exit 1
    fi
    print_success "Kept the old database as ${MIRA_PREVIOUS_DB}; created a fresh mira_service"
fi

# Run the fresh-install schema as the database superuser so extension and
# least-privilege grant setup can complete. The schema is a pure DDL contract:
# it creates no roles, no database, and refuses to run against a non-empty
# database, so it must target mira_service directly.
# The schema sizes every vector column from the embedding model, so resolve
# it first: the app reports the local model's size, or probes the configured
# remote endpoint for its vector length (deploy/lib/embedding_config.sh).
echo -ne "${DIM}${ARROW}${RESET} Resolving embedding model and vector length... "
if resolve_embedding_schema_args /opt/mira/app/venv/bin/python3 /opt/mira/app \
        "$CONFIG_EMBEDDING_PROVIDER" "$CONFIG_EMBEDDING_ENDPOINT" "$CONFIG_EMBEDDING_MODEL" "$CONFIG_EMBEDDING_API_KEY"; then
    echo -e "${CHECKMARK} ${DIM}${EMBEDDING_MODEL}, ${EMBEDDING_DIMENSIONS} dimensions${RESET}"
else
    echo -e "${ERROR}"
    print_error "Could not resolve the embedding model (reason above). A remote endpoint must answer POST /v1/embeddings for the configured model and key."
    exit 1
fi

# The injection screen's System One model must answer before the install
# commits: an enabled-but-broken screen parks MIRA's boot gate on every
# start. Local (self-hosted Kev) and remote gateways go through the same
# probe — deploy/lib/systemone_config.sh.
if [ "$CONFIG_INJECTION_SCREEN" = "yes" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Probing System One injection-screen model... "
    if probe_systemone /opt/mira/app/venv/bin/python3 /opt/mira/app \
            "$CONFIG_SYSTEMONE_PROVIDER" "$CONFIG_SYSTEMONE_ENDPOINT" "$CONFIG_SYSTEMONE_MODEL" "$CONFIG_SYSTEMONE_API_KEY"; then
        echo -e "${CHECKMARK} ${DIM}${CONFIG_SYSTEMONE_MODEL}${RESET}"
    else
        echo -e "${ERROR}"
        print_error "Could not reach the System One endpoint (reason above)."
        print_info "Point injection_screen at a reachable System One model, or disable it (injection_screen: no / MIRA_INJECTION_SCREEN_ENABLED=0)."
        exit 1
    fi
fi

echo -ne "${DIM}${ARROW}${RESET} Running fresh database schema (tables, indexes, RLS)... "
SCHEMA_FILE="/opt/mira/app/deploy/mira_service_schema.sql"
if [ ! -f "$SCHEMA_FILE" ]; then
    echo -e "${ERROR}"
    print_error "Schema file not found: $SCHEMA_FILE"
    exit 1
fi
# The apply emits hundreds of GRANT/COMMENT lines and is normally quiet, but a
# failure must never be swallowed: the schema's own refusal message (for example
# "requires an empty target database") is the entire diagnosis. Capture stdout
# and stderr together and print the tail when it fails.
if [ "$OS" = "linux" ]; then
    SCHEMA_CMD=(sudo -u postgres psql -d mira_service -v ON_ERROR_STOP=1 "${EMBEDDING_SCHEMA_ARGS[@]}" -f "$SCHEMA_FILE")
else
    SCHEMA_CMD=(psql -d mira_service -v ON_ERROR_STOP=1 "${EMBEDDING_SCHEMA_ARGS[@]}" -f "$SCHEMA_FILE")
fi
if SCHEMA_OUT="$("${SCHEMA_CMD[@]}" 2>&1)"; then
    echo -e "${CHECKMARK}"
else
    echo -e "${ERROR}"
    print_error "Failed to run schema file"
    printf '%s\n' "$SCHEMA_OUT" | tail -25 | sed 's/^/    /'
    print_info "The message above is psql's own reason; fix it and re-run."
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
        # NOTE: like the hosted chat-tier branch below, the failure is handled
        # inside the if/else so set -e cannot abort before this report.
        # The "run manually" guidance stays for the genuinely air-gapped
        # install whose UPDATE failed — it keys on the UPDATE's exit status,
        # not on offline_mode, and a successful rewrite never sees it.
        if sudo -u postgres psql -d mira_service -v ON_ERROR_STOP=1 -c "$OFFLINE_SQL" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to configure offline mode - you may need to run the UPDATE manually; aborting so the install cannot complete with lunaroute routes and no stored credential"
            exit 1
        fi
    elif [ "$OS" = "macos" ]; then
        if psql -d mira_service -v ON_ERROR_STOP=1 -c "$OFFLINE_SQL" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to configure offline mode - you may need to run the UPDATE manually; aborting so the install cannot complete with lunaroute routes and no stored credential"
            exit 1
        fi
    fi
fi

# Hosted install (offline_mode: no): rewrite the seeded model_configs routes to
# match the config's providers. The seed rows in mira_service_schema.sql ARE the
# lunaroute defaults, so a default config leaves them effectively untouched;
# any departure is applied here with UPDATEs against the live database — the
# same mechanism OFFLINE_SQL uses, never a string-patch of the schema file.
if [ "$CONFIG_OFFLINE_MODE" != "yes" ]; then
    # --- Chat tier: the 'primary' route ---
    if [ "$CONFIG_CHAT_PROVIDER_TYPE" = "openai" ]; then
        if [ -z "$CONFIG_CHAT_MODEL" ] || [ -z "$CONFIG_CHAT_ENDPOINT" ]; then
            echo -e "${ERROR}"
            print_error "OpenAI-compatible chat mode requires both chat_model and chat_endpoint, but got model='${CONFIG_CHAT_MODEL}' endpoint='${CONFIG_CHAT_ENDPOINT}'"
            print_info "Set chat_model and chat_endpoint in deploy-config.yml and re-run."
            exit 1
        fi
        CHAT_SQL="UPDATE model_configs SET dialect_name = 'openai', endpoint_url = '$CONFIG_CHAT_ENDPOINT', model = '$CONFIG_CHAT_MODEL', api_key_name = 'provider_key' WHERE name = 'primary';"
    else
        # Anthropic chat mode. The seed 'primary' row is an openai/lunaroute row
        # whose 'provider_key' Vault name Step 14 never stores in Anthropic mode,
        # so it MUST be rewritten to the anthropic dialect/endpoint/key here —
        # an empty chat_model would leave primary pointing at a route with no
        # stored credential (a broken first boot). Refuse it.
        if [ -z "$CONFIG_CHAT_MODEL" ]; then
            echo -e "${ERROR}"
            print_error "Anthropic chat mode requires a chat_model, but got '${CONFIG_CHAT_MODEL}'"
            print_info "Set chat_model to an Anthropic model (e.g. claude-opus-4-6) in deploy-config.yml and re-run."
            exit 1
        fi
        CHAT_SQL="UPDATE model_configs SET dialect_name = 'anthropic', endpoint_url = 'https://api.anthropic.com/v1/messages', model = '$CONFIG_CHAT_MODEL', api_key_name = 'anthropic_key' WHERE name = 'primary';"
    fi
    echo -ne "${DIM}${ARROW}${RESET} Configuring chat tier (${CONFIG_CHAT_MODEL})... "
    # NOTE: the psql calls are captured as `VAR="$(cmd || echo failed)"` so a
    # failure reaches the error branch below instead of aborting under
    # deploy.sh's set -e before it can report.
    if [ "$OS" = "linux" ]; then
        CHAT_OK="$(sudo -u postgres psql -d mira_service -v ON_ERROR_STOP=1 -c "$CHAT_SQL" > /dev/null 2>&1 || echo failed)"
    else
        CHAT_OK="$(psql -d mira_service -v ON_ERROR_STOP=1 -c "$CHAT_SQL" > /dev/null 2>&1 || echo failed)"
    fi
    if [ -n "$CHAT_OK" ]; then
        echo -e "${ERROR}"
        print_error "Failed to configure the chat tier (primary route); refusing to install a route with no valid provider"
        exit 1
    fi
    echo -e "${CHECKMARK}"

    # --- Subcortical tier: the four auxiliary routes ---
    # Rewrite only when the config departs from the lunaroute defaults, so the
    # default install keeps batch's glm-5.3-flash-background variant (a lunaroute
    # background serving the single subcortical_model key cannot express).
    DEF_SUB_ENDPOINT="https://gw.lunaroute.com/v1/chat/completions"
    DEF_SUB_MODEL="glm-5.3-flash"
    SUB_SQL=""
    if [ -n "$CONFIG_SUBCORTICAL_ENDPOINT" ] && [ "$CONFIG_SUBCORTICAL_ENDPOINT" != "$DEF_SUB_ENDPOINT" ]; then
        SUB_SQL="${SUB_SQL}UPDATE model_configs SET endpoint_url = '$CONFIG_SUBCORTICAL_ENDPOINT' WHERE name IN ('fast', 'batch', 'assessment', 'other');"
    fi
    if [ -n "$CONFIG_SUBCORTICAL_MODEL" ] && [ "$CONFIG_SUBCORTICAL_MODEL" != "$DEF_SUB_MODEL" ]; then
        SUB_SQL="${SUB_SQL}UPDATE model_configs SET model = '$CONFIG_SUBCORTICAL_MODEL' WHERE name IN ('fast', 'batch', 'assessment', 'other');"
    fi
    if [ -n "$SUB_SQL" ]; then
        echo -ne "${DIM}${ARROW}${RESET} Configuring subcortical tier (${CONFIG_SUBCORTICAL_MODEL:-glm-5.3-flash})... "
        if [ "$OS" = "linux" ]; then
            SUB_OK="$(sudo -u postgres psql -d mira_service -v ON_ERROR_STOP=1 -c "$SUB_SQL" > /dev/null 2>&1 || echo failed)"
        else
            SUB_OK="$(psql -d mira_service -v ON_ERROR_STOP=1 -c "$SUB_SQL" > /dev/null 2>&1 || echo failed)"
        fi
        if [ -n "$SUB_OK" ]; then
            echo -e "${ERROR}"
            print_error "Failed to configure the subcortical tier (fast/batch/assessment/other routes)"
            exit 1
        fi
        echo -e "${CHECKMARK}"
    fi
fi

# Update PostgreSQL passwords if custom password was set
# NOTE: the ALTER pairs run as `if` conditions so a failure reaches the
# explicit error report below (and its exit 1) instead of aborting silently
# under set -e before the operator sees why the install stopped.
# Percent-encode reserved characters so the embedded password forms a
# valid URL credential. The reader (PostgresClient._parse_database_url) hands
# the URL to libpq, which percent-decodes userinfo — both deploy writers
# (this file and deploy/docker/scripts/init-mira.sh) must encode by the same
# rule, or the app authenticates with the literal %XX sequence as password.
# The roles themselves keep the raw password (ALTER USER below, and the
# password="..." Vault fields).
DB_PASSWORD_URL_ENC=""
_pw="${CONFIG_DB_PASSWORD}"
for ((_i = 0; _i < ${#_pw}; _i++)); do
    _c=${_pw:_i:1}
    case "$_c" in
        [A-Za-z0-9._~-]) DB_PASSWORD_URL_ENC+="$_c" ;;
        *) printf -v _c '%%%02X' "'$_c"; DB_PASSWORD_URL_ENC+="$_c" ;;
    esac
done
if [ "$CONFIG_DB_PASSWORD" != "changethisifdeployingpwd" ]; then
    # On a re-run with a changed password, preserve the existing
    # Vault entry under a suffixed legacy key and write an access guide
    # BEFORE either store is updated. The old credential is never
    # overwritten in place - it is the operator's only fallback if the
    # database update fails or the new value is wrong.
    # Fail fast on an unreachable Vault: `vault kv get ... || true` below
    # cannot distinguish "no entry yet" from "Vault down", and a silent
    # no-op here would let the ALTER run without the legacy preservation.
    if ! vault status > /dev/null 2>&1; then
        print_error "Vault unreachable - cannot verify existing database credential for preservation"
        exit 1
    fi
    DB_EXISTING_PASSWORD=$(vault kv get -field=password secret/mira/database 2>/dev/null || true)
    if [ -n "$DB_EXISTING_PASSWORD" ] && [ "$DB_EXISTING_PASSWORD" != "$CONFIG_DB_PASSWORD" ]; then
        DB_LEGACY_KEY="secret/mira/database_legacy_$(date +%Y%m%d%H%M%S)"
        OLD_ADMIN_URL=$(vault kv get -field=admin_url secret/mira/database 2>/dev/null || true)
        OLD_USERNAME=$(vault kv get -field=username secret/mira/database 2>/dev/null || true)
        OLD_SERVICE_URL=$(vault kv get -field=service_url secret/mira/database 2>/dev/null || true)
        if vault kv put "$DB_LEGACY_KEY" \
            admin_url="${OLD_ADMIN_URL}" \
            password="${DB_EXISTING_PASSWORD}" \
            username="${OLD_USERNAME}" \
            service_url="${OLD_SERVICE_URL}" > /dev/null 2>&1; then
            print_info "Previous database credential preserved at ${DB_LEGACY_KEY}"
        else
            print_error "Failed to preserve the previous database credential in Vault - aborting before any credential update"
            exit 1
        fi
        # Update the primary Vault key only after the old entry is recorded,
        # so Vault and PostgreSQL are updated from the same new value.
        if vault kv put secret/mira/database \
            admin_url="postgresql://mira_admin:${DB_PASSWORD_URL_ENC}@localhost:5432/mira_service" \
            password="${CONFIG_DB_PASSWORD}" \
            username="mira_dbuser" \
            service_url="postgresql://mira_dbuser:${DB_PASSWORD_URL_ENC}@localhost:5432/mira_service" > /dev/null 2>&1; then
            print_info "Vault database credential updated to the new password"
        else
            print_error "Failed to update the Vault database credential - aborting before database update (old entry preserved at ${DB_LEGACY_KEY})"
            exit 1
        fi
        cat > /opt/vault/howtoaccess.txt <<EOF
MIRA Vault database credentials - written $(date)

Vault entries:
  1. secret/mira/database
     The NEW password set by this deploy re-run. This is the entry the
     running MIRA instance uses, and it matches the PostgreSQL roles
     mira_admin and mira_dbuser.
     Read it:  vault kv get -field=password secret/mira/database

  2. ${DB_LEGACY_KEY}
     The OLD password from the previous install, preserved unchanged.
     It is only useful against a database that still runs the old
     password (e.g. if this re-run's database update failed).
     Read it:  vault kv get -field=password ${DB_LEGACY_KEY}

The running instance reads secret/mira/database.
EOF
        chmod 600 /opt/vault/howtoaccess.txt
        print_info "Credential access guide written to /opt/vault/howtoaccess.txt"
    fi
    echo -ne "${DIM}${ARROW}${RESET} Updating database passwords... "
    # Abort on ALTER failure (sm9d): Step 14 below writes the custom
    # password into secret/mira/database, so continuing after a failed ALTER
    # would seed Vault with a credential the roles never received. The
    # `if`-shape is kept so the failure reaches this report under set -e.
    if [ "$OS" = "linux" ]; then
        if sudo -u postgres psql -c "ALTER USER mira_admin WITH PASSWORD '${CONFIG_DB_PASSWORD}';" > /dev/null 2>&1 && \
           sudo -u postgres psql -c "ALTER USER mira_dbuser WITH PASSWORD '${CONFIG_DB_PASSWORD}';" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to update database passwords - aborting before Vault stores the new credential (roles still hold the previous password)"
            exit 1
        fi
    elif [ "$OS" = "macos" ]; then
        if psql postgres -c "ALTER USER mira_admin WITH PASSWORD '${CONFIG_DB_PASSWORD}';" > /dev/null 2>&1 && \
           psql postgres -c "ALTER USER mira_dbuser WITH PASSWORD '${CONFIG_DB_PASSWORD}';" > /dev/null 2>&1; then
            echo -e "${CHECKMARK}"
        else
            echo -e "${ERROR}"
            print_error "Failed to update database passwords - aborting before Vault stores the new credential (roles still hold the previous password)"
            exit 1
        fi
    fi
fi

print_success "PostgreSQL configured"

print_header "Step 14: Vault Credential Storage"

# Build api_keys arguments based on chat provider type.
# (tsvt) The arguments are assembled as a bash array and passed directly:
# an eval re-parse would re-expand $ / quotes / backticks inside secret
# values and seed a mangled credential under a "Configured" status.
API_KEYS_ARGS=(secret/mira/api_keys)
if [ "$CONFIG_CHAT_PROVIDER_TYPE" = "openai" ]; then
    # OpenAI-compatible chat: provider_key = chat key, subcortical_key = subcortical key
    API_KEYS_ARGS+=(anthropic_key="${CONFIG_ANTHROPIC_KEY}" provider_key="${CONFIG_CHAT_API_KEY}" subcortical_key="${CONFIG_SUBCORTICAL_API_KEY}")
else
    # Anthropic chat: no provider_key needed
    API_KEYS_ARGS+=(anthropic_key="${CONFIG_ANTHROPIC_KEY}" subcortical_key="${CONFIG_SUBCORTICAL_API_KEY}")
fi
if [ -n "$CONFIG_KAGI_KEY" ]; then
    API_KEYS_ARGS+=(kagi_api_key="${CONFIG_KAGI_KEY}")
fi
if [ "$CONFIG_EMBEDDING_PROVIDER" = "remote" ] && [ -n "$CONFIG_EMBEDDING_API_KEY" ]; then
    API_KEYS_ARGS+=("${EMBEDDING_VAULT_KEY_NAME}=${CONFIG_EMBEDDING_API_KEY}")
fi
# Injection screen (M16): a remote System One gateway keys in Vault. On the
# lunaroute gateway that key is the reused chat key; any other remote endpoint
# carries its own explicit key. Local (unkeyed) and disabled installs seed
# nothing.
if [ "$CONFIG_INJECTION_SCREEN" = "yes" ] && [ "$CONFIG_SYSTEMONE_PROVIDER" = "remote" ]; then
    API_KEYS_ARGS+=("${SYSTEMONE_VAULT_KEY_NAME}=${CONFIG_SYSTEMONE_API_KEY}")
fi
vault_put_with_backup "${API_KEYS_ARGS[@]}"

# Preserved, never overwritten — see vault_put_if_not_exists: on a re-run that
# leaves the password at the sentinel default, the ALTER USER block above is
# skipped, so the roles still hold the previous custom password and this entry
# must stay in step with them. A changed password was already written above,
# together with its own legacy copy.
vault_put_if_not_exists secret/mira/database \
    admin_url="postgresql://mira_admin:${DB_PASSWORD_URL_ENC}@localhost:5432/mira_service" \
    password="${CONFIG_DB_PASSWORD}" \
    username="mira_dbuser" \
    service_url="postgresql://mira_dbuser:${DB_PASSWORD_URL_ENC}@localhost:5432/mira_service"

# userdata_encryption_key is the Fernet key every encrypted column in the
# per-user SQLite store is sealed with (utils/userdata_manager.py). It is
# carried forward unchanged on every re-install and generated only when no value
# exists: rotating it leaves the stored ciphertext unreadable. Vault must be
# reachable to read it — regenerating because Vault was down would brick
# encrypted user data, so an unreachable Vault aborts here instead.
if ! vault status > /dev/null 2>&1; then
    print_error "Vault unreachable - cannot read the existing userdata encryption key"
    exit 1
fi
CONFIG_USERDATA_ENCRYPTION_KEY="$(vault kv get -field=userdata_encryption_key secret/mira/services 2>/dev/null || true)"
CONFIG_DIAGNOSTICS_TOKEN="$(vault kv get -field=diagnostics_token secret/mira/services 2>/dev/null || true)"
if [ -z "$CONFIG_USERDATA_ENCRYPTION_KEY" ]; then
    CONFIG_USERDATA_ENCRYPTION_KEY="$(openssl rand -base64 32)"
fi
if [ -z "$CONFIG_DIAGNOSTICS_TOKEN" ]; then
    CONFIG_DIAGNOSTICS_TOKEN="$(openssl rand -base64 32)"
fi

# Optional SMTP relay for multi-user mode (MIRA_AUTH_MODE=multi). Vault is
# the only runtime SMTP source: the mail sender reads these Vault fields
# only and never reads MIRA_SMTP_* at runtime (auth/email_service.py).
# Exporting MIRA_SMTP_* while running the deployer persists a relay for
# the systemd service, which receives only VAULT_* Environment lines — and the
# write below applies them to secret/mira/services on every install, preserving
# the previous values first. Nothing here is required in the default single-user
# mode: no send happens, and no boot check demands it.
# (tsvt) Same array discipline as the api_keys write above: SMTP values are
# passed as single argv entries so $ / quotes / backticks in a relay password
# survive byte-identical and never see a second shell parse.
SMTP_ARGS=()
if [ -n "${MIRA_SMTP_HOST:-}" ]; then
    SMTP_ARGS+=(smtp_host="${MIRA_SMTP_HOST}")
    [ -n "${MIRA_SMTP_PORT:-}" ] && SMTP_ARGS+=(smtp_port="${MIRA_SMTP_PORT}")
    [ -n "${MIRA_SMTP_FROM:-}" ] && SMTP_ARGS+=(smtp_from="${MIRA_SMTP_FROM}")
    [ -n "${MIRA_SMTP_USER:-}" ] && SMTP_ARGS+=(smtp_user="${MIRA_SMTP_USER}")
    [ -n "${MIRA_SMTP_PASSWORD:-}" ] && SMTP_ARGS+=(smtp_password="${MIRA_SMTP_PASSWORD}")
    [ -n "${MIRA_SMTP_STARTTLS:-}" ] && SMTP_ARGS+=(smtp_starttls="${MIRA_SMTP_STARTTLS}")
fi

vault_put_with_backup secret/mira/services \
    app_url="http://localhost:1993" \
    valkey_url="valkey://localhost:6379" \
    userdata_encryption_key="${CONFIG_USERDATA_ENCRYPTION_KEY}" \
    diagnostics_token="${CONFIG_DIAGNOSTICS_TOKEN}" \
    "${SMTP_ARGS[@]}"

print_success "All credentials configured in Vault"
