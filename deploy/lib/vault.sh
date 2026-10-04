# deploy/lib/vault.sh
# HashiCorp Vault helper functions
# Source this file - do not execute directly
#
# Requires: lib/output.sh and lib/services.sh sourced first
# Requires: LOUD_MODE variable set

# Vault helper: Check if Vault is initialized
vault_is_initialized() {
    check_exists file "/opt/vault/init-keys.txt"
}

# Vault helper: Check if Vault is sealed
vault_is_sealed() {
    # vault status returns:
    # - exit code 0 when unsealed
    # - exit code 2 when sealed
    # - exit code 1 on error
    vault status > /dev/null 2>&1
    local exit_code=$?

    if [ $exit_code -eq 2 ]; then
        return 0  # Sealed (true)
    elif [ $exit_code -eq 0 ]; then
        return 1  # Unsealed (false)
    else
        # Error - assume sealed to be safe
        return 0
    fi
}

# Vault helper: Extract credential from init-keys.txt
# Usage: vault_extract_credential "Unseal Key 1" or "Initial Root Token"
vault_extract_credential() {
    local cred_type="$1"

    # Debug output in loud mode (to stderr so it doesn't pollute command substitution)
    if [ "$LOUD_MODE" = true ]; then
        echo "" >&2
        echo "DEBUG: Contents of /opt/vault/init-keys.txt:" >&2
        cat /opt/vault/init-keys.txt >&2
        echo "" >&2
        echo "DEBUG: Attempting to extract: $cred_type" >&2
    fi

    grep "$cred_type" /opt/vault/init-keys.txt | awk '{print $NF}'
}

# Vault helper: Unseal vault if sealed
vault_unseal() {
    if ! vault_is_sealed; then
        return 0  # Already unsealed
    fi

    local unseal_key=$(vault_extract_credential "Unseal Key 1")
    if [ -z "$unseal_key" ]; then
        print_error "Cannot unseal: unseal key not found in init-keys.txt"
        return 1
    fi

    run_with_status "Unsealing Vault" \
        vault operator unseal "$unseal_key"
}

# Vault helper: Authenticate with root token
vault_authenticate() {
    if ! vault_is_initialized; then
        print_error "Cannot authenticate: Vault not initialized"
        return 1
    fi

    local root_token=$(vault_extract_credential "Initial Root Token")
    if [ -z "$root_token" ]; then
        print_error "Cannot authenticate: root token not found in init-keys.txt"
        return 1
    fi

    run_with_status "Authenticating with Vault" vault login "$root_token"
}

# Vault helper: Check if AppRole exists
vault_approle_exists() {
    vault read auth/approle/role/mira > /dev/null 2>&1
}

# Vault helper: Full initialization orchestration
vault_initialize() {
    if vault_is_initialized; then
        print_info "Vault already initialized - checking state"

        # Unseal if needed (checks sealed state first)
        vault_unseal || return 1

        # Authenticate with root token
        vault_authenticate || return 1

        # Ensure KV2 secrets engine is enabled
        if ! vault secrets list | grep -q "^secret/"; then
            run_with_status "Enabling KV2 secrets engine" \
                vault secrets enable -version=2 -path=secret kv
        fi

        # Ensure AppRole exists
        if ! vault_approle_exists; then
            print_info "AppRole not found - creating it"

            # Enable AppRole if not enabled
            vault auth enable approle 2>/dev/null || true

            # Create policy if needed
            if ! vault policy read mira-policy > /dev/null 2>&1; then
                cat > /tmp/mira-policy.hcl <<'EOF'
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
path "secret/metadata/*" {
  capabilities = ["list", "read", "delete"]
}
EOF
                run_with_status "Writing policy to Vault" \
                    vault policy write mira-policy /tmp/mira-policy.hcl
            fi

            run_with_status "Creating AppRole" \
                vault write auth/approle/role/mira policies="mira-policy" token_ttl=1h token_max_ttl=4h
        fi

        # Ensure role-id and secret-id files exist
        if [ ! -f /opt/vault/role-id.txt ]; then
            vault read -field=role_id auth/approle/role/mira/role-id > /opt/vault/role-id.txt
        fi
        if [ ! -f /opt/vault/secret-id.txt ]; then
            vault write -field=secret_id -f auth/approle/role/mira/secret-id > /opt/vault/secret-id.txt
        fi

        return 0
    fi

    # Full initialization for new Vault
    echo -ne "${DIM}${ARROW}${RESET} Initializing Vault... "
    if vault operator init -key-shares=1 -key-threshold=1 > /opt/vault/init-keys.txt 2>&1; then
        echo -e "${CHECKMARK}"
        chmod 600 /opt/vault/init-keys.txt
    else
        echo -e "${ERROR}"
        print_error "Failed to initialize Vault"
        return 1
    fi

    vault_unseal || return 1
    vault_authenticate || return 1

    # Enable KV2 secrets engine
    run_with_status "Enabling KV2 secrets engine" \
        vault secrets enable -version=2 -path=secret kv

    # Enable AppRole authentication
    run_with_status "Enabling AppRole authentication" \
        vault auth enable approle

    # Create policy
    cat > /tmp/mira-policy.hcl <<'EOF'
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
path "secret/metadata/*" {
  capabilities = ["list", "read", "delete"]
}
EOF

    run_with_status "Writing policy to Vault" \
        vault policy write mira-policy /tmp/mira-policy.hcl

    run_with_status "Creating AppRole" \
        vault write auth/approle/role/mira policies="mira-policy" token_ttl=1h token_max_ttl=4h

    # Extract credentials
    vault read -field=role_id auth/approle/role/mira/role-id > /opt/vault/role-id.txt
    vault write -field=secret_id -f auth/approle/role/mira/secret-id > /opt/vault/secret-id.txt
}

# Vault helper: Store secret only if it doesn't exist
# Usage: vault_put_if_not_exists SECRET_PATH KEY1=VALUE1 KEY2=VALUE2 ...
#
# Reserved for secret/mira/database. It is the one entry that must NOT be
# overwritten on a re-run: when a re-run leaves CONFIG_DB_PASSWORD at the
# sentinel default, the ALTER USER block is skipped, so the roles keep the
# previous custom password and the matching Vault entry has to survive
# untouched. Every other entry belongs on vault_put_with_backup.
vault_put_if_not_exists() {
    local secret_path="$1"
    shift

    if vault kv get "$secret_path" &> /dev/null; then
        print_info "Secret already exists at $secret_path (preserving existing values)"
        return 0
    fi

    if ! run_with_status "Storing secret at $secret_path" vault kv put "$secret_path" "$@"; then
        print_error "Failed to store secret at $secret_path - deployment cannot continue"
        exit 1
    fi
}

# Vault helper: copy every field of a KV v2 secret to another path, verbatim.
# Usage: vault_copy_secret SOURCE_PATH DEST_PATH
#
# The CLI has no copy and no field-agnostic put, so the JSON body is read and
# rewritten through the HTTP API. Values never pass through a shell — newlines,
# quotes, backticks, and $ survive byte-identical — so the backup is a faithful
# copy of the operator's only credential, not a re-serialization of it.
vault_copy_secret() {
    local src_path="$1"
    local dst_path="$2"

    if ! vault status > /dev/null 2>&1; then
        print_error "Vault unreachable - cannot back up ${src_path}"
        return 1
    fi

    VAULT_ADDR="${VAULT_ADDR}" VAULT_TOKEN="$(vault print token)" \
        python3 - "$src_path" "$dst_path" <<'PY'
import json, os, sys, urllib.error, urllib.request

addr = os.environ["VAULT_ADDR"].rstrip("/")
token = os.environ["VAULT_TOKEN"]
src, dst = sys.argv[1], sys.argv[2]


def api_path(cli_path):
    # A CLI path is "<mount>/<key>", e.g. secret/mira/api_keys; the HTTP API
    # address for a KV v2 secret is "<mount>/data/<key>". Only the first
    # segment is the mount.
    mount, _, key = cli_path.partition("/")
    return f"{mount}/data/{key}"


def call(method, path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{addr}/v1/{path}",
        data=body,
        method=method,
        headers={"X-Vault-Token": token, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read() or b"{}")


try:
    fields = call("GET", api_path(src))["data"]["data"]
    call("POST", api_path(dst), {"data": fields})
except (urllib.error.URLError, KeyError, ValueError) as exc:
    sys.stderr.write(f"vault backup failed: {exc}\n")
    sys.exit(1)
PY
}

# Vault helper: Store a secret, keeping any previous value first.
# Usage: vault_put_with_backup SECRET_PATH KEY1=VALUE1 KEY2=VALUE2 ...
#
# A re-install lands on a Vault that still holds the previous install's
# credentials, and the paths the app reads are the same paths. The database's
# rename-aside precedent applies: preserve the old thing under a new name, then
# write the fresh one. The previous value is copied verbatim to
# "${secret_path}_legacy_${MIRA_LEGACY_STAMP}" — never overwritten in place —
# and the new values are then written to the primary path the app reads. A
# failed backup aborts: replacing the operator's only copy with no backup is
# not recoverable. Each backup path is recorded in MIRA_VAULT_BACKUPS so
# finalize.sh can report it at the end of the install.
#
# Exit codes are checked rather than trusted: `vault kv get` answers 2 both for
# "no value at this path" and for an unreachable Vault, so the reachability
# check comes first and a get that fails any other way aborts. Treating an
# unreadable existing secret as absent would overwrite it with no backup.
vault_put_with_backup() {
    local secret_path="$1"
    shift

    if ! vault status > /dev/null 2>&1; then
        print_error "Vault unreachable - cannot check ${secret_path} for a value to preserve"
        exit 1
    fi

    local get_rc=0
    vault kv get "$secret_path" > /dev/null 2>&1 || get_rc=$?
    if [ "$get_rc" -eq 2 ]; then
        : # no previous value at this path — nothing to preserve
    elif [ "$get_rc" -ne 0 ]; then
        print_error "Could not read ${secret_path} (vault kv get exit ${get_rc}) - aborting before it is overwritten"
        exit 1
    else
        local backup_path="${secret_path}_legacy_${MIRA_LEGACY_STAMP}"
        if ! vault_copy_secret "$secret_path" "$backup_path"; then
            print_error "Could not back up ${secret_path} to ${backup_path} - aborting before any credential is overwritten"
            exit 1
        fi
        MIRA_VAULT_BACKUPS+=("$backup_path")
        print_info "Previous ${secret_path} preserved at ${backup_path}"
    fi

    if ! run_with_status "Storing secret at $secret_path" vault kv put "$secret_path" "$@"; then
        print_error "Failed to store secret at $secret_path - deployment cannot continue"
        exit 1
    fi
}
