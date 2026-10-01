# deploy/lib/systemone_config.sh
# System One reachability probe for the injection screen
# Source this file - do not execute directly
#
# The injection screen's System One model is app config only — System One has
# no schema artifact (judgments are ephemeral, nothing is stored), so unlike
# embedding_config.sh this sets NO psql variables. Its one job is proving the
# configured endpoint answers the TypeSafe /v1/systemone wire contract before
# the install commits: an enabled-but-broken screen would park MIRA's boot gate
# (utils/power_on_self_test.py:_check_injection_screen) on every start.

# Vault key name under secret/mira/api_keys for a remote endpoint's bearer token.
SYSTEMONE_VAULT_KEY_NAME="systemone_key"

# probe_systemone PYTHON APP_DIR PROVIDER ENDPOINT MODEL API_KEY
#
# PYTHON runs with MIRA's requirements installed; APP_DIR is the MIRA checkout
# it imports from. PROVIDER is local or remote; API_KEY applies to remote only
# (piped to the probe on stdin, never on the command line). Prints the model
# name on success. On failure the probe's reason is on stderr and the function
# returns 1.
probe_systemone() {
    local python="$1" app_dir="$2" provider="$3" endpoint="$4" model="$5" api_key="$6"
    local describe='import sys; from clients.systemone_client import describe_for_installer; print(describe_for_installer(sys.argv[1:]))'

    case "$provider" in
        local)
            # Self-hosted endpoint: no Authorization header, no token.
            (cd "$app_dir" && "$python" -c "$describe" describe local "$endpoint" "$model") || return 1
            ;;
        remote)
            (cd "$app_dir" && printf '%s' "$api_key" | "$python" -c "$describe" describe remote "$endpoint" "$model") || return 1
            ;;
        *)
            echo "systemone provider must be local or remote, got '$provider'" >&2
            return 1
            ;;
    esac
}
