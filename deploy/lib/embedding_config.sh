# deploy/lib/embedding_config.sh
# Embedding-space arguments for deploy/mira_service_schema.sql
# Source this file - do not execute directly
#
# The schema takes five psql variables (see its embedding_config table). The
# model name and vector length come from the app itself —
# clients/embeddings_provider.py:describe_for_installer, which probes a
# remote endpoint for its vector length — never from literals in this file.

# Vault key name under secret/mira/api_keys for a remote endpoint's bearer token.
EMBEDDING_VAULT_KEY_NAME="embeddings_key"

# resolve_embedding_schema_args PYTHON APP_DIR PROVIDER ENDPOINT MODEL API_KEY
#
# PYTHON runs with MIRA's requirements installed; APP_DIR is the MIRA checkout
# it imports from. PROVIDER is local or remote; ENDPOINT, MODEL, and API_KEY
# apply to remote only (API_KEY may be empty for an endpoint that takes none).
# Sets EMBEDDING_SCHEMA_ARGS (psql -v arguments), EMBEDDING_MODEL, and
# EMBEDDING_DIMENSIONS. On failure the probe's reason is on stderr and the
# function returns 1.
resolve_embedding_schema_args() {
    local python="$1" app_dir="$2" provider="$3" endpoint="$4" model="$5" api_key="$6"
    local describe='import sys; from clients.embeddings_provider import describe_for_installer; print(describe_for_installer(sys.argv[1:]))'
    local described key_name=""

    case "$provider" in
        local)
            described=$(cd "$app_dir" && "$python" -c "$describe" describe local) || return 1
            endpoint=""
            ;;
        remote)
            described=$(cd "$app_dir" && printf '%s' "$api_key" | "$python" -c "$describe" describe remote "$endpoint" "$model") || return 1
            if [ -n "$api_key" ]; then
                key_name="$EMBEDDING_VAULT_KEY_NAME"
            fi
            ;;
        *)
            echo "embedding provider must be local or remote, got '$provider'" >&2
            return 1
            ;;
    esac

    read -r EMBEDDING_MODEL EMBEDDING_DIMENSIONS <<< "$described"
    case "$EMBEDDING_DIMENSIONS" in
        ''|*[!0-9]*)
            echo "embedding probe did not report a vector length (got: '$described')" >&2
            return 1
            ;;
    esac

    EMBEDDING_SCHEMA_ARGS=(
        -v "embedding_provider=$provider"
        -v "embedding_model=$EMBEDDING_MODEL"
        -v "embedding_endpoint_url=$endpoint"
        -v "embedding_api_key_name=$key_name"
        -v "embedding_dimensions=$EMBEDDING_DIMENSIONS"
    )
}
