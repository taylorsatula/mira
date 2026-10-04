# deploy/lib/config_file.sh
# Flat-YAML config parser for `deploy.sh --config <file>`
# Source this file - do not execute directly
#
# Requires: lib/output.sh sourced first (print_error)
# Sourced by: config.sh, only when CONFIG_FILE is set (deploy.sh --config)
#
# The file format is deliberately restricted to flat `key: value` pairs —
# no nesting, no lists — so it parses in plain bash (bash 3.x compatible for
# macOS) with zero dependencies: pyyaml/yq are not guaranteed on a fresh
# install host. Every key in YAML_KNOWN_KEYS is REQUIRED; an unfilled
# `__SET_ME__` placeholder aborts before sudo is requested; an empty value
# ("") means the same as the interview's Enter: a lunaroute-gateway tier reuses
# the chat key, anything else proceeds with a PLACEHOLDER_* value (the final
# banner explains the Vault fixup).

YAML_SEEN_KEYS=""

yaml_fail() {
    print_error "$1"
    exit 1
}

yaml_store() {
    # yaml_store <known-key> <value> — whitelist gate; duplicates abort.
    local key="$1" value="$2"
    case ",$YAML_SEEN_KEYS," in
        *",$key,"*) yaml_fail "Duplicate key in config file: '$key'" ;;
    esac
    YAML_SEEN_KEYS="$YAML_SEEN_KEYS,$key"
    eval "YAML_$key=\"\$value\""
}

parse_yaml_config() {
    local file="$1" line key value
    [ -f "$file" ] || yaml_fail "Config file not found: $file"
    [ -s "$file" ] || yaml_fail "Config file is empty: $file"
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|'#'*) continue ;;
        esac
        case "$line" in
            ' '*|'	'*) yaml_fail "Nested YAML is not supported — flat 'key: value' lines only. Offending line: $line" ;;
        esac
        case "$line" in
            *:*) : ;;
            *) yaml_fail "Malformed config line (expected 'key: value'): $line" ;;
        esac
        key="${line%%:*}"
        value="${line#*:}"
        # trim whitespace (bash 3.x-safe parameter expansion)
        key="${key#"${key%%[![:space:]]*}"}"
        key="${key%"${key##*[![:space:]]}"}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        # strip a trailing comment: ' #' through end of line — unless the
        # value is wrapped in a matching quote pair, whose interior is kept
        # verbatim (an internal ' #' survives; only the quotes are removed).
        # A '#' with no leading whitespace (URL fragments, passwords)
        # survives untouched either way.
        case "$value" in
            \"?*\"|\'?*\') ;;  # quote-wrapped: skip the comment strip
            *)
                value="${value%%[[:space:]]#*}"
                value="${value#"${value%%[![:space:]]*}"}"
                ;;
        esac
        # strip one matching pair of surrounding quotes
        case "$value" in
            '"'*) value="${value#\"}"; value="${value%\"}" ;;
            "'"*) value="${value#\'}"; value="${value%\'}" ;;
        esac
        case "$key" in
            offline_mode|local_model_choice|custom_gguf|build_llama_cpp|llama_main_url|llama_small_url|llama_main_model|llama_small_model|chat_provider_type|chat_endpoint|chat_api_key|chat_model|anthropic_key|anthropic_batch_key|subcortical_endpoint|subcortical_api_key|subcortical_model|kagi_api_key|embedding_provider|embedding_endpoint|embedding_model|embedding_api_key|injection_screen|systemone_provider|systemone_endpoint|systemone_model|systemone_api_key|timezone|db_password|install_playwright|install_systemd|start_mira_now|overwrite_existing|stop_occupied_ports)
                yaml_store "$key" "$value" ;;
            *) yaml_fail "Unknown config key: '$key' (see deploy-config.example.yml)" ;;
        esac
    done < "$file"

    # every key is required — the template ships one line per key, so a
    # missing key means the file was hand-edited badly; name it.
    local k
    for k in offline_mode local_model_choice custom_gguf build_llama_cpp llama_main_url llama_small_url llama_main_model llama_small_model chat_provider_type chat_endpoint chat_api_key chat_model anthropic_key subcortical_endpoint subcortical_api_key subcortical_model kagi_api_key embedding_provider embedding_endpoint embedding_model embedding_api_key injection_screen systemone_provider systemone_endpoint systemone_model systemone_api_key timezone db_password install_playwright install_systemd start_mira_now overwrite_existing stop_occupied_ports; do
        case ",$YAML_SEEN_KEYS," in
            *",$k,"*) : ;;
            *) yaml_fail "Config file is missing required key: '$k'" ;;
        esac
    done

    # unfilled template placeholders abort before sudo is requested
    local bad="" v
    for k in offline_mode local_model_choice custom_gguf build_llama_cpp llama_main_url llama_small_url llama_main_model llama_small_model chat_provider_type chat_endpoint chat_api_key chat_model anthropic_key subcortical_endpoint subcortical_api_key subcortical_model kagi_api_key embedding_provider embedding_endpoint embedding_model embedding_api_key injection_screen systemone_provider systemone_endpoint systemone_model systemone_api_key timezone db_password install_playwright install_systemd start_mira_now overwrite_existing stop_occupied_ports; do
        eval "v=\"\$YAML_$k\""
        [ "$v" = "__SET_ME__" ] && bad="$bad $k"
    done
    [ -z "$bad" ] || yaml_fail "Unfilled placeholder(s) — replace __SET_ME__ with a value, or \"\" to skip:${bad}"

    # enum validation
    [ "$YAML_offline_mode" = "yes" ] || [ "$YAML_offline_mode" = "no" ] || yaml_fail "offline_mode must be yes or no"
    if [ "$YAML_offline_mode" = "no" ]; then
        [ "$YAML_chat_provider_type" = "anthropic" ] || [ "$YAML_chat_provider_type" = "openai" ] || yaml_fail "chat_provider_type must be anthropic or openai"
        [ -n "$YAML_subcortical_endpoint" ] || yaml_fail "subcortical_endpoint must not be empty"
        [ -n "$YAML_subcortical_model" ] || yaml_fail "subcortical_model must not be empty"
        if [ "$YAML_chat_provider_type" = "openai" ]; then
            [ -n "$YAML_chat_endpoint" ] || yaml_fail "chat_endpoint must not be empty for an openai provider"
            [ -n "$YAML_chat_model" ] || yaml_fail "chat_model must not be empty for an openai provider"
        else
            # The primary route is rewritten from chat_model at Step 13; an
            # empty value would install a route with no model.
            [ -n "$YAML_chat_model" ] || yaml_fail "chat_model must not be empty for an anthropic provider (e.g. claude-opus-4-6)"
        fi
    else
        [ "$YAML_local_model_choice" = "auto" ] || [ "$YAML_local_model_choice" = "custom" ] || yaml_fail "local_model_choice must be auto or custom"
        [ -n "$YAML_llama_main_url" ] || yaml_fail "llama_main_url must not be empty"
        [ -n "$YAML_llama_small_url" ] || yaml_fail "llama_small_url must not be empty"
        [ -n "$YAML_llama_main_model" ] || yaml_fail "llama_main_model must not be empty"
        [ -n "$YAML_llama_small_model" ] || yaml_fail "llama_small_model must not be empty"
        if [ "$YAML_local_model_choice" = "custom" ] && [ -z "$YAML_custom_gguf" ]; then
            yaml_fail "custom_gguf must not be empty when local_model_choice is custom"
        fi
    fi
    [ "$YAML_embedding_provider" = "local" ] || [ "$YAML_embedding_provider" = "remote" ] || yaml_fail "embedding_provider must be local or remote"
    if [ "$YAML_embedding_provider" = "remote" ]; then
        [ -n "$YAML_embedding_endpoint" ] || yaml_fail "embedding_endpoint must not be empty when embedding_provider is remote"
        [ -n "$YAML_embedding_model" ] || yaml_fail "embedding_model must not be empty when embedding_provider is remote"
    elif [ -n "$YAML_embedding_endpoint$YAML_embedding_model$YAML_embedding_api_key" ]; then
        yaml_fail "embedding_endpoint, embedding_model, and embedding_api_key apply only when embedding_provider is remote; set them to \"\""
    fi
    # Injection screen: an enabled screen needs a reachable System One model
    # (probed before the install commits); a disabled one probes and seeds
    # nothing, so all four systemone_* keys must be empty.
    if [ "$YAML_injection_screen" = "yes" ]; then
        [ "$YAML_systemone_provider" = "local" ] || [ "$YAML_systemone_provider" = "remote" ] || yaml_fail "systemone_provider must be local or remote"
        [ -n "$YAML_systemone_endpoint" ] || yaml_fail "systemone_endpoint must not be empty when injection_screen is yes"
        [ -n "$YAML_systemone_model" ] || yaml_fail "systemone_model must not be empty when injection_screen is yes"
        if [ "$YAML_systemone_provider" = "remote" ]; then
            # Lunaroute is full-service: an empty key means "reuse the chat
            # key". Any other remote endpoint must supply its own.
            if [ -z "$YAML_systemone_api_key" ]; then
                case "$YAML_systemone_endpoint" in
                    *gw.lunaroute.com*) : ;;
                    *) yaml_fail "systemone_api_key must not be empty when systemone_provider is remote (or point the endpoint at lunaroute, which reuses the chat key)" ;;
                esac
            fi
        else
            [ -z "$YAML_systemone_api_key" ] || yaml_fail "systemone_api_key must be \"\" when systemone_provider is local (self-hosted endpoints take no token)"
        fi
    elif [ -n "$YAML_systemone_provider$YAML_systemone_endpoint$YAML_systemone_model$YAML_systemone_api_key" ]; then
        yaml_fail "systemone_* keys apply only when injection_screen is yes; set them to \"\""
    fi
    local k2 v2
    for k2 in build_llama_cpp install_playwright install_systemd start_mira_now overwrite_existing stop_occupied_ports injection_screen; do
        eval "v2=\"\$YAML_$k2\""
        [ "$v2" = "yes" ] || [ "$v2" = "no" ] || yaml_fail "$k2 must be yes or no"
    done
}

apply_yaml_config() {
    # Map parsed YAML keys onto the same CONFIG_*/STATUS_* state the
    # interview produces — later phases cannot tell the two paths apart.
    # Requires parse_yaml_config to have run (YAML_* populated).
    CONFIG_OFFLINE_MODE="$YAML_offline_mode"
    CONFIG_BUILD_LLAMA_CPP="$YAML_build_llama_cpp"
    if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
        # Placeholder keys so Vault validation passes (interview parity)
        CONFIG_ANTHROPIC_KEY="OFFLINE_MODE_PLACEHOLDER"
        CONFIG_ANTHROPIC_BATCH_KEY="OFFLINE_MODE_PLACEHOLDER"
        CONFIG_LOCAL_MODEL_CHOICE="$YAML_local_model_choice"
        CONFIG_CUSTOM_GGUF="$YAML_custom_gguf"
        CONFIG_LLAMA_MAIN_MODEL="$YAML_llama_main_model"
        CONFIG_LLAMA_SMALL_MODEL="$YAML_llama_small_model"
        # postgresql.sh reads these env names; exporting here gives the
        # config file a direct path into the offline SQL rewrite.
        export MIRA_LLAMA_MAIN_URL="$YAML_llama_main_url"
        export MIRA_LLAMA_SMALL_URL="$YAML_llama_small_url"
        STATUS_CHAT_PROVIDER="${CHECKMARK} Local llama-server"
        STATUS_CHAT_KEY="${DIM}N/A (local)${RESET}"
        STATUS_SUBCORTICAL="${DIM}N/A (local)${RESET}"
        STATUS_SUBCORTICAL_KEY="${DIM}N/A (local)${RESET}"
    else
        CONFIG_CHAT_PROVIDER_TYPE="$YAML_chat_provider_type"
        if [ "$CONFIG_CHAT_PROVIDER_TYPE" = "openai" ]; then
            CONFIG_CHAT_ENDPOINT="$YAML_chat_endpoint"
            CONFIG_CHAT_MODEL="$YAML_chat_model"
            if [ -z "$YAML_chat_api_key" ]; then
                CONFIG_CHAT_API_KEY="PLACEHOLDER_SET_THIS_LATER"
                STATUS_CHAT_KEY="${WARNING} NOT SET - You must configure this before using MIRA"
            else
                CONFIG_CHAT_API_KEY="$YAML_chat_api_key"
                STATUS_CHAT_KEY="${CHECKMARK} Configured"
            fi
            # Interview parity: background Anthropic routes get placeholders
            CONFIG_ANTHROPIC_KEY="PLACEHOLDER_NOT_CONFIGURED"
            CONFIG_ANTHROPIC_BATCH_KEY="PLACEHOLDER_NOT_CONFIGURED"
            STATUS_CHAT_PROVIDER="${CHECKMARK} OpenAI-compatible (${CONFIG_CHAT_ENDPOINT})"
        else
            CONFIG_CHAT_MODEL="$YAML_chat_model"
            if [ -z "$YAML_anthropic_key" ]; then
                CONFIG_ANTHROPIC_KEY="PLACEHOLDER_SET_THIS_LATER"
                CONFIG_CHAT_API_KEY=""
                STATUS_CHAT_KEY="${WARNING} NOT SET - You must configure this before using MIRA"
            else
                CONFIG_ANTHROPIC_KEY="$YAML_anthropic_key"
                CONFIG_CHAT_API_KEY="$YAML_anthropic_key"
                STATUS_CHAT_KEY="${CHECKMARK} Configured"
            fi
            STATUS_CHAT_PROVIDER="${CHECKMARK} Anthropic"
        fi
        CONFIG_SUBCORTICAL_ENDPOINT="$YAML_subcortical_endpoint"
        STATUS_SUBCORTICAL="${CHECKMARK} ${CONFIG_SUBCORTICAL_ENDPOINT}"
        if [ -n "$YAML_subcortical_api_key" ]; then
            CONFIG_SUBCORTICAL_API_KEY="$YAML_subcortical_api_key"
            STATUS_SUBCORTICAL_KEY="${CHECKMARK} Configured"
        elif is_lunaroute_full_service "$CONFIG_SUBCORTICAL_ENDPOINT"; then
            CONFIG_SUBCORTICAL_API_KEY="$CONFIG_CHAT_API_KEY"
            STATUS_SUBCORTICAL_KEY="${CHECKMARK} Reusing your lunaroute key"
        else
            CONFIG_SUBCORTICAL_API_KEY="PLACEHOLDER_SET_THIS_LATER"
            STATUS_SUBCORTICAL_KEY="${WARNING} NOT SET - You must configure this before using MIRA"
        fi
        CONFIG_SUBCORTICAL_MODEL="$YAML_subcortical_model"
    fi
    if [ -z "$YAML_kagi_api_key" ]; then
        CONFIG_KAGI_KEY=""
        STATUS_KAGI="${DIM}Skipped${RESET}"
    else
        CONFIG_KAGI_KEY="$YAML_kagi_api_key"
        STATUS_KAGI="${CHECKMARK} Configured"
    fi
    CONFIG_EMBEDDING_PROVIDER="$YAML_embedding_provider"
    CONFIG_EMBEDDING_ENDPOINT="$YAML_embedding_endpoint"
    CONFIG_EMBEDDING_MODEL="$YAML_embedding_model"
    CONFIG_EMBEDDING_API_KEY="$YAML_embedding_api_key"
    if [ "$CONFIG_EMBEDDING_PROVIDER" = "remote" ]; then
        if [ -z "$CONFIG_EMBEDDING_API_KEY" ] && is_lunaroute_full_service "$CONFIG_EMBEDDING_ENDPOINT"; then
            CONFIG_EMBEDDING_API_KEY="$CONFIG_CHAT_API_KEY"
        fi
        STATUS_EMBEDDINGS="${CHECKMARK} Remote: ${CONFIG_EMBEDDING_MODEL} at ${CONFIG_EMBEDDING_ENDPOINT}"
    else
        STATUS_EMBEDDINGS="${CHECKMARK} Local model"
    fi
    CONFIG_INJECTION_SCREEN="$YAML_injection_screen"
    CONFIG_SYSTEMONE_PROVIDER="$YAML_systemone_provider"
    CONFIG_SYSTEMONE_ENDPOINT="$YAML_systemone_endpoint"
    CONFIG_SYSTEMONE_MODEL="$YAML_systemone_model"
    CONFIG_SYSTEMONE_API_KEY="$YAML_systemone_api_key"
    if [ "$CONFIG_INJECTION_SCREEN" = "yes" ] && [ "$CONFIG_SYSTEMONE_PROVIDER" = "remote" ] \
        && [ -z "$CONFIG_SYSTEMONE_API_KEY" ] && is_lunaroute_full_service "$CONFIG_SYSTEMONE_ENDPOINT"; then
        CONFIG_SYSTEMONE_API_KEY="$CONFIG_CHAT_API_KEY"
    fi
    if [ "$CONFIG_INJECTION_SCREEN" = "yes" ]; then
        STATUS_SYSTEMONE="${CHECKMARK} On: ${CONFIG_SYSTEMONE_PROVIDER} ${CONFIG_SYSTEMONE_MODEL} at ${CONFIG_SYSTEMONE_ENDPOINT}"
    else
        STATUS_SYSTEMONE="${DIM}Disabled${RESET}"
    fi
    if [ -z "$YAML_db_password" ]; then
        CONFIG_DB_PASSWORD="changethisifdeployingpwd"
        STATUS_DB_PASSWORD="${DIM}Using default password${RESET}"
    else
        CONFIG_DB_PASSWORD="$YAML_db_password"
        STATUS_DB_PASSWORD="${CHECKMARK} Custom password set"
    fi
    if [ -n "$YAML_timezone" ]; then
        CONFIG_TIMEZONE="$YAML_timezone"
    else
        CONFIG_TIMEZONE=""   # config.sh resolves "" to the system timezone
    fi
    if [ "$YAML_install_playwright" = "yes" ]; then
        CONFIG_INSTALL_PLAYWRIGHT="yes"
        STATUS_PLAYWRIGHT="${CHECKMARK} Will be installed"
    else
        CONFIG_INSTALL_PLAYWRIGHT="no"
        STATUS_PLAYWRIGHT="${YELLOW}Skipped${RESET}"
    fi
    if [ "$YAML_install_systemd" = "yes" ]; then
        CONFIG_INSTALL_SYSTEMD="yes"
        if [ "$YAML_start_mira_now" = "yes" ]; then
            STATUS_SYSTEMD="${CHECKMARK} Will be installed and started"
        else
            STATUS_SYSTEMD="${CHECKMARK} Will be installed (not started)"
        fi
    else
        CONFIG_INSTALL_SYSTEMD="no"
        STATUS_SYSTEMD="${DIM}Skipped${RESET}"
    fi
    CONFIG_START_MIRA_NOW="$YAML_start_mira_now"
    print_info "Configuration loaded from $CONFIG_FILE"
}
