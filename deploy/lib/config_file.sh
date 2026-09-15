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
# ("") means the same as the interview's Enter-to-skip (the deploy proceeds
# with PLACEHOLDER_* values and the final banner explains the Vault fixup).

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
        # strip a trailing comment: ' #' through end of line. A '#' with no
        # leading whitespace (URL fragments, passwords) survives untouched.
        value="${value%%[[:space:]]#*}"
        value="${value#"${value%%[![:space:]]*}"}"
        # strip one matching pair of surrounding quotes
        case "$value" in
            '"'*) value="${value#\"}"; value="${value%\"}" ;;
            "'"*) value="${value#\'}"; value="${value%\'}" ;;
        esac
        case "$key" in
            offline_mode|local_model_choice|custom_gguf|build_llama_cpp|llama_main_url|llama_small_url|llama_main_model|llama_small_model|chat_provider_type|chat_endpoint|chat_api_key|chat_model|anthropic_key|anthropic_batch_key|subcortical_endpoint|subcortical_api_key|subcortical_model|kagi_api_key|timezone|db_password|install_playwright|install_systemd|start_mira_now|overwrite_existing|stop_occupied_ports)
                yaml_store "$key" "$value" ;;
            *) yaml_fail "Unknown config key: '$key' (see deploy-config.example.yml)" ;;
        esac
    done < "$file"

    # every key is required — the template ships one line per key, so a
    # missing key means the file was hand-edited badly; name it.
    local k
    for k in offline_mode local_model_choice custom_gguf build_llama_cpp llama_main_url llama_small_url llama_main_model llama_small_model chat_provider_type chat_endpoint chat_api_key chat_model anthropic_key anthropic_batch_key subcortical_endpoint subcortical_api_key subcortical_model kagi_api_key timezone db_password install_playwright install_systemd start_mira_now overwrite_existing stop_occupied_ports; do
        case ",$YAML_SEEN_KEYS," in
            *",$k,"*) : ;;
            *) yaml_fail "Config file is missing required key: '$k'" ;;
        esac
    done

    # unfilled template placeholders abort before sudo is requested
    local bad="" v
    for k in offline_mode local_model_choice custom_gguf build_llama_cpp llama_main_url llama_small_url llama_main_model llama_small_model chat_provider_type chat_endpoint chat_api_key chat_model anthropic_key anthropic_batch_key subcortical_endpoint subcortical_api_key subcortical_model kagi_api_key timezone db_password install_playwright install_systemd start_mira_now overwrite_existing stop_occupied_ports; do
        eval "v=\"\$YAML_$k\""
        [ "$v" = "__SET_ME__" ] && bad="$bad $k"
    done
    [ -z "$bad" ] || yaml_fail "Unfilled placeholder(s) — replace __SET_ME__ with a value, or \"\" to skip:${bad}"

    # enum validation
    [ "$YAML_offline_mode" = "yes" ] || [ "$YAML_offline_mode" = "no" ] || yaml_fail "offline_mode must be yes or no"
    if [ "$YAML_offline_mode" = "no" ]; then
        [ "$YAML_chat_provider_type" = "anthropic" ] || [ "$YAML_chat_provider_type" = "generic" ] || yaml_fail "chat_provider_type must be anthropic or generic"
        [ -n "$YAML_subcortical_endpoint" ] || yaml_fail "subcortical_endpoint must not be empty"
        [ -n "$YAML_subcortical_model" ] || yaml_fail "subcortical_model must not be empty"
        if [ "$YAML_chat_provider_type" = "generic" ]; then
            [ -n "$YAML_chat_endpoint" ] || yaml_fail "chat_endpoint must not be empty for a generic provider"
            [ -n "$YAML_chat_model" ] || yaml_fail "chat_model must not be empty for a generic provider"
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
    local k2 v2
    for k2 in build_llama_cpp install_playwright install_systemd start_mira_now overwrite_existing stop_occupied_ports; do
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
        if [ "$CONFIG_CHAT_PROVIDER_TYPE" = "generic" ]; then
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
            STATUS_CHAT_PROVIDER="${CHECKMARK} Generic (${CONFIG_CHAT_ENDPOINT})"
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
            if [ -z "$YAML_anthropic_batch_key" ]; then
                CONFIG_ANTHROPIC_BATCH_KEY="$CONFIG_ANTHROPIC_KEY"
            else
                CONFIG_ANTHROPIC_BATCH_KEY="$YAML_anthropic_batch_key"
            fi
            STATUS_CHAT_PROVIDER="${CHECKMARK} Anthropic"
        fi
        CONFIG_SUBCORTICAL_ENDPOINT="$YAML_subcortical_endpoint"
        STATUS_SUBCORTICAL="${CHECKMARK} ${CONFIG_SUBCORTICAL_ENDPOINT}"
        if [ -z "$YAML_subcortical_api_key" ]; then
            CONFIG_SUBCORTICAL_API_KEY="PLACEHOLDER_SET_THIS_LATER"
            STATUS_SUBCORTICAL_KEY="${WARNING} NOT SET - You must configure this before using MIRA"
        else
            CONFIG_SUBCORTICAL_API_KEY="$YAML_subcortical_api_key"
            STATUS_SUBCORTICAL_KEY="${CHECKMARK} Configured"
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
