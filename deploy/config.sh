# deploy/config.sh
# Interactive configuration gathering for MIRA deployment
# Source this file - do not execute directly
#
# Requires: lib/output.sh and lib/services.sh sourced first
# Requires: LOUD_MODE variable set
#
# Sets: CONFIG_*, STATUS_*, OS, DISTRO

# Initialize configuration state (using simple variables for Bash 3.x compatibility)
CONFIG_ANTHROPIC_KEY=""
CONFIG_ANTHROPIC_BATCH_KEY=""
CONFIG_KAGI_KEY=""
CONFIG_DB_PASSWORD=""
CONFIG_INSTALL_PLAYWRIGHT=""
CONFIG_INSTALL_SYSTEMD=""
CONFIG_START_MIRA_NOW=""
CONFIG_OFFLINE_MODE=""
CONFIG_LOCAL_MODEL_CHOICE=""       # auto / custom
CONFIG_LLAMA_MAIN_MODEL=""            # GGUF served on port 3090; set per docs/OFFLINE_MODELS.md
CONFIG_LLAMA_SMALL_MODEL=""           # GGUF served on port 3092; set per docs/OFFLINE_MODELS.md
CONFIG_CHAT_PROVIDER_TYPE=""
CONFIG_CHAT_ENDPOINT=""
CONFIG_CHAT_API_KEY=""
CONFIG_CHAT_MODEL=""
CONFIG_SUBCORTICAL_ENDPOINT=""
CONFIG_SUBCORTICAL_API_KEY=""
CONFIG_SUBCORTICAL_MODEL=""
CONFIG_BUILD_LLAMA_CPP=""
CONFIG_TIMEZONE=""                   # IANA name; empty = this host's system timezone
CONFIG_EMBEDDING_PROVIDER=""         # local / remote
CONFIG_EMBEDDING_ENDPOINT=""         # remote only: POST /v1/embeddings URL
CONFIG_EMBEDDING_MODEL=""            # remote only
CONFIG_EMBEDDING_API_KEY=""          # remote only; empty for an endpoint that takes none
CONFIG_INJECTION_SCREEN=""           # yes / no
CONFIG_SYSTEMONE_PROVIDER=""         # enabled only: local / remote
CONFIG_SYSTEMONE_ENDPOINT=""         # enabled only: POST /v1/systemone URL
CONFIG_SYSTEMONE_MODEL=""            # enabled only
CONFIG_SYSTEMONE_API_KEY=""          # remote only; lunaroute reuses the chat key, others explicit
STATUS_CHAT_PROVIDER=""
STATUS_CHAT_KEY=""
STATUS_SUBCORTICAL=""
STATUS_SUBCORTICAL_KEY=""
STATUS_KAGI=""
STATUS_EMBEDDINGS=""
STATUS_SYSTEMONE=""
STATUS_DB_PASSWORD=""
STATUS_TIMEZONE=""
STATUS_PLAYWRIGHT=""
STATUS_SYSTEMD=""
STATUS_MIRA_SERVICE=""

# True when the endpoint is the lunaroute gateway and the chat tier holds a
# usable key. Lunaroute is full-service: one key covers chat, subcortical,
# embeddings, and System One, so no other tier needs its own key prompt.
is_lunaroute_full_service() {
    case "$1" in
        *gw.lunaroute.com*) : ;;
        *) return 1 ;;
    esac
    [ "$CONFIG_CHAT_PROVIDER_TYPE" = "openai" ] || return 1
    [ -n "${CONFIG_CHAT_API_KEY:-}" ] || return 1
    [ "$CONFIG_CHAT_API_KEY" != "PLACEHOLDER_SET_THIS_LATER" ] || return 1
    return 0
}

# Detect this host's IANA timezone — the default for CONFIG_TIMEZONE when the
# operator leaves it empty. Mirrors utils/timezone_utils.py get_default_timezone()
# so the install default and the app-side fallback agree.
detect_system_timezone() {
    if [ -f /etc/timezone ]; then
        head -1 /etc/timezone | tr -d '[:space:]'
        return
    fi
    if [ -L /etc/localtime ]; then
        readlink /etc/localtime | sed 's|.*zoneinfo/||'
        return
    fi
    echo "UTC"
}
SYSTEM_TIMEZONE="$(detect_system_timezone)"

# --config <file>: parse and apply the declarative config, bypassing the
# interview entirely. Fails before sudo is requested on unfilled __SET_ME__
# placeholders, duplicate/unknown/missing keys, or bad enum values.
if [ -n "$CONFIG_FILE" ]; then
    source "${SCRIPT_DIR}/lib/config_file.sh"
    parse_yaml_config "$CONFIG_FILE"
    apply_yaml_config
fi

if [ -t 1 ]; then clear; fi
echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════╗"
echo "║   MIRA Deployment Script (main)        ║"
echo "╚════════════════════════════════════════╝"
echo -e "${RESET}"
[ "$LOUD_MODE" = true ] && print_info "Running in verbose mode (--loud)"
echo ""

print_header "Pre-flight Checks"

# Check available disk space (need at least 10GB)
echo -ne "${DIM}${ARROW}${RESET} Checking disk space... "
AVAILABLE_SPACE=$(df -k /opt 2>/dev/null | awk 'NR==2 {print $4}')
if [ -z "$AVAILABLE_SPACE" ]; then
    # /opt is absent on most macOS hosts; fall back to the root filesystem.
    AVAILABLE_SPACE=$(df -k / | awk 'NR==2 {print $4}')
fi
REQUIRED_SPACE=10485760  # 10GB in KB (df -k reports 1K blocks on every platform)
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo -e "${ERROR}"
    print_error "Insufficient disk space. Need at least 10GB free, found $(($AVAILABLE_SPACE / 1024 / 1024))GB"
    exit 1
fi
echo -e "${CHECKMARK}"

# Check if installation already exists
if [ -d "/opt/mira/app" ]; then
    echo ""
    print_warning "Existing MIRA installation found at /opt/mira/app"
    if [ -n "$CONFIG_FILE" ]; then
        if [ "$YAML_overwrite_existing" = "yes" ]; then
            print_info "Proceeding with overwrite (overwrite_existing: yes)"
        else
            print_info "Installation cancelled (overwrite_existing: no)."
            exit 0
        fi
    else
        read -p "$(echo -e ${YELLOW}This will OVERWRITE the existing installation. Continue? ${RESET})(y/n): " OVERWRITE
        if [[ ! "$OVERWRITE" =~ ^[Yy](es)?$ ]]; then
            print_info "Installation cancelled."
            exit 0
        fi
        print_info "Proceeding with overwrite..."
    fi
    echo ""
fi

print_success "Pre-flight checks passed"

# Detect operating system (needed for port stop logic and later steps)
OS_TYPE=$(uname -s)
case "$OS_TYPE" in
    Linux*)
        OS="linux"
        # Detect Linux distribution family
        if [ -f /etc/redhat-release ] || [ -f /etc/fedora-release ]; then
            DISTRO="fedora"
        elif [ -f /etc/debian_version ]; then
            DISTRO="debian"
        else
            # Fall back to checking /etc/os-release
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                case "$ID" in
                    fedora|rhel|centos|rocky|alma)
                        DISTRO="fedora"
                        ;;
                    debian|ubuntu|linuxmint|pop)
                        DISTRO="debian"
                        ;;
                    arch)
                        DISTRO="arch"
                        ;;
                    *)
                        # Check ID_LIKE for derivatives
                        case "$ID_LIKE" in
                            *fedora*|*rhel*)
                                DISTRO="fedora"
                                ;;
                            *debian*|*ubuntu*)
                                DISTRO="debian"
                                ;;
                            *arch*)
                                DISTRO="arch"
                                ;;
                            *)
                                DISTRO="unknown"
                                ;;
                        esac
                        ;;
                esac
            else
                DISTRO="unknown"
            fi
        fi
        ;;
    Darwin*)
        OS="macos"
        DISTRO=""
        ;;
    *)
        echo ""
        print_error "Unsupported operating system: $OS_TYPE"
        print_info "Supported: Linux (Debian/Ubuntu, Fedora/RHEL/CentOS, Arch) and macOS"
        print_info "For other platforms, see manual installation: docs/MANUAL_INSTALL.md"
        exit 1
        ;;
esac

# --config on macOS: systemd cannot be installed; force the interview's
# macOS semantics so the summary and finalize agree (OS is detected above;
# apply_yaml_config runs before detection and cannot see it).
if [ -n "$CONFIG_FILE" ] && [ "$OS" = "macos" ]; then
    CONFIG_INSTALL_SYSTEMD="no"
    # start_mira_now flows through from the YAML: macOS supervision is a
    # launchd agent (finalize.sh Step 15b2), so "start now" is meaningful.
    STATUS_SYSTEMD="${DIM}N/A (macOS — launchd agent)${RESET}"
fi

print_header "Port Availability Check"

echo -ne "${DIM}${ARROW}${RESET} Checking ports 1993, 8200, 6379, 5432... "
PORTS_IN_USE=""
for PORT in 1993 8200 6379 5432; do
    # One sanctioned probe (lib/services.sh:port_probe_status — lsof → ss →
    # netstat). Indeterminate must never report a false pass: with no probe
    # tool at all, port occupancy cannot be verified and the deploy stops.
    # `|| PROBE=$?` form: a bare call returning 1 (free) would trip set -e.
    PROBE=0
    port_probe_status "$PORT" || PROBE=$?
    case $PROBE in
        0) PORTS_IN_USE="$PORTS_IN_USE $PORT";;
        2)
            echo -e "${ERROR}"
            print_error "Port check indeterminate: no port probe tool (lsof, ss, or netstat) is installed."
            print_info "Install lsof or iproute2 (ss) so port availability can be verified, then re-run."
            exit 1;;
    esac
done

if [ -n "$PORTS_IN_USE" ]; then
    echo -e "${WARNING}"
    print_warning "The following ports are already in use:$PORTS_IN_USE"
    print_info "MIRA requires: 1993 (app), 8200 (vault), 6379 (valkey), 5432 (postgresql)"
    if [ -n "$CONFIG_FILE" ]; then
        if [ "$YAML_stop_occupied_ports" = "yes" ]; then
            print_info "Stopping services on occupied ports (stop_occupied_ports: yes)"
            CONTINUE="y"
        else
            print_info "Installation cancelled (stop_occupied_ports: no). Free up the required ports and try again."
            exit 0
        fi
    else
        read -p "$(echo -e ${YELLOW}Stop existing services and continue?${RESET}) (y/n): " CONTINUE
        if [[ ! "$CONTINUE" =~ ^[Yy](es)?$ ]]; then
            print_info "Installation cancelled. Free up the required ports and try again."
            exit 0
        fi
    fi
    echo ""

    # Stop services on occupied ports using unified stop_service function
    print_info "Stopping services on occupied ports..."
    for PORT in $PORTS_IN_USE; do
        case $PORT in
            8200)
                # Vault - canonical method per OS, fallback to port-based stop
                if [ "$OS" = "linux" ]; then
                    echo -ne "${DIM}${ARROW}${RESET} Stopping Vault (port 8200)... "
                    if check_exists service_systemctl vault; then
                        stop_service vault systemctl && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    else
                        stop_service "Vault" port 8200 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    fi
                elif [ "$OS" = "macos" ]; then
                    echo -ne "${DIM}${ARROW}${RESET} Stopping Vault (port 8200)... "
                    # launchd agent: bootout, not kill — the KeepAlive agent
                    # resurrects a killed process, and the legacy vault.pid
                    # file no longer exists (the background-process path is
                    # gone). Port-based kill is the fallback for pre-launchd
                    # installs.
                    if [ -f "$HOME/Library/LaunchAgents/com.mira.vault.plist" ]; then
                        launchctl bootout "gui/$(id -u)/com.mira.vault" 2>/dev/null && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    else
                        stop_service "Vault" port 8200 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    fi
                fi
                ;;
            6379)
                # Valkey - canonical method per OS
                echo -ne "${DIM}${ARROW}${RESET} Stopping Valkey (port 6379)... "
                if [ "$OS" = "linux" ]; then
                    # One sanctioned resolver (lib/services.sh): Debian/Ubuntu
                    # ship valkey-server.service, Fedora/Arch valkey.service.
                    VUNIT=$(resolve_valkey_unit || true)
                    if [ -n "$VUNIT" ] && systemctl is-active --quiet "$VUNIT"; then
                        stop_service "${VUNIT%.service}" systemctl && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    else
                        stop_service "Valkey" port 6379 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    fi
                elif [ "$OS" = "macos" ]; then
                    if check_exists service_brew valkey; then
                        stop_service valkey brew && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    else
                        stop_service "Valkey" port 6379 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    fi
                fi
                ;;
            5432)
                # PostgreSQL - canonical method per OS
                echo -ne "${DIM}${ARROW}${RESET} Stopping PostgreSQL (port 5432)... "
                if [ "$OS" = "linux" ]; then
                    # One sanctioned resolver (lib/services.sh): PGDG Fedora
                    # ships postgresql-17.service, Debian/Arch postgresql.service.
                    PUNIT=$(resolve_pg_unit || true)
                    if [ -n "$PUNIT" ] && systemctl is-active --quiet "$PUNIT"; then
                        stop_service "${PUNIT%.service}" systemctl && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    else
                        stop_service "PostgreSQL" port 5432 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    fi
                elif [ "$OS" = "macos" ]; then
                    if check_exists service_brew postgresql@17; then
                        stop_service postgresql@17 brew && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    else
                        stop_service "PostgreSQL" port 5432 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                    fi
                fi
                ;;
            1993)
                # MIRA - canonical method per OS
                echo -ne "${DIM}${ARROW}${RESET} Stopping MIRA (port 1993)... "
                if [ "$OS" = "linux" ] && check_exists service_systemctl mira; then
                    stop_service mira systemctl && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                else
                    stop_service "MIRA" port 1993 && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                fi
                ;;
            *)
                # Unknown service - use port-based stop
                echo -ne "${DIM}${ARROW}${RESET} Stopping process on port $PORT... "
                stop_service "Unknown" port $PORT && echo -e "${CHECKMARK}" || echo -e "${WARNING}"
                ;;
        esac
    done
    echo ""
else
    echo -e "${CHECKMARK}"
fi

print_success "Port check passed"

# --config mode already applied every choice above; skip the interview.
if [ -z "$CONFIG_FILE" ]; then

print_header "LLM Provider Configuration"

echo -e "${BOLD}${BLUE}LLM Provider${RESET}"
read -p "$(echo -e ${CYAN}Use local LLM via llama-server?${RESET}) (y/n, default=n): " USE_LOCAL_LLM_INPUT
if [[ "$USE_LOCAL_LLM_INPUT" =~ ^[Yy](es)?$ ]]; then
    CONFIG_OFFLINE_MODE="yes"
    # Placeholder keys so Vault validation passes
    CONFIG_ANTHROPIC_KEY="OFFLINE_MODE_PLACEHOLDER"
    CONFIG_ANTHROPIC_BATCH_KEY="OFFLINE_MODE_PLACEHOLDER"
    STATUS_CHAT_PROVIDER="${CHECKMARK} Local llama-server"
    STATUS_CHAT_KEY="${DIM}N/A (local)${RESET}"
    STATUS_SUBCORTICAL="${DIM}N/A (local)${RESET}"
    STATUS_SUBCORTICAL_KEY="${DIM}N/A (local)${RESET}"

    echo ""
    echo -e "${BOLD}Local Model Configuration${RESET}"
    echo -e "${DIM}   Two llama-server instances are expected — a capable main model on${RESET}"
    echo -e "${DIM}   port 3090 and a small fast model on port 3092. The fast model backs${RESET}"
    echo -e "${DIM}   the analysis/subcortical route, which runs on every message, so speed${RESET}"
    echo -e "${DIM}   matters more there than capability.${RESET}"
    echo ""
    echo -e "${DIM}   Supply the GGUF files yourself; docs/OFFLINE_MODELS.md covers where they${RESET}"
    echo -e "${DIM}   go and how to start both servers. The recommended split assumes roughly${RESET}"
    echo -e "${DIM}   48GB of VRAM across two cards.${RESET}"
    echo ""
    echo -e "${DIM}   If your hardware differs or lacks sufficient VRAM, choose custom${RESET}"
    echo -e "${DIM}   and download your own GGUF model after install completes.${RESET}"
    echo ""
    echo "     1. Recommended split (main + small; see docs/OFFLINE_MODELS.md)"
    echo "     2. Custom (bring your own GGUF model)"
    read -p "$(echo -e ${CYAN}Model selection${RESET}) [1-2, default=1]: " LOCAL_MODEL_CHOICE
    if [[ "$LOCAL_MODEL_CHOICE" == "2" ]]; then
        CONFIG_LOCAL_MODEL_CHOICE="custom"
        echo ""
        echo -e "${DIM}   Enter the path or URL of your GGUF model file:${RESET}"
        read -p "$(echo -e ${CYAN}Custom GGUF model${RESET}): " CUSTOM_GGUF_INPUT
        if [ -n "$CUSTOM_GGUF_INPUT" ]; then
            CONFIG_CUSTOM_GGUF="$CUSTOM_GGUF_INPUT"
        fi
    else
        CONFIG_LOCAL_MODEL_CHOICE="auto"
    fi
    # Both interview branches (auto and custom) seed model_configs.model via
    # postgresql.sh, so collect the two names here either way — mirroring
    # the --config path (deploy/lib/config_file.sh, llama_main_model/llama_small_model).
    echo ""
    read -p "$(echo -e ${CYAN}Main model name${RESET}) (default=local-main): " LLAMA_MAIN_MODEL_INPUT
    CONFIG_LLAMA_MAIN_MODEL="${LLAMA_MAIN_MODEL_INPUT:-local-main}"
    read -p "$(echo -e ${CYAN}Small model name${RESET}) (default=local-small): " LLAMA_SMALL_MODEL_INPUT
    CONFIG_LLAMA_SMALL_MODEL="${LLAMA_SMALL_MODEL_INPUT:-local-small}"
else
    CONFIG_OFFLINE_MODE="no"

    # Chat Provider
    echo -e "${BOLD}${BLUE}1. Chat Provider${RESET}"
    echo -e "${DIM}   Pick your main chat provider:${RESET}"
    echo "     1. OpenAI-compatible endpoint (default — lunaroute gateway)"
    echo "     2. Anthropic"
    read -p "$(echo -e ${CYAN}Select provider${RESET}) [1-2, default=1]: " CHAT_PROVIDER_CHOICE

    case "${CHAT_PROVIDER_CHOICE:-1}" in
        1)
            CONFIG_CHAT_PROVIDER_TYPE="openai"

            # OpenAI-compatible endpoint URL
            echo ""
            read -p "$(echo -e ${CYAN}Endpoint URL${RESET}) [default: https://gw.lunaroute.com/v1/chat/completions]: " CHAT_ENDPOINT_INPUT
            CONFIG_CHAT_ENDPOINT="${CHAT_ENDPOINT_INPUT:-https://gw.lunaroute.com/v1/chat/completions}"

            # API key
            echo -e "${BOLD}${BLUE}   Chat API Key${RESET}"
            while true; do
                read -p "$(echo -e ${CYAN}Enter key${RESET}) (or Enter to skip): " CHAT_KEY_INPUT
                if [ -z "$CHAT_KEY_INPUT" ]; then
                    CONFIG_CHAT_API_KEY="PLACEHOLDER_SET_THIS_LATER"
                    STATUS_CHAT_KEY="${WARNING} NOT SET - You must configure this before using MIRA"
                    break
                fi
                CONFIG_CHAT_API_KEY="$CHAT_KEY_INPUT"
                STATUS_CHAT_KEY="${CHECKMARK} Configured"
                break
            done

            # Model
            read -p "$(echo -e ${CYAN}Model name${RESET}) [default: glm-5.3]: " CHAT_MODEL_INPUT
            CONFIG_CHAT_MODEL="${CHAT_MODEL_INPUT:-glm-5.3}"

            # Anthropic placeholders (background tasks won't work without real keys)
            CONFIG_ANTHROPIC_KEY="PLACEHOLDER_NOT_CONFIGURED"
            CONFIG_ANTHROPIC_BATCH_KEY="PLACEHOLDER_NOT_CONFIGURED"

            STATUS_CHAT_PROVIDER="${CHECKMARK} OpenAI-compatible (${CONFIG_CHAT_ENDPOINT})"

            ;;
        *)
            CONFIG_CHAT_PROVIDER_TYPE="anthropic"

            # Anthropic API Key
            echo ""
            echo -e "${BOLD}${BLUE}   Anthropic API Key${RESET} ${DIM}(console.anthropic.com/settings/keys)${RESET}"
            while true; do
                read -p "$(echo -e ${CYAN}Enter key${RESET}) (or Enter to skip): " ANTHROPIC_KEY_INPUT
                if [ -z "$ANTHROPIC_KEY_INPUT" ]; then
                    CONFIG_ANTHROPIC_KEY="PLACEHOLDER_SET_THIS_LATER"
                    CONFIG_CHAT_API_KEY=""
                    STATUS_CHAT_KEY="${WARNING} NOT SET - You must configure this before using MIRA"
                    break
                fi
                if [[ $ANTHROPIC_KEY_INPUT =~ ^sk-ant- ]]; then
                    CONFIG_ANTHROPIC_KEY="$ANTHROPIC_KEY_INPUT"
                    CONFIG_CHAT_API_KEY="$ANTHROPIC_KEY_INPUT"
                    STATUS_CHAT_KEY="${CHECKMARK} Configured"
                    break
                else
                    print_warning "This doesn't look like a valid Anthropic API key (should start with 'sk-ant-')"
                    read -p "$(echo -e ${YELLOW}Continue anyway?${RESET}) (y=yes, n=exit, t=try again): " CONFIRM
                    if [[ "$CONFIRM" =~ ^[Yy](es)?$ ]]; then
                        CONFIG_ANTHROPIC_KEY="$ANTHROPIC_KEY_INPUT"
                        CONFIG_CHAT_API_KEY="$ANTHROPIC_KEY_INPUT"
                        STATUS_CHAT_KEY="${CHECKMARK} Configured (unvalidated)"
                        break
                    elif [[ "$CONFIRM" =~ ^[Tt](ry)?$ ]]; then
                        continue
                    else
                        CONFIG_ANTHROPIC_KEY="PLACEHOLDER_SET_THIS_LATER"
                        CONFIG_CHAT_API_KEY=""
                        STATUS_CHAT_KEY="${WARNING} NOT SET"
                        break
                    fi
                fi
            done

            # Model (default: claude-opus-4-6)
            read -p "$(echo -e ${CYAN}Model${RESET}) [default: claude-opus-4-6]: " CHAT_MODEL_INPUT
            CONFIG_CHAT_MODEL="${CHAT_MODEL_INPUT:-claude-opus-4-6}"

            STATUS_CHAT_PROVIDER="${CHECKMARK} Anthropic"
            ;;
    esac

    # Subcortical
    echo -e "${BOLD}${BLUE}2. Subcortical${RESET}"
    echo -e "${DIM}   Runs on every message for memory retrieval (query expansion, entity extraction).${RESET}"
    echo -e "${DIM}   A fast inference provider works best here; the default is the lunaroute${RESET}"
    echo -e "${DIM}   gateway serving glm-5.3-flash (the same key as the chat tier works).${RESET}"
    echo ""
    read -p "$(echo -e ${CYAN}Endpoint URL${RESET}) [default: https://gw.lunaroute.com/v1/chat/completions]: " SUBCORTICAL_ENDPOINT_INPUT
    CONFIG_SUBCORTICAL_ENDPOINT="${SUBCORTICAL_ENDPOINT_INPUT:-https://gw.lunaroute.com/v1/chat/completions}"
    STATUS_SUBCORTICAL="${CHECKMARK} ${CONFIG_SUBCORTICAL_ENDPOINT}"

    # Subcortical API Key — lunaroute is full-service: one key covers every tier.
    if is_lunaroute_full_service "$CONFIG_SUBCORTICAL_ENDPOINT"; then
        CONFIG_SUBCORTICAL_API_KEY="$CONFIG_CHAT_API_KEY"
        STATUS_SUBCORTICAL_KEY="${CHECKMARK} Reusing your lunaroute key"
    else
        echo -e "${BOLD}${BLUE}   Subcortical API Key${RESET}"
        while true; do
            read -p "$(echo -e ${CYAN}Enter key${RESET}) (or Enter to skip): " SUBCORTICAL_KEY_INPUT
            if [ -z "$SUBCORTICAL_KEY_INPUT" ]; then
                CONFIG_SUBCORTICAL_API_KEY="PLACEHOLDER_SET_THIS_LATER"
                STATUS_SUBCORTICAL_KEY="${WARNING} NOT SET - You must configure this before using MIRA"
                break
            fi
            # Validate gsk_ prefix if Groq endpoint detected
            if [[ "$CONFIG_SUBCORTICAL_ENDPOINT" == *"groq.com"* ]] && [[ ! $SUBCORTICAL_KEY_INPUT =~ ^gsk_ ]]; then
                print_warning "This doesn't look like a valid Groq API key (should start with 'gsk_')"
                read -p "$(echo -e ${YELLOW}Continue anyway?${RESET}) (y=yes, t=try again): " CONFIRM
                if [[ "$CONFIRM" =~ ^[Yy](es)?$ ]]; then
                    CONFIG_SUBCORTICAL_API_KEY="$SUBCORTICAL_KEY_INPUT"
                    STATUS_SUBCORTICAL_KEY="${CHECKMARK} Configured (unvalidated)"
                    break
                elif [[ "$CONFIRM" =~ ^[Tt](ry)?$ ]]; then
                    continue
                fi
            else
                CONFIG_SUBCORTICAL_API_KEY="$SUBCORTICAL_KEY_INPUT"
                STATUS_SUBCORTICAL_KEY="${CHECKMARK} Configured"
                break
            fi
        done
    fi

    # Subcortical Model
    read -p "$(echo -e ${CYAN}Model${RESET}) [default: glm-5.3-flash]: " SUBCORTICAL_MODEL_INPUT
    CONFIG_SUBCORTICAL_MODEL="${SUBCORTICAL_MODEL_INPUT:-glm-5.3-flash}"
fi

# Kagi Search API Key (optional — works with any provider)
echo -e "${BOLD}${BLUE}3. Kagi Search API Key${RESET} ${DIM}(OPTIONAL - kagi.com/settings?p=api)${RESET}"
read -p "$(echo -e ${CYAN}Enter key${RESET}) (or Enter to skip): " KAGI_KEY_INPUT
if [ -z "$KAGI_KEY_INPUT" ]; then
    CONFIG_KAGI_KEY=""
    STATUS_KAGI="${DIM}Skipped${RESET}"
else
    CONFIG_KAGI_KEY="$KAGI_KEY_INPUT"
    STATUS_KAGI="${CHECKMARK} Configured"
fi

# Embeddings (remote lunaroute by default; local model when air-gapped)
echo -e "${BOLD}${BLUE}4. Embeddings${RESET}"
echo -e "${DIM}   A remote OpenAI-compatible POST /v1/embeddings endpoint skips the local${RESET}"
echo -e "${DIM}   PyTorch model. The default is the lunaroute gateway serving emb-nomic-moe${RESET}"
echo -e "${DIM}   (the same key as the chat tier works). The installer probes it for its${RESET}"
echo -e "${DIM}   vector length. The choice is permanent for this install: stored memories${RESET}"
echo -e "${DIM}   are only comparable with the model that made them.${RESET}"
if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
    REMOTE_EMBEDDINGS_DEFAULT="n"
    echo -e "${DIM}   Offline default: no (local model, downloaded during install).${RESET}"
else
    REMOTE_EMBEDDINGS_DEFAULT="y"
fi
read -p "$(echo -e ${CYAN}Use a remote embedding endpoint?${RESET}) (y/n, default=${REMOTE_EMBEDDINGS_DEFAULT}): " REMOTE_EMBEDDINGS_INPUT
REMOTE_EMBEDDINGS_INPUT="${REMOTE_EMBEDDINGS_INPUT:-$REMOTE_EMBEDDINGS_DEFAULT}"
if [[ "$REMOTE_EMBEDDINGS_INPUT" =~ ^[Yy](es)?$ ]]; then
    CONFIG_EMBEDDING_PROVIDER="remote"
    while [ -z "$CONFIG_EMBEDDING_ENDPOINT" ]; do
        read -p "$(echo -e ${CYAN}Endpoint URL${RESET}) [default: https://gw.lunaroute.com/v1/embeddings]: " EMBEDDING_ENDPOINT_INPUT
        CONFIG_EMBEDDING_ENDPOINT="${EMBEDDING_ENDPOINT_INPUT:-https://gw.lunaroute.com/v1/embeddings}"
    done
    while [ -z "$CONFIG_EMBEDDING_MODEL" ]; do
        read -p "$(echo -e ${CYAN}Model${RESET}) [default: emb-nomic-moe]: " EMBEDDING_MODEL_INPUT
        CONFIG_EMBEDDING_MODEL="${EMBEDDING_MODEL_INPUT:-emb-nomic-moe}"
    done
    # Lunaroute is full-service: the chat key already covers embeddings, so no
    # second prompt. Any other endpoint supplies its own key.
    if is_lunaroute_full_service "$CONFIG_EMBEDDING_ENDPOINT"; then
        CONFIG_EMBEDDING_API_KEY="$CONFIG_CHAT_API_KEY"
    else
        read -p "$(echo -e ${CYAN}API key${RESET}) (or Enter for an endpoint that takes none): " CONFIG_EMBEDDING_API_KEY
    fi
    STATUS_EMBEDDINGS="${CHECKMARK} Remote: ${CONFIG_EMBEDDING_MODEL} at ${CONFIG_EMBEDDING_ENDPOINT}"
else
    CONFIG_EMBEDDING_PROVIDER="local"
    STATUS_EMBEDDINGS="${CHECKMARK} Local model"
fi

# Injection Screen (System One decision model)
echo -e "${BOLD}${BLUE}5. Injection Screen${RESET} ${DIM}(screens external content before it reaches MIRA)${RESET}"
echo -e "${DIM}   Fetched pages, email, and files pass through a System One decision${RESET}"
echo -e "${DIM}   model that judges manipulation attempts. Disabled mode still wraps them,${RESET}"
echo -e "${DIM}   never passes them raw.${RESET}"
if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
    # Air-gapped has no djev; a local Kev on the LAN can still opt in.
    SCREEN_DEFAULT_INPUT="n"
    echo -e "${DIM}   Offline default: no.${RESET}"
else
    SCREEN_DEFAULT_INPUT="y"
fi
read -p "$(echo -e ${CYAN}Enable the injection screen?${RESET}) (y/n, default=$SCREEN_DEFAULT_INPUT): " INJECTION_SCREEN_INPUT
INJECTION_SCREEN_INPUT="${INJECTION_SCREEN_INPUT:-$SCREEN_DEFAULT_INPUT}"
if [[ "$INJECTION_SCREEN_INPUT" =~ ^[Yy](es)?$ ]]; then
    CONFIG_INJECTION_SCREEN="yes"
    read -p "$(echo -e ${CYAN}Provider${RESET}) (local=self-hosted no-key endpoint, remote=hosted gateway) [local/remote, default=remote]: " SYSTEMONE_PROVIDER_INPUT
    CONFIG_SYSTEMONE_PROVIDER="${SYSTEMONE_PROVIDER_INPUT:-remote}"
    while [ -z "$CONFIG_SYSTEMONE_ENDPOINT" ]; do
        read -p "$(echo -e ${CYAN}Endpoint URL${RESET}) [default: https://gw.lunaroute.com/v1/systemone]: " SYSTEMONE_ENDPOINT_INPUT
        CONFIG_SYSTEMONE_ENDPOINT="${SYSTEMONE_ENDPOINT_INPUT:-https://gw.lunaroute.com/v1/systemone}"
    done
    while [ -z "$CONFIG_SYSTEMONE_MODEL" ]; do
        read -p "$(echo -e ${CYAN}Model${RESET}) [default: djev]: " SYSTEMONE_MODEL_INPUT
        CONFIG_SYSTEMONE_MODEL="${SYSTEMONE_MODEL_INPUT:-djev}"
    done
    if [ "$CONFIG_SYSTEMONE_PROVIDER" = "remote" ]; then
        # Lunaroute is full-service: the chat key covers System One too.
        if is_lunaroute_full_service "$CONFIG_SYSTEMONE_ENDPOINT"; then
            CONFIG_SYSTEMONE_API_KEY="$CONFIG_CHAT_API_KEY"
        else
            echo -e "${DIM}    The installer probes the endpoint before committing and stores${RESET}"
            echo -e "${DIM}    the key in Vault as systemone_key.${RESET}"
            while [ -z "$CONFIG_SYSTEMONE_API_KEY" ]; do
                read -p "$(echo -e ${CYAN}API key${RESET}): " CONFIG_SYSTEMONE_API_KEY
            done
        fi
        STATUS_SYSTEMONE="${CHECKMARK} On: remote ${CONFIG_SYSTEMONE_MODEL} at ${CONFIG_SYSTEMONE_ENDPOINT}"
    else
        STATUS_SYSTEMONE="${CHECKMARK} On: local ${CONFIG_SYSTEMONE_MODEL} at ${CONFIG_SYSTEMONE_ENDPOINT}"
    fi
else
    CONFIG_INJECTION_SCREEN="no"
    STATUS_SYSTEMONE="${DIM}Disabled${RESET}"
fi

# Database Password (optional - defaults to changethisifdeployingpwd)
echo -e "${BOLD}${BLUE}6. Database Password${RESET} ${DIM}(OPTIONAL - default: changethisifdeployingpwd)${RESET}"
read -p "$(echo -e ${CYAN}Enter password${RESET}) (or Enter for default): " DB_PASSWORD_INPUT
if [ -z "$DB_PASSWORD_INPUT" ]; then
    CONFIG_DB_PASSWORD="changethisifdeployingpwd"
    STATUS_DB_PASSWORD="${DIM}Using default password${RESET}"
else
    CONFIG_DB_PASSWORD="$DB_PASSWORD_INPUT"
    STATUS_DB_PASSWORD="${CHECKMARK} Custom password set"
fi

# Timezone (defaults to this machine's system timezone; the app validates
# the IANA name at boot and fails fast on garbage)
echo -e "${BOLD}${BLUE}7. Timezone${RESET} ${DIM}(IANA name — Enter uses this machine's: ${SYSTEM_TIMEZONE})${RESET}"
read -p "$(echo -e ${CYAN}Timezone${RESET}): " TIMEZONE_INPUT
if [ -z "$TIMEZONE_INPUT" ]; then
    CONFIG_TIMEZONE="$SYSTEM_TIMEZONE"
else
    CONFIG_TIMEZONE="$TIMEZONE_INPUT"
fi
STATUS_TIMEZONE="${CHECKMARK} ${CONFIG_TIMEZONE}"

# Playwright Browser Installation (optional)
echo -e "${BOLD}${BLUE}8. Playwright Browser${RESET} ${DIM}(OPTIONAL - for JS-heavy webpage extraction)${RESET}"
read -p "$(echo -e ${CYAN}Install Playwright?${RESET}) (y/n, default=y): " PLAYWRIGHT_INPUT
# Default to yes if user just presses Enter
if [ -z "$PLAYWRIGHT_INPUT" ]; then
    PLAYWRIGHT_INPUT="y"
fi
if [[ "$PLAYWRIGHT_INPUT" =~ ^[Yy](es)?$ ]]; then
    CONFIG_INSTALL_PLAYWRIGHT="yes"
    STATUS_PLAYWRIGHT="${CHECKMARK} Will be installed"
else
    CONFIG_INSTALL_PLAYWRIGHT="no"
    STATUS_PLAYWRIGHT="${YELLOW}Skipped${RESET}"
fi

# Systemd service option (Linux only)
echo -e "${BOLD}${BLUE}9. Service Supervision${RESET} ${DIM}(OPTIONAL - auto-start on boot; systemd on Linux, launchd on macOS)${RESET}"
if [ "$OS" = "linux" ]; then
    read -p "$(echo -e ${CYAN}Install as systemd service?${RESET}) (y/n): " SYSTEMD_INPUT
    if [[ "$SYSTEMD_INPUT" =~ ^[Yy](es)?$ ]]; then
        CONFIG_INSTALL_SYSTEMD="yes"
        read -p "$(echo -e ${CYAN}Start MIRA now?${RESET}) (y/n): " START_NOW_INPUT
        if [[ "$START_NOW_INPUT" =~ ^[Yy](es)?$ ]]; then
            CONFIG_START_MIRA_NOW="yes"
            STATUS_SYSTEMD="${CHECKMARK} Will be installed and started"
        else
            CONFIG_START_MIRA_NOW="no"
            STATUS_SYSTEMD="${CHECKMARK} Will be installed (not started)"
        fi
    else
        CONFIG_INSTALL_SYSTEMD="no"
        CONFIG_START_MIRA_NOW="no"
        STATUS_SYSTEMD="${DIM}Skipped${RESET}"
    fi
elif [ "$OS" = "macos" ]; then
    CONFIG_INSTALL_SYSTEMD="no"
    read -p "$(echo -e ${CYAN}Start MIRA now and at every login via launchd?${RESET}) (y/n): " START_NOW_INPUT
    if [ -z "$START_NOW_INPUT" ] || [[ "$START_NOW_INPUT" =~ ^[Yy](es)?$ ]]; then
        CONFIG_START_MIRA_NOW="yes"
        STATUS_SYSTEMD="${CHECKMARK} LaunchAgent will be installed and started"
    else
        CONFIG_START_MIRA_NOW="no"
        STATUS_SYSTEMD="${CHECKMARK} LaunchAgent will be installed (starts at login)"
    fi
fi

fi

# Timezone resolution: an empty CONFIG_TIMEZONE (yml "" or interview Enter)
# means "this host's system timezone" — resolve to a concrete IANA name so
# the systemd unit always carries an explicit value.
if [ -z "$CONFIG_TIMEZONE" ]; then
    CONFIG_TIMEZONE="$SYSTEM_TIMEZONE"
fi
[ -n "$STATUS_TIMEZONE" ] || STATUS_TIMEZONE="${CHECKMARK} ${CONFIG_TIMEZONE}"

echo ""
echo -e "${BOLD}Configuration Summary:${RESET}"
if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
    if [ "$CONFIG_LOCAL_MODEL_CHOICE" = "auto" ]; then
        echo -e "  LLM Provider:    ${CYAN}Local llama-server${RESET}"
        echo -e "  Main Model:      ${CYAN}${CONFIG_LLAMA_MAIN_MODEL:-configure per docs/OFFLINE_MODELS.md}${RESET}"
        echo -e "  Small Model:     ${CYAN}${CONFIG_LLAMA_SMALL_MODEL:-configure per docs/OFFLINE_MODELS.md}${RESET}"
    else
        echo -e "  LLM Provider:    ${CYAN}Local llama-server (custom)${RESET}"
        echo -e "  Model:           ${CYAN}${CONFIG_CUSTOM_GGUF:-TBD}${RESET}"
        echo -e "  Main Model:      ${CYAN}${CONFIG_LLAMA_MAIN_MODEL:-configure per docs/OFFLINE_MODELS.md}${RESET}"
        echo -e "  Small Model:     ${CYAN}${CONFIG_LLAMA_SMALL_MODEL:-configure per docs/OFFLINE_MODELS.md}${RESET}"
    fi
else
    echo -e "  Chat Provider:   ${STATUS_CHAT_PROVIDER}"
    echo -e "  Chat Model:      ${CYAN}${CONFIG_CHAT_MODEL}${RESET}"
    echo -e "  Chat Key:        ${STATUS_CHAT_KEY}"
    echo -e "  Subcortical:     ${STATUS_SUBCORTICAL}"
    echo -e "  Subcortical Key: ${STATUS_SUBCORTICAL_KEY}"
    echo -e "  Subcortical Mdl: ${CYAN}${CONFIG_SUBCORTICAL_MODEL}${RESET}"
fi
echo -e "  Kagi:            ${STATUS_KAGI}"
echo -e "  Embeddings:      ${STATUS_EMBEDDINGS}"
echo -e "  Injection Scr:   ${STATUS_SYSTEMONE}"
echo -e "  DB Password:     ${STATUS_DB_PASSWORD}"
echo -e "  Timezone:        ${STATUS_TIMEZONE}"
echo -e "  Playwright:      ${STATUS_PLAYWRIGHT}"
echo -e "  Systemd Service: ${STATUS_SYSTEMD}"
echo ""
