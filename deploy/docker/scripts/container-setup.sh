#!/bin/bash
# MIRA Container Interactive Setup
# Adapted from deploy/config.sh for Docker container context
#
# This script is sourced (not executed) by init-mira.sh when running
# interactively with a TTY attached.
#
# Differences from host deploy/config.sh:
# - No disk space check (container handles this)
# - No port availability check (container manages ports)
# - No systemd/service management options
# - No Playwright install prompt (pre-installed in image)
# - No existing installation check (clean container)

# Source helper libraries
source /opt/mira/app/deploy/lib/output.sh
source /opt/mira/app/deploy/lib/services.sh

LOUD_MODE=false

# Initialize configuration state
CONFIG_PROVIDER_KEY=""
CONFIG_SYSTEMONE_API_KEY=""
CONFIG_KAGI_KEY=""
CONFIG_DB_PASSWORD=""
CONFIG_OFFLINE_MODE=""
CONFIG_PROVIDER_NAME=""
CONFIG_PROVIDER_ENDPOINT=""
CONFIG_PROVIDER_KEY_PREFIX=""
CONFIG_PROVIDER_MODEL=""

# Container-specific settings
OS="linux"
DISTRO="debian"

clear
echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════╗"
echo "║   MIRA First-Boot Configuration        ║"
echo "╚════════════════════════════════════════╝"
echo -e "${RESET}"
echo ""

print_header "API Key Configuration"

# Note: Offline mode not supported in standard Docker image
# (would require llama.cpp bundled or external llama-server)
CONFIG_OFFLINE_MODE="no"

# (hn32) No Anthropic key step: the all-five-routes UPDATE binds every route
# to subcortical_key, so an Anthropic key would sit in Vault with no reader.

# OpenAI-compatible Provider Selection
echo -e "${BOLD}${BLUE}1. Fast Inference Provider${RESET} ${DIM}(OpenAI-compatible)${RESET}"
echo ""
echo -e "${DIM}   Select your preferred provider:${RESET}"
echo "     1. Lunaroute (default — OpenAI-compatible gateway)"
echo "     2. Groq"
echo "     3. OpenRouter"
echo "     4. Together AI"
echo "     5. Fireworks AI"
echo "     6. Cerebras"
echo "     7. SambaNova"
echo "     8. Other (custom endpoint)"
read -p "$(echo -e ${CYAN}Select provider${RESET}) [1-8, default=1]: " PROVIDER_CHOICE

case "${PROVIDER_CHOICE:-1}" in
    1)
        CONFIG_PROVIDER_NAME="Lunaroute"
        CONFIG_PROVIDER_ENDPOINT="https://gw.lunaroute.com/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX=""
        ;;
    2)
        CONFIG_PROVIDER_NAME="Groq"
        CONFIG_PROVIDER_ENDPOINT="https://api.groq.com/openai/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX="gsk_"
        ;;
    3)
        CONFIG_PROVIDER_NAME="OpenRouter"
        CONFIG_PROVIDER_ENDPOINT="https://openrouter.ai/api/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX="sk-or-"
        ;;
    4)
        CONFIG_PROVIDER_NAME="Together AI"
        CONFIG_PROVIDER_ENDPOINT="https://api.together.xyz/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX=""
        ;;
    5)
        CONFIG_PROVIDER_NAME="Fireworks AI"
        CONFIG_PROVIDER_ENDPOINT="https://api.fireworks.ai/inference/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX=""
        ;;
    6)
        CONFIG_PROVIDER_NAME="Cerebras"
        CONFIG_PROVIDER_ENDPOINT="https://api.cerebras.ai/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX=""
        ;;
    7)
        CONFIG_PROVIDER_NAME="SambaNova"
        CONFIG_PROVIDER_ENDPOINT="https://api.sambanova.ai/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX=""
        ;;
    8)
        CONFIG_PROVIDER_NAME="Custom"
        read -p "$(echo -e ${CYAN}Enter custom endpoint URL${RESET}): " CONFIG_PROVIDER_ENDPOINT
        CONFIG_PROVIDER_KEY_PREFIX=""
        ;;
    *)
        CONFIG_PROVIDER_NAME="Lunaroute"
        CONFIG_PROVIDER_ENDPOINT="https://gw.lunaroute.com/v1/chat/completions"
        CONFIG_PROVIDER_KEY_PREFIX=""
        ;;
esac

print_success "Provider: $CONFIG_PROVIDER_NAME"

# Providers whose models list needs the API key (Lunaroute, Groq) confirm their
# model AFTER key entry (step 2b) via prefill_provider_model; skip the pre-key
# prompt for them. Everyone else prompts here with a per-provider default.
if [ "$CONFIG_PROVIDER_NAME" != "Groq" ] && [ "$CONFIG_PROVIDER_NAME" != "Lunaroute" ]; then
    echo ""
    print_info "MIRA needs a model name compatible with ${CONFIG_PROVIDER_NAME}."
    case "$CONFIG_PROVIDER_NAME" in
        "OpenRouter")
            print_info "Example: meta-llama/llama-3.3-70b-instruct:free"
            # Live check against the public models list: prefill the suggestion only if OpenRouter actually serves it.
            prefill_provider_model "meta-llama/llama-3.3-70b-instruct:free"
            ;;
        "Together AI")
            print_info "Example: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
            CONFIG_PROVIDER_MODEL="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
            ;;
        "Fireworks AI")
            print_info "Example: accounts/fireworks/models/llama-v3p1-70b-instruct"
            CONFIG_PROVIDER_MODEL="accounts/fireworks/models/llama-v3p1-70b-instruct"
            ;;
        "Cerebras")
            print_info "Example: llama-3.3-70b"
            CONFIG_PROVIDER_MODEL="llama-3.3-70b"
            ;;
        "SambaNova")
            print_info "Example: Meta-Llama-3.1-70B-Instruct"
            CONFIG_PROVIDER_MODEL="Meta-Llama-3.1-70B-Instruct"
            ;;
    esac
    if [ -n "$CONFIG_PROVIDER_MODEL" ]; then
        read -p "$(echo -e ${CYAN}Model name${RESET}) [default: ${CONFIG_PROVIDER_MODEL}]: " MODEL_INPUT
        CONFIG_PROVIDER_MODEL="${MODEL_INPUT:-$CONFIG_PROVIDER_MODEL}"
    else
        print_info "Pick a model from your provider's website and enter its exact name."
        read -p "$(echo -e ${CYAN}Model name${RESET}): " CONFIG_PROVIDER_MODEL
    fi
fi

# Provider API Key (required)
echo -e "${BOLD}${BLUE}1b. ${CONFIG_PROVIDER_NAME} API Key${RESET} ${DIM}(REQUIRED)${RESET}"
while true; do
    read -p "$(echo -e ${CYAN}Enter key${RESET}): " PROVIDER_KEY_INPUT
    if [ -z "$PROVIDER_KEY_INPUT" ]; then
        print_warning "Provider API key is required for internal LLM operations."
        continue
    fi
    # Validate key prefix if provider has one
    if [ -n "$CONFIG_PROVIDER_KEY_PREFIX" ]; then
        if [[ $PROVIDER_KEY_INPUT =~ ^${CONFIG_PROVIDER_KEY_PREFIX} ]]; then
            CONFIG_PROVIDER_KEY="$PROVIDER_KEY_INPUT"
            print_success "Provider key configured"
            break
        else
            print_warning "This doesn't look like a valid ${CONFIG_PROVIDER_NAME} API key"
            read -p "$(echo -e ${YELLOW}Continue anyway?${RESET}) (y=yes, t=try again): " CONFIRM
            if [[ "$CONFIRM" =~ ^[Yy](es)?$ ]]; then
                CONFIG_PROVIDER_KEY="$PROVIDER_KEY_INPUT"
                print_success "Provider key configured (unvalidated)"
                break
            fi
        fi
    else
        CONFIG_PROVIDER_KEY="$PROVIDER_KEY_INPUT"
        print_success "Provider key configured"
        break
    fi
done

# Lunaroute/Groq prompts for no model above: the key (collected in 2b) is
# required for their models-list check, so prefill-or-leave-unset happens here.
if [ "$CONFIG_PROVIDER_NAME" = "Groq" ]; then
    prefill_provider_model "qwen/qwen3.6-27b"
elif [ "$CONFIG_PROVIDER_NAME" = "Lunaroute" ]; then
    prefill_provider_model "glm-5.3-flash"
fi

# Kagi API Key (optional)
echo -e "${BOLD}${BLUE}2. Kagi Search API Key${RESET} ${DIM}(OPTIONAL - kagi.com/settings?p=api)${RESET}"
read -p "$(echo -e ${CYAN}Enter key${RESET}) (or Enter to skip): " KAGI_KEY_INPUT
if [ -z "$KAGI_KEY_INPUT" ]; then
    CONFIG_KAGI_KEY=""
    print_info "Kagi search skipped (web search will be limited)"
else
    CONFIG_KAGI_KEY="$KAGI_KEY_INPUT"
    print_success "Kagi key configured"
fi

# Database Password (optional)
echo -e "${BOLD}${BLUE}3. Database Password${RESET} ${DIM}(OPTIONAL - for internal PostgreSQL)${RESET}"
read -p "$(echo -e ${CYAN}Enter password${RESET}) (or Enter for default): " DB_PASSWORD_INPUT
if [ -z "$DB_PASSWORD_INPUT" ]; then
    CONFIG_DB_PASSWORD="changethisifdeployingpwd"
    print_info "Using default database password"
else
    CONFIG_DB_PASSWORD="$DB_PASSWORD_INPUT"
    print_success "Custom database password set"
fi

# Injection Screen System One Key (optional)
# (40dz) The app's injection screen (on for a token-bearing install) reads
# its bearer token from Vault as systemone_key; without this step no shipped
# container path ever wrote it, and every gated sidebar dispatch died with a
# KeyError at first use. Enter skips: the screen stays off (the launcher's
# default) and external content is structurally wrapped only — fail-closed.
echo -e "${BOLD}${BLUE}4. Injection Screen System One Key${RESET} ${DIM}(OPTIONAL - hosted /v1/systemone gateway token)${RESET}"
read -p "$(echo -e ${CYAN}Enter key${RESET}) (or Enter to keep the screen off): " SYSTEMONE_KEY_INPUT
if [ -z "$SYSTEMONE_KEY_INPUT" ]; then
    CONFIG_SYSTEMONE_API_KEY=""
    print_info "Injection screen stays off; external content is structurally wrapped only"
else
    CONFIG_SYSTEMONE_API_KEY="$SYSTEMONE_KEY_INPUT"
    print_success "Injection screen enabled: token stored in Vault as systemone_key"
fi

# Configuration Summary
echo ""
print_header "Configuration Summary"
echo -e "  Provider:        ${CYAN}$CONFIG_PROVIDER_NAME${RESET}"
echo -e "  Provider Key:    ${GREEN}****${CONFIG_PROVIDER_KEY: -4}${RESET}"
if [ -n "$CONFIG_PROVIDER_MODEL" ]; then
    echo -e "  Provider Model:  ${CYAN}$CONFIG_PROVIDER_MODEL${RESET}"
fi
if [ -n "$CONFIG_KAGI_KEY" ]; then
    echo -e "  Kagi Key:        ${GREEN}Configured${RESET}"
else
    echo -e "  Kagi Key:        ${DIM}Skipped${RESET}"
fi
if [ -n "$CONFIG_SYSTEMONE_API_KEY" ]; then
    echo -e "  Inj. Screen:     ${GREEN}On (systemone_key in Vault)${RESET}"
else
    echo -e "  Inj. Screen:     ${DIM}Off${RESET}"
fi
echo ""

read -p "$(echo -e ${CYAN}Proceed with this configuration?${RESET}) (y/n): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy](es)?$ ]]; then
    print_error "Configuration cancelled. Restarting setup..."
    # This script is sourced into the PID-1 shell (init-mira.sh); exec would
    # replace the entrypoint and end the container before provisioning. Call
    # a fresh pass normally, then return from this pass so control unwinds to
    # init-mira.sh with the accepted CONFIG_* exports intact.
    source /opt/mira/container-setup.sh
    return
fi

# Export configuration for init-mira.sh to use
export CONFIG_PROVIDER_KEY
export CONFIG_SYSTEMONE_API_KEY
export CONFIG_PROVIDER_NAME
export CONFIG_PROVIDER_ENDPOINT
export CONFIG_PROVIDER_MODEL
export CONFIG_KAGI_KEY
export CONFIG_DB_PASSWORD
export CONFIG_OFFLINE_MODE

print_success "Configuration complete. Proceeding with initialization..."
