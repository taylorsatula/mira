# deploy/finalize.sh
# Systemd service, cleanup, and success message
# Source this file - do not execute directly
#
# Requires: lib/output.sh and lib/services.sh sourced first
# Requires: OS, DISTRO, MIRA_USER, MIRA_GROUP, CONFIG_*, STATUS_*, LOUD_MODE variables set

# Validate required variables
: "${OS:?Error: OS must be set}"
: "${MIRA_USER:?Error: MIRA_USER must be set}"

# One mechanism for the install-time environment on every start path: the
# systemd unit reads this file via EnvironmentFile, the no-systemd launcher
# (Step 15b) and the container's s6 mira/run source it when present. Non-secret
# values only — the bearer token lives in Vault, never in this file.
echo -ne "${DIM}${ARROW}${RESET} Writing /opt/mira/systemone.env... "
SYSTEMONE_ENV_FILE="/opt/mira/systemone.env"
{
    # The install's timezone choice, explicit on every start path: the app maps
    # MIRA_TIMEZONE → SystemConfig.timezone (config/config_manager.py). It was
    # collected by config.sh and dropped on the floor here — without this line
    # the app silently falls back to host detection and the interview/YAML
    # answer never reaches the installed instance.
    if [ -n "${CONFIG_TIMEZONE:-}" ]; then
        echo "MIRA_TIMEZONE=${CONFIG_TIMEZONE}"
    fi
    # The install's user-name choice, same mechanism: the app maps
    # MIRA_LOCAL_SESSION_FIRST_NAME → SystemConfig.local_session_first_name,
    # which create_local_session uses for the local account's first_name and
    # orientation content. config.sh resolves "" to "Friend" so this always
    # carries a concrete name; an upgraded install that predates the key keeps
    # the app-side "Friend" default (the env var is simply absent).
    if [ -n "${CONFIG_USER_NAME:-}" ]; then
        echo "MIRA_LOCAL_SESSION_FIRST_NAME=${CONFIG_USER_NAME}"
    fi
    if [ "${CONFIG_INJECTION_SCREEN}" = "yes" ]; then
        echo "MIRA_INJECTION_SCREEN_ENABLED=1"
        echo "MIRA_SYSTEMONE_PROVIDER=${CONFIG_SYSTEMONE_PROVIDER}"
        echo "MIRA_SYSTEMONE_ENDPOINT=${CONFIG_SYSTEMONE_ENDPOINT}"
        echo "MIRA_SYSTEMONE_MODEL=${CONFIG_SYSTEMONE_MODEL}"
    else
        echo "MIRA_INJECTION_SCREEN_ENABLED=0"
    fi
} > "$SYSTEMONE_ENV_FILE"
chmod 600 "$SYSTEMONE_ENV_FILE"
chown "$MIRA_USER:$MIRA_GROUP" "$SYSTEMONE_ENV_FILE" 2>/dev/null || chown "$MIRA_USER" "$SYSTEMONE_ENV_FILE"
echo -e "${CHECKMARK}"

# Systemd service installation (Linux only, if user opted in)
if [ "${CONFIG_INSTALL_SYSTEMD}" = "yes" ] && [ "$OS" = "linux" ]; then
    print_header "Step 15: Systemd Service Configuration"

    # Extract Vault credentials from files
    echo -ne "${DIM}${ARROW}${RESET} Reading Vault credentials... "
    VAULT_ROLE_ID=$(cat /opt/vault/role-id.txt)
    VAULT_SECRET_ID=$(cat /opt/vault/secret-id.txt)

    if [ -z "$VAULT_ROLE_ID" ] || [ -z "$VAULT_SECRET_ID" ]; then
        echo -e "${ERROR}"
        print_error "Failed to read Vault credentials from /opt/vault/"
        print_info "Skipping systemd service creation"
        CONFIG_INSTALL_SYSTEMD="failed"
        STATUS_MIRA_SERVICE="${ERROR} Configuration failed"
    else
        echo -e "${CHECKMARK}"

        # Create systemd service file
        echo -ne "${DIM}${ARROW}${RESET} Creating systemd service file... "

        # Resolve both dependency unit names by what this host actually ships
        # (lib/services.sh resolvers — one sanctioned mechanism, shared with
        # config.sh's port-stop path). A wrong name in Requires= would make
        # systemd refuse to start mira.service.
        PG_SERVICE=$(resolve_pg_unit || echo "postgresql.service")
        VALKEY_SERVICE=$(resolve_valkey_unit || echo "valkey.service")

        sudo tee /etc/systemd/system/mira.service > /dev/null <<EOF
[Unit]
Description=MIRA - AI Assistant with Persistent Memory
Documentation=https://github.com/taylorsatula/mira-OSS
Requires=vault.service ${PG_SERVICE} ${VALKEY_SERVICE}
After=vault.service ${PG_SERVICE} ${VALKEY_SERVICE} vault-unseal.service
ConditionPathExists=/opt/mira/app/main.py

[Service]
Type=simple
User=$MIRA_USER
Group=$MIRA_GROUP
WorkingDirectory=/opt/mira/app
EnvironmentFile=/opt/mira/systemone.env
Environment="VAULT_ADDR=http://127.0.0.1:8200"
Environment="VAULT_ROLE_ID=$VAULT_ROLE_ID"
Environment="VAULT_SECRET_ID=$VAULT_SECRET_ID"
Environment="MIRA_LOG_DIR=/opt/mira/logs"
ExecStart=/opt/mira/app/venv/bin/python3 /opt/mira/app/main.py
Restart=on-failure
RestartSec=10
TimeoutStartSec=60
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mira

[Install]
WantedBy=multi-user.target
EOF
        echo -e "${CHECKMARK}"

        # Reload systemd and enable service
        run_quiet sudo systemctl daemon-reload

        run_with_status "Enabling MIRA service for auto-start on boot" \
            sudo systemctl enable mira.service

        print_success "Systemd service configured"
        print_info "Service will auto-start on system boot"

        # Start service if user chose to during configuration
        if [ "${CONFIG_START_MIRA_NOW}" = "yes" ]; then
            echo ""
            start_service mira.service systemctl

            # Give service a moment to start
            sleep 2

            # Check if service started successfully
            if sudo systemctl is-active --quiet mira.service; then
                print_success "MIRA service is running"
                print_info "View logs: journalctl -u mira -f"
                STATUS_MIRA_SERVICE="${CHECKMARK} Running"
            else
                print_warning "MIRA service may have failed to start"
                print_info "Check status: systemctl status mira"
                print_info "View logs: journalctl -u mira -n 50"
                STATUS_MIRA_SERVICE="${ERROR} Start failed"
            fi
        else
            print_info "To start later: sudo systemctl start mira"
            print_info "To view logs: journalctl -u mira -f"
            STATUS_MIRA_SERVICE="${DIM}Not started${RESET}"
        fi
    fi
elif [ "$OS" = "macos" ]; then
    # systemd does not exist on macOS, so CONFIG_INSTALL_SYSTEMD is "no" by
    # construction here — reporting that as "user opted out" was false: the
    # operator opted in to launchd, which IS the macOS supervisor (Step 15b2).
    print_header "Step 15: Service Supervision"
    print_info "Supervision is the LaunchAgent installed in the next step"
elif [ "${CONFIG_INSTALL_SYSTEMD}" = "no" ]; then
    print_header "Step 15: Systemd Service Configuration"
    print_info "Skipping systemd service installation (user opted out)"
fi

# Write a launcher that exports Vault env vars before starting MIRA.
# On Linux with systemd these vars are baked into the unit; macOS has no
# equivalent, and a Linux user who declined (or failed) systemd gets no unit
# at all — so both need the launcher. The server itself reads Vault at
# startup (POST gate, preload_secrets) and fails fast without these env vars.
if [ "$OS" = "macos" ] || { [ "$OS" = "linux" ] && [ "${CONFIG_INSTALL_SYSTEMD}" != "yes" ]; }; then
    print_header "Step 15b: MIRA Launcher Script"

    RUN_SH="/opt/mira/app/run.sh"
    echo -ne "${DIM}${ARROW}${RESET} Writing $RUN_SH... "
    # The PATH prefix covers launchd's minimal environment (brew tools, and
    # anything bash_tool shells out to); the bounded seal-status wait covers
    # launchd's lack of agent ordering — the POST gate treats a sealed or
    # unreachable Vault as infrastructure failure, so the launcher waits for
    # Vault to be up AND unsealed before starting the server.
    cat > "$RUN_SH" <<'LAUNCHER'
#!/bin/bash
# MIRA launcher — exports Vault env vars and starts the server.
set -e
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
cd "$(dirname "$0")"
export VAULT_ADDR=http://127.0.0.1:8200
for i in $(seq 1 120); do
    S=$(curl -sf http://127.0.0.1:8200/v1/sys/seal-status 2>/dev/null || true)
    case "$S" in *'"sealed":false'*) break;; esac
    sleep 1
done
export VAULT_ROLE_ID=$(cat /opt/vault/role-id.txt)
export VAULT_SECRET_ID=$(cat /opt/vault/secret-id.txt)
export MIRA_LOG_DIR=/opt/mira/logs
# set -a: sourced assignments must be EXPORTED to reach the server process —
# a plain `.` sets shell variables only, so python never saw
# MIRA_INJECTION_SCREEN_ENABLED=0 and the boot gate parked (observed live on
# macOS; systemd's EnvironmentFile= exports natively, run.sh must too).
if [ -f /opt/mira/systemone.env ]; then
    set -a
    . /opt/mira/systemone.env
    set +a
fi
exec venv/bin/python3 main.py "$@"
LAUNCHER
    chmod +x "$RUN_SH"
    echo -e "${CHECKMARK}"
    print_info "Start MIRA with: $RUN_SH"
fi

# macOS supervision: a per-user LaunchAgent runs run.sh at login and restarts
# it after a failed exit — the platform equivalent of mira.service
# (Restart=on-failure). Written even when start_mira_now is no: agents in
# ~/Library/LaunchAgents load at the next login.
if [ "$OS" = "macos" ]; then
    print_header "Step 15b2: MIRA LaunchAgent"
    echo -ne "${DIM}${ARROW}${RESET} Writing com.mira.app.plist... "
    mkdir -p "$HOME/Library/LaunchAgents" /opt/mira/logs
    cat > "$HOME/Library/LaunchAgents/com.mira.app.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mira.app</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/mira/app/run.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/opt/mira/app</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>/opt/mira/logs/mira-launchd.log</string>
    <key>StandardErrorPath</key>
    <string>/opt/mira/logs/mira-launchd.log</string>
</dict>
</plist>
EOF
    echo -e "${CHECKMARK}"
    launchctl bootout "gui/$(id -u)/com.mira.app" 2>/dev/null || true
    if [ "${CONFIG_START_MIRA_NOW}" = "yes" ]; then
        echo -ne "${DIM}${ARROW}${RESET} Starting MIRA (launchd)... "
        # One sanctioned launchd reload path (lib/services.sh) — bootstrap
        # during the bootout teardown above fails with "Bootstrap failed: 5".
        if launchd_reload_agent "$HOME/Library/LaunchAgents/com.mira.app.plist"; then
            echo -e "${CHECKMARK}"
            # The POST gate runs real infrastructure probes before the server
            # binds; on a first boot with a local embedding model this can
            # take minutes — poll generously before calling it failed.
            echo -ne "${DIM}${ARROW}${RESET} Waiting for MIRA to become healthy... "
            MIRA_HEALTHY=0
            for i in $(seq 1 120); do
                if curl -sf http://127.0.0.1:1993/v0/api/health 2>/dev/null | grep -q '"status":"healthy"'; then
                    MIRA_HEALTHY=1
                    break
                fi
                sleep 5
            done
            if [ "$MIRA_HEALTHY" = 1 ]; then
                echo -e "${CHECKMARK} ${DIM}(healthy after ~$((i * 5))s)${RESET}"
                STATUS_MIRA_SERVICE="${CHECKMARK} Running"
                print_info "View logs: tail -f /opt/mira/logs/mira-launchd.log"
            else
                echo -e "${ERROR}"
                print_warning "MIRA did not report healthy within 600 s"
                print_info "Check status: launchctl print gui/$(id -u)/com.mira.app"
                print_info "View logs: tail -100 /opt/mira/logs/mira-launchd.log"
                STATUS_MIRA_SERVICE="${ERROR} Start failed"
            fi
        else
            echo -e "${ERROR}"
            print_error "Failed to load the com.mira.app launchd agent"
            STATUS_MIRA_SERVICE="${ERROR} Start failed"
        fi
    else
        print_info "Agent written but not started (start_mira_now: no) — it loads at the next login."
        print_info "Start now: launchctl bootstrap gui/$(id -u) $HOME/Library/LaunchAgents/com.mira.app.plist"
        STATUS_MIRA_SERVICE="${DIM}Not started${RESET}"
    fi
fi

# The `mira` command: the deployed TUI chat client on its own venv
# (python.sh Step 5b), runnable from any directory — PYTHONPATH puts the
# tui/ package inside /opt/mira/app on sys.path so the user's cwd
# does not matter.
print_header "Step 15d: Terminal Chat Command (/usr/local/bin/mira)"
MIRA_BIN=/usr/local/bin/mira
echo -ne "${DIM}${ARROW}${RESET} Writing $MIRA_BIN... "
sudo tee "$MIRA_BIN" >/dev/null <<'MIRA_CLI'
#!/bin/sh
# MIRA terminal chat client — deployed by finalize.sh. Talk to this
# instance from any directory: `mira` (first run: mira --login --base-url
# http://localhost:1993 --chat).
export PYTHONPATH="/opt/mira/app${PYTHONPATH:+:$PYTHONPATH}"
# -P: `python3 -m` prepends the CURRENT DIRECTORY to sys.path ahead of
# PYTHONPATH, so `mira` run from inside a mira checkout silently resolved
# tui/ and utils/ (and read the checkout's VERSION) instead of the installed
# tree. -P suppresses that prepend; the install tree wins regardless of cwd.
exec /opt/mira/tui-venv/bin/python3 -P -m tui "$@"
MIRA_CLI
sudo chmod 755 "$MIRA_BIN"
echo -e "${CHECKMARK}"

# Write one-time credential dump to user's home directory
print_header "Step 15c: Credential Dump"

ROOT_TOKEN=$(vault_extract_credential "Initial Root Token")
UNSEAL_KEY=$(vault_extract_credential "Unseal Key 1")

CRED_FILE="$HOME/MIRA_credentials.txt"

_cred_write_failed=false

if ! cat > "$CRED_FILE" <<CREDS
================================================================================
  ⚠️  CRITICAL — READ BEFORE PROCEEDING  ⚠️
================================================================================

This file contains ALL credentials for your MIRA installation.

THIS IS THE ONLY TIME these credentials will be written to disk in plain text.

ACTION REQUIRED:
  1. Copy this file to a secure location (password manager, encrypted drive)
  2. DELETE this file immediately after saving
  3. If you lose these credentials, you can recover by logging into Vault:
     export VAULT_ADDR='http://127.0.0.1:8200'
     vault login <root_token>
     vault kv get secret/mira/api_keys

================================================================================

Vault Root Token (hvs_...):
  ${ROOT_TOKEN}

Vault Unseal Key:
  ${UNSEAL_KEY}

================================================================================

API Keys stored in Vault at secret/mira/api_keys:

  anthropic_key:        ${CONFIG_ANTHROPIC_KEY}
CREDS
then
    _cred_write_failed=true
fi

if [ "$_cred_write_failed" = false ]; then
    if [ "$CONFIG_OFFLINE_MODE" != "yes" ]; then
        cat >> "$CRED_FILE" <<CREDS
  provider_key:         ${CONFIG_CHAT_API_KEY:-N/A}
  subcortical_key:      ${CONFIG_SUBCORTICAL_API_KEY}
  kagi_api_key:         ${CONFIG_KAGI_KEY:-N/A}
CREDS
        if [ "$CONFIG_INJECTION_SCREEN" = "yes" ] && [ "$CONFIG_SYSTEMONE_PROVIDER" = "remote" ]; then
            cat >> "$CRED_FILE" <<CREDS
  systemone_key:        ${CONFIG_SYSTEMONE_API_KEY}
CREDS
        fi
    else
        cat >> "$CRED_FILE" <<CREDS
  (Offline/local mode — no external API keys configured)
CREDS
    fi

    cat >> "$CRED_FILE" <<'CREDS'

================================================================================
  ⚠️  REMEMBER TO DELETE THIS FILE AFTER SAVING SECURELY  ⚠️
================================================================================
CREDS

    chmod 600 "$CRED_FILE"

    print_warning "Credential dump written to: $CRED_FILE"
    echo ""
    print_info "⚠️  This is the ONLY time these credentials are written to disk."
    print_info "Save this file securely, then DELETE it."
    print_info "To retrieve later, use Vault CLI with your root token."
else
    print_error "Failed to write credential dump to $CRED_FILE"
    print_info "Credentials are still accessible via Vault CLI with your root token."
fi
echo ""

print_header "Step 16: Cleanup"

if [ "$LOUD_MODE" = true ]; then
    print_step "Flushing pip cache..."
    venv/bin/pip3 cache purge 2>/dev/null || print_info "pip cache purge skipped (cache may be empty)"
else
    run_with_status "Flushing pip cache" \
        venv/bin/pip3 cache purge 2>/dev/null || true
fi

# Remove temporary files silently
run_quiet rm -f /tmp/mira-policy.hcl

if [ "$OS" = "linux" ]; then
    run_quiet rm -f /tmp/vault_1.18.3_linux_*.zip
    run_quiet rm -f /tmp/vault
fi

print_success "Cleanup complete"

echo ""
echo ""
echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════╗"
echo "║       Deployment Complete! 🎉          ║"
echo "╚════════════════════════════════════════╝"
echo -e "${RESET}"
echo ""

print_success "MIRA installed to: /opt/mira/app"
print_success "All temporary files cleaned up"

echo ""
echo -e "${BOLD}${BLUE}Important Files${RESET} ${DIM}(/opt/vault/)${RESET}"
print_info "init-keys.txt (Vault unseal key and root token)"
print_info "role-id.txt (AppRole role ID)"
print_info "secret-id.txt (AppRole secret ID)"
echo ""
if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
    echo -e "${BOLD}${BLUE}LLM Provider${RESET}"
    echo -e "  Provider:     ${CYAN}Local llama-server${RESET}"
    if [ "$CONFIG_LOCAL_MODEL_CHOICE" = "auto" ]; then
        echo -e "  Main Model:   ${CYAN}${CONFIG_LLAMA_MAIN_MODEL:-configure per docs/OFFLINE_MODELS.md}${RESET} ${DIM}(port 3090)${RESET}"
        echo -e "  Small Model:  ${CYAN}${CONFIG_LLAMA_SMALL_MODEL:-configure per docs/OFFLINE_MODELS.md}${RESET} ${DIM}(port 3092)${RESET}"
        echo -e "  VRAM Target:  ${DIM}~48GB across two cards${RESET}"
        echo ""
        print_info "Before first MIRA startup, download models & start servers:"
        print_info "  docs/OFFLINE_MODELS.md"
        print_info ""
        print_info "Models stored at: /opt/mira/models/"
        print_info "Server logs at:   /opt/mira/logs/llama-{main,small}.log"
        print_info "Health check:     curl http://localhost:3090/health"
    else
        echo -e "  Mode:         ${CYAN}Custom (bring your own GGUF)${RESET}"
        echo -e "  Main Model:   ${CYAN}${CONFIG_LLAMA_MAIN_MODEL:-configure per docs/OFFLINE_MODELS.md}${RESET} ${DIM}(port 3090)${RESET}"
        echo -e "  Small Model:  ${CYAN}${CONFIG_LLAMA_SMALL_MODEL:-configure per docs/OFFLINE_MODELS.md}${RESET} ${DIM}(port 3092)${RESET}"
        echo ""
        print_info "Place your GGUF files in /opt/mira/models/ and configure llama-server manually."
    fi
    echo -e "  Inj. Screen:  ${STATUS_SYSTEMONE}"
else
    echo -e "${BOLD}${BLUE}Provider Configuration${RESET}"
    echo -e "  Chat Provider:   ${STATUS_CHAT_PROVIDER}"
    echo -e "  Chat Model:      ${CYAN}${CONFIG_CHAT_MODEL}${RESET}"
    echo -e "  Chat Key:        ${STATUS_CHAT_KEY}"
    echo -e "  Subcortical:     ${STATUS_SUBCORTICAL}"
    echo -e "  Subcortical Mdl: ${CYAN}${CONFIG_SUBCORTICAL_MODEL}${RESET}"
    echo -e "  Subcortical Key: ${STATUS_SUBCORTICAL_KEY}"
    echo -e "  Kagi:            ${STATUS_KAGI}"
    echo -e "  Embeddings:      ${STATUS_EMBEDDINGS}"
    echo -e "  Injection Scr:   ${STATUS_SYSTEMONE}"

    if [ "${CONFIG_CHAT_API_KEY}" = "PLACEHOLDER_SET_THIS_LATER" ] || [ "${CONFIG_CHAT_API_KEY}" = "PLACEHOLDER_NOT_CONFIGURED" ] || [ "${CONFIG_SUBCORTICAL_API_KEY}" = "PLACEHOLDER_SET_THIS_LATER" ]; then
        echo ""
        print_warning "Required API keys not configured!"
        print_info "MIRA will not work until you set the missing API keys."
        print_info "To configure later, use Vault CLI:"
        echo -e "${DIM}    export VAULT_ADDR='http://127.0.0.1:8200'${RESET}"
        echo -e "${DIM}    vault login <root-token-from-init-keys.txt>${RESET}"
        echo -e "${DIM}    vault kv put secret/mira/api_keys \\${RESET}"
        echo -e "${DIM}      anthropic_key=\"sk-ant-your-key\" \\${RESET}"
        echo -e "${DIM}      subcortical_key=\"your-lunaroute-key\" \\${RESET}"
        echo -e "${DIM}      provider_key=\"your-chat-provider-key\" \\${RESET}"
        echo -e "${DIM}      kagi_api_key=\"your-kagi-key\"${RESET}"
    fi
fi

echo ""
echo -e "${BOLD}${BLUE}Services Running${RESET}"
if [ "$OS" = "linux" ]; then
    print_info "Valkey: localhost:6379"
    print_info "Vault: http://localhost:8200 (systemd service)"
    print_info "PostgreSQL: localhost:5432 (systemd service)"
    if [ "${CONFIG_INSTALL_SYSTEMD}" = "yes" ]; then
        print_info "MIRA: http://localhost:1993 (systemd service - ${STATUS_MIRA_SERVICE})"
    fi
elif [ "$OS" = "macos" ]; then
    print_info "Valkey: localhost:6379 (brew services)"
    print_info "Vault: http://localhost:8200 (launchd agent com.mira.vault)"
    print_info "PostgreSQL: localhost:5432 (brew services)"
    print_info "MIRA: http://localhost:1993 (launchd agent com.mira.app - ${STATUS_MIRA_SERVICE})"
fi

echo ""
echo -e "${BOLD}${GREEN}Next Steps${RESET}"
echo -e "  ${CYAN}→${RESET} Chat from any terminal: ${BOLD}mira${RESET} (first run: ${BOLD}mira --login --base-url http://localhost:1993 --chat${RESET})"
if [ "${CONFIG_INSTALL_SYSTEMD}" = "yes" ] && [ "$OS" = "linux" ]; then
    if [[ "${STATUS_MIRA_SERVICE}" == *"Running"* ]]; then
        echo -e "  ${CYAN}→${RESET} MIRA is running at: ${BOLD}http://localhost:1993${RESET}"
        echo -e "  ${CYAN}→${RESET} Open the web UI: ${BOLD}http://localhost:1993/chat${RESET}"
        echo -e "  ${CYAN}→${RESET} Check status: ${BOLD}systemctl status mira${RESET}"
        echo -e "  ${CYAN}→${RESET} View logs: ${BOLD}journalctl -u mira -f${RESET}"
        echo -e "  ${CYAN}→${RESET} Stop MIRA: ${BOLD}sudo systemctl stop mira${RESET}"
    elif [[ "${STATUS_MIRA_SERVICE}" == *"failed"* ]]; then
        echo -e "  ${CYAN}→${RESET} Check logs: ${BOLD}journalctl -u mira -n 50${RESET}"
        echo -e "  ${CYAN}→${RESET} Check status: ${BOLD}systemctl status mira${RESET}"
        echo -e "  ${CYAN}→${RESET} Try starting: ${BOLD}sudo systemctl start mira${RESET}"
    else
        echo -e "  ${CYAN}→${RESET} Start MIRA: ${BOLD}sudo systemctl start mira${RESET}"
        echo -e "  ${CYAN}→${RESET} Open the web UI: ${BOLD}http://localhost:1993/chat${RESET}"
        echo -e "  ${CYAN}→${RESET} View logs: ${BOLD}journalctl -u mira -f${RESET}"
    fi
    echo ""
    print_info "MIRA will auto-start on system boot (systemd enabled)"
else
    if [ "$OS" = "linux" ]; then
        # No systemd unit exists (user opted out or install failed), so the
        # Step 15b launcher written above is the start path.
        echo -e "  ${CYAN}→${RESET} Start MIRA: ${BOLD}/opt/mira/app/run.sh${RESET}"
        echo -e "  ${CYAN}→${RESET} Open the web UI: ${BOLD}http://localhost:1993/chat${RESET}"
        echo -e "  ${CYAN}→${RESET} After a reboot, unseal Vault first: ${BOLD}/opt/vault/unseal.sh${RESET}"
    else
        echo -e "  ${CYAN}→${RESET} MIRA runs as a LaunchAgent: ${BOLD}com.mira.app${RESET} (starts at login)"
        echo -e "  ${CYAN}→${RESET} Open the web UI: ${BOLD}http://localhost:1993/chat${RESET}"
        echo -e "  ${CYAN}→${RESET} Status: ${BOLD}launchctl print gui/$(id -u)/com.mira.app${RESET}"
        echo -e "  ${CYAN}→${RESET} Logs: ${BOLD}tail -f /opt/mira/logs/mira-launchd.log${RESET}"
        echo -e "  ${CYAN}→${RESET} Stop: ${BOLD}launchctl bootout gui/$(id -u)/com.mira.app${RESET}"
    fi
fi

# Harden /opt/vault/ permissions — restrict to MIRA_USER only
if [ "$OS" = "linux" ]; then
    sudo chmod 700 /opt/vault
    sudo find /opt/vault -type f -exec chmod 600 {} \;
    sudo find /opt/vault -type d -exec chmod 700 {} \;
    # Ensure scripts remain executable by owner
    sudo chmod 700 /opt/vault/unseal.sh 2>/dev/null || true
elif [ "$OS" = "macos" ]; then
    chmod 700 /opt/vault
    find /opt/vault -type f -exec chmod 600 {} \;
    find /opt/vault -type d -exec chmod 700 {} \;
    chmod 700 /opt/vault/unseal.sh 2>/dev/null || true
fi

if [ "$OS" = "macos" ]; then
    echo ""
    echo -e "${BOLD}${YELLOW}macOS Notes${RESET}"
    print_info "Supervision is via per-user LaunchAgents (start at login, restart on failure):"
    print_info "  com.mira.vault        — Vault server"
    print_info "  com.mira.vault.unseal — auto-unseal at login (bounded wait)"
    print_info "  com.mira.app          — MIRA via /opt/mira/app/run.sh"
    print_info "Inspect one: launchctl print gui/$(id -u)/<label>"
    print_info "Stop one:    launchctl bootout gui/$(id -u)/<label>"
    print_info "PostgreSQL and Valkey are managed by brew services"
fi

# Degraded-mode banner: subcortical runs on every user turn for entity
# extraction / passage filtering / query expansion. Without a key, those calls
# silently no-op, which looks like working MIRA until a user wonders why
# recall is poor. Announce the state loudly at the end so it can't be missed.
if [ "${CONFIG_SUBCORTICAL_API_KEY}" = "PLACEHOLDER_SET_THIS_LATER" ]; then
    echo ""
    echo -e "${BOLD}${YELLOW}╔════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${YELLOW}║  MIRA installed, but running in DEGRADED MODE          ║${RESET}"
    echo -e "${BOLD}${YELLOW}╚════════════════════════════════════════════════════════╝${RESET}"
    print_warning "No subcortical API key was configured."
    print_info "Entity extraction, passage filtering, and query expansion"
    print_info "will silently no-op on every conversation turn — memory"
    print_info "retrieval will be noticeably less accurate."
    echo ""
    print_info "To enable, store your subcortical provider key in Vault (default: lunaroute):"
    echo -e "${DIM}    export VAULT_ADDR='http://127.0.0.1:8200'${RESET}"
    echo -e "${DIM}    vault login \$(grep 'Initial Root Token' /opt/vault/init-keys.txt | awk '{print \$NF}')${RESET}"
    echo -e "${DIM}    vault kv patch secret/mira/api_keys subcortical_key=\"your-lunaroute-key\"${RESET}"
    echo ""
    print_info "Then restart MIRA to pick up the new key."
fi

# A renamed-aside database: this install runs against a fresh, empty
# mira_service, so say where the old data went and how to bring it forward.
if [ -n "${MIRA_PREVIOUS_DB:-}" ]; then
    echo ""
    echo -e "${BOLD}${YELLOW}Your previous database was kept as ${MIRA_PREVIOUS_DB}${RESET}"
    print_info "This install runs against a new, empty mira_service — nothing was deleted."
    print_info "To bring your data forward, follow:"
    echo -e "${DIM}    /opt/mira/app/deploy/HOW_TO_MIGRATE_OLD_INSTALLS.txt${RESET}"
fi

# Vault entries copied aside: the app reads only the primary secret/mira/*
# paths, so name the backups — the operator's previous credentials are
# recoverable there verbatim.
if [ "${#MIRA_VAULT_BACKUPS[@]}" -gt 0 ]; then
    echo ""
    echo -e "${BOLD}${YELLOW}Your previous Vault credentials were kept, not overwritten${RESET}"
    for backup in "${MIRA_VAULT_BACKUPS[@]}"; do
        print_info "  vault kv get ${backup}"
    done
    print_info "The running MIRA instance reads the primary secret/mira/* paths."
fi

echo ""
