# deploy/vault.sh
# HashiCorp Vault binary installation, service configuration, and initialization
# Source this file - do not execute directly
#
# Requires: lib/output.sh, lib/services.sh, lib/vault.sh sourced first
# Requires: OS, DISTRO, MIRA_USER, MIRA_GROUP, LOUD_MODE variables set
#
# Sets: VAULT_ADDR (exported)

# Validate required variables
: "${OS:?Error: OS must be set}"
: "${MIRA_USER:?Error: MIRA_USER must be set (run python.sh first)}"
: "${MIRA_GROUP:?Error: MIRA_GROUP must be set (run python.sh first)}"

print_header "Step 8: HashiCorp Vault Setup"

if [ "$OS" = "linux" ]; then
    # Detect architecture
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)
            VAULT_ARCH="amd64"
            ;;
        aarch64|arm64)
            VAULT_ARCH="arm64"
            ;;
        *)
            print_error "Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac

    cd /tmp
    run_with_status "Downloading Vault 1.18.3 (${VAULT_ARCH})" \
        wget -q https://releases.hashicorp.com/vault/1.18.3/vault_1.18.3_linux_${VAULT_ARCH}.zip

    run_with_status "Extracting Vault binary" \
        unzip -o vault_1.18.3_linux_${VAULT_ARCH}.zip

    run_with_status "Installing to /usr/local/bin" \
        sudo mv vault /usr/local/bin/

    run_quiet sudo chmod +x /usr/local/bin/vault

    # Persistent SELinux label for the vault binary on Fedora/RHEL. chcon is
    # ephemeral — any restorecon/relabel reverts it and Vault then fails to
    # exec from systemd. semanage records the rule in policy (-a add, -m
    # modify when a re-deploy finds it already there) and restorecon applies
    # it. semanage ships in policycoreutils-python-utils, installed by the
    # dependencies.sh dnf branch.
    if [ "$DISTRO" = "fedora" ] && command -v getenforce &> /dev/null; then
        if [ "$(getenforce)" != "Disabled" ]; then
            run_with_status "Labeling Vault binary for SELinux (persistent)" \
                bash -c "sudo semanage fcontext -a -t bin_t /usr/local/bin/vault 2>/dev/null || sudo semanage fcontext -m -t bin_t /usr/local/bin/vault; sudo restorecon -F /usr/local/bin/vault"
        fi
    fi
elif [ "$OS" = "macos" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Verifying Vault installation... "
    if ! command -v vault &> /dev/null; then
        echo -e "${ERROR}"
        print_error "Vault installation failed. Please install manually: brew tap hashicorp/tap && brew install hashicorp/tap/vault"
        exit 1
    fi
    echo -e "${CHECKMARK}"
fi

run_with_status "Creating Vault directories" \
    sudo mkdir -p /opt/vault/data /opt/vault/config /opt/vault/logs

run_with_status "Setting Vault directory ownership" \
    sudo chown -R $MIRA_USER:$MIRA_GROUP /opt/vault

# Persistent SELinux labels for /opt/vault on Fedora/RHEL (see the binary
# block above for why semanage+restorecon, not chcon). var_lib_t is the
# service-data type Vault's file backend needs to read/write under an
# enforcing policy.
if [ "$OS" = "linux" ] && [ "$DISTRO" = "fedora" ] && command -v getenforce &> /dev/null; then
    if [ "$(getenforce)" != "Disabled" ]; then
        run_with_status "Labeling Vault directories for SELinux (persistent)" \
            bash -c "sudo semanage fcontext -a -t var_lib_t '/opt/vault(/.*)?' 2>/dev/null || sudo semanage fcontext -m -t var_lib_t '/opt/vault(/.*)?'; sudo restorecon -RF /opt/vault"
    fi
fi

echo -ne "${DIM}${ARROW}${RESET} Writing Vault configuration... "
cat > /opt/vault/config/vault.hcl <<'EOF'
storage "file" {
  path = "/opt/vault/data"
}

listener "tcp" {
  address     = "127.0.0.1:8200"
  tls_disable = 1
}

api_addr = "http://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"
ui = true

log_level = "Info"
EOF
echo -e "${CHECKMARK}"

print_header "Step 9: Vault Service Configuration"

if [ "$OS" = "linux" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Creating systemd service... "
    sudo tee /etc/systemd/system/vault.service > /dev/null <<EOF
[Unit]
Description=HashiCorp Vault
Documentation=https://www.vaultproject.io/docs/
Requires=network-online.target
After=network-online.target
ConditionFileNotEmpty=/opt/vault/config/vault.hcl

[Service]
Type=notify
User=$MIRA_USER
Group=$MIRA_GROUP
ProtectSystem=full
ProtectHome=no
PrivateTmp=yes
ExecStart=/usr/local/bin/vault server -config=/opt/vault/config/vault.hcl
ExecReload=/bin/kill --signal HUP \$MAINPID
KillMode=process
KillSignal=SIGINT
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF
    echo -e "${CHECKMARK}"

    run_quiet sudo systemctl daemon-reload
    run_with_status "Enabling Vault service" \
        sudo systemctl enable vault.service

    start_service vault.service systemctl
    sleep 2
elif [ "$OS" = "macos" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Installing Vault launchd agent... "
    # Supervisor parity with the vault.service unit above: a per-user
    # LaunchAgent with RunAtLoad+KeepAlive starts Vault at login and
    # restarts it on crash. A bare backgrounded process (the old path)
    # survived neither logout nor reboot while the deploy still claimed
    # "service configured and running".
    VAULT_BIN=$(command -v vault)
    mkdir -p "$HOME/Library/LaunchAgents"
    cat > "$HOME/Library/LaunchAgents/com.mira.vault.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mira.vault</string>
    <key>ProgramArguments</key>
    <array>
        <string>${VAULT_BIN}</string>
        <string>server</string>
        <string>-config=/opt/vault/config/vault.hcl</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/opt/vault</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/opt/vault/logs/vault.log</string>
    <key>StandardErrorPath</key>
    <string>/opt/vault/logs/vault.log</string>
</dict>
</plist>
EOF
    echo -e "${CHECKMARK}"

    echo -ne "${DIM}${ARROW}${RESET} Starting Vault service... "
    # One sanctioned launchd reload path (lib/services.sh:launchd_reload_agent):
    # re-deploys must replace a loaded agent, and killing the process alone
    # would just make KeepAlive resurrect it.
    if ! launchd_reload_agent "$HOME/Library/LaunchAgents/com.mira.vault.plist"; then
        echo -e "${ERROR}"
        print_error "Vault launchd agent failed to load. Check /opt/vault/logs/vault.log"
        exit 1
    fi
    echo -e "${CHECKMARK} ${DIM}(com.mira.vault)${RESET}"
fi

print_success "Vault service configured and running"

# Wait for Vault to be ready and check initialization state
echo -ne "${DIM}${ARROW}${RESET} Waiting for Vault to be ready... "
export VAULT_ADDR='http://127.0.0.1:8200'
VAULT_READY=0
for i in {1..30}; do
    if curl -s http://127.0.0.1:8200/v1/sys/health > /dev/null 2>&1; then
        VAULT_READY=1
        break
    fi
    sleep 1
done

if [ $VAULT_READY -eq 0 ]; then
    echo -e "${ERROR}"
    print_error "Vault did not become ready within 30 seconds"
    print_info "Check Vault logs: /opt/vault/logs/vault.log"
    exit 1
fi
echo -e "${CHECKMARK} ${DIM}(ready after ${i}s)${RESET}"

print_header "Step 10: Vault Initialization"

# Use unified vault_initialize function (handles check, unseal, auth, policy, AppRole)
vault_initialize
print_success "Vault fully configured"

# CRITICAL: Update mira.service immediately if it exists
# This ensures credentials are synced even if later migration phases fail
if [ "$OS" = "linux" ] && [ -f /etc/systemd/system/mira.service ]; then
    echo -ne "${DIM}${ARROW}${RESET} Syncing credentials to mira.service... "

    # Read fresh credentials from files
    NEW_ROLE_ID=$(cat /opt/vault/role-id.txt 2>/dev/null)
    NEW_SECRET_ID=$(cat /opt/vault/secret-id.txt 2>/dev/null)

    if [ -n "$NEW_ROLE_ID" ] && [ -n "$NEW_SECRET_ID" ]; then
        # Update the Environment lines in the existing service file
        sudo sed -i "s|^Environment=\"VAULT_ROLE_ID=.*\"|Environment=\"VAULT_ROLE_ID=$NEW_ROLE_ID\"|" /etc/systemd/system/mira.service
        sudo sed -i "s|^Environment=\"VAULT_SECRET_ID=.*\"|Environment=\"VAULT_SECRET_ID=$NEW_SECRET_ID\"|" /etc/systemd/system/mira.service
        sudo systemctl daemon-reload
        echo -e "${CHECKMARK}"
    else
        echo -e "${WARNING} ${DIM}(credential files not ready)${RESET}"
    fi
fi

print_header "Step 11: Auto-Unseal Configuration"

echo -ne "${DIM}${ARROW}${RESET} Creating unseal script... "
# One script for every platform. systemd runs it as a oneshot ordered
# After=vault.service; launchd runs it at login with NO ordering guarantee,
# so the bounded wait for the server lives here. Idempotent: unsealing an
# already-unsealed Vault is a no-op success (the old unconditional call
# exited 400 on re-runs, marking vault-unseal.service failed).
# The PATH prefix covers launchd's minimal environment (brew vault) and is
# a no-op on Linux.
cat > /opt/vault/unseal.sh <<'EOF'
#!/bin/bash
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
export VAULT_ADDR='http://127.0.0.1:8200'
for i in $(seq 1 60); do
    curl -sf http://127.0.0.1:8200/v1/sys/seal-status > /dev/null 2>&1 && break
    sleep 1
done
if ! curl -sf http://127.0.0.1:8200/v1/sys/seal-status > /dev/null 2>&1; then
    echo "unseal.sh: vault not reachable after 60 s" >&2
    exit 1
fi
if curl -sf http://127.0.0.1:8200/v1/sys/seal-status | grep -q '"sealed":true'; then
    UNSEAL_KEY=$(grep 'Unseal Key 1:' /opt/vault/init-keys.txt | awk '{print $NF}')
    vault operator unseal "$UNSEAL_KEY"
fi
EOF
echo -e "${CHECKMARK}"

run_quiet chmod +x /opt/vault/unseal.sh

if [ "$OS" = "linux" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Creating auto-unseal systemd service... "
    sudo tee /etc/systemd/system/vault-unseal.service > /dev/null <<'EOF'
[Unit]
Description=Vault Auto-Unseal
After=vault.service
Requires=vault.service

[Service]
Type=oneshot
ExecStart=/opt/vault/unseal.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    echo -e "${CHECKMARK}"

    run_quiet sudo systemctl daemon-reload
    run_with_status "Enabling auto-unseal service" \
        sudo systemctl enable vault-unseal.service
elif [ "$OS" = "macos" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Installing auto-unseal launchd agent... "
    cat > "$HOME/Library/LaunchAgents/com.mira.vault.unseal.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mira.vault.unseal</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/opt/vault/unseal.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/opt/vault/logs/unseal.log</string>
    <key>StandardErrorPath</key>
    <string>/opt/vault/logs/unseal.log</string>
</dict>
</plist>
EOF
    if ! launchd_reload_agent "$HOME/Library/LaunchAgents/com.mira.vault.unseal.plist"; then
        echo -e "${WARNING} ${DIM}(unseal agent failed to load; run /opt/vault/unseal.sh manually after restarts)${RESET}"
    else
        echo -e "${CHECKMARK} ${DIM}(com.mira.vault.unseal)${RESET}"
    fi
fi

print_success "Auto-unseal configured"
