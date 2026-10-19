#!/bin/bash
# Restart-time Vault unseal and root-token login, exec'd by the s6 oneshot
# vault-init/up once Vault answers on 127.0.0.1:8200.
#
# Vault returns sealed on every container start: the unseal key and initial
# root token persist in /opt/vault/init-keys.txt on the /opt/vault volume, but
# the seal state does not. This script extracts `Unseal Key 1` and `Initial
# Root Token` from that file and unseals only when `vault status` reports
# sealed; an already-unsealed Vault is left alone.
#
# No `set -e` by design: `vault status` exits 2 when sealed, which is the
# expected branch here, and errexit would abort every sealed restart before
# the unseal ran. Each failure path checks its own condition and exits 1 — a
# missing keys file, an unextractable key or token, an unexpected status code
# — which fails the oneshot, so the mira longrun never starts against a Vault
# it cannot read.

export VAULT_ADDR="http://127.0.0.1:8200"

echo "vault-unseal: Checking Vault status..."

# Check if init-keys.txt exists
if [ ! -f /opt/vault/init-keys.txt ]; then
    echo "vault-unseal: ERROR - init-keys.txt missing"
    exit 1
fi

# Check seal status using vault status exit codes
# 0 = unsealed, 2 = sealed, 1 = error
# Note: do NOT use set -e because vault status returns 2 when sealed (expected)
vault status > /dev/null 2>&1
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "vault-unseal: Vault already unsealed"
elif [ $STATUS -eq 2 ]; then
    echo "vault-unseal: Vault is sealed, unsealing..."
    UNSEAL_KEY=$(grep "Unseal Key 1" /opt/vault/init-keys.txt | awk '{print $NF}')
    if [ -z "$UNSEAL_KEY" ]; then
        echo "vault-unseal: ERROR - Could not extract unseal key"
        exit 1
    fi
    vault operator unseal "$UNSEAL_KEY"
    echo "vault-unseal: Unsealed successfully"
else
    echo "vault-unseal: ERROR - Vault status check failed (exit code: $STATUS)"
    exit 1
fi

# Authenticate with root token
echo "vault-unseal: Authenticating..."
ROOT_TOKEN=$(grep "Initial Root Token" /opt/vault/init-keys.txt | awk '{print $NF}')
if [ -z "$ROOT_TOKEN" ]; then
    echo "vault-unseal: ERROR - Could not extract root token"
    exit 1
fi
vault login "$ROOT_TOKEN" > /dev/null
echo "vault-unseal: Authenticated successfully"

echo "vault-unseal: Vault is ready"
