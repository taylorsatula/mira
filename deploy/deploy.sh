#!/bin/bash
# MIRA Deployment Orchestrator
# This is the main entry point for deploying MIRA
#
# Usage: ./deploy/deploy.sh [--loud]
#
# Quick start (downloads and runs):
#   git clone https://github.com/taylorsatula/mira-OSS.git /tmp/mira-install && /tmp/mira-install/deploy/deploy.sh
#
# Options:
#   --loud     Show verbose output during installation
#
# There is no in-place upgrade path: 2.0 installs the greenfield schema from
# deploy/mira_service_schema.sql into an empty database. A user salvaging data
# from an older install needs only pg_dump and manual work.
#
# The deployment is broken into modular scripts:
#   lib/output.sh     - Visual output functions (colors, spinners)
#   lib/services.sh   - Service management helpers
#   lib/vault.sh      - Vault-specific helper functions
#   config.sh         - Interactive configuration gathering
#   preflight.sh      - System detection and validation
#   dependencies.sh   - System package installation
#   python.sh         - Python environment and MIRA setup
#   vault.sh          - HashiCorp Vault setup
#   postgresql.sh     - Database setup and credential storage
#   finalize.sh       - CLI setup, systemd, cleanup

set -e

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Bootstrap: Clone repo if running standalone
# ============================================================================
# If lib/output.sh doesn't exist, we were likely curl'd standalone - clone the repo
if [ ! -f "${SCRIPT_DIR}/lib/output.sh" ]; then
    echo "Cloning MIRA repository..."
    CLONE_DIR="/tmp/mira-install-$$"
    git clone --depth 1 https://github.com/taylorsatula/mira-OSS.git "$CLONE_DIR"
    exec "$CLONE_DIR/deploy/deploy.sh" "$@"
fi

# Parse arguments
LOUD_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--loud" ]; then
        LOUD_MODE=true
    fi
done

# ============================================================================
# Source shared libraries
# ============================================================================
source "${SCRIPT_DIR}/lib/output.sh"
source "${SCRIPT_DIR}/lib/services.sh"
source "${SCRIPT_DIR}/lib/vault.sh"

# ============================================================================
# Phase 1: Configuration Gathering
# ============================================================================
# config.sh handles:
#   - Variable initialization (CONFIG_*, STATUS_*)
#   - OS/distro detection
#   - Disk space and port checks
#   - Interactive prompts for API keys, options
#   - Configuration summary
source "${SCRIPT_DIR}/config.sh"

# ============================================================================
# Phase 2: Pre-flight Validation
# ============================================================================
# preflight.sh handles:
#   - System detection display
#   - Root check
#   - Sudo elevation
source "${SCRIPT_DIR}/preflight.sh"

# ============================================================================
# Phase 3: System Dependencies
# ============================================================================
# dependencies.sh handles:
#   - Package installation (apt/dnf/brew)
#   - llama.cpp build & model download (offline/local mode only)
#   - Sets: PYTHON_VER
source "${SCRIPT_DIR}/dependencies.sh"

# ============================================================================
# Phase 4: Python & Application Setup
# ============================================================================
# python.sh handles:
#   - Python verification
#   - MIRA download and installation
#   - Config patching for offline/custom providers
#   - Virtual environment and dependencies
#   - Embedding model download
#   - Playwright browser setup
#   - Sets: PYTHON_CMD, MIRA_USER, MIRA_GROUP
source "${SCRIPT_DIR}/python.sh"

# ============================================================================
# Phase 5: Vault Setup
# ============================================================================
# vault.sh handles:
#   - Vault binary download/installation
#   - Service configuration
#   - Initialization and auto-unseal
#   - Sets: VAULT_ADDR (exported)
source "${SCRIPT_DIR}/vault.sh"

# ============================================================================
# Phase 6: Database & Credentials
# ============================================================================
# postgresql.sh handles:
#   - Starting services (macOS)
#   - PostgreSQL readiness check
#   - Schema deployment
#   - Password updates
#   - Vault credential storage
source "${SCRIPT_DIR}/postgresql.sh"

# ============================================================================
# Phase 7: Finalization
# ============================================================================
# finalize.sh handles:
#   - MIRA CLI wrapper script
#   - Shell alias
#   - Systemd service (Linux, optional)
#   - Cleanup
#   - Success message and next steps
source "${SCRIPT_DIR}/finalize.sh"
