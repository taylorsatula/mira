#!/bin/bash
# MIRA Deployment Orchestrator
# This is the main entry point for deploying MIRA
#
# Usage: ./deploy/deploy.sh [--loud] [--config <file>] [--local]
#
# --config bypasses the interactive interview entirely: copy
# deploy/deploy-config.example.yml, fill in the placeholders, and run
#   ./deploy/deploy.sh --config deploy-config.yml --loud
#
# --local installs the MIRA code from the CURRENT DIRECTORY (a mira-OSS
# checkout, typically with uncommitted dev changes) instead of downloading
# the main-branch tarball from GitHub. Run it from the repo root:
#   cd /path/to/mira-OSS && ./deploy/deploy.sh --config deploy-config.yml --local --loud
# Untracked runtime junk the GitHub tarball never contains is excluded
# (.git, venv, __pycache__, *.pyc, .env, data, logs, scratch); everything
# else — including uncommitted modifications — is installed to /opt/mira/app
# with the same ownership and downstream steps as the GitHub path.
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
CONFIG_FILE=""
LOCAL_SOURCE="false"
while [ $# -gt 0 ]; do
    case "$1" in
        --loud) LOUD_MODE=true ;;
        --local)
            # Capture cwd NOW, before any phase script changes directory:
            # python.sh installs from this tree instead of wget-ing GitHub.
            LOCAL_SOURCE="true"
            LOCAL_SOURCE_DIR="$(pwd)" ;;
        --config)
            shift
            CONFIG_FILE="${1:?--config requires a file path}" ;;
        --config=*) CONFIG_FILE="${1#*=}" ;;
        *)
            echo "Unknown option: $1 (supported: --loud, --local, --config <file>)"
            exit 1 ;;
    esac
    shift
done

# ============================================================================
# Source shared libraries
# ============================================================================
source "${SCRIPT_DIR}/lib/output.sh"
source "${SCRIPT_DIR}/lib/services.sh"
source "${SCRIPT_DIR}/lib/vault.sh"
source "${SCRIPT_DIR}/lib/embedding_config.sh"
source "${SCRIPT_DIR}/lib/systemone_config.sh"

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
#   - model_configs route rewrite (offline OFFLINE_SQL, or hosted chat+subcortical)
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
