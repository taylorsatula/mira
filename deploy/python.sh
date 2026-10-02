# deploy/python.sh
# Python verification, MIRA download, venv setup, dependencies, embedding model, Playwright
# Source this file - do not execute directly
#
# Requires: lib/output.sh and lib/services.sh sourced first
# Requires: OS, PYTHON_VER, CONFIG_*, LOUD_MODE, RELEASE_TAG variables set
# (RELEASE_TAG is defined and exported by deploy/deploy.sh — the single
# source of truth for the release the installer deploys)
#
# Sets: PYTHON_CMD, MIRA_USER, MIRA_GROUP

# Validate required variables
: "${OS:?Error: OS must be set}"
: "${PYTHON_VER:?Error: PYTHON_VER must be set (run dependencies.sh first)}"
: "${RELEASE_TAG:?Error: RELEASE_TAG must be set (deploy.sh defines and exports it)}"

print_header "Step 2: Python Verification"

echo -ne "${DIM}${ARROW}${RESET} Locating Python ${PYTHON_VER}+... "
if [ "$OS" = "linux" ]; then
    # Use the version detected in Step 1
    if ! command -v python${PYTHON_VER} &> /dev/null; then
        echo -e "${ERROR}"
        print_error "Python ${PYTHON_VER} not found after installation."
        exit 1
    fi
    PYTHON_CMD="python${PYTHON_VER}"
elif [ "$OS" = "macos" ]; then
    # PYTHON_VER already set by dependencies.sh to 3.12+ version
    # Check common Homebrew locations
    if command -v python${PYTHON_VER} &> /dev/null; then
        PYTHON_CMD="python${PYTHON_VER}"
    elif [ -f "/opt/homebrew/opt/python@${PYTHON_VER}/bin/python${PYTHON_VER}" ]; then
        PYTHON_CMD="/opt/homebrew/opt/python@${PYTHON_VER}/bin/python${PYTHON_VER}"
    elif [ -f "/usr/local/opt/python@${PYTHON_VER}/bin/python${PYTHON_VER}" ]; then
        PYTHON_CMD="/usr/local/opt/python@${PYTHON_VER}/bin/python${PYTHON_VER}"
    else
        echo -e "${ERROR}"
        print_error "Python ${PYTHON_VER} not found. Check Homebrew installation."
        exit 1
    fi
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "${CHECKMARK} ${DIM}$PYTHON_VERSION${RESET}"

# Validate Python version is 3.12 or higher
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]; }; then
    print_error "MIRA requires Python 3.12 or higher. Found: $PYTHON_VERSION"
    print_info "Please install Python 3.12+ and re-run the deployment script."
    exit 1
fi

print_header "Step 3: MIRA Download & Installation"

# Determine user/group for ownership
if [ "$OS" = "linux" ]; then
    MIRA_USER="$(whoami)"
    MIRA_GROUP="$(id -gn)"
elif [ "$OS" = "macos" ]; then
    MIRA_USER="$(whoami)"
    MIRA_GROUP="staff"
fi

run_with_status "Creating /opt/mira/app directory" \
    sudo mkdir -p /opt/mira/app

# Clear any previous install's code before the new payload lands: a
# retired module left behind by an overlay is imported at boot and parks
# the POST gate. Host-local state the install payload never contains
# (venv, .env, data, logs — the same set both install paths exclude) is
# preserved; data/ holds per-user storage (utils/userdata_manager.py
# base_dir) and venv/ is reused by Step 4.
run_with_status "Clearing previous install (code only)" \
    sudo find /opt/mira/app -mindepth 1 -maxdepth 1 \
        ! -name venv ! -name .env ! -name data ! -name logs \
        -exec rm -rf {} +

if [ "${LOCAL_SOURCE:-false}" = "true" ]; then
    # --local: install from the repo checkout deploy.sh was invoked from
    # (cwd captured at argument-parsing time) instead of wget-ing GitHub.
    # Same target, ownership, and downstream steps as the tarball path.
    LOCAL_SOURCE_DIR="${LOCAL_SOURCE_DIR:?--local requires the cwd captured by deploy.sh}"
    for f in main.py requirements.txt deploy/deploy.sh; do
        if [ ! -f "$LOCAL_SOURCE_DIR/$f" ]; then
            print_error "--local: '$LOCAL_SOURCE_DIR' is not a mira-OSS checkout (missing $f)"
            print_info "Run from the repo root: cd /path/to/mira-OSS && ./deploy/deploy.sh --local ..."
            exit 1
        fi
    done
    # Excludes are the untracked runtime junk the GitHub tarball never
    # contains; everything else — including uncommitted modifications —
    # is installed.
    run_with_status "Installing MIRA from local tree ($LOCAL_SOURCE_DIR)" \
        bash -c "sudo tar -C '$LOCAL_SOURCE_DIR' \\
            --exclude=.git --exclude=venv --exclude=__pycache__ \\
            --exclude='*.pyc' --exclude=.env --exclude=.claude --exclude=.DS_Store \\
            --exclude=data --exclude=logs --exclude=scratch \\
            -cf - . | sudo tar -C /opt/mira/app -xf -"
else
    # Download to /tmp to keep user's home directory clean
    cd /tmp

    # Pinned release tag: RELEASE_TAG (from deploy/deploy.sh) is the single
    # source of truth; this is the deploy/RELEASE.md procedure, now wired.
    # GitHub's tag archive extracts to mira-<tag without the leading "v">.
    TARBALL="mira-${RELEASE_TAG#v}.tar.gz"
    SRC_DIR="/tmp/mira-${RELEASE_TAG#v}"

    run_with_status "Downloading MIRA release ${RELEASE_TAG}" \
        wget -q -O "$TARBALL" "https://github.com/taylorsatula/mira-OSS/archive/refs/tags/${RELEASE_TAG}.tar.gz"

    run_with_status "Extracting archive" \
        tar -xzf "$TARBALL" -C /tmp

    run_with_status "Copying files to /opt/mira/app" \
        sudo cp -r "$SRC_DIR"/* /opt/mira/app/

    # Clean up immediately after copying
    run_quiet rm -f "/tmp/$TARBALL"
    run_quiet rm -rf "$SRC_DIR"
fi

run_with_status "Setting ownership to $MIRA_USER:$MIRA_GROUP" \
    sudo chown -R $MIRA_USER:$MIRA_GROUP /opt/mira

print_success "MIRA installed to /opt/mira/app"

# Offline mode: LLM endpoints are configured post-schema by postgresql.sh
# (UPDATEs all five model_configs rows to llama-server endpoints and models)
if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
    echo ""
    echo -e "${DIM}NOTE: the 'other' route has no outside vendor to consult when air-gapped.${RESET}"
    echo -e "${DIM}postgresql.sh points it at the small local instance so routing stays valid.${RESET}"
    echo ""
fi

# Hosted-install route configuration is a database concern, not a schema-text
# concern: postgresql.sh rewrites the seeded model_configs rows with UPDATEs
# after applying the schema (the same mechanism OFFLINE_SQL uses), matching
# the config's chat + subcortical providers. The seed rows in
# mira_service_schema.sql are the lunaroute defaults; nothing string-patches
# them here anymore.

print_header "Step 4: Python Environment Setup"

cd /opt/mira/app

# Check if venv already exists
echo -ne "${DIM}${ARROW}${RESET} Checking for existing virtual environment... "
if [ -f venv/bin/python3 ]; then
    VENV_PYTHON_VERSION=$(venv/bin/python3 --version 2>&1 | awk '{print $2}')
    echo -e "${CHECKMARK} ${DIM}$VENV_PYTHON_VERSION (existing)${RESET}"
    print_info "Reusing existing virtual environment"
else
    echo -e "${DIM}(not found)${RESET}"
    run_with_status "Creating virtual environment" \
        $PYTHON_CMD -m venv venv

    run_with_status "Initializing pip" \
        venv/bin/python3 -m ensurepip
fi

if [ "$CONFIG_EMBEDDING_PROVIDER" = "local" ]; then
    echo -ne "${DIM}${ARROW}${RESET} Checking PyTorch installation... "
    if check_exists package torch; then
        TORCH_VERSION=$(venv/bin/pip3 show torch | grep Version | awk '{print $2}')
        echo -e "${CHECKMARK} ${DIM}$TORCH_VERSION (existing)${RESET}"
        print_info "Note: If you have CUDA-enabled PyTorch, it will be preserved"
    else
        echo -e "${DIM}(not installed yet)${RESET}"
        if [ "$LOUD_MODE" = true ]; then
            print_step "Installing PyTorch CPU-only version..."
            venv/bin/pip3 install torch --index-url https://download.pytorch.org/whl/cpu
        else
            (venv/bin/pip3 install -q torch --index-url https://download.pytorch.org/whl/cpu) &
            show_progress $! "Installing PyTorch CPU-only"
        fi
    fi
else
    print_info "Remote embeddings: skipping PyTorch"
fi

print_header "Step 5: Python Dependencies"

# Count packages in requirements.txt
PACKAGE_COUNT=$(grep -c '^[^#]' requirements.txt 2>/dev/null || echo "many")
echo -e "${DIM}This is the one that is going to take a while (~${PACKAGE_COUNT} packages)${RESET}"
echo ""

if [ "$LOUD_MODE" = true ]; then
    print_step "Installing from requirements.txt..."
    venv/bin/pip3 install -r requirements.txt
else
    (venv/bin/pip3 install -q -r requirements.txt) &
    show_progress $! "Installing Python packages from requirements.txt"
    if [ $? -ne 0 ]; then
        print_error "Failed to install Python packages from requirements.txt"
        print_info "Run with --loud flag to see detailed error output"
        exit 1
    fi
fi

if [ "$CONFIG_EMBEDDING_PROVIDER" = "local" ]; then
    # Install sentence-transformers separately to ensure proper dependency resolution
    # (torch is installed first so deployments use the CPU wheel)
    echo -ne "${DIM}${ARROW}${RESET} Checking sentence-transformers... "
    if ! check_exists package sentence-transformers; then
        echo ""
        install_python_package sentence-transformers
        if [ $? -ne 0 ]; then
            print_error "Failed to install sentence-transformers"
            print_info "Run with --loud flag to see detailed error output"
            exit 1
        fi
    else
        install_python_package sentence-transformers  # This will show version if already installed
    fi
else
    print_info "Remote embeddings: skipping sentence-transformers"
fi

print_success "Python dependencies installed"

# Install TOAST log level into venv so it's available at interpreter startup
echo -ne "${DIM}${ARROW}${RESET} Installing custom log levels... "
SITE_PACKAGES=$(venv/bin/python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
cp "${SCRIPT_DIR}/_mira_log_levels.py" "$SITE_PACKAGES/"
echo "import _mira_log_levels" > "$SITE_PACKAGES/mira-log-levels.pth"
echo -e "${CHECKMARK}"

print_header "Step 6: Embedding Model Download"
if [ "$CONFIG_EMBEDDING_PROVIDER" = "local" ]; then

    # Download MongoDB leaf embedding model (768d asymmetric retrieval)
    echo -ne "${DIM}${ARROW}${RESET} Checking embedding model cache... "
    MODEL_CACHED=$(venv/bin/python3 << 'EOF'
from pathlib import Path

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

def check_model_cached(model_substring):
    """Check if a model is fully cached by looking for model directories and required files"""
    if not cache_dir.exists():
        return False

    model_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and model_substring in d.name]

    for model_dir in model_dirs:
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    has_config = (snapshot / "config.json").exists()
                    has_model = (snapshot / "pytorch_model.bin").exists() or (snapshot / "model.safetensors").exists()
                    if has_config and has_model:
                        return True
    return False

if check_model_cached("mdbr-leaf-ir-asym"):
    print("cached")
else:
    print("not_cached")
EOF
    )

    if [ "$MODEL_CACHED" = "cached" ]; then
        echo -e "${CHECKMARK} ${DIM}(MongoDB/mdbr-leaf-ir-asym already cached)${RESET}"
        print_info "To re-download: rm -rf ~/.cache/huggingface/hub/*mdbr-leaf*"
    else
        echo -e "${DIM}(not found)${RESET}"
        if [ "$LOUD_MODE" = true ]; then
            print_step "Downloading MongoDB/mdbr-leaf-ir-asym embedding model..."
            venv/bin/python3 << 'EOF'
from sentence_transformers import SentenceTransformer
print("→ Loading/downloading MongoDB/mdbr-leaf-ir-asym (768d)...")
SentenceTransformer("MongoDB/mdbr-leaf-ir-asym")
print("✓ mdbr-leaf-ir-asym ready")
EOF
        else
            (venv/bin/python3 << 'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("MongoDB/mdbr-leaf-ir-asym")
EOF
    ) &
            show_progress $! "Downloading MongoDB/mdbr-leaf-ir-asym embedding model"
        fi
    fi

    print_success "Embedding model ready"
else
    print_info "Remote embeddings (${CONFIG_EMBEDDING_MODEL}): no local embedding model to download"
fi

print_header "Step 7: Playwright Browser Setup"

if [ "${CONFIG_INSTALL_PLAYWRIGHT}" = "yes" ]; then
    # playwright is an optional dependency and is deliberately absent from
    # requirements.txt, so the package is installed here rather than in Step 5.
    # That makes the config prompt gate the package and the browser together:
    # opting out leaves web_tool._fetch_playwright() to report the
    # "playwright_unavailable" error code, and the POST check
    # CheckSpec("playwright_service", required=False, ...) to fail as advisory.
    echo -ne "${DIM}${ARROW}${RESET} Checking playwright package... "
    install_python_package playwright
    if [ $? -ne 0 ]; then
        print_error "Failed to install playwright"
        print_info "Run with --loud flag to see detailed error output"
        exit 1
    fi

    # Check if Playwright Chromium is already installed
    PLAYWRIGHT_CACHE="$HOME/.cache/ms-playwright"
    echo -ne "${DIM}${ARROW}${RESET} Checking Playwright cache... "
    if [ -d "$PLAYWRIGHT_CACHE" ] && ls "$PLAYWRIGHT_CACHE"/chromium-* >/dev/null 2>&1; then
        echo -e "${CHECKMARK} ${DIM}(already installed)${RESET}"
        print_info "To update browsers: venv/bin/playwright install chromium"
    else
        echo -e "${DIM}(not found)${RESET}"
        if [ "$LOUD_MODE" = true ]; then
            print_step "Installing Playwright Chromium browser..."
            venv/bin/playwright install chromium
        else
            (venv/bin/playwright install chromium > /dev/null 2>&1) &
            show_progress $! "Installing Playwright Chromium"
        fi
    fi

    # System dependencies - optional, may fail on newer Ubuntu
    if [ "$OS" = "linux" ]; then
        echo -ne "${DIM}${ARROW}${RESET} Installing Playwright system dependencies... "
        if sudo venv/bin/playwright install-deps > /tmp/playwright-deps.log 2>&1; then
            echo -e "${CHECKMARK}"
            rm -f /tmp/playwright-deps.log
        else
            echo -e "${WARNING}"
            print_warning "Some system dependencies failed to install"

            # Extract specific failed packages if possible
            FAILED_PACKAGES=$(grep "Unable to locate package" /tmp/playwright-deps.log 2>/dev/null | sed 's/.*Unable to locate package //' | head -3 | tr '\n' ' ')
            if [ -n "$FAILED_PACKAGES" ]; then
                print_info "Missing packages: $FAILED_PACKAGES"
            fi

            print_info "This is common on Ubuntu 24.04+ due to package name changes"
            print_info "Playwright should still work in headless mode for most sites"
            print_info "Full log saved to: /tmp/playwright-deps.log"
        fi
    elif [ "$OS" = "macos" ]; then
        print_info "Playwright browser dependencies are bundled on macOS"
    fi

    print_success "Playwright configured"
else
    print_info "Playwright installation skipped (user opted out)"
    print_info "Note: Advanced webpage extraction will not be available"
    print_info "Basic HTTP requests and web search will still work"
    print_success "Playwright setup skipped"
fi
