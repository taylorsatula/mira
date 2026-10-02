# deploy/dependencies.sh
# System package installation and local LLM setup (llama.cpp)
# Source this file - do not execute directly
#
# Requires: lib/output.sh and lib/services.sh sourced first
# Requires: OS, DISTRO, CONFIG_OFFLINE_MODE, CONFIG_LOCAL_MODEL_CHOICE, LOUD_MODE variables set
#
# Sets: PYTHON_VER

# Validate required variables
: "${OS:?Error: OS must be set}"
# DISTRO can be empty for macOS, so just check if variable exists
if [ -z "${DISTRO+x}" ]; then
    echo "Error: DISTRO variable must be set (can be empty string for macOS)"
    exit 1
fi

print_header "Step 1: System Dependencies"

if [ "$OS" = "linux" ] && [ "$DISTRO" = "debian" ]; then
    # Non-interactive debconf for every apt call in this branch: over ssh with
    # no controlling tty the Dialog/Readline frontends fail and fall back
    # noisily; postgresql-common's cluster prompts must answer deterministically.
    export DEBIAN_FRONTEND=noninteractive

    # Debian/Ubuntu: Add PostgreSQL APT repository for PostgreSQL 17.
    # Codename comes from /etc/os-release (guaranteed on every systemd
    # Debian-family image), not lsb_release — the lsb-release package is
    # absent on minimal/netinst images and the pgdg.list line then comes out
    # malformed ("-pgdg main" with no codename), breaking apt-get update.
    if [ ! -f /etc/apt/sources.list.d/pgdg.list ]; then
        run_with_status "Adding PostgreSQL APT repository" \
            bash -c 'sudo apt-get install -y ca-certificates wget > /dev/null 2>&1 && \
                     sudo install -d /usr/share/postgresql-common/pgdg && \
                     sudo wget -q -O /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc && \
                     echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(. /etc/os-release && echo "$VERSION_CODENAME")-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list > /dev/null'
    fi

    # Detect Python version to use (newest available, 3.12+ required)
    PYTHON_VER=$(python3 --version 2>&1 | sed -n 's/Python \([0-9]*\.[0-9]*\).*/\1/p')

    if [ "$LOUD_MODE" = true ]; then
        print_step "Updating package lists..."
        sudo apt-get update
        print_step "Installing system packages (Python ${PYTHON_VER})..."
        # valkey-server, not valkey: Ubuntu splits the package and ships no
        # 'valkey' metapackage (Debian/Fedora do) — asking apt for 'valkey' on
        # Ubuntu LTS aborts this step with 'Unable to locate package'.
        # Playwright Chromium runtime deps (full set, matching the Docker
        # base image) — a subset leaves Playwright's host validation
        # warning that browsers cannot launch.
        # Contrib modules ship inside postgresql-17 (pg_trgm, pgcrypto, cube;
        # verified on noble and resolute) — do NOT add an unversioned
        # postgresql-contrib: it resolves to the distro's own postgres major
        # (16 on noble, 18 on resolute), whose auto-created cluster races the
        # pinned 17 cluster for port 5432.
        sudo apt-get install -y \
            build-essential \
            cmake \
            git \
            python${PYTHON_VER}-venv \
            python${PYTHON_VER}-dev \
            libpq-dev \
            postgresql-server-dev-17 \
            unzip \
            wget \
            curl \
            postgresql-17 \
            postgresql-17-pgvector \
            valkey-server \
            libnss3 \
            libnspr4 \
            libatk1.0-0t64 \
            libatk-bridge2.0-0t64 \
            libcups2t64 \
            libdrm2 \
            libxkbcommon0 \
            libxcomposite1 \
            libxdamage1 \
            libxfixes3 \
            libxrandr2 \
            libgbm1 \
            libpango-1.0-0 \
            libcairo2 \
            libasound2t64 \
            libatspi2.0-0t64
    else
        # Silent mode with progress indicator
        (sudo apt-get update > /dev/null 2>&1) &
        show_progress $! "Updating package lists"

        # valkey-server + the contrib/postgresql version pins — see the
        # loud-branch comments above
        (sudo apt-get install -y \
            build-essential cmake git python${PYTHON_VER}-venv python${PYTHON_VER}-dev libpq-dev \
            postgresql-server-dev-17 unzip wget curl postgresql-17 \
            postgresql-17-pgvector valkey-server \
            libnss3 libnspr4 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 \
            libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
            libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2t64 \
            libatspi2.0-0t64 > /dev/null 2>&1) &
        show_progress $! "Installing system packages"
    fi
elif [ "$OS" = "linux" ] && [ "$DISTRO" = "fedora" ]; then
    # Check minimum Fedora version (PGDG dropped support for F-40 and earlier)
    FEDORA_VER=$(rpm -E %fedora 2>/dev/null || echo 0)
    if [ "$FEDORA_VER" -lt 41 ]; then
        print_error "Fedora $FEDORA_VER is not supported — MIRA requires Fedora 41+"
        print_info "PostgreSQL 17 + PGDG repository are unavailable on older releases."
        exit 1
    fi

    # PGDG publishes its Fedora/RHEL repos for x86_64 only. On any other
    # arch the repo URL 404s and the dnf install below fails with noise —
    # fail loud here with the manual path instead.
    if [ "$(uname -m)" != "x86_64" ]; then
        print_error "Automated Fedora/RHEL install requires x86_64 (PGDG publishes no $(uname -m) repos)."
        print_info "Install PostgreSQL 17+, pgvector, Valkey, and Vault from distro sources,"
        print_info "then follow docs/MANUAL_INSTALL.md and re-run with the packages present."
        exit 1
    fi

    # Fedora/RHEL: Add PostgreSQL PGDG repository for PostgreSQL 17.
    # Guard on the real package names: the download FILE is named
    # pgdg-fedora-repo-latest.noarch.rpm but the PACKAGE it installs is
    # pgdg-fedora-repo — guarding on the file name never matched and
    # reinstalled the repo RPM on every run.
    if ! rpm -q pgdg-fedora-repo > /dev/null 2>&1 && ! rpm -q pgdg-redhat-repo > /dev/null 2>&1; then
        if [ -f /etc/fedora-release ]; then
            run_with_status "Adding PostgreSQL PGDG repository" \
                sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/F-$(rpm -E %fedora)-x86_64/pgdg-fedora-repo-latest.noarch.rpm
        else
            # RHEL/CentOS/Rocky/Alma
            run_with_status "Adding PostgreSQL PGDG repository" \
                sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-$(rpm -E %rhel)-x86_64/pgdg-redhat-repo-latest.noarch.rpm
        fi
        # Import the PGDG signing key non-interactively: dnf5 otherwise prompts
        # at the first makecache ("Is this ok [y/N]: Importing OpenPGP key"),
        # which stalls an unattended deploy. The repo RPM ships the key file;
        # the import is idempotent.
        run_with_status "Importing PGDG signing key" \
            bash -c 'sudo rpm --import /etc/pki/rpm-gpg/PGDG-RPM-GPG-KEY-*' \
            || print_warning "PGDG key import failed; dnf may prompt to import it during package install"
    fi

    # (No dnf module disable here: Fedora 41+ ships dnf5, which has no
    # module interface, and Fedora has shipped no modular PostgreSQL since
    # F39 — the old `dnf module disable postgresql` was dead weight.)

    # Determine correct development tools group name
    # Fedora uses "development-tools", RHEL/Rocky/Alma use "Development Tools"
    if [ -f /etc/fedora-release ]; then
        DEV_TOOLS_GROUP="@development-tools"
    else
        DEV_TOOLS_GROUP="@Development Tools"
    fi

    if [ "$LOUD_MODE" = true ]; then
        print_step "Updating package lists..."
        sudo dnf makecache
        print_step "Installing system packages..."
        # Playwright Chromium runtime deps — same set as the apt branch.
        sudo dnf install -y \
            "$DEV_TOOLS_GROUP" \
            python3-devel \
            python3-pip \
            libpq-devel \
            postgresql17-server \
            postgresql17-contrib \
            postgresql17-devel \
            pgvector_17 \
            unzip \
            wget \
            curl \
            valkey \
            policycoreutils-python-utils \
            atk \
            at-spi2-atk \
            at-spi2-core \
            libXcomposite \
            nss \
            nspr \
            cups-libs \
            alsa-lib \
            mesa-libgbm \
            libxkbcommon \
            libXdamage \
            libXfixes \
            libXrandr \
            pango \
            cairo
    else
        # Silent mode with progress indicator
        (sudo dnf makecache > /dev/null 2>&1) &
        show_progress $! "Updating package lists"

        (sudo dnf install -y \
            "$DEV_TOOLS_GROUP" python3-devel python3-pip libpq-devel \
            postgresql17-server postgresql17-contrib postgresql17-devel pgvector_17 \
            unzip wget curl valkey policycoreutils-python-utils \
            atk at-spi2-atk at-spi2-core libXcomposite \
            nss nspr cups-libs alsa-lib mesa-libgbm libxkbcommon \
            libXdamage libXfixes libXrandr pango cairo > /dev/null 2>&1) &
        show_progress $! "Installing system packages"
    fi

    # Initialize PostgreSQL database cluster if not already done.
    # `sudo test`: /var/lib/pgsql is mode 0700 postgres — an unprivileged -d
    # test always reads "missing" (EACCES on traversal), so a plain test ran
    # initdb on every re-deploy and aborted on "Data directory is not empty".
    if ! sudo test -d /var/lib/pgsql/17/data/base; then
        run_with_status "Initializing PostgreSQL database cluster" \
            sudo /usr/pgsql-17/bin/postgresql-17-setup initdb
    fi

    # Configure pg_hba.conf: password auth for TCP (host) connections only.
    # MIRA connects via localhost TCP with the provisioned role password, so
    # the host lines must be scram-sha-256 (PGDG initdb defaults them to
    # ident). The unix-socket (local) lines keep their initdb default
    # (ident/peer): that is how every `sudo -u postgres psql` provisioning
    # call authenticates — the postgres role has no password, so rewriting
    # local lines to scram locks the deploy out of its own database.
    # sudo test/grep: the data dir is 0700 postgres — unprivileged guards
    # silently evaluate false and made this whole block dead code (observed
    # live on Fedora 44: the edits never ran; PGDG's initdb defaults happened
    # to be local-peer + host-scram, masking it). EL derivatives may default
    # to ident — the normalization must actually execute.
    PG_HBA="/var/lib/pgsql/17/data/pg_hba.conf"
    if sudo test -f "$PG_HBA"; then
        if ! sudo grep -qE "^host[[:space:]]+all[[:space:]]+all[[:space:]]+127\\.0\\.0\\.1/32[[:space:]]+scram-sha-256" "$PG_HBA" 2>/dev/null; then
            run_with_status "Configuring PostgreSQL authentication (scram-sha-256 for TCP)" \
                bash -c "sudo sed -i -E 's#^host[[:space:]]+all[[:space:]]+all[[:space:]]+127\\.0\\.0\\.1/32[[:space:]]+(ident|peer|trust|md5)#host    all             all             127.0.0.1/32            scram-sha-256#' $PG_HBA && \
                         sudo sed -i -E 's#^host[[:space:]]+all[[:space:]]+all[[:space:]]+::1/128[[:space:]]+(ident|peer|trust|md5)#host    all             all             ::1/128                 scram-sha-256#' $PG_HBA"
        fi
    fi

    # Enable and start PostgreSQL service
    run_with_status "Enabling PostgreSQL service" \
        sudo systemctl enable postgresql-17
    run_with_status "Starting PostgreSQL service" \
        sudo systemctl start postgresql-17
    # Apply the pg_hba edits to an already-running cluster (on re-deploys the
    # start above is a no-op and the config would never be re-read).
    run_quiet sudo systemctl reload postgresql-17 || true

    # Enable and start Valkey service
    run_with_status "Enabling Valkey service" \
        sudo systemctl enable valkey
    run_with_status "Starting Valkey service" \
        sudo systemctl start valkey

    # Detect Python version after installation
    PYTHON_VER=$(python3 --version 2>&1 | sed -n 's/Python \([0-9]*\.[0-9]*\).*/\1/p')

elif [ "$OS" = "linux" ] && [ "$DISTRO" = "arch" ]; then
    # Arch Linux (pacman). Rolling release — `python` is always the current
    # CPython (≥3.12 floor satisfied; venv+ensurepip verified working).
    # Full system upgrade first: Arch does not support partial upgrades, so
    # installing fresh packages against stale synced DBs is not an option.
    # atk/at-spi2-atk are merged into at-spi2-core upstream (verified: the
    # separate package names no longer exist in the repos).
    if [ "$LOUD_MODE" = true ]; then
        print_step "Upgrading system packages (Arch rolling — full upgrade required)..."
        sudo pacman -Syu --noconfirm
        print_step "Installing system packages..."
        # Playwright Chromium runtime deps — same set as the apt/dnf branches.
        sudo pacman -S --noconfirm --needed \
            base-devel \
            cmake \
            git \
            wget \
            curl \
            unzip \
            python \
            python-pip \
            postgresql \
            postgresql-libs \
            pgvector \
            valkey \
            nss \
            nspr \
            at-spi2-core \
            cups \
            libdrm \
            libxkbcommon \
            libxcomposite \
            libxdamage \
            libxfixes \
            libxrandr \
            mesa \
            pango \
            cairo \
            alsa-lib
    else
        (sudo pacman -Syu --noconfirm > /dev/null 2>&1) &
        show_progress $! "Upgrading system packages (Arch rolling)"

        (sudo pacman -S --noconfirm --needed \
            base-devel cmake git wget curl unzip \
            python python-pip postgresql postgresql-libs pgvector valkey \
            nss nspr at-spi2-core cups libdrm libxkbcommon \
            libxcomposite libxdamage libxfixes libxrandr \
            mesa pango cairo alsa-lib > /dev/null 2>&1) &
        show_progress $! "Installing system packages"
    fi

    # Initialize the PostgreSQL cluster if not already done. Arch's package
    # creates the postgres user and /var/lib/postgres/data but deliberately
    # does NOT run initdb (unlike Debian's pg_createcluster and Fedora's
    # postgresql-17-setup). Idempotent: PG_VERSION marks an initialized
    # cluster. `sudo test`: the data dir is postgres-owned and may not be
    # traversable by the deploy user (same trap as the Fedora guard).
    if ! sudo test -f /var/lib/postgres/data/PG_VERSION; then
        run_with_status "Initializing PostgreSQL database cluster" \
            sudo -u postgres initdb -D /var/lib/postgres/data --locale C.UTF-8 --encoding UTF8
    fi

    # Password auth for TCP connections only — Arch's initdb defaults every
    # line to trust. The app connects via localhost TCP with the provisioned
    # role password (host lines → scram-sha-256); the unix-socket (local)
    # lines keep trust so every `sudo -u postgres psql` provisioning call
    # still authenticates. Same split as the Fedora branch.
    PG_HBA="/var/lib/postgres/data/pg_hba.conf"
    if sudo test -f "$PG_HBA"; then
        if ! sudo grep -qE "^host[[:space:]]+all[[:space:]]+all[[:space:]]+127\\.0\\.0\\.1/32[[:space:]]+scram-sha-256" "$PG_HBA" 2>/dev/null; then
            run_with_status "Configuring PostgreSQL authentication (scram-sha-256 for TCP)" \
                bash -c "sudo sed -i -E 's#^host[[:space:]]+all[[:space:]]+all[[:space:]]+127\\.0\\.0\\.1/32[[:space:]]+(trust|ident|peer|md5)#host    all             all             127.0.0.1/32            scram-sha-256#' $PG_HBA && \
                         sudo sed -i -E 's#^host[[:space:]]+all[[:space:]]+all[[:space:]]+::1/128[[:space:]]+(trust|ident|peer|md5)#host    all             all             ::1/128                 scram-sha-256#' $PG_HBA"
        fi
    fi

    run_with_status "Enabling PostgreSQL service" \
        sudo systemctl enable postgresql
    run_with_status "Starting PostgreSQL service" \
        sudo systemctl start postgresql
    # Apply pg_hba edits to an already-running cluster (re-deploys).
    run_quiet sudo systemctl reload postgresql || true

    run_with_status "Enabling Valkey service" \
        sudo systemctl enable valkey
    run_with_status "Starting Valkey service" \
        sudo systemctl start valkey

    # Detect Python version after installation
    PYTHON_VER=$(python3 --version 2>&1 | sed -n 's/Python \([0-9]*\.[0-9]*\).*/\1/p')

elif [ "$OS" = "macos" ]; then
    # macOS Homebrew package installation
    # Check if Homebrew is installed
    echo -ne "${DIM}${ARROW}${RESET} Checking for Homebrew... "
    # Homebrew installed but absent from PATH is the common non-login-shell /
    # ssh / CI case — probe the canonical prefixes before declaring it missing.
    if ! command -v brew &> /dev/null; then
        for BREW_PREFIX_CANDIDATE in /opt/homebrew/bin /usr/local/bin; do
            if [ -x "$BREW_PREFIX_CANDIDATE/brew" ]; then
                PATH="$BREW_PREFIX_CANDIDATE:$PATH"
                export PATH
                break
            fi
        done
    fi
    if ! command -v brew &> /dev/null; then
        echo -e "${ERROR}"
        print_error "Homebrew is not installed. Please install Homebrew first:"
        print_info "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    echo -e "${CHECKMARK}"

    # Non-interactive Homebrew. Installing a formula whose dependencies got
    # refreshed makes brew offer to upgrade UNRELATED installed dependents
    # behind a y/n prompt (observed live: installing MIRA's set proposed a
    # php upgrade via ca-certificates) — an unattended --config deploy hangs
    # there, and a MIRA install must never upgrade packages it does not own.
    export NONINTERACTIVE=1
    export HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK=1
    export HOMEBREW_NO_INSTALL_UPGRADE=1
    export HOMEBREW_NO_ENV_HINTS=1

    # Detect Python version to use (newest available, 3.12+ required)
    # Check for Python 3.12+ in descending order of preference
    PYTHON_VER=""
    for ver in 3.14 3.13 3.12; do
        if command -v python${ver} &> /dev/null; then
            PYTHON_VER="${ver}"
            break
        fi
    done
    
    # If no suitable version found, default to 3.12 for installation
    if [ -z "$PYTHON_VER" ]; then
        PYTHON_VER="3.12"
    fi

    # valkey and redis both install `redis-*` binaries — if redis formula is
    # present, `brew install valkey` aborts with a conflict error. Unlink redis
    # preemptively (the formula stays installed; just its symlinks are removed)
    # so the install can proceed. Users can `brew link redis` later if they
    # want both around.
    if brew list --formula 2>/dev/null | grep -q "^redis$"; then
        echo -ne "${DIM}${ARROW}${RESET} Unlinking redis formula (conflicts with valkey install)... "
        brew unlink redis > /dev/null 2>&1 || true
        echo -e "${CHECKMARK}"
    fi

    # Install only what is missing: `brew install` of an already-present
    # formula prints "Error: <f> is already installed" (exit 0 — verified on
    # brew 7.0.7), which reads as failure in a successful re-deploy log.
    # A missing formula in the list still installs normally.
    BREW_MISSING=""
    for formula in python@${PYTHON_VER} wget curl postgresql@17 pgvector valkey hashicorp/tap/vault; do
        brew list --formula "${formula##*/}" > /dev/null 2>&1 || BREW_MISSING="$BREW_MISSING $formula"
    done

    if [ "$LOUD_MODE" = true ]; then
        print_step "Updating Homebrew..."
        # Tolerate non-zero exit from `brew update` — a single broken third-party
        # tap (e.g. one whose remote has been deleted) makes brew update exit
        # non-zero, which `set -e` would otherwise turn into a silent deploy
        # abort. The subsequent `brew install` does its own index refresh, so
        # stale indices aren't actually a concern here.
        brew update || print_warning "brew update reported errors (continuing; usually a broken tap)"
        print_step "Adding HashiCorp tap..."
        brew tap hashicorp/tap
        if [ -n "$BREW_MISSING" ]; then
            print_step "Installing dependencies via Homebrew:$BREW_MISSING"
            brew install $BREW_MISSING
        else
            print_success "All Homebrew dependencies already installed"
        fi
    else
        (brew update > /dev/null 2>&1 || true) &
        show_progress $! "Updating Homebrew"

        (brew tap hashicorp/tap > /dev/null 2>&1) &
        show_progress $! "Adding HashiCorp tap"

        if [ -n "$BREW_MISSING" ]; then
            (brew install $BREW_MISSING > /dev/null 2>&1) &
            show_progress $! "Installing dependencies via Homebrew"
        fi
    fi

    # Homebrew clears the sudo credential timestamp (see ensure_sudo in
    # lib/services.sh) — re-establish elevation before the next sudo step
    # (Step 3's mkdir/chown under /opt/mira).
    ensure_sudo

    print_info "Playwright will install its own browser dependencies"
elif [ "$OS" = "linux" ]; then
    # Unsupported Linux distribution
    print_error "Unsupported Linux distribution: $DISTRO"
    print_info "Supported distributions:"
    print_info "  - Debian/Ubuntu and derivatives (apt)"
    print_info "  - Fedora/RHEL/Rocky/Alma/CentOS (dnf)"
    print_info "  - Arch and derivatives (pacman)"
    print_info ""
    print_info "For other distributions, install these dependencies manually:"
    print_info "  - Python 3.12+ with venv and dev headers"
    print_info "  - PostgreSQL 17 with pgvector extension"
    print_info "  - Valkey (Redis-compatible)"
    print_info "  - Build tools (gcc, make, etc.)"
    print_info "  - libpq development headers"
    print_info ""
    print_info "Then see docs/MANUAL_INSTALL.md for the remaining steps."
    exit 1
fi

print_success "System dependencies installed"

# Local LLM setup via llama.cpp (only for offline/local mode)
if [ "$CONFIG_OFFLINE_MODE" = "yes" ]; then
    # build_llama_cpp: no serves deployments whose llama-server runs on
    # another machine (remote offline mode via --config) — nothing to build
    # locally. Interview mode leaves the variable unset; the default matches
    # the historical build-always behavior.
    if [ "${CONFIG_BUILD_LLAMA_CPP:-yes}" = "no" ]; then
        print_header "Step 1b: llama.cpp Setup"
        print_info "Skipping llama.cpp build (build_llama_cpp: no — remote llama-server)"
        print_success "llama.cpp setup complete"
    else
    print_header "Step 1b: llama.cpp Setup"

    LLAMA_MODELS_DIR="/opt/mira/models"
    LLAMA_MAIN_PORT=8080
    LLAMA_SMALL_PORT=8081

    # --- Detect or build llama-server ---
    echo -ne "${DIM}${ARROW}${RESET} Checking for llama-server... "
    if command -v llama-server &> /dev/null; then
        echo -e "${CHECKMARK} ${DIM}(found in PATH)${RESET}"
    else
        echo -e "${DIM}(not found, building from source)${RESET}"

        # Check for required build tools
        MISSING_TOOLS=""
        for tool in cmake git g++; do
            if ! command -v $tool &> /dev/null; then
                MISSING_TOOLS="$MISSING_TOOLS $tool"
            fi
        done

        if [ -n "$MISSING_TOOLS" ]; then
            print_error "Missing build tools:$MISSING_TOOLS"
            print_info "Install them first, then re-run deploy."
            if [ "$OS" = "linux" ] && [ "$DISTRO" = "debian" ]; then
                print_info "  sudo apt install -y cmake git build-essential"
            elif [ "$OS" = "linux" ] && [ "$DISTRO" = "fedora" ]; then
                print_info "  sudo dnf install -y cmake git gcc-c++"
            elif [ "$OS" = "macos" ]; then
                print_info "  brew install cmake"
            fi
            exit 1
        fi

        # Detect CUDA availability
        USE_CUDA="OFF"
        if command -v nvcc &> /dev/null; then
            USE_CUDA="ON"
        fi

        # Clone and build llama.cpp
        BUILD_DIR="/tmp/llama.cpp-build"
        rm -rf "$BUILD_DIR"

        if [ "$LOUD_MODE" = true ]; then
            print_step "Cloning llama.cpp..."
            git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$BUILD_DIR"
            if [ "$USE_CUDA" = "ON" ]; then
                print_step "Building llama.cpp with CUDA support (this may take several minutes)..."
            else
                print_step "Building llama.cpp (CPU only, no CUDA detected)..."
            fi
            cd "$BUILD_DIR" && cmake -B build -DGGML_CUDA=$USE_CUDA -DLLAMA_SERVER=ON
            cd "$BUILD_DIR" && cmake --build build --config Release -j$(cpu_count)
            run_with_status "Installing llama.cpp" \
                sudo cmake --install build
        else
            (git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$BUILD_DIR" > /dev/null 2>&1 && \
             cd "$BUILD_DIR" && cmake -B build -DGGML_CUDA=$USE_CUDA -DLLAMA_SERVER=ON > /dev/null 2>&1 && \
             cmake --build build --config Release -j$(cpu_count) > /dev/null 2>&1 && \
             sudo cmake --install build > /dev/null 2>&1) &
            if [ "$USE_CUDA" = "ON" ]; then
                PROGRESS_MSG="Building llama.cpp from source (CUDA)"
            else
                PROGRESS_MSG="Building llama.cpp from source (CPU)"
            fi
            if show_progress $! "$PROGRESS_MSG"; then
                echo -e "${CHECKMARK}"
            else
                print_error "llama.cpp build failed"
                exit 1
            fi
        fi
        rm -rf "$BUILD_DIR"

        # Verify installation
        if ! command -v llama-server &> /dev/null; then
            print_error "llama-server not found after build — installation may have failed"
            exit 1
        fi
        echo -e "${CHECKMARK} ${DIM}(built & installed)${RESET}"
    fi

    # Create models directory (downloaded later by standalone script)
    run_with_status "Creating models directory" \
        sudo mkdir -p "$LLAMA_MODELS_DIR"
    run_quiet sudo chown -R $(whoami): "$LLAMA_MODELS_DIR"

    if [ "$CONFIG_LOCAL_MODEL_CHOICE" = "custom" ]; then
        print_info "Custom model mode — place your GGUF files in $LLAMA_MODELS_DIR/"
        if [ -n "${CONFIG_CUSTOM_GGUF:-}" ]; then
            print_info "User-specified model: $CONFIG_CUSTOM_GGUF"
        fi
    fi

    print_success "llama.cpp setup complete"
    fi
fi
