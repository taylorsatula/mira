# MIRA Application Image
# Thin layer on top of mira-base with just the application code.
# Rebuilds in seconds after code changes.
#
# Prerequisites: docker build -t mira-base:latest -f deploy/docker/Dockerfile.base .
# Build:         docker build -t mira:latest -f deploy/docker/Dockerfile .
# Run:           docker run -it -v mira-data:/opt/vault -p 1993:1993 mira:latest

ARG BASE_IMAGE=mira-base:latest
FROM ${BASE_IMAGE}

# The source commit this image was built from: pass
# --build-arg MIRA_REVISION=$(git rev-parse HEAD); an image built from a
# release asset falls back to that asset's RELEASE_COMMIT below.
ARG MIRA_REVISION=""

# Labels
LABEL org.opencontainers.image.revision="${MIRA_REVISION}"
LABEL org.opencontainers.image.title="MIRA"
LABEL org.opencontainers.image.description="MIRA - Memory-Integrated Reasoning Assistant"
LABEL org.opencontainers.image.source="https://github.com/taylorsatula/mira"

# Copy application code
COPY --chown=mira:mira . /opt/mira/app/

# Build record for the running image.
RUN rev="${MIRA_REVISION:-$(cat /opt/mira/app/RELEASE_COMMIT 2>/dev/null || true)}" \
    && printf 'revision=%s\nbuilt_at=%s\n' "${rev:-unrecorded}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > /opt/mira/image-build.txt

# Copy s6 service definitions
COPY deploy/docker/s6-rc.d/ /etc/s6-overlay/s6-rc.d/

# Copy configuration files
COPY deploy/docker/config/vault.hcl /opt/vault/config/vault.hcl
COPY deploy/docker/config/valkey.conf /etc/valkey/valkey.conf

# Copy scripts
COPY deploy/docker/scripts/init-mira.sh /init-mira.sh
COPY deploy/docker/scripts/container-setup.sh /opt/mira/container-setup.sh
COPY deploy/docker/scripts/healthcheck.sh /opt/mira/healthcheck.sh
COPY deploy/docker/scripts/vault-unseal.sh /opt/mira/app/deploy/docker/scripts/vault-unseal.sh
RUN chmod +x /init-mira.sh /opt/mira/container-setup.sh /opt/mira/healthcheck.sh \
    /opt/mira/app/deploy/docker/scripts/vault-unseal.sh \
    && chmod +x /etc/s6-overlay/s6-rc.d/mira/run \
                /etc/s6-overlay/s6-rc.d/vault/run \
                /etc/s6-overlay/s6-rc.d/valkey/run \
                /etc/s6-overlay/s6-rc.d/postgresql/run \
                /etc/s6-overlay/s6-rc.d/vault-init/up

# Expose ports
EXPOSE 1993

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD /opt/mira/healthcheck.sh

# Volumes for persistent data. /opt/mira/app/data holds per-user storage
# (userdata.db, artifacts, imagegen outputs) anchored by
# utils/userdata_manager.py:base_dir — without a volume, container recreation
# destroys it. init-mira.sh chowns it to mira on every start (volume
# mountpoints come up root-owned and the app writes as mira). /opt/mira/backups
# holds the pre-upgrade backups the step runner takes; a bare VOLUME is
# anonymous, so it persists across container recreation only when run with a
# named mount (-v mira-backups:/opt/mira/backups, README).
VOLUME ["/var/lib/postgresql/17/main", "/opt/vault", "/var/lib/valkey", "/home/mira/.cache/huggingface", "/opt/mira/app/data", "/opt/mira/backups"]

# Use our custom entrypoint that handles first-boot, then hands off to s6
ENTRYPOINT ["/init-mira.sh"]
