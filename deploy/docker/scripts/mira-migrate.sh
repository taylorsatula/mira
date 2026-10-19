#!/bin/bash
# deploy/docker/scripts/mira-migrate.sh — the container's database step run,
# every start, as the s6 oneshot mira-migrate (before the mira longrun).
#
# Waits for an unsealed Vault and a listening PostgreSQL, 60 s each, exiting 3
# when either never arrives — a timeout that fails the oneshot rather than a
# migration failure, and it triggers no restore. Then exports the AppRole the
# runner authenticates with and runs `scripts.migrate migrate` as root, so the
# runner's superuser seam is `sudo -u postgres`, writing this run's backups
# under /opt/mira/backups.
#
# The exit code is the migration's. Only a migration failure (exit 1) with
# this run's manifest on disk triggers a restore of that backup, and the
# restore's own exit code decides whether the closing message says the data
# was put back or was not. Any nonzero exit fails the oneshot, so mira never
# starts on a half-upgraded database; the operator reruns the previous image
# on the same volumes.
set -uo pipefail

export VAULT_ADDR=http://127.0.0.1:8200
export PATH="/usr/lib/postgresql/17/bin:${PATH}"

for i in $(seq 1 60); do
    [ "$(curl -sf http://127.0.0.1:8200/v1/sys/health | jq -r '.sealed' 2>/dev/null)" = "false" ] && break
    [ "$i" -eq 60 ] && { echo "mira-migrate: Vault not unsealed after 60 s"; exit 3; }
    sleep 1
done
for i in $(seq 1 60); do
    pg_isready -q -h 127.0.0.1 && break
    [ "$i" -eq 60 ] && { echo "mira-migrate: PostgreSQL not ready after 60 s"; exit 3; }
    sleep 1
done
export VAULT_ROLE_ID="$(cat /opt/vault/role-id.txt)"
export VAULT_SECRET_ID="$(cat /opt/vault/secret-id.txt)"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
cd /opt/mira/app
rc=0
PYTHONPATH=/opt/mira/app /opt/mira/venv/bin/python3 -P -m scripts.migrate migrate \
    --stamp "$STAMP" --backups-dir /opt/mira/backups --env-file /opt/mira/systemone.env || rc=$?
if [ "$rc" -eq 1 ] && [ -f "/opt/mira/backups/${STAMP}/manifest.json" ]; then
    restore_rc=0
    PYTHONPATH=/opt/mira/app /opt/mira/venv/bin/python3 -P -m scripts.migrate restore "$STAMP" \
        --backups-dir /opt/mira/backups || restore_rc=$?
    if [ "$restore_rc" -eq 0 ]; then
        echo "MIRA did not start: the database upgrade failed and the data was put back. Run the previous image on the same volumes."
    else
        echo "MIRA did not start: the database upgrade failed and the data was NOT put back. Do what the message above says about your data."
    fi
fi
exit "$rc"
