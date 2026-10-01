#!/bin/bash
# MIRA Database Deployment Script
# Deploys mira_service database with unified schema
# For fresh PostgreSQL installations

set -e  # Exit on any error

# Resolve the schema next to this script so the tool works from any directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==================================================================="
echo "=== MIRA Database Deployment                                    ==="
echo "==================================================================="
echo ""

# =====================================================================
# STEP 1: Find PostgreSQL superuser
# =====================================================================

echo "Step 1: Detecting PostgreSQL superuser..."

# Check if current user is a superuser
CURRENT_USER=$(whoami)
IS_SUPERUSER=$(psql -U $CURRENT_USER -h localhost -d postgres -tAc "SELECT COUNT(*) FROM pg_roles WHERE rolname = '$CURRENT_USER' AND rolsuper = true" 2>/dev/null || echo "0")

if [ "$IS_SUPERUSER" = "1" ]; then
    SUPERUSER=$CURRENT_USER
    echo "✓ Using current user as superuser: $SUPERUSER"
else
    # Try to find a superuser
    SUPERUSER=$(psql -U $CURRENT_USER -h localhost -d postgres -tAc "SELECT rolname FROM pg_roles WHERE rolsuper = true LIMIT 1" 2>/dev/null || echo "")

    if [ -z "$SUPERUSER" ]; then
        echo "✗ Error: No PostgreSQL superuser found"
        echo "Please run as a PostgreSQL superuser or specify one manually"
        exit 1
    fi

    echo "✓ Using detected superuser: $SUPERUSER"
fi

# =====================================================================
# STEP 2: Check if mira_service already exists
# =====================================================================

echo ""
echo "Step 2: Checking for existing mira_service database..."

if psql -U $SUPERUSER -h localhost -lqt | cut -d \| -f 1 | grep -qw mira_service; then
    echo "✗ Error: mira_service database already exists"
    echo "Please drop it first: dropdb -U $SUPERUSER mira_service"
    exit 1
else
    echo "✓ No existing mira_service database found"
fi

# =====================================================================
# STEP 3: Provision roles and database, then deploy clean schema
# =====================================================================

echo ""
echo "Step 3: Provisioning roles and database..."

# The schema is a pure DDL contract: it creates no roles and no database, and
# refuses to run against a database that already has tables. Provision both
# here, then apply the schema to the empty mira_service database.
psql -U $SUPERUSER -h localhost -d postgres -v ON_ERROR_STOP=1 -c "\
DO \$roles\$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mira_admin') THEN
        CREATE ROLE mira_admin LOGIN PASSWORD 'changethisifdeployingpwd' BYPASSRLS;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mira_dbuser') THEN
        CREATE ROLE mira_dbuser LOGIN PASSWORD 'changethisifdeployingpwd';
    END IF;
END \$roles\$;"

psql -U $SUPERUSER -h localhost -d postgres -v ON_ERROR_STOP=1 -c "CREATE DATABASE mira_service OWNER mira_admin"

echo ""
echo "Step 4: Deploying mira_service schema..."

# The schema sizes its vector columns from the embedding model. Local by
# default; for a remote OpenAI-compatible endpoint set
# MIRA_EMBEDDING_PROVIDER=remote, MIRA_EMBEDDING_ENDPOINT, MIRA_EMBEDDING_MODEL
# (the bearer token is read from the terminal). MIRA_PYTHON must have MIRA's
# requirements installed (default: the checkout's venv, else python3).
source "${SCRIPT_DIR}/lib/embedding_config.sh"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -z "${MIRA_PYTHON:-}" ]; then
    if [ -x "${APP_DIR}/venv/bin/python3" ]; then MIRA_PYTHON="${APP_DIR}/venv/bin/python3"; else MIRA_PYTHON="python3"; fi
fi
EMBEDDING_PROVIDER="${MIRA_EMBEDDING_PROVIDER:-local}"
EMBEDDING_API_KEY=""
if [ "$EMBEDDING_PROVIDER" = "remote" ]; then
    read -r -s -p "Embedding endpoint bearer token (Enter for none): " EMBEDDING_API_KEY
    echo ""
fi
resolve_embedding_schema_args "$MIRA_PYTHON" "$APP_DIR" "$EMBEDDING_PROVIDER" \
    "${MIRA_EMBEDDING_ENDPOINT:-}" "${MIRA_EMBEDDING_MODEL:-}" "$EMBEDDING_API_KEY"
echo "✓ Embedding model ${EMBEDDING_MODEL}, ${EMBEDDING_DIMENSIONS} dimensions"
if [ -n "$EMBEDDING_API_KEY" ]; then
    echo "  Store the token in Vault as secret/mira/api_keys ${EMBEDDING_VAULT_KEY_NAME}=<token>"
fi

# (40dz) The injection screen's System One client (provider=remote by
# default) reads its bearer token from Vault as systemone_key — collect it
# here like the embedding token so the operator is not left to guess the
# field name. A local (self-hosted) endpoint takes no token: set
# MIRA_SYSTEMONE_PROVIDER=local.
SYSTEMONE_API_KEY=""
if [ "${MIRA_SYSTEMONE_PROVIDER:-remote}" = "remote" ]; then
    read -r -s -p "Injection-screen System One bearer token (Enter for none): " SYSTEMONE_API_KEY
    echo ""
fi
if [ -n "$SYSTEMONE_API_KEY" ]; then
    echo "  Store the token in Vault as secret/mira/api_keys systemone_key=<token>"
fi

SCHEMA_STATUS=0
psql -U $SUPERUSER -h localhost -d mira_service -v ON_ERROR_STOP=1 "${EMBEDDING_SCHEMA_ARGS[@]}" -f "${SCRIPT_DIR}/mira_service_schema.sql" > /dev/null 2>&1 || SCHEMA_STATUS=$?

if [ $SCHEMA_STATUS -eq 0 ]; then
    echo "✓ Schema deployed successfully"
else
    echo "✗ Schema deployment failed"
    exit 1
fi

# =====================================================================
# STEP 5: Verify deployment
# =====================================================================

echo ""
echo "Step 5: Verifying deployment..."

# Check database exists
DB_EXISTS=$(psql -U $SUPERUSER -h localhost -lqt | cut -d \| -f 1 | grep -w mira_service | wc -l)
if [ "$DB_EXISTS" -eq "1" ]; then
    echo "✓ Database mira_service created"
else
    echo "✗ Database mira_service not found"
    exit 1
fi

# Check roles
echo ""
echo "Roles provisioned by deployment tooling:"
psql -U $SUPERUSER -h localhost -d postgres -c "
SELECT
    rolname,
    rolsuper as superuser,
    rolcreaterole as create_role,
    rolcreatedb as create_db,
    CASE
        WHEN rolname = 'mira_admin' THEN 'Database owner (migrations, schema)'
        WHEN rolname = 'mira_dbuser' THEN 'Application runtime (data only)'
    END as purpose
FROM pg_roles
WHERE rolname IN ('mira_admin', 'mira_dbuser')
ORDER BY rolname;
" 2>/dev/null

# Check tables
echo ""
echo "Tables created:"
psql -U $SUPERUSER -h localhost -d mira_service -c "
SELECT schemaname, tablename
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;
" 2>/dev/null | head -20

# Count rows (should all be 0 on fresh install)
echo ""
echo "Table row counts (should be 0):"
psql -U $SUPERUSER -h localhost -d mira_service -c "
SELECT 'users' as table, COUNT(*) FROM users
UNION ALL SELECT 'continuums', COUNT(*) FROM continuums
UNION ALL SELECT 'messages', COUNT(*) FROM messages
UNION ALL SELECT 'memories', COUNT(*) FROM memories
UNION ALL SELECT 'entities', COUNT(*) FROM entities;
" 2>/dev/null

echo ""
echo "==================================================================="
echo "✓ Database deployment complete!"
echo ""
echo "Next steps:"
echo "1. Both roles were created with the sentinel password 'changethisifdeployingpwd'."
echo "   Set real passwords (ALTER ROLE mira_admin/mira_dbuser WITH PASSWORD ...), then add them to Vault (mira/database):"
echo "   admin_url: postgresql://mira_admin:<mira_admin password>@localhost:5432/mira_service"
echo "   service_url: postgresql://mira_dbuser:<mira_dbuser password>@localhost:5432/mira_service"
echo "   username: mira_dbuser"
echo "   password: <mira_dbuser password>"
echo ""
echo "2. Update application config to use mira_service"
echo "3. Start the application: python main.py"
echo ""
echo "4. The injection screen's System One model is app config only — no schema"
echo "   artifact and no Vault step here. Enable it with MIRA_INJECTION_SCREEN_ENABLED"
echo "   (+ MIRA_SYSTEMONE_PROVIDER/ENDPOINT/MODEL); for a remote provider seed"
echo "   Vault secret/mira/api_keys systemone_key=<token> manually."
echo "==================================================================="
