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

psql -U $SUPERUSER -h localhost -d mira_service -v ON_ERROR_STOP=1 -f "${SCRIPT_DIR}/mira_service_schema.sql" > /dev/null 2>&1

if [ $? -eq 0 ]; then
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
echo "   service_url: postgresql://mira_dbuser:<mira_dbuser password>@localhost:5432/mira_service"
echo "   username: mira_admin"
echo "   password: <mira_admin password>"
echo ""
echo "2. Update application config to use mira_service"
echo "3. Start the application: python main.py"
echo "==================================================================="
