#!/bin/bash
# Setup script for PostgreSQL test environment

echo "Setting up PostgreSQL test environment for secure_auth tests..."

# Default values
DEFAULT_HOST="localhost"
DEFAULT_PORT="5432"
DEFAULT_DB="auth_db"
DEFAULT_USER="postgres"

# Prompt for database configuration
read -p "PostgreSQL host [$DEFAULT_HOST]: " DB_HOST
DB_HOST=${DB_HOST:-$DEFAULT_HOST}

read -p "PostgreSQL port [$DEFAULT_PORT]: " DB_PORT
DB_PORT=${DB_PORT:-$DEFAULT_PORT}

read -p "Database name [$DEFAULT_DB]: " DB_NAME
DB_NAME=${DB_NAME:-$DEFAULT_DB}

read -p "Database user [$DEFAULT_USER]: " DB_USER
DB_USER=${DB_USER:-$DEFAULT_USER}

read -s -p "Database password: " DB_PASSWORD
echo

# Export environment variables
export AUTH_DB_HOST="$DB_HOST"
export AUTH_DB_PORT="$DB_PORT"
export AUTH_DB_NAME="$DB_NAME"
export AUTH_DB_USER="$DB_USER"
export AUTH_DB_PASSWORD="$DB_PASSWORD"

# Generate master key if not set
if [ -z "$AUTH_MASTER_KEY" ]; then
    export AUTH_MASTER_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
    echo "Generated AUTH_MASTER_KEY"
fi

# Test database connection
echo "Testing PostgreSQL connection..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "SELECT version();" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ PostgreSQL connection successful"
    
    # Create main database if it doesn't exist
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || \
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME"
    
    echo "✓ Database '$DB_NAME' is ready"
    echo
    echo "Environment variables set:"
    echo "  AUTH_DB_HOST=$AUTH_DB_HOST"
    echo "  AUTH_DB_PORT=$AUTH_DB_PORT"
    echo "  AUTH_DB_NAME=$AUTH_DB_NAME"
    echo "  AUTH_DB_USER=$AUTH_DB_USER"
    echo "  AUTH_DB_PASSWORD=***"
    echo "  AUTH_MASTER_KEY=***"
    echo
    echo "You can now run tests with: pytest tests/test_secure_auth/"
else
    echo "✗ Failed to connect to PostgreSQL"
    echo "Please check your PostgreSQL installation and credentials"
    exit 1
fi