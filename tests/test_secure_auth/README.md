# Secure Auth PostgreSQL Test Suite

This test suite tests the secure authentication system with PostgreSQL, ensuring proper testing of PostgreSQL-specific features like JSONB, UUID, INET, and ARRAY types.

## Prerequisites

1. **PostgreSQL Installation**: Ensure PostgreSQL is installed and running on your system.
   ```bash
   # macOS
   brew install postgresql
   brew services start postgresql
   
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   sudo systemctl start postgresql
   
   # Check PostgreSQL is running
   psql --version
   ```

2. **Python Dependencies**: Install test requirements
   ```bash
   pip install -r test_requirements.txt
   ```

## Configuration

The test suite uses environment variables for PostgreSQL connection. Set these before running tests:

```bash
# Required environment variables
export AUTH_DB_HOST=localhost
export AUTH_DB_PORT=5432
export AUTH_DB_NAME=auth_db
export AUTH_DB_USER=postgres
export AUTH_DB_PASSWORD=your_password

# Optional: Set a master key for testing (auto-generated if not set)
export AUTH_MASTER_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
```

### Alternative: Use .env file
Create a `.env` file in the project root:
```
AUTH_DB_HOST=localhost
AUTH_DB_PORT=5432
AUTH_DB_NAME=auth_db
AUTH_DB_USER=postgres
AUTH_DB_PASSWORD=your_password
```

## Database Setup

The test suite automatically creates a test database (`{AUTH_DB_NAME}_test`) and enables required extensions:
- `uuid-ossp` for UUID generation

No manual database creation is needed - the test fixtures handle everything.

## Running Tests

### Run all tests
```bash
pytest tests/test_secure_auth/
```

### Run specific test file
```bash
pytest tests/test_secure_auth/test_auth_service.py
```

### Run with verbose output
```bash
pytest tests/test_secure_auth/ -v
```

### Run with coverage
```bash
pytest tests/test_secure_auth/ --cov=secure_auth --cov-report=html
```

### Run tests in parallel
```bash
pytest tests/test_secure_auth/ -n auto
```

## Test Architecture

### Shared Fixtures (conftest.py)

The `conftest.py` file provides shared PostgreSQL fixtures:

- **`test_database_url`**: Constructs PostgreSQL URL for test database
- **`test_database`**: Creates/drops test database with extensions
- **`db_engine`**: Provides SQLAlchemy engine for the test database
- **`db_session`**: Provides isolated database session with automatic rollback
- **`test_user`**, **`admin_user`**, etc.: Pre-configured test users
- **`auth_master_key`**: Test master key for cryptographic operations
- **`setup_test_environment`**: Auto-configures test environment variables

### Test Isolation

Each test runs in an isolated transaction that is rolled back after the test completes, ensuring:
- No test data persists between tests
- Tests can run in any order
- Parallel test execution is safe

### PostgreSQL-Specific Testing

The tests validate PostgreSQL-specific features:
- **JSONB columns**: `user_metadata`, `security_flags`, `forensic_data`
- **UUID primary keys**: All models use UUID with `gen_random_uuid()`
- **INET type**: IP address storage with proper validation
- **ARRAY types**: `trusted_device_ids`, `known_ip_addresses`
- **PostgreSQL indexes**: Composite indexes for performance

## Troubleshooting

### Database Connection Errors
```
ValueError: Missing required environment variables: AUTH_DB_HOST, AUTH_DB_PORT...
```
**Solution**: Ensure all AUTH_DB_* environment variables are set.

### Permission Denied
```
psql: FATAL: password authentication failed for user "postgres"
```
**Solution**: Check AUTH_DB_PASSWORD matches your PostgreSQL user password.

### Database Already Exists
The test suite automatically handles existing test databases by dropping and recreating them.

### Extension Not Found
```
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"
```
**Solution**: Ensure you have PostgreSQL contrib package installed:
```bash
# Ubuntu/Debian
sudo apt-get install postgresql-contrib

# macOS (included with brew install)
```

## CI/CD Integration

For CI environments, use PostgreSQL service containers:

### GitHub Actions Example
```yaml
services:
  postgres:
    image: postgres:14
    env:
      POSTGRES_PASSWORD: postgres
    options: >-
      --health-cmd pg_isready
      --health-interval 10s
      --health-timeout 5s
      --health-retries 5
    ports:
      - 5432:5432

env:
  AUTH_DB_HOST: localhost
  AUTH_DB_PORT: 5432
  AUTH_DB_NAME: auth_db
  AUTH_DB_USER: postgres
  AUTH_DB_PASSWORD: postgres
```

### Docker Compose Example
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: auth_db
    ports:
      - "5432:5432"
  
  tests:
    build: .
    environment:
      AUTH_DB_HOST: postgres
      AUTH_DB_PORT: 5432
      AUTH_DB_NAME: auth_db
      AUTH_DB_USER: postgres
      AUTH_DB_PASSWORD: postgres
    depends_on:
      - postgres
    command: pytest tests/test_secure_auth/
```

## Development Tips

1. **Use transaction fixtures**: The `db_session` fixture automatically rolls back changes
2. **Don't mock security functions**: Test real cryptographic operations
3. **Test PostgreSQL features**: Validate JSONB queries, array operations, etc.
4. **Check indexes**: Ensure composite indexes are used for performance
5. **Validate constraints**: Test unique constraints, foreign keys, etc.

## Security Testing Focus

These tests focus on security validation:
- **Attack simulation**: Brute force, SQL injection, timing attacks
- **Cryptographic properties**: Token entropy, salt uniqueness, hash security
- **Rate limiting**: Distributed attacks, escalation, persistence
- **Session security**: Context binding, hijacking prevention
- **Input validation**: Sanitization, length limits, encoding