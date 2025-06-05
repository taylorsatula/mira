"""
Shared test fixtures for secure_auth tests using PostgreSQL.

This module provides database fixtures that use PostgreSQL instead of SQLite,
enabling proper testing of PostgreSQL-specific features like JSONB, UUID, INET, and ARRAY types.
"""

import os
import pytest
import secrets
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from secure_auth.models import Base, User, get_database_url


@pytest.fixture(scope="session")
def test_database_url():
    """
    Provide PostgreSQL test database URL.
    
    Uses AUTH_DB_* environment variables with a test database suffix.
    """
    # Get base database configuration from environment
    host = os.environ.get('AUTH_DB_HOST', 'localhost')
    port = os.environ.get('AUTH_DB_PORT', '5432')
    base_name = os.environ.get('AUTH_DB_NAME', 'auth_db')
    username = os.environ.get('AUTH_DB_USER', 'postgres')
    password = os.environ.get('AUTH_DB_PASSWORD', '')
    
    # Create test database name
    test_db_name = f"{base_name}_test"
    
    # Return test database URL
    return f"postgresql://{username}:{password}@{host}:{port}/{test_db_name}"


@pytest.fixture(scope="session")
def test_database(test_database_url):
    """
    Create and configure test database for the entire test session.
    
    This fixture:
    1. Creates a test database
    2. Enables required PostgreSQL extensions
    3. Creates all tables
    4. Tears down the database after tests complete
    """
    # Parse database name from URL
    db_name = test_database_url.split('/')[-1]
    base_url = test_database_url.rsplit('/', 1)[0]
    
    # Connect to postgres database to create test database
    admin_engine = create_engine(f"{base_url}/postgres", isolation_level='AUTOCOMMIT')
    
    try:
        # Drop test database if it exists
        with admin_engine.connect() as conn:
            # Check if database exists
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": db_name}
            )
            if result.scalar():
                # Terminate existing connections
                conn.execute(
                    text("""
                        SELECT pg_terminate_backend(pg_stat_activity.pid)
                        FROM pg_stat_activity
                        WHERE pg_stat_activity.datname = :dbname
                        AND pid <> pg_backend_pid()
                    """),
                    {"dbname": db_name}
                )
                # Drop the database
                conn.execute(text(f'DROP DATABASE "{db_name}"'))
            
            # Create test database
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    finally:
        admin_engine.dispose()
    
    # Connect to test database and set up
    engine = create_engine(test_database_url, echo=False)
    
    try:
        # Enable required extensions
        with engine.connect() as conn:
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            conn.commit()
        
        # Create all tables
        Base.metadata.create_all(engine)
        
        yield test_database_url
        
    finally:
        # Cleanup
        engine.dispose()
        
        # Drop test database
        admin_engine = create_engine(f"{base_url}/postgres", isolation_level='AUTOCOMMIT')
        try:
            with admin_engine.connect() as conn:
                # Terminate connections
                conn.execute(
                    text("""
                        SELECT pg_terminate_backend(pg_stat_activity.pid)
                        FROM pg_stat_activity
                        WHERE pg_stat_activity.datname = :dbname
                        AND pid <> pg_backend_pid()
                    """),
                    {"dbname": db_name}
                )
                # Drop database
                conn.execute(text(f'DROP DATABASE "{db_name}"'))
        finally:
            admin_engine.dispose()


@pytest.fixture
def db_engine(test_database):
    """Provide a database engine for the test database."""
    engine = create_engine(test_database, echo=False)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """
    Provide a database session with automatic rollback after each test.
    
    This ensures test isolation by rolling back any changes made during the test.
    """
    Session = sessionmaker(bind=db_engine)
    session = Session()
    
    # Begin a transaction
    session.begin()
    
    yield session
    
    # Rollback the transaction
    session.rollback()
    session.close()


@pytest.fixture
def test_user(db_session):
    """Create a test user in the database."""
    user = User(
        email="test@example.com",
        tenant_id="default",
        is_active=True,
        email_verified=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def admin_user(db_session):
    """Create an admin test user in the database."""
    user = User(
        email="admin@example.com",
        tenant_id="default",
        is_active=True,
        is_admin=True,
        email_verified=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def inactive_user(db_session):
    """Create an inactive test user in the database."""
    user = User(
        email="inactive@example.com",
        tenant_id="default",
        is_active=False,
        email_verified=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def locked_user(db_session):
    """Create a locked test user in the database."""
    user = User(
        email="locked@example.com",
        tenant_id="default",
        is_active=True,
        email_verified=True,
        account_locked_until=datetime.utcnow() + timedelta(hours=1)
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def multi_tenant_users(db_session):
    """Create users in different tenants for testing isolation."""
    users = []
    
    # Create users in different tenants
    for tenant in ["tenant_a", "tenant_b", "tenant_c"]:
        user = User(
            email=f"user@{tenant}.com",
            tenant_id=tenant,
            is_active=True,
            email_verified=True
        )
        db_session.add(user)
        users.append(user)
    
    db_session.commit()
    
    # Refresh all users
    for user in users:
        db_session.refresh(user)
    
    return users


@pytest.fixture
def auth_master_key():
    """Provide a test master key for authentication services."""
    key = secrets.token_hex(32)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('AUTH_MASTER_KEY', key)
        yield key


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """
    Automatically set up test environment variables.
    
    This fixture runs automatically for all tests and ensures
    required environment variables are set for testing.
    """
    # Set default test environment variables if not already set
    test_env = {
        'AUTH_DB_HOST': os.environ.get('AUTH_DB_HOST', 'localhost'),
        'AUTH_DB_PORT': os.environ.get('AUTH_DB_PORT', '5432'),
        'AUTH_DB_NAME': os.environ.get('AUTH_DB_NAME', 'auth_db'),
        'AUTH_DB_USER': os.environ.get('AUTH_DB_USER', 'postgres'),
        'AUTH_DB_PASSWORD': os.environ.get('AUTH_DB_PASSWORD', 'postgres'),
        'AUTH_MASTER_KEY': os.environ.get('AUTH_MASTER_KEY', secrets.token_hex(32))
    }
    
    for key, value in test_env.items():
        if key not in os.environ:
            monkeypatch.setenv(key, value)