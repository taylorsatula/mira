#!/usr/bin/env python3
"""
Test runner for secure_auth PostgreSQL tests.

This script ensures the PostgreSQL test environment is properly configured
before running the test suite.
"""

import os
import sys
import subprocess
import secrets


def check_postgresql_env():
    """Check if PostgreSQL environment variables are set."""
    required_vars = [
        'AUTH_DB_HOST',
        'AUTH_DB_PORT',
        'AUTH_DB_NAME',
        'AUTH_DB_USER',
        'AUTH_DB_PASSWORD'
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables or run: source setup_test_env.sh")
        return False
    
    return True


def generate_master_key():
    """Generate AUTH_MASTER_KEY if not set."""
    if not os.environ.get('AUTH_MASTER_KEY'):
        master_key = secrets.token_hex(32)
        os.environ['AUTH_MASTER_KEY'] = master_key
        print("✓ Generated AUTH_MASTER_KEY")


def test_postgresql_connection():
    """Test PostgreSQL connection."""
    import psycopg2
    
    try:
        conn = psycopg2.connect(
            host=os.environ['AUTH_DB_HOST'],
            port=os.environ['AUTH_DB_PORT'],
            database='postgres',  # Connect to default DB first
            user=os.environ['AUTH_DB_USER'],
            password=os.environ['AUTH_DB_PASSWORD']
        )
        conn.close()
        print("✓ PostgreSQL connection successful")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False


def run_tests(args):
    """Run pytest with provided arguments."""
    # Default to current directory if no args provided
    if not args:
        args = ['tests/test_secure_auth/']
    
    # Add pytest to the command
    cmd = ['pytest'] + args
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run pytest
    result = subprocess.run(cmd, env=os.environ.copy())
    return result.returncode


def main():
    """Main entry point."""
    print("🔒 Secure Auth PostgreSQL Test Runner")
    print("=" * 60)
    
    # Check environment
    if not check_postgresql_env():
        return 1
    
    # Generate master key if needed
    generate_master_key()
    
    # Test connection
    if not test_postgresql_connection():
        return 1
    
    # Get test arguments (everything after script name)
    test_args = sys.argv[1:]
    
    # Run tests
    return run_tests(test_args)


if __name__ == '__main__':
    sys.exit(main())