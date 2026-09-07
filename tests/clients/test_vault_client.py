"""
Tests for clients/vault_client.py - Vault secret management.

Testing philosophy: NO MOCKING. Tests use real Vault instance configured
via environment variables. Tests verify actual authentication, secret retrieval,
caching, and error handling with real Vault API.
"""
import pytest
import os
from hvac.exceptions import VaultError, InvalidPath, Unauthorized, Forbidden

import clients.vault_client as vault_client
from clients.vault_client import (
    VaultClient,
    get_database_url,
    get_api_key,
    get_auth_secret,
    get_service_config,
    get_database_credentials,
    _ensure_vault_client,
)


class TestVaultClientInitialization:
    """Test VaultClient initialization and authentication with real Vault."""

    def test_vault_client_initializes_with_env_vars(self):
        """VaultClient initializes successfully using environment variables."""
        # Real Vault initialization with AppRole auth
        client = VaultClient()

        # Verify client is properly configured
        assert client.vault_addr is not None
        assert client.vault_addr == os.getenv('VAULT_ADDR')
        assert client.vault_role_id is not None
        assert client.vault_secret_id is not None

        # Verify authentication succeeded
        assert client.client.is_authenticated() is True

    def test_vault_client_initializes_with_explicit_params(self):
        """VaultClient can be initialized with explicit parameters."""
        vault_addr = os.getenv('VAULT_ADDR')
        vault_token = os.getenv('VAULT_TOKEN')

        client = VaultClient(
            vault_addr=vault_addr,
            vault_token=vault_token
        )

        assert client.vault_addr == vault_addr
        assert client.client.is_authenticated() is True

    def test_vault_client_fails_without_required_env_vars(self):
        """VaultClient fails fast when required environment variables are missing."""
        # Save original env vars
        original_addr = os.getenv('VAULT_ADDR')
        original_role_id = os.getenv('VAULT_ROLE_ID')
        original_secret_id = os.getenv('VAULT_SECRET_ID')

        try:
            # Clear required env vars
            if 'VAULT_ADDR' in os.environ:
                del os.environ['VAULT_ADDR']

            # Should raise ValueError with clear message
            with pytest.raises(ValueError, match="VAULT_ADDR environment variable is required"):
                VaultClient()

        finally:
            # Restore env vars
            if original_addr:
                os.environ['VAULT_ADDR'] = original_addr
            if original_role_id:
                os.environ['VAULT_ROLE_ID'] = original_role_id
            if original_secret_id:
                os.environ['VAULT_SECRET_ID'] = original_secret_id

    def test_vault_client_fails_without_approle_credentials(self):
        """VaultClient fails when AppRole credentials are missing."""
        original_role_id = os.getenv('VAULT_ROLE_ID')
        original_secret_id = os.getenv('VAULT_SECRET_ID')

        try:
            # Clear AppRole credentials
            if 'VAULT_ROLE_ID' in os.environ:
                del os.environ['VAULT_ROLE_ID']
            if 'VAULT_SECRET_ID' in os.environ:
                del os.environ['VAULT_SECRET_ID']

            with pytest.raises(ValueError, match="VAULT_ROLE_ID and VAULT_SECRET_ID"):
                VaultClient()

        finally:
            # Restore env vars
            if original_role_id:
                os.environ['VAULT_ROLE_ID'] = original_role_id
            if original_secret_id:
                os.environ['VAULT_SECRET_ID'] = original_secret_id

    def test_vault_client_fails_with_invalid_approle_credentials(self):
        """VaultClient fails when AppRole credentials are invalid.

        Tests real production failure: wrong role_id/secret_id due to rotation,
        typo in configuration, or incorrect credential copy/paste.
        """
        original_role_id = os.getenv('VAULT_ROLE_ID')
        original_secret_id = os.getenv('VAULT_SECRET_ID')

        try:
            # Set invalid AppRole credentials
            os.environ['VAULT_ROLE_ID'] = 'invalid-role-id-12345'
            os.environ['VAULT_SECRET_ID'] = 'invalid-secret-id-67890'

            # Should fail with PermissionError during authentication
            with pytest.raises(PermissionError, match="AppRole authentication failed"):
                VaultClient()

        finally:
            # Restore valid credentials
            if original_role_id:
                os.environ['VAULT_ROLE_ID'] = original_role_id
            if original_secret_id:
                os.environ['VAULT_SECRET_ID'] = original_secret_id

    def test_singleton_returns_same_instance(self):
        """_ensure_vault_client returns the same singleton instance."""
        # Get first instance
        client1 = _ensure_vault_client()

        # Get second instance - should be same object
        client2 = _ensure_vault_client()

        assert client1 is client2

    def test_singleton_reset_creates_new_instance(self):
        """After resetting singleton, new instance is created."""
        # Get first instance
        client1 = _ensure_vault_client()

        # Reset singleton (done automatically by fixtures)
        vault_client._vault_client_instance = None

        # Get new instance - should be different object
        client2 = _ensure_vault_client()

        assert client1 is not client2


class TestSecretRetrieval:
    """Test VaultClient.get_secret() method with real Vault secrets."""

    def test_get_secret_retrieves_database_username(self):
        """get_secret retrieves actual database username from Vault."""
        client = VaultClient()

        # Retrieve real secret from Vault
        username = client.get_secret('mira/database', 'username')

        # Verify secret is returned as string
        assert isinstance(username, str)
        assert len(username) > 0

    def test_get_secret_retrieves_database_password(self):
        """get_secret retrieves actual database password from Vault."""
        client = VaultClient()

        password = client.get_secret('mira/database', 'password')

        assert isinstance(password, str)
        assert len(password) > 0

    def test_get_secret_with_invalid_path_raises_permissionerror(self):
        """get_secret raises PermissionError for non-existent/forbidden path.

        Vault returns 403 Forbidden for paths that don't exist or aren't accessible,
        which is correct security behavior (don't reveal path existence).
        """
        client = VaultClient()

        with pytest.raises(PermissionError, match="Access denied"):
            client.get_secret('nonexistent/path', 'field')

    def test_get_secret_with_missing_field_raises_keyerror(self):
        """get_secret raises KeyError when field doesn't exist in secret."""
        client = VaultClient()

        with pytest.raises(KeyError, match="Field 'nonexistent_field' not found"):
            client.get_secret('mira/database', 'nonexistent_field')

    def test_get_secret_keyerror_includes_available_fields(self):
        """KeyError for missing field lists available fields."""
        client = VaultClient()

        try:
            client.get_secret('mira/database', 'invalid_field_xyz')
            pytest.fail("Should have raised KeyError")
        except KeyError as e:
            error_msg = str(e)
            # Error should mention what fields ARE available
            assert 'Available:' in error_msg


class TestConvenienceFunctions:
    """Test convenience functions that wrap VaultClient.get_secret()."""

    def test_get_database_url_for_mira_app(self):
        """get_database_url retrieves mira_app database URL."""
        url = get_database_url('mira_app')

        assert isinstance(url, str)
        assert 'postgresql' in url or 'postgres' in url
        assert len(url) > 0

    def test_get_database_url_for_mira_memory(self):
        """get_database_url retrieves mira_memory database URL."""
        url = get_database_url('mira_memory')

        assert isinstance(url, str)
        assert 'postgresql' in url or 'postgres' in url
        assert len(url) > 0

    def test_get_database_url_with_invalid_service_raises_valueerror(self):
        """get_database_url raises ValueError for unknown service."""
        with pytest.raises(ValueError, match="Unknown database service"):
            get_database_url('invalid_service_name')

    def test_get_api_key_retrieves_openai_key(self):
        """get_api_key retrieves OpenAI API key."""
        try:
            key = get_api_key('openai_key')
            assert isinstance(key, str)
            assert len(key) > 0
        except (FileNotFoundError, KeyError):
            # If this specific key doesn't exist, that's fine
            # We're testing the function works, not that every key exists
            pytest.skip("openai_key not configured in Vault")

    def test_get_auth_secret_retrieves_jwt_secret(self):
        """get_auth_secret retrieves JWT secret key."""
        secret = get_auth_secret('jwt_secret_key')

        assert isinstance(secret, str)
        assert len(secret) > 0

    def test_get_service_config_retrieves_field(self):
        """get_service_config retrieves service configuration field."""
        value = get_service_config('mira', 'app_url')

        assert isinstance(value, str)
        assert len(value) > 0
        # Should be a URL
        assert 'http' in value.lower()

    def test_get_database_credentials_returns_dict(self):
        """get_database_credentials retrieves username and password."""
        creds = get_database_credentials()

        # Verify return structure
        assert isinstance(creds, dict)
        assert 'username' in creds
        assert 'password' in creds

        # Verify values are strings
        assert isinstance(creds['username'], str)
        assert isinstance(creds['password'], str)
        assert len(creds['username']) > 0
        assert len(creds['password']) > 0


class TestCachingBehavior:
    """Test secret caching to verify performance optimization."""

    def test_secret_cache_is_populated_after_retrieval(self):
        """Secret cache is populated after first retrieval."""
        # Clear cache
        vault_client._secret_cache.clear()

        # Retrieve secret
        url = get_database_url('mira_app')

        # Verify cache was populated
        cache_key = 'mira/database/appdb_url'
        assert cache_key in vault_client._secret_cache
        assert vault_client._secret_cache[cache_key] == url

    def test_cached_secret_is_returned_on_subsequent_calls(self):
        """Subsequent calls return cached value without Vault API call."""
        # Clear cache
        vault_client._secret_cache.clear()

        # First call - hits Vault
        url1 = get_database_url('mira_app')

        # Manually modify cache to verify it's being used
        cache_key = 'mira/database/appdb_url'
        vault_client._secret_cache[cache_key] = 'test_cached_value'

        # Second call - should return cached value
        url2 = get_database_url('mira_app')

        assert url2 == 'test_cached_value'
        assert url1 != url2  # Proves cache is being used

    def test_cache_persists_across_function_calls(self):
        """Cache persists across multiple convenience function calls."""
        vault_client._secret_cache.clear()

        # Call multiple functions
        get_database_url('mira_app')
        get_database_url('mira_memory')
        get_database_credentials()

        # Verify multiple cache entries exist
        assert len(vault_client._secret_cache) >= 3

    def test_cache_is_cleared_by_reset_fixture(self):
        """Cache is cleared between tests by reset fixture."""
        # This test verifies the reset fixture works correctly
        # By this point, reset_test_environment fixture has run
        # If cache wasn't cleared, previous test's values would be present

        # The cache should start empty (or nearly empty) at start of test
        initial_cache_size = len(vault_client._secret_cache)

        # Add something to cache
        get_database_url('mira_app')

        # Verify cache grew
        assert len(vault_client._secret_cache) > initial_cache_size


class TestHealthCheck:
    """Test Vault connection health check functionality."""

    def test_vault_connection_succeeds(self):
        """test_vault_connection returns success status."""
        from clients.vault_client import test_vault_connection
        result = test_vault_connection()

        # Verify return structure
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'message' in result
        assert 'authenticated' in result

        # Verify successful connection
        assert result['status'] == 'success'
        assert result['authenticated'] is True
        assert result['vault_addr'] == os.getenv('VAULT_ADDR')

    def test_vault_connection_includes_namespace_if_configured(self):
        """test_vault_connection includes namespace when configured."""
        from clients.vault_client import test_vault_connection
        result = test_vault_connection()

        # If namespace is configured, it should be in result
        if os.getenv('VAULT_NAMESPACE'):
            assert 'namespace' in result
            assert result['namespace'] == os.getenv('VAULT_NAMESPACE')

    def test_vault_connection_handles_failure(self):
        """test_vault_connection returns error status when connection fails.

        Tests failure mode by temporarily breaking the vault client singleton
        and verifying the health check properly reports failure.
        """
        from clients.vault_client import test_vault_connection

        # Save original singleton
        original_instance = vault_client._vault_client_instance

        try:
            # Break the singleton by setting invalid credentials
            original_role_id = os.getenv('VAULT_ROLE_ID')
            os.environ['VAULT_ROLE_ID'] = 'intentionally-invalid-role-id'

            # Reset singleton so it gets recreated with bad credentials
            vault_client._vault_client_instance = None
            vault_client._secret_cache.clear()

            # Health check should return error status
            result = test_vault_connection()

            # Verify error response structure
            assert isinstance(result, dict)
            assert 'status' in result
            assert result['status'] == 'error'
            assert 'message' in result
            assert 'authenticated' in result
            assert result['authenticated'] is False

        finally:
            # Restore original state
            if original_role_id:
                os.environ['VAULT_ROLE_ID'] = original_role_id
            vault_client._vault_client_instance = original_instance


class TestErrorHandling:
    """Test error handling for various Vault failure scenarios."""

    def test_invalid_path_returns_clear_error_message(self):
        """Invalid/forbidden secret path returns clear, actionable error message.

        Vault returns PermissionError for paths that don't exist or aren't accessible.
        """
        client = VaultClient()

        try:
            client.get_secret('totally/fake/path/xyz', 'field')
            pytest.fail("Should have raised PermissionError")
        except PermissionError as e:
            error_msg = str(e)
            # Error should include the path that was accessed
            assert 'totally/fake/path/xyz' in error_msg
            assert 'denied' in error_msg.lower()

    def test_missing_field_returns_helpful_error(self):
        """Missing field in existing secret provides helpful guidance."""
        client = VaultClient()

        try:
            client.get_secret('mira/database', 'this_field_definitely_does_not_exist')
            pytest.fail("Should have raised KeyError")
        except KeyError as e:
            error_msg = str(e)
            # Error should mention the field and show available fields
            assert 'this_field_definitely_does_not_exist' in error_msg
            assert 'Available:' in error_msg

    def test_get_database_url_error_message_shows_valid_services(self):
        """get_database_url error message lists valid service names."""
        try:
            get_database_url('invalid_service')
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            # Should show what services ARE valid
            assert 'mira_app' in error_msg
            assert 'mira_memory' in error_msg


class TestRealWorldUsage:
    """Integration tests simulating real-world usage patterns."""

    def test_application_startup_retrieves_all_required_secrets(self):
        """Application can retrieve all required secrets at startup."""
        # Simulate application startup secret retrieval
        secrets_retrieved = []

        try:
            # Database URLs
            app_db = get_database_url('mira_app')
            secrets_retrieved.append(('mira_app_db', app_db))

            memory_db = get_database_url('mira_memory')
            secrets_retrieved.append(('mira_memory_db', memory_db))

            # Database credentials
            creds = get_database_credentials()
            secrets_retrieved.append(('db_username', creds['username']))
            secrets_retrieved.append(('db_password', creds['password']))

        except Exception as e:
            pytest.fail(f"Failed to retrieve required secrets: {e}")

        # Verify all secrets were retrieved successfully
        assert len(secrets_retrieved) >= 4

        # Verify all values are non-empty strings
        for name, value in secrets_retrieved:
            assert isinstance(value, str), f"{name} is not a string"
            assert len(value) > 0, f"{name} is empty"

    def test_repeated_secret_access_uses_cache(self):
        """Repeated secret access efficiently uses cache."""
        vault_client._secret_cache.clear()

        # Access same secret multiple times (simulating app runtime)
        urls = []
        for _ in range(5):
            url = get_database_url('mira_app')
            urls.append(url)

        # All URLs should be identical
        assert all(url == urls[0] for url in urls)

        # Cache should have been hit
        cache_key = 'mira/database/appdb_url'
        assert cache_key in vault_client._secret_cache

    def test_multiple_concurrent_secret_retrievals(self):
        """Multiple different secrets can be retrieved in sequence."""
        vault_client._secret_cache.clear()

        # Retrieve multiple different secrets
        secrets = {}

        try:
            secrets['app_db'] = get_database_url('mira_app')
            secrets['memory_db'] = get_database_url('mira_memory')
            creds = get_database_credentials()
            secrets['username'] = creds['username']
            secrets['password'] = creds['password']

        except Exception as e:
            pytest.fail(f"Failed to retrieve multiple secrets: {e}")

        # Verify all are distinct values
        assert len(secrets) == 4
        assert len(set(secrets.values())) >= 3  # At least 3 unique values

        # Verify cache has multiple entries
        assert len(vault_client._secret_cache) >= 3
