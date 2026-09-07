"""
Fixtures for simulating infrastructure failures in tests.

These are not mocks of functionality but rather tools to simulate
failure conditions for testing error handling and resilience.
"""
import pytest


@pytest.fixture
def vault_unavailable(monkeypatch):
    """Simulate Vault unavailability for error testing."""
    def mock_get_secret(*args, **kwargs):
        raise RuntimeError("Vault connection failed")
    
    monkeypatch.setattr("clients.vault_client.VaultClient.get_secret", mock_get_secret)


@pytest.fixture
def valkey_unavailable(monkeypatch):
    """Simulate Valkey unavailability for error testing."""
    def mock_health_check(*args, **kwargs):
        raise Exception("Valkey connection failed")
    
    monkeypatch.setattr("clients.valkey_client.ValkeyClient.health_check", mock_health_check)


@pytest.fixture
def database_unavailable(monkeypatch):
    """Simulate database unavailability for error testing."""
    def mock_get_connection(*args, **kwargs):
        raise Exception("Database connection failed")
    
    monkeypatch.setattr("clients.postgres_client.PostgresClient.get_connection", mock_get_connection)


@pytest.fixture
def llm_provider_unavailable(monkeypatch):
    """Simulate LLM provider unavailability for error testing."""
    def mock_generate(*args, **kwargs):
        raise Exception("LLM API unavailable")
    
    monkeypatch.setattr("clients.llm_provider.LLMProvider.generate_response", mock_generate)


@pytest.fixture
def network_timeout(monkeypatch):
    """Simulate network timeout conditions."""
    import asyncio
    
    async def mock_timeout(*args, **kwargs):
        await asyncio.sleep(0.1)
        raise asyncio.TimeoutError("Network request timed out")
    
    # Can be applied to various network operations as needed
    return mock_timeout


@pytest.fixture
def mock_webauthn_credentials():
    """Provide mock WebAuthn credentials for testing."""
    return {
        "registration_options": {
            "challenge": "test_challenge_123",
            "rp": {"name": "MIRA", "id": "localhost"},
            "user": {
                "id": "test_user_id",
                "name": "test@example.com",
                "displayName": "Test User"
            }
        },
        "registration_response": {
            "id": "test_credential_id",
            "rawId": "dGVzdF9jcmVkZW50aWFsX2lk",  # base64 encoded
            "response": {
                "attestationObject": "mock_attestation_object",
                "clientDataJSON": "mock_client_data_json"
            },
            "type": "public-key"
        }
    }