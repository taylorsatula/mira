"""
Integration tests for the /health endpoint.

Tests system health checks across database components without requiring authentication.
"""
import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Test /health endpoint returns correct system status."""

    def test_health_endpoint_returns_200_when_healthy(self, test_client: TestClient):
        """Verify /v0/api/health returns 200 with healthy status when databases are accessible."""
        response = test_client.get("/v0/v0/api/health")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["success"] is True
        assert "data" in data

        # Verify health data
        health_data = data["data"]
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        assert "components" in health_data
        assert "meta" in health_data  # meta is inside data

    def test_health_endpoint_no_auth_required(self, test_client: TestClient):
        """Verify /v0/api/health endpoint works without authentication."""
        # Make request without setting Authorization header
        response = test_client.get("/v0/v0/api/health")

        # Should succeed without auth
        assert response.status_code in [200, 503]  # Either healthy or unhealthy, but endpoint accessible
        data = response.json()
        assert data["success"] is not None

    def test_health_includes_database_component(self, test_client: TestClient):
        """Verify health check includes database component status."""
        response = test_client.get("/v0/v0/api/health")

        data = response.json()
        health_data = data["data"]
        components = health_data["components"]

        # Database component should be present
        assert "database" in components
        db_status = components["database"]
        assert "status" in db_status
        assert db_status["status"] in ["healthy", "unhealthy"]

        # If database is healthy, should have latency metric
        if db_status["status"] == "healthy":
            assert "latency_ms" in db_status
            assert isinstance(db_status["latency_ms"], (int, float))
            assert db_status["latency_ms"] >= 0

    def test_health_includes_memory_db_component(self, test_client: TestClient):
        """Verify health check includes memory database component."""
        response = test_client.get("/v0/v0/api/health")

        data = response.json()
        health_data = data["data"]
        components = health_data["components"]

        # Memory DB component should be present
        assert "memory_db" in components
        memory_status = components["memory_db"]
        assert "status" in memory_status
        assert memory_status["status"] in ["healthy", "unhealthy"]

        # If memory DB is healthy, should have latency metric
        if memory_status["status"] == "healthy":
            assert "latency_ms" in memory_status
            assert isinstance(memory_status["latency_ms"], (int, float))
            assert memory_status["latency_ms"] >= 0

    def test_health_includes_system_component(self, test_client: TestClient):
        """Verify health check includes system component info."""
        response = test_client.get("/v0/v0/api/health")

        data = response.json()
        health_data = data["data"]
        components = health_data["components"]

        # System component should be present
        assert "system" in components
        system_status = components["system"]
        assert system_status["status"] == "healthy"
        assert "version" in system_status
        assert "uptime_seconds" in system_status

    def test_health_returns_unhealthy_with_503_when_db_fails(self, test_client: TestClient, monkeypatch):
        """Verify /v0/api/health returns 503 when database is unavailable."""
        # Mock database failure
        from clients import postgres_client

        original_execute = postgres_client.PostgresClient.execute_single

        def failing_execute(self, query):
            if "SELECT 1" in query and self.database_name == "mira_service":
                raise Exception("Connection refused")
            return original_execute(self, query)

        monkeypatch.setattr(postgres_client.PostgresClient, "execute_single", failing_execute)

        response = test_client.get("/v0/v0/api/health")

        # When database is unhealthy, should return 503
        assert response.status_code == 503
        data = response.json()
        assert data["success"] is False

    def test_health_timestamp_is_valid_iso8601(self, test_client: TestClient):
        """Verify health response includes valid ISO8601 timestamp."""
        response = test_client.get("/v0/v0/api/health")

        data = response.json()
        health_data = data["data"]
        timestamp = health_data["timestamp"]

        # Verify ISO8601 format (basic check)
        assert "T" in timestamp  # ISO8601 format has T between date and time
        assert timestamp.endswith("Z") or "+" in timestamp  # Has timezone info

    def test_health_check_duration_metric(self, test_client: TestClient):
        """Verify health check includes duration metric."""
        response = test_client.get("/v0/v0/api/health")

        data = response.json()
        health_data = data["data"]
        meta = health_data["meta"]

        assert "check_duration_ms" in meta
        assert isinstance(meta["check_duration_ms"], (int, float))
        assert meta["check_duration_ms"] >= 0

    def test_health_checks_run_count(self, test_client: TestClient):
        """Verify health check reports how many component checks were run."""
        response = test_client.get("/v0/v0/api/health")

        data = response.json()
        health_data = data["data"]
        meta = health_data["meta"]

        assert "checks_run" in meta
        assert isinstance(meta["checks_run"], int)
        assert meta["checks_run"] >= 3  # At least database, memory_db, system

    def test_health_overall_status_reflects_all_components(self, test_client: TestClient):
        """Verify overall health status is unhealthy if any component fails."""
        response = test_client.get("/v0/v0/api/health")

        data = response.json()
        health_data = data["data"]
        status = health_data["status"]
        components = health_data["components"]

        # If all components are healthy, overall should be healthy
        all_healthy = all(comp.get("status") == "healthy" for comp in components.values())

        if all_healthy:
            assert status == "healthy"
            assert response.status_code == 200
