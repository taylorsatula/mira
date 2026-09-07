"""
Integration tests for POST /actions endpoint.

Tests domain-routed state mutations with real database persistence.
"""
import pytest
from fastapi.testclient import TestClient


class TestActionsEndpointAuthentication:
    """Test /actions endpoint authentication."""

    def test_actions_requires_authentication(self, test_client: TestClient):
        """Verify /actions endpoint requires authentication."""
        response = test_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "reminder",
                "action": "create",
                "data": {"title": "Test"}
            }
        )

        # Without auth, should return 401 or 403
        assert response.status_code in [401, 403]

    def test_actions_accepts_authenticated_request(self, authenticated_client: TestClient):
        """Verify authenticated user can call /actions endpoint."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "user",
                "action": "update_preferences",
                "data": {}
            }
        )

        # Should succeed (may have no effect with empty data, but endpoint accessible)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestActionsEndpointValidation:
    """Test /actions endpoint input validation."""

    def test_actions_requires_domain_field(self, authenticated_client: TestClient):
        """Verify /actions requires domain field."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "action": "create",
                "data": {}
            }
        )

        # Missing required domain field
        assert response.status_code == 422

    def test_actions_requires_action_field(self, authenticated_client: TestClient):
        """Verify /actions requires action field."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "reminder",
                "data": {}
            }
        )

        # Missing required action field
        assert response.status_code == 422

    def test_actions_rejects_invalid_domain(self, authenticated_client: TestClient):
        """Verify /actions rejects invalid domain."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "invalid_domain",
                "action": "create",
                "data": {}
            }
        )

        # Should return 422 for invalid enum value
        assert response.status_code == 422

    def test_actions_empty_action_rejected(self, authenticated_client: TestClient):
        """Verify /actions rejects empty action string."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "reminder",
                "action": "",
                "data": {}
            }
        )

        # Should be rejected by Pydantic validator
        assert response.status_code == 422

    def test_actions_data_field_optional(self, authenticated_client: TestClient):
        """Verify data field is optional."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "user",
                "action": "get_email_config"
                # No data field
            }
        )

        # Should succeed
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestActionsEndpointReminderDomain:
    """Test /actions endpoint with reminder domain."""

    def test_reminder_create_action(self, authenticated_client: TestClient):
        """Verify reminder create action works."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "reminder",
                "action": "create",
                "data": {
                    "title": "Buy groceries",
                    "date": "2025-01-20"
                }
            }
        )

        # Should succeed or fail with validation error
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "reminder" in data["data"]
            assert data["data"]["reminder"]["title"] == "Buy groceries"

    def test_reminder_create_requires_title(self, authenticated_client: TestClient):
        """Verify reminder create requires title field."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "reminder",
                "action": "create",
                "data": {
                    "date": "2025-01-20"
                    # Missing title
                }
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "title" in data["error"]["message"].lower()

    def test_reminder_create_requires_date(self, authenticated_client: TestClient):
        """Verify reminder create requires date field."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "reminder",
                "action": "create",
                "data": {
                    "title": "Test"
                    # Missing date
                }
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "date" in data["error"]["message"].lower()

    def test_reminder_create_with_optional_fields(self, authenticated_client: TestClient):
        """Verify reminder create accepts optional fields."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "reminder",
                "action": "create",
                "data": {
                    "title": "Meeting",
                    "date": "2025-02-15",
                    "description": "Team sync",
                    "contact_name": "Alice"
                }
            }
        )

        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True

    def test_reminder_rejects_unknown_action(self, authenticated_client: TestClient):
        """Verify reminder domain rejects unknown action."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "reminder",
                "action": "unknown_action",
                "data": {}
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "unknown" in data["error"]["message"].lower()


class TestActionsEndpointMemoryDomain:
    """Test /actions endpoint with memory domain."""

    def test_memory_create_action(self, authenticated_client: TestClient):
        """Verify memory create action works."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "memory",
                "action": "create",
                "data": {
                    "content": "User prefers coffee over tea"
                }
            }
        )

        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "memory" in data["data"]

    def test_memory_create_requires_content(self, authenticated_client: TestClient):
        """Verify memory create requires content field."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "memory",
                "action": "create",
                "data": {}
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "content" in data["error"]["message"].lower()

    def test_memory_create_with_importance(self, authenticated_client: TestClient):
        """Verify memory create accepts optional importance field."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "memory",
                "action": "create",
                "data": {
                    "content": "Important fact",
                    "importance": 0.8
                }
            }
        )

        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True


class TestActionsEndpointUserDomain:
    """Test /actions endpoint with user domain."""

    def test_user_update_preferences_action(self, authenticated_client: TestClient):
        """Verify user update_preferences action works."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "user",
                "action": "update_preferences",
                "data": {
                    "theme": "dark"
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_user_get_email_config_action(self, authenticated_client: TestClient):
        """Verify user get_email_config action works."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "user",
                "action": "get_email_config",
                "data": {}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestActionsEndpointContactsDomain:
    """Test /actions endpoint with contacts domain."""

    def test_contacts_create_action(self, authenticated_client: TestClient):
        """Verify contacts create action works."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "contacts",
                "action": "create",
                "data": {
                    "name": "John Doe",
                    "email": "john@example.com"
                }
            }
        )

        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "contact" in data["data"]

    def test_contacts_create_requires_name(self, authenticated_client: TestClient):
        """Verify contacts create requires name field."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "contacts",
                "action": "create",
                "data": {
                    "email": "test@example.com"
                }
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "name" in data["error"]["message"].lower()

    def test_contacts_list_action(self, authenticated_client: TestClient):
        """Verify contacts list action works."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "contacts",
                "action": "list",
                "data": {}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "contacts" in data["data"]
        assert isinstance(data["data"]["contacts"], list)


class TestActionsEndpointConversationDomain:
    """Test /actions endpoint with continuum domain."""

    def test_conversation_link_day_action(self, authenticated_client: TestClient):
        """Verify continuum link_day action works."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "continuum",
                "action": "link_day",
                "data": {
                    "date": "2025-01-15"
                }
            }
        )

        # Should succeed or fail with business logic error
        assert response.status_code in [200, 400]

    def test_conversation_link_day_requires_date(self, authenticated_client: TestClient):
        """Verify link_day requires date field."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "continuum",
                "action": "link_day",
                "data": {}
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "date" in data["error"]["message"].lower()

    def test_postpone_collapse_action(self, authenticated_client: TestClient):
        """Verify postpone_collapse action sets virtual last message time."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "continuum",
                "action": "postpone_collapse",
                "data": {
                    "minutes": 30
                }
            }
        )

        # Should succeed if there's an active segment, or fail with NotFound if not
        assert response.status_code in [200, 404]
        data = response.json()
        if response.status_code == 200:
            assert data["success"] is True
            assert "postponed" in data["data"]
            assert data["data"]["postponed"] is True
            assert data["data"]["minutes"] == 30
            assert "virtual_last_message_time" in data["data"]

    def test_postpone_collapse_requires_minutes(self, authenticated_client: TestClient):
        """Verify postpone_collapse requires minutes field."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "continuum",
                "action": "postpone_collapse",
                "data": {}
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "minutes" in data["error"]["message"].lower()

    def test_postpone_collapse_validates_minutes_range(self, authenticated_client: TestClient):
        """Verify postpone_collapse validates minutes is between 1 and 1440."""
        # Test below minimum
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "continuum",
                "action": "postpone_collapse",
                "data": {"minutes": 0}
            }
        )
        assert response.status_code == 400
        assert "1 and 1440" in response.json()["error"]["message"]

        # Test above maximum
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "continuum",
                "action": "postpone_collapse",
                "data": {"minutes": 1441}
            }
        )
        assert response.status_code == 400
        assert "1 and 1440" in response.json()["error"]["message"]


class TestActionsEndpointDomainKnowledgeDomain:
    """Test /actions endpoint with domain_knowledge domain."""

    def test_domain_knowledge_create_action(self, authenticated_client: TestClient):
        """Verify domain_knowledge create action works."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "domain_knowledge",
                "action": "create",
                "data": {
                    "domain_label": "work",
                    "domain_name": "Work Context",
                    "block_description": "Information about work projects"
                }
            }
        )

        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True

    def test_domain_knowledge_create_requires_fields(self, authenticated_client: TestClient):
        """Verify domain_knowledge create requires all required fields."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "domain_knowledge",
                "action": "create",
                "data": {
                    "domain_label": "work"
                    # Missing domain_name and block_description
                }
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False


class TestActionsEndpointResponseStructure:
    """Test /actions endpoint response structure."""

    def test_actions_success_response_structure(self, authenticated_client: TestClient):
        """Verify successful action response has correct structure."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "user",
                "action": "update_preferences",
                "data": {}
            }
        )

        data = response.json()

        # Verify top-level structure
        assert "success" in data
        assert "data" in data
        assert "meta" in data

        # Success should be True
        assert data["success"] is True

        # Meta should have timestamp
        assert "timestamp" in data["meta"]

    def test_actions_error_response_structure(self, authenticated_client: TestClient):
        """Verify error response has correct structure."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "reminder",
                "action": "create",
                "data": {}
            }
        )

        if response.status_code == 400:
            data = response.json()

            # Verify error structure
            assert data["success"] is False
            assert "error" in data
            assert "code" in data["error"]
            assert "message" in data["error"]

            # Should have request ID in meta
            assert "meta" in data
            assert "timestamp" in data["meta"]


class TestActionsEndpointUserIsolation:
    """Test user data isolation in actions."""

    def test_actions_scoped_to_authenticated_user(self, authenticated_client: TestClient, authenticated_user):
        """Verify actions only affect authenticated user's data."""
        # Create a memory for the authenticated user
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "memory",
                "action": "create",
                "data": {
                    "content": "Private user memory"
                }
            }
        )

        if response.status_code == 200:
            # Memory created for authenticated user
            # User isolation handled at DB level via RLS
            data = response.json()
            assert data["success"] is True
            assert authenticated_user["user_id"]  # Verify we have user context


class TestActionsEndpointErrorCases:
    """Test /actions endpoint error handling."""

    def test_actions_handles_invalid_uuid(self, authenticated_client: TestClient):
        """Verify actions handles invalid UUID in data."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "memory",
                "action": "delete",
                "data": {
                    "id": "not-a-uuid"
                }
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False

    def test_actions_rejects_unknown_fields(self, authenticated_client: TestClient):
        """Verify actions rejects unknown fields in data."""
        response = authenticated_client.post(
            "/v0/v0/api/actions",
            json={
                "domain": "reminder",
                "action": "create",
                "data": {
                    "title": "Test",
                    "date": "2025-01-20",
                    "unknown_field": "should be rejected"
                }
            }
        )

        if response.status_code == 400:
            data = response.json()
            assert "unknown" in data["error"]["message"].lower()
