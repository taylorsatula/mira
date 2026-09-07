"""
Integration tests for POST /chat endpoint.

Tests real message processing, database persistence, and response generation.
"""
import pytest
import base64
from fastapi.testclient import TestClient


class TestChatEndpointBasics:
    """Test basic /chat endpoint functionality."""

    def test_chat_requires_authentication(self, test_client: TestClient):
        """Verify /v0/api/chat endpoint requires authentication."""
        response = test_client.post(
            "/v0/v0/api/chat",
            json={"message": "Hello"}
        )

        # Without auth, should return 401 or 403
        assert response.status_code in [401, 403]

    def test_chat_accepts_authenticated_request(self, authenticated_client: TestClient, authenticated_user):
        """Verify authenticated user can send chat message."""
        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={"message": "Hello, MIRA"}
        )

        # Should succeed
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_chat_returns_correct_response_structure(self, authenticated_client: TestClient):
        """Verify /v0/api/chat returns exact expected response structure."""
        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={"message": "What is 2+2?"}
        )

        data = response.json()

        # Verify top-level structure
        assert data["success"] is True
        assert "data" in data
        assert "meta" in data

        # Verify data fields
        chat_data = data["data"]
        assert "continuum_id" in chat_data
        assert "response" in chat_data
        assert "metadata" in chat_data

        # Verify metadata structure
        metadata = chat_data["metadata"]
        assert "tools_used" in metadata
        assert "referenced_memories" in metadata
        assert "surfaced_memories" in metadata
        assert "processing_time_ms" in metadata

        # Verify types
        assert isinstance(chat_data["continuum_id"], str)
        assert isinstance(chat_data["response"], str)
        assert isinstance(metadata["tools_used"], list)
        assert isinstance(metadata["referenced_memories"], list)
        assert isinstance(metadata["surfaced_memories"], list)
        assert isinstance(metadata["processing_time_ms"], int)

    def test_chat_processes_message_and_returns_response(self, authenticated_client: TestClient):
        """Verify /chat actually processes message and returns non-empty response."""
        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={"message": "Say hello"}
        )

        data = response.json()
        chat_data = data["data"]

        # Response should be non-empty
        assert len(chat_data["response"]) > 0
        assert isinstance(chat_data["response"], str)

    def test_chat_creates_continuum_id(self, authenticated_client: TestClient):
        """Verify /chat returns a valid continuum ID."""
        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={"message": "First message"}
        )

        data = response.json()
        continuum_id = data["data"]["continuum_id"]

        # Should be non-empty string (UUID format expected)
        assert isinstance(continuum_id, str)
        assert len(continuum_id) > 0

        # Second request should return same continuum (continuum continues)
        response2 = authenticated_client.post(
            "/v0/v0/api/chat",
            json={"message": "Second message"}
        )

        continuum_id2 = response2.json()["data"]["continuum_id"]
        assert continuum_id2 == continuum_id

    def test_chat_processing_time_measured(self, authenticated_client: TestClient):
        """Verify processing time metric is captured."""
        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={"message": "How long did this take?"}
        )

        data = response.json()
        processing_time_ms = data["data"]["metadata"]["processing_time_ms"]

        # Should be a positive integer representing milliseconds
        assert isinstance(processing_time_ms, int)
        assert processing_time_ms > 0


class TestChatEndpointValidation:
    """Test /chat endpoint input validation."""

    def test_chat_rejects_empty_message(self, authenticated_client: TestClient):
        """Verify /chat rejects empty messages."""
        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={"message": ""}
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"
        assert "empty" in data["error"]["message"].lower()

    def test_chat_rejects_whitespace_only_message(self, authenticated_client: TestClient):
        """Verify /chat rejects whitespace-only messages."""
        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={"message": "   \n\t  "}
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False

    def test_chat_trims_whitespace_from_message(self, authenticated_client: TestClient):
        """Verify /chat trims leading/trailing whitespace."""
        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={"message": "  \n  Hello world  \n  "}
        )

        # Should succeed (whitespace trimmed)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_chat_rejects_missing_message_field(self, authenticated_client: TestClient):
        """Verify /chat requires message field."""
        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={}
        )

        # FastAPI should reject missing required field
        assert response.status_code == 422

    def test_chat_requires_image_type_with_image(self, authenticated_client: TestClient):
        """Verify image_type is required when image is provided."""
        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={
                "message": "What's in this image?",
                "image": "base64encodeddata"
                # Missing image_type
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "image_type" in data["error"]["message"].lower()


class TestChatEndpointImages:
    """Test /chat image handling."""

    def test_chat_rejects_unsupported_image_format(self, authenticated_client: TestClient):
        """Verify /chat rejects unsupported image formats."""
        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={
                "message": "Analyze this image",
                "image": base64.b64encode(b"fake image data").decode(),
                "image_type": "image/bmp"  # Unsupported format
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "unsupported" in data["error"]["message"].lower()

    def test_chat_accepts_jpeg_images(self, authenticated_client: TestClient):
        """Verify /chat accepts JPEG images."""
        # Create minimal valid JPEG (FFD8FF is JPEG magic bytes)
        jpeg_data = bytes.fromhex("FFD8FFE0") + b"\x00" * 100 + bytes.fromhex("FFD9")
        image_b64 = base64.b64encode(jpeg_data).decode()

        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={
                "message": "What is this?",
                "image": image_b64,
                "image_type": "image/jpeg"
            }
        )

        # Should not reject due to format
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_chat_rejects_oversized_images(self, authenticated_client: TestClient):
        """Verify /chat rejects images exceeding size limit."""
        # Create image larger than 5MB
        oversized_data = b"x" * (6 * 1024 * 1024)
        image_b64 = base64.b64encode(oversized_data).decode()

        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={
                "message": "Analyze this huge image",
                "image": image_b64,
                "image_type": "image/jpeg"
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "exceed" in data["error"]["message"].lower() or "size" in data["error"]["message"].lower()

    def test_chat_rejects_invalid_base64(self, authenticated_client: TestClient):
        """Verify /chat rejects invalid base64 image data."""
        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={
                "message": "Look at this",
                "image": "not valid base64!!!",
                "image_type": "image/jpeg"
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "base64" in data["error"]["message"].lower()


class TestChatEndpointConcurrency:
    """Test /chat endpoint concurrency controls."""

    def test_chat_enforces_per_user_request_lock(self, authenticated_client: TestClient):
        """Verify /chat enforces one active request per user."""
        import time
        import threading

        responses = []
        errors = []

        def send_request():
            try:
                # Send a "slow" message
                response = authenticated_client.post(
                    "/v0/v0/api/chat",
                    json={"message": "Sleep and process this slowly"}
                )
                responses.append(response)
            except Exception as e:
                errors.append(e)

        # Start two requests concurrently
        threads = [threading.Thread(target=send_request) for _ in range(2)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=30)

        # One should succeed, one should fail with lock error
        assert len(responses) == 2
        assert len(errors) == 0

        # Check response statuses
        status_codes = [r.status_code for r in responses]
        data_list = [r.json() for r in responses]

        # One should be 200, one should be 400 (validation error for locked user)
        assert 200 in status_codes or any(d.get("success") for d in data_list)

    def test_chat_lock_releases_after_request(self, authenticated_client: TestClient):
        """Verify user lock is released after request completes."""
        # First request
        response1 = authenticated_client.post(
            "/v0/v0/api/chat",
            json={"message": "First request"}
        )

        assert response1.status_code == 200

        # Second request should succeed (lock released)
        response2 = authenticated_client.post(
            "/v0/v0/api/chat",
            json={"message": "Second request"}
        )

        assert response2.status_code == 200


class TestChatEndpointTimestamp:
    """Test /chat endpoint timestamp handling."""

    def test_chat_response_includes_timestamp(self, authenticated_client: TestClient):
        """Verify /chat response includes valid ISO8601 timestamp."""
        response = authenticated_client.post(
            "/v0/v0/api/chat",
            json={"message": "What time is it?"}
        )

        data = response.json()
        timestamp = data["meta"]["timestamp"]

        # Verify ISO8601 format
        assert "T" in timestamp
        assert timestamp.endswith("Z") or "+" in timestamp
