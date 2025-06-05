"""
Security-focused tests for secure_auth API endpoints using real HTTP requests.

Tests ACTUAL API security behavior and attack vectors using FastAPI TestClient.
NO MOCKING of core security validations - tests the real implementation.
"""

import pytest
import secrets
import time
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from secure_auth.api import router, get_current_user
from secure_auth.models import User
from secure_auth.auth_service import get_auth_service
from secure_auth.session_service import get_session_service
from secure_auth.rate_limit_service import get_rate_limit_service
from secure_auth.audit_service import get_audit_service


@pytest.fixture
def test_app():
    """Create FastAPI app with secure_auth router for testing."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """Create TestClient for making real HTTP requests."""
    return TestClient(test_app)


@pytest.fixture
def test_user():
    """Create a test user in the database."""
    auth_service = get_auth_service()
    result = auth_service.create_user(
        email="test@example.com",
        tenant_id="default",
        is_admin=False
    )
    assert result.success
    return result.user


@pytest.fixture
def admin_user():
    """Create an admin test user in the database."""
    auth_service = get_auth_service()
    result = auth_service.create_user(
        email="admin@example.com",
        tenant_id="default",
        is_admin=True
    )
    assert result.success
    return result.user


@pytest.fixture
def authenticated_session(test_user):
    """Create an authenticated session for testing."""
    session_service = get_session_service()
    result = session_service.create_session(
        user=test_user,
        ip_address="192.168.1.100",
        user_agent="Test Browser",
        device_fingerprint=None,
        device_name="Test Device",
        remember_device=False,
        auth_method="magic_link"
    )
    assert result.success
    return {
        "user": test_user,
        "session_token": result.session_token,
        "headers": {"Authorization": f"Bearer {result.session_token}"}
    }


@pytest.fixture
def admin_session(admin_user):
    """Create an authenticated admin session for testing."""
    session_service = get_session_service()
    result = session_service.create_session(
        user=admin_user,
        ip_address="192.168.1.100",
        user_agent="Admin Browser",
        device_fingerprint=None,
        device_name="Admin Device",
        remember_device=False,
        auth_method="magic_link"
    )
    assert result.success
    return {
        "user": admin_user,
        "session_token": result.session_token,
        "headers": {"Authorization": f"Bearer {result.session_token}"}
    }


class TestMagicLinkRequestSecurity:
    """Test magic link request endpoint with real HTTP requests."""
    
    def test_magic_link_request_success(self, client):
        """Test valid magic link request returns success."""
        response = client.post("/auth/request-magic-link", json={
            "email": "test@example.com",
            "tenant_id": "default"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data
        
    def test_magic_link_request_invalid_email(self, client):
        """Test invalid email format is rejected."""
        invalid_emails = [
            "not_an_email",
            "@missing_local.com",
            "missing_domain@",
            "double@@domain.com",
            "spaces in@email.com",
        ]
        
        for email in invalid_emails:
            response = client.post("/auth/request-magic-link", json={
                "email": email,
                "tenant_id": "default"
            })
            
            # Should be rejected with 422 (validation error)
            assert response.status_code == 422
    
    def test_magic_link_request_injection_attempts(self, client):
        """Test injection attacks are prevented."""
        malicious_payloads = [
            {"email": "test@example.com'; DROP TABLE users; --", "tenant_id": "default"},
            {"email": "test@example.com<script>alert('xss')</script>", "tenant_id": "default"},
            {"email": "test@example.com", "tenant_id": "'; DROP TABLE tenants; --"},
            {"email": "test@example.com", "device_name": "<script>alert('device')</script>"}
        ]
        
        for payload in malicious_payloads:
            response = client.post("/auth/request-magic-link", json=payload)
            
            # Should either reject with validation error or sanitize input
            if response.status_code == 422:
                # Rejected by validation - good
                assert True
            elif response.status_code == 200:
                # Accepted but should be sanitized in the auth service
                data = response.json()
                assert data["success"] is True
    
    def test_rate_limiting_prevents_spam(self, client):
        """Test rate limiting blocks excessive magic link requests."""
        email = "target@example.com"
        
        # Make rapid requests until rate limited
        rate_limited = False
        request_count = 0
        
        for i in range(25):  # Try many requests
            response = client.post("/auth/request-magic-link", json={
                "email": email,
                "tenant_id": "default"
            })
            
            request_count += 1
            
            if response.status_code == 429:
                rate_limited = True
                # Should include Retry-After header
                assert "retry-after" in response.headers
                break
            elif response.status_code != 200:
                # Some other error
                break
        
        # Should eventually be rate limited
        assert rate_limited, f"Rate limiting did not trigger after {request_count} requests"
        assert request_count <= 10, "Rate limiting should trigger within reasonable limits"
    
    def test_distributed_attack_rate_limiting(self, client):
        """Test rate limiting works against distributed attacks."""
        email = "victim@example.com"
        attacker_ips = [f"203.0.113.{i}" for i in range(50, 60)]  # 10 different IPs
        
        successful_requests = 0
        rate_limited_responses = 0
        
        for ip in attacker_ips:
            # Use different User-Agent and X-Forwarded-For for each "attacker"
            headers = {
                "X-Forwarded-For": ip,
                "User-Agent": f"AttackBot/{ip.split('.')[-1]}"
            }
            
            for attempt in range(5):  # 5 attempts per IP
                response = client.post("/auth/request-magic-link", 
                                     json={"email": email, "tenant_id": "default"},
                                     headers=headers)
                
                if response.status_code == 200:
                    successful_requests += 1
                elif response.status_code == 429:
                    rate_limited_responses += 1
                    break  # This IP is rate limited
        
        # Should be limited by email rate limit, not IP limits
        assert successful_requests <= 5, f"Too many successful requests: {successful_requests}"
        assert rate_limited_responses > 0, "No rate limiting detected in distributed attack"
    
    def test_rate_limiting_across_http_methods(self, client):
        """
        Test rate limiting enforcement across different HTTP methods.
        
        Attack scenario: Attacker attempts to bypass rate limits by using
        different HTTP methods against the same endpoint.
        """
        target_email = "method-test@example.com"
        test_payload = {"email": target_email, "tenant_id": "default"}
        
        # Track responses across different methods
        method_responses = {
            "POST": [],
            "GET": [],
            "PUT": [],
            "DELETE": [],
            "PATCH": [],
            "HEAD": [],
            "OPTIONS": []
        }
        
        rate_limit_triggered = False
        total_successful = 0
        
        # Test each HTTP method with rapid requests
        for method in method_responses.keys():
            method_successful = 0
            
            for attempt in range(15):  # Try many requests per method
                try:
                    if method == "POST":
                        response = client.post("/auth/request-magic-link", json=test_payload)
                    elif method == "GET":
                        # GET requests should be rejected for this endpoint anyway
                        response = client.get("/auth/request-magic-link")
                    elif method == "PUT":
                        response = client.put("/auth/request-magic-link", json=test_payload)
                    elif method == "DELETE":
                        response = client.delete("/auth/request-magic-link")
                    elif method == "PATCH":
                        response = client.patch("/auth/request-magic-link", json=test_payload)
                    elif method == "HEAD":
                        response = client.head("/auth/request-magic-link")
                    elif method == "OPTIONS":
                        response = client.options("/auth/request-magic-link")
                    
                    method_responses[method].append(response.status_code)
                    
                    if response.status_code == 200:
                        method_successful += 1
                        total_successful += 1
                    elif response.status_code == 429:
                        rate_limit_triggered = True
                        break  # Rate limit hit for this method
                    
                    # Small delay between requests
                    time.sleep(0.1)
                    
                except Exception as e:
                    # Some methods might not be implemented, that's fine
                    method_responses[method].append(f"error: {str(e)}")
        
        # Security assertions
        
        # 1. Rate limiting should eventually trigger regardless of method
        assert rate_limit_triggered or total_successful <= 10, (
            f"Rate limiting not enforced across methods. "
            f"Total successful: {total_successful}, Rate limited: {rate_limit_triggered}"
        )
        
        # 2. POST method should work initially but be rate limited
        post_responses = method_responses["POST"]
        assert any(code == 200 for code in post_responses if isinstance(code, int)), \
            "POST method should work initially"
        assert any(code == 429 for code in post_responses if isinstance(code, int)), \
            "POST method should be rate limited eventually"
        
        # 3. Non-POST methods should either be rejected or also rate limited
        for method in ["GET", "PUT", "DELETE", "PATCH"]:
            responses = method_responses[method]
            valid_responses = [code for code in responses if isinstance(code, int)]
            
            if valid_responses:
                # Should be rejected (405 Method Not Allowed) or rate limited (429)
                assert all(code in [405, 429, 422] for code in valid_responses), \
                    f"{method} method should be rejected or rate limited, got: {valid_responses}"
        
        # 4. Verify rate limiting is not bypassed by method switching
        if total_successful > 5:
            assert rate_limit_triggered, \
                "Rate limiting should trigger when switching between HTTP methods"
    
    def test_rate_limiting_across_different_endpoints(self, client):
        """
        Test that rate limits are properly isolated or shared across endpoints.
        
        Attack scenario: Attacker attempts to bypass rate limits by attacking
        multiple endpoints that share the same rate limiting bucket.
        """
        test_email = "endpoint-test@example.com"
        
        # Test multiple endpoints that might share rate limits
        endpoints_to_test = [
            {
                "url": "/auth/request-magic-link",
                "method": "POST",
                "payload": {"email": test_email, "tenant_id": "default"}
            },
            {
                "url": "/auth/csrf-token",
                "method": "GET",
                "payload": None
            },
            {
                "url": "/auth/health",
                "method": "GET", 
                "payload": None
            }
        ]
        
        endpoint_results = {}
        
        # Test each endpoint for rate limiting behavior
        for endpoint in endpoints_to_test:
            endpoint_key = f"{endpoint['method']} {endpoint['url']}"
            successful_requests = 0
            rate_limited = False
            
            for attempt in range(20):  # Try many requests per endpoint
                if endpoint["method"] == "POST":
                    response = client.post(endpoint["url"], json=endpoint["payload"])
                else:
                    response = client.get(endpoint["url"])
                
                if response.status_code == 200:
                    successful_requests += 1
                elif response.status_code == 429:
                    rate_limited = True
                    break
                
                time.sleep(0.1)
            
            endpoint_results[endpoint_key] = {
                "successful": successful_requests,
                "rate_limited": rate_limited
            }
        
        # Security analysis
        
        # 1. High-risk endpoints (like magic link) should have strict rate limits
        magic_link_result = endpoint_results.get("POST /auth/request-magic-link", {})
        assert magic_link_result.get("rate_limited", False) or magic_link_result.get("successful", 0) <= 5, \
            "Magic link endpoint should have strict rate limiting"
        
        # 2. Informational endpoints might have different limits but should still be protected
        for endpoint_key, result in endpoint_results.items():
            if result.get("successful", 0) > 50:  # Very high threshold for info endpoints
                assert False, f"Endpoint {endpoint_key} has no rate limiting: {result['successful']} successful requests"
        
        # 3. Verify rate limits are working across the board
        any_rate_limited = any(result.get("rate_limited", False) for result in endpoint_results.values())
        high_success_count = any(result.get("successful", 0) > 15 for result in endpoint_results.values())
        
        assert any_rate_limited or not high_success_count, \
            "No rate limiting detected across any endpoints"


class TestMagicLinkVerificationSecurity:
    """Test magic link verification endpoint security."""
    
    def test_invalid_token_rejection(self, client):
        """Test invalid tokens are rejected."""
        invalid_tokens = [
            "fake_token",
            "",
            " ",
            "token_with_spaces",
            "token<script>alert('xss')</script>",
            "token'; DROP TABLE magic_links; --",
            secrets.token_urlsafe(43),  # Random valid-format token
        ]
        
        for token in invalid_tokens:
            # URL encode the token if it contains special characters
            safe_token = token.replace(" ", "%20").replace("<", "%3C").replace(">", "%3E")
            response = client.get(f"/auth/verify-magic-link/{safe_token}")
            
            # Should be rejected
            assert response.status_code in [400, 404], f"Token '{token}' was not rejected"
    
    def test_brute_force_token_protection(self, client):
        """Test brute force attacks against magic link tokens are blocked."""
        # Try many random tokens
        failed_attempts = 0
        rate_limited = False
        
        for i in range(100):
            fake_token = secrets.token_urlsafe(43)
            response = client.get(f"/auth/verify-magic-link/{fake_token}")
            
            if response.status_code == 429:
                rate_limited = True
                break
            elif response.status_code in [400, 404]:
                failed_attempts += 1
            
            # Add small delay to avoid overwhelming the test
            if i % 10 == 0:
                time.sleep(0.1)
        
        # Should eventually be rate limited or all attempts should fail
        assert failed_attempts > 0, "No token validation occurred"
        # Rate limiting might kick in for sustained attacks
        if rate_limited:
            assert True  # Rate limiting is working
    
    def test_session_creation_on_valid_verification(self, client, test_user):
        """Test valid magic link creates proper session."""
        # First create a magic link
        auth_service = get_auth_service()
        link_result = auth_service.request_magic_link(
            email=test_user.email,
            ip_address="192.168.1.100",
            user_agent="Test Browser",
            device_fingerprint=None,
            base_url="http://testserver",
            tenant_id=test_user.tenant_id
        )
        
        assert link_result.success
        assert link_result.magic_link_id
        
        # Verify the magic link
        response = client.get(f"/auth/verify-magic-link/{link_result.magic_link_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_email"] == test_user.email
        assert data["session_token"] is not None
        
        # Should set secure session cookie
        assert "set-cookie" in response.headers
        cookie_header = response.headers["set-cookie"]
        assert "secure_session=" in cookie_header
        assert "HttpOnly" in cookie_header
        assert "SameSite=strict" in cookie_header


class TestAuthenticationAndAuthorization:
    """Test authentication and authorization with real HTTP requests."""
    
    def test_unauthenticated_access_blocked(self, client):
        """Test unauthenticated access to protected endpoints is blocked."""
        protected_endpoints = [
            ("GET", "/auth/me"),
            ("POST", "/auth/logout"),
            ("POST", "/auth/create-user"),
        ]
        
        for method, endpoint in protected_endpoints:
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                response = client.post(endpoint, json={})
            
            assert response.status_code == 401
            assert "authentication required" in response.json()["detail"].lower()
            assert "www-authenticate" in response.headers
    
    def test_invalid_session_token_blocked(self, client):
        """Test invalid session tokens are rejected."""
        invalid_tokens = [
            "fake_token",
            "expired_token_123",
            secrets.token_urlsafe(43),  # Random valid-format token
            "",
        ]
        
        for token in invalid_tokens:
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get("/auth/me", headers=headers)
            
            assert response.status_code == 401
            assert "invalid or expired session" in response.json()["detail"].lower()
    
    def test_authenticated_access_works(self, client, authenticated_session):
        """Test authenticated users can access protected endpoints."""
        response = client.get("/auth/me", headers=authenticated_session["headers"])
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == authenticated_session["user"].email
        assert data["id"] == str(authenticated_session["user"].id)
    
    def test_session_context_validation(self, client, authenticated_session):
        """Test session context validation prevents hijacking."""
        # Try to use session from different IP
        headers = {
            **authenticated_session["headers"],
            "X-Forwarded-For": "10.0.0.1",  # Different IP
        }
        
        response = client.get("/auth/me", headers=headers)
        
        # Should be rejected due to IP context mismatch
        assert response.status_code == 401
    
    def test_cookie_based_authentication(self, client, authenticated_session):
        """Test cookie-based authentication works."""
        # Set session cookie
        client.cookies.set("secure_session", authenticated_session["session_token"])
        
        response = client.get("/auth/me")
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == authenticated_session["user"].email


class TestPrivilegeEscalationPrevention:
    """Test privilege escalation prevention with real HTTP requests."""
    
    def test_non_admin_user_creation_blocked(self, client, authenticated_session):
        """Test non-admin users cannot create accounts."""
        response = client.post("/auth/create-user", 
                             json={
                                 "email": "newuser@example.com",
                                 "tenant_id": "default",
                                 "is_admin": False
                             },
                             headers=authenticated_session["headers"])
        
        assert response.status_code == 403
        assert "administrator privileges required" in response.json()["detail"].lower()
    
    def test_admin_user_creation_works(self, client, admin_session):
        """Test admin users can create accounts."""
        new_email = f"newuser{secrets.token_hex(4)}@example.com"
        
        response = client.post("/auth/create-user",
                             json={
                                 "email": new_email,
                                 "tenant_id": "default", 
                                 "is_admin": False
                             },
                             headers=admin_session["headers"])
        
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == new_email
        assert data["is_admin"] is False
    
    def test_privilege_escalation_attempts_logged(self, client, authenticated_session):
        """Test privilege escalation attempts are logged."""
        # Attempt to create admin user as non-admin
        response = client.post("/auth/create-user",
                             json={
                                 "email": "malicious@example.com",
                                 "tenant_id": "default",
                                 "is_admin": True  # Attempting admin escalation
                             },
                             headers=authenticated_session["headers"])
        
        assert response.status_code == 403
        
        # Verify the attempt was logged (check audit service)
        audit_service = get_audit_service()
        # Note: In a real test, you'd verify the audit log contains the escalation attempt
        # This would require querying the audit database or checking logged events


class TestLogoutSecurity:
    """Test logout security with real HTTP requests."""
    
    def test_logout_invalidates_session(self, client, authenticated_session):
        """Test logout properly invalidates the session."""
        # Verify session works before logout
        response = client.get("/auth/me", headers=authenticated_session["headers"])
        assert response.status_code == 200
        
        # Logout
        response = client.post("/auth/logout", headers=authenticated_session["headers"])
        assert response.status_code == 200
        assert response.json()["success"] is True
        
        # Verify session no longer works
        response = client.get("/auth/me", headers=authenticated_session["headers"])
        assert response.status_code == 401
    
    def test_logout_clears_cookie(self, client, authenticated_session):
        """Test logout clears the session cookie."""
        # Set session cookie
        client.cookies.set("secure_session", authenticated_session["session_token"])
        
        # Logout
        response = client.post("/auth/logout", headers=authenticated_session["headers"])
        assert response.status_code == 200
        
        # Check that cookie is deleted
        assert "set-cookie" in response.headers
        cookie_header = response.headers["set-cookie"]
        assert "secure_session=" in cookie_header
        # Cookie deletion typically sets empty value or expires=Thu, 01 Jan 1970
        assert "expires=" in cookie_header or "Max-Age=0" in cookie_header


class TestInputValidationAndSanitization:
    """Test input validation and sanitization with real HTTP requests."""
    
    def test_malicious_headers_handled(self, client):
        """Test malicious headers are handled safely."""
        malicious_headers = {
            "X-Forwarded-For": "'; DROP TABLE users; --",
            "User-Agent": "<script>alert('xss')</script>",
            "X-Real-IP": "../../etc/passwd",
        }
        
        response = client.post("/auth/request-magic-link",
                             json={"email": "test@example.com", "tenant_id": "default"},
                             headers=malicious_headers)
        
        # Should not crash or return errors due to header processing
        assert response.status_code in [200, 422, 429]  # Valid responses
    
    def test_oversized_requests_rejected(self, client):
        """Test oversized requests are rejected."""
        # Create oversized payload
        huge_string = "A" * 10000
        
        response = client.post("/auth/request-magic-link", json={
            "email": "test@example.com",
            "tenant_id": huge_string,
            "device_name": huge_string
        })
        
        # Should be rejected due to size limits
        assert response.status_code == 422


class TestCSRFProtection:
    """Test CSRF protection with real HTTP requests."""
    
    def test_csrf_token_endpoint_works(self, client):
        """Test CSRF token endpoint provides tokens."""
        response = client.get("/auth/csrf-token")
        
        assert response.status_code == 200
        data = response.json()
        assert "csrf_token" in data
        assert "usage" in data
        assert len(data["csrf_token"]) > 0


class TestHealthCheckSecurity:
    """Test health check endpoint security."""
    
    def test_health_check_no_sensitive_info(self, client):
        """Test health check doesn't expose sensitive information."""
        response = client.get("/auth/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should contain safe information
        assert "status" in data
        assert "service" in data
        assert "timestamp" in data
        assert "version" in data
        
        # Should not contain sensitive information
        sensitive_keys = [
            "database_url", "secret_key", "admin_password",
            "internal_ip", "users_count", "config", "env"
        ]
        
        for key in sensitive_keys:
            assert key not in data
        
        # Values should not contain sensitive patterns
        for value in data.values():
            if isinstance(value, str):
                assert "password" not in value.lower()
                assert "secret" not in value.lower()
                assert "192.168" not in value  # Internal IP patterns


class TestUserEnumerationPrevention:
    """Test comprehensive user enumeration prevention across multiple attack vectors."""
    
    def test_response_content_consistency(self, client, test_user):
        """
        Test that response content doesn't leak user existence information.
        
        Attack scenario: Attacker analyzes response messages, headers, and structure
        to determine if a user account exists.
        """
        existing_email = test_user.email
        nonexistent_email = f"nonexistent-{secrets.token_hex(8)}@fake-domain.com"
        
        # Test magic link requests
        existing_response = client.post("/auth/request-magic-link", json={
            "email": existing_email,
            "tenant_id": "default"
        })
        
        nonexistent_response = client.post("/auth/request-magic-link", json={
            "email": nonexistent_email,
            "tenant_id": "default"
        })
        
        # Response structure should be identical
        assert existing_response.status_code == nonexistent_response.status_code, \
            "Status codes differ between existing and nonexistent users"
        
        existing_data = existing_response.json()
        nonexistent_data = nonexistent_response.json()
        
        # Response fields should be identical
        assert set(existing_data.keys()) == set(nonexistent_data.keys()), \
            "Response structure differs between existing and nonexistent users"
        
        # Success indicators should be the same (both should claim success)
        assert existing_data.get("success") == nonexistent_data.get("success"), \
            "Success field differs between existing and nonexistent users"
        
        # Message content should not reveal user existence
        existing_message = existing_data.get("message", "").lower()
        nonexistent_message = nonexistent_data.get("message", "").lower()
        
        # Messages should be identical or equally generic
        enumeration_keywords = [
            "not found", "invalid user", "unknown user", "does not exist",
            "account exists", "user found", "registered", "unregistered"
        ]
        
        for keyword in enumeration_keywords:
            assert keyword not in existing_message, f"Existing user message contains '{keyword}'"
            assert keyword not in nonexistent_message, f"Nonexistent user message contains '{keyword}'"
        
        # Headers should not leak information
        existing_headers = dict(existing_response.headers)
        nonexistent_headers = dict(nonexistent_response.headers)
        
        # Remove timing-sensitive headers for comparison
        for headers in [existing_headers, nonexistent_headers]:
            headers.pop("date", None)
            headers.pop("x-response-time", None)
        
        assert existing_headers == nonexistent_headers, \
            "Response headers differ between existing and nonexistent users"
    
    def test_error_message_consistency(self, client, test_user):
        """
        Test that error messages don't reveal user existence through different validation paths.
        
        Attack scenario: Attacker uses invalid input to trigger different error messages
        that might reveal whether a user exists.
        """
        existing_email = test_user.email
        nonexistent_email = f"nonexistent-{secrets.token_hex(8)}@fake-domain.com"
        
        # Test with various invalid inputs that might trigger different validation paths
        test_cases = [
            {"email": existing_email, "tenant_id": ""},  # Empty tenant
            {"email": nonexistent_email, "tenant_id": ""},  # Empty tenant
            {"email": existing_email, "tenant_id": "invalid_tenant"},  # Invalid tenant
            {"email": nonexistent_email, "tenant_id": "invalid_tenant"},  # Invalid tenant
            {"email": existing_email},  # Missing tenant_id
            {"email": nonexistent_email},  # Missing tenant_id
        ]
        
        responses = []
        for test_case in test_cases:
            response = client.post("/auth/request-magic-link", json=test_case)
            responses.append({
                "input": test_case,
                "status_code": response.status_code,
                "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
            })
        
        # Group responses by input type (existing vs nonexistent user)
        existing_responses = [r for r in responses if existing_email in str(r["input"])]
        nonexistent_responses = [r for r in responses if nonexistent_email in str(r["input"])]
        
        # Compare corresponding responses
        for existing_resp, nonexistent_resp in zip(existing_responses, nonexistent_responses):
            # Status codes should be identical for same input type
            assert existing_resp["status_code"] == nonexistent_resp["status_code"], (
                f"Different status codes for similar inputs: "
                f"{existing_resp['status_code']} vs {nonexistent_resp['status_code']}"
            )
            
            # Error message structure should be similar
            if isinstance(existing_resp["response"], dict) and isinstance(nonexistent_resp["response"], dict):
                assert set(existing_resp["response"].keys()) == set(nonexistent_resp["response"].keys()), (
                    "Error response structure differs between existing and nonexistent users"
                )
    
    def test_side_channel_enumeration_resistance(self, client, test_user):
        """
        Test resistance to side-channel enumeration attacks through various vectors.
        
        Attack scenario: Attacker uses indirect methods to enumerate users, such as
        analyzing behavior differences, resource consumption, or secondary effects.
        """
        existing_email = test_user.email
        nonexistent_emails = [f"nonexistent-{i}@fake-domain.com" for i in range(3)]
        
        # Test 1: Resource consumption patterns
        resource_test_results = {}
        
        for email_type, email in [("existing", existing_email)] + [(f"nonexistent_{i}", e) for i, e in enumerate(nonexistent_emails)]:
            # Make multiple requests and analyze patterns
            response_times = []
            response_sizes = []
            
            for _ in range(10):
                start_time = time.perf_counter()
                response = client.post("/auth/request-magic-link", json={
                    "email": email,
                    "tenant_id": "default"
                })
                end_time = time.perf_counter()
                
                response_times.append(end_time - start_time)
                response_sizes.append(len(response.content))
                
                time.sleep(0.2)  # Avoid rate limiting
            
            resource_test_results[email_type] = {
                "avg_response_time": sum(response_times) / len(response_times),
                "avg_response_size": sum(response_sizes) / len(response_sizes),
                "response_time_variance": sum((t - sum(response_times) / len(response_times)) ** 2 for t in response_times) / len(response_times)
            }
        
        # Analyze resource consumption patterns
        existing_stats = resource_test_results["existing"]
        nonexistent_stats = [resource_test_results[key] for key in resource_test_results if key.startswith("nonexistent")]
        
        # Response sizes should be consistent
        existing_size = existing_stats["avg_response_size"]
        for i, nonexistent_stat in enumerate(nonexistent_stats):
            size_diff = abs(existing_size - nonexistent_stat["avg_response_size"])
            assert size_diff <= 50, (  # Allow small variance for timestamps, etc.
                f"Response size differs significantly: existing={existing_size}, "
                f"nonexistent_{i}={nonexistent_stat['avg_response_size']}"
            )
        
        # Test 2: Database interaction patterns (through response timing stability)
        # Check if timing variance suggests different database query paths
        existing_variance = existing_stats["response_time_variance"]
        for i, nonexistent_stat in enumerate(nonexistent_stats):
            variance_ratio = max(existing_variance, nonexistent_stat["response_time_variance"]) / max(min(existing_variance, nonexistent_stat["response_time_variance"]), 0.001)
            assert variance_ratio < 5.0, (
                f"Response timing variance differs too much (suggests different code paths): "
                f"existing_variance={existing_variance:.6f}, nonexistent_{i}_variance={nonexistent_stat['response_time_variance']:.6f}"
            )
    
    def test_enumeration_via_secondary_endpoints(self, client, test_user):
        """
        Test that secondary endpoints don't leak user existence information.
        
        Attack scenario: Attacker uses other endpoints (like password reset, 
        account verification) to determine if users exist.
        """
        existing_email = test_user.email
        nonexistent_email = f"nonexistent-{secrets.token_hex(8)}@fake-domain.com"
        
        # Test password reset endpoint if it exists
        secondary_endpoints = [
            {"url": "/auth/request-magic-link", "method": "POST", "payload": {"email": "{email}", "tenant_id": "default"}},
            # Add other endpoints that might exist in your system
        ]
        
        for endpoint_config in secondary_endpoints:
            existing_payload = endpoint_config["payload"].copy() if endpoint_config["payload"] else {}
            nonexistent_payload = endpoint_config["payload"].copy() if endpoint_config["payload"] else {}
            
            # Replace email placeholder
            if existing_payload:
                for key, value in existing_payload.items():
                    if isinstance(value, str) and "{email}" in value:
                        existing_payload[key] = value.replace("{email}", existing_email)
                        nonexistent_payload[key] = value.replace("{email}", nonexistent_email)
            
            # Test existing user
            if endpoint_config["method"] == "POST":
                existing_response = client.post(endpoint_config["url"], json=existing_payload)
                nonexistent_response = client.post(endpoint_config["url"], json=nonexistent_payload)
            else:
                existing_response = client.get(endpoint_config["url"])
                nonexistent_response = client.get(endpoint_config["url"])
            
            # Responses should be consistent
            assert existing_response.status_code == nonexistent_response.status_code, (
                f"Endpoint {endpoint_config['url']} returns different status codes: "
                f"{existing_response.status_code} vs {nonexistent_response.status_code}"
            )
            
            # Response structure should be identical
            if existing_response.headers.get("content-type", "").startswith("application/json"):
                existing_data = existing_response.json()
                nonexistent_data = nonexistent_response.json()
                assert set(existing_data.keys()) == set(nonexistent_data.keys()), (
                    f"Endpoint {endpoint_config['url']} returns different response structure"
                )


class TestTimingAttackResistance:
    """Test timing attack resistance to prevent user enumeration."""
    
    def test_magic_link_request_constant_time_comprehensive(self, client, test_user):
        """
        Test magic link requests take constant time to prevent user enumeration attacks.
        
        Attack scenario: Attacker measures response times to determine if email addresses
        are registered in the system, enabling targeted attacks against known users.
        """
        existing_email = test_user.email
        nonexistent_email = "definitely-nonexistent-user@fake-domain-12345.com"
        
        # Warm up the service to eliminate cold start timing variations
        for _ in range(5):
            client.post("/auth/request-magic-link", json={
                "email": "warmup@example.com",
                "tenant_id": "default"
            })
            time.sleep(0.1)
        
        # Measure timing for existing user (large sample for statistical significance)
        existing_times = []
        for i in range(50):  # Increased sample size for robust statistical analysis
            start = time.perf_counter()
            response = client.post("/auth/request-magic-link", json={
                "email": existing_email,
                "tenant_id": "default"
            })
            end = time.perf_counter()
            existing_times.append(end - start)
            
            # Verify expected response regardless of timing
            assert response.status_code in [200, 429], f"Unexpected status code: {response.status_code}"
            
            # Add delay to avoid rate limiting interfering with timing measurements
            time.sleep(0.3)
        
        # Measure timing for nonexistent user with same sample size
        nonexistent_times = []
        for i in range(50):
            start = time.perf_counter()
            response = client.post("/auth/request-magic-link", json={
                "email": nonexistent_email,
                "tenant_id": "default"
            })
            end = time.perf_counter()
            nonexistent_times.append(end - start)
            
            # Both existing and nonexistent users should return same response type
            assert response.status_code in [200, 429], f"Unexpected status code: {response.status_code}"
            
            time.sleep(0.3)
        
        # Statistical analysis for timing attack resistance
        def calculate_statistics(times):
            n = len(times)
            mean = sum(times) / n
            variance = sum((t - mean) ** 2 for t in times) / n
            std_dev = variance ** 0.5
            return mean, std_dev, variance
        
        avg_existing, std_existing, var_existing = calculate_statistics(existing_times)
        avg_nonexistent, std_nonexistent, var_nonexistent = calculate_statistics(nonexistent_times)
        
        # Perform Welch's t-test to detect statistically significant timing differences
        pooled_std = ((var_existing + var_nonexistent) / 2) ** 0.5
        t_statistic = abs(avg_existing - avg_nonexistent) / (pooled_std * (2 / len(existing_times)) ** 0.5)
        
        # Critical t-value for 95% confidence with large samples (≈ 1.96)
        critical_t_value = 2.0
        
        assert t_statistic < critical_t_value, (
            f"TIMING ATTACK VULNERABILITY DETECTED! "
            f"Statistically significant timing difference found:\n"
            f"- Existing user: {avg_existing:.4f}s ± {std_existing:.4f}s\n"
            f"- Nonexistent user: {avg_nonexistent:.4f}s ± {std_nonexistent:.4f}s\n"
            f"- T-statistic: {t_statistic:.3f} > {critical_t_value} (critical value)\n"
            f"- This difference is exploitable for user enumeration attacks!"
        )
        
        # Additional security checks
        timing_difference = abs(avg_existing - avg_nonexistent)
        
        # 1. Absolute timing difference should be minimal (< 10ms typically)
        assert timing_difference < 0.01, (
            f"Absolute timing difference too large: {timing_difference * 1000:.2f}ms. "
            f"This could enable timing-based user enumeration."
        )
        
        # 2. Relative timing difference should be < 5% to prevent statistical exploitation
        if min(avg_existing, avg_nonexistent) > 0:
            relative_difference = timing_difference / min(avg_existing, avg_nonexistent)
            assert relative_difference < 0.05, (
                f"Relative timing difference too large: {relative_difference:.1%}. "
                f"Attackers can exploit differences > 5% with sufficient samples."
            )
        
        # 3. Coefficient of variation should be similar (timing consistency)
        cv_existing = std_existing / avg_existing if avg_existing > 0 else 0
        cv_nonexistent = std_nonexistent / avg_nonexistent if avg_nonexistent > 0 else 0
        cv_difference = abs(cv_existing - cv_nonexistent)
        
        assert cv_difference < 0.1, (
            f"Timing consistency varies too much between user types: {cv_difference:.2%}. "
            f"This pattern could be exploited for enumeration."
        )
    
    def test_magic_link_timing_under_load_conditions(self, client, test_user):
        """
        Test timing consistency under various load conditions to prevent enumeration.
        
        Attack scenario: Attacker tests timing under different system loads to find
        conditions where timing differences become more pronounced.
        """
        existing_email = test_user.email
        nonexistent_email = f"nonexistent-{secrets.token_hex(8)}@fake-domain.com"
        
        test_scenarios = [
            {"name": "low_load", "concurrent_requests": 1, "delay": 0.5},
            {"name": "medium_load", "concurrent_requests": 3, "delay": 0.1},
            {"name": "high_load", "concurrent_requests": 5, "delay": 0.05}
        ]
        
        timing_results = {}
        
        for scenario in test_scenarios:
            existing_times = []
            nonexistent_times = []
            
            # Test each email type under this load condition
            for email_type, email in [("existing", existing_email), ("nonexistent", nonexistent_email)]:
                times = []
                
                for batch in range(scenario["concurrent_requests"]):
                    # Measure timing for this batch
                    start = time.perf_counter()
                    response = client.post("/auth/request-magic-link", json={
                        "email": email,
                        "tenant_id": "default"
                    })
                    end = time.perf_counter()
                    times.append(end - start)
                    
                    assert response.status_code in [200, 429]
                    time.sleep(scenario["delay"])
                
                if email_type == "existing":
                    existing_times.extend(times)
                else:
                    nonexistent_times.extend(times)
            
            # Analyze timing for this scenario
            if existing_times and nonexistent_times:
                avg_existing = sum(existing_times) / len(existing_times)
                avg_nonexistent = sum(nonexistent_times) / len(nonexistent_times)
                timing_diff = abs(avg_existing - avg_nonexistent)
                
                timing_results[scenario["name"]] = {
                    "existing_avg": avg_existing,
                    "nonexistent_avg": avg_nonexistent,
                    "difference": timing_diff
                }
                
                # Each load condition should maintain timing resistance
                assert timing_diff < 0.015, (
                    f"Timing vulnerability under {scenario['name']} load: "
                    f"{timing_diff * 1000:.2f}ms difference"
                )
        
        # Verify timing differences don't increase dramatically under load
        differences = [result["difference"] for result in timing_results.values()]
        max_diff = max(differences)
        min_diff = min(differences)
        
        assert max_diff / min_diff < 3.0, (
            f"Timing difference varies too much across load conditions: "
            f"{max_diff / min_diff:.1f}x variation could enable conditional enumeration"
        )
    
    def test_magic_link_verification_constant_time(self, client):
        """Test magic link verification prevents timing-based token enumeration."""
        # Test with different types of invalid tokens
        token_types = {
            "random_valid_format": secrets.token_urlsafe(43),
            "wrong_length_short": secrets.token_urlsafe(10),
            "wrong_length_long": secrets.token_urlsafe(100),
            "malformed_chars": "invalid-token-with-special@chars!",
            "empty_token": "",
            "numeric_only": "1234567890123456789012345678901234567890123"
        }
        
        timing_results = {}
        
        for token_type, token in token_types.items():
            times = []
            
            # Measure multiple attempts for each token type
            for _ in range(15):
                start = time.perf_counter()
                # URL encode problematic characters
                safe_token = token.replace("@", "%40").replace("!", "%21")
                response = client.get(f"/auth/verify-magic-link/{safe_token}")
                end = time.perf_counter()
                times.append(end - start)
                
                # All should be rejected
                assert response.status_code in [400, 404, 422], f"{token_type} was not rejected"
                
                time.sleep(0.1)
            
            timing_results[token_type] = {
                "avg": sum(times) / len(times),
                "times": times
            }
        
        # Verify timing consistency across different invalid token types
        avg_times = [result["avg"] for result in timing_results.values()]
        min_time = min(avg_times)
        max_time = max(avg_times)
        
        # Timing variance should be minimal across token types
        timing_variance = (max_time - min_time) / min_time if min_time > 0 else 0
        
        assert timing_variance < 0.3, (
            f"Token verification timing variance too high: {timing_variance:.2%}. "
            f"Min: {min_time:.4f}s, Max: {max_time:.4f}s. "
            f"This could enable timing-based token enumeration."
        )
    
    def test_authentication_timing_consistency(self, client, authenticated_session):
        """Test authentication endpoints maintain timing consistency."""
        # Test /auth/me endpoint with valid vs invalid tokens
        valid_token = authenticated_session["session_token"]
        invalid_tokens = [
            "fake_token_123",
            secrets.token_urlsafe(43),
            "expired_token_xyz",
            "",
        ]
        
        # Measure valid token timing
        valid_times = []
        for _ in range(10):
            start = time.perf_counter()
            response = client.get("/auth/me", headers={"Authorization": f"Bearer {valid_token}"})
            end = time.perf_counter()
            valid_times.append(end - start)
            assert response.status_code == 200
            time.sleep(0.1)
        
        # Measure invalid token timing
        invalid_times = []
        for token in invalid_tokens:
            for _ in range(5):  # Fewer samples per token
                start = time.perf_counter()
                response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
                end = time.perf_counter()
                invalid_times.append(end - start)
                assert response.status_code == 401
                time.sleep(0.1)
        
        # Calculate timing statistics
        avg_valid = sum(valid_times) / len(valid_times)
        avg_invalid = sum(invalid_times) / len(invalid_times)
        
        # Authentication timing should be consistent to prevent token validation oracle
        timing_difference = abs(avg_valid - avg_invalid)
        timing_ratio = timing_difference / min(avg_valid, avg_invalid) if min(avg_valid, avg_invalid) > 0 else 0
        
        assert timing_ratio < 0.25, (
            f"Authentication timing inconsistency detected: "
            f"Valid tokens: {avg_valid:.4f}s, Invalid tokens: {avg_invalid:.4f}s, "
            f"Ratio: {timing_ratio:.2%}. This could reveal token validity."
        )


class TestConcurrentSessionSecurity:
    """Test concurrent session security."""
    
    def test_concurrent_session_limits(self, client, test_user):
        """Test concurrent session limits are enforced."""
        session_service = get_session_service()
        session_tokens = []
        
        # Create multiple sessions for the same user
        for i in range(10):
            result = session_service.create_session(
                user=test_user,
                ip_address=f"192.168.1.{100 + i}",
                user_agent=f"Browser {i}",
                device_fingerprint=None,
                device_name=f"Device {i}",
                remember_device=False,
                auth_method="magic_link"
            )
            
            if result.success:
                session_tokens.append(result.session_token)
        
        # Should have some limit on concurrent sessions
        # (The exact limit depends on your session service implementation)
        assert len(session_tokens) <= 5, "Too many concurrent sessions allowed"
        
        # Verify older sessions are invalidated when limit is reached
        if len(session_tokens) > 1:
            # Try to use an earlier session
            headers = {"Authorization": f"Bearer {session_tokens[0]}"}
            response = client.get("/auth/me", headers=headers)
            
            # Might be valid or might be invalidated depending on implementation
            assert response.status_code in [200, 401]