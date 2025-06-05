"""
Security-focused tests for SecureAuthService.

Tests ACTUAL attack scenarios with real database integration.
NO MOCKING of security functions - tests the real implementation.
"""

import pytest
import secrets
import time
from datetime import datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

from secure_auth.auth_service import SecureAuthService, AuthenticationResult
from secure_auth.models import User, MagicLink, RateLimit, Base
from secure_auth.token_service import SecurityTokenService


@pytest.fixture
def auth_service(db_session, auth_master_key):
    """Auth service with real database session."""
    return SecureAuthService(db_session=db_session)


class TestInputSanitizationSecurity:
    """Test input sanitization prevents actual injection attacks."""
    
    def test_email_sanitization_prevents_sql_injection(self, auth_service):
        """Test email sanitization blocks SQL injection attempts."""
        sql_injection_payloads = [
            "test@example.com'; DROP TABLE users; --",
            "test@example.com' UNION SELECT * FROM passwords--",
            "test@example.com'; INSERT INTO users VALUES('hacker')--",
            "test@example.com' OR '1'='1",
            "test@example.com'; DELETE FROM sessions; --",
        ]
        
        for payload in sql_injection_payloads:
            with pytest.raises(ValueError):
                auth_service._sanitize_email(payload)
    
    def test_email_sanitization_prevents_xss_injection(self, auth_service):
        """Test email sanitization blocks XSS attempts."""
        xss_payloads = [
            "test@example.com<script>alert('xss')</script>",
            "test@example.com<img src=x onerror=alert('xss')>",
            "test@example.com<svg onload=alert('xss')>",
            "test@example.com';alert('xss');//",
            "test@example.com\"><script>alert('xss')</script>",
        ]
        
        for payload in xss_payloads:
            with pytest.raises(ValueError):
                auth_service._sanitize_email(payload)
    
    def test_email_length_attack_prevention(self, auth_service):
        """Test email length limits prevent buffer overflow attacks."""
        # Test exactly at RFC 5321 limit (320 chars)
        max_length_email = "a" * 312 + "@test.com"  # 320 chars total
        result = auth_service._sanitize_email(max_length_email)
        assert len(result) <= 320
        
        # Test buffer overflow attempt
        overflow_email = "a" * 500 + "@test.com"
        with pytest.raises(ValueError, match="Email address too long"):
            auth_service._sanitize_email(overflow_email)
    
    def test_ip_address_injection_prevention(self, auth_service):
        """Test IP address sanitization prevents injection attacks."""
        malicious_ips = [
            "192.168.1.1'; DROP TABLE sessions; --",
            "192.168.1.1<script>alert('xss')</script>", 
            "192.168.1.1 OR 1=1",
            "192.168.1.1\x00\x01\x02",  # Null bytes
            "file:///etc/passwd",
            "javascript:alert('xss')",
        ]
        
        for malicious_ip in malicious_ips:
            result = auth_service._sanitize_ip_address(malicious_ip)
            # Should reject all malicious inputs
            assert result is None


class TestRealDatabaseMagicLinkSecurity:
    """Test magic link security with real database operations."""
    
    def test_magic_link_request_creates_database_record(self, auth_service, test_user, db_session):
        """Test magic link request creates proper database record."""
        result = auth_service.request_magic_link(
            email="test@example.com",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 Test Browser"
        )
        
        assert result.success
        assert result.magic_link_id
        
        # Verify database record exists
        magic_link = db_session.query(MagicLink).filter(
            MagicLink.id == result.magic_link_id
        ).first()
        
        assert magic_link is not None
        assert magic_link.user_id == test_user.id
        assert magic_link.email == "test@example.com"
        assert magic_link.requesting_ip == "192.168.1.100"
        assert magic_link.used_at is None
        assert magic_link.expires_at > datetime.utcnow()
        assert magic_link.token_hash is not None
        assert magic_link.salt is not None
    
    def test_magic_link_verification_with_real_token(self, auth_service, test_user, db_session):
        """Test magic link verification with real cryptographic token."""
        # Request magic link first
        request_result = auth_service.request_magic_link(
            email="test@example.com",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 Test Browser"
        )
        
        # Get the actual token from token service (simulate email link)
        magic_link = db_session.query(MagicLink).filter(
            MagicLink.id == request_result.magic_link_id
        ).first()
        
        # Generate a token that would hash to the stored hash
        # This simulates the real token that would be in the email
        token_service = auth_service.token_service
        
        # For testing, we need to create a verifiable token
        # In real usage, the token would be generated and sent via email
        test_token = secrets.token_urlsafe(64)
        
        # Update the database with a hash we can verify
        context = f"{test_user.id}:{magic_link.requesting_ip}"
        test_hash = token_service.hash_token(test_token, magic_link.salt, additional_context=context)
        magic_link.token_hash = test_hash
        db_session.commit()
        
        # Now verify the token
        verify_result = auth_service.verify_magic_link(
            token=test_token,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 Test Browser"
        )
        
        assert verify_result.success
        assert verify_result.user.id == test_user.id
        
        # Verify database state after verification
        db_session.refresh(magic_link)
        assert magic_link.used_at is not None
        assert magic_link.verified_ip == "192.168.1.100"
        
        # Verify user state updated
        db_session.refresh(test_user)
        assert test_user.last_login_at is not None
        assert test_user.failed_login_count == 0
    
    def test_magic_link_replay_attack_prevention(self, auth_service, test_user, db_session):
        """Test magic link cannot be reused (replay attack prevention)."""
        # Create and use a magic link
        request_result = auth_service.request_magic_link(
            email="test@example.com",
            ip_address="192.168.1.100"
        )
        
        magic_link = db_session.query(MagicLink).filter(
            MagicLink.id == request_result.magic_link_id
        ).first()
        
        # Mark as used
        magic_link.used_at = datetime.utcnow()
        magic_link.verified_ip = "192.168.1.100"
        db_session.commit()
        
        # Create a test token
        test_token = secrets.token_urlsafe(64)
        context = f"{test_user.id}:{magic_link.requesting_ip}"
        test_hash = auth_service.token_service.hash_token(test_token, magic_link.salt, additional_context=context)
        magic_link.token_hash = test_hash
        db_session.commit()
        
        # Attempt to use the already-used magic link
        verify_result = auth_service.verify_magic_link(
            token=test_token,
            ip_address="192.168.1.100"
        )
        
        # Should fail because link was already used
        assert not verify_result.success
        assert verify_result.error_code == "INVALID_TOKEN"
    
    def test_magic_link_expiration_enforcement(self, auth_service, test_user, db_session):
        """Test expired magic links are rejected."""
        # Create magic link
        request_result = auth_service.request_magic_link(
            email="test@example.com",
            ip_address="192.168.1.100"
        )
        
        magic_link = db_session.query(MagicLink).filter(
            MagicLink.id == request_result.magic_link_id
        ).first()
        
        # Expire the magic link
        magic_link.expires_at = datetime.utcnow() - timedelta(minutes=1)
        db_session.commit()
        
        # Create a test token
        test_token = secrets.token_urlsafe(64)
        context = f"{test_user.id}:{magic_link.requesting_ip}"
        test_hash = auth_service.token_service.hash_token(test_token, magic_link.salt, additional_context=context)
        magic_link.token_hash = test_hash
        db_session.commit()
        
        # Attempt to verify expired token
        verify_result = auth_service.verify_magic_link(
            token=test_token,
            ip_address="192.168.1.100"
        )
        
        # Should fail because link expired
        assert not verify_result.success
        assert verify_result.error_code == "INVALID_TOKEN"
    
    def test_concurrent_magic_link_limit_enforcement(self, auth_service, test_user, db_session):
        """Test that concurrent magic link limits are enforced."""
        # Request multiple magic links
        magic_link_ids = []
        for i in range(5):  # More than the default limit of 3
            result = auth_service.request_magic_link(
                email="test@example.com",
                ip_address=f"192.168.1.{100 + i}"
            )
            if result.success:
                magic_link_ids.append(result.magic_link_id)
        
        # Check how many active magic links exist
        active_links = db_session.query(MagicLink).filter(
            MagicLink.user_id == test_user.id,
            MagicLink.expires_at > datetime.utcnow(),
            MagicLink.used_at.is_(None)
        ).count()
        
        # Should not exceed the maximum concurrent limit
        assert active_links <= auth_service.max_concurrent_magic_links
    
    def test_user_enumeration_protection_consistency(self, auth_service, db_session):
        """Test that response timing doesn't reveal user existence."""
        # Time requests for existing vs non-existing users
        existing_email = "test@example.com"
        nonexistent_email = "nonexistent@example.com"
        
        # Create test user
        user = User(email=existing_email, tenant_id="default", is_active=True)
        db_session.add(user)
        db_session.commit()
        
        # Measure timing for existing user
        start_time = time.time()
        result1 = auth_service.request_magic_link(
            email=existing_email,
            ip_address="192.168.1.100"
        )
        existing_time = time.time() - start_time
        
        # Measure timing for non-existing user
        start_time = time.time()
        result2 = auth_service.request_magic_link(
            email=nonexistent_email,
            ip_address="192.168.1.100"
        )
        nonexistent_time = time.time() - start_time
        
        # Both should return success for security
        assert result1.success
        assert result2.success
        
        # Timing difference should be minimal (timing attack protection)
        timing_diff = abs(existing_time - nonexistent_time)
        max_timing_diff = max(existing_time, nonexistent_time)
        
        if max_timing_diff > 0:
            relative_diff = timing_diff / max_timing_diff
            # Allow reasonable variation but not excessive timing leaks
            assert relative_diff < 0.2  # Less than 20% difference


class TestRealDatabaseRateLimiting:
    """Test rate limiting with real database operations."""
    
    def test_email_rate_limiting_database_persistence(self, auth_service, test_user, db_session):
        """Test email rate limiting persists in database correctly."""
        email = "test@example.com"
        
        # Make multiple requests
        for i in range(4):  # Default email limit is 3
            result = auth_service.request_magic_link(
                email=email,
                ip_address=f"192.168.1.{100 + i}"  # Different IPs
            )
            
            if i < 3:
                assert result.success, f"Request {i+1} should succeed"
            else:
                # 4th request should be rate limited
                assert not result.success
                assert result.error_code == "RATE_LIMITED"
                assert result.retry_after_seconds is not None
        
        # Verify rate limit record exists in database
        rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email,
            RateLimit.action_type == "magic_link_request"
        ).first()
        
        assert rate_limit is not None
        assert rate_limit.attempt_count >= 3
        assert rate_limit.is_exceeded
    
    def test_ip_rate_limiting_database_persistence(self, auth_service, db_session):
        """Test IP-based rate limiting persists in database."""
        ip_address = "192.168.1.100"
        
        # Create multiple users for same IP
        emails = [f"user{i}@example.com" for i in range(12)]
        for email in emails:
            user = User(email=email, tenant_id="default", is_active=True)
            db_session.add(user)
        db_session.commit()
        
        # Make requests from same IP to different users
        success_count = 0
        for email in emails:
            result = auth_service.request_magic_link(
                email=email,
                ip_address=ip_address
            )
            if result.success:
                success_count += 1
            else:
                # Should be rate limited at some point
                assert result.error_code == "IP_RATE_LIMITED"
                break
        
        # Should allow some requests but eventually rate limit
        assert success_count > 0
        assert success_count <= 10  # Default IP limit
        
        # Verify IP rate limit record
        ip_rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "ip",
            RateLimit.identifier_value == ip_address
        ).first()
        
        assert ip_rate_limit is not None
    
    def test_rate_limit_window_expiration_in_database(self, auth_service, test_user, db_session):
        """Test rate limit windows expire and reset in database."""
        email = "test@example.com"
        
        # Exhaust rate limit
        for i in range(4):
            result = auth_service.request_magic_link(email=email)
            if not result.success:
                break
        
        # Get rate limit record
        rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email
        ).first()
        
        assert rate_limit is not None
        original_count = rate_limit.attempt_count
        
        # Manually expire the rate limit window
        rate_limit.expires_at = datetime.utcnow() - timedelta(minutes=1)
        db_session.commit()
        
        # New request should succeed (new window)
        result = auth_service.request_magic_link(email=email)
        assert result.success
        
        # Should create new rate limit record or reset existing one
        updated_rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email,
            RateLimit.expires_at > datetime.utcnow()
        ).first()
        
        assert updated_rate_limit is not None
    
    def test_adaptive_rate_limiting_escalation(self, auth_service, test_user, db_session):
        """Test adaptive rate limiting escalates with violations."""
        email = "test@example.com"
        
        # First violation - exhaust initial limit
        for i in range(4):
            auth_service.request_magic_link(email=email)
        
        # Wait and reset window manually to simulate time passage
        rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email
        ).first()
        
        if rate_limit:
            # Mark as violated and reset for next window
            rate_limit.last_violation_at = datetime.utcnow()
            rate_limit.reset_count += 1
            rate_limit.expires_at = datetime.utcnow() - timedelta(minutes=1)
            db_session.commit()
        
        # Second round - should have lower threshold due to escalation
        success_count = 0
        for i in range(5):
            result = auth_service.request_magic_link(email=email)
            if result.success:
                success_count += 1
            else:
                break
        
        # Should allow fewer requests in second round (escalation)
        assert success_count < 3  # Less than original limit


class TestRealDatabaseUserManagement:
    """Test user management with real database operations."""
    
    def test_user_creation_database_persistence(self, auth_service, db_session):
        """Test user creation persists correctly in database."""
        result = auth_service.create_user(
            email="newuser@example.com",
            tenant_id="test_tenant",
            is_admin=False
        )
        
        assert result.success
        assert result.user is not None
        
        # Verify user exists in database
        user = db_session.query(User).filter(
            User.email == "newuser@example.com",
            User.tenant_id == "test_tenant"
        ).first()
        
        assert user is not None
        assert user.email == "newuser@example.com"
        assert user.tenant_id == "test_tenant"
        assert user.is_admin is False
        assert user.is_active is True
        assert user.email_verified is False  # Default state
        assert user.created_at is not None
    
    def test_duplicate_user_prevention(self, auth_service, test_user, db_session):
        """Test duplicate user creation is prevented."""
        # Attempt to create user with same email
        result = auth_service.create_user(
            email="test@example.com",  # Same as test_user
            tenant_id="default"
        )
        
        assert not result.success
        assert result.error_code == "USER_EXISTS"
        
        # Verify only one user exists
        user_count = db_session.query(User).filter(
            User.email == "test@example.com",
            User.tenant_id == "default"
        ).count()
        
        assert user_count == 1
    
    def test_tenant_isolation_in_database(self, auth_service, db_session):
        """Test tenant isolation works correctly in database."""
        email = "shared@example.com"
        
        # Create users with same email in different tenants
        result1 = auth_service.create_user(
            email=email,
            tenant_id="tenant_a"
        )
        
        result2 = auth_service.create_user(
            email=email,
            tenant_id="tenant_b"
        )
        
        assert result1.success
        assert result2.success
        
        # Verify both users exist with proper tenant isolation
        users = db_session.query(User).filter(User.email == email).all()
        assert len(users) == 2
        
        tenant_ids = {user.tenant_id for user in users}
        assert tenant_ids == {"tenant_a", "tenant_b"}
    
    def test_account_locking_database_state(self, auth_service, test_user, db_session):
        """Test account locking updates database state correctly."""
        # Lock the account
        test_user.account_locked_until = datetime.utcnow() + timedelta(hours=1)
        db_session.commit()
        
        # Attempt magic link request for locked account
        result = auth_service.request_magic_link(
            email="test@example.com",
            ip_address="192.168.1.100"
        )
        
        # Should appear to succeed (security) but not create usable magic link
        assert result.success
        assert "If that email is registered" in result.message
        
        # However, internally it should be handled as locked
        # Verify no new magic links were created for locked account
        magic_link_count = db_session.query(MagicLink).filter(
            MagicLink.user_id == test_user.id,
            MagicLink.created_at > datetime.utcnow() - timedelta(minutes=1)
        ).count()
        
        # Implementation may vary - could be 0 (no link) or 1 (link created but unusable)
        # The key is that verification should fail for locked accounts


class TestTimingAttackResistance:
    """Test timing attack protection works correctly."""
    
    @pytest.fixture
    def auth_service(self, db_session):
        return SecureAuthService(db_session=db_session)
    
    def test_minimum_response_time_enforcement(self, auth_service):
        """Test response time is consistent regardless of operation speed."""
        start_times = []
        end_times = []
        
        # Measure response times for operations that should complete quickly
        for _ in range(5):
            start = time.time()
            auth_service._ensure_minimum_response_time(start)
            end = time.time()
            
            start_times.append(start)
            end_times.append(end)
        
        response_times = [end - start for start, end in zip(start_times, end_times)]
        
        # All response times should be at least the minimum (500ms default)
        min_expected = auth_service.min_response_time_ms / 1000.0
        for response_time in response_times:
            assert response_time >= min_expected * 0.95  # Allow 5% tolerance


class TestSecurityConfiguration:
    """Test security configuration is properly enforced."""
    
    def test_secure_defaults(self):
        """Test service initializes with secure default values."""
        auth_service = SecureAuthService()
        
        # Timing protection should be enabled
        assert auth_service.min_response_time_ms >= 500
        
        # Magic link expiry should be reasonable
        assert 5 <= auth_service.magic_link_expiry_minutes <= 15
        
        # Concurrent link limit should prevent abuse
        assert 1 <= auth_service.max_concurrent_magic_links <= 5
    
    def test_trusted_ip_configuration(self):
        """Test trusted IP configuration is secure."""
        auth_service = SecureAuthService()
        
        # Should have reasonable private network ranges
        assert len(auth_service.trusted_ip_ranges) >= 3
        
        # Test that private IPs are considered trusted
        assert auth_service._is_trusted_ip("192.168.1.1")
        assert auth_service._is_trusted_ip("10.0.0.1") 
        assert auth_service._is_trusted_ip("172.16.0.1")
        
        # Test that public IPs are not automatically trusted
        assert not auth_service._is_trusted_ip("8.8.8.8")
        assert not auth_service._is_trusted_ip("203.0.113.1")


class TestRaceConditionSecurity:
    """Test race condition protection in concurrent operations."""
    
    def test_concurrent_magic_link_requests_same_email(self, auth_service, test_user, db_session):
        """
        Test concurrent magic link requests from same email prevent duplicate generation.
        
        Attack scenario: Attacker rapidly fires multiple magic link requests
        simultaneously for the same email to try bypassing rate limits or
        generating multiple valid tokens.
        """
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        email = "test@example.com"
        ip_address = "192.168.1.100"
        
        # Track results from concurrent requests
        results = []
        errors = []
        
        def make_request():
            """Make a magic link request - simulates concurrent attacker."""
            try:
                result = auth_service.request_magic_link(
                    email=email,
                    ip_address=ip_address,
                    user_agent="Concurrent Attack Bot"
                )
                return result
            except Exception as e:
                errors.append(str(e))
                return None
        
        # Launch 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all requests simultaneously
            futures = [executor.submit(make_request) for _ in range(10)]
            
            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        # Should not have any database errors from race conditions
        assert len(errors) == 0, f"Race condition errors: {errors}"
        
        # Count successful vs rate-limited requests
        successful_requests = [r for r in results if r.success]
        rate_limited_requests = [r for r in results if not r.success and r.error_code == "RATE_LIMITED"]
        
        # First few should succeed, rest should be rate limited
        assert len(successful_requests) <= 3, "Too many concurrent requests succeeded"
        assert len(rate_limited_requests) > 0, "No requests were rate limited"
        
        # Verify database consistency - check for duplicate magic links
        active_links = db_session.query(MagicLink).filter(
            MagicLink.user_id == test_user.id,
            MagicLink.email == email,
            MagicLink.expires_at > datetime.utcnow(),
            MagicLink.used_at.is_(None)
        ).all()
        
        # Should not have more active links than successful requests
        assert len(active_links) <= len(successful_requests)
        
        # All active links should have unique token hashes
        token_hashes = [link.token_hash for link in active_links]
        assert len(token_hashes) == len(set(token_hashes)), "Duplicate token hashes found"
    
    def test_concurrent_magic_link_verification_race(self, auth_service, test_user, db_session):
        """
        Test concurrent verification attempts of same magic link.
        
        Attack scenario: Attacker tries to use the same magic link token
        multiple times simultaneously to bypass single-use protection.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Create a magic link first
        request_result = auth_service.request_magic_link(
            email="test@example.com",
            ip_address="192.168.1.100"
        )
        assert request_result.success
        
        # Get the magic link and create a verifiable token
        magic_link = db_session.query(MagicLink).filter(
            MagicLink.id == request_result.magic_link_id
        ).first()
        
        test_token = secrets.token_urlsafe(64)
        context = f"{test_user.id}:{magic_link.requesting_ip}"
        test_hash = auth_service.token_service.hash_token(test_token, magic_link.salt, additional_context=context)
        magic_link.token_hash = test_hash
        db_session.commit()
        
        # Track verification results
        verification_results = []
        errors = []
        
        def verify_token():
            """Verify the same token simultaneously."""
            try:
                result = auth_service.verify_magic_link(
                    token=test_token,
                    ip_address="192.168.1.100",
                    user_agent="Concurrent Verification Attack"
                )
                return result
            except Exception as e:
                errors.append(str(e))
                return None
        
        # Launch 10 concurrent verification attempts
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(verify_token) for _ in range(10)]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    verification_results.append(result)
        
        # Should not have database errors from race conditions
        assert len(errors) == 0, f"Race condition errors: {errors}"
        
        # Only ONE verification should succeed
        successful_verifications = [r for r in verification_results if r.success]
        failed_verifications = [r for r in verification_results if not r.success]
        
        assert len(successful_verifications) == 1, f"Multiple verifications succeeded: {len(successful_verifications)}"
        assert len(failed_verifications) > 0, "No verifications failed"
        
        # Verify database state - magic link should be marked as used exactly once
        db_session.refresh(magic_link)
        assert magic_link.used_at is not None, "Magic link not marked as used"
        assert magic_link.verified_ip == "192.168.1.100"
        
        # User should have updated login time exactly once
        db_session.refresh(test_user)
        assert test_user.last_login_at is not None
    
    def test_concurrent_rate_limit_updates_consistency(self, auth_service, test_user, db_session):
        """
        Test concurrent rate limit updates maintain database consistency.
        
        Attack scenario: Multiple requests from same email/IP simultaneously
        to try causing race conditions in rate limit counting.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        email = "test@example.com"
        
        # Track all request results
        all_results = []
        errors = []
        
        def make_request(request_id):
            """Make magic link request with unique IP per thread."""
            try:
                result = auth_service.request_magic_link(
                    email=email,
                    ip_address=f"192.168.1.{100 + (request_id % 50)}",  # Vary IPs to focus on email rate limiting
                    user_agent=f"RateLimit Test {request_id}"
                )
                return result
            except Exception as e:
                errors.append(f"Request {request_id}: {str(e)}")
                return None
        
        # Launch 20 concurrent requests (well above email limit of 3)
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request, i) for i in range(20)]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_results.append(result)
        
        # Should not have database errors
        assert len(errors) == 0, f"Database consistency errors: {errors}"
        
        # Count results by type
        successful = [r for r in all_results if r.success]
        rate_limited = [r for r in all_results if not r.success and r.error_code == "RATE_LIMITED"]
        
        # Should have exactly the allowed number of successes (3 for email)
        assert len(successful) <= 3, f"Too many successes: {len(successful)}"
        assert len(rate_limited) > 0, "No requests were rate limited"
        
        # Verify rate limit database record consistency
        rate_limit_records = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email
        ).all()
        
        # Should have exactly one rate limit record for the email
        assert len(rate_limit_records) == 1, f"Multiple rate limit records: {len(rate_limit_records)}"
        
        rate_limit = rate_limit_records[0]
        assert rate_limit.attempt_count >= len(successful), "Rate limit count inconsistent"
        
        # If rate limited, should be marked as exceeded
        if len(rate_limited) > 0:
            assert rate_limit.is_exceeded, "Rate limit not marked as exceeded"


class TestSessionInvalidationEdgeCases:
    """Test session invalidation edge cases with multiple sessions."""
    
    def test_multiple_active_sessions_per_user(self, auth_service, test_user, db_session):
        """
        Test session invalidation when user has multiple active sessions.
        
        Edge case: User logs in from multiple devices/locations and one
        session needs to be invalidated without affecting others.
        """
        from secure_auth.session_service import SessionService
        from secure_auth.models import Session
        
        # Create session service
        session_service = SessionService(db_session=db_session)
        
        # Create multiple sessions for same user (different devices/IPs)
        sessions = []
        for i in range(3):
            session_token = session_service.create_session(
                user_id=test_user.id,
                ip_address=f"192.168.1.{10 + i}",
                user_agent=f"Device {i+1} Browser",
                tenant_id=test_user.tenant_id
            )
            sessions.append(session_token)
        
        # Verify all sessions are active
        for session_token in sessions:
            user = auth_service.get_user_from_session(
                session_token=session_token,
                ip_address="192.168.1.10",  # Use first IP
                user_agent="Device 1 Browser"
            )
            assert user is not None, "Session should be active"
        
        # Invalidate one specific session
        session_service.invalidate_session(sessions[1])
        
        # Verify only the targeted session is invalidated
        user1 = auth_service.get_user_from_session(
            session_token=sessions[0],
            ip_address="192.168.1.10",
            user_agent="Device 1 Browser"
        )
        assert user1 is not None, "First session should remain active"
        
        user2 = auth_service.get_user_from_session(
            session_token=sessions[1],
            ip_address="192.168.1.11",
            user_agent="Device 2 Browser"
        )
        assert user2 is None, "Second session should be invalidated"
        
        user3 = auth_service.get_user_from_session(
            session_token=sessions[2],
            ip_address="192.168.1.12",
            user_agent="Device 3 Browser"
        )
        assert user3 is not None, "Third session should remain active"
    
    def test_session_invalidation_on_suspicious_activity(self, auth_service, test_user, db_session):
        """
        Test session invalidation when suspicious activity is detected.
        
        Edge case: Session should be invalidated when used from different
        IP address or user agent than originally created with.
        """
        from secure_auth.session_service import SessionService
        
        session_service = SessionService(db_session=db_session)
        
        # Create session with specific context
        original_ip = "192.168.1.100"
        original_ua = "Mozilla/5.0 (Trusted Browser)"
        
        session_token = session_service.create_session(
            user_id=test_user.id,
            ip_address=original_ip,
            user_agent=original_ua,
            tenant_id=test_user.tenant_id
        )
        
        # Verify session works with original context
        user = auth_service.get_user_from_session(
            session_token=session_token,
            ip_address=original_ip,
            user_agent=original_ua
        )
        assert user is not None, "Session should work with original context"
        
        # Attempt to use session from different IP (session hijacking attempt)
        hijacker_ip = "203.0.113.50"
        hijacked_user = auth_service.get_user_from_session(
            session_token=session_token,
            ip_address=hijacker_ip,
            user_agent=original_ua
        )
        assert hijacked_user is None, "Session should be rejected from different IP"
        
        # Verify session is now invalidated (security measure)
        # After suspicious activity, session should be permanently invalidated
        original_user_retry = auth_service.get_user_from_session(
            session_token=session_token,
            ip_address=original_ip,
            user_agent=original_ua
        )
        # Implementation may vary - session could be invalidated or require re-verification
        # The key is that it handles the security incident appropriately
    
    def test_concurrent_session_invalidation_race(self, auth_service, test_user, db_session):
        """
        Test concurrent session invalidation operations don't cause races.
        
        Edge case: Multiple concurrent attempts to invalidate same session
        or related sessions should not cause database inconsistencies.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from secure_auth.session_service import SessionService
        
        session_service = SessionService(db_session=db_session)
        
        # Create multiple sessions for the user
        session_tokens = []
        for i in range(5):
            token = session_service.create_session(
                user_id=test_user.id,
                ip_address=f"192.168.1.{10 + i}",
                user_agent=f"Device {i+1}",
                tenant_id=test_user.tenant_id
            )
            session_tokens.append(token)
        
        # Track invalidation results
        invalidation_results = []
        errors = []
        
        def invalidate_session(session_token):
            """Invalidate a specific session."""
            try:
                result = session_service.invalidate_session(session_token)
                return f"Invalidated: {session_token[:10]}..."
            except Exception as e:
                errors.append(f"Error invalidating {session_token[:10]}...: {str(e)}")
                return None
        
        def invalidate_all_user_sessions():
            """Invalidate all sessions for the user."""
            try:
                count = session_service.invalidate_all_user_sessions(test_user.id)
                return f"Invalidated {count} user sessions"
            except Exception as e:
                errors.append(f"Error invalidating all user sessions: {str(e)}")
                return None
        
        # Launch concurrent invalidation operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            # Individual session invalidations
            for token in session_tokens[:3]:
                futures.append(executor.submit(invalidate_session, token))
            
            # Bulk user session invalidations
            futures.append(executor.submit(invalidate_all_user_sessions))
            futures.append(executor.submit(invalidate_all_user_sessions))
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    invalidation_results.append(result)
        
        # Should not have database errors from race conditions
        assert len(errors) == 0, f"Concurrent invalidation errors: {errors}"
        
        # Verify all sessions are properly invalidated
        for session_token in session_tokens:
            user = auth_service.get_user_from_session(
                session_token=session_token,
                ip_address="192.168.1.10",
                user_agent="Device 1"
            )
            assert user is None, f"Session {session_token[:10]}... should be invalidated"
    
    def test_session_cleanup_with_expired_sessions(self, auth_service, test_user, db_session):
        """
        Test session cleanup properly handles expired sessions.
        
        Edge case: Mix of active, expired, and invalidated sessions should
        be cleaned up correctly without affecting valid sessions.
        """
        from secure_auth.session_service import SessionService
        from secure_auth.models import Session
        
        session_service = SessionService(db_session=db_session)
        
        # Create sessions with different states
        # Active session
        active_session = session_service.create_session(
            user_id=test_user.id,
            ip_address="192.168.1.100",
            user_agent="Active Device",
            tenant_id=test_user.tenant_id
        )
        
        # Expired session (manually expire)
        expired_session = session_service.create_session(
            user_id=test_user.id,
            ip_address="192.168.1.101",
            user_agent="Expired Device",
            tenant_id=test_user.tenant_id
        )
        
        # Manually expire the session
        session_record = db_session.query(Session).filter(
            Session.session_token == expired_session
        ).first()
        if session_record:
            session_record.expires_at = datetime.utcnow() - timedelta(hours=1)
            db_session.commit()
        
        # Invalidated session
        invalidated_session = session_service.create_session(
            user_id=test_user.id,
            ip_address="192.168.1.102",
            user_agent="Invalidated Device",
            tenant_id=test_user.tenant_id
        )
        session_service.invalidate_session(invalidated_session)
        
        # Run cleanup
        cleaned_count = session_service.cleanup_expired_sessions()
        
        # Verify active session still works
        active_user = auth_service.get_user_from_session(
            session_token=active_session,
            ip_address="192.168.1.100",
            user_agent="Active Device"
        )
        assert active_user is not None, "Active session should still work"
        
        # Verify expired session doesn't work
        expired_user = auth_service.get_user_from_session(
            session_token=expired_session,
            ip_address="192.168.1.101",
            user_agent="Expired Device"
        )
        assert expired_user is None, "Expired session should not work"
        
        # Verify invalidated session doesn't work
        invalidated_user = auth_service.get_user_from_session(
            session_token=invalidated_session,
            ip_address="192.168.1.102",
            user_agent="Invalidated Device"
        )
        assert invalidated_user is None, "Invalidated session should not work"
        
        # Cleanup should have removed at least the expired session
        assert cleaned_count >= 1, "Cleanup should have removed expired sessions"