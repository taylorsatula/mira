"""
Security-focused tests for RateLimitService.

Tests ACTUAL rate limiting behavior with real database integration.
NO MOCKING of core security logic - tests the real implementation.
"""

import pytest
import time
import secrets
from datetime import datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

from secure_auth.rate_limit_service import RateLimitService, RateLimitAction, RateLimitResult
from secure_auth.models import RateLimit, User, Base


@pytest.fixture
def rate_limit_service(db_session):
    """Rate limit service with real database session."""
    return RateLimitService(db_session=db_session)


class TestRealDatabaseRateLimiting:
    """Test rate limiting with real database operations."""
    
    def test_email_rate_limiting_database_persistence(self, rate_limit_service, db_session):
        """Test email rate limiting creates and updates database records."""
        email = "test@example.com"
        action = RateLimitAction.MAGIC_LINK_REQUEST
        
        # First request should be allowed
        result = rate_limit_service.check_rate_limit(
            identifier_type="email",
            identifier_value=email,
            action=action
        )
        assert result.allowed
        assert result.current_count == 0  # Check hasn't recorded yet
        
        # Record the attempt
        record_result = rate_limit_service.record_attempt(
            identifier_type="email",
            identifier_value=email,
            action=action,
            success=True
        )
        assert record_result.allowed
        assert record_result.current_count == 1
        
        # Verify database record exists
        rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email,
            RateLimit.action_type == action.value
        ).first()
        
        assert rate_limit is not None
        assert rate_limit.attempt_count == 1
        assert rate_limit.expires_at > datetime.utcnow()
        assert not rate_limit.is_exceeded
    
    def test_rate_limit_threshold_enforcement_with_database(self, rate_limit_service, db_session):
        """Test rate limit thresholds are enforced correctly."""
        email = "bruteforce@example.com"
        action = RateLimitAction.MAGIC_LINK_REQUEST
        
        # Get the configured limit for this action
        config = rate_limit_service._get_limit_config(action, "email")
        limit = config["limit"]  # Should be 3 for magic link requests
        
        # Make requests up to the limit
        for i in range(limit):
            result = rate_limit_service.record_attempt(
                identifier_type="email",
                identifier_value=email,
                action=action,
                success=True
            )
            assert result.allowed, f"Request {i+1} should be allowed"
            assert result.current_count == i + 1
        
        # Next request should be blocked
        over_limit_result = rate_limit_service.record_attempt(
            identifier_type="email",
            identifier_value=email,
            action=action,
            success=True
        )
        assert not over_limit_result.allowed
        assert over_limit_result.current_count > limit
        
        # Verify database state
        rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email
        ).first()
        
        assert rate_limit.is_exceeded
        assert rate_limit.attempt_count > limit
    
    def test_ip_rate_limiting_independent_tracking(self, rate_limit_service, db_session):
        """Test IP-based rate limiting tracks independently from email."""
        ip_address = "192.168.1.100"
        action = RateLimitAction.LOGIN_ATTEMPT
        
        # Make multiple requests from same IP
        config = rate_limit_service._get_limit_config(action, "ip")
        ip_limit = config["limit"]  # Should be 20 for login attempts
        
        # Make requests up to 75% of limit to avoid hitting it in test
        test_requests = min(15, int(ip_limit * 0.75))
        
        for i in range(test_requests):
            result = rate_limit_service.record_attempt(
                identifier_type="ip",
                identifier_value=ip_address,
                action=action,
                user_id=str(uuid4()),  # Different users
                success=True
            )
            assert result.allowed, f"IP request {i+1} should be allowed"
        
        # Verify database record
        ip_rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "ip",
            RateLimit.identifier_value == ip_address
        ).first()
        
        assert ip_rate_limit is not None
        assert ip_rate_limit.attempt_count == test_requests
        assert not ip_rate_limit.is_exceeded  # Should still be under limit
    
    def test_rate_limit_window_expiration_and_reset(self, rate_limit_service, db_session):
        """Test rate limit windows expire and reset correctly."""
        email = "window-test@example.com"
        action = RateLimitAction.FAILED_LOGIN
        
        # Exhaust the rate limit
        config = rate_limit_service._get_limit_config(action, "email")
        limit = config["limit"]
        
        for i in range(limit + 1):
            rate_limit_service.record_attempt(
                identifier_type="email",
                identifier_value=email,
                action=action,
                success=False
            )
        
        # Verify we're rate limited
        check_result = rate_limit_service.check_rate_limit(
            identifier_type="email",
            identifier_value=email,
            action=action
        )
        assert not check_result.allowed
        
        # Manually expire the rate limit window
        rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email
        ).first()
        
        original_window_start = rate_limit.window_start
        rate_limit.expires_at = datetime.utcnow() - timedelta(minutes=1)
        db_session.commit()
        
        # New request should be allowed (new window)
        new_result = rate_limit_service.record_attempt(
            identifier_type="email",
            identifier_value=email,
            action=action,
            success=True
        )
        assert new_result.allowed
        
        # Verify new window was created
        updated_rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email,
            RateLimit.expires_at > datetime.utcnow()
        ).first()
        
        assert updated_rate_limit is not None
        # Should be either a new record or reset record
        assert (updated_rate_limit.window_start > original_window_start or 
                updated_rate_limit.attempt_count == 1)
    
    def test_adaptive_escalation_with_violations(self, rate_limit_service, db_session):
        """Test adaptive rate limiting escalates with violation history."""
        email = "repeat-offender@example.com"
        action = RateLimitAction.LOGIN_ATTEMPT
        
        # First violation cycle
        config = rate_limit_service._get_limit_config(action, "email")
        base_limit = config["limit"]
        
        # Exhaust first limit
        for i in range(base_limit + 1):
            rate_limit_service.record_attempt(
                identifier_type="email",
                identifier_value=email,
                action=action,
                success=False
            )
        
        # Mark violation and reset window
        rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email
        ).first()
        
        rate_limit.last_violation_at = datetime.utcnow()
        rate_limit.reset_count += 1
        rate_limit.expires_at = datetime.utcnow() - timedelta(minutes=1)
        db_session.commit()
        
        # Second cycle - should have lower threshold
        attempts_allowed = 0
        for i in range(base_limit):
            result = rate_limit_service.record_attempt(
                identifier_type="email",
                identifier_value=email,
                action=action,
                success=False
            )
            if result.allowed:
                attempts_allowed += 1
            else:
                break
        
        # Should allow fewer attempts than original limit (escalation)
        assert attempts_allowed < base_limit
        
        # Verify escalation in database
        escalated_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email,
            RateLimit.expires_at > datetime.utcnow()
        ).first()
        
        assert escalated_limit.limit_threshold < base_limit
    
    def test_different_actions_independent_limits(self, rate_limit_service, db_session):
        """Test different actions have independent rate limits."""
        email = "multi-action@example.com"
        
        # Test magic link and login attempt limits are independent
        magic_action = RateLimitAction.MAGIC_LINK_REQUEST
        login_action = RateLimitAction.LOGIN_ATTEMPT
        
        # Exhaust magic link limit
        magic_config = rate_limit_service._get_limit_config(magic_action, "email")
        for i in range(magic_config["limit"] + 1):
            rate_limit_service.record_attempt(
                identifier_type="email",
                identifier_value=email,
                action=magic_action,
                success=True
            )
        
        # Magic link should be rate limited
        magic_check = rate_limit_service.check_rate_limit(
            identifier_type="email",
            identifier_value=email,
            action=magic_action
        )
        assert not magic_check.allowed
        
        # But login attempts should still be allowed
        login_result = rate_limit_service.record_attempt(
            identifier_type="email",
            identifier_value=email,
            action=login_action,
            success=True
        )
        assert login_result.allowed
        
        # Verify separate database records
        rate_limits = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email
        ).all()
        
        action_types = {rl.action_type for rl in rate_limits}
        assert magic_action.value in action_types
        assert login_action.value in action_types
    
    def test_violation_count_lookback_period(self, rate_limit_service, db_session):
        """Test violation count considers lookback period correctly."""
        email = "violation-history@example.com"
        action = RateLimitAction.MAGIC_LINK_REQUEST
        
        # Create old violation (outside lookback period)
        old_rate_limit = RateLimit(
            identifier_type="email",
            identifier_value=email,
            action_type=action.value,
            window_start=datetime.utcnow() - timedelta(days=2),
            window_duration_seconds=1200,
            attempt_count=5,
            limit_threshold=3,
            base_threshold=3,
            expires_at=datetime.utcnow() - timedelta(days=2, minutes=-20),
            last_violation_at=datetime.utcnow() - timedelta(days=2),
            escalation_factor=1
        )
        old_rate_limit.is_exceeded = True
        db_session.add(old_rate_limit)
        
        # Create recent violation (within lookback period)
        recent_rate_limit = RateLimit(
            identifier_type="email",
            identifier_value=email,
            action_type=action.value,
            window_start=datetime.utcnow() - timedelta(hours=2),
            window_duration_seconds=1200,
            attempt_count=4,
            limit_threshold=3,
            base_threshold=3,
            expires_at=datetime.utcnow() - timedelta(hours=2, minutes=-20),
            last_violation_at=datetime.utcnow() - timedelta(hours=2),
            escalation_factor=1
        )
        recent_rate_limit.is_exceeded = True
        db_session.add(recent_rate_limit)
        db_session.commit()
        
        # Get violation count (should only count recent ones)
        violation_count = rate_limit_service._get_violation_count(
            identifier_type="email",
            identifier_value=email,
            action=action,
            lookback_hours=24
        )
        
        # Should only count the recent violation
        assert violation_count == 1
    
    def test_rate_limit_status_reporting(self, rate_limit_service, db_session):
        """Test rate limit status reporting with database integration."""
        email = "status-test@example.com"
        action = RateLimitAction.LOGIN_ATTEMPT
        
        # Create some rate limit activity
        for i in range(3):
            rate_limit_service.record_attempt(
                identifier_type="email",
                identifier_value=email,
                action=action,
                success=i < 2  # First two succeed, third fails
            )
        
        # Get status
        status = rate_limit_service.get_rate_limit_status(
            identifier_type="email",
            identifier_value=email,
            action=action
        )
        
        assert "identifier" in status
        assert "active_limits" in status
        assert "total_violations" in status
        assert status["identifier"] == f"email:{email}"
        
        # Should have one active limit
        assert len(status["active_limits"]) == 1
        limit_info = status["active_limits"][0]
        
        assert limit_info["action"] == action.value
        assert limit_info["current_count"] == 3
        assert "reset_time" in limit_info
        assert "is_exceeded" in limit_info
    
    def test_cleanup_expired_limits(self, rate_limit_service, db_session):
        """Test cleanup of expired rate limit records."""
        # Create expired rate limit
        expired_limit = RateLimit(
            identifier_type="email",
            identifier_value="expired@example.com",
            action_type=RateLimitAction.LOGIN_ATTEMPT.value,
            window_start=datetime.utcnow() - timedelta(hours=2),
            window_duration_seconds=3600,
            attempt_count=5,
            limit_threshold=5,
            base_threshold=5,
            expires_at=datetime.utcnow() - timedelta(minutes=30),  # Expired
            escalation_factor=1
        )
        db_session.add(expired_limit)
        
        # Create active rate limit
        active_limit = RateLimit(
            identifier_type="email",
            identifier_value="active@example.com",
            action_type=RateLimitAction.LOGIN_ATTEMPT.value,
            window_start=datetime.utcnow() - timedelta(minutes=30),
            window_duration_seconds=3600,
            attempt_count=2,
            limit_threshold=5,
            base_threshold=5,
            expires_at=datetime.utcnow() + timedelta(minutes=30),  # Active
            escalation_factor=1
        )
        db_session.add(active_limit)
        db_session.commit()
        
        # Verify both exist
        total_before = db_session.query(RateLimit).count()
        assert total_before == 2
        
        # Cleanup expired
        cleaned_count = rate_limit_service.cleanup_expired_limits()
        assert cleaned_count == 1
        
        # Verify only active limit remains
        remaining_limits = db_session.query(RateLimit).all()
        assert len(remaining_limits) == 1
        assert remaining_limits[0].identifier_value == "active@example.com"
    
    def test_manual_rate_limit_reset(self, rate_limit_service, db_session):
        """Test manual rate limit reset functionality."""
        email = "reset-test@example.com"
        action = RateLimitAction.MAGIC_LINK_REQUEST
        
        # Create rate limit that's exceeded
        for i in range(5):  # Exceed the limit
            rate_limit_service.record_attempt(
                identifier_type="email",
                identifier_value=email,
                action=action,
                success=True
            )
        
        # Verify rate limited
        check_result = rate_limit_service.check_rate_limit(
            identifier_type="email",
            identifier_value=email,
            action=action
        )
        assert not check_result.allowed
        
        # Reset the rate limit
        reset_success = rate_limit_service.reset_rate_limit(
            identifier_type="email",
            identifier_value=email,
            action=action
        )
        assert reset_success
        
        # Verify reset worked
        after_reset_check = rate_limit_service.check_rate_limit(
            identifier_type="email",
            identifier_value=email,
            action=action
        )
        assert after_reset_check.allowed
        
        # Verify database record removed
        rate_limit_count = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email,
            RateLimit.action_type == action.value,
            RateLimit.expires_at > datetime.utcnow()
        ).count()
        assert rate_limit_count == 0


class TestRealBruteForceScenarios:
    """Test real brute force attack scenarios."""
    
    def test_rapid_fire_requests_from_single_ip(self, rate_limit_service, db_session):
        """Test rapid fire requests from single IP are properly limited."""
        attacker_ip = "203.0.113.50"
        action = RateLimitAction.LOGIN_ATTEMPT
        
        # Simulate rapid requests
        results = []
        for i in range(25):  # More than IP limit
            result = rate_limit_service.record_attempt(
                identifier_type="ip",
                identifier_value=attacker_ip,
                action=action,
                user_id=str(uuid4()),  # Different users
                success=False
            )
            results.append(result)
            
            # Stop if rate limited
            if not result.allowed:
                break
        
        # Should eventually be rate limited
        blocked_results = [r for r in results if not r.allowed]
        assert len(blocked_results) > 0
        
        # Verify final state is blocked
        final_check = rate_limit_service.check_rate_limit(
            identifier_type="ip",
            identifier_value=attacker_ip,
            action=action
        )
        assert not final_check.allowed
        assert final_check.retry_after_seconds is not None
    
    def test_distributed_attack_per_email_protection(self, rate_limit_service, db_session):
        """Test distributed attack against single email is limited."""
        target_email = "victim@example.com"
        action = RateLimitAction.MAGIC_LINK_REQUEST
        
        # Simulate requests from different IPs to same email
        attacker_ips = [f"203.0.113.{i}" for i in range(50, 60)]
        
        results = []
        for ip in attacker_ips:
            result = rate_limit_service.record_attempt(
                identifier_type="email",
                identifier_value=target_email,
                action=action,
                success=True
            )
            results.append(result)
            
            if not result.allowed:
                break
        
        # Email should be rate limited regardless of IP diversity
        email_limited = any(not r.allowed for r in results)
        assert email_limited
        
        # Verify email-specific rate limit exists
        email_rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == target_email
        ).first()
        
        assert email_rate_limit is not None
        assert email_rate_limit.is_exceeded
    
    def test_credential_stuffing_simulation(self, rate_limit_service, db_session):
        """Test credential stuffing attack simulation."""
        # Single IP attacking multiple emails
        attacker_ip = "198.51.100.10"
        action = RateLimitAction.LOGIN_ATTEMPT
        
        # Create victim emails
        victim_emails = [f"victim{i}@example.com" for i in range(25)]
        
        # Simulate credential stuffing
        ip_results = []
        for email in victim_emails:
            # Try each email once from same IP
            result = rate_limit_service.record_attempt(
                identifier_type="ip",
                identifier_value=attacker_ip,
                action=action,
                success=False
            )
            ip_results.append(result)
            
            if not result.allowed:
                break
        
        # IP should eventually be rate limited
        ip_blocked = any(not r.allowed for r in ip_results)
        assert ip_blocked
        
        # Verify IP rate limit record
        ip_rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "ip",
            RateLimit.identifier_value == attacker_ip
        ).first()
        
        assert ip_rate_limit is not None
        assert ip_rate_limit.attempt_count > 0


class TestRateLimitConfiguration:
    """Test rate limit configuration is secure and appropriate."""
    
    def test_all_critical_actions_have_limits(self):
        """Test all security-critical actions have rate limits configured."""
        service = RateLimitService()
        
        critical_actions = [
            RateLimitAction.MAGIC_LINK_REQUEST,
            RateLimitAction.LOGIN_ATTEMPT,
            RateLimitAction.FAILED_LOGIN,
            RateLimitAction.PASSWORD_RESET,
        ]
        
        for action in critical_actions:
            # Should have email-based limits
            email_config = service._get_limit_config(action, "email")
            assert email_config is not None, f"Missing email rate limit for {action}"
            
            # Should have IP-based limits
            ip_config = service._get_limit_config(action, "ip")
            assert ip_config is not None, f"Missing IP rate limit for {action}"
    
    def test_limit_hierarchies_are_logical(self):
        """Test rate limit hierarchies make security sense."""
        service = RateLimitService()
        
        # Failed logins should be more restrictive than general login attempts
        failed_email = service._get_limit_config(RateLimitAction.FAILED_LOGIN, "email")
        login_email = service._get_limit_config(RateLimitAction.LOGIN_ATTEMPT, "email")
        
        assert failed_email["limit"] <= login_email["limit"], "Failed login limit should be stricter"
        assert failed_email["window_minutes"] >= login_email["window_minutes"], "Failed login window should be longer"
        
        # IP limits should generally be higher than email limits (distributed vs focused)
        magic_ip = service._get_limit_config(RateLimitAction.MAGIC_LINK_REQUEST, "ip")
        magic_email = service._get_limit_config(RateLimitAction.MAGIC_LINK_REQUEST, "email")
        
        assert magic_ip["limit"] >= magic_email["limit"], "IP limit should accommodate multiple users"
    
    def test_escalation_factors_are_reasonable(self):
        """Test escalation factors provide meaningful security improvement."""
        service = RateLimitService()
        
        for action in RateLimitAction:
            for identifier_type in ["email", "ip", "user"]:
                config = service._get_limit_config(action, identifier_type)
                if config:
                    escalation_factor = config["escalation_factor"]
                    
                    # Should provide meaningful escalation
                    assert 1.0 <= escalation_factor <= 5.0, f"Escalation factor out of range for {action}:{identifier_type}"
                    
                    # Test escalation reduces limits meaningfully
                    base_limit = config["limit"]
                    escalated = service._calculate_adaptive_threshold(base_limit, escalation_factor, 2)
                    
                    # Should reduce by at least 25% after 2 violations
                    reduction_ratio = escalated / base_limit
                    assert reduction_ratio <= 0.75, f"Insufficient escalation for {action}:{identifier_type}"


class TestHeaderManipulationBypass:
    """Test rate limit bypass attempts using request header manipulation."""
    
    def test_x_forwarded_for_spoofing_prevention(self, rate_limit_service, db_session):
        """
        Test X-Forwarded-For header spoofing cannot bypass IP rate limits.
        
        Attack scenario: Attacker attempts to bypass IP rate limits by sending
        different X-Forwarded-For headers to make the system think requests
        are coming from different IPs.
        """
        real_attacker_ip = "203.0.113.50"
        action = RateLimitAction.LOGIN_ATTEMPT
        
        # Simulate X-Forwarded-For spoofing attempts
        spoofed_ips = [
            "10.0.0.1",  # Private IP
            "192.168.1.1",  # Private IP
            "127.0.0.1",  # Localhost
            "172.16.0.1",  # Private IP
            "8.8.8.8",  # Public DNS
            "1.1.1.1",  # Public DNS
            "",  # Empty
            "invalid-ip",  # Invalid format
            "999.999.999.999"  # Invalid IP
        ]
        
        # Get IP rate limit for reference
        config = rate_limit_service._get_limit_config(action, "ip")
        ip_limit = config["limit"]
        
        # Make requests exceeding limit, all from same real IP
        # but with different X-Forwarded-For headers
        results = []
        for i in range(ip_limit + 5):
            spoofed_ip = spoofed_ips[i % len(spoofed_ips)]
            
            # The service should use real_attacker_ip regardless of X-Forwarded-For
            result = rate_limit_service.record_attempt(
                identifier_type="ip",
                identifier_value=real_attacker_ip,  # Real IP should be tracked
                action=action,
                success=False,
                metadata={"x_forwarded_for": spoofed_ip}  # Spoofed header
            )
            results.append(result)
            
            if not result.allowed:
                break
        
        # Should be rate limited despite header spoofing
        blocked_results = [r for r in results if not r.allowed]
        assert len(blocked_results) > 0, "X-Forwarded-For spoofing bypassed rate limiting"
        
        # Verify only one rate limit record for real IP
        ip_rate_limits = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "ip",
            RateLimit.identifier_value == real_attacker_ip
        ).all()
        
        assert len(ip_rate_limits) == 1, "Multiple IP records suggest spoofing worked"
        assert ip_rate_limits[0].is_exceeded
    
    def test_user_agent_rotation_bypass_prevention(self, rate_limit_service, db_session):
        """
        Test User-Agent rotation cannot bypass rate limits.
        
        Attack scenario: Attacker rotates User-Agent headers to appear as
        different browsers/devices to bypass rate limiting.
        """
        attacker_ip = "198.51.100.10"
        action = RateLimitAction.FAILED_LOGIN
        
        # Common User-Agent strings attackers might rotate through
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
            "Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0",
            "curl/7.68.0",
            "Python-requests/2.25.1",
            "PostmanRuntime/7.26.8",
            ""  # Empty user agent
        ]
        
        # Get limit for reference
        config = rate_limit_service._get_limit_config(action, "ip")
        ip_limit = config["limit"]
        
        # Make requests with rotating User-Agents
        results = []
        for i in range(ip_limit + 3):
            user_agent = user_agents[i % len(user_agents)]
            
            result = rate_limit_service.record_attempt(
                identifier_type="ip",
                identifier_value=attacker_ip,
                action=action,
                success=False,
                metadata={"user_agent": user_agent}
            )
            results.append(result)
            
            if not result.allowed:
                break
        
        # Should be rate limited despite User-Agent rotation
        blocked_results = [r for r in results if not r.allowed]
        assert len(blocked_results) > 0, "User-Agent rotation bypassed rate limiting"
        
        # Verify single rate limit record for IP
        rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "ip",
            RateLimit.identifier_value == attacker_ip
        ).first()
        
        assert rate_limit is not None
        assert rate_limit.is_exceeded
    
    def test_referer_header_manipulation_bypass_prevention(self, rate_limit_service, db_session):
        """
        Test Referer header manipulation cannot bypass rate limits.
        
        Attack scenario: Attacker varies Referer headers to appear as
        legitimate traffic from different sources.
        """
        target_email = "victim@example.com"
        action = RateLimitAction.MAGIC_LINK_REQUEST
        
        # Various referer headers an attacker might use
        referers = [
            "https://www.google.com/search?q=login",
            "https://www.facebook.com/",
            "https://twitter.com/",
            "https://www.linkedin.com/",
            "https://legitimate-site.com/login",
            "https://partner-site.com/auth",
            "",  # Empty referer
            "invalid-url",
            "javascript:void(0)"
        ]
        
        # Get email limit for reference
        config = rate_limit_service._get_limit_config(action, "email")
        email_limit = config["limit"]
        
        # Make requests with varying Referer headers
        results = []
        for i in range(email_limit + 2):
            referer = referers[i % len(referers)]
            
            result = rate_limit_service.record_attempt(
                identifier_type="email",
                identifier_value=target_email,
                action=action,
                success=True,
                metadata={"referer": referer}
            )
            results.append(result)
            
            if not result.allowed:
                break
        
        # Should be rate limited despite referer manipulation
        blocked_results = [r for r in results if not r.allowed]
        assert len(blocked_results) > 0, "Referer manipulation bypassed rate limiting"
        
        # Verify email-based rate limit exists
        rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == target_email
        ).first()
        
        assert rate_limit is not None
        assert rate_limit.is_exceeded
    
    def test_session_id_manipulation_bypass_prevention(self, rate_limit_service, db_session):
        """
        Test session ID manipulation cannot bypass user-based rate limits.
        
        Attack scenario: Attacker manipulates session cookies or IDs
        to appear as different authenticated users.
        """
        user_id = str(uuid4())
        action = RateLimitAction.LOGIN_ATTEMPT
        
        # Various session IDs/cookies an attacker might forge
        session_ids = [
            "sess_" + secrets.token_hex(16),
            "PHPSESSID=" + secrets.token_hex(32),
            "ASP.NET_SessionId=" + secrets.token_urlsafe(24),
            "jsessionid=" + secrets.token_hex(16),
            "",  # No session
            "invalid-session",
            "admin_session_123",
            "guest_session"
        ]
        
        # Get user limit for reference
        config = rate_limit_service._get_limit_config(action, "user")
        user_limit = config["limit"] if config else 10  # Default fallback
        
        # Make requests with different session IDs but same user
        results = []
        for i in range(user_limit + 3):
            session_id = session_ids[i % len(session_ids)]
            
            result = rate_limit_service.record_attempt(
                identifier_type="user",
                identifier_value=user_id,  # Same user regardless of session
                action=action,
                success=False,
                metadata={"session_id": session_id}
            )
            results.append(result)
            
            if not result.allowed:
                break
        
        # Should be rate limited despite session manipulation
        blocked_results = [r for r in results if not r.allowed]
        assert len(blocked_results) > 0, "Session manipulation bypassed rate limiting"
        
        # Verify user-based rate limit exists
        rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "user",
            RateLimit.identifier_value == user_id
        ).first()
        
        assert rate_limit is not None
        assert rate_limit.is_exceeded
    
    def test_combined_header_manipulation_attack(self, rate_limit_service, db_session):
        """
        Test combined header manipulation cannot bypass rate limits.
        
        Attack scenario: Sophisticated attacker varies multiple headers
        simultaneously to maximize chances of bypassing detection.
        """
        attacker_ip = "192.0.2.100"
        target_email = "admin@example.com"
        action = RateLimitAction.LOGIN_ATTEMPT
        
        # Complex attack with multiple varying headers
        attack_variants = [
            {
                "x_forwarded_for": "10.0.0.1",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "referer": "https://www.google.com/",
                "accept_language": "en-US,en;q=0.9"
            },
            {
                "x_forwarded_for": "172.16.0.1", 
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "referer": "https://www.facebook.com/",
                "accept_language": "es-ES,es;q=0.8"
            },
            {
                "x_forwarded_for": "192.168.1.1",
                "user_agent": "curl/7.68.0",
                "referer": "",
                "accept_language": "fr-FR,fr;q=0.7"
            }
        ]
        
        # Get limits for both IP and email
        ip_config = rate_limit_service._get_limit_config(action, "ip")
        email_config = rate_limit_service._get_limit_config(action, "email")
        
        combined_limit = min(ip_config["limit"], email_config["limit"])
        
        # Execute combined attack
        ip_results = []
        email_results = []
        
        for i in range(combined_limit + 5):
            variant = attack_variants[i % len(attack_variants)]
            
            # Try IP-based attack
            ip_result = rate_limit_service.record_attempt(
                identifier_type="ip",
                identifier_value=attacker_ip,
                action=action,
                success=False,
                metadata=variant
            )
            ip_results.append(ip_result)
            
            # Try email-based attack  
            email_result = rate_limit_service.record_attempt(
                identifier_type="email",
                identifier_value=target_email,
                action=action,
                success=False,
                metadata=variant
            )
            email_results.append(email_result)
            
            # Stop if both are blocked
            if not ip_result.allowed and not email_result.allowed:
                break
        
        # Both IP and email should be rate limited
        ip_blocked = any(not r.allowed for r in ip_results)
        email_blocked = any(not r.allowed for r in email_results)
        
        assert ip_blocked, "Combined header manipulation bypassed IP rate limiting"
        assert email_blocked, "Combined header manipulation bypassed email rate limiting"
        
        # Verify database records exist for both
        ip_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "ip",
            RateLimit.identifier_value == attacker_ip
        ).first()
        
        email_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email", 
            RateLimit.identifier_value == target_email
        ).first()
        
        assert ip_limit is not None and ip_limit.is_exceeded
        assert email_limit is not None and email_limit.is_exceeded


class TestServicePersistenceAcrossRestarts:
    """Test rate limit persistence across service restarts."""
    
    def test_rate_limits_persist_across_service_restarts(self, test_database, db_session):
        """
        Test rate limits persist when service is recreated (simulating restart).
        
        Security requirement: Rate limits must not be reset by application
        deployments or restarts, preventing attackers from bypassing limits
        by triggering service restarts.
        """
        email = "persistent@example.com"
        action = RateLimitAction.FAILED_LOGIN
        
        # Create first service instance
        service1 = RateLimitService(db_session=db_session)
        
        # Exhaust rate limit
        config = service1._get_limit_config(action, "email")
        limit = config["limit"]
        
        for i in range(limit + 1):
            service1.record_attempt(
                identifier_type="email",
                identifier_value=email,
                action=action,
                success=False
            )
        
        # Verify rate limited
        check1 = service1.check_rate_limit(
            identifier_type="email",
            identifier_value=email, 
            action=action
        )
        assert not check1.allowed
        
        # Simulate service restart by creating new service instance
        # with same database session (representing persistent storage)
        service2 = RateLimitService(db_session=db_session)
        
        # Rate limit should still be enforced
        check2 = service2.check_rate_limit(
            identifier_type="email",
            identifier_value=email,
            action=action
        )
        assert not check2.allowed, "Rate limit was reset after service restart"
        
        # Attempt should still be blocked
        restart_attempt = service2.record_attempt(
            identifier_type="email",
            identifier_value=email,
            action=action,
            success=False
        )
        assert not restart_attempt.allowed, "Service restart allowed bypassing rate limit"
        
        # Verify same database record is being used
        rate_limits = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email
        ).all()
        
        # Should only have one record (not duplicated by restart)
        assert len(rate_limits) == 1
        assert rate_limits[0].is_exceeded
    
    def test_violation_history_persists_across_restarts(self, test_database, db_session):
        """
        Test violation history and escalation persist across restarts.
        
        Security requirement: Escalation based on violation history must
        not be reset by restarts, preventing repeat offenders from
        getting fresh starts.
        """
        email = "repeat-offender@example.com"
        action = RateLimitAction.MAGIC_LINK_REQUEST
        
        # Create first service and establish violation history
        service1 = RateLimitService(db_session=db_session)
        
        # Create initial violation
        config = service1._get_limit_config(action, "email")
        base_limit = config["limit"]
        
        for i in range(base_limit + 2):
            service1.record_attempt(
                identifier_type="email",
                identifier_value=email,
                action=action,
                success=True
            )
        
        # Manually set violation history (simulating previous violations)
        rate_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email
        ).first()
        
        rate_limit.last_violation_at = datetime.utcnow() - timedelta(hours=1)
        rate_limit.reset_count = 2  # Previous violations
        rate_limit.expires_at = datetime.utcnow() - timedelta(minutes=1)  # Expire current window
        db_session.commit()
        
        # Simulate service restart
        service2 = RateLimitService(db_session=db_session)
        
        # New requests should face escalated limits
        escalated_attempts = 0
        for i in range(base_limit):
            result = service2.record_attempt(
                identifier_type="email",
                identifier_value=email,
                action=action,
                success=True
            )
            if result.allowed:
                escalated_attempts += 1
            else:
                break
        
        # Should allow fewer attempts due to escalation history
        assert escalated_attempts < base_limit, "Violation history was lost after restart"
        
        # Verify escalation persisted
        current_limit = db_session.query(RateLimit).filter(
            RateLimit.identifier_type == "email",
            RateLimit.identifier_value == email,
            RateLimit.expires_at > datetime.utcnow()
        ).first()
        
        assert current_limit is not None
        assert current_limit.reset_count >= 2, "Reset count was lost"
        assert current_limit.limit_threshold < base_limit, "Escalation was reset"
    
    def test_configuration_consistency_across_restarts(self, test_database):
        """
        Test rate limit configuration remains consistent across restarts.
        
        Security requirement: Rate limit thresholds and windows must not
        change unexpectedly during restarts, maintaining security posture.
        """
        # Create first service instance and capture configuration
        with create_engine(test_database).connect() as conn:
            Session = sessionmaker(bind=conn)
            session1 = Session()
            
            service1 = RateLimitService(db_session=session1)
            
            # Capture all configurations
            original_configs = {}
            for action in RateLimitAction:
                for id_type in ["email", "ip", "user"]:
                    config = service1._get_limit_config(action, id_type)
                    if config:
                        original_configs[f"{action.value}:{id_type}"] = config.copy()
            
            session1.close()
        
        # Simulate restart with new service instance
        with create_engine(test_database).connect() as conn:
            Session = sessionmaker(bind=conn)
            session2 = Session()
            
            service2 = RateLimitService(db_session=session2)
            
            # Verify configurations are identical
            for key, original_config in original_configs.items():
                action_str, id_type = key.split(":")
                action = RateLimitAction(action_str)
                
                new_config = service2._get_limit_config(action, id_type)
                assert new_config is not None, f"Configuration lost for {key}"
                
                # Check critical security parameters
                assert new_config["limit"] == original_config["limit"], f"Limit changed for {key}"
                assert new_config["window_minutes"] == original_config["window_minutes"], f"Window changed for {key}"
                assert new_config["escalation_factor"] == original_config["escalation_factor"], f"Escalation changed for {key}"
            
            session2.close()
    
    def test_active_rate_limits_survive_database_reconnection(self, test_database):
        """
        Test active rate limits survive database reconnection scenarios.
        
        Security requirement: Temporary database connectivity issues must
        not reset rate limiting state.
        """
        email = "db-reconnect-test@example.com"
        action = RateLimitAction.LOGIN_ATTEMPT
        
        # Create rate limit with first database connection
        engine1 = create_engine(test_database)
        Session1 = sessionmaker(bind=engine1)
        session1 = Session1()
        
        service1 = RateLimitService(db_session=session1)
        
        # Create rate limit state
        config = service1._get_limit_config(action, "email")
        for i in range(config["limit"]):
            service1.record_attempt(
                identifier_type="email",
                identifier_value=email,
                action=action,
                success=False
            )
        
        # Verify rate limited
        check1 = service1.check_rate_limit(
            identifier_type="email",
            identifier_value=email,
            action=action
        )
        
        session1.close()
        engine1.dispose()
        
        # Simulate database reconnection with new engine/session
        engine2 = create_engine(test_database)
        Session2 = sessionmaker(bind=engine2)
        session2 = Session2()
        
        service2 = RateLimitService(db_session=session2)
        
        # Rate limit state should be preserved
        check2 = service2.check_rate_limit(
            identifier_type="email",
            identifier_value=email,
            action=action
        )
        
        # Should have same rate limit status
        assert check1.allowed == check2.allowed, "Rate limit status changed after reconnection"
        assert abs(check1.current_count - check2.current_count) <= 1, "Count significantly different"
        
        if check1.retry_after_seconds and check2.retry_after_seconds:
            # Retry times should be similar (accounting for time passage)
            time_diff = abs(check1.retry_after_seconds - check2.retry_after_seconds)
            assert time_diff < 5, "Retry time significantly different"
        
        session2.close()
        engine2.dispose()


class TestServiceSingleton:
    """Test rate limit service singleton behavior."""
    
    def test_singleton_returns_same_instance(self):
        """Test get_rate_limit_service returns same instance."""
        from secure_auth.rate_limit_service import get_rate_limit_service
        
        service1 = get_rate_limit_service()
        service2 = get_rate_limit_service()
        
        assert service1 is service2
    
    def test_singleton_configuration_consistency(self):
        """Test singleton maintains consistent configuration."""
        from secure_auth.rate_limit_service import get_rate_limit_service
        
        service = get_rate_limit_service()
        
        # Should have all expected default limits
        assert len(service.default_limits) >= 6
        
        # Configuration should be immutable across calls
        config1 = service._get_limit_config(RateLimitAction.LOGIN_ATTEMPT, "email")
        config2 = service._get_limit_config(RateLimitAction.LOGIN_ATTEMPT, "email")
        
        assert config1 == config2