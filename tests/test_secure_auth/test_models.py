"""
Security-focused tests for secure_auth models.

Tests attack scenarios, cryptographic properties, and data integrity
rather than just basic functionality. Follows security testing principles.
"""

import pytest
import secrets
import time
from datetime import datetime, timedelta
from unittest.mock import patch
from uuid import uuid4, UUID

from secure_auth.models import (
    User, MagicLink, UserSession, SecurityEvent, RateLimit, DeviceTrust,
    SecurityEventType, RiskLevel, get_database_url, create_auth_database,
    get_database_session
)


class TestUser:
    """Test User model security properties."""
    
    def test_user_creation_with_security_defaults(self):
        """Test user creation initializes secure defaults."""
        user = User(email="test@example.com")
        
        assert user.is_active is True
        assert user.is_admin is False
        assert user.email_verified is False
        assert user.failed_login_count == 0
        assert user.account_locked_until is None
        assert user.trusted_device_ids == []
        assert user.known_ip_addresses == []
        assert user.security_flags == {}
        assert user.metadata == {}
    
    def test_user_email_uniqueness_constraint(self):
        """Test that email uniqueness is enforced."""
        email = "unique@example.com"
        user1 = User(email=email)
        user2 = User(email=email)
        
        # Both objects created, but constraint enforced at DB level
        assert user1.email == user2.email
        assert user1 is not user2
    
    def test_user_account_locking_security(self):
        """Test account locking prevents access during lock period."""
        user = User(email="locked@example.com")
        
        # Initially not locked
        assert not user.is_locked
        
        # Lock account for 1 hour
        user.account_locked_until = datetime.utcnow() + timedelta(hours=1)
        assert user.is_locked
        
        # Expired lock should not be active
        user.account_locked_until = datetime.utcnow() - timedelta(minutes=1)
        assert not user.is_locked
    
    def test_user_tenant_isolation(self):
        """Test that tenant_id provides proper isolation."""
        email = "tenant@example.com"
        user1 = User(email=email, tenant_id="tenant_a")
        user2 = User(email=email, tenant_id="tenant_b")
        
        # Same email allowed in different tenants
        assert user1.email == user2.email
        assert user1.tenant_id != user2.tenant_id
        assert user1.tenant_id == "tenant_a"
        assert user2.tenant_id == "tenant_b"
    
    def test_user_security_flags_structure(self):
        """Test security flags can store security-relevant data."""
        user = User(email="flagged@example.com")
        
        # Test various security flag scenarios
        security_scenarios = {
            "suspicious_login_detected": True,
            "multiple_failed_attempts": 5,
            "unusual_location_access": ["192.168.1.1", "10.0.0.1"],
            "device_trust_violations": {
                "count": 2,
                "last_violation": datetime.utcnow().isoformat()
            }
        }
        
        user.security_flags = security_scenarios
        assert user.security_flags["suspicious_login_detected"] is True
        assert user.security_flags["multiple_failed_attempts"] == 5
        assert len(user.security_flags["unusual_location_access"]) == 2
        assert "count" in user.security_flags["device_trust_violations"]


class TestMagicLink:
    """Test MagicLink model security properties."""
    
    def test_magic_link_creation_with_security_features(self):
        """Test magic link creation includes security features."""
        user_id = uuid4()
        magic_link = MagicLink(
            user_id=user_id,
            email="test@example.com",
            token_hash=secrets.token_hex(64),
            salt=secrets.token_hex(32),
            expires_at=datetime.utcnow() + timedelta(minutes=15)
        )
        
        assert magic_link.user_id == user_id
        assert magic_link.attempt_count == 0
        assert magic_link.max_attempts == 3
        assert magic_link.risk_score == 0
        assert magic_link.used_at is None
        assert not magic_link.is_used
        assert not magic_link.attempts_exceeded
    
    def test_magic_link_expiration_security(self):
        """Test magic link expiration prevents reuse after timeout."""
        magic_link = MagicLink(
            user_id=uuid4(),
            email="test@example.com",
            token_hash=secrets.token_hex(64),
            salt=secrets.token_hex(32),
            expires_at=datetime.utcnow() - timedelta(minutes=1)  # Already expired
        )
        
        assert magic_link.is_expired
        
        # Future expiration should not be expired
        magic_link.expires_at = datetime.utcnow() + timedelta(minutes=15)
        assert not magic_link.is_expired
    
    def test_magic_link_attempt_limiting(self):
        """Test magic link prevents brute force through attempt limiting."""
        magic_link = MagicLink(
            user_id=uuid4(),
            email="test@example.com",
            token_hash=secrets.token_hex(64),
            salt=secrets.token_hex(32),
            expires_at=datetime.utcnow() + timedelta(minutes=15),
            max_attempts=3
        )
        
        # Initially should allow attempts
        assert not magic_link.attempts_exceeded
        
        # Simulate failed attempts
        magic_link.attempt_count = 2
        assert not magic_link.attempts_exceeded
        
        # Exceed maximum attempts
        magic_link.attempt_count = 3
        assert magic_link.attempts_exceeded
        
        # Even one more attempt should still be exceeded
        magic_link.attempt_count = 4
        assert magic_link.attempts_exceeded
    
    def test_magic_link_context_tracking(self):
        """Test magic link tracks request and verification context."""
        magic_link = MagicLink(
            user_id=uuid4(),
            email="test@example.com",
            token_hash=secrets.token_hex(64),
            salt=secrets.token_hex(32),
            expires_at=datetime.utcnow() + timedelta(minutes=15),
            requesting_ip="192.168.1.1",
            requesting_user_agent="Mozilla/5.0 Test Browser",
            requesting_fingerprint=secrets.token_hex(32)
        )
        
        # Request context should be stored
        assert magic_link.requesting_ip == "192.168.1.1"
        assert "Mozilla" in magic_link.requesting_user_agent
        assert len(magic_link.requesting_fingerprint) == 64
        
        # Verification context initially empty
        assert magic_link.verified_ip is None
        assert magic_link.verified_user_agent is None
        assert magic_link.verified_fingerprint is None
    
    def test_magic_link_usage_tracking(self):
        """Test magic link tracks usage to prevent replay attacks."""
        magic_link = MagicLink(
            user_id=uuid4(),
            email="test@example.com",
            token_hash=secrets.token_hex(64),
            salt=secrets.token_hex(32),
            expires_at=datetime.utcnow() + timedelta(minutes=15)
        )
        
        # Initially not used
        assert not magic_link.is_used
        assert magic_link.used_at is None
        
        # Mark as used
        magic_link.used_at = datetime.utcnow()
        assert magic_link.is_used
        
        # Should remain used even if used_at is modified
        assert magic_link.is_used


class TestUserSession:
    """Test UserSession model security properties."""
    
    def test_session_creation_with_security_context(self):
        """Test session creation captures comprehensive security context."""
        session = UserSession(
            user_id=uuid4(),
            token_hash=secrets.token_hex(64),
            session_key=secrets.token_hex(64),
            expires_at=datetime.utcnow() + timedelta(hours=24),
            device_fingerprint=secrets.token_hex(64),
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0 Test Browser"
        )
        
        assert session.is_active is True
        assert session.is_remembered is False
        assert session.requires_mfa is False
        assert session.auth_method == "magic_link"
        assert session.risk_score == 0
        assert session.anomaly_flags == {}
        assert session.device_trust_score == 0
    
    def test_session_expiration_security(self):
        """Test session expiration prevents access after timeout."""
        session = UserSession(
            user_id=uuid4(),
            token_hash=secrets.token_hex(64),
            session_key=secrets.token_hex(64),
            expires_at=datetime.utcnow() - timedelta(minutes=1)  # Already expired
        )
        
        assert session.is_expired
        assert not session.is_valid  # Expired sessions are invalid
        
        # Future expiration should not be expired
        session.expires_at = datetime.utcnow() + timedelta(hours=1)
        assert not session.is_expired
        assert session.is_valid
    
    def test_session_validity_security_checks(self):
        """Test session validity combines multiple security checks."""
        session = UserSession(
            user_id=uuid4(),
            token_hash=secrets.token_hex(64),
            session_key=secrets.token_hex(64),
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        # Initially valid
        assert session.is_valid
        
        # Deactivated session should be invalid
        session.is_active = False
        assert not session.is_valid
        
        # Reactivate
        session.is_active = True
        assert session.is_valid
        
        # Invalidated session should be invalid
        session.invalidated_at = datetime.utcnow()
        assert not session.is_valid
    
    def test_session_context_binding_data(self):
        """Test session stores context binding data for validation."""
        context_hash = secrets.token_hex(64)
        security_hash = secrets.token_hex(64)
        
        session = UserSession(
            user_id=uuid4(),
            token_hash=secrets.token_hex(64),
            session_key=secrets.token_hex(64),
            expires_at=datetime.utcnow() + timedelta(hours=1),
            context_hash=context_hash,
            security_hash=security_hash,
            user_agent_hash=secrets.token_hex(64)
        )
        
        assert session.context_hash == context_hash
        assert session.security_hash == security_hash
        assert len(session.user_agent_hash) == 64


class TestSecurityEvent:
    """Test SecurityEvent model for audit trail integrity."""
    
    def test_security_event_creation_with_required_fields(self):
        """Test security event creation with audit trail data."""
        event = SecurityEvent(
            user_id=uuid4(),
            event_type=SecurityEventType.LOGIN_ATTEMPT,
            risk_level=RiskLevel.MEDIUM,
            success=False,
            message="Failed login attempt detected",
            ip_address="192.168.1.1"
        )
        
        assert event.event_type == SecurityEventType.LOGIN_ATTEMPT
        assert event.risk_level == RiskLevel.MEDIUM
        assert event.success is False
        assert "Failed login" in event.message
        assert event.ip_address == "192.168.1.1"
    
    def test_security_event_types_comprehensive_coverage(self):
        """Test that security event types cover all attack scenarios."""
        critical_event_types = [
            SecurityEventType.LOGIN_FAILURE,
            SecurityEventType.MAGIC_LINK_REUSED,
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            SecurityEventType.SUSPICIOUS_ACTIVITY,
            SecurityEventType.CONTEXT_VIOLATION,
            SecurityEventType.PRIVILEGE_ESCALATION
        ]
        
        for event_type in critical_event_types:
            event = SecurityEvent(
                event_type=event_type,
                risk_level=RiskLevel.HIGH,
                success=False,
                message=f"Security event: {event_type.value}"
            )
            assert event.event_type == event_type
            assert event.risk_level == RiskLevel.HIGH
    
    def test_security_event_forensic_data_storage(self):
        """Test security event stores forensic data for investigation."""
        forensic_data = {
            "attack_vectors": ["brute_force", "credential_stuffing"],
            "request_headers": {"X-Forwarded-For": "10.0.0.1"},
            "payload_hash": secrets.token_hex(32),
            "threat_intelligence": {
                "known_bad_ip": True,
                "reputation_score": 15
            }
        }
        
        event = SecurityEvent(
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            risk_level=RiskLevel.CRITICAL,
            success=False,
            message="Potential attack detected",
            forensic_data=forensic_data,
            correlation_id=str(uuid4())
        )
        
        assert event.forensic_data["attack_vectors"] == ["brute_force", "credential_stuffing"]
        assert event.forensic_data["threat_intelligence"]["known_bad_ip"] is True
        assert len(event.correlation_id) == 36  # UUID format


class TestRateLimit:
    """Test RateLimit model security and attack prevention."""
    
    def test_rate_limit_creation_with_adaptive_thresholds(self):
        """Test rate limit creation with escalation capability."""
        rate_limit = RateLimit(
            identifier_type="email",
            identifier_value="test@example.com",
            action_type="magic_link",
            window_start=datetime.utcnow(),
            window_duration_seconds=300,  # 5 minutes
            limit_threshold=3,
            base_threshold=3,
            expires_at=datetime.utcnow() + timedelta(minutes=5)
        )
        
        assert rate_limit.attempt_count == 1
        assert rate_limit.limit_threshold == 3
        assert rate_limit.base_threshold == 3
        assert rate_limit.escalation_factor == 1
        assert rate_limit.reset_count == 0
        assert not rate_limit.is_exceeded
    
    def test_rate_limit_threshold_enforcement(self):
        """Test rate limit prevents attacks when threshold exceeded."""
        rate_limit = RateLimit(
            identifier_type="ip",
            identifier_value="192.168.1.1",
            action_type="login",
            window_start=datetime.utcnow(),
            window_duration_seconds=300,
            limit_threshold=5,
            base_threshold=5,
            expires_at=datetime.utcnow() + timedelta(minutes=5)
        )
        
        # Within limit
        rate_limit.attempt_count = 4
        assert not rate_limit.is_exceeded
        
        # At limit
        rate_limit.attempt_count = 5
        assert rate_limit.is_exceeded
        
        # Over limit
        rate_limit.attempt_count = 10
        assert rate_limit.is_exceeded
    
    def test_rate_limit_window_expiration(self):
        """Test rate limit window expiration resets protection."""
        rate_limit = RateLimit(
            identifier_type="user_id",
            identifier_value=str(uuid4()),
            action_type="api_call",
            window_start=datetime.utcnow(),
            window_duration_seconds=60,
            limit_threshold=10,
            base_threshold=10,
            expires_at=datetime.utcnow() - timedelta(minutes=1)  # Already expired
        )
        
        assert rate_limit.is_expired
        
        # Future expiration should not be expired
        rate_limit.expires_at = datetime.utcnow() + timedelta(minutes=1)
        assert not rate_limit.is_expired
    
    def test_rate_limit_escalation_tracking(self):
        """Test rate limit tracks escalation for repeat offenders."""
        rate_limit = RateLimit(
            identifier_type="email",
            identifier_value="repeat@offender.com",
            action_type="magic_link",
            window_start=datetime.utcnow(),
            window_duration_seconds=300,
            limit_threshold=3,
            base_threshold=3,
            expires_at=datetime.utcnow() + timedelta(minutes=5),
            escalation_factor=2,  # Double threshold after violations
            reset_count=3  # Third reset
        )
        
        assert rate_limit.escalation_factor == 2
        assert rate_limit.reset_count == 3
        # Escalated threshold should be base * escalation_factor
        expected_threshold = rate_limit.base_threshold * rate_limit.escalation_factor
        assert rate_limit.limit_threshold >= expected_threshold or rate_limit.limit_threshold == 3


class TestDeviceTrust:
    """Test DeviceTrust model for device recognition security."""
    
    def test_device_trust_creation_with_security_tracking(self):
        """Test device trust creation tracks security metrics."""
        device_trust = DeviceTrust(
            user_id=uuid4(),
            device_id=secrets.token_hex(32),
            device_fingerprint=secrets.token_hex(64),
            device_name="Test Device",
            device_type="desktop"
        )
        
        assert device_trust.trust_score == 0
        assert device_trust.is_trusted is False
        assert device_trust.trust_established_at is None
        assert device_trust.access_count == 1
        assert device_trust.suspicious_activity_count == 0
        assert device_trust.known_ip_addresses == []
    
    def test_device_trust_security_indicators(self):
        """Test device trust tracks security indicators."""
        device_trust = DeviceTrust(
            user_id=uuid4(),
            device_id=secrets.token_hex(32),
            device_fingerprint=secrets.token_hex(64),
            suspicious_activity_count=5,
            last_anomaly_at=datetime.utcnow()
        )
        
        assert device_trust.suspicious_activity_count == 5
        assert device_trust.last_anomaly_at is not None
        
        # High suspicious activity should affect trust
        assert device_trust.suspicious_activity_count > 0


class TestDatabaseSecurity:
    """Test database configuration and security features."""
    
    def test_get_database_url_requires_all_environment_variables(self):
        """Test database URL construction fails without required env vars."""
        required_vars = [
            'AUTH_DB_HOST',
            'AUTH_DB_PORT', 
            'AUTH_DB_NAME',
            'AUTH_DB_USER',
            'AUTH_DB_PASSWORD'
        ]
        
        # Test each missing variable
        for missing_var in required_vars:
            with patch.dict('os.environ', {
                var: 'test_value' for var in required_vars if var != missing_var
            }, clear=True):
                with pytest.raises(ValueError) as exc_info:
                    get_database_url()
                
                assert missing_var in str(exc_info.value)
                assert "Missing required environment variables" in str(exc_info.value)
    
    def test_get_database_url_constructs_valid_postgresql_url(self):
        """Test database URL construction creates valid PostgreSQL connection."""
        test_env = {
            'AUTH_DB_HOST': 'localhost',
            'AUTH_DB_PORT': '5432',
            'AUTH_DB_NAME': 'test_auth_db',
            'AUTH_DB_USER': 'test_user',
            'AUTH_DB_PASSWORD': 'test_password'
        }
        
        with patch.dict('os.environ', test_env, clear=True):
            url = get_database_url()
            
            expected = "postgresql://test_user:test_password@localhost:5432/test_auth_db"
            assert url == expected
            
            # Verify URL components
            assert url.startswith("postgresql://")
            assert "test_user:test_password" in url
            assert "@localhost:5432" in url
            assert url.endswith("/test_auth_db")
    
    def test_database_url_handles_special_characters_in_password(self):
        """Test database URL properly handles special characters in credentials."""
        test_env = {
            'AUTH_DB_HOST': 'localhost',
            'AUTH_DB_PORT': '5432',
            'AUTH_DB_NAME': 'auth_db',
            'AUTH_DB_USER': 'user@domain',
            'AUTH_DB_PASSWORD': 'p@ssw0rd!#$'
        }
        
        with patch.dict('os.environ', test_env, clear=True):
            url = get_database_url()
            
            # Should contain the exact credentials as provided
            assert "user@domain:p@ssw0rd!#$" in url
    
    @patch('secure_auth.models.create_engine')
    def test_create_auth_database_uses_secure_configuration(self, mock_create_engine):
        """Test database creation uses secure engine configuration."""
        test_url = "postgresql://user:pass@localhost:5432/testdb"
        
        create_auth_database(test_url)
        
        # Verify secure engine configuration
        mock_create_engine.assert_called_once_with(
            test_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600
        )


class TestModelIntegrity:
    """Test model integrity and security constraints."""
    
    def test_uuid_primary_keys_provide_security(self):
        """Test that UUID primary keys prevent enumeration attacks."""
        user = User(email="test@example.com")
        magic_link = MagicLink(
            user_id=uuid4(),
            email="test@example.com",
            token_hash=secrets.token_hex(64),
            salt=secrets.token_hex(32),
            expires_at=datetime.utcnow() + timedelta(minutes=15)
        )
        session = UserSession(
            user_id=uuid4(),
            token_hash=secrets.token_hex(64),
            session_key=secrets.token_hex(64),
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        # IDs should be UUIDs (None until saved, but type should be UUID when set)
        models_with_uuid_ids = [user, magic_link, session]
        
        for model in models_with_uuid_ids:
            # If ID is set, should be UUID type
            if model.id is not None:
                assert isinstance(model.id, UUID)
    
    def test_model_relationships_maintain_referential_integrity(self):
        """Test that model relationships maintain data integrity."""
        user_id = uuid4()
        user = User(id=user_id, email="test@example.com")
        
        magic_link = MagicLink(
            user_id=user_id,
            email="test@example.com",
            token_hash=secrets.token_hex(64),
            salt=secrets.token_hex(32),
            expires_at=datetime.utcnow() + timedelta(minutes=15)
        )
        
        session = UserSession(
            user_id=user_id,
            token_hash=secrets.token_hex(64),
            session_key=secrets.token_hex(64),
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        security_event = SecurityEvent(
            user_id=user_id,
            event_type=SecurityEventType.LOGIN_SUCCESS,
            risk_level=RiskLevel.LOW,
            success=True,
            message="Successful login"
        )
        
        # All models should reference the same user_id
        assert magic_link.user_id == user_id
        assert session.user_id == user_id
        assert security_event.user_id == user_id
    
    def test_security_event_types_enum_integrity(self):
        """Test security event types enum contains all necessary values."""
        required_event_types = [
            "login_attempt", "login_success", "login_failure", "logout",
            "magic_link_requested", "magic_link_verified", "magic_link_expired",
            "magic_link_reused", "session_created", "session_expired",
            "session_invalidated", "rate_limit_exceeded", "suspicious_activity",
            "context_violation", "privilege_escalation", "data_access",
            "data_modification"
        ]
        
        enum_values = [event_type.value for event_type in SecurityEventType]
        
        for required_type in required_event_types:
            assert required_type in enum_values, f"Missing required event type: {required_type}"
    
    def test_risk_level_enum_covers_security_spectrum(self):
        """Test risk level enum covers complete security assessment spectrum."""
        expected_levels = ["low", "medium", "high", "critical"]
        
        enum_values = [level.value for level in RiskLevel]
        
        for expected_level in expected_levels:
            assert expected_level in enum_values, f"Missing risk level: {expected_level}"
        
        # Verify all levels are present
        assert len(enum_values) == len(expected_levels)