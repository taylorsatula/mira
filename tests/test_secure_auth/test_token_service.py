"""
Security-focused tests for SecurityTokenService.

Tests ACTUAL cryptographic properties and attack resistance.
NO MOCKING of cryptographic functions - tests the real implementation.
"""

import pytest
import secrets
import time
import os
import hashlib
import hmac
from unittest.mock import patch
from datetime import datetime, timedelta

from secure_auth.token_service import SecurityTokenService, TokenType


class TestCryptographicSecurity:
    """Test core cryptographic security properties."""
    
    @pytest.fixture
    def token_service(self):
        """Real token service with test master key."""
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': secrets.token_hex(32)}):
            return SecurityTokenService()
    
    def test_master_key_validation_security(self):
        """Test master key validation prevents weak keys."""
        # Missing key should fail
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="AUTH_MASTER_KEY environment variable is required"):
                SecurityTokenService()
        
        # Invalid hex should fail
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': 'not_hex_string'}):
            with pytest.raises(ValueError, match="AUTH_MASTER_KEY must be a valid hex string"):
                SecurityTokenService()
        
        # Short key should fail (less than 64 hex chars = 32 bytes)
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': 'abcd1234'}):
            with pytest.raises(ValueError):
                SecurityTokenService()
        
        # Valid key should work
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': secrets.token_hex(32)}):
            service = SecurityTokenService()
            assert service is not None
    
    def test_token_generation_entropy(self, token_service):
        """Test generated tokens have cryptographically secure entropy."""
        tokens = []
        salts = []
        
        # Generate many tokens to test for patterns
        for _ in range(1000):
            token, salt = token_service.generate_token()
            tokens.append(token)
            salts.append(salt)
        
        # All tokens and salts must be unique
        assert len(set(tokens)) == 1000, "Duplicate tokens detected - insufficient entropy"
        assert len(set(salts)) == 1000, "Duplicate salts detected - insufficient entropy"
        
        # Test token lengths are consistent
        token_lengths = [len(token) for token in tokens]
        assert len(set(token_lengths)) == 1, "Inconsistent token lengths"
        
        # Test salt lengths are consistent and secure (32 bytes = 64 hex chars)
        salt_lengths = [len(salt) for salt in salts]
        assert all(length == 64 for length in salt_lengths), "Salt length not 32 bytes"
    
    def test_token_character_distribution(self, token_service):
        """Test token character distribution for entropy quality."""
        tokens = []
        for _ in range(100):
            token, _ = token_service.generate_token()
            tokens.append(token)
        
        # Combine all tokens for character analysis
        all_chars = ''.join(tokens)
        char_counts = {}
        for char in all_chars:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Character distribution should be roughly even
        if char_counts:
            total_chars = len(all_chars)
            expected_freq = total_chars / len(char_counts)
            
            for char, count in char_counts.items():
                frequency = count / total_chars
                expected = expected_freq / total_chars
                
                # No character should deviate too much from expected frequency
                deviation = abs(frequency - expected) / expected
                assert deviation < 0.5, f"Character '{char}' frequency too skewed: {deviation:.2%}"
    
    def test_salt_randomness_quality(self, token_service):
        """Test salt generation produces cryptographically random values."""
        salts = []
        for _ in range(500):
            _, salt = token_service.generate_token()
            salts.append(salt)
        
        # Convert hex salts to bytes for entropy analysis
        salt_bytes = [bytes.fromhex(salt) for salt in salts]
        
        # All salts should be different
        assert len(set(salts)) == 500, "Duplicate salts detected"
        
        # Test byte distribution within salts
        all_bytes = b''.join(salt_bytes)
        byte_counts = [0] * 256
        
        for byte_val in all_bytes:
            byte_counts[byte_val] += 1
        
        # Byte values should be roughly evenly distributed
        total_bytes = len(all_bytes)
        expected_count = total_bytes / 256
        
        for i, count in enumerate(byte_counts):
            if expected_count > 0:
                deviation = abs(count - expected_count) / expected_count
                # Allow reasonable deviation for random data
                assert deviation < 0.3, f"Byte value {i} too skewed: {deviation:.2%}"


class TestHashingSecurityProperties:
    """Test cryptographic hashing security properties."""
    
    @pytest.fixture
    def token_service(self):
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': secrets.token_hex(32)}):
            return SecurityTokenService()
    
    def test_pbkdf2_iteration_count_security(self, token_service):
        """Test PBKDF2 iteration count meets security standards."""
        # OWASP recommends minimum 100,000 iterations for PBKDF2-SHA256
        assert token_service._default_iterations >= 100000
        
        # Test that iteration count is actually used
        token = "test_token"
        salt = secrets.token_hex(32)
        
        # Hash with different iteration counts should produce different results
        hash_low = token_service.hash_token(token, salt, iterations=1000)
        hash_high = token_service.hash_token(token, salt, iterations=100000)
        
        assert hash_low != hash_high, "Iteration count not affecting hash output"
    
    def test_salt_prevents_rainbow_table_attacks(self, token_service):
        """Test different salts produce different hashes for same token."""
        same_token = "identical_password"
        
        hashes = []
        salts = []
        
        # Generate same token with different salts
        for _ in range(100):
            _, salt = token_service.generate_token()
            token_hash = token_service.hash_token(same_token, salt)
            hashes.append(token_hash)
            salts.append(salt)
        
        # All salts should be unique
        assert len(set(salts)) == 100, "Duplicate salts generated"
        
        # All hashes should be unique despite same input token
        assert len(set(hashes)) == 100, "Same hash produced for identical tokens with different salts"
    
    def test_hash_avalanche_effect(self, token_service):
        """Test small input changes cause large hash changes (avalanche effect)."""
        base_token = "baseline_token_12345"
        salt = secrets.token_hex(32)
        base_hash = token_service.hash_token(base_token, salt)
        
        # Test various small modifications
        modifications = [
            "Baseline_token_12345",  # Capitalize first letter
            "baseline_token_12346",  # Change last digit
            "baseline_token_1234",   # Remove last character  
            "baseline_token_123456", # Add character
            "baseline_Token_12345",  # Capitalize middle letter
            "baseline token 12345",  # Change underscore to space
        ]
        
        for modified_token in modifications:
            modified_hash = token_service.hash_token(modified_token, salt)
            
            # Hash should be completely different
            assert modified_hash != base_hash, f"No avalanche effect for: {modified_token}"
            
            # Count different characters in hex representation
            diff_count = sum(1 for a, b in zip(base_hash, modified_hash) if a != b)
            total_chars = len(base_hash)
            diff_ratio = diff_count / total_chars
            
            # At least 50% of output should change (good avalanche effect)
            assert diff_ratio >= 0.5, f"Poor avalanche effect ({diff_ratio:.1%}) for: {modified_token}"
    
    def test_context_binding_security(self, token_service):
        """Test context binding prevents cross-context token reuse."""
        token = "context_test_token"
        salt = secrets.token_hex(32)
        
        # Hash with different contexts
        contexts = [
            "user123:192.168.1.1",
            "user123:192.168.1.2",  # Different IP
            "user456:192.168.1.1",  # Different user
            "user123:10.0.0.1",     # Different network
            None,                    # No context
        ]
        
        hashes = []
        for context in contexts:
            hash_val = token_service.hash_token(token, salt, additional_context=context)
            hashes.append(hash_val)
        
        # All hashes should be different
        assert len(set(hashes)) == len(contexts), "Context binding not working"
        
        # Verify verification only works with correct context
        correct_hash = token_service.hash_token(token, salt, additional_context=contexts[0])
        
        assert token_service.verify_token(token, correct_hash, salt, additional_context=contexts[0])
        assert not token_service.verify_token(token, correct_hash, salt, additional_context=contexts[1])
        assert not token_service.verify_token(token, correct_hash, salt, additional_context=contexts[2])


class TestTimingAttackResistance:
    """Test resistance to timing-based attacks."""
    
    @pytest.fixture
    def token_service(self):
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': secrets.token_hex(32)}):
            return SecurityTokenService()
    
    def test_constant_time_verification(self, token_service):
        """Test token verification takes constant time regardless of correctness."""
        # Generate a valid token and hash
        token, salt = token_service.generate_token()
        valid_hash = token_service.hash_token(token, salt)
        
        correct_times = []
        incorrect_times = []
        
        # Measure timing for correct verifications
        for _ in range(50):
            start = time.perf_counter()
            result = token_service.verify_token(token, valid_hash, salt)
            end = time.perf_counter()
            assert result is True
            correct_times.append(end - start)
        
        # Measure timing for incorrect verifications
        for _ in range(50):
            wrong_token = secrets.token_urlsafe(32)
            start = time.perf_counter()
            result = token_service.verify_token(wrong_token, valid_hash, salt)
            end = time.perf_counter()
            assert result is False
            incorrect_times.append(end - start)
        
        # Calculate average times
        avg_correct = sum(correct_times) / len(correct_times)
        avg_incorrect = sum(incorrect_times) / len(incorrect_times)
        
        # Timing difference should be minimal (< 5% difference)
        if max(avg_correct, avg_incorrect) > 0:
            timing_diff = abs(avg_correct - avg_incorrect) / max(avg_correct, avg_incorrect)
            assert timing_diff < 0.05, f"Timing difference too large: {timing_diff:.1%}"
    
    def test_verification_error_timing_consistency(self, token_service):
        """Test error conditions don't leak information through timing."""
        token, salt = token_service.generate_token()
        valid_hash = token_service.hash_token(token, salt)
        
        error_scenarios = [
            (None, valid_hash, salt),           # None token
            ("", valid_hash, salt),             # Empty token
            (token, "invalid_hash", salt),      # Invalid hash
            (token, valid_hash, "invalid_salt"), # Invalid salt
            ("wrong_token", valid_hash, salt),  # Wrong token
        ]
        
        timings = []
        for scenario_token, scenario_hash, scenario_salt in error_scenarios:
            scenario_times = []
            
            for _ in range(20):
                start = time.perf_counter()
                try:
                    result = token_service.verify_token(scenario_token, scenario_hash, scenario_salt)
                    assert result is False
                except:
                    pass  # Some scenarios may raise exceptions
                end = time.perf_counter()
                scenario_times.append(end - start)
            
            timings.append(scenario_times)
        
        # All error scenarios should have similar timing characteristics
        avg_times = [sum(times) / len(times) for times in timings]
        
        if max(avg_times) > 0:
            for i, avg_time in enumerate(avg_times):
                for j, other_avg in enumerate(avg_times[i+1:], i+1):
                    timing_diff = abs(avg_time - other_avg) / max(avg_time, other_avg)
                    assert timing_diff < 0.2, f"Error timing leak between scenarios {i} and {j}: {timing_diff:.1%}"


class TestSessionContextSecurity:
    """Test session context generation security."""
    
    @pytest.fixture
    def token_service(self):
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': secrets.token_hex(32)}):
            return SecurityTokenService()
    
    def test_context_hash_deterministic(self, token_service):
        """Test context hash is deterministic for same inputs."""
        ip = "192.168.1.100"
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        fp = "device_fingerprint_12345"
        
        # Generate same context hash multiple times
        hashes = []
        for _ in range(10):
            context_hash = token_service.generate_session_context_hash(ip, ua, fp)
            hashes.append(context_hash)
        
        # All hashes should be identical for same inputs
        assert len(set(hashes)) == 1, "Context hash not deterministic"
    
    def test_context_hash_sensitivity(self, token_service):
        """Test context hash changes with any input change."""
        base_ip = "192.168.1.100"
        base_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        base_fp = "device_fingerprint_12345"
        
        base_hash = token_service.generate_session_context_hash(base_ip, base_ua, base_fp)
        
        # Test changes to each component
        variations = [
            ("192.168.1.101", base_ua, base_fp),  # IP change
            (base_ip, "Mozilla/5.0 (Different Browser)", base_fp),  # UA change
            (base_ip, base_ua, "different_fingerprint"),  # FP change
            (None, base_ua, base_fp),  # Missing IP
            (base_ip, None, base_fp),  # Missing UA
            (base_ip, base_ua, None),  # Missing FP
            (None, None, None),        # All missing
        ]
        
        for ip, ua, fp in variations:
            variant_hash = token_service.generate_session_context_hash(ip, ua, fp)
            assert variant_hash != base_hash, f"Context hash didn't change for: {ip}, {ua}, {fp}"
    
    def test_device_fingerprint_generation(self, token_service):
        """Test device fingerprint generation security properties."""
        # Test deterministic generation
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        resolution = "1920x1080"
        timezone = "America/New_York"
        language = "en-US"
        
        fingerprints = []
        for _ in range(5):
            fp = token_service.generate_device_fingerprint(ua, resolution, timezone, language)
            fingerprints.append(fp)
        
        # Should be deterministic
        assert len(set(fingerprints)) == 1, "Device fingerprint not deterministic"
        
        # Test sensitivity to changes
        base_fp = token_service.generate_device_fingerprint(ua, resolution, timezone, language)
        
        variations = [
            ("Different User Agent", resolution, timezone, language),
            (ua, "1366x768", timezone, language),
            (ua, resolution, "Europe/London", language),
            (ua, resolution, timezone, "fr-FR"),
            (None, resolution, timezone, language),
        ]
        
        for var_ua, var_res, var_tz, var_lang in variations:
            var_fp = token_service.generate_device_fingerprint(var_ua, var_res, var_tz, var_lang)
            assert var_fp != base_fp, f"Fingerprint didn't change for variation"
    
    def test_context_includes_master_key_binding(self, token_service):
        """Test context hash includes master key for security."""
        # Generate context hash
        context_hash = token_service.generate_session_context_hash(
            "192.168.1.1", "Mozilla/5.0", "fingerprint"
        )
        
        # Create new service with different master key
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': secrets.token_hex(32)}):
            different_service = SecurityTokenService()
            different_hash = different_service.generate_session_context_hash(
                "192.168.1.1", "Mozilla/5.0", "fingerprint"
            )
        
        # Hashes should be different with different master keys
        assert context_hash != different_hash, "Context hash not bound to master key"


class TestEncryptionSecurity:
    """Test encryption/decryption security properties."""
    
    @pytest.fixture
    def token_service(self):
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': secrets.token_hex(32)}):
            return SecurityTokenService()
    
    def test_encryption_roundtrip_integrity(self, token_service):
        """Test encryption/decryption maintains data integrity."""
        test_data = [
            "simple_string",
            "Complex Data! @#$%^&*()_+-={}[]|\\:;\"'<>?,./ with symbols",
            "unicode_test_ñáéíóú_中文_русский_العربية",
            "a" * 1000,  # Long string
            "",  # Empty string
            "Line1\nLine2\nLine3\n",  # Multiline
        ]
        
        for data in test_data:
            encrypted, nonce = token_service.encrypt_sensitive_data(data)
            decrypted = token_service.decrypt_sensitive_data(encrypted)
            
            assert decrypted == data, f"Roundtrip failed for: {data[:50]}..."
    
    def test_encryption_randomness(self, token_service):
        """Test encryption produces different outputs for same input."""
        data = "test_encryption_data"
        
        encryptions = []
        for _ in range(100):
            encrypted, nonce = token_service.encrypt_sensitive_data(data)
            encryptions.append(encrypted)
        
        # All encryptions should be unique (due to random nonce)
        assert len(set(encryptions)) == 100, "Encryption not producing unique outputs"
    
    def test_encryption_with_aad(self, token_service):
        """Test authenticated encryption with additional data."""
        data = "sensitive_data"
        aad = "additional_authenticated_data"
        
        # Encrypt with AAD
        encrypted, nonce = token_service.encrypt_sensitive_data(data, aad)
        
        # Should decrypt correctly with same AAD
        decrypted = token_service.decrypt_sensitive_data(encrypted, aad)
        assert decrypted == data
        
        # Should fail with different AAD
        wrong_aad = "wrong_additional_data"
        failed_decrypt = token_service.decrypt_sensitive_data(encrypted, wrong_aad)
        assert failed_decrypt is None
        
        # Should fail with missing AAD
        failed_decrypt = token_service.decrypt_sensitive_data(encrypted)
        assert failed_decrypt is None
    
    def test_encryption_tamper_detection(self, token_service):
        """Test encryption detects tampering."""
        data = "tamper_test_data"
        encrypted, nonce = token_service.encrypt_sensitive_data(data)
        
        # Tamper with encrypted data
        encrypted_bytes = bytes.fromhex(encrypted)
        tampered_bytes = bytearray(encrypted_bytes)
        
        # Flip a bit in the middle
        tampered_bytes[len(tampered_bytes) // 2] ^= 0x01
        tampered_hex = tampered_bytes.hex()
        
        # Decryption should fail for tampered data
        result = token_service.decrypt_sensitive_data(tampered_hex)
        assert result is None, "Tampered data should not decrypt"


class TestCSRFTokenSecurity:
    """Test CSRF token security properties."""
    
    @pytest.fixture
    def token_service(self):
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': secrets.token_hex(32)}):
            return SecurityTokenService()
    
    def test_csrf_token_pair_generation(self, token_service):
        """Test CSRF token pair generation produces secure tokens."""
        pairs = []
        
        for _ in range(100):
            token_value, token_hash = token_service.generate_csrf_token_pair()
            pairs.append((token_value, token_hash))
        
        # All values should be unique
        values = [pair[0] for pair in pairs]
        hashes = [pair[1] for pair in pairs]
        
        assert len(set(values)) == 100, "Duplicate CSRF token values"
        assert len(set(hashes)) == 100, "Duplicate CSRF token hashes"
    
    def test_csrf_token_verification_security(self, token_service):
        """Test CSRF token verification security properties."""
        token_value, token_hash = token_service.generate_csrf_token_pair()
        
        # Valid pair should verify
        assert token_service.verify_csrf_token_pair(token_value, token_hash)
        
        # Wrong combinations should fail
        wrong_value, _ = token_service.generate_csrf_token_pair()
        _, wrong_hash = token_service.generate_csrf_token_pair()
        
        assert not token_service.verify_csrf_token_pair(wrong_value, token_hash)
        assert not token_service.verify_csrf_token_pair(token_value, wrong_hash)
        assert not token_service.verify_csrf_token_pair(wrong_value, wrong_hash)
    
    def test_csrf_timing_attack_resistance(self, token_service):
        """Test CSRF verification resists timing attacks."""
        token_value, token_hash = token_service.generate_csrf_token_pair()
        
        valid_times = []
        invalid_times = []
        
        # Time valid verifications
        for _ in range(50):
            start = time.perf_counter()
            result = token_service.verify_csrf_token_pair(token_value, token_hash)
            end = time.perf_counter()
            assert result is True
            valid_times.append(end - start)
        
        # Time invalid verifications
        for _ in range(50):
            wrong_value = secrets.token_urlsafe(32)
            start = time.perf_counter()
            result = token_service.verify_csrf_token_pair(wrong_value, token_hash)
            end = time.perf_counter()
            assert result is False
            invalid_times.append(end - start)
        
        # Timing should be similar
        avg_valid = sum(valid_times) / len(valid_times)
        avg_invalid = sum(invalid_times) / len(invalid_times)
        
        if max(avg_valid, avg_invalid) > 0:
            timing_diff = abs(avg_valid - avg_invalid) / max(avg_valid, avg_invalid)
            assert timing_diff < 0.1, f"CSRF timing difference too large: {timing_diff:.1%}"


class TestTokenExpiryLogic:
    """Test token expiry calculation security."""
    
    @pytest.fixture
    def token_service(self):
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': secrets.token_hex(32)}):
            return SecurityTokenService()
    
    def test_token_expiry_times_secure(self, token_service):
        """Test token expiry times are within secure bounds."""
        now = datetime.utcnow()
        
        # Test all token types
        expiry_tests = [
            (TokenType.MAGIC_LINK, timedelta(minutes=5), timedelta(minutes=15)),
            (TokenType.SESSION, timedelta(hours=4), timedelta(hours=12)),
            (TokenType.CSRF, timedelta(hours=8), timedelta(hours=24)),
            (TokenType.API_KEY, timedelta(days=30), timedelta(days=400)),
            (TokenType.REFRESH, timedelta(days=3), timedelta(days=14)),
        ]
        
        for token_type, min_expiry, max_expiry in expiry_tests:
            expiry = token_service.get_token_expiry(token_type)
            time_diff = expiry - now
            
            assert min_expiry <= time_diff <= max_expiry, \
                f"{token_type} expiry {time_diff} outside secure range [{min_expiry}, {max_expiry}]"
    
    def test_remember_device_expiry_extension(self, token_service):
        """Test remember device extends session expiry appropriately."""
        # Regular session expiry
        regular_expiry = token_service.get_token_expiry(TokenType.SESSION, remember_device=False)
        
        # Remember device expiry
        remember_expiry = token_service.get_token_expiry(TokenType.SESSION, remember_device=True)
        
        # Remember device should extend expiry
        assert remember_expiry > regular_expiry
        
        # But not excessively (should be reasonable)
        time_diff = remember_expiry - regular_expiry
        assert timedelta(days=7) <= time_diff <= timedelta(days=90)


class TestServiceConfiguration:
    """Test service configuration security."""
    
    def test_secure_default_configuration(self):
        """Test service initializes with secure defaults."""
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': secrets.token_hex(32)}):
            service = SecurityTokenService()
            
            # Iteration count should meet security standards
            assert service._default_iterations >= 100000
            
            # Salt length should be adequate (32 bytes)
            assert service._salt_length >= 32
            
            # Token length should be adequate (64 bytes default)
            assert service._token_length >= 32
    
    def test_master_key_isolation(self):
        """Test master key is properly isolated between instances."""
        key1 = secrets.token_hex(32)
        key2 = secrets.token_hex(32)
        
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': key1}):
            service1 = SecurityTokenService()
            token1, salt1 = service1.generate_token()
            hash1 = service1.hash_token(token1, salt1)
        
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': key2}):
            service2 = SecurityTokenService()
            # Same token and salt, different master key
            hash2 = service2.hash_token(token1, salt1)
        
        # Different master keys should produce different hashes
        assert hash1 != hash2, "Master key not properly isolated"
    
    def test_singleton_behavior_security(self):
        """Test singleton behavior doesn't compromise security."""
        from secure_auth.token_service import get_token_service, _token_service
        
        # Clear singleton
        import secure_auth.token_service
        secure_auth.token_service._token_service = None
        
        with patch.dict('os.environ', {'AUTH_MASTER_KEY': secrets.token_hex(32)}):
            service1 = get_token_service()
            service2 = get_token_service()
            
            # Should return same instance
            assert service1 is service2
            
            # Should function correctly
            token, salt = service1.generate_token()
            hash_val = service1.hash_token(token, salt)
            assert service2.verify_token(token, hash_val, salt)