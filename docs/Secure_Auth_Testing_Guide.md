# Secure Authentication System Testing Guide for Claude

## 🎯 **Core Testing Philosophy**

Claude, when testing the `secure_auth` system, your primary goal is **security validation, not code coverage**. Every test should answer: "How would an attacker try to break this, and would my test catch it?"

**Think like an attacker, test like a defender.** Your tests should simulate real attack scenarios that would happen in production, not just verify that code executes without errors.

## 🚨 **Critical Testing Principles**

### **1. Test Attack Scenarios, Not Happy Paths**
```python
# ❌ BAD - Tests that the code works
def test_magic_link_verification_success():
    result = auth_service.verify_magic_link("valid_token")
    assert result.success is True

# ✅ GOOD - Tests that attacks fail
def test_magic_link_brute_force_protection():
    # Try 1000 random tokens against same magic link
    for _ in range(1000):
        result = auth_service.verify_magic_link(secrets.token_urlsafe(43))
        assert result.success is False
    # Verify original link is still invalidated after attacks
```

### **2. Never Mock Core Security Functions**
```python
# ❌ BAD - Mocks the security, tests nothing
with patch.object(auth_service.token_service, 'verify_token', return_value=True):
    result = auth_service.verify_magic_link("fake_token")

# ✅ GOOD - Tests actual cryptographic verification
def test_token_verification_cryptographic_security():
    token, salt = token_service.generate_token()
    hash_value = token_service.hash_token(token, salt)
    
    # Correct token should verify
    assert token_service.verify_token(token, hash_value, salt) is True
    
    # Similar tokens should not verify
    assert token_service.verify_token(token + "x", hash_value, salt) is False
```

### **4. Use Real Database Integration**
```python
# ✅ EXCELLENT - Tests with real database persistence
def test_rate_limiting_blocks_distributed_attack(rate_limit_service, db_session):
    """Test that attackers cannot bypass rate limits with multiple IPs."""
    target_email = "victim@example.com"
    
    # Simulate distributed attack from 10 different IPs
    attacker_ips = [f"203.0.113.{i}" for i in range(50, 60)]
    
    for ip in attacker_ips:
        result = rate_limit_service.record_attempt(
            identifier_type="email", identifier_value=target_email, action=MAGIC_LINK
        )
        if not result.allowed:
            break
    
    # Verify database shows email is rate limited
    rate_limit = db_session.query(RateLimit).filter(
        RateLimit.identifier_value == target_email
    ).first()
    assert rate_limit.is_exceeded
```

### **3. Test Advanced Cryptographic Security Properties**
```python
# ✅ EXCELLENT - Validates comprehensive cryptographic properties
def test_token_entropy_and_randomness():
    """Test token generation has cryptographically secure entropy."""
    tokens = []
    salts = []
    
    # Generate large sample for statistical analysis
    for _ in range(1000):
        token, salt = token_service.generate_token()
        tokens.append(token)
        salts.append(salt)
    
    # 1. All tokens and salts must be unique
    assert len(set(tokens)) == 1000, "Duplicate tokens detected - insufficient entropy"
    assert len(set(salts)) == 1000, "Duplicate salts detected - insufficient entropy"
    
    # 2. Character distribution should be roughly uniform
    all_chars = ''.join(tokens)
    char_counts = {}
    for char in all_chars:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    if char_counts:
        total_chars = len(all_chars)
        expected_freq = total_chars / len(char_counts)
        
        for char, count in char_counts.items():
            frequency = count / total_chars
            expected = expected_freq / total_chars
            deviation = abs(frequency - expected) / expected
            assert deviation < 0.3, f"Character '{char}' frequency too skewed: {deviation:.2%}"
    
    # 3. Test salt randomness quality
    salt_bytes = [bytes.fromhex(salt) for salt in salts[:100]]
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
            assert deviation < 0.5, f"Byte value {i} too skewed: {deviation:.2%}"


def test_hash_avalanche_effect():
    """
    Test that small input changes cause large hash changes.
    
    Security property: Cryptographic avalanche effect prevents
    attackers from using hash similarities to guess inputs.
    """
    base_token = "baseline_token_12345"
    salt = secrets.token_hex(32)
    base_hash = token_service.hash_token(base_token, salt)
    
    # Test single character modifications
    modifications = [
        "Baseline_token_12345",  # Case change
        "baseline_token_12346",  # Last digit change  
        "baseline_token_1234",   # Character removal
        "xbaseline_token_12345", # Character addition
        "baseline_Token_12345",  # Middle case change
        "baseline token 12345",  # Punctuation change
    ]
    
    for modified in modifications:
        modified_hash = token_service.hash_token(modified, salt)
        
        # Hash should be completely different
        assert modified_hash != base_hash, f"No avalanche effect for: {modified}"
        
        # Count different characters (Hamming distance)
        diff_count = sum(1 for a, b in zip(base_hash, modified_hash) if a != b)
        total_chars = len(base_hash)
        diff_ratio = diff_count / total_chars
        
        # At least 50% of output should change (good avalanche effect)
        assert diff_ratio >= 0.5, f"Poor avalanche: {diff_ratio:.1%} for {modified}"


def test_pbkdf2_iteration_security():
    """Test PBKDF2 iteration count meets security standards."""
    # OWASP recommends minimum 100,000 iterations for PBKDF2-SHA256
    assert token_service._default_iterations >= 100000
    
    # Test that iteration count actually affects hash output
    token = "test_token"
    salt = secrets.token_hex(32)
    
    hash_low = token_service.hash_token(token, salt, iterations=1000)
    hash_high = token_service.hash_token(token, salt, iterations=100000)
    
    assert hash_low != hash_high, "Iteration count not affecting hash output"


def test_salt_prevents_rainbow_table_attacks():
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
    assert len(set(hashes)) == 100, "Same hash for identical tokens with different salts"


def test_context_binding_security():
    """Test context binding prevents cross-context token reuse."""
    token = "context_test_token"
    salt = secrets.token_hex(32)
    
    # Hash with different contexts
    contexts = [
        "user123:192.168.1.1",
        "user123:192.168.1.2",  # Different IP
        "user456:192.168.1.1",  # Different user
        "user123:10.0.0.1",     # Different network
        None,                   # No context
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
```

## 🔒 **Domain-Specific Security Tests**

### **Real Attack Simulation Examples**

Use these patterns as templates for your security tests:

### **Brute Force Attack Simulation**
```python
def test_brute_force_attack_prevention():
    """Test system blocks sustained brute force attacks."""
    attacker_ip = "203.0.113.50"
    
    # Simulate rapid-fire login attempts
    results = []
    for i in range(25):  # More than reasonable limit
        result = auth_service.attempt_login(
            email=f"victim{i}@example.com",
            ip_address=attacker_ip,
            user_agent="Attack Bot"
        )
        results.append(result)
        if not result.allowed:
            break
    
    # Attack should be blocked before completion
    blocked_count = sum(1 for r in results if not r.allowed)
    assert blocked_count > 0, "Brute force attack was not blocked"
```

### **Authentication Bypass Prevention**
```python
def test_authentication_bypass_attempts():
    """Test that common bypass techniques fail."""
    bypass_attempts = [
        ("", "empty_token"),
        ("admin", "predictable_token"),
        ("../../../etc/passwd", "path_traversal"),
        ("'; DROP TABLE users; --", "sql_injection"),
        ("null", "null_injection"),
        ("undefined", "undefined_injection")
    ]
    
    for attempt, description in bypass_attempts:
        result = auth_service.verify_magic_link(attempt)
        assert result.success is False, f"Bypass succeeded with {description}"
```

### **Session Hijacking Prevention**
```python
def test_session_context_binding_prevents_hijacking():
    """Test that sessions cannot be used from different contexts."""
    # Create session with specific context
    original_ip = "192.168.1.1"
    original_ua = "Original Browser"
    
    session_token = create_authenticated_session(
        user, ip_address=original_ip, user_agent=original_ua
    )
    
    # Verify session works with original context
    user_from_session = auth_service.get_user_from_session(
        session_token, ip_address=original_ip, user_agent=original_ua
    )
    assert user_from_session is not None
    
    # Verify session fails with different IP
    user_hijacked = auth_service.get_user_from_session(
        session_token, ip_address="10.0.0.1", user_agent=original_ua
    )
    assert user_hijacked is None
    
    # Verify session fails with different User-Agent
    user_hijacked = auth_service.get_user_from_session(
        session_token, ip_address=original_ip, user_agent="Hijacker Browser"
    )
    assert user_hijacked is None
```

### **Rate Limiting Effectiveness**
```python
def test_distributed_rate_limit_attack_prevention():
    """Test that attackers cannot bypass rate limits with multiple IPs."""
    email = "target@example.com"
    
    # Simulate distributed attack from multiple IPs
    attack_ips = [f"192.168.1.{i}" for i in range(1, 21)]  # 20 different IPs
    
    successful_requests = 0
    for ip in attack_ips:
        for attempt in range(5):  # 5 attempts per IP
            result = auth_service.request_magic_link(
                email=email, ip_address=ip, user_agent="Attack Bot"
            )
            if result.success:
                successful_requests += 1
    
    # Should be limited by email rate limit (3), not IP limits
    assert successful_requests <= 3, "Rate limiting failed against distributed attack"
```

### **Advanced Timing Attack Resistance with Statistical Analysis**
```python
def test_advanced_timing_attack_resistance():
    """
    Test timing attack resistance using statistical analysis.
    
    Attack scenario: Sophisticated attacker uses statistical methods to detect
    timing differences across large sample sizes with confidence intervals.
    """
    existing_email = test_user.email
    nonexistent_email = "nonexistent@example.com"
    
    existing_times = []
    nonexistent_times = []
    
    # Large sample size for statistical significance (50+ samples)
    for _ in range(50):
        start = time.perf_counter()
        auth_service.request_magic_link(email=existing_email)
        end = time.perf_counter()
        existing_times.append(end - start)
        
        start = time.perf_counter()
        auth_service.request_magic_link(email=nonexistent_email)
        end = time.perf_counter()
        nonexistent_times.append(end - start)
        
        time.sleep(0.1)  # Avoid rate limiting
    
    # Perform Welch's t-test for statistical significance
    def calculate_t_statistic(times1, times2):
        mean1, mean2 = sum(times1)/len(times1), sum(times2)/len(times2)
        var1 = sum((t - mean1)**2 for t in times1) / len(times1)
        var2 = sum((t - mean2)**2 for t in times2) / len(times2)
        pooled_std = ((var1 + var2) / 2) ** 0.5
        return abs(mean1 - mean2) / (pooled_std * (2/len(times1))**0.5)
    
    t_stat = calculate_t_statistic(existing_times, nonexistent_times)
    # Critical t-value for 95% confidence ≈ 2.0
    assert t_stat < 2.0, f"Statistically significant timing difference: {t_stat:.3f}"
    
    # Additional checks for timing attack resistance
    avg_existing = sum(existing_times) / len(existing_times)
    avg_nonexistent = sum(nonexistent_times) / len(nonexistent_times)
    
    # 1. Absolute timing difference should be minimal (< 10ms)
    timing_difference = abs(avg_existing - avg_nonexistent)
    assert timing_difference < 0.01, f"Timing difference too large: {timing_difference*1000:.2f}ms"
    
    # 2. Relative timing difference should be < 5%
    if min(avg_existing, avg_nonexistent) > 0:
        relative_diff = timing_difference / min(avg_existing, avg_nonexistent)
        assert relative_diff < 0.05, f"Relative timing difference: {relative_diff:.1%}"
    
    # 3. Coefficient of variation should be similar (timing consistency)
    def cv(times): 
        mean = sum(times) / len(times)
        std = (sum((t - mean)**2 for t in times) / len(times))**0.5
        return std / mean if mean > 0 else 0
    
    cv_existing = cv(existing_times)
    cv_nonexistent = cv(nonexistent_times)
    cv_diff = abs(cv_existing - cv_nonexistent)
    assert cv_diff < 0.1, f"Timing consistency varies too much: {cv_diff:.2%}"
```

### **Race Condition and Concurrent Operation Security**
```python
def test_concurrent_operation_security():
    """
    Test that concurrent operations don't create security vulnerabilities.
    
    Attack scenario: Attacker launches simultaneous requests to exploit
    race conditions in security checks or state updates.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    email = "race-test@example.com"
    results = []
    errors = []
    
    def attempt_magic_link():
        try:
            return auth_service.request_magic_link(
                email=email,
                ip_address="192.168.1.100",
                user_agent="Concurrent Attack Bot"
            )
        except Exception as e:
            errors.append(str(e))
            return None
    
    # Launch 20 concurrent requests
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(attempt_magic_link) for _ in range(20)]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    # Security validations for race conditions
    
    # 1. Should not have race condition errors  
    assert len(errors) == 0, f"Race conditions detected: {errors}"
    
    # 2. Only expected number should succeed (rate limit consistency)
    successful = [r for r in results if r.success]
    assert len(successful) <= 3, f"Race condition bypassed rate limit: {len(successful)}"
    
    # 3. Verify database consistency - no duplicate records
    rate_limits = db_session.query(RateLimit).filter(
        RateLimit.identifier_value == email
    ).all()
    assert len(rate_limits) == 1, "Race condition created duplicate records"
    
    # 4. All tokens should be unique (no duplicates from race conditions)
    if len(successful) > 1:
        magic_links = db_session.query(MagicLink).filter(
            MagicLink.email == email
        ).all()
        token_hashes = [ml.token_hash for ml in magic_links]
        assert len(token_hashes) == len(set(token_hashes)), "Duplicate tokens from race condition"


def test_concurrent_session_invalidation_race():
    """
    Test concurrent session invalidation doesn't cause database inconsistencies.
    
    Attack scenario: Multiple concurrent attempts to invalidate same session
    or related sessions should not cause race conditions.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
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
        try:
            result = session_service.invalidate_session(session_token)
            return f"Invalidated: {session_token[:10]}..."
        except Exception as e:
            errors.append(f"Error invalidating {session_token[:10]}...: {str(e)}")
            return None
    
    def invalidate_all_user_sessions():
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
```

### **Service Persistence Across Restarts**
```python
def test_security_state_survives_service_restart():
    """
    Test that security state persists across service restarts.
    
    Security requirement: Attackers must not be able to reset rate limits
    or security state by triggering application restarts.
    """
    email = "persistent-attacker@example.com"
    action = RateLimitAction.MAGIC_LINK_REQUEST
    
    # Create first service instance and exhaust rate limit
    service1 = RateLimitService(db_session=db_session)
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
    assert not check1.allowed, "Initial rate limit not working"
    
    # Simulate service restart with new instance (same database)
    service2 = RateLimitService(db_session=db_session)
    
    # Rate limit should still be enforced after "restart"
    check2 = service2.check_rate_limit(
        identifier_type="email",
        identifier_value=email,
        action=action
    )
    assert not check2.allowed, "Service restart reset rate limit"
    
    # New attempt should still be blocked
    restart_attempt = service2.record_attempt(
        identifier_type="email",
        identifier_value=email,
        action=action,
        success=False
    )
    assert not restart_attempt.allowed, "Service restart allowed bypass"
    
    # Verify same database record is being used (not duplicated)
    rate_limits = db_session.query(RateLimit).filter(
        RateLimit.identifier_type == "email",
        RateLimit.identifier_value == email
    ).all()
    assert len(rate_limits) == 1, "Service restart created duplicate records"


def test_violation_history_persists_across_restarts():
    """
    Test violation history and escalation persist across service restarts.
    
    Security requirement: Escalation based on violation history must
    not be reset by restarts, preventing repeat offenders from getting fresh starts.
    """
    email = "repeat-offender@example.com"
    action = RateLimitAction.MAGIC_LINK_REQUEST
    
    # Create first service and establish violation history
    service1 = RateLimitService(db_session=db_session)
    config = service1._get_limit_config(action, "email")
    base_limit = config["limit"]
    
    # Create initial violation
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
    rate_limit.expires_at = datetime.utcnow() - timedelta(minutes=1)  # Expire window
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
    
    # Verify escalation data persisted
    current_limit = db_session.query(RateLimit).filter(
        RateLimit.identifier_type == "email",
        RateLimit.identifier_value == email,
        RateLimit.expires_at > datetime.utcnow()
    ).first()
    
    assert current_limit is not None
    assert current_limit.reset_count >= 2, "Reset count was lost"
    assert current_limit.limit_threshold < base_limit, "Escalation was reset"
```

### **Infrastructure Security Attack Vectors**
```python
def test_reverse_proxy_header_spoofing_prevention():
    """
    Test that attackers cannot spoof reverse proxy headers.
    
    Attack scenario: Attacker sends crafted X-Forwarded-For headers
    to bypass IP-based security controls.
    """
    real_attacker_ip = "203.0.113.50"
    spoofed_headers = [
        "10.0.0.1",           # Private IP
        "127.0.0.1",          # Localhost  
        "192.168.1.1",        # Private IP
        "",                   # Empty
        "invalid-ip",         # Invalid format
        "'; DROP TABLE ips;", # Injection attempt
        "999.999.999.999",    # Invalid IP range
        "file:///etc/passwd", # Path traversal attempt
    ]
    
    for spoofed_ip in spoofed_headers:
        # Service should use real IP, not spoofed header
        result = rate_limit_service.record_attempt(
            identifier_type="ip",
            identifier_value=real_attacker_ip,  # Real IP should be tracked
            action=RateLimitAction.LOGIN_ATTEMPT,
            metadata={"x_forwarded_for": spoofed_ip}  # Spoofed header
        )
        
        # Should still be tracked under real IP
        if not result.allowed:
            break
    
    # Verify only real IP has rate limit record
    rate_limits = db_session.query(RateLimit).filter(
        RateLimit.identifier_type == "ip"
    ).all()
    
    ip_values = [rl.identifier_value for rl in rate_limits]
    assert real_attacker_ip in ip_values, "Real IP not tracked"
    assert all(spoofed not in ip_values for spoofed in spoofed_headers), \
        "Spoofed headers were trusted"


def test_user_agent_rotation_bypass_prevention():
    """
    Test User-Agent rotation cannot bypass rate limits.
    
    Attack scenario: Attacker rotates User-Agent headers to appear as
    different browsers/devices to bypass rate limiting.
    """
    attacker_ip = "198.51.100.10"
    
    # Common User-Agent strings attackers might rotate through
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
        "curl/7.68.0",
        "Python-requests/2.25.1",
        "PostmanRuntime/7.26.8",
        ""  # Empty user agent
    ]
    
    # Get limit for reference
    config = rate_limit_service._get_limit_config(
        RateLimitAction.FAILED_LOGIN, "ip"
    )
    ip_limit = config["limit"]
    
    # Make requests with rotating User-Agents
    results = []
    for i in range(ip_limit + 3):
        user_agent = user_agents[i % len(user_agents)]
        
        result = rate_limit_service.record_attempt(
            identifier_type="ip",
            identifier_value=attacker_ip,
            action=RateLimitAction.FAILED_LOGIN,
            success=False,
            metadata={"user_agent": user_agent}
        )
        results.append(result)
        
        if not result.allowed:
            break
    
    # Should be rate limited despite User-Agent rotation
    blocked_results = [r for r in results if not r.allowed]
    assert len(blocked_results) > 0, "User-Agent rotation bypassed rate limiting"


def test_combined_header_manipulation_attack():
    """
    Test combined header manipulation cannot bypass rate limits.
    
    Attack scenario: Sophisticated attacker varies multiple headers
    simultaneously to maximize chances of bypassing detection.
    """
    attacker_ip = "192.0.2.100"
    target_email = "admin@example.com"
    
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
    ip_config = rate_limit_service._get_limit_config(RateLimitAction.LOGIN_ATTEMPT, "ip")
    email_config = rate_limit_service._get_limit_config(RateLimitAction.LOGIN_ATTEMPT, "email")
    
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
            action=RateLimitAction.LOGIN_ATTEMPT,
            success=False,
            metadata=variant
        )
        ip_results.append(ip_result)
        
        # Try email-based attack  
        email_result = rate_limit_service.record_attempt(
            identifier_type="email",
            identifier_value=target_email,
            action=RateLimitAction.LOGIN_ATTEMPT,
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
```

### **Error Condition Information Leakage Prevention**
```python
def test_error_conditions_dont_leak_information():
    """
    Test that error conditions don't leak sensitive information.
    
    Attack scenario: Attacker analyzes error messages, status codes, and timing
    to gather information about system internals or user existence.
    """
    error_scenarios = [
        ("nonexistent@example.com", "user_not_found"),
        ("", "empty_email"),
        ("invalid-email", "invalid_format"),
        ("admin@" + "x" * 1000 + ".com", "oversized_input"),
        ("test@example.com'; DROP TABLE users;--", "sql_injection_attempt"),
        ("test@example.com<script>alert('xss')</script>", "xss_attempt"),
    ]
    
    error_responses = []
    error_timings = []
    
    for email, scenario in error_scenarios:
        start = time.perf_counter()
        response = client.post("/auth/request-magic-link", json={"email": email})
        end = time.perf_counter()
        
        error_responses.append((scenario, response.json(), response.status_code))
        error_timings.append(end - start)
    
    # 1. All error responses should be structurally similar
    status_codes = [r[2] for r in error_responses]
    response_structures = [set(r[1].keys()) for r in error_responses]
    
    # Should use consistent error format
    assert len(set(status_codes)) <= 2, "Inconsistent error status codes leak information"
    assert all(struct == response_structures[0] for struct in response_structures), \
        "Error response structure varies"
    
    # 2. No error should contain sensitive details
    for scenario, response, _ in error_responses:
        response_text = str(response).lower()
        sensitive_keywords = [
            "internal", "database", "stack", "debug", "sql", "query",
            "exception", "traceback", "file", "line", "function"
        ]
        assert not any(kw in response_text for kw in sensitive_keywords), \
            f"Sensitive info leaked in {scenario}: {response_text}"
    
    # 3. Error timing should be consistent (timing attack prevention)
    if len(error_timings) > 1:
        avg_timing = sum(error_timings) / len(error_timings)
        for i, timing in enumerate(error_timings):
            if avg_timing > 0:
                timing_variation = abs(timing - avg_timing) / avg_timing
                assert timing_variation < 0.3, \
                    f"Error timing variation too high for scenario {error_scenarios[i][1]}: {timing_variation:.1%}"


def test_authentication_error_consistency():
    """
    Test authentication errors don't reveal information through different paths.
    
    Attack scenario: Attacker uses different invalid inputs to trigger
    different error messages that might reveal system details.
    """
    invalid_token_scenarios = [
        ("", "empty_token"),
        ("invalid_token_123", "random_invalid"),
        (secrets.token_urlsafe(43), "valid_format_invalid_token"),
        ("a" * 1000, "oversized_token"),
        ("../../../etc/passwd", "path_traversal"),
        ("'; DROP TABLE sessions; --", "sql_injection"),
    ]
    
    timing_results = {}
    response_results = {}
    
    for token, scenario in invalid_token_scenarios:
        times = []
        responses = []
        
        # Test multiple times for timing consistency
        for _ in range(10):
            start = time.perf_counter()
            response = client.get(f"/auth/verify-magic-link/{token}")
            end = time.perf_counter()
            
            times.append(end - start)
            responses.append((response.status_code, response.json()))
        
        timing_results[scenario] = times
        response_results[scenario] = responses
    
    # All scenarios should have similar response structure
    for scenario, responses in response_results.items():
        status_codes = [r[0] for r in responses]
        response_bodies = [r[1] for r in responses]
        
        # Status codes should be consistent within scenario
        assert len(set(status_codes)) == 1, f"Inconsistent status codes in {scenario}"
        
        # Response structure should be consistent
        if response_bodies:
            first_keys = set(response_bodies[0].keys())
            assert all(set(body.keys()) == first_keys for body in response_bodies), \
                f"Inconsistent response structure in {scenario}"
    
    # Timing should be consistent across different error types
    avg_times = {scenario: sum(times)/len(times) for scenario, times in timing_results.items()}
    timing_values = list(avg_times.values())
    
    if timing_values:
        max_time = max(timing_values)
        min_time = min(timing_values)
        if min_time > 0:
            timing_spread = (max_time - min_time) / min_time
            assert timing_spread < 0.5, f"Error timing spread too high: {timing_spread:.1%}"
```

## 🛡️ **Security Test Categories**

### **1. Cryptographic Security**
- Token randomness and entropy validation
- Hash avalanche effect testing
- Salt uniqueness verification
- Constant-time comparison validation
- PBKDF2 iteration count security
- Context binding effectiveness

### **2. Authentication Security**
- Bypass attempt prevention
- Credential stuffing protection
- Brute force resistance
- Token reuse prevention
- Session context validation
- Multi-factor authentication flows

### **3. Session Security**
- Context binding enforcement
- Session fixation prevention
- Concurrent session limits
- Proper session invalidation
- Cross-device session management
- Session hijacking resistance

### **4. Rate Limiting Security**
- Distributed attack prevention
- Escalation effectiveness
- Independence between limit types
- Performance under attack load
- Header manipulation bypass prevention
- Service restart persistence

### **5. Infrastructure Security**
- Reverse proxy header validation
- User-Agent manipulation resistance
- Combined header attack prevention
- Load balancer security integration
- CDN security configuration

### **6. Race Condition Security**
- Concurrent operation safety
- Database consistency under load
- Session invalidation races
- Rate limit consistency
- Token generation uniqueness

### **7. Information Leakage Prevention**
- Error message consistency
- Timing attack resistance
- Response structure uniformity
- Sensitive data exposure prevention
- Debug information filtering

### **8. Input Security**
- Injection attack prevention
- Input sanitization validation
- Length limit enforcement
- Malicious payload handling
- Unicode normalization attacks
- Encoding bypass attempts

## 🚫 **Anti-Patterns to Avoid**

### **❌ Don't Test Configuration Values**
```python
# BAD - Tests hardcoded values, not security
def test_rate_limit_config():
    assert service.default_limits[MAGIC_LINK]["email"]["limit"] == 3
```

### **❌ Don't Mock Security Functions**
```python
# BAD - Mocks away the security being tested
with patch.object(service, '_verify_token', return_value=True):
    result = service.authenticate(user)
```

### **❌ Don't Test Object Property Assignment**
```python
# BAD - Tests ORM functionality, not security
def test_user_creation():
    user = User(email="test@example.com")
    assert user.email == "test@example.com"
```

### **❌ Don't Test Only Single-Threaded Scenarios**
```python
# BAD - Misses race conditions and concurrent attacks
def test_rate_limiting():
    service.record_attempt("email", "test@example.com")
    result = service.record_attempt("email", "test@example.com") 
    assert not result.allowed

# GOOD - Tests concurrent access patterns
def test_concurrent_rate_limiting():
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(service.record_attempt, "email", "test@example.com") 
                  for _ in range(10)]
        results = [f.result() for f in futures]
    # Validate results show proper race condition handling
```

### **❌ Don't Ignore Service Restart Scenarios**
```python
# BAD - Tests only in-memory state
def test_rate_limit_persistence():
    service.exhaust_rate_limit("test@example.com")
    assert not service.check_rate_limit("test@example.com").allowed

# GOOD - Tests across service lifecycle
def test_rate_limit_survives_restart():
    service1.exhaust_rate_limit("test@example.com")
    service2 = create_new_service_instance()  # Simulates restart
    assert not service2.check_rate_limit("test@example.com").allowed
```

### **❌ Don't Use Inadequate Sample Sizes for Statistical Tests**
```python
# BAD - Too few samples for timing attack testing
def test_timing_attack_resistance():
    times1 = [measure_operation() for _ in range(5)]  # Too small
    times2 = [measure_operation() for _ in range(5)]
    assert abs(avg(times1) - avg(times2)) < 0.1

# GOOD - Statistically significant sample sizes
def test_timing_attack_resistance():
    times1 = [measure_operation() for _ in range(50)]  # Large enough for stats
    times2 = [measure_operation() for _ in range(50)]
    # Perform proper statistical analysis (t-test, etc.)
```

### **❌ Don't Mock Infrastructure Security Components**
```python
# BAD - Mocks away the security being tested
with patch('request.headers.get') as mock_header:
    mock_header.return_value = "10.0.0.1"
    result = security_check(request)

# GOOD - Tests with real header processing
def test_header_spoofing_prevention():
    # Use real TestClient with actual headers
    response = client.post("/api/endpoint", 
                          headers={"X-Forwarded-For": "spoofed_ip"})
    # Verify spoofing is detected and blocked
```

## ✅ **Test Structure Guidelines**

### **Use Descriptive Test Names**
```python
def test_magic_link_prevents_replay_attacks()
def test_session_invalidation_on_suspicious_activity()
def test_rate_limiting_blocks_credential_stuffing()
```

### **Include Threat Context in Docstrings**
```python
def test_token_verification_timing_attack_resistance():
    """
    Verify that token verification takes constant time regardless of input.
    
    Attack scenario: Attacker measures response times to determine if
    tokens are close to valid values, enabling faster brute force attacks.
    """
```

### **Test Edge Cases and Boundary Conditions**
```python
def test_rate_limit_boundary_conditions():
    """Test behavior exactly at rate limit boundaries."""
    # Test at limit - 1 (should work)
    # Test at limit (should work but hit limit)  
    # Test at limit + 1 (should be blocked)
```

## 🎯 **Coverage Goals**

Aim for **100% code coverage** but focus on **security scenario coverage**:

1. **Every authentication path** should be tested against bypass attempts
2. **Every rate limit** should be tested against distributed attacks  
3. **Every token operation** should be tested against cryptographic attacks
4. **Every session operation** should be tested against hijacking attempts
5. **Every input** should be tested against injection attacks

## 📋 **Test Implementation Checklist**

For each test file, ensure you have:

- [ ] **Real attack simulation tests** with realistic attack volumes (25+ attempts, multiple IPs)
- [ ] **Database integration tests** that verify security state persistence
- [ ] **Boundary condition tests** for all limits and thresholds (at limit-1, at limit, at limit+1)
- [ ] **Escalation behavior validation** for repeat offenders
- [ ] **Cross-attack-vector independence tests** (email vs IP vs user rate limits)
- [ ] **Window expiration and reset tests** for time-based security
- [ ] **Configuration logic validation** (are security settings logical?)
- [ ] **Cryptographic property validation** for all token operations
- [ ] **Timing attack resistance tests** for all authentication operations
- [ ] **Input validation tests** with malicious payloads
- [ ] **Concurrent operation tests** for race conditions
- [ ] **Error handling tests** that don't leak information

## 🎯 **Success Indicators**

Your tests are security-focused when:
- Test names describe attack scenarios, not just functionality
- Tests use realistic attack volumes and patterns
- You test with real databases, not mocks
- Tests verify that attacks actually fail, not just that code runs
- You validate mathematical security properties (entropy, escalation)
- **You test concurrent operations for race conditions**
- **You validate security state persists across service restarts**
- **You use statistical analysis for timing attack resistance**
- **You test infrastructure-level attack vectors (header spoofing, etc.)**
- **You verify error conditions don't leak sensitive information**
- **You simulate complex, multi-vector attacks**
- **You test boundary conditions and edge cases systematically**

Remember Claude: Your tests should make an attacker's job harder, not just verify that the code compiles and runs. Think like a sophisticated adversary who combines multiple attack techniques simultaneously.