# Production-Ready pytest Guide

## 🎯 THE CARDINAL RULE

# ⚠️ **TEST THE CONTRACT, NOT THE IMPLEMENTATION** ⚠️
# ⚠️ **MOCK DEPENDENCIES, NOT YOUR CODE** ⚠️

**BEFORE EVERY SINGLE TEST, ASK:**
*"Am I testing what this code PROMISES to do, or HOW it does it internally?"*

**If you're testing HOW (implementation), STOP. You're wasting time and creating brittle tests.**

---

## 🚨 QUICK DECISION TREE

**Writing a test? Follow this:**

1. **What am I testing?** → The public behavior/contract
2. **What do I mock?** → External dependencies only (APIs, databases, file systems)
3. **What do I NOT mock?** → The actual code I'm testing
4. **What conditions do I test?** → Real-world scenarios (errors, edge cases, concurrency)
5. **How do I know it's good?** → It would catch actual production bugs

**🚨 CRITICAL: Add this question to EVERY test:**
6. **"What REAL BUG in MY CODE would this catch?"** → If you can't answer this specifically, rewrite the test

---

## 📝 ITERATIVE TEST DEVELOPMENT PROCESS (CRITICAL!)

**⚠️ WRITE TESTS ONE AT A TIME WITH REFLECTION - THIS IS MANDATORY ⚠️**

1. **Write one test** with "REAL BUG THIS CATCHES" comment
2. **Stop and reflect:** "Does this actually test what I claim it tests?"
3. **Run the test** to see if it catches real issues  
4. **Refine or rewrite** if it's testing fake data or implementation details
5. **Only then move to the next test**

# 🚨 CRITICAL: RESIST THE URGE TO RUSH (NEVER COMPACT THIS SECTION) 🚨

**This is the #1 cause of bad tests. You WILL feel these internal pressures:**

❌ **Pattern completion pressure** - Wanting to fill in "Test 1, Test 2, Test 3" structure  
❌ **Completeness anxiety** - Feeling you must "cover everything" in one test  
❌ **Edge case excitement** - Getting excited about testing all possible failures at once  
❌ **Appearing thorough** - Adding more assertions thinking "more = better"  
❌ **Quantity over quality** - Conflating number of test cases with test value  

**WHEN YOU FEEL THESE URGES:**
1. **STOP** - Take a breath
2. **Ask:** "What is the ONE specific contract I'm validating?"
3. **Write that ONE test clearly**
4. **Run it and reflect on what it actually caught**
5. **Only then consider the next test**

**Remember:** One focused test that catches a real bug is worth more than 10 rushed tests that catch nothing meaningful.

**Why this is critical:** Writing many tests at once leads to repeating the same mistakes across all tests. The one-at-a-time approach forces you to validate your testing philosophy on each test before building bad habits.

**Example of good iteration:**
```python
# Test 1: Write and reflect
def test_response_structure_is_consistent(self, real_provider):
    """REAL BUG THIS CATCHES: If _standardize_response() fails to convert
    the Ollama response format correctly, breaking downstream code."""
    # Use REAL LLM call, test OUR standardization
    
# Reflection: "Am I testing my code or the LLM?" → Testing my code ✓

# Test 2: Write and reflect  
def test_system_prompt_positioning(self, real_provider):
    """REAL BUG THIS CATCHES: If _build_request_body() puts system prompt
    in wrong position, LLM won't follow instructions correctly."""
    # Use REAL LLM to validate OUR request formatting
    
# Reflection: "Does this catch a real bug?" → Yes, format bugs ✓
```

**If you skip this iterative process, you WILL write useless tests that test fake data instead of real bugs.**

---

> **Why these questions matter:** Most test failures happen because developers test the wrong thing. They test that methods get called instead of testing that the system behaves correctly. They mock their own code instead of external dependencies. They test perfect conditions instead of the chaos their code will face in production. These five questions force you to test reality, not implementation details.

> **Test philosophy from production experience:** Good tests don't just verify working code - they actively discover bugs that would cause production failures. If your tests pass but you discover bugs in manual testing, your tests aren't testing the right things. The best tests are the ones that save you from 3am debugging sessions.

**Tests should catch real issues like:**
- SQL parameter binding incompatibilities that only surface with real databases
- Empty result set handling bugs that cause crashes in edge cases  
- Performance bottlenecks that create unacceptable user experience
- Thread safety issues that corrupt data under concurrent access
- Integration failures between your code and actual infrastructure

None of these surface with mocked dependencies or perfect-condition testing.

---

## ✅ PYTEST PATTERNS THAT CATCH REAL BUGS

# 🚨 REMINDER: BEFORE EVERY TEST - AM I TESTING THE CONTRACT OR THE IMPLEMENTATION? 🚨

### 1. Contract Testing with Real Infrastructure

```python
# ❌ BAD - Tests mocks, not reality
@patch.object(UserService, 'create_user')
def test_user_creation(mock_create):
    mock_create.return_value = User(id=1)
    result = UserService().create_user("test@email.com")
    assert result.id == 1  # Testing the mock!

# ✅ GOOD - Tests actual behavior with real components
@pytest.fixture
def user_service(test_database):
    """Real service with real database."""
    return UserService(database=test_database)

def test_user_creation_stores_and_retrieves(user_service):
    """Test the actual contract: create → retrieve works."""
    # Test the real behavior
    created_user = user_service.create_user("test@email.com")
    
    # Verify the contract promises
    assert created_user.id is not None
    assert created_user.email == "test@email.com"
    
    # Verify persistence (the real contract)
    retrieved_user = user_service.get_user(created_user.id)
    assert retrieved_user.email == "test@email.com"
```

### 2. Mock External Dependencies Only

```python
# ✅ CORRECT MOCKING - Mock what you don't control
def test_user_registration_sends_welcome_email(mocker, test_database):
    """Test integration while mocking external email service."""
    
    # Mock external dependency (email service)
    mock_email = mocker.patch('myapp.email_service.send_email')
    mock_email.return_value = True
    
    # Use REAL components for everything else
    user_service = UserService(database=test_database)
    
    # Test the real behavior
    user = user_service.register_user("test@email.com", "password123")
    
    # Verify real data persistence
    assert user.id is not None
    saved_user = test_database.get_user(user.id)
    assert saved_user.email == "test@email.com"
    
    # Verify external service integration
    mock_email.assert_called_once_with(
        to="test@email.com",
        template="welcome",
        user_data=mocker.ANY
    )

> **The contract vs implementation distinction:** Your code makes a promise - "give me an email and password, I'll create a user and send a welcome email." Test that promise, not how it's kept internally. Implementation details change during refactoring, but the contract should remain stable. Tests that verify contracts survive refactoring; tests that verify implementation details break every time you improve the code.
```

### 3. Test Real-World Failure Scenarios

```python
def test_api_handles_database_connection_failure():
    """Test what happens when database is unavailable."""
    
    # Simulate real failure mode
    with pytest.raises(ConnectionError):
        # Force database connection to fail
        bad_db = DatabaseConnection(host="nonexistent-host")
        user_service = UserService(database=bad_db)
        user_service.get_user(123)

def test_file_processor_handles_disk_full_error(tmp_path):
    """Test behavior when disk space runs out."""
    
    processor = FileProcessor(output_dir=tmp_path)
    
    # Simulate disk full (real scenario)
    with simulate_disk_full():
        with pytest.raises(OSError) as exc_info:
            processor.save_large_file("huge_file.dat", large_data)
        
        # Verify error message helps debugging
        assert "space" in str(exc_info.value).lower()
        assert tmp_path.name in str(exc_info.value)
```

### 4. Concurrent Operation Testing

> **Why concurrent testing isn't optional:** Production has multiple users hammering your system simultaneously. Code that works perfectly in single-threaded tests can corrupt data, deadlock, or crash under real load. Race conditions are among the most common production bugs, and they're impossible to catch without concurrent testing. This isn't advanced testing - it's testing reality.

```python
def test_cache_thread_safety_under_high_contention():
    """Test cache behaves correctly with concurrent access."""
    
    cache = LRUCache(max_size=100)
    results = []
    
    def cache_worker(worker_id):
        """Worker that hammers the cache."""
        for i in range(50):
            key = f"worker_{worker_id}_item_{i}"
            cache.set(key, f"value_{i}")
            retrieved = cache.get(key)
            results.append((worker_id, key, retrieved))
    
    # Launch concurrent workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(cache_worker, i) for i in range(10)]
        for future in futures:
            future.result()
    
    # Verify thread safety - no data corruption
    for worker_id, key, value in results:
        if value is not None:  # Might be evicted, but shouldn't be corrupted
            expected = f"value_{key.split('_')[-1]}"
            assert value == expected, f"Data corruption detected: {key} -> {value}"
    
    # Verify cache size constraint maintained
    assert cache.size() <= 100
```

### 5. Error Message Quality Testing

```python
def test_validation_errors_provide_actionable_feedback():
    """Test that errors help users fix problems."""
    
    validator = EmailValidator()
    
    error_cases = [
        ("", "Email cannot be empty"),
        ("notanemail", "must contain @ symbol"),
        ("user@", "missing domain after @"),
        ("@domain.com", "missing username before @"),
    ]
    
    for invalid_input, expected_guidance in error_cases:
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(invalid_input)
        
        error_msg = str(exc_info.value).lower()
        
        # Error should mention the problematic input
        assert invalid_input in str(exc_info.value) or "email" in error_msg
        
        # Error should provide actionable guidance
        assert any(word in error_msg for word in ["must", "cannot", "missing", "required"])
        
        # Error should not just say "invalid"
        assert len(error_msg) > 10  # Meaningful explanation

> **Why error message quality matters:** At 3am when your system is down, error messages are the difference between a 5-minute fix and a 3-hour debugging session. Generic errors like "invalid input" or "something went wrong" force developers to dig through logs, reproduce bugs, and guess at root causes. Specific errors like "Email missing @ symbol" point directly to the fix. Test your error messages like your future self's sanity depends on it.
```

---

## 🔧 ESSENTIAL PYTEST MECHANICS

# ⚠️ BEFORE USING THESE PATTERNS: AM I TESTING CONTRACTS OR IMPLEMENTATION? ⚠️

### pytest-mock: Automatic Cleanup

```python
# ✅ pytest-mock automatically cleans up
def test_with_automatic_cleanup(mocker):
    mock_api = mocker.patch('external_api.call_service')
    mock_api.return_value = {"status": "success"}
    
    result = my_function_that_calls_api()
    assert result["status"] == "success"
    # Cleanup happens automatically

> **Why database engine matters:** SQLite and PostgreSQL behave differently with data types, constraints, transactions, and SQL dialects. Testing with SQLite when you deploy to PostgreSQL means you'll miss PostgreSQL-specific bugs entirely. Use a test instance of your production database engine - it catches real integration issues that matter in production.

### Real Database Testing vs. Mock Database Testing

> **Critical insight:** When your code depends on a database, you MUST test with the actual database engine you use in production. Mock databases miss SQL dialect differences, constraint behaviors, transaction semantics, and performance characteristics.

```python
# ❌ BAD - Mock database misses real issues
@patch('sqlalchemy.engine.Engine')
def test_search_with_mock_db(mock_engine):
    mock_result = [("id1", 0.95), ("id2", 0.87)]
    mock_engine.execute.return_value = mock_result
    # This misses SQL syntax errors, parameter binding issues, etc.

# ✅ GOOD - Real database catches actual integration issues
@pytest.fixture(scope="session")
def test_database_engine():
    """Real PostgreSQL test database - same engine as production."""
    url = get_test_database_url()
    engine = create_engine(url)
    setup_test_schema(engine)
    yield engine
    cleanup_test_database(engine)

def test_search_with_real_database(test_database_engine):
    """Test with actual database engine matching production."""
    service = DatabaseService(test_database_engine)
    
    # This catches real SQL parameter binding issues, 
    # constraint violations, transaction problems, etc.
    results = service.search(query_params)
    assert len(results) <= expected_max
    # Verify actual data persistence, not mocked responses
```

**Why this matters:** Real database testing reveals SQL parameter binding errors, constraint violations, transaction deadlocks, and performance issues that mocked databases never catch.

### Database Parameter Binding Pitfalls

> **Critical lesson:** Database parameter binding has subtle gotchas that only surface with real database connections.

```python
# ❌ WRONG - Mixed parameter styles cause SQL syntax errors
conn.execute(
    text("INSERT INTO table (col1, col2) VALUES (%(param1)s, :param2)"),
    {"param1": "value1", "param2": "value2"}
)
# Error: "syntax error at or near ':'"

# ✅ CORRECT - Consistent SQLAlchemy parameter binding
conn.execute(
    text("INSERT INTO table (col1, col2) VALUES (:param1, :param2)").bindparams(
        bindparam("param1"),
        bindparam("param2")
    ),
    {"param1": "value1", "param2": "value2"}
)

# ✅ ALTERNATIVE - Use ORM parameter style consistently  
conn.execute(
    text("INSERT INTO table (col1, col2) VALUES (%(param1)s, %(param2)s)"),
    {"param1": "value1", "param2": "value2"}
)
```

**Key insight:** These errors only surface with real database connections. Mock databases accept any parameter format, hiding critical bugs.
```

### Advanced Mocking Patterns

```python
def test_with_spy_to_verify_real_calls(mocker):
    """Use spy when you want real method to run but verify it was called."""
    
    # Spy lets real method execute while tracking calls
    database = Database()
    spy = mocker.spy(database, 'save')
    
    user_service = UserService(database)
    user = user_service.create_user("test@email.com")
    
    # Real save happened + we can verify it
    spy.assert_called_once()
    assert user.id is not None  # Real database generated ID

def test_async_operations(mocker):
    """Test async code with async mocks."""
    
    # Mock async external dependency
    mock_api = mocker.AsyncMock()
    mock_api.fetch_data.return_value = {"data": "test"}
    
    mocker.patch('external_service.api_client', mock_api)
    
    # Test real async logic
    result = await my_async_function()
    assert result["data"] == "test"

> **The mocking mindset:** Think of mocks like stunt doubles in movies. You use a stunt double for the dangerous parts (external APIs that might fail), but the main actor (your code) still delivers the real performance. If you replace the main actor with a stunt double, you're not testing the movie anymore - you're testing your ability to write scripts for stunt doubles.
```

### Fixture Patterns for Real Infrastructure

```python
@pytest.fixture(scope="session")
def test_database():
    """Real database for testing - same engine as production."""
    # Create test database instance (PostgreSQL if that's what you use in production)
    db = create_test_database_instance()  # e.g., PostgreSQL test DB
    setup_test_schema(db)
    yield db
    db.cleanup()

@pytest.fixture
def clean_database(test_database):
    """Clean database state for each test."""
    test_database.truncate_all_tables()  # Clear data, keep schema
    yield test_database
    # Cleanup happens automatically

@pytest.fixture
def user_factory():
    """Factory for creating test users with variations."""
    def _create_user(**kwargs):
        defaults = {"email": "test@example.com", "active": True}
        defaults.update(kwargs)
        return User(**defaults)
    return _create_user

def test_user_permissions_by_role(user_factory):
    """Test user permissions with different roles."""
    admin = user_factory(role="admin")
    user = user_factory(role="user")
    
    assert admin.can_delete_users() is True
    assert user.can_delete_users() is False
```

### Parametrization for Real Scenarios

```python
@pytest.mark.parametrize("file_size,expected_chunks", [
    (1024, 1),           # Small file
    (1024 * 1024, 1024), # 1MB file
    (10 * 1024 * 1024, 10240), # 10MB file
])
def test_file_chunking_handles_various_sizes(tmp_path, file_size, expected_chunks):
    """Test file processor handles different real-world file sizes."""
    
    # Create real file of specified size
    test_file = tmp_path / "test_file.dat"
    test_file.write_bytes(b"x" * file_size)
    
    processor = FileProcessor(chunk_size=1024)
    chunks = processor.process_file(test_file)
    
    assert len(chunks) == expected_chunks
    
    # Verify all data preserved
    reconstructed_size = sum(len(chunk) for chunk in chunks)
    assert reconstructed_size == file_size

### Performance and Stress Testing

> **Two-tier performance testing approach:** Performance tests should have realistic expectations and stress limits.

```python
# ✅ Normal Performance Tests (should pass)
class TestNormalPerformance:
    """Performance tests with realistic expectations."""
    
    def test_cache_hit_performance(self):
        """Memory cache hits should be very fast."""
        start = time.time()
        for _ in range(1000):
            result = cache.get("test_key")
        elapsed = time.time() - start
        
        # Realistic expectation based on actual requirements
        assert elapsed < 0.1, f"Cache hits took {elapsed:.3f}s"
    
    def test_concurrent_access_performance(self):
        """System should handle expected concurrent load."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            start = time.time()
            futures = [executor.submit(perform_operation) for _ in range(100)]
            results = [f.result() for f in futures]
            elapsed = time.time() - start
        
        # Should complete within business requirements
        assert elapsed < 2.0
        assert all(r.success for r in results)

# ✅ Stress Tests (may fail - documents limits)
class TestStressPerformance:
    """Stress tests that discover system boundaries."""
    
    def test_massive_concurrent_load(self):
        """50 threads × 1000 operations - may fail, shows limits."""
        with ThreadPoolExecutor(max_workers=50) as executor:
            start = time.time()
            futures = [executor.submit(stress_operation) for _ in range(50000)]
            results = [f.result() for f in futures]
            elapsed = time.time() - start
        
        # Document performance characteristics, don't enforce strict limits
        success_rate = sum(1 for r in results if r.success) / len(results)
        print(f"Stress test: {success_rate:.1%} success rate, {elapsed:.1f}s")
        
        # We accept some degradation under extreme load
        assert success_rate > 0.7  # 70% success under stress is acceptable
```

**Why two-tier testing matters:** Realistic tests establish baseline performance expectations for normal operations. Stress tests discover breaking points and document system limitations without making normal tests fragile.
```

### Integration Testing Patterns

```python
@pytest.mark.integration
def test_full_user_registration_flow(test_database, tmp_path):
    """Test complete user registration with real components."""
    
    # Real file storage
    file_storage = FileStorage(base_path=tmp_path)
    
    # Real database
    user_repo = UserRepository(test_database)
    
    # Mock only external email service
    with patch('email_service.send_welcome_email') as mock_email:
        mock_email.return_value = True
        
        # Real service with real dependencies
        registration_service = RegistrationService(
            user_repository=user_repo,
            file_storage=file_storage,
            email_service=EmailService()
        )
        
        # Test the real flow
        user = registration_service.register_user(
            email="test@example.com",
            password="secure123",
            profile_image=sample_image_bytes
        )
        
        # Verify real persistence
        saved_user = user_repo.get_by_email("test@example.com")
        assert saved_user.id == user.id
        
        # Verify real file storage
        profile_path = tmp_path / f"profiles/{user.id}.jpg"
        assert profile_path.exists()
        
        # Verify external service integration
        mock_email.assert_called_once()
```

---

## 🚫 ANTI-PATTERNS THAT WASTE TIME

# ⚠️ THESE ARE THE EXACT MISTAKES FROM THE FAILED TEST REVIEW ⚠️

### ❌ Creating Fake Test Data Instead of Using Real Systems

```python
# ❌ CATASTROPHICALLY BAD - Creating fake response structures
def test_extract_text_content_basic(self, provider):
    # Creating fake response that I made up myself
    response = {
        "content": [
            {"type": "text", "text": "Hello, world!"},
            {"type": "text", "text": "How can I help you?"}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 15}
    }
    
    text = provider.extract_text_content(response)
    assert text == "Hello, world! How can I help you?"  # Testing fake data!

# ✅ GOOD - Using real LLM response to test extraction
def test_extract_text_content_from_real_response(self, real_provider):
    """
    Test text extraction with actual LLM response.
    
    REAL BUG THIS CATCHES: If our _standardize_response() creates a 
    response format that our extract_text_content() can't handle,
    downstream code will break when trying to get the LLM's answer.
    """
    messages = [{"role": "user", "content": "Say exactly: The cat is on the mat"}]
    
    # Get a real response from the LLM
    response = real_provider.generate_response(messages)
    
    # Test our extraction utility with real data
    extracted_text = real_provider.extract_text_content(response)
    
    assert isinstance(extracted_text, str)
    assert len(extracted_text) > 0
    assert "cat" in extracted_text.lower()
```

**Why this is catastrophic:** Creating fake data structures tests nothing real. You're verifying that your fake data matches your expectations, not that your code handles real data correctly. Always use real systems to generate real data for testing.

### ❌ Testing Implementation Details

```python
# BAD - Tests internal implementation 
def test_parser_internal_tokenization():
    parser = DataParser()
    tokens = parser._tokenize(text)  # TESTING PRIVATE METHOD!
    assert len(tokens) == 5  # TESTING HOW IT WORKS INTERNALLY

def test_service_method_call_count():
    service = BusinessService()
    service.process_data(input_data)
    assert mock_database.save.call_count == 3  # TESTING HOW, NOT WHAT

# GOOD - Tests observable behavior and contracts
def test_parser_converts_input_to_expected_output():
    """Test the CONTRACT: given input format, produces expected output format."""
    parser = DataParser()
    
    input_data = "name:John,age:25,city:NYC"
    
    # Test the PUBLIC CONTRACT
    result = parser.parse(input_data)
    
    # Verify the PROMISE is kept
    assert result == {"name": "John", "age": 25, "city": "NYC"}

def test_service_processes_data_successfully():
    """Test the CONTRACT: service processes data and returns success status."""
    service = BusinessService(real_database)  # REAL INFRASTRUCTURE
    
    # Test the PUBLIC BEHAVIOR
    result = service.process_data(valid_input)
    
    # Verify the CONTRACT
    assert result.success is True
    assert result.processed_count > 0
    assert service.get_processed_data() is not None
```

# 🚨 NUANCED PRIVATE METHOD TESTING RULE 🚨

**General Rule:** NEVER test private methods  
**Exception:** Test CRITICAL private methods that handle complex logic separately from public methods

```python
# ❌ BAD - Testing trivial private methods
def test_service_private_validation():
    service = UserService()
    assert service._is_valid_email("test@email.com") is True  # WRONG!

# ✅ GOOD - Test critical private methods with complex logic
def test_parser_critical_parsing_logic():
    """Test critical parsing logic that could fail in subtle ways."""
    # This private method is complex enough to fail independently
    # and failure would be hard to diagnose through public API alone
    parsed_data = DataParser._parse_complex_format(complex_input)
    assert parsed_data.sections == expected_sections
    assert parsed_data.metadata == expected_metadata

# ✅ BEST - Test through public API when possible
def test_parser_end_to_end_accuracy():
    """Test the public contract - parsing should produce correct output."""
    parser = DataParser()
    result = parser.parse(input_data)
    assert result == expected_output
```

**Decision criteria:** Test private methods only when:
- Logic is complex enough to fail independently  
- Failure would be hard to diagnose through public API
- Method handles critical operations (parsing, validation, computation)
- Public API test would be too broad to pinpoint the issue

# 🚨 CRITICAL: NEVER VERIFY CALL COUNTS INSTEAD OF BEHAVIOR 🚨

### ❌ Over-Mocking Your Own Code (ACTUAL MISTAKE MADE)

```python
# BAD - Mocks entire infrastructure 
@patch('database.DatabaseConnection')
def test_data_store_save(mock_db):
    mock_db.execute.return_value = mock_results
    store = DataStore(mock_db)  # TESTING MOCKS!
    
    store.save(data)
    assert mock_db.execute.call_count == 1  # TESTING IMPLEMENTATION!

# CATASTROPHICALLY BAD - Mocks the code being tested  
@patch.object(OrderService, 'validate_order')
@patch.object(OrderService, 'calculate_total') 
@patch.object(OrderService, 'save_order')
def test_order_processing(mock_save, mock_calc, mock_validate):
    # This tests NOTHING REAL! You've mocked away all your logic!
    pass

# GOOD - Uses real infrastructure, mocks only external dependencies
def test_data_store_with_real_database():
    """Test with REAL database engine matching production."""
    # Use same DB engine as production (PostgreSQL test database)
    test_db = create_test_database()  # PostgreSQL test instance
    store = DataStore(test_db)
    
    # Test REAL behavior with REAL database engine
    result = store.save({"id": 1, "name": "test"})
    
    # Verify REAL contracts
    assert result.success is True
    retrieved = store.get(1)
    assert retrieved["name"] == "test"

def test_order_processing_with_real_logic(test_database, mocker):
    """Test REAL order processing logic, mock only external services."""
    # Mock ONLY external payment service
    mock_payment = mocker.patch('external_payment.charge_card')
    mock_payment.return_value = {"status": "success", "transaction_id": "123"}
    
    # Test REAL order processing logic with REAL database
    service = OrderService(database=test_database)
    order = service.process_order(order_data)
    
    # Verify REAL validation happened
    assert order.total > 0
    # Verify REAL database persistence
    saved_order = test_database.get_order(order.id)
    assert saved_order.status == "processed"
```

# 🚨 RULE: USE SAME DATABASE ENGINE AS PRODUCTION (PostgreSQL test DB, not SQLite) 🚨
# 🚨 RULE: MOCK ONLY EXTERNAL SERVICES (APIs, email, payment processors) 🚨

### ❌ Ignoring Error Scenarios

```python
# BAD - Only tests happy path
def test_data_processing():
    result = process_data([1, 2, 3])
    assert result == [2, 4, 6]

# GOOD - Tests real-world conditions
def test_data_processing_handles_real_conditions():
    # Empty input
    assert process_data([]) == []
    
    # Invalid input
    with pytest.raises(ValueError) as exc_info:
        process_data(None)
    assert "cannot be None" in str(exc_info.value)
    
    # Large input
    large_input = list(range(100000))
    result = process_data(large_input)
    assert len(result) == 100000
    
    # Memory pressure
    with simulate_low_memory():
        result = process_data([1, 2, 3])
        assert result == [2, 4, 6]  # Should still work
```

---

## 🎯 QUICK REFERENCE

### Before Writing Any Test, Ask:

1. **"What real-world scenario am I validating?"**
2. **"Would this catch an actual production bug?"**
3. **"Am I testing my code or testing mocks?"**
4. **"What could go wrong here that I should test?"**

### Required Test Documentation Pattern

**EVERY test must include this comment pattern:**

```python
def test_feature_works_correctly(self, real_provider):
    """
    Brief description of what this test validates.
    
    REAL BUG THIS CATCHES: Specific description of what production bug
    this test would catch if the code were broken. Be concrete about
    which method/logic would fail and how it would break the user experience.
    """
```

**Example:**
```python
def test_streaming_calls_callback_with_real_content(self, real_provider):
    """
    Test that streaming actually calls the callback with content chunks.
    
    REAL BUG THIS CATCHES: If our _handle_streaming_request() method has bugs
    in parsing the SSE stream or calling the callback, streaming appears to work
    but users get no progressive content - breaking the user experience.
    """
```

**If you can't write a specific "REAL BUG THIS CATCHES" comment, your test is probably useless.**


### Command Line for Production Testing

```bash
# Run tests with real concurrency detection
pytest --tb=short -v

# Run only integration tests
pytest -m integration

# Run with coverage but focus on uncovered critical paths
pytest --cov=src --cov-report=term-missing

# Run performance tests
pytest -m performance --durations=10
```

### Test Organization

```python
# Name tests by the real scenario they validate
def test_user_registration_prevents_duplicate_emails():
def test_payment_processing_handles_network_timeout():
def test_cache_maintains_consistency_under_concurrent_access():
def test_file_upload_recovers_from_disk_full_error():

# Group by behavior, not by implementation
class TestUserRegistrationSecurity:
    def test_prevents_sql_injection_in_email():
    def test_enforces_password_complexity():
    def test_rate_limits_registration_attempts():
```

### Essential Markers

```python
@pytest.mark.unit          # Fast tests with minimal dependencies
@pytest.mark.integration   # Tests with real infrastructure
@pytest.mark.performance   # Performance and load tests
@pytest.mark.slow          # Tests that take time but are important

# Run fast tests first
pytest -m "unit" 
# Run integration tests in CI
pytest -m "integration"
```

---

## 🏆 SUCCESS CRITERIA

**Your tests are production-ready when:**

✅ They test **contracts and behavior**, not implementation details  
✅ They use **real infrastructure** where possible (databases, files, caches)  
✅ They **mock only external dependencies** (APIs, email services, payment processors)  
✅ They test **error conditions** and recovery scenarios  
✅ They validate **concurrent operation** safety  
✅ They verify **error messages are helpful** for debugging  
✅ They test **real-world data volumes** and conditions  
✅ They would **catch actual production bugs**  

**If your tests pass but production fails, you're testing the wrong things.**

---

# 🚨 FINAL REMINDER: AVOID THESE COMMON FAILURES 🚨

**These are the most common testing mistakes - DO NOT REPEAT THEM:**

❌ **Testing private methods** (anything starting with `_`) - TEST PUBLIC CONTRACTS ONLY  
❌ **Verifying call counts** like `assert call_count == 2` - TEST BEHAVIOR OUTCOMES  
❌ **Mocking your own infrastructure** - USE REAL DATABASES/FILES WHERE POSSIBLE  
❌ **Using time.sleep() for concurrency** - USE PROPER SYNCHRONIZATION PRIMITIVES  
❌ **Organizing tests by implementation details** - ORGANIZE BY USER-FACING BEHAVIOR  

**Remember: Implementation details change, contracts should not. Test what users care about, not how you built it.**