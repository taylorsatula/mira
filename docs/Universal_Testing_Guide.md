# Universal Testing Methodology Guide for Claude

## 🎯 **Core Testing Philosophy**

Claude, when writing tests for ANY system, your primary goal is **real-world validation, not coverage theater**. Every test should answer: "Does this actually work under the conditions it will face in production?"

**Think like production, test like production, validate like production.**

## 🚨 **Universal Testing Principles**

### **1. Test Real-World Conditions, Not Ideal Scenarios**

**Bad**: Testing only happy paths and perfect conditions
**Good**: Testing the messy reality your code will face

```python
# ❌ TESTING THEATER - Tests perfect conditions
def test_data_processing():
    result = process_data([1, 2, 3, 4, 5])
    assert result == [2, 4, 6, 8, 10]

# ✅ REAL-WORLD TESTING - Tests actual conditions
def test_data_processing_real_conditions():
    # Empty data
    assert process_data([]) == []
    
    # Large datasets
    large_data = list(range(100000))
    result = process_data(large_data)
    assert len(result) == 100000
    
    # Mixed data types (if applicable)
    mixed_data = [1, 2.5, 3, None, 4]
    # Should either handle gracefully or fail explicitly
    
    # Memory pressure simulation
    with simulate_low_memory():
        result = process_data(moderate_data)
        assert result is not None
```

### **2. Test the Contract, Not the Implementation**

Tests should verify what a function/class promises to do, not how it does it internally. This ensures tests don't break during refactoring and makes them more maintainable.

```python
# ❌ IMPLEMENTATION TESTING - Tests internal details
def test_user_repository_uses_sql():
    repo = UserRepository()
    repo.create_user("test@example.com")
    assert "INSERT INTO users" in repo._last_query
    assert repo._connection_pool.active_connections == 1

# ✅ CONTRACT TESTING - Tests behavior promise
def test_user_repository_stores_and_retrieves_users():
    repo = UserRepository()
    
    # Contract: create_user returns user with ID
    user = repo.create_user("test@example.com")
    assert user.id is not None
    assert user.email == "test@example.com"
    
    # Contract: get_user retrieves the same user
    retrieved = repo.get_user(user.id)
    assert retrieved.email == "test@example.com"
    
    # Contract: get_user returns None for non-existent ID
    assert repo.get_user(99999) is None
```

### **3. Think Adversarially - What Can Go Wrong?**

For every function, ask:
- What if the input is malformed?
- What if dependencies fail?
- What if resources are exhausted?
- What if multiple users do this simultaneously?

```python
# ✅ ADVERSARIAL THINKING - API Testing
def test_api_endpoint_failure_modes():
    # Network failures
    with simulate_network_timeout():
        response = api_client.get("/data")
        assert response.status_code in [503, 504, 408]
    
    # Database connection failures
    with database_unavailable():
        response = api_client.get("/data")
        assert response.status_code == 503
        assert "temporarily unavailable" in response.json()["message"]
    
    # Resource exhaustion
    with high_cpu_load():
        response = api_client.get("/data")
        # Should either succeed or fail gracefully
        assert response.status_code != 500
```

### **3. Use Statistical Rigor for Performance and Reliability**

Don't just check "it works once" - validate statistical properties.

```python
# ✅ STATISTICAL VALIDATION - Performance Testing
def test_api_response_time_consistency():
    response_times = []
    
    # Large sample size for statistical significance
    for _ in range(100):
        start = time.perf_counter()
        response = api_client.get("/fast-endpoint")
        end = time.perf_counter()
        
        assert response.status_code == 200
        response_times.append(end - start)
    
    # Statistical analysis
    avg_time = sum(response_times) / len(response_times)
    std_dev = (sum((t - avg_time)**2 for t in response_times) / len(response_times))**0.5
    
    # Performance requirements
    assert avg_time < 0.1, f"Average response time too slow: {avg_time:.3f}s"
    
    # Consistency requirement (coefficient of variation)
    cv = std_dev / avg_time if avg_time > 0 else 0
    assert cv < 0.2, f"Response time too inconsistent: {cv:.1%} variation"
    
    # No outliers beyond 3 standard deviations
    outliers = [t for t in response_times if abs(t - avg_time) > 3 * std_dev]
    assert len(outliers) == 0, f"Performance outliers detected: {outliers}"
```

### **4. Test Actual Implementation, Not Mocks**

Mock external dependencies, but test your actual logic with real infrastructure.

```python
# ❌ OVER-MOCKING - Tests nothing real
@patch('database.query')
@patch('cache.get')
@patch('external_api.call')
def test_business_logic(mock_api, mock_cache, mock_db):
    mock_db.return_value = [{'id': 1}]
    mock_cache.return_value = None
    mock_api.return_value = {'status': 'ok'}
    
    result = business_function()
    assert result == 'expected'

# ✅ REAL IMPLEMENTATION - Tests actual behavior
def test_business_logic_with_real_infrastructure(test_db, test_cache):
    # Use real test database with real data
    test_db.insert('test_table', {'id': 1, 'data': 'test'})
    
    # Use real cache implementation
    test_cache.clear()
    
    # Mock only external dependencies
    with patch('external_api.call') as mock_api:
        mock_api.return_value = {'status': 'ok'}
        
        result = business_function()
        
        # Verify actual database interactions
        assert test_db.query_count() > 0
        # Verify actual cache behavior
        assert test_cache.size() > 0
```

### **5. Simulate Concurrent Operations**

Production has multiple users - test like it.

```python
# ✅ CONCURRENT TESTING - Database Operations
def test_concurrent_data_updates():
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    initial_value = 100
    update_count = 50
    
    def update_balance(amount):
        return database.update_balance(user_id=1, change=amount)
    
    # Launch concurrent updates
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(update_balance, 1) for _ in range(update_count)]
        results = [future.result() for future in as_completed(futures)]
    
    # Verify data consistency
    final_balance = database.get_balance(user_id=1)
    expected_balance = initial_value + update_count
    
    assert final_balance == expected_balance, \
        f"Race condition detected: expected {expected_balance}, got {final_balance}"
    
    # Verify all operations succeeded
    successful_updates = [r for r in results if r.success]
    assert len(successful_updates) == update_count, "Some updates failed unexpectedly"
```

### **6. Test Across Service Lifecycle**

Test persistence, restarts, and state transitions.

```python
# ✅ LIFECYCLE TESTING - Service Persistence
def test_cache_survives_restart():
    # Set up initial state
    cache_service.set("important_key", "critical_value", ttl=3600)
    assert cache_service.get("important_key") == "critical_value"
    
    # Simulate service restart
    cache_service.shutdown()
    new_cache_service = CacheService(same_config)
    
    # Verify persistence
    assert new_cache_service.get("important_key") == "critical_value"
    
    # Verify TTL is maintained
    time.sleep(2)
    assert new_cache_service.get("important_key") == "critical_value"
    
    # Verify expiration still works
    time.sleep(3600)
    assert new_cache_service.get("important_key") is None
```

### **7. Validate Boundary Conditions and Edge Cases**

Test the edges where bugs hide.

```python
# ✅ BOUNDARY TESTING - Data Processing
def test_data_processing_boundaries():
    processor = DataProcessor(max_items=1000)
    
    # Boundary conditions
    test_cases = [
        ([], "empty_input"),
        ([1], "single_item"),
        (list(range(999)), "max_minus_one"),
        (list(range(1000)), "exactly_at_max"),
        (list(range(1001)), "over_limit"),
        (list(range(10000)), "way_over_limit"),
    ]
    
    for data, description in test_cases:
        if len(data) <= 1000:
            # Should succeed
            result = processor.process(data)
            assert len(result) == len(data), f"Failed at {description}"
        else:
            # Should fail gracefully
            with pytest.raises(ValueError, match="exceeds maximum"):
                processor.process(data)
    
    # Edge case: exactly at memory limit
    with simulate_memory_limit(1024 * 1024):  # 1MB limit
        large_but_feasible = list(range(100))
        result = processor.process(large_but_feasible)
        assert result is not None
```

### **8. Test Error Conditions and Message Quality**

Don't just test that errors occur - test they provide helpful debugging information.

```python
# ✅ ERROR CONDITION AND MESSAGE TESTING - File Processing
def test_file_processor_error_handling():
    processor = FileProcessor()
    
    error_scenarios = [
        ("nonexistent_file.txt", FileNotFoundError, "file not found"),
        ("directory/", IsADirectoryError, "expected file, got directory"),
        ("empty_file.txt", ValueError, "file is empty"),
        ("corrupted_file.bin", DataCorruptionError, "file corrupted at byte"),
        ("huge_file.txt", MemoryError, "file too large"),
    ]
    
    for file_path, expected_error, expected_message_content in error_scenarios:
        setup_error_scenario(file_path, expected_error)
        
        with pytest.raises(expected_error) as exc_info:
            processor.process_file(file_path)
        
        # Verify error message quality for debugging
        error_message = str(exc_info.value).lower()
        assert expected_message_content in error_message, \
            f"Error message should help debug: {error_message}"
        assert file_path in str(exc_info.value), \
            "Error should include the problematic file path"
        
        # Verify system is still in good state after error
        assert processor.is_healthy()
        assert processor.can_process_valid_file()
    
    # Test partial failure with descriptive error
    with simulate_disk_full():
        result = processor.process_file("normal_file.txt")
        assert result.status == "partial_success"
        assert "disk space" in result.error_message
        assert "normal_file.txt" in result.error_message
        # Error should suggest recovery action
        assert any(keyword in result.error_message for keyword in 
                  ["free space", "try again", "contact admin"])


def test_validation_error_messages_are_actionable():
    """Test that validation errors tell users exactly what's wrong and how to fix it."""
    
    validator = EmailValidator()
    
    invalid_cases = [
        ("", "Email address cannot be empty"),
        ("notanemail", "Email address must contain @ symbol"),
        ("user@", "Email address missing domain after @"),
        ("@domain.com", "Email address missing username before @"),
        ("user@domain", "Email address domain must contain a dot"),
        ("user name@domain.com", "Email address cannot contain spaces"),
    ]
    
    for invalid_email, expected_guidance in invalid_cases:
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(invalid_email)
        
        error_message = str(exc_info.value)
        
        # Should contain the problematic input
        assert invalid_email in error_message or "provided email" in error_message
        
        # Should provide actionable guidance
        assert any(word in error_message.lower() for word in 
                  ["must", "should", "cannot", "required", "expected"])
        
        # Should not just say "invalid" - should explain why
        assert "invalid" not in error_message.lower() or len(error_message) > 20
```

### **9. Test Integration Points Thoroughly**

Every external dependency needs timeout and malformed response testing. These are the most common production failures.

```python
# ✅ INTEGRATION POINT TESTING - External API
def test_external_api_integration_failure_modes():
    api_client = ExternalAPIClient()
    
    # Test timeout scenarios
    with simulate_network_delay(seconds=10):
        with pytest.raises(TimeoutError) as exc_info:
            api_client.get_user_data(user_id=123)
        
        # Verify timeout is reasonable (not too short, not too long)
        assert "timeout" in str(exc_info.value).lower()
        assert api_client.timeout_seconds in [5, 10, 30]  # Reasonable values
    
    # Test malformed response scenarios
    malformed_responses = [
        ('{"incomplete": json', "Invalid JSON"),
        ('{"missing_required_field": "value"}', "Missing required field"),
        ('{"status": "error", "code": 500}', "API error response"),
        ('', "Empty response"),
        ('not json at all', "Non-JSON response"),
    ]
    
    for response_body, scenario in malformed_responses:
        with mock_api_response(response_body):
            with pytest.raises((ValueError, APIError)) as exc_info:
                api_client.get_user_data(user_id=123)
            
            # Error should mention the integration point
            error_msg = str(exc_info.value).lower()
            assert any(word in error_msg for word in ["api", "response", "external"])


def test_database_integration_failure_modes():
    """Test database connection and query failure scenarios."""
    
    db_client = DatabaseClient()
    
    # Connection timeout
    with simulate_db_connection_delay(seconds=5):
        with pytest.raises(ConnectionTimeout):
            db_client.get_user(user_id=123)
    
    # Connection refused
    with simulate_db_unavailable():
        with pytest.raises(ConnectionError) as exc_info:
            db_client.get_user(user_id=123)
        
        # Should suggest retry or fallback
        assert any(word in str(exc_info.value).lower() for word in 
                  ["retry", "unavailable", "connection"])
    
    # Query timeout on slow queries
    with simulate_slow_query():
        with pytest.raises(QueryTimeout):
            db_client.get_all_users()  # Potentially slow operation
    
    # Constraint violations
    with pytest.raises(IntegrityError) as exc_info:
        db_client.create_user(email="duplicate@example.com")  # Already exists
    
    # Error should be specific about constraint
    assert "email" in str(exc_info.value).lower()
    assert any(word in str(exc_info.value).lower() for word in 
              ["unique", "duplicate", "constraint"])


def test_file_system_integration_failure_modes():
    """Test file system operation failure scenarios."""
    
    file_handler = FileHandler()
    
    # Disk full
    with simulate_disk_full():
        with pytest.raises(OSError) as exc_info:
            file_handler.save_large_file("test.dat", large_data)
        
        assert "space" in str(exc_info.value).lower()
    
    # Permission denied
    with simulate_no_write_permission():
        with pytest.raises(PermissionError):
            file_handler.save_file("readonly_dir/test.txt", "data")
    
    # File locked by another process
    with simulate_file_locked("existing.txt"):
        with pytest.raises(PermissionError):
            file_handler.delete_file("existing.txt")
```

### **10. Test Resource Cleanup**

Verify that resources are properly cleaned up, especially in error cases. Resource leaks compound over time.

```python
# ✅ RESOURCE CLEANUP TESTING
def test_database_connections_are_cleaned_up():
    """Test database connections are properly closed in all scenarios."""
    
    connection_pool = DatabaseConnectionPool(max_connections=10)
    initial_active = connection_pool.active_count()
    
    # Normal operation cleanup
    with connection_pool.get_connection() as conn:
        conn.execute("SELECT 1")
        active_during = connection_pool.active_count()
        assert active_during == initial_active + 1
    
    # Connection should be returned to pool
    assert connection_pool.active_count() == initial_active
    
    # Error scenario cleanup
    try:
        with connection_pool.get_connection() as conn:
            conn.execute("INVALID SQL QUERY")
    except Exception:
        pass  # Expected to fail
    
    # Connection should still be cleaned up after error
    assert connection_pool.active_count() == initial_active
    
    # Multiple connections cleanup
    connections = []
    try:
        for i in range(5):
            conn = connection_pool.get_connection()
            connections.append(conn)
        
        assert connection_pool.active_count() == initial_active + 5
        
        # Simulate error during processing
        raise ValueError("Simulated processing error")
        
    except ValueError:
        # Cleanup should happen even with exception
        for conn in connections:
            conn.close()
    
    assert connection_pool.active_count() == initial_active


def test_file_handles_are_cleaned_up():
    """Test file handles are closed in error scenarios."""
    
    file_processor = FileProcessor()
    
    # Track open file descriptors
    import psutil
    process = psutil.Process()
    initial_fds = len(process.open_files())
    
    # Normal processing
    file_processor.process_file("normal_file.txt")
    assert len(process.open_files()) == initial_fds
    
    # Error during processing should still close files
    try:
        file_processor.process_file("file_that_causes_error.txt")
    except ProcessingError:
        pass
    
    assert len(process.open_files()) == initial_fds, "File handles leaked after error"
    
    # Batch processing with partial failures
    files = ["good1.txt", "bad.txt", "good2.txt"]
    try:
        file_processor.process_files(files)
    except BatchProcessingError:
        pass
    
    assert len(process.open_files()) == initial_fds, "File handles leaked in batch processing"


def test_thread_cleanup():
    """Test that background threads are properly terminated."""
    
    import threading
    
    processor = BackgroundProcessor()
    initial_thread_count = threading.active_count()
    
    # Start background processing
    processor.start()
    assert threading.active_count() > initial_thread_count
    
    # Normal shutdown
    processor.stop()
    processor.wait_for_completion(timeout=5)
    assert threading.active_count() == initial_thread_count
    
    # Error scenario - processor crash
    processor.start()
    processor._simulate_internal_error()  # Force error state
    
    # Should still clean up threads
    processor.emergency_shutdown()
    processor.wait_for_completion(timeout=5)
    assert threading.active_count() == initial_thread_count, "Threads not cleaned up after crash"
```

### **11. Test State Transitions**

For any stateful code, test that invalid operation sequences can't corrupt the state.

```python
# ✅ STATE TRANSITION TESTING
def test_order_state_machine_prevents_invalid_transitions():
    """Test that order processing prevents invalid state transitions."""
    
    order = Order()
    
    # Valid state transitions
    assert order.status == OrderStatus.PENDING
    
    order.confirm()
    assert order.status == OrderStatus.CONFIRMED
    
    order.ship()
    assert order.status == OrderStatus.SHIPPED
    
    order.deliver()
    assert order.status == OrderStatus.DELIVERED
    
    # Invalid transitions should be rejected
    new_order = Order()
    
    with pytest.raises(InvalidStateTransitionError):
        new_order.ship()  # Can't ship before confirming
    
    with pytest.raises(InvalidStateTransitionError):
        new_order.deliver()  # Can't deliver before shipping
    
    # Can't go backwards
    confirmed_order = Order()
    confirmed_order.confirm()
    
    with pytest.raises(InvalidStateTransitionError):
        confirmed_order.status = OrderStatus.PENDING
    
    # Cancelled orders can't be modified
    cancelled_order = Order()
    cancelled_order.cancel()
    
    with pytest.raises(InvalidStateTransitionError):
        cancelled_order.confirm()


def test_connection_state_machine_handles_concurrent_operations():
    """Test connection state under concurrent operations."""
    
    connection = NetworkConnection()
    
    def connect_operation():
        try:
            connection.connect()
            time.sleep(0.1)  # Simulate work
            connection.disconnect()
        except Exception as e:
            return str(e)
        return "success"
    
    # Concurrent connection attempts
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(connect_operation) for _ in range(10)]
        results = [future.result() for future in futures]
    
    # Should handle concurrent operations gracefully
    successful = [r for r in results if r == "success"]
    errors = [r for r in results if r != "success"]
    
    # At least some should succeed
    assert len(successful) > 0
    
    # Errors should be meaningful, not state corruption
    for error in errors:
        assert "already connected" in error or "connection in progress" in error
        assert "corrupt" not in error.lower()
        assert "invalid state" not in error.lower()
    
    # Final state should be clean
    assert connection.status in [ConnectionStatus.CONNECTED, ConnectionStatus.DISCONNECTED]


def test_cache_state_consistency_under_eviction():
    """Test cache maintains consistent state during eviction."""
    
    cache = LRUCache(max_size=100)
    
    # Fill cache to capacity
    for i in range(100):
        cache.set(f"key_{i}", f"value_{i}")
    
    assert cache.size() == 100
    assert cache.is_full()
    
    # Adding more should trigger eviction
    cache.set("new_key", "new_value")
    
    # State should remain consistent
    assert cache.size() == 100  # Size maintained
    assert "new_key" in cache  # New item added
    assert cache.get("new_key") == "new_value"
    
    # Oldest item should be evicted
    assert "key_0" not in cache
    
    # Concurrent operations during eviction
    def cache_operations():
        for i in range(50):
            cache.set(f"concurrent_{i}", f"value_{i}")
            cache.get(f"key_{i % 10}")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(cache_operations) for _ in range(5)]
        for future in futures:
            future.result()
    
    # Cache should maintain consistency
    assert cache.size() <= 100
    assert cache.size() > 0
    
    # All operations in cache should be valid
    for key, value in cache.items():
        assert cache.get(key) == value  # Consistency check
```

## 🚫 **Universal Anti-Patterns to Avoid**

### **❌ Don't Test Implementation Details**
```python
# BAD - Tests internal implementation
def test_cache_uses_redis():
    cache = Cache()
    assert isinstance(cache._backend, Redis)

# GOOD - Tests behavior
def test_cache_stores_and_retrieves_data():
    cache = Cache()
    cache.set("key", "value")
    assert cache.get("key") == "value"
```

### **❌ Don't Mock What You're Testing**
```python
# BAD - Mocks the actual functionality
@patch.object(UserService, 'create_user')
def test_user_creation(mock_create):
    mock_create.return_value = User(id=1)
    result = UserService().create_user("test@example.com")
    assert result.id == 1

# GOOD - Tests actual functionality
def test_user_creation(test_database):
    service = UserService(test_database)
    result = service.create_user("test@example.com")
    assert result.id is not None
    assert service.get_user(result.id).email == "test@example.com"
```

### **❌ Don't Use Tiny Sample Sizes for Statistical Tests**
```python
# BAD - Meaningless sample size
def test_performance():
    times = [measure_operation() for _ in range(3)]
    assert max(times) < 1.0

# GOOD - Statistically significant sample
def test_performance():
    times = [measure_operation() for _ in range(100)]
    avg_time = sum(times) / len(times)
    assert avg_time < 0.5
    assert max(times) < 1.0  # No outliers
```

### **❌ Don't Ignore Resource Constraints**
```python
# BAD - Tests in perfect conditions
def test_data_processing():
    result = process_large_dataset(million_records)
    assert len(result) == 1000000

# GOOD - Tests under realistic constraints
def test_data_processing_with_memory_limits():
    with memory_limit(512 * 1024 * 1024):  # 512MB limit
        result = process_large_dataset(million_records)
        assert len(result) == 1000000
        # Should work within memory constraints
```

### **❌ Don't Accept Vague Error Messages**
```python
# BAD - Unhelpful error testing
def test_validation_fails():
    with pytest.raises(ValueError):
        validate_email("invalid")

# GOOD - Verify error message quality
def test_validation_provides_helpful_errors():
    with pytest.raises(ValueError) as exc_info:
        validate_email("invalid")
    
    error_msg = str(exc_info.value)
    assert "email" in error_msg.lower()
    assert "must contain @" in error_msg.lower()
    assert "invalid" in error_msg  # Include the problematic input
```

### **❌ Don't Skip Integration Point Testing**
```python
# BAD - Only tests with mocked external dependencies
@patch('external_api.call')
def test_data_fetcher(mock_api):
    mock_api.return_value = {"data": "perfect"}
    result = fetch_user_data(123)
    assert result == {"data": "perfect"}

# GOOD - Tests integration failure modes
def test_data_fetcher_handles_api_failures():
    # Test timeout
    with simulate_api_timeout():
        with pytest.raises(TimeoutError):
            fetch_user_data(123)
    
    # Test malformed response
    with mock_api_response('{"incomplete": json'):
        with pytest.raises(ValueError):
            fetch_user_data(123)
```

### **❌ Don't Ignore Resource Cleanup**
```python
# BAD - No cleanup verification
def test_file_processing():
    processor = FileProcessor()
    result = processor.process_large_file("big_file.txt")
    assert result.success

# GOOD - Verify resource cleanup
def test_file_processing_cleans_up_resources():
    processor = FileProcessor()
    
    initial_fds = get_open_file_count()
    result = processor.process_large_file("big_file.txt")
    final_fds = get_open_file_count()
    
    assert result.success
    assert final_fds == initial_fds, "File descriptors leaked"
```

### **❌ Don't Skip State Transition Testing**
```python
# BAD - Only tests individual operations
def test_order_can_be_shipped():
    order = Order()
    order.ship()
    assert order.status == "shipped"

# GOOD - Tests state machine constraints
def test_order_state_machine():
    order = Order()
    
    # Can't ship unconfirmed order
    with pytest.raises(InvalidStateTransitionError):
        order.ship()
    
    # Valid sequence works
    order.confirm()
    order.ship()
    assert order.status == "shipped"
```

## 🎯 **Domain-Specific Applications**

### **Web API Testing**
Focus on: Concurrent requests, network failures, rate limiting, data consistency

### **Database Testing**
Focus on: Transactions, concurrent access, constraint validation, performance under load

### **Data Processing Testing**
Focus on: Memory usage, processing time, data consistency, error recovery

### **Machine Learning Testing**
Focus on: Model consistency, input validation, performance degradation, edge case predictions

### **Cache Testing**
Focus on: Eviction policies, concurrent access, persistence, cache invalidation

### **Message Queue Testing**
Focus on: Message ordering, delivery guarantees, backpressure handling, failure recovery

## ✅ **Implementation Guidelines**

### **Use Descriptive Test Names That Describe Real Scenarios**
```python
def test_api_handles_database_connection_failure_gracefully()
def test_cache_eviction_maintains_most_frequently_used_items()
def test_payment_processing_prevents_double_charging_under_concurrent_requests()
def test_file_upload_handles_network_interruption_with_resume_capability()
```

### **Include System Context in Test Documentation**
```python
def test_order_processing_under_peak_load():
    """
    Test that order processing maintains data consistency during peak traffic.
    
    Real-world scenario: Black Friday traffic spike where 1000+ orders/second
    are processed simultaneously. System should maintain inventory accuracy
    and prevent overselling.
    
    Failure modes tested:
    - Race conditions in inventory updates
    - Database deadlocks under high concurrency
    - Memory pressure from large order batches
    """
```

### **Test Progressive Failure Modes**
```python
def test_system_degradation_under_increasing_load():
    """Test how system behaves as load increases beyond capacity."""
    
    # Test at different load levels
    load_levels = [10, 50, 100, 200, 500, 1000]  # requests/second
    
    for load in load_levels:
        response_times = []
        error_rates = []
        
        # Generate load for 30 seconds
        for _ in range(load * 30):
            start = time.time()
            response = make_request()
            response_times.append(time.time() - start)
            error_rates.append(1 if response.status_code >= 400 else 0)
        
        avg_response_time = sum(response_times) / len(response_times)
        error_rate = sum(error_rates) / len(error_rates)
        
        print(f"Load: {load} req/s, Avg Response: {avg_response_time:.3f}s, Error Rate: {error_rate:.1%}")
        
        # Define acceptable degradation
        if load <= 100:
            assert avg_response_time < 0.1  # Fast under normal load
            assert error_rate < 0.01  # Very low error rate
        elif load <= 500:
            assert avg_response_time < 0.5  # Slower but acceptable
            assert error_rate < 0.05  # Some errors expected
        else:
            # Beyond capacity - should fail gracefully
            assert error_rate > 0.0  # Should start rejecting requests
            assert avg_response_time < 10.0  # But not hang forever
```

## 🏆 **Success Indicators**

Your tests are production-ready when:

- **Tests find real bugs** that would happen in production
- **Tests run under realistic conditions** (concurrency, resource limits, failures)
- **Tests validate statistical properties** when applicable (performance, consistency)
- **Tests use real infrastructure** components where possible
- **Tests simulate actual user behavior** and usage patterns
- **Tests verify graceful degradation** under stress
- **Tests check error handling** and recovery mechanisms
- **Tests validate data consistency** across operations
- **Tests verify system behavior** across service lifecycles
- **Tests focus on contracts and behavior**, not implementation details
- **Tests verify error messages are helpful** for debugging in production
- **Tests validate all integration points** (timeouts, malformed responses)
- **Tests verify resource cleanup** in both success and error scenarios
- **Tests validate state transitions** and prevent invalid operation sequences
- **Tests are maintainable** and clearly document what real-world scenario they're validating

## 🎯 **The Ultimate Test Question**

Before writing any test, ask yourself:

**"If this test passes but the code still fails in production, what am I not testing?"**

Then test that.

---

Remember Claude: Your tests should make the system more robust, not just make dashboards green. Test the conditions your code will actually face, and think like production thinks - messy, concurrent, failing, and unpredictable.