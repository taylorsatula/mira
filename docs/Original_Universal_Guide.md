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

### **2. Think Adversarially - What Can Go Wrong?**

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

### **8. Test Error Conditions Explicitly**

Don't just test success - test graceful failure.

```python
# ✅ ERROR CONDITION TESTING - File Processing
def test_file_processor_error_handling():
	processor = FileProcessor()
	
	error_scenarios = [
		("nonexistent_file.txt", FileNotFoundError),
		("directory/", IsADirectoryError),
		("empty_file.txt", ValueError),
		("corrupted_file.bin", DataCorruptionError),
		("huge_file.txt", MemoryError),
	]
	
	for file_path, expected_error in error_scenarios:
		setup_error_scenario(file_path, expected_error)
		
		with pytest.raises(expected_error):
			processor.process_file(file_path)
		
		# Verify system is still in good state after error
		assert processor.is_healthy()
		assert processor.can_process_valid_file()
	
	# Test partial failure recovery
	with simulate_disk_full():
		result = processor.process_file("normal_file.txt")
		assert result.status == "partial_success"
		assert "disk space" in result.error_message
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
- **Tests are maintainable** and clearly document what real-world scenario they're validating

## 🎯 **The Ultimate Test Question**

Before writing any test, ask yourself:

**"If this test passes but the code still fails in production, what am I not testing?"**

Then test that.

---

Remember Claude: Your tests should make the system more robust, not just make dashboards green. Test the conditions your code will actually face, and think like production thinks - messy, concurrent, failing, and unpredictable.