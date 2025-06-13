# MIRA - Testing Guide

## 🚨 Core Testing Principles (Non-Negotiable)

### Test Quality
- **Test the Contract, Not Implementation**: Test what the code PROMISES to do (input → output), not HOW it does it internally. If you're testing method call counts or internal state, STOP - you're testing the wrong thing.
- **Real Bug Documentation**: Every test MUST have a "REAL BUG THIS CATCHES" comment explaining exactly what production failure this would prevent. If you can't write this comment specifically, rewrite the test.
- **One Test at a Time**: Write tests individually with reflection between each one. Resist pattern completion pressure, completeness anxiety, and edge case excitement. Each test should validate ONE specific contract.
- **Use Real Data**: Never create fake response structures or mock data. Use real systems to generate real data for testing. Testing fake data only validates that your fake data matches your expectations.

### Environment Setup
- **Activate environment**: `conda activate mira`
- **Run tests**: `PYTHONPATH=/Users/taylut/Programming/GitHub/botwithmemory pytest tests/test_file.py::test_function`

### PostgreSQL Test Database
- **Active instance**: PostgreSQL running with `mira_app` user
- **Test database**: `lt_memory_test` 
- **Connection**: `postgresql://mira_app@localhost/lt_memory_test`
- **Permissions**: Grant with `psql -U god -h localhost -d lt_memory_test -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mira_app;"`


### Test Coverage Strategy
- **Public Contracts First**: Always test the public API contracts that users depend on - these are your primary responsibility
- **Important Private Methods**: Test private methods that contain significant logic, handle critical operations, or could fail in ways that are hard to diagnose through public APIs alone
- **Implementation Details to Avoid**: Don't test trivial private methods, method call counts, or internal state that doesn't affect observable behavior
- **Fix Upstream, Don't Accommodate**: If tests reveal flawed logic in the source code, fix the source code rather than editing tests to accommodate bad behavior. Tests should validate correct behavior, not enable incorrect behavior.

### Mocking Strategy
- **NEVER USE MOCK**: Never use mock never use mock never use mock never use mock never use unittest never use mock never use mock never use mock. Want to use Mock? Don't. I'd rather have you fail at writing a test than try to mock things. If there is something that should be mocked I will explicitly tell you. Incase you forgot: Don't use Mock.

### Debugging Strategy
- **Execute Code to Explore Hunches**: When tests fail unexpectedly or reveal confusing behavior, use executable code within the conversation to probe and understand root causes. Don't guess - investigate systematically.
- **How to Explore Hunches Effectively**:
  1. **Isolate the Problem**: Create minimal reproduction cases that test your specific hypothesis
  2. **Trace Data Flow**: Follow data from input to output, checking transformations at each step
  3. **Check Assumptions**: Verify that your assumptions about system behavior are actually correct
  4. **Test Edge Cases**: Probe boundary conditions, timezone handling, data type conversions
  5. **Inspect Real State**: Check database contents, file system state, memory contents - what's actually there vs. what you expect
  6. **Step Through Logic**: Execute the problematic code path step-by-step to see where it diverges from expectation
## 🎯 Essential Decision Tree

### Before Writing Any Test
1. **What REAL BUG in production would this catch?** → If you can't answer specifically, don't write the test
2. **Am I testing a CONTRACT or IMPLEMENTATION?** → CONTRACT: "Given X input, produces Y output" ✅ IMPLEMENTATION: "Method A calls Method B twice" ❌  
3. **What should I mock?** → External dependencies only (APIs, email services) ✅ Your own code ❌

### Test Documentation Pattern
```python
def test_feature_works_correctly(self):
    """
    Brief description of what this test validates.
    
    REAL BUG THIS CATCHES: Specific description of what production bug
    this test would catch if the code were broken. Be concrete about
    which method/logic would fail and how it would break user experience.
    """
```

## ✅ Testing Patterns That Work

### Public Contract Testing
```python
def test_user_creation_stores_and_retrieves(test_database):
    """
    REAL BUG THIS CATCHES: If user creation or retrieval has SQL bugs,
    users can't be saved or found, breaking authentication.
    """
    service = UserService(database=test_database)  # Real database
    
    created_user = service.create_user("test@email.com")
    assert created_user.id is not None
    
    # Verify the real contract - persistence works
    retrieved_user = service.get_user(created_user.id)  
    assert retrieved_user.email == "test@email.com"
```

### Important Private Method Testing
```python
def test_cache_key_handles_unicode(self, basic_cache):
    """
    REAL BUG THIS CATCHES: If _get_cache_key() fails with Unicode,
    international users can't cache their data, breaking the system.
    """
    unicode_text = "Hello 世界 🌍"
    key = basic_cache._get_cache_key(unicode_text)  # Important private logic
    
    assert isinstance(key, str)
    assert len(key) == 64  # SHA-256 hex length
    assert all(c in "0123456789abcdef" for c in key)

def test_path_generation_prevents_directory_traversal(self, basic_cache):
    """
    REAL BUG THIS CATCHES: If _get_cache_path() allows directory traversal,
    attackers could write files outside cache directory, compromising security.
    """
    cache_key = "a" * 64
    path = basic_cache._get_cache_path(cache_key)
    
    # Security: ensure path stays within cache directory
    assert basic_cache.cache_dir in path.parents or path == basic_cache.cache_dir
    assert ".." not in str(path)
```

### External Dependency Mocking
```python
def test_user_registration_sends_email(mocker, test_database):
    """
    REAL BUG THIS CATCHES: If email integration breaks, users don't get
    welcome emails, causing poor onboarding experience.
    """
    # Mock ONLY external email service
    mock_email = mocker.patch('external_email.send_welcome')
    mock_email.return_value = True
    
    # Use REAL components for everything else
    user_service = UserService(database=test_database)
    user = user_service.register_user("test@email.com", "password")
    
    # Verify real persistence
    saved_user = test_database.get_user(user.id)
    assert saved_user.email == "test@email.com"
    
    # Verify external integration
    mock_email.assert_called_once()
```

### Concurrent Access Testing
```python
def test_concurrent_writes_maintain_integrity(self, temp_cache_dir, sample_embeddings):
    """
    REAL BUG THIS CATCHES: If concurrent writes corrupt data, users get
    wrong results back, breaking similarity search accuracy.
    """
    cache = EmbeddingCache(str(temp_cache_dir))
    embedding = sample_embeddings["medium"]
    
    def write_and_verify(thread_id):
        text = f"thread_{thread_id}_data"
        cache.set(text, embedding)
        result = cache.get(text)
        if result is not None:
            np.testing.assert_array_equal(result, embedding)
            return True
        return False
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(write_and_verify, i) for i in range(20)]
        results = [f.result() for f in futures]
    
    assert all(results)  # All concurrent operations succeeded
```

### Error Handling Testing
```python
def test_corrupted_files_handled_gracefully(self, basic_cache, sample_embeddings):
    """
    REAL BUG THIS CATCHES: If corrupted cache files crash the system,
    users experience downtime instead of graceful degradation.
    """
    basic_cache.set("test", sample_embeddings["small"])
    
    # Corrupt the file (real-world scenario)
    cache_key = basic_cache._get_cache_key("test")
    cache_path = basic_cache._get_cache_path(cache_key)
    with open(cache_path, 'w') as f:
        f.write("corrupted data")
    
    # Should handle gracefully, not crash
    result = basic_cache.get("test")
    assert result is None
    assert not cache_path.exists()  # Corrupted file cleaned up
```

## 🚫 Critical Anti-Patterns

### Testing Implementation Details
- **Never Test Call Counts**: `assert mock.call_count == 2` tests HOW code works, not WHAT it accomplishes
- **Never Test Method Call Sequences**: Testing that public method calls specific private methods in specific order tests implementation, not contract
- **Never Test Trivial Private Methods**: Don't test simple getters, basic validation, or private methods with obvious logic

### Creating Fake Test Data  
- **Never Mock Your Own Response Structures**: Creating fake API responses tests your ability to create fake data, not your code's ability to handle real responses
- **Never Use Hardcoded Test Data**: Use real systems to generate real test data that reflects actual production formats

### Over-Mocking
- **Never Mock Your Own Infrastructure**: Mocking databases, file systems, or business logic you control hides integration bugs
- **Never Mock the Code Being Tested**: If you mock away all your logic, you're testing the mocks, not your code

## ⚡ Performance Testing Strategy

### Normal Performance Tests
- **Realistic Expectations**: Set performance expectations based on actual business requirements, not arbitrary speed goals
- **Baseline Characteristics**: Establish normal operating performance to detect regressions

### Stress Tests
- **Document System Limits**: Stress tests may fail - that's the point. They show where the system breaks down
- **Accept Degradation**: Under extreme load, some degradation is acceptable. Document what "good enough under stress" means

### Code Quality
- **Tests**: `pytest` or `pytest tests/test_file.py::test_function`
- **Lint**: `flake8`
- **Type check**: `mypy .`
- **Format**: `black .`

## ✅ Success Criteria

**Your tests are production-ready when:**

✅ **Contract-Focused**: Tests verify promises the code makes to users, not internal implementation details  
✅ **Real Bug Prevention**: Every test has a specific "REAL BUG THIS CATCHES" comment and would actually prevent production failures  
✅ **Comprehensive Coverage**: Tests public contracts AND important private methods that handle critical logic  
✅ **Real Infrastructure**: Uses real databases, files, caches where possible; mocks only external dependencies  
✅ **Real-World Conditions**: Tests error scenarios, concurrent access, and edge cases that happen in production  
✅ **Performance Awareness**: Includes both realistic performance expectations and stress tests that discover limits

**If your tests pass but production fails, you're testing the wrong things.**

## 🔍 Debugging Methodology: Execute Code to Explore Hunches

When tests fail unexpectedly or reveal confusing behavior, don't rationalize or work around the issue. Use executable code to systematically investigate root causes.

### Example: Timezone Bug Investigation

```python
# Test failed: expected dates ['2025-06-06', '2025-06-07', '2025-06-08'] 
# but got ['2025-06-05', '2025-06-06', '2025-06-07']

# Step 1: Isolate the problem - test date conversion logic
base_date = utc_now() - timedelta(days=3)
for day_offset in range(3):
    current_date = base_date + timedelta(days=day_offset)
    target_date = current_date.date()
    storage_datetime = ensure_utc(datetime.combine(target_date, datetime.min.time()))
    print(f'day_offset={day_offset}: target_date={target_date}, storage_datetime={storage_datetime}')

# Step 2: Check database state directly
with memory_manager.get_session() as session:
    all_archived = session.query(ArchivedConversation).all()
    print(f'Database contents: {[(ac.conversation_date, ac.message_count) for ac in all_archived]}')

# Step 3: Test retrieval logic
conv = session.query(ArchivedConversation).first()
print(f'Raw datetime: {conv.conversation_date}')
print(f'Raw .date(): {conv.conversation_date.date()}')
print(f'UTC converted: {ensure_utc(conv.conversation_date).date()}')

# Result: Found bug in retrieval - was calling .date() without UTC conversion first!
```

### Investigation Principles

1. **Create Minimal Test Cases**: Strip away complexity to isolate the specific behavior you're investigating
2. **Verify Assumptions**: Test that your understanding of how the system works is actually correct  
3. **Trace Data Transformations**: Follow data through each step to find where it diverges from expected
4. **Check Real State**: Look at actual database/file/memory contents, not just what your code thinks is there
5. **Test Boundary Conditions**: Timezone conversions, data type handling, edge cases often reveal bugs

### When to Use This Approach

- Test failures that don't make logical sense
- Unexpected behavior in integration with external systems
- Data corruption or transformation issues
- Performance problems with unclear root causes
- Any time you find yourself saying "that should work" but it doesn't

Remember: **Don't accommodate bugs with workarounds. Find and fix the root cause.**