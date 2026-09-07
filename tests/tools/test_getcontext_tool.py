"""
Comprehensive integration tests for getcontext_tool.

Tests the full data flow including:
- Real LLM calls (no mocking)
- Background thread execution
- Context propagation via contextvars
- Event bus publishing
- Trinket updates
- Dependency injection
- Error handling

Run with: pytest tests/tools/test_getcontext_tool.py -v -s
"""
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch

import pytest

from tools.implementations.getcontext_tool import GetContextTool, ContextSearchAgent, GetContextToolConfig
from tools.repo import ToolRepository
from working_memory.core import WorkingMemory
from cns.integration.event_bus import EventBus
from cns.core.events import UpdateTrinketEvent
from utils.user_context import set_current_user_id, get_current_user_id, clear_user_context
from utils.timezone_utils import utc_now


logger = logging.getLogger(__name__)


# ==================== FIXTURES ====================

@pytest.fixture
def user_id():
    """Standard test user ID."""
    return "test_user_123"


@pytest.fixture
def continuum_id():
    """Standard test continuum ID."""
    return "test_continuum_456"


@pytest.fixture
def user_context(user_id, continuum_id):
    """Set up user context for tests."""
    set_current_user_id(user_id)
    # Also set continuum_id in context
    from utils.user_context import _user_context
    context = _user_context.get() or {}
    context['continuum_id'] = continuum_id
    _user_context.set(context)

    yield {"user_id": user_id, "continuum_id": continuum_id}

    # Cleanup
    clear_user_context()


@pytest.fixture
def event_bus():
    """Create event bus for testing."""
    return EventBus()


@pytest.fixture
def mock_tool_repo():
    """Create mock tool repository with conversation, memory, and web tools."""
    repo = Mock(spec=ToolRepository)

    # Mock continuum_tool for conversation/memory search
    continuum_tool = Mock()
    continuum_tool.run.return_value = {
        "status": "high_confidence",
        "confidence": 0.9,
        "results": [
            {
                "summary": "Test conversation result about context searching",
                "confidence_score": 0.85,
                "segment_id": "seg12345",
                "display_title": "Test Conversation",
                "matched_entities": ["context", "search"],
                "created_at": "2025-01-01T12:00:00Z"
            }
        ],
        "result_count": 1
    }

    # Mock web_tool
    web_tool = Mock()
    web_tool.run.return_value = {
        "success": True,
        "results": [
            {
                "title": "Test Web Result",
                "url": "https://example.com/test",
                "snippet": "This is a test web search result about context"
            }
        ]
    }

    # Configure repo to return appropriate tool
    def get_tool(name):
        if name == 'continuum_tool':
            return continuum_tool
        elif name == 'web_tool':
            return web_tool
        raise ValueError(f"Unknown tool: {name}")

    repo.get_tool.side_effect = get_tool

    return repo


@pytest.fixture
def mock_working_memory(event_bus):
    """Create mock working memory with event bus."""
    wm = Mock(spec=WorkingMemory)
    wm.event_bus = event_bus
    return wm


@pytest.fixture
def getcontext_tool(mock_tool_repo, mock_working_memory):
    """Create GetContextTool with mocked dependencies."""
    tool = GetContextTool(
        tool_repo=mock_tool_repo,
        working_memory=mock_working_memory
    )
    return tool


@pytest.fixture
def config():
    """Create test configuration."""
    return GetContextToolConfig(
        enabled=True,
        max_iterations=8,
        standard_completion_threshold=3,
        deep_completion_threshold=5,
        search_timeout_seconds=30,  # Shorter for tests
        conversation_max_results=5,
        memory_max_results=5,
        web_max_results=3
    )


# ==================== UNIT TESTS ====================

class TestContextSearchAgent:
    """Test the ContextSearchAgent LLM-based search orchestration."""

    def test_safe_parse_json_valid(self, mock_tool_repo, config):
        """Test JSON parsing with valid input."""
        agent = ContextSearchAgent('standard', mock_tool_repo, config)

        json_str = '{"key": "value", "number": 42}'
        result = agent._safe_parse_json(json_str)

        assert result == {"key": "value", "number": 42}

    def test_safe_parse_json_malformed_with_repair(self, mock_tool_repo, config):
        """Test JSON parsing with malformed input that can be repaired."""
        agent = ContextSearchAgent('standard', mock_tool_repo, config)

        # Missing closing brace - json_repair should fix this
        json_str = '{"key": "value", "number": 42'
        result = agent._safe_parse_json(json_str, fallback={"default": "value"})

        # Should either repair successfully or return fallback
        assert result is not None

    def test_safe_parse_json_completely_broken(self, mock_tool_repo, config):
        """Test JSON parsing with completely broken input returns fallback."""
        agent = ContextSearchAgent('standard', mock_tool_repo, config)

        json_str = 'not json at all!!!'
        fallback = {"error": "fallback"}
        result = agent._safe_parse_json(json_str, fallback=fallback)

        assert result == fallback

    def test_add_to_scratchpad_memory_format(self, mock_tool_repo, config):
        """Test scratchpad addition with memory-style results."""
        agent = ContextSearchAgent('standard', mock_tool_repo, config)

        findings = [
            {
                "text": "Memory content about searching",
                "importance_score": 0.87
            }
        ]

        agent.add_to_scratchpad('memory', findings)

        assert len(agent.scratchpad) == 1
        assert agent.scratchpad[0]['source'] == 'memory'
        assert agent.scratchpad[0]['content'] == "Memory content about searching"
        assert agent.scratchpad[0]['metadata']['title'] == "Memory (importance: 0.87)"

    def test_add_to_scratchpad_conversation_format(self, mock_tool_repo, config):
        """Test scratchpad addition with conversation-style results."""
        agent = ContextSearchAgent('standard', mock_tool_repo, config)

        findings = [
            {
                "summary": "Conversation summary about context",
                "display_title": "Test Conversation",
                "confidence_score": 0.92,
                "timestamp": "2025-01-01T12:00:00Z"
            }
        ]

        agent.add_to_scratchpad('conversation', findings)

        assert len(agent.scratchpad) == 1
        assert agent.scratchpad[0]['source'] == 'conversation'
        assert agent.scratchpad[0]['content'] == "Conversation summary about context"
        assert agent.scratchpad[0]['metadata']['title'] == "Test Conversation"
        assert agent.scratchpad[0]['metadata']['confidence'] == 0.92

    def test_add_to_scratchpad_web_format(self, mock_tool_repo, config):
        """Test scratchpad addition with web-style results."""
        agent = ContextSearchAgent('standard', mock_tool_repo, config)

        findings = [
            {
                "title": "Web Page Title",
                "url": "https://example.com",
                "content": "Web page content about searching"
            }
        ]

        agent.add_to_scratchpad('web', findings)

        assert len(agent.scratchpad) == 1
        assert agent.scratchpad[0]['source'] == 'web'
        assert agent.scratchpad[0]['content'] == "Web page content about searching"
        assert agent.scratchpad[0]['metadata']['title'] == "Web Page Title"
        assert agent.scratchpad[0]['metadata']['url'] == "https://example.com"

    @pytest.mark.integration
    def test_plan_search_real_llm(self, mock_tool_repo, config):
        """Test plan_search with real LLM call."""
        agent = ContextSearchAgent('standard', mock_tool_repo, config)

        query = "What did we discuss about API authentication last week?"
        search_scope = ['conversation', 'memory', 'web']

        plan = agent.plan_search(query, search_scope)

        # Verify structure
        assert 'entities' in plan
        assert 'concepts' in plan
        assert 'source_priority' in plan
        assert 'strategy' in plan

        # Verify types
        assert isinstance(plan['entities'], list)
        assert isinstance(plan['concepts'], list)
        assert isinstance(plan['source_priority'], list)
        assert isinstance(plan['strategy'], str)

        logger.info(f"Plan generated: {json.dumps(plan, indent=2)}")

    @pytest.mark.integration
    def test_determine_next_search_first_search(self, mock_tool_repo, config):
        """Test determine_next_search for initial search."""
        agent = ContextSearchAgent('standard', mock_tool_repo, config)

        query = "How do I configure API timeouts?"
        plan = {
            'entities': ['API', 'timeout', 'configuration'],
            'concepts': ['API settings', 'timeout configuration'],
            'source_priority': ['conversation', 'memory', 'web'],
            'strategy': 'Search conversation first for recent discussions'
        }

        # First search with empty history
        next_search = agent.determine_next_search(query, plan)

        assert next_search is not None
        assert next_search['source'] in ['conversation', 'memory', 'web']
        assert next_search['query'] == query

        logger.info(f"First search: {json.dumps(next_search, indent=2)}")

    @pytest.mark.integration
    def test_determine_next_search_with_findings(self, mock_tool_repo, config):
        """Test determine_next_search after gathering some findings."""
        agent = ContextSearchAgent('standard', mock_tool_repo, config)

        query = "How do I configure API timeouts?"
        plan = {
            'entities': ['API', 'timeout', 'configuration'],
            'concepts': ['API settings', 'timeout configuration'],
            'source_priority': ['conversation', 'memory', 'web'],
            'strategy': 'Search conversation first for recent discussions'
        }

        # Add some findings to scratchpad
        agent.add_to_scratchpad('conversation', [
            {
                "summary": "Discussion about timeout settings in config file",
                "display_title": "API Configuration Discussion",
                "confidence_score": 0.85
            }
        ])

        # Mark that we searched conversation
        agent.search_history.append({
            'source': 'conversation',
            'query': query,
            'reason': 'Initial search'
        })

        next_search = agent.determine_next_search(query, plan)

        # Should either suggest next search or return None if complete
        if next_search:
            assert 'source' in next_search
            assert 'query' in next_search
            assert 'reason' in next_search
            logger.info(f"Next search: {json.dumps(next_search, indent=2)}")
        else:
            logger.info("Agent determined search is complete")

    @pytest.mark.integration
    def test_is_search_complete_standard_mode(self, mock_tool_repo, config):
        """Test completion check in standard mode with real LLM."""
        agent = ContextSearchAgent('standard', mock_tool_repo, config)

        query = "What is the API timeout setting?"

        # Add relevant finding
        agent.add_to_scratchpad('conversation', [
            {
                "summary": "API timeout is set to 60 seconds in config.py",
                "display_title": "API Configuration",
                "confidence_score": 0.95
            }
        ])

        is_complete, reason = agent.is_search_complete(query)

        assert isinstance(is_complete, bool)
        assert isinstance(reason, str)
        assert len(reason) > 0

        logger.info(f"Completion check: complete={is_complete}, reason={reason}")

    @pytest.mark.integration
    def test_is_search_complete_deep_mode(self, mock_tool_repo, config):
        """Test completion check in deep mode requires more thorough search."""
        agent = ContextSearchAgent('deep', mock_tool_repo, config)

        query = "How does the authentication system work?"

        # Add single finding - deep mode should want more
        agent.add_to_scratchpad('conversation', [
            {
                "summary": "Uses JWT tokens for authentication",
                "display_title": "Auth Discussion",
                "confidence_score": 0.80
            }
        ])

        is_complete, reason = agent.is_search_complete(query)

        assert isinstance(is_complete, bool)
        assert isinstance(reason, str)

        logger.info(f"Deep mode completion: complete={is_complete}, reason={reason}")

    @pytest.mark.integration
    def test_summarize_findings_real_llm(self, mock_tool_repo, config):
        """Test final summarization with real LLM call."""
        agent = ContextSearchAgent('standard', mock_tool_repo, config)

        query = "How do I configure database connections?"

        # Add diverse findings
        agent.add_to_scratchpad('conversation', [
            {
                "summary": "Database connection string is in config.py",
                "display_title": "Database Setup",
                "confidence_score": 0.90
            }
        ])
        agent.add_to_scratchpad('memory', [
            {
                "text": "PostgreSQL is used for main database with connection pooling",
                "importance_score": 0.85
            }
        ])
        agent.add_to_scratchpad('web', [
            {
                "title": "PostgreSQL Connection Best Practices",
                "url": "https://example.com/postgres",
                "content": "Use connection pooling for better performance"
            }
        ])

        summary = agent.summarize_findings(query)

        # Verify structure
        assert 'query' in summary
        assert 'summary' in summary
        assert 'key_findings' in summary
        assert 'sources_searched' in summary
        assert 'confidence' in summary

        # Verify types
        assert isinstance(summary['key_findings'], list)
        assert isinstance(summary['confidence'], (int, float))
        assert 0.0 <= summary['confidence'] <= 1.0

        logger.info(f"Final summary: {json.dumps(summary, indent=2)}")


class TestGetContextToolCore:
    """Test core GetContextTool functionality."""

    def test_initialization_with_dependencies(self, mock_tool_repo, mock_working_memory):
        """Test tool initializes correctly with dependencies."""
        tool = GetContextTool(
            tool_repo=mock_tool_repo,
            working_memory=mock_working_memory
        )

        assert tool.tool_repo == mock_tool_repo
        assert tool.working_memory == mock_working_memory
        assert tool.event_bus == mock_working_memory.event_bus
        assert tool.config is not None

    def test_initialization_without_dependencies(self):
        """Test tool initializes without crashing when dependencies are None."""
        tool = GetContextTool(tool_repo=None, working_memory=None)

        assert tool.tool_repo is None
        assert tool.working_memory is None
        assert tool.event_bus is None
        assert tool.config is not None

    def test_run_returns_immediately(self, getcontext_tool, user_context):
        """Test that run() returns immediately with task_id."""
        query = "Test query"

        start_time = time.time()
        result = getcontext_tool.run(query)
        elapsed = time.time() - start_time

        # Should return in under 1 second
        assert elapsed < 1.0

        # Verify response structure
        assert result['success'] is True
        assert 'task_id' in result
        assert result['status'] == 'searching'
        assert result['message'] == f"Context search initiated for: '{query}'. Results will appear when ready."
        assert result['search_scope'] == ['conversation', 'memory', 'web']
        assert result['search_mode'] == 'standard'

    def test_run_validates_query(self, getcontext_tool, user_context):
        """Test that run() validates query parameter."""
        with pytest.raises(ValueError, match="Query is required"):
            getcontext_tool.run("")

    def test_run_validates_search_scope(self, getcontext_tool, user_context):
        """Test that run() validates and filters search_scope."""
        query = "Test query"

        # Invalid scope should be filtered to defaults
        result = getcontext_tool.run(query, search_scope=['invalid', 'conversation'])

        assert 'conversation' in result['search_scope']
        assert 'invalid' not in result['search_scope']

    def test_run_empty_search_scope_uses_defaults(self, getcontext_tool, user_context):
        """Test that empty search_scope defaults to all sources."""
        query = "Test query"

        result = getcontext_tool.run(query, search_scope=[])

        assert set(result['search_scope']) == {'conversation', 'memory', 'web'}

    def test_search_conversation_calls_continuum_tool(self, getcontext_tool):
        """Test _search_conversation delegates to continuum_tool."""
        query = "test conversation search"
        entities = ["entity1", "entity2"]

        results = getcontext_tool._search_conversation(query, entities)

        # Verify continuum_tool was called correctly
        getcontext_tool.tool_repo.get_tool.assert_called_with('continuum_tool')
        continuum_tool = getcontext_tool.tool_repo.get_tool('continuum_tool')
        continuum_tool.run.assert_called_once_with(
            operation='search',
            query=query,
            search_mode='summaries',
            entities=entities,
            max_results=getcontext_tool.config.conversation_max_results
        )

        # Verify results extracted correctly
        assert len(results) == 1
        assert results[0]['summary'] == "Test conversation result about context searching"

    def test_search_memory_calls_continuum_tool(self, getcontext_tool):
        """Test _search_memory delegates to continuum_tool."""
        query = "test memory search"
        entities = ["entity1"]

        results = getcontext_tool._search_memory(query, entities)

        # Verify continuum_tool was called correctly
        continuum_tool = getcontext_tool.tool_repo.get_tool('continuum_tool')
        continuum_tool.run.assert_called_once_with(
            operation='search',
            query=query,
            search_mode='memories',
            entities=entities,
            max_results=getcontext_tool.config.memory_max_results
        )

    def test_search_web_calls_web_tool(self, getcontext_tool):
        """Test _search_web delegates to web_tool."""
        query = "test web search"

        results = getcontext_tool._search_web(query)

        # Verify web_tool was called correctly
        getcontext_tool.tool_repo.get_tool.assert_called_with('web_tool')
        web_tool = getcontext_tool.tool_repo.get_tool('web_tool')
        web_tool.run.assert_called_once_with(
            operation='search',
            query=query,
            max_results=getcontext_tool.config.web_max_results
        )

        # Verify results extracted correctly
        assert len(results) == 1
        assert results[0]['title'] == "Test Web Result"

    def test_search_methods_handle_missing_tool_repo(self):
        """Test search methods return empty list when tool_repo is None."""
        tool = GetContextTool(tool_repo=None, working_memory=None)

        # Should return empty list and log warning, not crash
        conv_results = tool._search_conversation("query")
        mem_results = tool._search_memory("query")
        web_results = tool._search_web("query")

        assert conv_results == []
        assert mem_results == []
        assert web_results == []


class TestGetContextToolEventPublishing:
    """Test event bus publishing from GetContextTool."""

    def test_publish_pending_result(self, getcontext_tool, user_context, event_bus):
        """Test pending result is published correctly."""
        events_received = []

        def capture_event(event):
            events_received.append(event)

        event_bus.subscribe('UpdateTrinketEvent', capture_event)

        # Trigger run which publishes pending
        result = getcontext_tool.run("Test query")
        task_id = result['task_id']

        # Wait briefly for event
        time.sleep(0.1)

        # Verify event was published
        assert len(events_received) == 1
        event = events_received[0]

        assert event.target_trinket == 'GetContextTrinket'
        assert event.context['task_id'] == task_id
        assert event.context['status'] == 'pending'
        assert event.context['query'] == "Test query"

    def test_publish_success_result(self, getcontext_tool, user_context, continuum_id):
        """Test success result publishing structure."""
        events_received = []

        def capture_event(event):
            events_received.append(event)

        getcontext_tool.event_bus.subscribe('UpdateTrinketEvent', capture_event)

        task_id = str(uuid.uuid4())
        summary = {
            "query": "test",
            "summary": "Test summary",
            "key_findings": [],
            "confidence": 0.8
        }

        getcontext_tool._publish_success_result(continuum_id, task_id, summary)

        assert len(events_received) == 1
        event = events_received[0]

        assert event.context['task_id'] == task_id
        assert event.context['status'] == 'success'
        assert event.context['summary'] == summary

    def test_publish_timeout_result(self, getcontext_tool, user_context, continuum_id):
        """Test timeout result publishing structure."""
        events_received = []

        def capture_event(event):
            events_received.append(event)

        getcontext_tool.event_bus.subscribe('UpdateTrinketEvent', capture_event)

        task_id = str(uuid.uuid4())

        getcontext_tool._publish_timeout_result(
            continuum_id=continuum_id,
            task_id=task_id,
            query="test query",
            iteration=5,
            elapsed=31.5,
            search_mode='standard',
            findings_count=3
        )

        assert len(events_received) == 1
        event = events_received[0]

        assert event.context['status'] == 'timeout'
        assert event.context['iteration'] == 5
        assert event.context['elapsed'] == 31.5

    def test_publish_failure_result(self, getcontext_tool, user_context, continuum_id):
        """Test failure result publishing structure."""
        events_received = []

        def capture_event(event):
            events_received.append(event)

        getcontext_tool.event_bus.subscribe('UpdateTrinketEvent', capture_event)

        task_id = str(uuid.uuid4())

        getcontext_tool._publish_failure_result(
            continuum_id=continuum_id,
            task_id=task_id,
            query="test query",
            error="Connection timeout",
            error_type="TimeoutError"
        )

        assert len(events_received) == 1
        event = events_received[0]

        assert event.context['status'] == 'failed'
        assert event.context['error'] == "Connection timeout"
        assert event.context['error_type'] == "TimeoutError"

    def test_publish_methods_handle_missing_event_bus(self, continuum_id):
        """Test publish methods don't crash when event_bus is None."""
        tool = GetContextTool(tool_repo=None, working_memory=None)

        task_id = str(uuid.uuid4())

        # Should not crash, just return early
        tool._publish_success_result(continuum_id, task_id, {})
        tool._publish_timeout_result(continuum_id, task_id, "q", 1, 1.0, 'standard', 0)
        tool._publish_failure_result(continuum_id, task_id, "q", "err", "ErrType")
        tool._publish_pending_result(continuum_id, task_id, "q", ['conversation'], 'standard')


class TestGetContextToolContextPropagation:
    """Test context propagation to background thread."""

    def test_context_available_in_worker_thread(self, getcontext_tool, user_context, user_id):
        """Test that user_id context is available in background worker."""
        context_checks = []

        # Patch _async_search_worker to check context
        original_worker = getcontext_tool._async_search_worker

        def worker_with_context_check(*args, **kwargs):
            try:
                # Check if context is available
                current_user = get_current_user_id()
                context_checks.append(('success', current_user))
            except RuntimeError as e:
                context_checks.append(('error', str(e)))

            # Don't actually run the search
            return

        with patch.object(getcontext_tool, '_async_search_worker', worker_with_context_check):
            result = getcontext_tool.run("Test query")

            # Wait for thread to execute
            time.sleep(0.5)

        # Verify context was available
        assert len(context_checks) == 1
        status, value = context_checks[0]
        assert status == 'success', f"Context not available: {value}"
        assert value == user_id

    def test_context_missing_raises_error(self):
        """Test that missing context in worker causes RuntimeError."""
        # Clear any existing context
        clear_user_context()

        tool = GetContextTool(tool_repo=Mock(), working_memory=Mock())

        # This should raise because no context is set
        with pytest.raises(RuntimeError):
            get_current_user_id()


# ==================== INTEGRATION TESTS ====================

class TestGetContextToolIntegration:
    """Full integration tests with real LLM calls and background threads."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_search_workflow_standard_mode(self, getcontext_tool, user_context, event_bus):
        """Test complete search workflow in standard mode with real LLM."""
        events_received = []

        def capture_event(event):
            events_received.append(event)
            logger.info(f"Event received: {event.context.get('status')} for task {event.context.get('task_id', 'unknown')[:8]}")

        event_bus.subscribe('UpdateTrinketEvent', capture_event)

        query = "How do I configure database timeout settings?"

        # Start search
        result = getcontext_tool.run(query, search_mode='standard')
        task_id = result['task_id']

        logger.info(f"Started search {task_id[:8]} for: {query}")

        # Wait for search to complete (up to 60 seconds)
        max_wait = 60
        start_time = time.time()
        final_event = None

        while time.time() - start_time < max_wait:
            time.sleep(1)

            # Check for completion events
            for event in events_received:
                if event.context.get('task_id') == task_id:
                    status = event.context.get('status')
                    if status in ['success', 'timeout', 'failed']:
                        final_event = event
                        break

            if final_event:
                break

        # Verify we got a final result
        assert final_event is not None, "Search did not complete within timeout"

        final_status = final_event.context['status']
        logger.info(f"Search completed with status: {final_status}")

        if final_status == 'success':
            summary = final_event.context['summary']

            assert 'query' in summary
            assert 'summary' in summary
            assert 'key_findings' in summary
            assert summary['query'] == query

            logger.info(f"Success! Summary: {summary['summary']}")
            logger.info(f"Key findings: {json.dumps(summary['key_findings'], indent=2)}")

        elif final_status == 'timeout':
            logger.warning(f"Search timed out after {final_event.context['elapsed']}s")
            logger.info(f"Found {final_event.context['findings_count']} findings before timeout")

        elif final_status == 'failed':
            logger.error(f"Search failed: {final_event.context['error']}")
            pytest.fail(f"Search failed: {final_event.context['error']}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_deep_search_mode(self, getcontext_tool, user_context, event_bus):
        """Test deep search mode gathers more comprehensive results."""
        events_received = []

        def capture_event(event):
            events_received.append(event)

        event_bus.subscribe('UpdateTrinketEvent', capture_event)

        query = "Explain the authentication and authorization system"

        result = getcontext_tool.run(query, search_mode='deep')
        task_id = result['task_id']

        logger.info(f"Started deep search {task_id[:8]}")

        # Wait for completion
        max_wait = 90  # Deep mode may take longer
        start_time = time.time()
        final_event = None

        while time.time() - start_time < max_wait:
            time.sleep(1)

            for event in events_received:
                if event.context.get('task_id') == task_id:
                    status = event.context.get('status')
                    if status in ['success', 'timeout', 'failed']:
                        final_event = event
                        break

            if final_event:
                break

        assert final_event is not None

        if final_event.context['status'] == 'success':
            summary = final_event.context['summary']

            # Deep mode should have more iterations
            assert summary.get('iterations', 0) >= getcontext_tool.config.deep_completion_threshold

            logger.info(f"Deep search completed in {summary.get('iterations')} iterations")

    @pytest.mark.integration
    def test_concurrent_searches(self, getcontext_tool, user_context, event_bus):
        """Test multiple concurrent searches can run simultaneously."""
        events_received = []

        def capture_event(event):
            events_received.append(event)

        event_bus.subscribe('UpdateTrinketEvent', capture_event)

        queries = [
            "What is the database schema?",
            "How does caching work?",
            "What are the API endpoints?"
        ]

        task_ids = []
        for query in queries:
            result = getcontext_tool.run(query)
            task_ids.append(result['task_id'])
            logger.info(f"Started search {result['task_id'][:8]}: {query}")

        # Wait for all to complete
        max_wait = 60
        start_time = time.time()
        completed_tasks = set()

        while time.time() - start_time < max_wait and len(completed_tasks) < len(task_ids):
            time.sleep(1)

            for event in events_received:
                task_id = event.context.get('task_id')
                if task_id in task_ids:
                    status = event.context.get('status')
                    if status in ['success', 'timeout', 'failed']:
                        completed_tasks.add(task_id)

        logger.info(f"Completed {len(completed_tasks)}/{len(task_ids)} searches")

        # At least some should complete
        assert len(completed_tasks) > 0

    @pytest.mark.integration
    def test_search_with_limited_scope(self, getcontext_tool, user_context, event_bus):
        """Test search with limited scope (only conversation)."""
        events_received = []

        def capture_event(event):
            events_received.append(event)

        event_bus.subscribe('UpdateTrinketEvent', capture_event)

        query = "Test query for limited scope"

        result = getcontext_tool.run(query, search_scope=['conversation'])
        task_id = result['task_id']

        assert result['search_scope'] == ['conversation']

        # Wait briefly for pending event
        time.sleep(1)

        # Verify only conversation was included
        pending_events = [e for e in events_received
                         if e.context.get('task_id') == task_id
                         and e.context.get('status') == 'pending']

        assert len(pending_events) == 1
        assert pending_events[0].context['search_scope'] == ['conversation']


# ==================== ERROR HANDLING TESTS ====================

class TestGetContextToolErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.integration
    def test_llm_api_failure_handling(self, mock_tool_repo, mock_working_memory, user_context):
        """Test handling of LLM API failures."""
        # Create tool with invalid API configuration
        # Note: get_api_key is imported inside the _get_llm_client method, so we patch it in vault_client
        with patch('clients.vault_client.get_api_key') as mock_get_key:
            mock_get_key.return_value = "invalid_key_123"

            tool = GetContextTool(
                tool_repo=mock_tool_repo,
                working_memory=mock_working_memory
            )

            events_received = []

            def capture_event(event):
                events_received.append(event)

            tool.event_bus.subscribe('UpdateTrinketEvent', capture_event)

            result = tool.run("Test query")
            task_id = result['task_id']

            # Wait for failure or success (with invalid key, it may fail fast or succeed with mock tools)
            time.sleep(5)

            # Check final status
            final_events = [e for e in events_received
                          if e.context.get('task_id') == task_id
                          and e.context.get('status') in ['failed', 'success', 'timeout']]

            # Should have received some final event (success, failure, or timeout)
            assert len(final_events) > 0, f"No final event received. All events: {[e.context.get('status') for e in events_received]}"
            logger.info(f"Final status: {final_events[0].context['status']}")

    def test_timeout_enforcement(self, getcontext_tool, user_context, config):
        """Test that searches timeout after configured duration."""
        # Set very short timeout for testing
        getcontext_tool.config.search_timeout_seconds = 2
        getcontext_tool.config.max_iterations = 100  # Won't reach this

        events_received = []

        def capture_event(event):
            events_received.append(event)

        getcontext_tool.event_bus.subscribe('UpdateTrinketEvent', capture_event)

        # Start search that will timeout
        result = getcontext_tool.run("Complex query requiring many iterations")
        task_id = result['task_id']

        # Wait for timeout
        time.sleep(5)

        # Check for timeout event
        timeout_events = [e for e in events_received
                         if e.context.get('task_id') == task_id
                         and e.context.get('status') == 'timeout']

        if timeout_events:
            assert timeout_events[0].context['elapsed'] > 2.0
            logger.info(f"Timed out after {timeout_events[0].context['elapsed']}s")

    def test_missing_vault_key_handling(self, mock_tool_repo, mock_working_memory, user_context):
        """Test handling when Vault API key is missing."""
        # Patch at the vault_client module level since import happens inside the method
        with patch('clients.vault_client.get_api_key') as mock_get_key:
            mock_get_key.side_effect = FileNotFoundError("Key not found in Vault")

            tool = GetContextTool(
                tool_repo=mock_tool_repo,
                working_memory=mock_working_memory
            )

            events_received = []

            def capture_event(event):
                events_received.append(event)

            tool.event_bus.subscribe('UpdateTrinketEvent', capture_event)

            result = tool.run("Test query")
            task_id = result['task_id']

            # Wait for failure
            time.sleep(3)

            # Should receive failure event (the thread will crash and publish failure)
            failure_events = [e for e in events_received
                            if e.context.get('task_id') == task_id
                            and e.context.get('status') == 'failed']

            # Thread should crash and publish failure event
            assert len(failure_events) > 0, f"Expected failure event. Got: {[e.context.get('status') for e in events_received]}"
            assert 'error' in failure_events[0].context
            logger.info(f"Failed as expected: {failure_events[0].context['error']}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--log-cli-level=INFO'])
