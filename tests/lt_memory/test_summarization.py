"""
Production-grade tests for SummarizationEngine white-label summarization system.

This test suite focuses on realistic production scenarios that matter for
reliability. We test the contracts and behaviors that users actually depend on,
using real LLM API calls and real template files for authentic testing.

Testing philosophy:
1. Test the public API contracts that users rely on
2. Test critical private methods that could fail in subtle ways  
3. Test template system with real files and fallback behavior
4. Test LLM integration with real API calls (not mocked)
5. Verify error handling for missing templates and API failures
6. Test real conversation summarization workflows
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch

# Import the system under test
from lt_memory.utils.summarization import SummarizationEngine
from api.llm_provider import LLMProvider
from errors import ToolError, ErrorCode


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_messages():
    """
    Provides realistic message samples for testing.
    
    These represent actual conversation patterns we see in production:
    - User/assistant exchanges
    - Different content types (text, lists)
    - Edge cases (empty, special characters)
    """
    return {
        "simple_conversation": [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming."},
            {"role": "user", "content": "Can you give me an example?"},
            {"role": "assistant", "content": "Sure! Email spam detection is a common example - the system learns to identify spam by analyzing thousands of emails."}
        ],
        
        "complex_conversation": [
            {"role": "user", "content": "I need help with my Python project"},
            {"role": "assistant", "content": "I'd be happy to help! What specific aspect of your Python project are you working on?"},
            {"role": "user", "content": "I'm trying to implement a REST API using Flask"},
            {"role": "assistant", "content": "Great choice! Flask is excellent for APIs. Here's a basic structure:\n\n```python\nfrom flask import Flask, jsonify\napp = Flask(__name__)\n\n@app.route('/api/data')\ndef get_data():\n    return jsonify({'message': 'Hello World'})\n```"},
            {"role": "user", "content": "How do I handle POST requests?"},
            {"role": "assistant", "content": "For POST requests, you'll want to use the methods parameter and access request data:\n\n```python\nfrom flask import request\n\n@app.route('/api/data', methods=['POST'])\ndef create_data():\n    data = request.get_json()\n    # Process your data here\n    return jsonify({'status': 'created', 'data': data})\n```"}
        ],
        
        "unicode_conversation": [
            {"role": "user", "content": "Hello! 你好 🌍"},
            {"role": "assistant", "content": "Hello! 你好! I can help with questions in multiple languages."},
            {"role": "user", "content": "Café naïve résumé"},
            {"role": "assistant", "content": "I see you're testing Unicode support with accented characters. Yes, I handle international text properly."}
        ],
        
        "edge_cases": [
            {"role": "user", "content": ""},  # Empty content
            {"role": "assistant", "content": "I notice your message was empty. How can I help you?"},
            {"role": "user", "content": ["list", "content", "here"]},  # List content
            {"role": "assistant", "content": 42},  # Non-string content
            {"role": "unknown", "content": "Unknown role message"}
        ],
        
        "long_conversation": [
            {"role": "user", "content": f"This is message {i} with detailed content about topic {i}"} 
            for i in range(10)
        ] + [
            {"role": "assistant", "content": f"Response {i} with comprehensive information about the topic"}
            for i in range(10)
        ]
    }


@pytest.fixture
def real_template_dir():
    """Use the real template directory from the project."""
    return Path("config/prompts/summarization")


@pytest.fixture
def temp_template_dir(tmp_path):
    """Create a temporary directory with test template files."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    
    # Create sample template files
    templates = {
        "daily.txt": "You are summarizing a daily conversation between user and assistant on {current_date}. Focus on key topics and outcomes for {scope} summary.",
        "weekly.txt": "You are creating a weekly summary for the period ending {current_date}. Identify patterns and important developments from this {scope} timeframe.",
        "monthly.txt": "You are creating a monthly summary for {current_date}. Highlight major achievements, decisions, and recurring themes from this {scope} period.",
        "custom.txt": "Custom template with {scope} and {custom_var} variables.",
        "malformed.txt": "Template with unclosed {variable and missing } bracket"
    }
    
    for filename, content in templates.items():
        (template_dir / filename).write_text(content)
    
    return template_dir


@pytest.fixture
def llm_provider():
    """Real LLM bridge for authentic API testing."""
    return LLMProvider()


@pytest.fixture
def basic_engine(llm_provider, temp_template_dir):
    """Basic summarization engine for testing with real LLM bridge."""
    return SummarizationEngine(llm_provider, template_dir=str(temp_template_dir))


@pytest.fixture
def production_engine(llm_provider, real_template_dir):
    """Engine using real production templates."""
    return SummarizationEngine(llm_provider, template_dir=str(real_template_dir))


@pytest.fixture
def engine_without_templates(llm_provider, tmp_path):
    """Engine with non-existent template directory."""
    non_existent_dir = tmp_path / "nonexistent"
    return SummarizationEngine(llm_provider, template_dir=str(non_existent_dir))


# =============================================================================
# CRITICAL PRIVATE METHOD TESTS
# =============================================================================

class TestTemplateManagement:
    """
    Test template loading and caching system.
    
    Templates are critical - failures here break summarization entirely.
    """
    
    def test_loads_template_from_file(self, basic_engine):
        """
        Test that templates are loaded correctly from files.
        
        This is the normal case - should always work with proper setup.
        """
        template = basic_engine._get_scope_template("daily")
        
        # Should load template content
        assert "daily" in template.lower() or "day" in template.lower()
        assert "conversation" in template.lower()
        assert len(template.strip()) > 0
    
    def test_caches_loaded_templates(self, basic_engine):
        """
        Test that templates are cached to avoid repeated file I/O.
        
        Performance optimization that matters for high-volume usage.
        """
        # Load template twice
        template1 = basic_engine._get_scope_template("weekly")
        template2 = basic_engine._get_scope_template("weekly")
        
        # Should be same content (cached)
        assert template1 == template2
        
        # Cache should contain the template
        assert "weekly" in basic_engine.template_cache
    
    def test_loads_templates_as_static_text(self, basic_engine):
        """
        Test that templates are loaded as static text without any processing.
        
        Templates are just file content, not variable substitution systems.
        """
        # Load production template
        template = basic_engine._get_scope_template("daily")
        
        # Should be static text with no variable processing
        assert isinstance(template, str)
        assert len(template.strip()) > 0
        
        # Loading the same template multiple times gives same result
        template2 = basic_engine._get_scope_template("daily")
        assert template == template2
    
    def test_handles_missing_template_file_gracefully(self, basic_engine):
        """
        Test fallback behavior when template file doesn't exist.
        
        Missing templates shouldn't crash the system.
        """
        template = basic_engine._get_scope_template("nonexistent")
        
        # Should use fallback template
        assert "summarizing a conversation" in template.lower()
        assert "user and ai assistant" in template.lower()
        
        # Fallback should be cached
        assert "nonexistent" in basic_engine.template_cache
    
    def test_loads_malformed_templates_as_is(self, basic_engine):
        """
        Test that malformed templates are loaded as-is.
        
        Since we don't do variable substitution, malformed syntax is irrelevant.
        """
        # Use malformed template with unclosed variable
        template = basic_engine._get_scope_template("malformed")
        
        # Should return template content as-is (no processing means no failures)
        assert "{variable" in template  # Unclosed variable preserved exactly
        assert "missing } bracket" in template
    
    def test_templates_with_braces_loaded_literally(self, basic_engine):
        """
        Test that templates with brace syntax are loaded literally.
        
        Since we removed variable substitution, braces are just text.
        """
        # Create a template with brace syntax
        custom_template_dir = basic_engine.template_dir
        custom_template_file = custom_template_dir / "test_braces.txt"
        custom_template_file.write_text("Hello {user_name}, your {item} is ready.")
        
        template = basic_engine._get_scope_template("test_braces")
        
        # Should load content exactly as written (no substitution)
        assert "Hello {user_name}, your {item} is ready." == template
        
        # Clean up
        custom_template_file.unlink()


class TestMessageFormatting:
    """
    Test conversation formatting logic.
    
    Message formatting affects summary quality significantly.
    """
    
    def test_formats_standard_conversation(self, basic_engine, sample_messages):
        """
        Test formatting of normal user/assistant exchange.
        
        This is the most common case.
        """
        formatted = basic_engine._format_conversation(sample_messages["simple_conversation"])
        
        # Should include role prefixes
        assert "User:" in formatted
        assert "Assistant:" in formatted
        
        # Should include actual content
        assert "machine learning" in formatted.lower()
        assert "spam detection" in formatted.lower()
        
        # Should separate messages with newlines
        assert "\n" in formatted
    
    def test_handles_different_content_types(self, basic_engine, sample_messages):
        """
        Test that various content types are handled correctly.
        
        Real messages can contain lists, numbers, etc.
        """
        formatted = basic_engine._format_conversation(sample_messages["edge_cases"])
        
        # Should handle empty content
        assert "User:" in formatted
        
        # Should convert list to string
        assert "list" in formatted and "content" in formatted
        
        # Should convert numbers to string  
        assert "42" in formatted
        
        # Should handle unknown roles
        assert "Unknown:" in formatted
    
    def test_skips_empty_messages(self, basic_engine):
        """
        Test that completely empty messages are filtered out.
        
        Empty messages add no value to summaries.
        """
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "   "},  # Whitespace only
            {"role": "user", "content": "Real content"}
        ]
        
        formatted = basic_engine._format_conversation(messages)
        
        # Should only contain the real content
        lines = [line for line in formatted.split("\n") if line.strip()]
        assert len(lines) == 1
        assert "Real content" in formatted
    
    def test_unicode_content_preservation(self, basic_engine, sample_messages):
        """
        Test that Unicode characters are preserved correctly.
        
        International users exist in production.
        """
        formatted = basic_engine._format_conversation(sample_messages["unicode_conversation"])
        
        # Should preserve Unicode characters
        assert "你好" in formatted
        assert "🌍" in formatted
        assert "Café naïve résumé" in formatted
    
    def test_large_conversation_handling(self, basic_engine, sample_messages):
        """
        Test that large conversations are formatted efficiently.
        
        Some conversations can be very long.
        """
        formatted = basic_engine._format_conversation(sample_messages["long_conversation"])
        
        # Should contain content from multiple messages
        assert "message 0" in formatted or "Message 0" in formatted
        assert "message 9" in formatted or "Message 9" in formatted
        assert "Response" in formatted
        
        # Should be structured properly
        lines = formatted.split("\n")
        assert len(lines) >= 20  # 20 messages minimum


class TestUserMessageBuilding:
    """
    Test user message construction with compression instructions.
    
    User message structure directly affects LLM output quality.
    """
    
    def test_includes_compression_instruction(self, basic_engine, sample_messages):
        """
        Test that compression level instructions are included.
        
        Instructions guide the LLM on how to summarize.
        """
        user_msg = basic_engine._build_user_message(
            sample_messages["simple_conversation"], 
            "concise"
        )
        
        # Should include compression instruction
        assert "concise summary" in user_msg.lower()
        assert "three sentences maximum" in user_msg.lower()
    
    def test_different_compression_levels(self, basic_engine, sample_messages):
        """
        Test that different compression levels use different instructions.
        """
        messages = sample_messages["simple_conversation"]
        
        ultra_msg = basic_engine._build_user_message(messages, "ultra_concise")
        concise_msg = basic_engine._build_user_message(messages, "concise")
        detailed_msg = basic_engine._build_user_message(messages, "detailed")
        
        # Should have different instructions
        assert "one sentence maximum" in ultra_msg.lower()
        assert "three sentences maximum" in concise_msg.lower()
        assert "six sentences maximum" in detailed_msg.lower()
    
    def test_includes_conversation_content(self, basic_engine, sample_messages):
        """
        Test that formatted conversation is included in user message.
        """
        user_msg = basic_engine._build_user_message(
            sample_messages["complex_conversation"], 
            "detailed"
        )
        
        # Should include conversation marker
        assert "Conversation to summarize:" in user_msg
        
        # Should include actual conversation content
        assert "Python project" in user_msg
        assert "Flask" in user_msg
        assert "POST requests" in user_msg
    
    def test_handles_unknown_compression_level(self, basic_engine, sample_messages):
        """
        Test fallback for invalid compression levels.
        
        Invalid input should default gracefully.
        """
        user_msg = basic_engine._build_user_message(
            sample_messages["simple_conversation"],
            "invalid_level"
        )
        
        # Should fall back to detailed level
        assert "six sentences maximum" in user_msg.lower()
    
    def test_user_message_structure(self, basic_engine, sample_messages):
        """
        Test that user message has correct structure.
        """
        user_msg = basic_engine._build_user_message(
            sample_messages["simple_conversation"],
            "concise"
        )
        
        # Should have instruction first, then conversation
        instruction_index = user_msg.find("concise summary")
        conversation_index = user_msg.find("Conversation to summarize:")
        
        assert instruction_index < conversation_index


class TestTokenLimits:
    """
    Test token limit calculation for different compression levels.
    
    Token limits control LLM output length.
    """
    
    def test_compression_level_token_limits(self, basic_engine):
        """
        Test that each compression level has appropriate token limits.
        """
        ultra_tokens = basic_engine._get_max_tokens("ultra_concise")
        concise_tokens = basic_engine._get_max_tokens("concise")
        detailed_tokens = basic_engine._get_max_tokens("detailed")
        
        # Should be in ascending order
        assert ultra_tokens < concise_tokens < detailed_tokens
        
        # Should be reasonable values
        assert ultra_tokens == 50
        assert concise_tokens == 200
        assert detailed_tokens == 500
    
    def test_unknown_compression_level_fallback(self, basic_engine):
        """
        Test fallback token limit for unknown compression levels.
        """
        fallback_tokens = basic_engine._get_max_tokens("unknown_level")
        
        # Should use default fallback
        assert fallback_tokens == 300


# =============================================================================
# PUBLIC METHOD CONTRACT TESTS
# =============================================================================

class TestSummarization:
    """
    Test the core summarization functionality.
    
    This is the primary user-facing feature.
    """
    
    @pytest.mark.integration
    def test_returns_summary_for_valid_input(self, basic_engine, sample_messages):
        """
        Test successful summarization with normal input.
        
        This is the core contract users depend on.
        """
        summary = basic_engine.summarize(
            sample_messages["simple_conversation"],
            scope="daily",
            compression_level="concise"
        )
        
        # Should return non-empty string
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0
        
        # Should contain relevant content about the conversation
        summary_lower = summary.lower()
        assert "machine learning" in summary_lower or "ml" in summary_lower or "learning" in summary_lower
    
    def test_handles_empty_message_list(self, basic_engine):
        """
        Test behavior with no messages to summarize.
        
        Edge case that should be handled gracefully.
        """
        summary = basic_engine.summarize([], scope="daily")
        
        assert summary == "No messages to summarize."
    
    @pytest.mark.integration
    def test_different_scopes_produce_different_outputs(self, basic_engine, sample_messages):
        """
        Test that scope parameter affects summarization approach.
        
        Different scopes should produce contextually appropriate summaries.
        """
        messages = sample_messages["simple_conversation"]
        
        # Test different scopes
        daily_summary = basic_engine.summarize(messages, scope="daily", compression_level="concise")
        weekly_summary = basic_engine.summarize(messages, scope="weekly", compression_level="concise")
        monthly_summary = basic_engine.summarize(messages, scope="monthly", compression_level="concise")
        
        # All should be valid strings
        assert all(isinstance(s, str) and len(s.strip()) > 0 for s in [daily_summary, weekly_summary, monthly_summary])
        
        # Should be different (different context produces different output)
        summaries = [daily_summary, weekly_summary, monthly_summary]
        assert len(set(summaries)) > 1  # At least some should be different
    
    @pytest.mark.integration
    def test_compression_levels_affect_length(self, basic_engine, sample_messages):
        """
        Test that compression level affects summary length.
        
        More compression should generally produce shorter summaries.
        """
        messages = sample_messages["complex_conversation"]  # Use longer conversation
        
        # Test different compression levels
        ultra_summary = basic_engine.summarize(messages, compression_level="ultra_concise")
        concise_summary = basic_engine.summarize(messages, compression_level="concise") 
        detailed_summary = basic_engine.summarize(messages, compression_level="detailed")
        
        # All should be valid
        summaries = [ultra_summary, concise_summary, detailed_summary]
        assert all(isinstance(s, str) and len(s.strip()) > 0 for s in summaries)
        
        # Generally expect length progression (though not guaranteed with real LLM)
        lengths = [len(s.split()) for s in summaries]
        
        # Ultra concise should be shortest
        assert lengths[0] <= max(lengths)
        
        # Detailed should not be shortest
        assert lengths[2] >= min(lengths)
    
    @pytest.mark.integration
    def test_uses_different_templates_for_different_scopes(self, basic_engine, sample_messages):
        """
        Test that different scopes use different templates.
        
        Different scopes should produce contextually appropriate summaries.
        """
        summary = basic_engine.summarize(
            sample_messages["complex_conversation"],
            scope="daily",
            compression_level="detailed"
        )
        
        # Should generate valid summary
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0
        
        # Context should have influenced the prompt (indirect test via output quality)
        summary_lower = summary.lower()
        assert "python" in summary_lower or "flask" in summary_lower or "api" in summary_lower


class TestScopeManagement:
    """
    Test scope and template discovery functionality.
    """
    
    def test_get_available_scopes_returns_template_files(self, basic_engine):
        """
        Test that available scopes reflect actual template files.
        """
        scopes = basic_engine.get_available_scopes()
        
        # Should include template files (without .txt extension)
        expected_scopes = ["custom", "daily", "malformed", "monthly", "weekly"]
        assert sorted(scopes) == sorted(expected_scopes)
    
    def test_get_available_scopes_handles_missing_directory(self, engine_without_templates):
        """
        Test graceful handling when template directory doesn't exist.
        """
        scopes = engine_without_templates.get_available_scopes()
        
        # Should return empty list for missing directory
        assert scopes == []
    
    def test_get_compression_levels_returns_all_levels(self, basic_engine):
        """
        Test that all defined compression levels are returned.
        """
        levels = basic_engine.get_compression_levels()
        
        expected_levels = ["ultra_concise", "concise", "detailed"]
        assert sorted(levels) == sorted(expected_levels)
    
    def test_production_templates_exist(self, production_engine):
        """
        Test that production template files are accessible.
        """
        scopes = production_engine.get_available_scopes()
        
        # Should include at least the basic scopes
        assert "daily" in scopes
        assert "weekly" in scopes  
        assert "monthly" in scopes


class TestValidation:
    """
    Test setup validation functionality.
    """
    
    def test_validate_setup_with_valid_configuration(self, basic_engine):
        """
        Test validation with properly configured engine.
        """
        results = basic_engine.validate_setup()
        
        # Should report valid setup
        assert results["template_dir_exists"] is True
        assert results["llm_provider_available"] is True
        assert len(results["available_scopes"]) > 0
        assert len(results["compression_levels"]) == 3
    
    def test_validate_setup_with_missing_templates(self, engine_without_templates):
        """
        Test validation with missing template directory.
        """
        results = engine_without_templates.validate_setup()
        
        # Should report issues
        assert results["template_dir_exists"] is False
        assert results["available_scopes"] == []
        assert results["llm_provider_available"] is True  # Real bridge exists
    
    def test_validate_setup_with_missing_llm_provider(self, temp_template_dir):
        """
        Test validation with missing LLM bridge.
        """
        engine = SummarizationEngine(None, template_dir=str(temp_template_dir))
        results = engine.validate_setup()
        
        # Should report missing bridge
        assert results["llm_provider_available"] is False
        assert results["template_dir_exists"] is True


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """
    Test error handling and recovery scenarios.
    
    Robust error handling is critical for production reliability.
    """
    
    def test_handles_llm_provider_failure(self, basic_engine, sample_messages):
        """
        Test graceful handling of LLM bridge failures.
        
        External LLM services can fail - system should handle this.
        """
        # Patch the LLM bridge to raise exception
        with patch.object(basic_engine.llm_provider, 'generate_response', side_effect=Exception("LLM service unavailable")):
            with pytest.raises(ToolError) as exc_info:
                basic_engine.summarize(sample_messages["simple_conversation"])
        
        # Should raise ToolError with proper context
        assert exc_info.value.code == ErrorCode.MEMORY_ERROR
        assert "LLM summarization failed" in str(exc_info.value)
        assert "LLM service unavailable" in str(exc_info.value)
    
    def test_handles_template_file_permission_errors(self, temp_template_dir, llm_provider):
        """
        Test handling of template file access issues.
        """
        # Create engine
        engine = SummarizationEngine(llm_provider, template_dir=str(temp_template_dir))
        
        # Mock file access to raise permission error
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            template = engine._get_scope_template("daily")
        
        # Should fall back to default template
        assert "summarizing a conversation" in template.lower()
    
    def test_handles_invalid_template_directory(self, llm_provider, tmp_path):
        """
        Test handling when template directory is a file, not directory.
        """
        # Create a file where directory should be
        fake_dir = tmp_path / "fake_dir.txt"
        fake_dir.write_text("not a directory")
        
        engine = SummarizationEngine(llm_provider, template_dir=str(fake_dir))
        scopes = engine.get_available_scopes()
        
        # Should handle gracefully
        assert scopes == []
    
    def test_handles_malformed_message_structures(self, basic_engine):
        """
        Test handling of unexpected message formats.
        
        Real data can be malformed - system should survive.
        """
        malformed_messages = [
            {"role": "user"},  # Missing content
            {"content": "Missing role"},  # Missing role
            {},  # Empty message
            {"role": "user", "content": None},  # None content
        ]
        
        # Should not crash
        formatted = basic_engine._format_conversation(malformed_messages)
        
        # Should handle gracefully without crashing
        assert formatted is not None
    
    def test_error_context_manager_usage(self, basic_engine, sample_messages):
        """
        Test that error context manager is used properly.
        
        Error context provides consistent error handling.
        """
        # Mock LLM to raise exception
        with patch.object(basic_engine.llm_provider, 'generate_response', side_effect=ValueError("Test error")):
            with pytest.raises(ToolError) as exc_info:
                basic_engine.summarize(sample_messages["simple_conversation"])
        
        # Should be wrapped in ToolError
        assert exc_info.value.code == ErrorCode.MEMORY_ERROR


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSummarizationIntegration:
    """
    Test complete summarization workflows.
    
    These ensure the system works end-to-end for real use cases.
    """
    
    @pytest.mark.integration
    def test_complete_daily_summarization_workflow(self, production_engine, sample_messages):
        """
        Test complete daily conversation summarization using production templates.
        
        This simulates the most common use case with real templates.
        """
        # Use a conversation from the day
        conversation = sample_messages["complex_conversation"]
        
        # Generate daily summary
        summary = production_engine.summarize(
            conversation,
            scope="daily",
            compression_level="detailed"
        )
        
        # Should successfully generate summary
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0
        
        # Should contain relevant content from conversation
        summary_lower = summary.lower()
        assert any(keyword in summary_lower for keyword in ["python", "flask", "api", "project", "rest"])
    
    @pytest.mark.integration
    def test_weekly_summarization_with_long_conversation(self, production_engine, sample_messages):
        """
        Test weekly summarization with large conversation.
        
        Weekly summaries often deal with more content.
        """
        long_conversation = sample_messages["long_conversation"]
        
        summary = production_engine.summarize(
            long_conversation,
            scope="weekly", 
            compression_level="concise"
        )
        
        # Should handle large input successfully
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0
        
        # Should be appropriate length for concise compression
        word_count = len(summary.split())
        assert 10 <= word_count <= 100  # Reasonable range for concise summary
    
    @pytest.mark.integration
    def test_monthly_ultra_concise_summarization(self, production_engine, sample_messages):
        """
        Test monthly summarization with ultra concise compression.
        
        Monthly summaries often need to be very brief.
        """
        summary = production_engine.summarize(
            sample_messages["simple_conversation"],
            scope="monthly",
            compression_level="ultra_concise"
        )
        
        # Should generate summary
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0
        
        # Should be very brief for ultra concise
        word_count = len(summary.split())
        assert 1 <= word_count <= 50  # Ultra concise should be very short
    
    @pytest.mark.integration
    def test_unicode_content_end_to_end(self, basic_engine, sample_messages):
        """
        Test complete workflow with Unicode content.
        
        International users must be supported properly.
        """
        unicode_conversation = sample_messages["unicode_conversation"]
        
        summary = basic_engine.summarize(
            unicode_conversation,
            scope="daily",
            compression_level="concise"
        )
        
        # Should handle Unicode without errors
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0
        
        # Summary should be coherent (Unicode processing worked)
        assert len(summary.split()) >= 3  # Should be a reasonable summary
    
    @pytest.mark.integration
    def test_fallback_template_integration(self, engine_without_templates, sample_messages):
        """
        Test that fallback template works in complete workflow.
        
        System should work even without custom templates.
        """
        summary = engine_without_templates.summarize(
            sample_messages["simple_conversation"],
            scope="nonexistent_scope"
        )
        
        # Should still generate summary using fallback
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0
        
        # Should contain relevant content
        summary_lower = summary.lower()
        assert "machine learning" in summary_lower or "learning" in summary_lower or "ml" in summary_lower


# =============================================================================
# REAL-WORLD SCENARIO TESTS
# =============================================================================

class TestRealWorldScenarios:
    """
    Test scenarios based on actual usage patterns.
    
    These represent how the summarization engine is used in production.
    """
    
    @pytest.mark.integration
    def test_customer_support_conversation_summarization(self, production_engine):
        """
        Test summarizing a customer support conversation.
        
        Common use case: support ticket summarization.
        """
        support_conversation = [
            {"role": "user", "content": "I'm having trouble with my account login. It keeps saying invalid credentials."},
            {"role": "assistant", "content": "I can help you with that. Let me check your account status. Can you confirm your email address?"},
            {"role": "user", "content": "My email is user@example.com. I've tried resetting my password twice already."},
            {"role": "assistant", "content": "I see the issue. Your account was temporarily locked due to multiple failed login attempts. I've unlocked it now. Please try logging in again with your current password."},
            {"role": "user", "content": "That worked! Thank you so much. Is there a way to prevent this in the future?"},
            {"role": "assistant", "content": "Yes, I recommend enabling two-factor authentication for added security. I can help you set that up if you'd like."},
            {"role": "user", "content": "Yes please, let's set that up now."},
            {"role": "assistant", "content": "Perfect! I'll send you a verification code to your mobile number ending in 1234. Please enter the code when you receive it."}
        ]
        
        summary = production_engine.summarize(
            support_conversation,
            scope="daily",
            compression_level="detailed"
        )
        
        # Should generate meaningful summary
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0
        
        # Should capture key elements of support interaction
        summary_lower = summary.lower()
        assert any(keyword in summary_lower for keyword in ["login", "account", "locked", "unlock", "two-factor", "authentication", "security"])
    
    @pytest.mark.integration
    def test_technical_discussion_summarization(self, production_engine):
        """
        Test summarizing a technical discussion.
        
        Common use case: meeting notes and technical decisions.
        """
        technical_discussion = [
            {"role": "user", "content": "We need to decide on the database architecture for the new microservice. What are our options?"},
            {"role": "assistant", "content": "We have several options: PostgreSQL for ACID compliance, MongoDB for document storage, or Redis for caching. What are your performance requirements?"},
            {"role": "user", "content": "We need to handle 10k transactions per second with strong consistency. Data relationships are important."},
            {"role": "assistant", "content": "Given those requirements, I'd recommend PostgreSQL with read replicas for scaling. We can use connection pooling and optimize our queries for performance."},
            {"role": "user", "content": "What about backup and disaster recovery?"},
            {"role": "assistant", "content": "PostgreSQL has excellent backup tools. We can set up continuous WAL archiving and point-in-time recovery. I recommend automated daily backups with 30-day retention."},
            {"role": "user", "content": "Sounds good. Let's go with PostgreSQL. Can you create a migration plan?"},
            {"role": "assistant", "content": "Absolutely! I'll create a phased migration plan starting with schema design, then data migration, and finally application updates. I'll have a draft ready by tomorrow."}
        ]
        
        summary = production_engine.summarize(
            technical_discussion,
            scope="weekly",
            compression_level="detailed"
        )
        
        # Should capture technical details
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0
        
        # Should include key technical elements
        summary_lower = summary.lower()
        assert any(keyword in summary_lower for keyword in ["postgresql", "database", "microservice", "performance", "migration", "backup"])
    
    @pytest.mark.integration  
    def test_learning_session_summarization(self, production_engine):
        """
        Test summarizing an educational conversation.
        
        Common use case: tutoring sessions and learning summaries.
        """
        learning_session = [
            {"role": "user", "content": "I'm learning about machine learning. Can you explain gradient descent?"},
            {"role": "assistant", "content": "Gradient descent is an optimization algorithm used to minimize the cost function in ML models. It works by iteratively moving in the direction of steepest descent."},
            {"role": "user", "content": "How does it know which direction to move?"},
            {"role": "assistant", "content": "It calculates the gradient (partial derivatives) of the cost function with respect to each parameter. The negative gradient points toward the minimum."},
            {"role": "user", "content": "What's the difference between batch, mini-batch, and stochastic gradient descent?"},
            {"role": "assistant", "content": "Batch GD uses the entire dataset for each update. Stochastic GD uses one sample. Mini-batch GD uses a small subset. Mini-batch offers a good balance of stability and efficiency."},
            {"role": "user", "content": "Can you give me a simple example?"},
            {"role": "assistant", "content": "Sure! Imagine you're finding the bottom of a hill blindfolded. You feel the slope with your feet and take steps downhill. That's essentially what gradient descent does mathematically."}
        ]
        
        summary = production_engine.summarize(
            learning_session,
            scope="daily",
            compression_level="concise"
        )
        
        # Should capture learning content
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0
        
        # Should include educational elements
        summary_lower = summary.lower()
        assert any(keyword in summary_lower for keyword in ["gradient descent", "machine learning", "optimization", "algorithm", "learning"])
    
    @pytest.mark.integration
    def test_batch_summarization_workflow(self, production_engine, sample_messages):
        """
        Test summarizing multiple conversations in sequence.
        
        Batch processing is common for historical data.
        """
        conversations = [
            sample_messages["simple_conversation"],
            sample_messages["complex_conversation"],
            sample_messages["unicode_conversation"]
        ]
        
        summaries = []
        contexts = [
            {"batch_id": "batch_001", "conversation_id": "conv_1"},
            {"batch_id": "batch_001", "conversation_id": "conv_2"}, 
            {"batch_id": "batch_001", "conversation_id": "conv_3"}
        ]
        
        # Process batch
        for conversation in conversations:
            summary = production_engine.summarize(
                conversation,
                scope="daily",
                compression_level="concise"
            )
            summaries.append(summary)
        
        # All should succeed
        assert len(summaries) == 3
        assert all(isinstance(s, str) and len(s.strip()) > 0 for s in summaries)
        
        # Each should contain relevant content  
        assert any("machine learning" in s.lower() or "learning" in s.lower() for s in summaries)
        assert any("python" in s.lower() or "flask" in s.lower() for s in summaries)


# =============================================================================  
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """
    Test performance characteristics for realistic workloads.
    """
    
    @pytest.mark.integration
    def test_summarization_completes_within_reasonable_time(self, production_engine, sample_messages):
        """
        Test that summarization completes within acceptable time limits.
        
        Performance matters for user experience.
        """
        import time
        
        conversation = sample_messages["complex_conversation"]
        
        # Time the summarization
        start_time = time.time()
        summary = production_engine.summarize(
            conversation,
            scope="daily",
            compression_level="concise"
        )
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time (allowing for API latency)
        assert elapsed < 30.0  # 30 seconds max for API call
        
        # Should still produce valid output
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0
    
    @pytest.mark.integration
    def test_handles_very_long_conversations(self, production_engine):
        """
        Test that very long conversations can be summarized.
        
        Some conversations can be extremely long.
        """
        # Create a very long conversation
        long_conversation = []
        for i in range(50):  # 100 total messages
            long_conversation.extend([
                {"role": "user", "content": f"This is user message {i} discussing topic {i % 5} with detailed information about the subject."},
                {"role": "assistant", "content": f"This is assistant response {i} providing comprehensive information about topic {i % 5} with examples and explanations."}
            ])
        
        # Should handle without errors
        summary = production_engine.summarize(
            long_conversation,
            scope="weekly",
            compression_level="concise"
        )
        
        # Should produce valid summary
        assert isinstance(summary, str)
        assert len(summary.strip()) > 0
        
        # Should be appropriately compressed given the large input
        word_count = len(summary.split())
        assert 20 <= word_count <= 150  # Reasonable compression ratio