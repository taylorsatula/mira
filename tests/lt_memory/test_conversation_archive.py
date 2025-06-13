"""
Production-grade tests for ConversationArchive.

Testing philosophy: TEST THE CONTRACT, NOT THE IMPLEMENTATION
Before every test: "What real production bug in OUR CODE would this catch?"

The ConversationArchive manages temporal storage and retrieval of conversation history.
We test the public contracts that users depend on: archiving daily conversations,
retrieving them by date/range, and generating progressive summaries.

Testing approach:
1. Use real PostgreSQL database (same engine as production)
2. Test public API contracts that users rely on
3. Test critical private methods that could fail in subtle ways
4. Focus on date handling, message filtering, and summary generation
5. Test error handling and edge cases
6. Verify database persistence and retrieval accuracy
"""

import pytest
from datetime import datetime, date, timedelta
from pathlib import Path

# Import the system under test
from lt_memory.managers.conversation_archive import ConversationArchive
from lt_memory.managers.memory_manager import MemoryManager
from lt_memory.models.base import ArchivedConversation
from config.config_manager import AppConfig
from utils.timezone_utils import utc_now, ensure_utc, format_utc_iso
from conversation import Message
from sqlalchemy import text


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def test_config():
    """
    Real configuration for testing with PostgreSQL test database.
    
    Uses actual config structure to catch configuration-related bugs.
    """
    config = AppConfig()
    
    # Override with test database
    config.memory.database_url = "postgresql://mira_admin@localhost:5432/lt_memory_test"
    config.memory.db_pool_size = 5
    config.memory.db_pool_max_overflow = 10
    config.memory.embedding_dim = 1024
    
    # Ensure paths exist
    config.paths.data_dir = "/tmp/test_lt_memory_cache"
    Path(config.paths.data_dir).mkdir(exist_ok=True)
    
    return config


@pytest.fixture
def clean_test_database(test_config):
    """
    Provides a clean test database for each test.
    
    Ensures tests don't interfere with each other by cleaning data, not schema.
    """
    from sqlalchemy import create_engine
    from lt_memory.models.base import Base
    
    def clean_database():
        """Helper function to clean all data from tables."""
        engine = create_engine(test_config.memory.database_url)
        
        # Just clean the data from tables if they exist
        with engine.connect() as conn:
            # Delete data in reverse dependency order
            for table in reversed(Base.metadata.sorted_tables):
                try:
                    conn.execute(table.delete())
                except Exception:
                    # Table might not exist yet - that's fine
                    pass
            conn.commit()
        
        engine.dispose()
    
    # Clean before test starts
    clean_database()
    
    yield test_config
    
    # Clean up data after test
    clean_database()


@pytest.fixture
def memory_manager(clean_test_database):
    """Real MemoryManager instance with clean database."""
    return MemoryManager(clean_test_database)


@pytest.fixture
def conversation_archive(memory_manager):
    """Real ConversationArchive instance."""
    return ConversationArchive(memory_manager)


@pytest.fixture
def real_conversation_with_messages():
    """Create a real conversation with actual Message objects for testing."""
    yesterday = utc_now() - timedelta(days=1)
    today = utc_now()
    
    # Create real Message objects using the actual class
    messages = [
        Message(role="user", content="Hello, what's the weather like?", created_at=yesterday),
        Message(role="assistant", content="The weather is sunny and 75°F today.", created_at=yesterday),
        Message(role="user", content="Great! Can you remind me about my meeting?", created_at=yesterday),
        Message(role="assistant", content="You have a team meeting at 2 PM today.", created_at=yesterday),
        Message(role="user", content="What time is it now?", created_at=today),
        Message(role="assistant", content="It's currently 10:30 AM.", created_at=today),
    ]
    
    # Create real conversation object
    class RealConversation:
        def __init__(self):
            self.messages = messages
    
    return RealConversation()


# =============================================================================
# CORE CONTRACT TESTS
# =============================================================================

class TestConversationArchiveCore:
    """
    Test the fundamental contracts of ConversationArchive.
    
    Focus on the public API promises that users depend on.
    """
    
    def test_archive_day_stores_and_retrieves_correctly(self, conversation_archive, real_conversation_with_messages):
        """
        Test the core contract: archive_day() stores conversation and can be retrieved.
        
        REAL BUG THIS CATCHES: If archive_day() has bugs in message filtering by date,
        database storage, or summary generation, daily conversations are lost or corrupted,
        breaking the entire conversation history system that users depend on.
        """
        # Archive yesterday's conversation
        target_date = (utc_now() - timedelta(days=1)).date()
        
        # Test the real archival process
        result = conversation_archive.archive_day(real_conversation_with_messages, target_date)
        
        # Verify the contract promises
        assert result["success"] is True
        assert result["date"] == target_date.isoformat()
        assert result["message_count"] == 4  # Only yesterday's messages
        assert "archive_id" in result
        assert result["original_messages"] == 6  # Total messages
        assert result["remaining_messages"] == 2  # Today's messages remain
        
        # Verify actual database persistence (the real contract)
        retrieved_conversation = conversation_archive.get_conversation_by_date(target_date)
        assert retrieved_conversation is not None
        assert retrieved_conversation["id"] == result["archive_id"]
        assert retrieved_conversation["message_count"] == 4
        assert len(retrieved_conversation["messages"]) == 4
        assert retrieved_conversation["summary"] is not None
        assert len(retrieved_conversation["summary"]) > 0
        
        # Verify messages are correctly filtered and stored
        stored_messages = retrieved_conversation["messages"]
        for msg in stored_messages:
            msg_date = datetime.fromisoformat(msg["created_at"].replace('Z', '+00:00')).date()
            assert msg_date == target_date
        
        # Verify live conversation was modified correctly
        assert len(real_conversation_with_messages.messages) == 2  # Only today's messages remain
        for msg in real_conversation_with_messages.messages:
            assert msg.created_at.date() != target_date

    def test_archive_day_handles_empty_conversation_gracefully(self, conversation_archive):
        """
        Test that archive_day handles conversations with no messages correctly.
        
        REAL BUG THIS CATCHES: If archive_day() crashes or behaves incorrectly when given
        an empty conversation, automated archival processes will fail, breaking the
        daily archival system that keeps conversation history manageable.
        """
        target_date = (utc_now() - timedelta(days=1)).date()
        
        # Create conversation with no messages
        class EmptyConversation:
            def __init__(self):
                self.messages = []
        
        empty_conversation = EmptyConversation()
        
        # Test archival of empty conversation
        result = conversation_archive.archive_day(empty_conversation, target_date)
        
        # Verify graceful handling
        assert result["success"] is False
        assert result["date"] == target_date.isoformat()
        assert result["message_count"] == 0
        assert "No messages found in conversation" in result["message"]
        
        # Verify no database entry was created
        retrieved_conversation = conversation_archive.get_conversation_by_date(target_date)
        assert retrieved_conversation is None
        
        # Verify conversation unchanged
        assert len(empty_conversation.messages) == 0

    def test_get_conversations_by_range_retrieves_multiple_days(self, conversation_archive):
        """
        Test that get_conversations_by_range retrieves all conversations in date range.
        
        REAL BUG THIS CATCHES: If get_conversations_by_range() has bugs in date filtering
        or SQL query construction, users get incomplete conversation history when browsing
        multiple days, breaking the web interface and historical context features.
        """
        # Create separate conversations for 3 different days
        base_date = utc_now() - timedelta(days=3)
        archived_dates = []
        
        for day_offset in range(3):
            current_date = base_date + timedelta(days=day_offset)
            
            # Create messages for this specific day only
            messages = [
                Message(role="user", content=f"Question from day {day_offset}", created_at=current_date),
                Message(role="assistant", content=f"Answer from day {day_offset}", created_at=current_date),
            ]
            
            class DayConversation:
                def __init__(self, messages):
                    self.messages = messages
            
            # Create a fresh conversation for each day
            conversation = DayConversation(messages)
            
            # Archive this day's conversation
            target_date = current_date.date()
            result = conversation_archive.archive_day(conversation, target_date)
            assert result["success"] is True
            archived_dates.append(target_date)
        
        # Test range retrieval
        start_date = base_date.date()
        end_date = (base_date + timedelta(days=2)).date()
        
        conversations = conversation_archive.get_conversations_by_range(start_date, end_date)
        
        # Verify all 3 conversations retrieved
        assert len(conversations) == 3
        
        # Verify they're in chronological order
        dates = [conv["date"] for conv in conversations]
        assert dates == sorted(dates)
        
        # Verify the date range matches what we actually archived
        expected_dates = [date.isoformat() for date in archived_dates]
        assert dates == expected_dates
        
        # Verify each conversation has correct content
        for i, conv in enumerate(conversations):
            assert conv["message_count"] == 2
            assert len(conv["messages"]) == 2
            assert f"day {i}" in conv["messages"][0]["content"]
            assert conv["summary"] is not None

    def test_get_week_conversations_retrieves_monday_to_sunday(self, conversation_archive):
        """
        Test that get_week_conversations retrieves all conversations from Monday to Sunday.
        
        REAL BUG THIS CATCHES: If get_week_conversations() has bugs in week boundary 
        calculation or delegates incorrectly to get_conversations_by_range(), users get 
        incomplete weekly conversation history, breaking weekly summaries and context.
        """
        # Create conversations for a specific week (use a known Monday)
        # June 2, 2025 is a Monday
        monday = date(2025, 6, 2)
        
        # Create conversations for each day of the week
        week_dates = []
        for day_offset in range(7):  # Monday through Sunday
            current_date = monday + timedelta(days=day_offset)
            week_dates.append(current_date)
            
            # Create messages for this day
            messages = [
                Message(role="user", content=f"Question for {current_date.strftime('%A')}", 
                       created_at=utc_now().replace(year=current_date.year, month=current_date.month, day=current_date.day)),
                Message(role="assistant", content=f"Answer for {current_date.strftime('%A')}", 
                       created_at=utc_now().replace(year=current_date.year, month=current_date.month, day=current_date.day)),
            ]
            
            class DayConversation:
                def __init__(self, messages):
                    self.messages = messages
            
            conversation = DayConversation(messages)
            
            # Archive this day's conversation
            result = conversation_archive.archive_day(conversation, current_date)
            assert result["success"] is True
        
        # Test with different dates within the week (should all return same 7 conversations)
        test_dates = [
            monday,           # Monday
            monday + timedelta(days=3),  # Thursday  
            monday + timedelta(days=6),  # Sunday
        ]
        
        for test_date in test_dates:
            week_conversations = conversation_archive.get_week_conversations(test_date)
            
            # Should get all 7 conversations
            assert len(week_conversations) == 7
            
            # Should be in chronological order (Monday to Sunday)
            retrieved_dates = [conv["date"] for conv in week_conversations]
            expected_dates = [date.isoformat() for date in week_dates]
            assert retrieved_dates == expected_dates
            
            # Verify content matches day of week
            for i, conv in enumerate(week_conversations):
                day_name = week_dates[i].strftime('%A')
                assert day_name in conv["messages"][0]["content"]
                assert day_name in conv["messages"][1]["content"]
                assert conv["message_count"] == 2

    def test_get_month_conversations_retrieves_full_month(self, conversation_archive):
        """
        Test that get_month_conversations retrieves all conversations for the entire month.
        
        REAL BUG THIS CATCHES: If get_month_conversations() has bugs in month boundary 
        calculation (especially December->January), users get incomplete monthly conversation 
        history, breaking monthly summaries and reports.
        """
        # Test with a specific month - use February 2025 (28 days, non-leap year)
        # and December 2024 (31 days, tests year transition)
        test_months = [
            (date(2025, 2, 15), "February 2025"),  # Mid-February
            (date(2024, 12, 25), "December 2024"), # December (tests year boundary)
        ]
        
        for target_date, month_name in test_months:
            # Calculate the actual month boundaries
            month_start = target_date.replace(day=1)
            if target_date.month == 12:
                month_end = target_date.replace(year=target_date.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                month_end = target_date.replace(month=target_date.month + 1, day=1) - timedelta(days=1)
            
            # Create conversations for first, middle, and last days of month
            test_days = [month_start, target_date, month_end]
            archived_dates = []
            
            for day in test_days:
                messages = [
                    Message(role="user", content=f"Question for {day} in {month_name}", 
                           created_at=utc_now().replace(year=day.year, month=day.month, day=day.day)),
                    Message(role="assistant", content=f"Answer for {day} in {month_name}", 
                           created_at=utc_now().replace(year=day.year, month=day.month, day=day.day)),
                ]
                
                class DayConversation:
                    def __init__(self, messages):
                        self.messages = messages
                
                conversation = DayConversation(messages)
                
                # Archive this day's conversation
                result = conversation_archive.archive_day(conversation, day)
                assert result["success"] is True
                archived_dates.append(day)
            
            # Test month retrieval with the target date
            month_conversations = conversation_archive.get_month_conversations(target_date)
            
            # Should get all 3 conversations we created
            assert len(month_conversations) == 3
            
            # Should be in chronological order
            retrieved_dates = [conv["date"] for conv in month_conversations]
            expected_dates = [day.isoformat() for day in sorted(archived_dates)]
            assert retrieved_dates == expected_dates
            
            # Verify content matches the month
            for conv in month_conversations:
                assert month_name in conv["messages"][0]["content"]
                assert month_name in conv["messages"][1]["content"]
                assert conv["message_count"] == 2

    def test_get_archive_stats_calculates_totals_correctly(self, conversation_archive):
        """
        Test that get_archive_stats calculates conversation and message totals correctly.
        
        REAL BUG THIS CATCHES: If get_archive_stats() has bugs in aggregation queries
        (like using scalar() instead of sum()), admin dashboards show wrong statistics,
        misleading users about system usage and storage requirements.
        """
        # Create conversations with different message counts to test aggregation
        test_data = [
            (date(2025, 7, 1), 3, "First conversation"),
            (date(2025, 7, 2), 5, "Second conversation"), 
            (date(2025, 7, 3), 2, "Third conversation"),
        ]
        
        total_expected_conversations = len(test_data)
        total_expected_messages = sum(msg_count for _, msg_count, _ in test_data)
        
        for target_date, message_count, description in test_data:
            # Create conversation with specific number of messages
            messages = []
            for i in range(message_count):
                messages.extend([
                    Message(role="user", content=f"{description} - message {i+1} user", 
                           created_at=utc_now().replace(year=target_date.year, month=target_date.month, day=target_date.day)),
                    Message(role="assistant", content=f"{description} - message {i+1} assistant", 
                           created_at=utc_now().replace(year=target_date.year, month=target_date.month, day=target_date.day)),
                ])
            
            class TestConversation:
                def __init__(self, messages):
                    self.messages = messages
            
            conversation = TestConversation(messages)
            
            # Archive this conversation
            result = conversation_archive.archive_day(conversation, target_date)
            assert result["success"] is True
            assert result["message_count"] == message_count * 2  # user + assistant pairs
        
        # Get archive statistics
        stats = conversation_archive.get_archive_stats()
        
        # Verify conversation count
        assert stats["total_archived_conversations"] == total_expected_conversations
        
        # Verify total message count (this will catch the aggregation bug)
        expected_total_messages = total_expected_messages * 2  # user + assistant pairs
        assert stats["total_archived_messages"] == expected_total_messages
        
        # Verify date range
        assert stats["oldest_archive"] == "2025-07-01"
        assert stats["newest_archive"] == "2025-07-03"
        
        # Verify all required fields are present
        required_fields = ["total_archived_conversations", "total_archived_messages", 
                          "oldest_archive", "newest_archive"]
        for field in required_fields:
            assert field in stats
            assert stats[field] is not None

    def test_get_recent_context_summaries_retrieves_correct_yesterday_content(self, conversation_archive):
        """
        Test that get_recent_context_summaries retrieves the actual content from yesterday.
        
        REAL BUG THIS CATCHES: If get_recent_context_summaries() has bugs in date calculation
        or retrieval logic, it could return the wrong day's summary or no summary, breaking
        working memory's ability to maintain conversation context across sessions.
        """
        # Set up specific dates with identifiable content
        today = date(2025, 8, 15)  # Thursday
        yesterday = date(2025, 8, 14)  # Wednesday  
        day_before = date(2025, 8, 13)  # Tuesday
        
        # Create conversations with distinct, identifiable content
        conversations_data = [
            (day_before, "Tuesday", "I worked on the database schema"),
            (yesterday, "Wednesday", "I fixed the authentication bug"), 
        ]
        
        for conv_date, day_name, unique_content in conversations_data:
            messages = [
                Message(role="user", content=f"What did I do on {day_name}?", 
                       created_at=utc_now().replace(year=conv_date.year, month=conv_date.month, day=conv_date.day)),
                Message(role="assistant", content=f"On {day_name}, {unique_content}.", 
                       created_at=utc_now().replace(year=conv_date.year, month=conv_date.month, day=conv_date.day)),
            ]
            
            class DayConversation:
                def __init__(self, messages):
                    self.messages = messages
            
            conversation = DayConversation(messages)
            result = conversation_archive.archive_day(conversation, conv_date)
            assert result["success"] is True
        
        # Test the actual contract: get context for "today"
        context = conversation_archive.get_recent_context_summaries(today)
        
        # Verify yesterday's summary contains the correct content
        assert context["yesterday"] is not None
        assert context["yesterday"]["date"] == yesterday.isoformat()
        
        # This is the key test: verify we got YESTERDAY's content, not some other day
        yesterday_summary = context["yesterday"]["summary"]
        assert "Wednesday" in yesterday_summary or "authentication bug" in yesterday_summary
        # Should NOT contain Tuesday's content
        assert "Tuesday" not in yesterday_summary
        assert "database schema" not in yesterday_summary

    def test_get_recent_context_summaries_handles_missing_data_gracefully(self, conversation_archive):
        """
        Test that get_recent_context_summaries handles missing data gracefully.
        
        REAL BUG THIS CATCHES: If get_recent_context_summaries() crashes when yesterday's
        conversation or weekly summaries are missing, working memory initialization fails,
        breaking the entire conversation system.
        """
        # Use a date where we haven't archived any conversations
        future_date = date(2025, 12, 25)
        
        # Get context for a date with no archived data
        context = conversation_archive.get_recent_context_summaries(future_date)
        
        # Should return proper structure with None values
        assert isinstance(context, dict)
        assert "yesterday" in context
        assert "weekly" in context
        assert context["yesterday"] is None
        assert context["weekly"] is None

    def test_generate_weekly_summary_synthesizes_diverse_daily_activities(self, conversation_archive):
        """
        Test that generate_weekly_summary creates meaningful synthesis from diverse daily activities.
        
        REAL BUG THIS CATCHES: If generate_weekly_summary() fails to properly aggregate diverse
        daily summaries or produces generic/meaningless output, users get low-quality weekly
        context that doesn't help with continuity, breaking the progressive summarization system.
        """
        # Create a week of diverse, realistic daily activities
        monday = date(2025, 9, 1)
        
        # Realistic diverse daily scenarios with different themes and complexities
        week_scenarios = [
            (0, "Monday", [
                "Started working on the new authentication system design",
                "Had a productive meeting with the security team about OAuth implementation", 
                "Identified three potential security vulnerabilities in the current system"
            ]),
            (1, "Tuesday", [
                "Debugged the payment processing pipeline that was failing for international customers",
                "Discovered the issue was in currency conversion rounding",
                "Fixed the bug and deployed the patch to production"
            ]),
            (2, "Wednesday", [
                "Reviewed code for the mobile app refactoring project",
                "Gave feedback on API endpoint design and suggested performance improvements",
                "Mentored junior developer on database optimization techniques"
            ]),
            (3, "Thursday", [
                "Attended the quarterly business review meeting",
                "Presented technical roadmap for Q4 including infrastructure scaling plans",
                "Discussed budget allocation for cloud services and team expansion"
            ]),
            (4, "Friday", [
                "Completed performance testing on the new caching layer",
                "Results showed 40% improvement in response times",
                "Documented findings and recommendations for production rollout"
            ])
        ]
        
        # Create conversations for each day with rich, diverse content
        for day_offset, day_name, activities in week_scenarios:
            current_date = monday + timedelta(days=day_offset)
            
            # Create multiple message exchanges to simulate real conversation
            messages = []
            for i, activity in enumerate(activities):
                messages.extend([
                    Message(role="user", content=f"Tell me about the {activity.split()[0].lower()} work I did on {day_name}", 
                           created_at=utc_now().replace(year=current_date.year, month=current_date.month, day=current_date.day)),
                    Message(role="assistant", content=f"On {day_name}, you {activity}.", 
                           created_at=utc_now().replace(year=current_date.year, month=current_date.month, day=current_date.day)),
                ])
            
            class DiverseConversation:
                def __init__(self, messages):
                    self.messages = messages
            
            conversation = DiverseConversation(messages)
            result = conversation_archive.archive_day(conversation, current_date)
            assert result["success"] is True
        
        # Generate weekly summary
        wednesday = monday + timedelta(days=2)
        weekly_summary = conversation_archive.generate_weekly_summary(wednesday)
        
        # Basic validation
        assert weekly_summary is not None
        assert isinstance(weekly_summary, str)
        assert len(weekly_summary) > 50  # Should be substantial
        
        # Test for meaningful synthesis - the summary should contain key themes
        key_themes = [
            "authentication", "security",  # Monday theme
            "payment", "bug", "international",  # Tuesday theme  
            "code review", "mobile", "mentoring",  # Wednesday theme
            "quarterly", "business", "roadmap",  # Thursday theme
            "performance", "testing", "caching"  # Friday theme
        ]
        
        # Count how many themes are represented in the summary
        themes_found = 0
        for theme in key_themes:
            if theme.lower() in weekly_summary.lower():
                themes_found += 1
        
        # A good weekly summary should capture multiple themes from the week
        assert themes_found >= 4, f"Weekly summary should capture multiple themes, only found {themes_found}: {weekly_summary}"
        
        # Test for synthesis quality - should not just be a list
        # Good summaries synthesize rather than just enumerate
        list_indicators = weekly_summary.lower().count("monday") + weekly_summary.lower().count("tuesday") + weekly_summary.lower().count("wednesday")
        
        # If it mentions many specific days, it might be just listing rather than synthesizing
        assert list_indicators <= 2, f"Weekly summary should synthesize, not just list daily activities: {weekly_summary}"
        
        # Test for technical content preservation - should maintain important details
        technical_terms = ["authentication", "OAuth", "payment", "API", "performance", "caching"]
        technical_found = sum(1 for term in technical_terms if term.lower() in weekly_summary.lower())
        assert technical_found >= 3, f"Weekly summary should preserve technical context: {weekly_summary}"

    def test_generate_weekly_summary_handles_missing_conversations_gracefully(self, conversation_archive):
        """
        Test that generate_weekly_summary handles weeks with no conversations gracefully.
        
        REAL BUG THIS CATCHES: If generate_weekly_summary() crashes when no conversations
        exist for a week, automated weekly summary generation fails, breaking batch
        processing and scheduled summarization tasks.
        """
        # Use a date range where no conversations exist
        empty_week_date = date(2025, 10, 15)
        
        # Try to generate weekly summary for empty week
        weekly_summary = conversation_archive.generate_weekly_summary(empty_week_date)
        
        # Should return None gracefully, not crash
        assert weekly_summary is None

    def test_generate_monthly_summary_synthesizes_month_long_project_evolution(self, conversation_archive):
        """
        Test that generate_monthly_summary creates meaningful synthesis from a month of evolving work.
        
        REAL BUG THIS CATCHES: If generate_monthly_summary() fails to synthesize daily summaries
        and weekly summaries into coherent monthly context, or produces generic output, users lose
        long-term project context essential for strategic planning and project continuity.
        """
        # Create a month-long project evolution: October 2025
        # Simulate a realistic software development project across 4 weeks
        october_2025 = date(2025, 10, 1)
        
        # Week 1: Project Planning & Setup (Oct 1-7)
        week1_scenarios = [
            (0, "Project kickoff meeting - defined requirements for new user dashboard"),
            (1, "Created initial database schema design for user analytics"),
            (2, "Set up development environment and CI/CD pipeline"),
            (3, "Designed API endpoints for user behavior tracking"),
            (4, "Initial frontend mockups and user experience flow"),
        ]
        
        # Week 2: Core Development (Oct 8-14) 
        week2_scenarios = [
            (7, "Implemented user authentication and session management"),
            (8, "Built core database models and migration scripts"),
            (9, "Developed REST API endpoints for user data collection"),
            (10, "Created frontend components for dashboard visualization"),
            (11, "Added real-time data streaming using WebSocket connections"),
        ]
        
        # Week 3: Integration & Testing (Oct 15-21)
        week3_scenarios = [
            (14, "Integrated frontend with backend APIs and resolved CORS issues"),
            (15, "Implemented comprehensive unit tests achieving 85% coverage"),
            (16, "Performance testing revealed bottlenecks in database queries"),
            (17, "Optimized database indexes and reduced API response time by 60%"),
            (18, "User acceptance testing with beta users and collected feedback"),
        ]
        
        # Week 4: Deployment & Polish (Oct 22-28)
        week4_scenarios = [
            (21, "Fixed critical security vulnerability in user data access controls"),
            (22, "Deployed to staging environment and conducted final integration tests"),
            (23, "Production deployment with blue-green deployment strategy"),
            (24, "Monitored system performance and resolved minor UI issues"),
            (25, "Documentation updates and team knowledge transfer session"),
        ]
        
        all_scenarios = week1_scenarios + week2_scenarios + week3_scenarios + week4_scenarios
        
        # Create daily conversations for the entire month
        for day_offset, activity in all_scenarios:
            current_date = october_2025 + timedelta(days=day_offset)
            
            messages = [
                Message(role="user", content=f"What was the main focus of my work on {current_date.strftime('%B %d')}?", 
                       created_at=utc_now().replace(year=current_date.year, month=current_date.month, day=current_date.day)),
                Message(role="assistant", content=f"On {current_date.strftime('%B %d')}, you {activity}.", 
                       created_at=utc_now().replace(year=current_date.year, month=current_date.month, day=current_date.day)),
            ]
            
            class ProjectDayConversation:
                def __init__(self, messages):
                    self.messages = messages
            
            conversation = ProjectDayConversation(messages)
            result = conversation_archive.archive_day(conversation, current_date)
            assert result["success"] is True
        
        # Generate weekly summaries for some weeks to test mixed summarization
        week2_date = october_2025 + timedelta(days=9)  # Oct 10
        week3_date = october_2025 + timedelta(days=16)  # Oct 17
        
        weekly_summary_1 = conversation_archive.generate_weekly_summary(week2_date)
        weekly_summary_2 = conversation_archive.generate_weekly_summary(week3_date)
        
        assert weekly_summary_1 is not None
        assert weekly_summary_2 is not None
        
        # Generate monthly summary
        monthly_summary = conversation_archive.generate_monthly_summary(october_2025)
        
        # Basic validation
        assert monthly_summary is not None
        assert isinstance(monthly_summary, str)
        assert len(monthly_summary) > 100  # Should be comprehensive
        
        # Test for project evolution themes across the month
        project_phases = [
            "planning", "requirements", "design", "schema",  # Week 1 themes
            "implementation", "authentication", "API", "frontend",  # Week 2 themes  
            "integration", "testing", "performance", "optimization",  # Week 3 themes
            "deployment", "security", "production", "monitoring"  # Week 4 themes
        ]
        
        phases_found = 0
        for phase in project_phases:
            if phase.lower() in monthly_summary.lower():
                phases_found += 1
        
        # A good monthly summary should capture the project evolution
        assert phases_found >= 8, f"Monthly summary should capture project evolution, only found {phases_found}: {monthly_summary}"
        
        # Test for high-level synthesis - should focus on outcomes and progress
        outcome_indicators = ["achieved", "completed", "improved", "implemented", "resolved", "optimized"]
        outcomes_found = sum(1 for indicator in outcome_indicators if indicator.lower() in monthly_summary.lower())
        assert outcomes_found >= 3, f"Monthly summary should highlight outcomes: {monthly_summary}"
        
        # Test for technical depth preservation
        technical_specifics = ["database", "API", "frontend", "authentication", "performance", "deployment"]
        technical_found = sum(1 for term in technical_specifics if term.lower() in monthly_summary.lower())
        assert technical_found >= 4, f"Monthly summary should preserve technical context: {monthly_summary}"
        
        # Test against over-detailed daily listing
        date_mentions = monthly_summary.lower().count("october")
        assert date_mentions <= 3, f"Monthly summary should synthesize, not list daily details: {monthly_summary}"
        
        # Verify monthly summary incorporates weekly summaries if present
        # Should contain higher-level insights than just daily summaries
        strategic_terms = ["project", "development", "progress", "milestone", "achievement"]
        strategic_found = sum(1 for term in strategic_terms if term.lower() in monthly_summary.lower())
        assert strategic_found >= 2, f"Monthly summary should include strategic perspective: {monthly_summary}"

    def test_generate_monthly_summary_handles_missing_data_gracefully(self, conversation_archive):
        """
        Test that generate_monthly_summary handles months with no conversations gracefully.
        
        REAL BUG THIS CATCHES: If generate_monthly_summary() crashes when no conversations
        exist for a month, automated monthly reporting fails, breaking long-term analytics
        and historical context generation.
        """
        # Use a future month with no data
        empty_month = date(2025, 12, 15)
        
        monthly_summary = conversation_archive.generate_monthly_summary(empty_month)
        
        # Should return None gracefully
        assert monthly_summary is None

    def test_get_weekly_summary_retrieves_stored_weekly_summary(self, conversation_archive):
        """
        Test that get_weekly_summary retrieves the correct weekly summary for a given date.
        
        REAL BUG THIS CATCHES: If get_weekly_summary() fails to retrieve the correct weekly
        summary or returns summaries from wrong dates, working memory loses weekly context,
        breaking conversation continuity and progressive summarization chains.
        """
        # Create a week of conversations and generate a weekly summary
        monday = date(2025, 11, 3)  # Using a unique week
        
        # Create distinct conversations for each day including Sunday
        week_activities = [
            (0, "Started working on machine learning model optimization"),
            (1, "Improved neural network architecture and hyperparameter tuning"),
            (2, "Conducted extensive training on larger dataset"),
            (3, "Evaluated model performance and achieved 95% accuracy"),
            (4, "Deployed optimized model to production environment"),
            (5, "Conducted weekend testing and performance analysis"),
            (6, "Reviewed week's progress and planned next week's goals"),  # Sunday
        ]
        
        for day_offset, activity in week_activities:
            current_date = monday + timedelta(days=day_offset)
            
            messages = [
                Message(role="user", content=f"What did I accomplish on {current_date.strftime('%A')}?", 
                       created_at=utc_now().replace(year=current_date.year, month=current_date.month, day=current_date.day)),
                Message(role="assistant", content=f"You {activity}.", 
                       created_at=utc_now().replace(year=current_date.year, month=current_date.month, day=current_date.day)),
            ]
            
            class WeeklyTestConversation:
                def __init__(self, messages):
                    self.messages = messages
            
            conversation = WeeklyTestConversation(messages)
            result = conversation_archive.archive_day(conversation, current_date)
            assert result["success"] is True
        
        # Generate weekly summary for this week
        wednesday = monday + timedelta(days=2)
        generated_summary = conversation_archive.generate_weekly_summary(wednesday)
        assert generated_summary is not None
        
        # Verify weekly summary is stored on Sunday and accessible from any day in the week
        sunday = monday + timedelta(days=6)
        
        # First verify Sunday has the weekly summary
        sunday_conversation = conversation_archive.get_conversation_by_date(sunday)
        assert sunday_conversation is not None
        assert sunday_conversation.get("weekly_summary") == generated_summary
        
        # Verify that any date in the week can retrieve the weekly summary
        for day_offset in range(7):  # Monday through Sunday
            test_date = monday + timedelta(days=day_offset)
            day_name = test_date.strftime('%A')
            
            retrieved_summary = conversation_archive.get_weekly_summary(test_date)
            assert retrieved_summary == generated_summary, f"Failed to retrieve weekly summary from {day_name}"
            
        # Verify non-Sunday conversations don't have weekly_summary field
        for day_offset in range(6):  # Monday through Saturday  
            test_date = monday + timedelta(days=day_offset)
            day_conversation = conversation_archive.get_conversation_by_date(test_date)
            assert day_conversation.get("weekly_summary") is None, f"{test_date.strftime('%A')} should not have weekly_summary"

    def test_get_weekly_summary_handles_missing_summary_gracefully(self, conversation_archive):
        """
        Test that get_weekly_summary returns None when no weekly summary exists.
        
        REAL BUG THIS CATCHES: If get_weekly_summary() crashes when no weekly summary
        exists for a date, working memory initialization fails when trying to get
        recent context, breaking conversation startup.
        """
        # Use a date where we have conversations but no weekly summary generated
        isolated_date = date(2025, 11, 20)
        
        # Create a single day's conversation but don't generate weekly summary
        messages = [
            Message(role="user", content="Quick question about the weather", 
                   created_at=utc_now().replace(year=isolated_date.year, month=isolated_date.month, day=isolated_date.day)),
            Message(role="assistant", content="It's sunny today.", 
                   created_at=utc_now().replace(year=isolated_date.year, month=isolated_date.month, day=isolated_date.day)),
        ]
        
        class IsolatedConversation:
            def __init__(self, messages):
                self.messages = messages
        
        conversation = IsolatedConversation(messages)
        result = conversation_archive.archive_day(conversation, isolated_date)
        assert result["success"] is True
        
        # Try to get weekly summary - should return None gracefully
        weekly_summary = conversation_archive.get_weekly_summary(isolated_date)
        assert weekly_summary is None
        
        # Test with a completely non-existent date
        future_date = date(2025, 12, 31)
        weekly_summary_future = conversation_archive.get_weekly_summary(future_date)
        assert weekly_summary_future is None

    def test_weekly_summary_storage_and_retrieval_debug(self, conversation_archive):
        """
        Debug test to understand weekly summary storage mechanism.
        
        REAL BUG THIS CATCHES: If weekly summary storage doesn't work correctly,
        the entire weekly summarization system is broken.
        """
        # Create a single day's conversation
        test_date = date(2025, 11, 25)
        
        messages = [
            Message(role="user", content="I completed the API integration project", 
                   created_at=utc_now().replace(year=test_date.year, month=test_date.month, day=test_date.day)),
            Message(role="assistant", content="Great work on completing the API integration!", 
                   created_at=utc_now().replace(year=test_date.year, month=test_date.month, day=test_date.day)),
        ]
        
        class SimpleConversation:
            def __init__(self, messages):
                self.messages = messages
        
        conversation = SimpleConversation(messages)
        result = conversation_archive.archive_day(conversation, test_date)
        assert result["success"] is True
        
        # Verify conversation was stored
        stored_conv = conversation_archive.get_conversation_by_date(test_date)
        assert stored_conv is not None
        assert stored_conv.get("weekly_summary") is None  # Should be None initially
        
        # Generate weekly summary for this date
        weekly_summary = conversation_archive.generate_weekly_summary(test_date)
        assert weekly_summary is not None
        
        # Now try to retrieve it
        retrieved_summary = conversation_archive.get_weekly_summary(test_date)
        
        # DEBUG: Let's see what we get
        print(f"Generated summary: {weekly_summary}")
        print(f"Retrieved summary: {retrieved_summary}")
        
        # Check if the conversation now has a weekly summary
        updated_conv = conversation_archive.get_conversation_by_date(test_date)
        print(f"Updated conversation weekly_summary: {updated_conv.get('weekly_summary') if updated_conv else 'No conversation'}")
        
        # This should work
        assert retrieved_summary is not None
        assert retrieved_summary == weekly_summary

    def test_get_monthly_summary_retrieves_stored_monthly_summary(self, conversation_archive):
        """
        Test that get_monthly_summary retrieves monthly summaries from the 1st of the month.
        
        REAL BUG THIS CATCHES: If get_monthly_summary() fails to retrieve monthly summaries
        from the correct date (1st of month) or has date calculation bugs, users lose 
        monthly context for long-term planning and historical review.
        """
        # Create conversations for a month and generate monthly summary
        march_2025 = date(2025, 3, 1)  # Start with March 1st
        
        # Create conversations for key days including the 1st
        month_activities = [
            (0, "March project kickoff and goal setting"),      # March 1st
            (14, "Mid-month progress review and adjustments"),  # March 15th  
            (30, "Month-end completion and retrospective"),     # March 31st
        ]
        
        for day_offset, activity in month_activities:
            current_date = march_2025 + timedelta(days=day_offset)
            
            messages = [
                Message(role="user", content=f"What was accomplished on {current_date.strftime('%B %d')}?", 
                       created_at=utc_now().replace(year=current_date.year, month=current_date.month, day=current_date.day)),
                Message(role="assistant", content=f"On {current_date.strftime('%B %d')}, you {activity}.", 
                       created_at=utc_now().replace(year=current_date.year, month=current_date.month, day=current_date.day)),
            ]
            
            class MonthlyTestConversation:
                def __init__(self, messages):
                    self.messages = messages
            
            conversation = MonthlyTestConversation(messages)
            result = conversation_archive.archive_day(conversation, current_date)
            assert result["success"] is True
        
        # Generate monthly summary (using any date in March)
        mid_march = date(2025, 3, 15)
        generated_summary = conversation_archive.generate_monthly_summary(mid_march)
        assert generated_summary is not None
        assert len(generated_summary) > 0
        
        # Verify monthly summary is stored on March 1st
        march_first_conversation = conversation_archive.get_conversation_by_date(march_2025)
        assert march_first_conversation is not None
        assert march_first_conversation.get("monthly_summary") == generated_summary
        
        # Test retrieval from different dates within March
        test_dates = [
            date(2025, 3, 1),   # March 1st (where it's stored)
            date(2025, 3, 15),  # March 15th (mid-month)
            date(2025, 3, 31),  # March 31st (end of month)
        ]
        
        for test_date in test_dates:
            retrieved_summary = conversation_archive.get_monthly_summary(test_date)
            assert retrieved_summary == generated_summary, f"Failed to retrieve monthly summary for {test_date}"
            assert "March" in retrieved_summary or "project" in retrieved_summary
        
        # Verify non-first-of-month conversations don't have monthly_summary
        for day_offset in [14, 30]:  # March 15th and 31st
            test_date = march_2025 + timedelta(days=day_offset)
            day_conversation = conversation_archive.get_conversation_by_date(test_date)
            assert day_conversation.get("monthly_summary") is None, f"{test_date} should not have monthly_summary"

    def test_get_monthly_summary_handles_missing_summary_gracefully(self, conversation_archive):
        """
        Test that get_monthly_summary returns None when no monthly summary exists.
        
        REAL BUG THIS CATCHES: If get_monthly_summary() crashes when no monthly summary
        exists, historical review and context retrieval fails, breaking long-term memory features.
        """
        # Use a month where no monthly summary was generated
        june_2025 = date(2025, 6, 15)
        
        # Create a conversation but don't generate monthly summary
        messages = [
            Message(role="user", content="Brief June conversation", 
                   created_at=utc_now().replace(year=2025, month=6, day=15)),
            Message(role="assistant", content="Brief response in June.", 
                   created_at=utc_now().replace(year=2025, month=6, day=15)),
        ]
        
        class BriefConversation:
            def __init__(self, messages):
                self.messages = messages
        
        conversation = BriefConversation(messages)
        result = conversation_archive.archive_day(conversation, june_2025)
        assert result["success"] is True
        
        # Try to get monthly summary - should return None gracefully
        monthly_summary = conversation_archive.get_monthly_summary(june_2025)
        assert monthly_summary is None
        
        # Test with completely non-existent month
        future_month = date(2026, 1, 15)
        future_summary = conversation_archive.get_monthly_summary(future_month)
        assert future_summary is None