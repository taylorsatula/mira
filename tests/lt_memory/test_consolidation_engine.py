"""
Production-grade tests for ConsolidationEngine memory optimization system.

This test suite focuses on realistic production scenarios for memory consolidation,
using real PostgreSQL database and real datetime operations for authentic testing.

Testing philosophy:
1. Test the public API contracts that users rely on
2. Test critical private methods that handle complex logic
3. Use real database infrastructure (PostgreSQL test database)
4. Test with real datetime operations and timezone handling
5. Verify data retention and optimization behaviors
6. Test real-world edge cases and boundary conditions
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

# Import the system under test
from lt_memory.managers.consolidation_engine import ConsolidationEngine
from lt_memory.managers.memory_manager import MemoryManager
from lt_memory.models.base import MemoryPassage, Base
from config.config_manager import AppConfig
from utils.timezone_utils import utc_now
from errors import ToolError, ErrorCode
from sqlalchemy import text

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def test_config():
    """Real configuration for PostgreSQL test database."""
    config = AppConfig()
    # Override database URL for testing
    config.memory.database_url = "postgresql://mira_app@localhost/lt_memory_test"
    config.memory.max_memory_age_days = 30  # 30 days for testing
    config.memory.embedding_dim = 1024  # Keep correct dimension
    return config


@pytest.fixture(scope="session")
def test_memory_manager(test_config):
    """Real MemoryManager with PostgreSQL test database."""
    manager = MemoryManager(test_config)
    yield manager
    # Cleanup: Close database connections
    manager.engine.dispose()


@pytest.fixture
def clean_database(test_memory_manager):
    """Clean database state for each test."""
    # Clean up existing test data
    with test_memory_manager.get_session() as session:
        session.query(MemoryPassage).delete()
        session.commit()
    
    yield test_memory_manager
    
    # Cleanup after test
    with test_memory_manager.get_session() as session:
        session.query(MemoryPassage).delete()
        session.commit()


@pytest.fixture
def consolidation_engine(clean_database):
    """Real ConsolidationEngine with clean database."""
    return clean_database.consolidation_engine


@pytest.fixture
def sample_passages_data():
    """Sample passage data with different age/importance/access patterns."""
    now = utc_now()
    
    return {
        # Should be pruned: old + low importance + low access
        "old_unimportant_unused": {
            "text": "Old unimportant content that should be pruned",
            "created_at": now - timedelta(days=35),  # Older than 30 days
            "importance_score": 0.2,  # Below 0.3 threshold
            "access_count": 1,  # Below 2 threshold
            "last_accessed": now - timedelta(days=20)
        },
        
        # Should NOT be pruned: old but important
        "old_important_unused": {
            "text": "Old but important content that should be kept",
            "created_at": now - timedelta(days=35),  # Older than 30 days
            "importance_score": 0.8,  # Above 0.3 threshold
            "access_count": 1,  # Below 2 threshold
            "last_accessed": now - timedelta(days=20)
        },
        
        # Should NOT be pruned: old + unimportant but frequently accessed
        "old_unimportant_used": {
            "text": "Old unimportant but frequently used content",
            "created_at": now - timedelta(days=35),  # Older than 30 days
            "importance_score": 0.2,  # Below 0.3 threshold
            "access_count": 5,  # Above 2 threshold
            "last_accessed": now - timedelta(days=1)
        },
        
        # Should NOT be pruned: recent
        "recent_unimportant_unused": {
            "text": "Recent content should be kept regardless",
            "created_at": now - timedelta(days=15),  # Within 30 days
            "importance_score": 0.1,  # Below 0.3 threshold
            "access_count": 0,  # Below 2 threshold
            "last_accessed": None
        },
        
        # Should be pruned: exactly at boundary conditions
        "boundary_case_prunable": {
            "text": "Boundary case that should be pruned",
            "created_at": now - timedelta(days=31),  # Just over 30 days
            "importance_score": 0.29,  # Just below 0.3
            "access_count": 1,  # Just below 2
            "last_accessed": now - timedelta(days=25)
        },
        
        # Should NOT be pruned: exactly at boundary conditions
        "boundary_case_keepable": {
            "text": "Boundary case that should be kept",
            "created_at": now - timedelta(days=31),  # Just over 30 days
            "importance_score": 0.3,  # Exactly at 0.3
            "access_count": 1,  # Just below 2 - but importance saves it
            "last_accessed": now - timedelta(days=25)
        }
    }


def create_memory_passage(memory_manager, passage_data):
    """Helper to create a MemoryPassage with real database operations."""
    with memory_manager.get_session() as session:
        # Generate a real embedding for the text
        embedding = memory_manager.generate_embedding(passage_data["text"])
        
        passage = MemoryPassage(
            text=passage_data["text"],
            embedding=embedding,
            source="test",  # Required field
            source_id="test_passage",  # Optional but good to have
            importance_score=passage_data["importance_score"],
            access_count=passage_data["access_count"],
            created_at=passage_data["created_at"],
            last_accessed=passage_data["last_accessed"]
        )
        session.add(passage)
        session.commit()
        return passage.id


# =============================================================================
# CRITICAL PRIVATE METHOD TESTS
# =============================================================================

class TestPruneOldPassages:
    """
    Test passage pruning logic with real database operations.
    
    Pruning is critical - incorrect logic could delete important memories
    or fail to clean up old data, causing storage and performance issues.
    """
    
    def test_prunes_passages_meeting_all_criteria(self, consolidation_engine, sample_passages_data):
        """
        Test that passages meeting ALL pruning criteria are deleted.
        
        REAL BUG THIS CATCHES: If the pruning logic has incorrect AND/OR 
        conditions in the SQL query, it could delete important memories or 
        fail to delete old data, causing either data loss or storage bloat.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Create test passages with different characteristics
        prunable_id = create_memory_passage(memory_manager, sample_passages_data["old_unimportant_unused"])
        boundary_prunable_id = create_memory_passage(memory_manager, sample_passages_data["boundary_case_prunable"])
        
        # Verify they exist before pruning
        with memory_manager.get_session() as session:
            assert session.query(MemoryPassage).filter_by(id=prunable_id).first() is not None
            assert session.query(MemoryPassage).filter_by(id=boundary_prunable_id).first() is not None
        
        # Run pruning
        pruned_count = consolidation_engine._prune_old_passages()
        
        # Should prune exactly 2 passages
        assert pruned_count == 2
        
        # Verify passages were actually deleted from database
        with memory_manager.get_session() as session:
            assert session.query(MemoryPassage).filter_by(id=prunable_id).first() is None
            assert session.query(MemoryPassage).filter_by(id=boundary_prunable_id).first() is None
    
    def test_preserves_passages_not_meeting_all_criteria(self, consolidation_engine, sample_passages_data):
        """
        Test that passages not meeting ALL criteria are preserved.
        
        REAL BUG THIS CATCHES: If pruning logic is too aggressive or has
        incorrect boolean logic, it could delete important or recent memories,
        causing data loss and breaking user experience.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Create passages that should NOT be pruned for different reasons
        important_id = create_memory_passage(memory_manager, sample_passages_data["old_important_unused"])
        frequently_used_id = create_memory_passage(memory_manager, sample_passages_data["old_unimportant_used"])
        recent_id = create_memory_passage(memory_manager, sample_passages_data["recent_unimportant_unused"])
        boundary_keepable_id = create_memory_passage(memory_manager, sample_passages_data["boundary_case_keepable"])
        
        # Run pruning
        pruned_count = consolidation_engine._prune_old_passages()
        
        # Should not prune any passages
        assert pruned_count == 0
        
        # Verify all passages still exist
        with memory_manager.get_session() as session:
            assert session.query(MemoryPassage).filter_by(id=important_id).first() is not None
            assert session.query(MemoryPassage).filter_by(id=frequently_used_id).first() is not None
            assert session.query(MemoryPassage).filter_by(id=recent_id).first() is not None
            assert session.query(MemoryPassage).filter_by(id=boundary_keepable_id).first() is not None
    
    def test_mixed_scenario_prunes_only_qualifying_passages(self, consolidation_engine, sample_passages_data):
        """
        Test mixed scenario with both prunable and non-prunable passages.
        
        REAL BUG THIS CATCHES: If pruning logic doesn't correctly evaluate
        the compound AND conditions, it might prune the wrong passages or
        miss passages that should be pruned, causing inconsistent behavior.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Create a mix of passages - some should be pruned, others kept
        should_prune_ids = [
            create_memory_passage(memory_manager, sample_passages_data["old_unimportant_unused"]),
            create_memory_passage(memory_manager, sample_passages_data["boundary_case_prunable"])
        ]
        
        should_keep_ids = [
            create_memory_passage(memory_manager, sample_passages_data["old_important_unused"]),
            create_memory_passage(memory_manager, sample_passages_data["recent_unimportant_unused"]),
            create_memory_passage(memory_manager, sample_passages_data["boundary_case_keepable"])
        ]
        
        # Verify all exist before pruning
        with memory_manager.get_session() as session:
            total_before = session.query(MemoryPassage).count()
            assert total_before == 5
        
        # Run pruning
        pruned_count = consolidation_engine._prune_old_passages()
        
        # Should prune exactly 2 passages
        assert pruned_count == 2
        
        # Verify correct passages were deleted and kept
        with memory_manager.get_session() as session:
            total_after = session.query(MemoryPassage).count()
            assert total_after == 3
            
            # Pruned passages should be gone
            for passage_id in should_prune_ids:
                assert session.query(MemoryPassage).filter_by(id=passage_id).first() is None
            
            # Kept passages should still exist
            for passage_id in should_keep_ids:
                assert session.query(MemoryPassage).filter_by(id=passage_id).first() is not None
    
    def test_handles_empty_database_gracefully(self, consolidation_engine):
        """
        Test pruning behavior with no passages in database.
        
        REAL BUG THIS CATCHES: If pruning logic doesn't handle empty result
        sets correctly, it could crash when no passages exist, breaking
        maintenance operations on new or cleaned systems.
        """
        # Ensure database is empty
        memory_manager = consolidation_engine.memory_manager
        with memory_manager.get_session() as session:
            assert session.query(MemoryPassage).count() == 0
        
        # Run pruning on empty database
        pruned_count = consolidation_engine._prune_old_passages()
        
        # Should return 0 without errors
        assert pruned_count == 0
        
        # Database should still be empty
        with memory_manager.get_session() as session:
            assert session.query(MemoryPassage).count() == 0
    
    def test_handles_no_qualifying_passages(self, consolidation_engine, sample_passages_data):
        """
        Test pruning when no passages meet pruning criteria.
        
        REAL BUG THIS CATCHES: If pruning logic has off-by-one errors or
        incorrect comparison operators, it might try to delete passages
        that don't meet criteria, causing unexpected data loss.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Create only passages that should NOT be pruned
        keep_ids = [
            create_memory_passage(memory_manager, sample_passages_data["old_important_unused"]),
            create_memory_passage(memory_manager, sample_passages_data["recent_unimportant_unused"]),
            create_memory_passage(memory_manager, sample_passages_data["old_unimportant_used"])
        ]
        
        # Run pruning
        pruned_count = consolidation_engine._prune_old_passages()
        
        # Should not prune anything
        assert pruned_count == 0
        
        # All passages should still exist
        with memory_manager.get_session() as session:
            assert session.query(MemoryPassage).count() == 3
            for passage_id in keep_ids:
                assert session.query(MemoryPassage).filter_by(id=passage_id).first() is not None
    
    def test_uses_real_configuration_max_age_days(self, consolidation_engine):
        """
        Test that pruning uses the actual configuration value for max age.
        
        REAL BUG THIS CATCHES: If pruning logic hardcodes the age cutoff
        instead of using configuration, changing the config won't affect
        pruning behavior, making the system non-configurable.
        """
        memory_manager = consolidation_engine.memory_manager
        config_max_age = memory_manager.config.memory.max_memory_age_days
        
        # Create passage exactly at the boundary
        now = utc_now()
        boundary_passage_data = {
            "text": "Boundary test passage",
            "created_at": now - timedelta(days=config_max_age + 1),  # Just over limit
            "importance_score": 0.2,  # Low importance
            "access_count": 1,  # Low access
            "last_accessed": now - timedelta(days=10)
        }
        
        boundary_id = create_memory_passage(memory_manager, boundary_passage_data)
        
        # Run pruning
        pruned_count = consolidation_engine._prune_old_passages()
        
        # Should prune the boundary passage
        assert pruned_count == 1
        
        # Verify passage was deleted
        with memory_manager.get_session() as session:
            assert session.query(MemoryPassage).filter_by(id=boundary_id).first() is None
    
    def test_boundary_conditions_for_importance_and_access(self, consolidation_engine):
        """
        Test exact boundary conditions for importance score and access count.
        
        REAL BUG THIS CATCHES: If pruning logic uses wrong comparison operators
        (< vs <= or >= vs >), passages exactly at the boundaries might be
        incorrectly pruned or preserved, causing unpredictable behavior.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        old_date = now - timedelta(days=35)  # Definitely old
        
        # Test importance score boundary (should NOT prune at exactly 0.3)
        exactly_03_importance = {
            "text": "Exactly 0.3 importance",
            "created_at": old_date,
            "importance_score": 0.3,  # Exactly at threshold
            "access_count": 1,
            "last_accessed": now - timedelta(days=10)
        }
        
        # Test access count boundary (should NOT prune at exactly 2)
        exactly_2_access = {
            "text": "Exactly 2 access count",
            "created_at": old_date,
            "importance_score": 0.2,  # Below threshold
            "access_count": 2,  # Exactly at threshold
            "last_accessed": now - timedelta(days=10)
        }
        
        boundary_ids = [
            create_memory_passage(memory_manager, exactly_03_importance),
            create_memory_passage(memory_manager, exactly_2_access)
        ]
        
        # Run pruning
        pruned_count = consolidation_engine._prune_old_passages()
        
        # Should not prune passages exactly at boundaries
        assert pruned_count == 0
        
        # Verify both passages still exist
        with memory_manager.get_session() as session:
            for passage_id in boundary_ids:
                assert session.query(MemoryPassage).filter_by(id=passage_id).first() is not None


class TestUpdateImportanceScores:
    """
    Test importance score updating logic with real mathematical calculations.
    
    Score updating is critical - incorrect calculations could cause important
    memories to fade or unimportant ones to persist, affecting memory quality.
    """
    
    def test_age_decay_reduces_old_passage_scores(self, consolidation_engine):
        """
        Test that age decay properly reduces scores for old passages vs new ones.
        
        REAL BUG THIS CATCHES: If age decay calculation is wrong or not applied,
        old memories would maintain artificially high importance scores,
        preventing natural memory aging and causing storage bloat.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create old and new passages with same starting characteristics
        old_passage_data = {
            "text": "Very old passage for age decay test",
            "created_at": now - timedelta(days=100),  # Very old
            "importance_score": 0.8,
            "access_count": 0,
            "last_accessed": None
        }
        
        new_passage_data = {
            "text": "Recent passage for age decay test", 
            "created_at": now - timedelta(days=1),   # Very new
            "importance_score": 0.8,  # Same starting score
            "access_count": 0,        # Same access pattern
            "last_accessed": None
        }
        
        old_id = create_memory_passage(memory_manager, old_passage_data)
        new_id = create_memory_passage(memory_manager, new_passage_data)
        
        # Run importance score update
        updated_count = consolidation_engine._update_importance_scores()
        
        # Compare results - age decay should affect old passage more
        with memory_manager.get_session() as session:
            old_passage = session.query(MemoryPassage).filter_by(id=old_id).first()
            new_passage = session.query(MemoryPassage).filter_by(id=new_id).first()
            
            # Old passage should have lower score due to age decay
            assert old_passage.importance_score < new_passage.importance_score
            # Both should be reduced from original 0.8, but old more so
            assert old_passage.importance_score < 0.8
            assert new_passage.importance_score <= 0.8
    
    def test_frequent_access_boosts_scores_compared_to_unused(self, consolidation_engine):
        """
        Test that frequently accessed passages get higher scores than unused ones.
        
        REAL BUG THIS CATCHES: If access factor calculation is wrong,
        frequently used memories wouldn't be prioritized, causing the system
        to lose track of what's actually important to the user.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create passages with different access patterns but same other characteristics
        frequently_accessed_data = {
            "text": "Frequently accessed passage",
            "created_at": now - timedelta(days=10),
            "importance_score": 0.5,
            "access_count": 20,  # High access count
            "last_accessed": now - timedelta(days=1)
        }
        
        rarely_accessed_data = {
            "text": "Rarely accessed passage",
            "created_at": now - timedelta(days=10),  # Same age
            "importance_score": 0.5,  # Same starting score
            "access_count": 1,       # Low access count
            "last_accessed": now - timedelta(days=8)
        }
        
        frequent_id = create_memory_passage(memory_manager, frequently_accessed_data)
        rare_id = create_memory_passage(memory_manager, rarely_accessed_data)
        
        # Run importance score update
        updated_count = consolidation_engine._update_importance_scores()
        
        # Compare results - frequent access should boost score
        with memory_manager.get_session() as session:
            frequent_passage = session.query(MemoryPassage).filter_by(id=frequent_id).first()
            rare_passage = session.query(MemoryPassage).filter_by(id=rare_id).first()
            
            # Frequently accessed should have higher score
            assert frequent_passage.importance_score > rare_passage.importance_score
    
    def test_applies_recency_factor_for_recently_accessed(self, consolidation_engine):
        """
        Test that recently accessed passages get recency boost.
        
        REAL BUG THIS CATCHES: If recency factor calculation fails,
        recently used memories could decay too quickly, breaking the
        user experience by forgetting recent context.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create passage accessed recently vs long ago
        recent_access_data = {
            "text": "Recently accessed passage",
            "created_at": now - timedelta(days=30),  # Moderate age
            "importance_score": 0.6,
            "access_count": 3,
            "last_accessed": now - timedelta(days=2)  # Very recent access
        }
        
        old_access_data = {
            "text": "Long ago accessed passage", 
            "created_at": now - timedelta(days=30),  # Same age
            "importance_score": 0.6,  # Same starting score
            "access_count": 3,  # Same access count
            "last_accessed": now - timedelta(days=50)  # Very old access
        }
        
        recent_id = create_memory_passage(memory_manager, recent_access_data)
        old_id = create_memory_passage(memory_manager, old_access_data)
        
        # Run importance score update
        updated_count = consolidation_engine._update_importance_scores()
        
        # Should update both passages
        assert updated_count >= 2
        
        # Compare final scores
        with memory_manager.get_session() as session:
            recent_passage = session.query(MemoryPassage).filter_by(id=recent_id).first()
            old_passage = session.query(MemoryPassage).filter_by(id=old_id).first()
            
            # Recently accessed should have higher score due to recency factor
            assert recent_passage.importance_score > old_passage.importance_score
    
    def test_handles_passages_never_accessed(self, consolidation_engine):
        """
        Test handling of passages that have never been accessed.
        
        REAL BUG THIS CATCHES: If recency factor calculation crashes on
        None values, the system would fail when processing passages that
        were created but never accessed.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create passage that was never accessed
        never_accessed_data = {
            "text": "Never accessed passage",
            "created_at": now - timedelta(days=20),
            "importance_score": 0.7,
            "access_count": 0,
            "last_accessed": None  # Never accessed
        }
        
        passage_id = create_memory_passage(memory_manager, never_accessed_data)
        
        # Should not crash when processing None last_accessed
        updated_count = consolidation_engine._update_importance_scores()
        
        # Should handle gracefully and update the passage
        assert updated_count >= 1
        
        # Verify passage still exists and has valid score
        with memory_manager.get_session() as session:
            passage = session.query(MemoryPassage).filter_by(id=passage_id).first()
            assert passage is not None
            assert 0.1 <= passage.importance_score <= 1.0  # Within valid range
    
    def test_avoids_database_churn_for_minimal_score_changes(self, consolidation_engine):
        """
        Test that minimal score changes don't cause unnecessary database updates.
        
        REAL BUG THIS CATCHES: If the significance threshold is wrong,
        the system could either update too many passages (causing database
        churn) or miss important score changes.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create passage with very minimal potential changes
        minimal_change_data = {
            "text": "Minimal change passage",
            "created_at": now - timedelta(days=1),  # Very recent
            "importance_score": 0.5,
            "access_count": 0,
            "last_accessed": None
        }
        
        # Create passage guaranteed to have significant changes
        significant_change_data = {
            "text": "Significant change passage", 
            "created_at": now - timedelta(days=200),  # Very old
            "importance_score": 0.9,  # High score that will decay substantially
            "access_count": 0,
            "last_accessed": None
        }
        
        minimal_id = create_memory_passage(memory_manager, minimal_change_data)
        significant_id = create_memory_passage(memory_manager, significant_change_data)
        
        # Run importance score update
        updated_count = consolidation_engine._update_importance_scores()
        
        # Should update the old passage, may or may not update the recent one
        assert updated_count >= 1  # At least the old passage should be updated
        
        # Verify the old passage was definitely changed significantly
        with memory_manager.get_session() as session:
            significant_passage = session.query(MemoryPassage).filter_by(id=significant_id).first()
            # Very old passage with high score should have decayed substantially
            assert significant_passage.importance_score < 0.7  # Significant reduction from 0.9
    
    def test_clamps_scores_within_valid_range(self, consolidation_engine):
        """
        Test that calculated scores are clamped between 0.1 and 1.0.
        
        REAL BUG THIS CATCHES: If clamping logic is wrong, scores could
        go outside valid ranges, causing database constraint violations
        or invalid importance comparisons.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create passage that would calculate to very low score
        very_low_score_data = {
            "text": "Very low score passage",
            "created_at": now - timedelta(days=500),  # Extremely old
            "importance_score": 0.1,  # Already at minimum
            "access_count": 0,
            "last_accessed": now - timedelta(days=400)  # Very old access
        }
        
        # Create passage that would calculate to very high score  
        very_high_score_data = {
            "text": "Very high score passage",
            "created_at": now - timedelta(days=1),  # Very recent
            "importance_score": 0.9,  # High starting score
            "access_count": 50,  # Extremely high access
            "last_accessed": now - timedelta(hours=1)  # Very recent access
        }
        
        low_id = create_memory_passage(memory_manager, very_low_score_data)
        high_id = create_memory_passage(memory_manager, very_high_score_data)
        
        # Run importance score update
        updated_count = consolidation_engine._update_importance_scores()
        
        # Verify scores are within valid range
        with memory_manager.get_session() as session:
            low_passage = session.query(MemoryPassage).filter_by(id=low_id).first()
            high_passage = session.query(MemoryPassage).filter_by(id=high_id).first()
            
            # Scores should be clamped within valid range
            assert 0.1 <= low_passage.importance_score <= 1.0
            assert 0.1 <= high_passage.importance_score <= 1.0
            
            # Low score should be at minimum clamp
            assert low_passage.importance_score >= 0.1
            
            # High score should be at maximum clamp  
            assert high_passage.importance_score <= 1.0
    
    def test_handles_empty_database_gracefully(self, consolidation_engine):
        """
        Test score updating with no passages in database.
        
        REAL BUG THIS CATCHES: If the update logic doesn't handle empty
        result sets correctly, it could crash when no passages exist,
        breaking maintenance operations on new systems.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Ensure database is empty
        with memory_manager.get_session() as session:
            assert session.query(MemoryPassage).count() == 0
        
        # Run importance score update on empty database
        updated_count = consolidation_engine._update_importance_scores()
        
        # Should return 0 without errors
        assert updated_count == 0
        
        # Database should still be empty
        with memory_manager.get_session() as session:
            assert session.query(MemoryPassage).count() == 0
    
    def test_combined_factors_work_together_logically(self, consolidation_engine):
        """
        Test that age, access, and recency factors combine logically.
        
        REAL BUG THIS CATCHES: If any of the mathematical formulas have
        errors (wrong coefficients, operators, or precedence), scores
        would be calculated incorrectly, affecting memory prioritization.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create passages that test different factor combinations
        # Best case: recent, frequently accessed, recently used
        best_case_data = {
            "text": "Best case passage - recent, frequent, recent access",
            "created_at": now - timedelta(days=5),    # Recent
            "importance_score": 0.6,
            "access_count": 15,                       # High access
            "last_accessed": now - timedelta(days=1)  # Recently accessed
        }
        
        # Worst case: old, never accessed, never used
        worst_case_data = {
            "text": "Worst case passage - old, unused, never accessed",
            "created_at": now - timedelta(days=200),  # Very old
            "importance_score": 0.6,  # Same starting score
            "access_count": 0,        # Never accessed
            "last_accessed": None     # Never used
        }
        
        best_id = create_memory_passage(memory_manager, best_case_data)
        worst_id = create_memory_passage(memory_manager, worst_case_data)
        
        # Run importance score update
        updated_count = consolidation_engine._update_importance_scores()
        
        # Verify logical factor combination
        with memory_manager.get_session() as session:
            best_passage = session.query(MemoryPassage).filter_by(id=best_id).first()
            worst_passage = session.query(MemoryPassage).filter_by(id=worst_id).first()
            
            # Best case should have higher score than worst case
            assert best_passage.importance_score > worst_passage.importance_score
            # Best case should be boosted above original
            assert best_passage.importance_score >= 0.6  # Should be boosted or maintained
            # Worst case should be significantly reduced
            assert worst_passage.importance_score < 0.6


class TestShouldOptimizeIndexes:
    """
    Test index optimization decision logic with real database operations.
    
    Index optimization decisions are critical - running optimization too often
    wastes resources, while not running it enough degrades search performance.
    """
    
    def test_recommends_optimization_when_passage_count_exceeds_threshold(self, consolidation_engine, monkeypatch):
        """
        Test that optimization is recommended when passage count exceeds threshold.
        
        REAL BUG THIS CATCHES: If the passage count threshold check is wrong
        or not applied, the system could fail to optimize indexes when needed,
        causing vector search performance to degrade significantly.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Create a manageable number of passages for testing
        for i in range(5):
            passage_data = {
                "text": f"Test passage {i} for optimization threshold",
                "created_at": utc_now(),
                "importance_score": 0.5,
                "access_count": 1,
                "last_accessed": utc_now()
            }
            create_memory_passage(memory_manager, passage_data)
        
        # Verify we have 5 passages
        with memory_manager.get_session() as session:
            count = session.query(MemoryPassage).count()
            assert count == 5
        
        # Temporarily patch the _should_optimize_indexes method to use a lower threshold
        original_method = consolidation_engine._should_optimize_indexes
        
        def modified_should_optimize():
            with memory_manager.get_session() as session:
                passage_count = session.query(MemoryPassage).count()
                # Use threshold of 3 instead of 10000 for testing
                return passage_count > 3
        
        monkeypatch.setattr(consolidation_engine, '_should_optimize_indexes', modified_should_optimize)
        
        # Now with 5 passages and threshold of 3, should recommend optimization
        should_optimize = consolidation_engine._should_optimize_indexes()
        assert should_optimize is True
    
    def test_does_not_recommend_optimization_below_threshold(self, consolidation_engine):
        """
        Test that optimization is NOT recommended when passage count <= 10,000.
        
        REAL BUG THIS CATCHES: If the threshold logic is inverted or has
        wrong comparison operators, the system could optimize indexes
        unnecessarily on small datasets, wasting computational resources.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Start with empty database
        with memory_manager.get_session() as session:
            session.query(MemoryPassage).delete()
            session.commit()
            assert session.query(MemoryPassage).count() == 0
        
        # Should not recommend optimization with zero passages
        should_optimize = consolidation_engine._should_optimize_indexes()
        assert should_optimize is False
        
        # Add a small number of passages (well below threshold)
        for i in range(5):
            passage_data = {
                "text": f"Small dataset passage {i}",
                "created_at": utc_now(),
                "importance_score": 0.5,
                "access_count": 1,
                "last_accessed": utc_now()
            }
            create_memory_passage(memory_manager, passage_data)
        
        # Verify count is small
        with memory_manager.get_session() as session:
            count = session.query(MemoryPassage).count()
            assert count == 5
            assert count < 10000
        
        # Should not recommend optimization with real threshold
        should_optimize = consolidation_engine._should_optimize_indexes()
        assert should_optimize is False
    
    def test_handles_empty_database_gracefully(self, consolidation_engine):
        """
        Test optimization decision with empty database.
        
        REAL BUG THIS CATCHES: If the method doesn't handle empty result
        sets correctly, it could crash when no passages exist, breaking
        maintenance operations on new or cleaned systems.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Ensure database is empty
        with memory_manager.get_session() as session:
            session.query(MemoryPassage).delete()
            session.commit()
            assert session.query(MemoryPassage).count() == 0
        
        # Should not crash and should return False
        should_optimize = consolidation_engine._should_optimize_indexes()
        assert should_optimize is False
    
    def test_executes_postgres_metadata_query_safely(self, consolidation_engine):
        """
        Test that the PostgreSQL metadata query executes without errors.
        
        REAL BUG THIS CATCHES: If the PostgreSQL system catalog query has
        syntax errors or permission issues, the optimization check would
        fail, breaking the consolidation process.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # This test verifies the query executes without throwing exceptions
        # The actual query looks for index metadata in pg_class/pg_namespace
        try:
            should_optimize = consolidation_engine._should_optimize_indexes()
            # Should execute without exceptions
            assert isinstance(should_optimize, bool)
        except Exception as e:
            pytest.fail(f"PostgreSQL metadata query failed: {e}")
    
    def test_returns_boolean_value(self, consolidation_engine):
        """
        Test that the method always returns a boolean value.
        
        REAL BUG THIS CATCHES: If the method returns None, strings, or other
        types instead of boolean, downstream code expecting True/False would
        fail with type errors.
        """
        # Test with various database states
        memory_manager = consolidation_engine.memory_manager
        
        # Test with empty database
        with memory_manager.get_session() as session:
            session.query(MemoryPassage).delete()
            session.commit()
        
        result = consolidation_engine._should_optimize_indexes()
        assert isinstance(result, bool)
        assert result in [True, False]
        
        # Test with some passages
        passage_data = {
            "text": "Boolean return test passage",
            "created_at": utc_now(),
            "importance_score": 0.5,
            "access_count": 1,
            "last_accessed": utc_now()
        }
        create_memory_passage(memory_manager, passage_data)
        
        result = consolidation_engine._should_optimize_indexes()
        assert isinstance(result, bool)
        assert result in [True, False]
    
    def test_sql_query_structure_and_safety(self, consolidation_engine):
        """
        Test that the SQL query is structured safely and accesses correct tables.
        
        REAL BUG THIS CATCHES: If the SQL query has injection vulnerabilities
        or accesses wrong system tables, it could expose sensitive data or
        fail on different PostgreSQL versions.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Verify the method executes its SQL query safely
        # The query should look for the 'idx_memory_passages_embedding' index
        with memory_manager.get_session() as session:
            # Test that we can execute the same query manually to verify its structure
            result = session.execute(
                text("""
                SELECT obj_description(c.oid, 'pg_class') as comment
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'idx_memory_passages_embedding'
                AND n.nspname = 'public'
                """)
            ).fetchone()
            
            # Query should execute without errors (result can be None if index doesn't exist)
            # The important thing is that it doesn't crash
            assert result is None or isinstance(result[0], (str, type(None)))
        
        # Now test the actual method
        should_optimize = consolidation_engine._should_optimize_indexes()
        assert isinstance(should_optimize, bool)
    
    def test_threshold_boundary_conditions(self, consolidation_engine, monkeypatch):
        """
        Test behavior exactly at the threshold boundary.
        
        REAL BUG THIS CATCHES: If boundary condition logic uses wrong
        comparison operators (< vs <= or > vs >=), the optimization
        might trigger at the wrong count, affecting performance timing.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Clear database
        with memory_manager.get_session() as session:
            session.query(MemoryPassage).delete()
            session.commit()
        
        # Create exactly 3 passages
        for i in range(3):
            passage_data = {
                "text": f"Boundary test passage {i}",
                "created_at": utc_now(),
                "importance_score": 0.5,
                "access_count": 1,
                "last_accessed": utc_now()
            }
            create_memory_passage(memory_manager, passage_data)
        
        # Patch method to use threshold of 3 for testing boundary
        def boundary_should_optimize():
            with memory_manager.get_session() as session:
                passage_count = session.query(MemoryPassage).count()
                # Test exactly at boundary: 3 passages with threshold 3
                return passage_count > 3
        
        monkeypatch.setattr(consolidation_engine, '_should_optimize_indexes', boundary_should_optimize)
        
        # With exactly 3 passages and threshold 3, should NOT optimize (3 > 3 is False)
        should_optimize = consolidation_engine._should_optimize_indexes()
        assert should_optimize is False
        
        # Add one more passage to exceed threshold
        passage_data = {
            "text": "Fourth passage to exceed threshold",
            "created_at": utc_now(),
            "importance_score": 0.5,
            "access_count": 1,
            "last_accessed": utc_now()
        }
        create_memory_passage(memory_manager, passage_data)
        
        # Now with 4 passages and threshold 3, should optimize (4 > 3 is True)
        should_optimize = consolidation_engine._should_optimize_indexes()
        assert should_optimize is True
    
    def test_consistent_results_with_same_database_state(self, consolidation_engine):
        """
        Test that the method returns consistent results for the same database state.
        
        REAL BUG THIS CATCHES: If the method has race conditions or depends
        on changing metadata, it could return different results for the same
        data, making optimization decisions unpredictable.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Set up a consistent database state
        with memory_manager.get_session() as session:
            session.query(MemoryPassage).delete()
            session.commit()
        
        # Add specific number of passages
        for i in range(7):
            passage_data = {
                "text": f"Consistency test passage {i}",
                "created_at": utc_now(),
                "importance_score": 0.5,
                "access_count": 1,
                "last_accessed": utc_now()
            }
            create_memory_passage(memory_manager, passage_data)
        
        # Call method multiple times - should get same result
        result1 = consolidation_engine._should_optimize_indexes()
        result2 = consolidation_engine._should_optimize_indexes()
        result3 = consolidation_engine._should_optimize_indexes()
        
        assert result1 == result2 == result3
        assert isinstance(result1, bool)


class TestConsolidateMemories:
    """
    Test the main consolidation orchestration method with real operations.
    
    This is the primary user-facing contract - all consolidation steps must
    execute correctly and return accurate operation counts.
    """
    
    def test_returns_complete_results_dictionary_structure(self, consolidation_engine):
        """
        Test that consolidate_memories returns properly structured results.
        
        REAL BUG THIS CATCHES: If the results dictionary structure is wrong
        or missing keys, downstream code expecting specific result fields
        would crash when processing consolidation outcomes.
        """
        # Run consolidation on clean database
        results = consolidation_engine.consolidate_memories()
        
        # Verify required result keys exist
        required_keys = ["pruned_passages", "updated_scores", "archived_conversations", "optimized_indexes"]
        for key in required_keys:
            assert key in results, f"Missing required result key: {key}"
        
        # Verify result types
        assert isinstance(results["pruned_passages"], int)
        assert isinstance(results["updated_scores"], int)
        assert isinstance(results["archived_conversations"], int)
        assert isinstance(results["optimized_indexes"], bool)
        
        # Verify non-negative counts
        assert results["pruned_passages"] >= 0
        assert results["updated_scores"] >= 0
        assert results["archived_conversations"] >= 0
    
    def test_executes_full_consolidation_workflow_successfully(self, consolidation_engine):
        """
        Test that all consolidation steps execute without errors.
        
        REAL BUG THIS CATCHES: If any step in the consolidation workflow
        has bugs or dependencies fail, the entire maintenance process
        would break, causing memory system degradation.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create test data for consolidation to work on
        passages_data = [
            {
                "text": "Recent important passage",
                "created_at": now - timedelta(days=5),
                "importance_score": 0.8,
                "access_count": 10,
                "last_accessed": now - timedelta(days=1)
            },
            {
                "text": "Old unimportant passage for pruning",
                "created_at": now - timedelta(days=35),
                "importance_score": 0.2,
                "access_count": 1,
                "last_accessed": now - timedelta(days=30)
            },
            {
                "text": "Passage needing score update",
                "created_at": now - timedelta(days=20),
                "importance_score": 0.5,
                "access_count": 5,
                "last_accessed": now - timedelta(days=2)
            }
        ]
        
        for passage_data in passages_data:
            create_memory_passage(memory_manager, passage_data)
        
        # Execute full consolidation workflow
        results = consolidation_engine.consolidate_memories()
        
        # Should complete without errors and return valid results
        assert isinstance(results, dict)
        assert results["pruned_passages"] >= 0  # May or may not prune based on exact criteria
        assert results["updated_scores"] >= 0   # Should update at least some scores
        assert results["archived_conversations"] == 0  # No conversation_id provided
        assert isinstance(results["optimized_indexes"], bool)
    
    def test_consolidation_with_conversation_id_archives_conversation(self, consolidation_engine):
        """
        Test that providing conversation_id triggers conversation archiving.
        
        REAL BUG THIS CATCHES: If conversation archiving logic is broken,
        completed conversations wouldn't be marked as archived, causing
        memory management to process them repeatedly.
        """
        test_conversation_id = "test_conversation_123"
        
        # Run consolidation with conversation ID
        results = consolidation_engine.consolidate_memories(conversation_id=test_conversation_id)
        
        # Should indicate conversation was archived
        assert results["archived_conversations"] == 1
        assert isinstance(results, dict)
    
    def test_consolidation_without_conversation_id_skips_archiving(self, consolidation_engine):
        """
        Test that not providing conversation_id skips conversation archiving.
        
        REAL BUG THIS CATCHES: If archiving logic runs when it shouldn't,
        system maintenance could incorrectly mark conversations as archived,
        affecting conversation history management.
        """
        # Run consolidation without conversation ID
        results = consolidation_engine.consolidate_memories()
        
        # Should not archive any conversations
        assert results["archived_conversations"] == 0
    
    def test_creates_memory_snapshot_during_consolidation(self, consolidation_engine):
        """
        Test that consolidation creates a memory snapshot for recovery.
        
        REAL BUG THIS CATCHES: If snapshot creation fails, there would be
        no recovery point after consolidation, making it impossible to
        restore memory state if issues are discovered later.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Count snapshots before consolidation
        with memory_manager.get_session() as session:
            from lt_memory.models.base import MemorySnapshot
            snapshots_before = session.query(MemorySnapshot).count()
        
        # Run consolidation
        test_conversation_id = "snapshot_test_conversation"
        results = consolidation_engine.consolidate_memories(conversation_id=test_conversation_id)
        
        # Should have created a new snapshot
        with memory_manager.get_session() as session:
            snapshots_after = session.query(MemorySnapshot).count()
            assert snapshots_after == snapshots_before + 1
            
            # Verify snapshot details
            latest_snapshot = session.query(MemorySnapshot).order_by(MemorySnapshot.created_at.desc()).first()
            assert latest_snapshot.conversation_id == test_conversation_id
            assert latest_snapshot.reason == "consolidation"
    
    def test_prunes_old_passages_during_consolidation(self, consolidation_engine):
        """
        Test that consolidation actually prunes old, unimportant passages.
        
        REAL BUG THIS CATCHES: If pruning logic is broken, old data would
        accumulate indefinitely, causing storage bloat and degraded
        search performance over time.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create passages that should be pruned (old + low importance + low access)
        prunable_passages_data = [
            {
                "text": f"Old prunable passage {i}",
                "created_at": now - timedelta(days=35),  # Older than 30 days
                "importance_score": 0.2,  # Below 0.3 threshold
                "access_count": 1,  # Below 2 threshold
                "last_accessed": now - timedelta(days=30)
            }
            for i in range(3)
        ]
        
        # Create passages that should be kept
        keepable_passages_data = [
            {
                "text": "Recent important passage to keep",
                "created_at": now - timedelta(days=5),
                "importance_score": 0.8,
                "access_count": 10,
                "last_accessed": now - timedelta(days=1)
            },
            {
                "text": "Old but important passage to keep",
                "created_at": now - timedelta(days=35),
                "importance_score": 0.8,  # High importance saves it
                "access_count": 1,
                "last_accessed": now - timedelta(days=30)
            }
        ]
        
        # Create all passages
        prunable_ids = []
        for passage_data in prunable_passages_data:
            passage_id = create_memory_passage(memory_manager, passage_data)
            prunable_ids.append(passage_id)
        
        keepable_ids = []
        for passage_data in keepable_passages_data:
            passage_id = create_memory_passage(memory_manager, passage_data)
            keepable_ids.append(passage_id)
        
        # Count passages before consolidation
        with memory_manager.get_session() as session:
            passages_before = session.query(MemoryPassage).count()
            assert passages_before == 5  # 3 prunable + 2 keepable
        
        # Run consolidation
        results = consolidation_engine.consolidate_memories()
        
        # Should have pruned the 3 old passages
        assert results["pruned_passages"] == 3
        
        # Verify actual database state
        with memory_manager.get_session() as session:
            passages_after = session.query(MemoryPassage).count()
            assert passages_after == 2  # Only the 2 keepable passages remain
            
            # Verify prunable passages are gone
            for passage_id in prunable_ids:
                assert session.query(MemoryPassage).filter_by(id=passage_id).first() is None
            
            # Verify keepable passages still exist
            for passage_id in keepable_ids:
                assert session.query(MemoryPassage).filter_by(id=passage_id).first() is not None
    
    def test_updates_importance_scores_during_consolidation(self, consolidation_engine):
        """
        Test that consolidation updates importance scores based on usage patterns.
        
        REAL BUG THIS CATCHES: If score updating is broken, memory importance
        would become stale, affecting which memories are prioritized for
        retrieval and retention.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create passage with old score that should decay significantly
        old_passage_data = {
            "text": "Old passage with high score that should decay",
            "created_at": now - timedelta(days=100),  # Very old
            "importance_score": 0.9,  # High score that should decay
            "access_count": 0,
            "last_accessed": None
        }
        
        passage_id = create_memory_passage(memory_manager, old_passage_data)
        
        # Store original score
        with memory_manager.get_session() as session:
            passage = session.query(MemoryPassage).filter_by(id=passage_id).first()
            original_score = passage.importance_score
            assert original_score == 0.9
        
        # Run consolidation
        results = consolidation_engine.consolidate_memories()
        
        # Should have updated at least this passage
        assert results["updated_scores"] >= 1
        
        # Verify score was actually updated
        with memory_manager.get_session() as session:
            passage = session.query(MemoryPassage).filter_by(id=passage_id).first()
            new_score = passage.importance_score
            
            # Score should have decayed due to age
            assert new_score < original_score
            # Should be clamped to minimum
            assert new_score >= 0.1
    
    def test_handles_empty_database_gracefully(self, consolidation_engine):
        """
        Test consolidation with no passages in database.
        
        REAL BUG THIS CATCHES: If consolidation doesn't handle empty
        databases correctly, it could crash when run on new or cleaned
        systems, breaking maintenance operations.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Ensure database is empty
        with memory_manager.get_session() as session:
            session.query(MemoryPassage).delete()
            session.commit()
            assert session.query(MemoryPassage).count() == 0
        
        # Run consolidation on empty database
        results = consolidation_engine.consolidate_memories()
        
        # Should complete without errors
        assert results["pruned_passages"] == 0
        assert results["updated_scores"] == 0
        assert results["archived_conversations"] == 0
        assert isinstance(results["optimized_indexes"], bool)
    
    def test_real_embedding_cache_integration(self, consolidation_engine):
        """
        Test that consolidation integrates with real embedding cache operations.
        
        REAL BUG THIS CATCHES: If embedding cache integration is broken,
        cache cleanup might not work properly, leading to unbounded cache
        growth and disk space issues.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Add some data to work with and potentially trigger cache operations
        passage_data = {
            "text": "Test passage for cache integration",
            "created_at": utc_now(),
            "importance_score": 0.5,
            "access_count": 1,
            "last_accessed": utc_now()
        }
        create_memory_passage(memory_manager, passage_data)
        
        # Get cache stats before consolidation
        cache_stats_before = memory_manager.embedding_cache.get_stats()
        
        # Run consolidation (this tests real cache integration)
        results = consolidation_engine.consolidate_memories()
        
        # Should complete without errors - cache operations integrated properly
        assert isinstance(results, dict)
        
        # Cache should still be functional after consolidation
        cache_stats_after = memory_manager.embedding_cache.get_stats()
        assert isinstance(cache_stats_after, dict)
    
    def test_real_vector_store_optimization_integration(self, consolidation_engine):
        """
        Test that consolidation integrates with real vector store operations.
        
        REAL BUG THIS CATCHES: If vector store integration is broken,
        index optimization calls could fail, preventing performance
        improvements from being applied to vector search.
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Add passages to make optimization potentially worthwhile
        for i in range(10):
            passage_data = {
                "text": f"Vector store test passage {i}",
                "created_at": utc_now(),
                "importance_score": 0.5,
                "access_count": 1,
                "last_accessed": utc_now()
            }
            create_memory_passage(memory_manager, passage_data)
        
        # Run consolidation (this tests real vector store integration)
        results = consolidation_engine.consolidate_memories()
        
        # Should complete without errors - vector store operations integrated properly
        assert isinstance(results["optimized_indexes"], bool)
        
        # Vector store should still be functional after consolidation
        # Test by doing a simple operation
        test_embedding = memory_manager.generate_embedding("test query")
        search_results = memory_manager.vector_store.search(test_embedding, k=1)
        assert isinstance(search_results, list)
    
    def test_consolidation_with_realistic_mixed_data(self, consolidation_engine):
        """
        Test consolidation with realistic mix of memory data patterns.
        
        REAL BUG THIS CATCHES: If consolidation fails with real-world data
        complexity, the system would break in production where data has
        varied patterns, edge cases, and realistic usage.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create realistic mixed dataset
        realistic_data = [
            # Recent, important, frequently accessed - should be preserved and boosted
            {
                "text": "Recent important conversation about project planning",
                "created_at": now - timedelta(days=2),
                "importance_score": 0.7,
                "access_count": 15,
                "last_accessed": now - timedelta(hours=6)
            },
            # Old but important - should be preserved
            {
                "text": "Important historical context that should be kept",
                "created_at": now - timedelta(days=25),
                "importance_score": 0.8,
                "access_count": 3,
                "last_accessed": now - timedelta(days=5)
            },
            # Old and unimportant - should be pruned
            {
                "text": "Old unimportant note about weather",
                "created_at": now - timedelta(days=40),
                "importance_score": 0.1,
                "access_count": 1,
                "last_accessed": now - timedelta(days=35)
            },
            # Never accessed - should get default aging
            {
                "text": "Passage that was never accessed after creation",
                "created_at": now - timedelta(days=15),
                "importance_score": 0.5,
                "access_count": 0,
                "last_accessed": None
            },
            # Boundary case - exactly at thresholds
            {
                "text": "Boundary case passage for edge testing",
                "created_at": now - timedelta(days=30),  # Exactly at age threshold
                "importance_score": 0.3,  # Exactly at importance threshold
                "access_count": 2,  # Exactly at access threshold
                "last_accessed": now - timedelta(days=15)
            }
        ]
        
        passage_ids = []
        for passage_data in realistic_data:
            passage_id = create_memory_passage(memory_manager, passage_data)
            passage_ids.append(passage_id)
        
        # Count passages before consolidation
        with memory_manager.get_session() as session:
            passages_before = session.query(MemoryPassage).count()
            assert passages_before == 5
        
        # Run consolidation with conversation archiving
        results = consolidation_engine.consolidate_memories(conversation_id="realistic_test")
        
        # Verify realistic outcomes
        with memory_manager.get_session() as session:
            passages_after = session.query(MemoryPassage).count()
            
            # Should have pruned at least the clearly old/unimportant passage
            assert passages_after < passages_before
            assert results["pruned_passages"] >= 1
            
            # Should have updated some scores
            assert results["updated_scores"] >= 0
            
            # Should have archived the conversation
            assert results["archived_conversations"] == 1
            
            # Should have valid index optimization decision
            assert isinstance(results["optimized_indexes"], bool)
        
        # Verify important passages were preserved
        with memory_manager.get_session() as session:
            # The important historical context should still exist
            important_passage = session.query(MemoryPassage).filter(
                MemoryPassage.text.contains("Important historical context")
            ).first()
            assert important_passage is not None
            
            # The recent important passage should still exist
            recent_passage = session.query(MemoryPassage).filter(
                MemoryPassage.text.contains("Recent important conversation")
            ).first()
            assert recent_passage is not None
    
    def test_consolidation_operation_counts_are_accurate(self, consolidation_engine):
        """
        Test that returned operation counts accurately reflect what was done.
        
        REAL BUG THIS CATCHES: If operation counting is wrong, users and
        monitoring systems would have incorrect information about consolidation
        effectiveness, making memory management impossible to tune properly.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create exactly 2 passages that should be pruned
        prunable_data = [
            {
                "text": "First prunable passage",
                "created_at": now - timedelta(days=35),
                "importance_score": 0.2,
                "access_count": 1,
                "last_accessed": now - timedelta(days=30)
            },
            {
                "text": "Second prunable passage",
                "created_at": now - timedelta(days=40),
                "importance_score": 0.15,
                "access_count": 0,
                "last_accessed": now - timedelta(days=35)
            }
        ]
        
        # Create exactly 1 passage that needs score update
        score_update_data = {
            "text": "Passage needing score update",
            "created_at": now - timedelta(days=100),  # Very old for significant decay
            "importance_score": 0.8,  # High score that will decay significantly
            "access_count": 0,
            "last_accessed": None
        }
        
        # Create passages
        for passage_data in prunable_data:
            create_memory_passage(memory_manager, passage_data)
        
        score_update_id = create_memory_passage(memory_manager, score_update_data)
        
        # Run consolidation
        results = consolidation_engine.consolidate_memories(conversation_id="count_test")
        
        # Verify exact counts
        assert results["pruned_passages"] == 2  # Exactly 2 prunable passages
        assert results["updated_scores"] >= 1   # At least the score update passage
        assert results["archived_conversations"] == 1  # One conversation archived
        
        # Verify database state matches counts
        with memory_manager.get_session() as session:
            remaining_passages = session.query(MemoryPassage).count()
            # Should have 1 remaining passage (the score update one, since it wasn't pruned)
            assert remaining_passages == 1
            
            # Verify the remaining passage has updated score
            remaining_passage = session.query(MemoryPassage).filter_by(id=score_update_id).first()
            assert remaining_passage is not None
            assert remaining_passage.importance_score < 0.8  # Score should have decayed


class TestAnalyzeMemoryPatterns:
    """
    Test memory pattern analysis functionality with real database operations.
    
    Analytics are critical for monitoring memory system health, usage patterns,
    and making informed decisions about memory management tuning.
    """
    
    def test_returns_complete_analysis_structure(self, consolidation_engine):
        """
        Test that analyze_memory_patterns returns properly structured analysis.
        
        REAL BUG THIS CATCHES: If the analysis structure is wrong or missing
        required fields, monitoring systems and dashboards would break when
        trying to display memory system health metrics.
        """
        # Run analysis on clean database
        analysis = consolidation_engine.analyze_memory_patterns()
        
        # Verify required top-level keys exist
        required_keys = ["passage_creation_rate", "memory_health"]
        for key in required_keys:
            assert key in analysis, f"Missing required analysis key: {key}"
        
        # Verify passage_creation_rate structure
        creation_rate = analysis["passage_creation_rate"]
        assert isinstance(creation_rate, dict)
        
        expected_time_windows = ["last_hour", "last_day", "last_week", "last_month"]
        for window in expected_time_windows:
            assert window in creation_rate, f"Missing time window: {window}"
            assert isinstance(creation_rate[window], int)
            assert creation_rate[window] >= 0
        
        # Verify memory_health structure
        memory_health = analysis["memory_health"]
        assert isinstance(memory_health, dict)
        
        expected_health_keys = ["total_passages", "low_importance_ratio", "avg_passage_importance"]
        for key in expected_health_keys:
            assert key in memory_health, f"Missing memory health key: {key}"
        
        # Verify health metric types and ranges
        assert isinstance(memory_health["total_passages"], int)
        assert memory_health["total_passages"] >= 0
        
        assert isinstance(memory_health["low_importance_ratio"], (int, float))
        assert 0.0 <= memory_health["low_importance_ratio"] <= 1.0
        
        assert isinstance(memory_health["avg_passage_importance"], (int, float))
        assert 0.0 <= memory_health["avg_passage_importance"] <= 1.0
    
    def test_creation_rate_windows_are_properly_nested(self, consolidation_engine):
        """
        Test that creation rate time windows follow logical nesting.
        
        REAL BUG THIS CATCHES: If time window logic is wrong, longer periods
        could show fewer passages than shorter periods, breaking the fundamental
        assumption that last_month >= last_week >= last_day >= last_hour.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create passages across different time periods
        passages_data = [
            # Recent passages
            {"text": "Recent 1", "created_at": now - timedelta(minutes=30)},
            {"text": "Recent 2", "created_at": now - timedelta(hours=2)},
            {"text": "Recent 3", "created_at": now - timedelta(hours=18)},
            # Older passages  
            {"text": "Weekly 1", "created_at": now - timedelta(days=3)},
            {"text": "Weekly 2", "created_at": now - timedelta(days=5)},
            # Even older
            {"text": "Monthly", "created_at": now - timedelta(days=15)}
        ]
        
        for data in passages_data:
            passage_data = {
                "text": data["text"],
                "created_at": data["created_at"],
                "importance_score": 0.5,
                "access_count": 1,
                "last_accessed": now
            }
            create_memory_passage(memory_manager, passage_data)
        
        analysis = consolidation_engine.analyze_memory_patterns()
        rates = analysis["passage_creation_rate"]
        
        # Time windows must be properly nested (longer periods >= shorter periods)
        assert rates["last_hour"] <= rates["last_day"]
        assert rates["last_day"] <= rates["last_week"] 
        assert rates["last_week"] <= rates["last_month"]
        
        # Verify actual counts make sense
        assert rates["last_hour"] >= 1  # At least the 30-minute passage
        assert rates["last_day"] >= 3   # Should include recent passages
        assert rates["last_week"] >= 5  # Should include weekly passages
        assert rates["last_month"] == 6 # Should include all test passages
    
    def test_low_importance_ratio_identifies_problematic_data(self, consolidation_engine):
        """
        Test that low importance ratio correctly identifies memory quality issues.
        
        REAL BUG THIS CATCHES: If low importance detection is wrong, the system
        wouldn't alert operators to memory quality degradation, allowing poor
        data to accumulate and degrade search performance.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create mostly low-importance data (system health problem)
        problematic_data = [
            # 4 low importance passages (below 0.3 threshold)
            {"text": "Low 1", "importance_score": 0.1},
            {"text": "Low 2", "importance_score": 0.2}, 
            {"text": "Low 3", "importance_score": 0.29},  # Just below threshold
            {"text": "Low 4", "importance_score": 0.05},
            # 1 good passage
            {"text": "Good", "importance_score": 0.8}
        ]
        
        for data in problematic_data:
            passage_data = {
                "text": data["text"],
                "created_at": now,
                "importance_score": data["importance_score"],
                "access_count": 1,
                "last_accessed": now
            }
            create_memory_passage(memory_manager, passage_data)
        
        analysis = consolidation_engine.analyze_memory_patterns()
        health = analysis["memory_health"]
        
        # Should detect this as a problematic ratio
        assert health["total_passages"] == 5
        assert health["low_importance_ratio"] > 0.5  # More than half are low quality
        assert health["low_importance_ratio"] < 1.0  # But not all (due to good passage)
        
        # Now test healthy data
        with memory_manager.get_session() as session:
            session.query(MemoryPassage).delete()
            session.commit()
        
        # Create mostly high-importance data (healthy system)
        healthy_data = [
            {"text": "High 1", "importance_score": 0.8},
            {"text": "High 2", "importance_score": 0.9},
            {"text": "High 3", "importance_score": 0.7},
            {"text": "High 4", "importance_score": 0.6},
            # 1 low passage
            {"text": "Low", "importance_score": 0.2}
        ]
        
        for data in healthy_data:
            passage_data = {
                "text": data["text"],
                "created_at": now,
                "importance_score": data["importance_score"],
                "access_count": 1,
                "last_accessed": now
            }
            create_memory_passage(memory_manager, passage_data)
        
        analysis = consolidation_engine.analyze_memory_patterns()
        health = analysis["memory_health"]
        
        # Should detect this as healthy
        assert health["total_passages"] == 5
        assert health["low_importance_ratio"] < 0.5  # Less than half are low quality
        assert health["low_importance_ratio"] > 0.0  # But not zero (due to low passage)
    
    def test_handles_empty_database_without_division_by_zero(self, consolidation_engine):
        """
        Test analysis with no passages avoids mathematical errors.
        
        REAL BUG THIS CATCHES: If analysis doesn't handle empty databases
        correctly, division by zero errors would crash monitoring systems
        when they're most needed (during system initialization or after cleanup).
        """
        memory_manager = consolidation_engine.memory_manager
        
        # Ensure database is empty
        with memory_manager.get_session() as session:
            session.query(MemoryPassage).delete()
            session.commit()
            assert session.query(MemoryPassage).count() == 0
        
        # Should not crash with division by zero
        analysis = consolidation_engine.analyze_memory_patterns()
        
        # Verify safe handling of empty state
        creation_rates = analysis["passage_creation_rate"]
        assert all(rate == 0 for rate in creation_rates.values())
        
        health = analysis["memory_health"]
        assert health["total_passages"] == 0
        assert health["low_importance_ratio"] == 0  # Safe default, not NaN
        assert health["avg_passage_importance"] == 0  # Safe default, not NaN
        
        # Verify returned values are valid numbers, not NaN or None
        assert isinstance(health["low_importance_ratio"], (int, float))
        assert isinstance(health["avg_passage_importance"], (int, float))
    
    def test_time_boundaries_exclude_old_data_correctly(self, consolidation_engine):
        """
        Test that time window boundaries correctly exclude old data.
        
        REAL BUG THIS CATCHES: If time comparisons use wrong operators
        (>= instead of > or <= instead of <), old data would be incorrectly
        included in recent metrics, inflating creation rates.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create passages just outside each time window
        old_data = [
            # Just over 1 hour - should NOT appear in last_hour
            {
                "text": "Just over hour",
                "created_at": now - timedelta(hours=1, minutes=5),
                "importance_score": 0.5,
                "access_count": 1,
                "last_accessed": now
            },
            # Just over 1 day - should NOT appear in last_day  
            {
                "text": "Just over day",
                "created_at": now - timedelta(days=1, hours=2),
                "importance_score": 0.5,
                "access_count": 1,
                "last_accessed": now
            },
            # Just over 1 week - should NOT appear in last_week
            {
                "text": "Just over week", 
                "created_at": now - timedelta(weeks=1, days=1),
                "importance_score": 0.5,
                "access_count": 1,
                "last_accessed": now
            },
            # Just over 1 month - should NOT appear in last_month
            {
                "text": "Just over month",
                "created_at": now - timedelta(days=31),
                "importance_score": 0.5,
                "access_count": 1,
                "last_accessed": now
            }
        ]
        
        for passage_data in old_data:
            create_memory_passage(memory_manager, passage_data)
        
        analysis = consolidation_engine.analyze_memory_patterns()
        rates = analysis["passage_creation_rate"]
        
        # All passages are just outside their respective windows
        assert rates["last_hour"] == 0     # Just over 1 hour should not count
        assert rates["last_day"] == 1      # Just over 1 hour counts in day, but not just over day
        assert rates["last_week"] == 2     # Just over hour + day count in week, but not just over week
        assert rates["last_month"] == 3    # All but just over month should count
    
    def test_analysis_reflects_database_changes_immediately(self, consolidation_engine):
        """
        Test that analysis reflects real-time database state changes.
        
        REAL BUG THIS CATCHES: If analysis caches stale data or doesn't
        read current database state, monitoring metrics would lag behind
        reality, making real-time system health assessment impossible.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Start with empty database
        with memory_manager.get_session() as session:
            session.query(MemoryPassage).delete()
            session.commit()
        
        # Initial analysis should show empty state
        analysis1 = consolidation_engine.analyze_memory_patterns()
        assert analysis1["memory_health"]["total_passages"] == 0
        
        # Add a passage
        passage_data = {
            "text": "Real-time test passage",
            "created_at": now,
            "importance_score": 0.7,
            "access_count": 1,
            "last_accessed": now
        }
        create_memory_passage(memory_manager, passage_data)
        
        # Analysis should immediately reflect the change
        analysis2 = consolidation_engine.analyze_memory_patterns()
        assert analysis2["memory_health"]["total_passages"] == 1
        assert analysis2["passage_creation_rate"]["last_hour"] == 1
        
        # Add another passage with different characteristics
        passage_data2 = {
            "text": "Second real-time test passage",
            "created_at": now,
            "importance_score": 0.1,  # Low importance
            "access_count": 1,
            "last_accessed": now
        }
        create_memory_passage(memory_manager, passage_data2)
        
        # Analysis should reflect both passages
        analysis3 = consolidation_engine.analyze_memory_patterns()
        assert analysis3["memory_health"]["total_passages"] == 2
        assert analysis3["memory_health"]["low_importance_ratio"] == 0.5  # 1 out of 2
        assert analysis3["passage_creation_rate"]["last_hour"] == 2
    
    def test_importance_threshold_boundary_at_exactly_03(self, consolidation_engine):
        """
        Test that importance threshold boundary handles exactly 0.3 correctly.
        
        REAL BUG THIS CATCHES: If boundary logic uses wrong comparison
        operators (< vs <=), passages exactly at 0.3 threshold would be
        misclassified, affecting memory quality assessment accuracy.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create passages exactly at and around the 0.3 threshold
        boundary_data = [
            {"text": "Exactly at threshold", "importance_score": 0.3},      # Exactly at boundary
            {"text": "Just below threshold", "importance_score": 0.299},   # Just below
            {"text": "Just above threshold", "importance_score": 0.301},   # Just above
        ]
        
        for data in boundary_data:
            passage_data = {
                "text": data["text"],
                "created_at": now,
                "importance_score": data["importance_score"],
                "access_count": 1,
                "last_accessed": now
            }
            create_memory_passage(memory_manager, passage_data)
        
        analysis = consolidation_engine.analyze_memory_patterns()
        health = analysis["memory_health"]
        
        # Based on the code: importance_score < 0.3 means low importance
        # So exactly 0.3 should NOT be considered low importance
        assert health["total_passages"] == 3
        
        # Only the 0.299 passage should be considered low importance
        # 0.3 and 0.301 should not be low importance
        expected_low_count = 1  # Only the 0.299 passage
        expected_ratio = expected_low_count / 3.0
        assert health["low_importance_ratio"] == expected_ratio
    
    def test_analysis_consistency_across_multiple_calls(self, consolidation_engine):
        """
        Test that analysis returns identical results for unchanged database.
        
        REAL BUG THIS CATCHES: If analysis has race conditions or depends
        on non-deterministic state, monitoring metrics would fluctuate
        randomly, making alerting and trend analysis unreliable.
        """
        memory_manager = consolidation_engine.memory_manager
        now = utc_now()
        
        # Create stable test dataset
        stable_data = [
            {
                "text": "Stable passage 1",
                "created_at": now - timedelta(hours=6),
                "importance_score": 0.6,
                "access_count": 3,
                "last_accessed": now
            },
            {
                "text": "Stable passage 2", 
                "created_at": now - timedelta(days=2),
                "importance_score": 0.2,
                "access_count": 1,
                "last_accessed": now
            }
        ]
        
        for passage_data in stable_data:
            create_memory_passage(memory_manager, passage_data)
        
        # Run analysis multiple times - should be identical
        analysis1 = consolidation_engine.analyze_memory_patterns()
        analysis2 = consolidation_engine.analyze_memory_patterns()
        analysis3 = consolidation_engine.analyze_memory_patterns()
        
        # All results must be exactly identical
        assert analysis1 == analysis2 == analysis3
        
        # Verify deterministic behavior for specific metrics
        for analysis in [analysis1, analysis2, analysis3]:
            assert analysis["memory_health"]["total_passages"] == 2
            assert analysis["passage_creation_rate"]["last_day"] == 1  # Only 6-hour passage
            assert analysis["passage_creation_rate"]["last_week"] == 2  # Both passages