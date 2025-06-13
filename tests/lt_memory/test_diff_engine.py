"""
Production-grade tests for DiffEngine differential versioning system.

This test suite focuses on realistic production scenarios that matter for
reliability. We test the contracts and behaviors that users actually depend on,
not theoretical edge cases that will never occur.

Testing philosophy:
1. Test the public API contracts that users rely on
2. Test critical private methods that could fail in subtle ways
3. Focus on scenarios that actually happen in production
4. Avoid impossible scenarios that would force unnecessary defensive coding
"""

import pytest
import time
from typing import List, Dict, Any, Tuple
from unittest.mock import patch

# Import the system under test
from lt_memory.utils.diff_engine import (
    DiffEngine, 
    DiffCorruptionError, 
    DiffValidationResult,
    Hunk
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_texts():
    """
    Provides realistic text samples that represent actual usage.
    
    These samples cover the main use cases we see in production:
    - Standard text documents
    - Code files
    - Empty files (new file creation)
    - Unicode content (international users)
    """
    return {
        # Standard document - most common case
        "document": """# Project Plan
Last updated: 2024-01-15

## Overview
This document outlines our Q1 initiatives.

## Goals
1. Launch new feature
2. Improve performance
3. Fix critical bugs
""",
        
        # Empty content - new file creation
        "empty": "",
        
        # Code file - common use case
        "code": """def process_data(items):
    '''Process a list of items.'''
    results = []
    for item in items:
        if item.valid:
            results.append(item.process())
    return results
""",
        
        # Unicode content - international users exist
        "unicode": "# README\n\nSupports international text: Hello 世界! Привет мир! 🎉\n",
        
        # Single line - config files, etc.
        "single_line": "API_KEY=secret123"
    }


@pytest.fixture
def create_version_sequence():
    """
    Helper to create realistic version sequences for testing.
    
    This simulates how documents actually evolve in production:
    small incremental changes rather than massive rewrites.
    """
    def _create_sequence(base_content: str, num_versions: int = 5) -> List[str]:
        versions = [base_content]
        current = base_content
        
        for i in range(1, num_versions):
            # Simulate realistic edits
            if i == 1:
                # Add a line
                current = current.replace("## Goals", "## Goals\n0. Set up environment")
            elif i == 2:
                # Modify existing line
                current = current.replace("Launch new feature", "Launch payment feature")
            elif i == 3:
                # Delete a line
                lines = current.splitlines()
                lines = [l for l in lines if "Fix critical bugs" not in l]
                current = "\n".join(lines) + "\n"
            elif i == 4:
                # Add section
                current = current + "\n## Timeline\nQ1 2024\n"
            else:
                # Minor edit
                current = current.replace(f"Version {i-1}", f"Version {i}")
            
            versions.append(current)
        
        return versions
    
    return _create_sequence


# =============================================================================
# CRITICAL PRIVATE METHOD TESTS
# =============================================================================

class TestDiffComputation:
    """
    Test the core diff computation logic.
    
    These tests ensure the diff algorithm works correctly for common cases
    we see in production. We don't test impossible scenarios.
    """
    
    def test_basic_diff_creation(self):
        """
        Test that basic diffs are created correctly.
        
        This is the fundamental operation - if this fails, nothing works.
        """
        old = "Line 1\nLine 2\nLine 3\n"
        new = "Line 1\nLine 2 modified\nLine 3\n"
        
        result = DiffEngine._compute_unified_diff(old, new)
        
        # Should produce a diff
        assert len(result["operations"]) > 0
        assert result["size"] > 0
        
        # Should contain the change - using real diff format
        ops_text = '\n'.join(result["operations"])
        assert "-Line 2\n" in ops_text  # Real format includes newline
        assert "+Line 2 modified\n" in ops_text  # Real format includes newline
    
    def test_empty_file_handling(self):
        """
        Test creating and modifying empty files.
        
        Common case: new files start empty, then get content.
        """
        # Empty to content
        result1 = DiffEngine._compute_unified_diff("", "New content\n")
        assert any("+New content\n" in op for op in result1["operations"])
        
        # Content to empty  
        result2 = DiffEngine._compute_unified_diff("Old content\n", "")
        assert any("-Old content\n" in op for op in result2["operations"])
    
    def test_unicode_content_diff(self, sample_texts):
        """
        Test that Unicode content is handled correctly.
        
        Production reality: international users exist.
        """
        old = sample_texts["unicode"]
        new = old.replace("世界", "世界!!!")
        
        result = DiffEngine._compute_unified_diff(old, new)
        
        # Should handle Unicode without errors
        ops_text = '\n'.join(result["operations"])
        assert "世界" in ops_text


class TestDiffApplication:
    """
    Test applying diffs to reconstruct content.
    
    This is the most critical operation - errors here mean data loss.
    """
    
    def test_apply_basic_diff(self):
        """
        Test applying a simple diff.
        
        If this fails, version reconstruction is broken.
        """
        base = "Line 1\nLine 2\nLine 3\n"
        new = "Line 1\nLine 2 modified\nLine 3\n"
        
        # Generate real diff operations using the engine
        diff_data = DiffEngine._compute_unified_diff(base, new)
        real_diff_ops = diff_data["operations"]
        
        result = DiffEngine._apply_diff_operations(base, real_diff_ops)
        
        # This should produce the expected result
        assert result == new
    
    def test_apply_multi_hunk_diff(self):
        """
        Test applying diff with multiple change regions.
        
        Common in real edits where changes are scattered.
        """
        base = "\n".join([f"Line {i}" for i in range(1, 11)]) + "\n"
        diff_ops = [
            "@@ -2,1 +2,1 @@",
            "- Line 2",
            "+ Line 2 modified",
            "@@ -7,2 +7,2 @@", 
            "- Line 7",
            "- Line 8",
            "+ Line 7 changed",
            "+ Line 8 changed"
        ]
        
        result = DiffEngine._apply_diff_operations(base, diff_ops)
        
        # Both changes should be applied
        assert "Line 2 modified" in result
        assert "Line 7 changed" in result
        assert "Line 8 changed" in result
        assert "Line 5" in result  # Unchanged lines preserved
    
    def test_apply_to_wrong_content_fails_validation(self):
        """
        Test that validation catches when diff is applied to wrong base.
        
        This prevents silent corruption from applying diffs out of order.
        """
        base = "Line 1\nLine 2\nLine 3\n"
        wrong_base = "Different Line 1\nDifferent Line 2\nLine 3\n"
        
        # Create diff for correct base
        new_content = "Line 1\nLine 2 modified\nLine 3\n"
        diff_version = DiffEngine.create_diff_version(base, new_content, 2)
        diff_ops = diff_version["forward_operations"]
        
        # Validation should catch mismatch
        validation = DiffEngine.validate_diff(wrong_base, diff_ops)
        assert validation.valid is False
        assert any("mismatch" in issue.lower() for issue in validation.issues)


class TestDiffParsing:
    """
    Test parsing of diff format into structured data.
    
    Parsing errors would cause reconstruction failures.
    """
    
    def test_parse_standard_diff(self):
        """
        Test parsing a well-formed diff.
        
        This is the normal case - should always work.
        """
        diff_ops = [
            "@@ -1,3 +1,3 @@",
            "  Line 1",
            "- Line 2",
            "+ Line 2 new",
            "  Line 3"
        ]
        
        hunks = DiffEngine._parse_hunks(diff_ops)
        
        assert len(hunks) == 1
        hunk = hunks[0]
        assert hunk.old_start == 0  # 0-based
        assert hunk.old_count == 3
        assert len(hunk.operations) == 4
    
    def test_parse_malformed_header_fails(self):
        """
        Test that malformed headers are rejected.
        
        This catches corruption before it causes problems.
        """
        diff_ops = [
            "@@ CORRUPTED @@",
            "  Line 1"
        ]
        
        with pytest.raises(ValueError) as exc_info:
            DiffEngine._parse_hunks(diff_ops)
        
        assert "Invalid hunk header" in str(exc_info.value)


# =============================================================================
# PUBLIC API CONTRACT TESTS
# =============================================================================

class TestVersionCreation:
    """
    Test the public API for creating versions.
    
    These methods are the primary interface users depend on.
    """
    
    def test_create_base_version(self, sample_texts):
        """
        Test creating the initial version.
        
        Every version history starts with this.
        """
        content = sample_texts["document"]
        base = DiffEngine.create_base_version(content)
        
        # Verify the contract
        assert base["type"] == "base"
        assert base["content"] == content
        assert base["size"] == len(content)
        assert base["line_count"] == len(content.splitlines())
    
    def test_create_diff_version(self):
        """
        Test creating a differential version.
        
        This is how we store changes efficiently.
        """
        old = "Original line\n"
        new = "Modified line\n"
        
        diff = DiffEngine.create_diff_version(old, new, version=2)
        
        # Verify the contract
        assert diff["type"] == "diff"
        assert diff["version"] == 2
        assert "forward_operations" in diff
        assert "reverse_operations" in diff
        assert diff["checksum"] == DiffEngine._content_checksum(new)
    
    def test_diff_validation_prevents_corruption(self):
        """
        Test that invalid diffs are rejected at creation time.
        
        Better to fail fast than corrupt data later.
        """
        old = "Line 1\nLine 2\n"
        new = "Line 1\nLine 2 modified\n"
        
        # Mock a validation failure
        with patch.object(DiffEngine, 'validate_diff') as mock_validate:
            mock_validate.return_value = DiffValidationResult(
                valid=False,
                issues=["Test corruption"],
                hunk_count=1
            )
            
            with pytest.raises(ValueError) as exc_info:
                DiffEngine.create_diff_version(old, new, 2)
            
            assert "invalid" in str(exc_info.value).lower()


class TestVersionReconstruction:
    """
    Test reconstructing historical versions.
    
    This is the core value proposition - get any past version back.
    """
    
    def test_reconstruct_base_version(self):
        """
        Test that version 1 returns the base content.
        
        Simplest case but must work perfectly.
        """
        base_content = "Original content\n"
        
        result = DiffEngine.reconstruct_version(base_content, [], 1)
        assert result == base_content
    
    def test_reconstruct_through_diff_chain(self, create_version_sequence):
        """
        Test reconstructing through multiple diffs.
        
        This is the normal production case.
        """
        # Create realistic version sequence
        versions = create_version_sequence("Version 1\n", num_versions=5)
        
        # Build diff history
        base_content = versions[0]
        diff_history = []
        
        for i in range(1, len(versions)):
            old = versions[i-1]
            new = versions[i]
            diff = DiffEngine.create_diff_version(old, new, version=i+1)
            diff_history.append(diff)
        
        # Test reconstructing each version
        for i, expected in enumerate(versions):
            version_num = i + 1
            result = DiffEngine.reconstruct_version(base_content, diff_history, version_num)
            assert result == expected, f"Version {version_num} mismatch"
    
    def test_reconstruct_uses_snapshots(self, sample_texts):
        """
        Test that snapshots are used for efficiency.
        
        Important for performance with many versions.
        """
        # Create enough versions to trigger snapshot
        versions = []
        base = sample_texts["document"]
        current = base
        
        for i in range(15):
            if i > 0:
                current = current.replace(f"Version {i}", f"Version {i+1}")
            versions.append(current)
        
        # Build history with snapshots
        base_content = versions[0]
        diff_history = []
        
        for i in range(1, len(versions)):
            version_num = i + 1
            
            if DiffEngine.should_create_snapshot(version_num):
                snapshot = DiffEngine.create_snapshot_version(versions[i], version_num)
                diff_history.append(snapshot)
            else:
                diff = DiffEngine.create_diff_version(versions[i-1], versions[i], version_num)
                diff_history.append(diff)
        
        # Verify snapshot was created at version 11
        snapshots = [h for h in diff_history if h["type"] == "snapshot"]
        assert any(s["version"] == 11 for s in snapshots)
        
        # Reconstruct version 13 (should use snapshot)
        result = DiffEngine.reconstruct_version(base_content, diff_history, 13)
        assert result == versions[12]
    
    def test_checksum_validation_detects_corruption(self):
        """
        Test that corrupted content is detected via checksums.
        
        Last line of defense against data corruption.
        """
        base = "Original\n"
        new = "Modified\n"
        
        diff = DiffEngine.create_diff_version(base, new, 2)
        
        # Corrupt the checksum
        diff["checksum"] = "wrongchecksum"
        
        with pytest.raises(DiffCorruptionError) as exc_info:
            DiffEngine.reconstruct_version(base, [diff], 2)
        
        assert "checksum mismatch" in str(exc_info.value).lower()
    
    def test_missing_version_detected(self):
        """
        Test that missing versions in the chain are detected.
        
        Prevents silent data corruption from incomplete histories.
        """
        base = "Version 1\n"
        
        # Create history with gap (missing version 2)
        diff_v3 = DiffEngine.create_diff_version("Version 2\n", "Version 3\n", 3)
        
        with pytest.raises(ValueError) as exc_info:
            DiffEngine.reconstruct_version(base, [diff_v3], 3)
        
        assert "Missing version data for version 2" in str(exc_info.value)


class TestDiffValidation:
    """
    Test the validation that prevents corrupted diffs from being applied.
    
    This is critical for data integrity.
    """
    
    def test_valid_diff_passes(self):
        """
        Test that correct diffs pass validation.
        
        We shouldn't reject good data.
        """
        base = "Line 1\nLine 2\nLine 3\n"
        new = "Line 1\nLine 2 modified\nLine 3\n"
        
        # Generate real diff operations using the engine
        diff_data = DiffEngine._compute_unified_diff(base, new)
        real_diff_ops = diff_data["operations"]
        
        result = DiffEngine.validate_diff(base, real_diff_ops)
        assert result.valid is True
        assert len(result.issues) == 0
    
    def test_wrong_base_content_detected(self):
        """
        Test that applying diff to wrong version is caught.
        
        Common error that must be prevented.
        """
        base = "Line 1\nLine 2\nLine 3\n"
        diff_ops = [
            "@@ -1,2 +1,2 @@",
            "  Wrong Line 1",  # Doesn't match base
            "- Line 2",
            "+ Line 2 modified"
        ]
        
        result = DiffEngine.validate_diff(base, diff_ops)
        assert result.valid is False
        assert any("mismatch" in issue for issue in result.issues)
    
    def test_out_of_bounds_detected(self):
        """
        Test that diffs referencing non-existent lines are caught.
        
        Prevents array index errors.
        """
        base = "Line 1\nLine 2\n"  # Only 2 lines
        diff_ops = [
            "@@ -5,1 +5,1 @@",  # Line 5 doesn't exist
            "- Line 5",
            "+ Line 5 modified"
        ]
        
        result = DiffEngine.validate_diff(base, diff_ops)
        assert result.valid is False
        assert any("beyond file end" in issue for issue in result.issues)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestRealWorldScenarios:
    """
    Test complete workflows that represent actual usage patterns.
    
    These tests ensure the system works end-to-end for real use cases.
    """
    
    def test_document_evolution(self, sample_texts, create_version_sequence):
        """
        Test a document evolving through multiple edits.
        
        This simulates the most common use case.
        """
        # Start with a document and evolve it
        versions = create_version_sequence(sample_texts["document"], num_versions=10)
        
        # Build complete history
        base_content = versions[0]
        diff_history = []
        
        for i in range(1, len(versions)):
            old = versions[i-1]
            new = versions[i]
            version_num = i + 1
            
            if DiffEngine.should_create_snapshot(version_num):
                diff_history.append(DiffEngine.create_snapshot_version(new, version_num))
            else:
                diff_history.append(DiffEngine.create_diff_version(old, new, version_num))
        
        # Verify we can get any version back
        for i in range(len(versions)):
            version_num = i + 1
            reconstructed = DiffEngine.reconstruct_version(base_content, diff_history, version_num)
            assert reconstructed == versions[i]
        
        # Verify storage efficiency
        total_diff_size = sum(h.get("diff_size", 0) for h in diff_history if h["type"] == "diff")
        total_content_size = sum(len(v) for v in versions)
        
        # Small edits should compress well
        assert total_diff_size < total_content_size * 0.3
    
    def test_code_file_evolution(self, sample_texts):
        """
        Test version control for code files.
        
        Code has specific patterns: small frequent changes.
        """
        versions = [
            # Version 1: Initial code
            sample_texts["code"],
            
            # Version 2: Add parameter
            sample_texts["code"].replace("def process_data(items):", 
                                       "def process_data(items, validate=True):"),
            
            # Version 3: Add validation check
            sample_texts["code"].replace("if item.valid:",
                                       "if validate and item.valid:"),
            
            # Version 4: Add docstring detail
            sample_texts["code"].replace("'''Process a list of items.'''",
                                       "'''Process a list of items.\n    \n    Args:\n        items: List of items to process\n    '''"),
        ]
        
        # Build history
        base_content = versions[0]
        diff_history = []
        
        for i in range(1, len(versions)):
            diff = DiffEngine.create_diff_version(versions[i-1], versions[i], i+1)
            diff_history.append(diff)
        
        # Verify all versions work
        for i, expected in enumerate(versions):
            result = DiffEngine.reconstruct_version(base_content, diff_history, i+1)
            assert result == expected
    
    def test_corrupted_diff_recovery_via_snapshot(self):
        """
        Test that snapshots enable recovery from corruption.
        
        This is why we have snapshots - corruption resilience.
        """
        # Create 15 versions to get snapshot at 11
        versions = [f"Content for version {i+1}\n" for i in range(15)]
        
        base_content = versions[0]
        diff_history = []
        
        for i in range(1, len(versions)):
            version_num = i + 1
            
            if DiffEngine.should_create_snapshot(version_num):
                diff_history.append(DiffEngine.create_snapshot_version(versions[i], version_num))
            else:
                diff_history.append(DiffEngine.create_diff_version(versions[i-1], versions[i], version_num))
        
        # Corrupt diff at version 7
        for h in diff_history:
            if h.get("version") == 7 and h["type"] == "diff":
                h["forward_operations"] = ["@@ -99,1 +99,1 @@", "  Corrupted"]
                break
        
        # Can't reconstruct version 8 (depends on corrupted 7)
        with pytest.raises(DiffCorruptionError):
            DiffEngine.reconstruct_version(base_content, diff_history, 8)
        
        # But CAN reconstruct version 13 (uses snapshot at 11)
        result = DiffEngine.reconstruct_version(base_content, diff_history, 13)
        assert result == versions[12]


# =============================================================================
# PERFORMANCE TESTS (only what matters)
# =============================================================================

class TestPerformance:
    """
    Test performance for realistic scenarios.
    
    We only test performance that would actually impact users.
    """
    
    def test_large_file_handling(self):
        """
        Test that oversized content is rejected rather than creating bloated diffs.
        
        This prevents storage bloat from inappropriate content.
        """
        # 1MB single line - inappropriate for line-based diffing
        large_content = "x" * (1024 * 1024)
        modified = large_content[:-100] + "y" * 100
        
        # Should reject oversized diff creation with clear error
        with pytest.raises(ValueError) as exc_info:
            DiffEngine.create_diff_version(large_content, modified, 2)
        
        # Error should be informative
        error_msg = str(exc_info.value).lower()
        assert "diff too large" in error_msg
        assert "oversized lines" in error_msg
    
    def test_many_versions_performance(self):
        """
        Test performance with 100 versions.
        
        This is a realistic number for active documents.
        """
        versions = []
        for i in range(100):
            content = f"Document version {i+1}\n"
            content += "Common content that doesn't change\n" * 20
            content += f"Changing line: {i * 100}\n"
            versions.append(content)
        
        # Build history
        base_content = versions[0]
        diff_history = []
        
        start = time.time()
        for i in range(1, len(versions)):
            version_num = i + 1
            
            if DiffEngine.should_create_snapshot(version_num):
                diff_history.append(DiffEngine.create_snapshot_version(versions[i], version_num))
            else:
                diff_history.append(DiffEngine.create_diff_version(versions[i-1], versions[i], version_num))
        
        build_time = time.time() - start
        assert build_time < 5.0, f"Building 100 versions took {build_time:.2f}s"
        
        # Test reconstruction is fast
        start = time.time()
        result = DiffEngine.reconstruct_version(base_content, diff_history, 95)
        recon_time = time.time() - start
        
        assert recon_time < 0.5, f"Reconstruction took {recon_time:.2f}s"
        assert result == versions[94]


# =============================================================================
# UTILITY TESTS
# =============================================================================

class TestUtilities:
    """
    Test utility methods that support the main functionality.
    """
    
    def test_checksum_consistency(self):
        """
        Test that checksums are deterministic and unique.
        
        Critical for integrity checking.
        """
        content1 = "Test content\n"
        content2 = "Different content\n"
        
        # Same content = same checksum
        check1a = DiffEngine._content_checksum(content1)
        check1b = DiffEngine._content_checksum(content1)
        assert check1a == check1b
        
        # Different content = different checksum
        check2 = DiffEngine._content_checksum(content2)
        assert check1a != check2
        
        # Format is correct
        assert len(check1a) == 16
        assert all(c in "0123456789abcdef" for c in check1a)
    
    def test_reconstruction_plan(self):
        """
        Test planning optimal reconstruction paths.
        
        This optimizes performance for large histories.
        """
        # Create history with snapshot at 11
        diff_history = []
        for v in range(2, 15):
            if v == 11:
                diff_history.append({"type": "snapshot", "version": v})
            else:
                diff_history.append({"type": "diff", "version": v})
        
        # Plan should use snapshot
        plan = DiffEngine.get_reconstruction_plan(diff_history, target_version=14)
        
        assert plan["start_version"] == 11
        assert plan["start_type"] == "snapshot"
        assert plan["operations_needed"] == 3  # 11->12->13->14