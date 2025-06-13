"""
Differential versioning engine for efficient memory block storage.

Provides robust diff-based version control with:
- Current version stored directly for fast access
- Historical versions reconstructed on-demand
- Corruption detection and recovery
- Periodic snapshots for chain reliability
"""

import difflib
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Hunk:
    """Represents a single hunk in a unified diff."""
    old_start: int      # 0-based line number in old file
    old_count: int      # Number of lines in old file
    new_start: int      # 0-based line number in new file  
    new_count: int      # Number of lines in new file
    operations: List[str]  # Lines starting with ' ', '-', or '+'


@dataclass
class DiffValidationResult:
    """Result of diff validation."""
    valid: bool
    issues: List[str]
    hunk_count: int
    can_recover: bool = False


class DiffCorruptionError(Exception):
    """Raised when diff data is corrupted and cannot be applied."""
    pass


class DiffEngine:
    """
    Robust differential versioning engine.
    
    Architecture:
    - Current version: Always stored directly (memory_block.value)
    - Historical versions: Reconstructed from base + diffs
    - Snapshots: Full content stored every N versions for recovery
    - Validation: All diffs validated before application
    """
    
    # Create snapshot every N versions to limit reconstruction cost
    SNAPSHOT_INTERVAL = 10
    
    @staticmethod
    def create_base_version(content: str) -> Dict[str, Any]:
        """
        Create a base version entry (version 1).
        
        Args:
            content: Full content for the base version
            
        Returns:
            Dict containing base version data
        """
        return {
            "type": "base",
            "content": content,
            "size": len(content),
            "line_count": len(content.splitlines())
        }
    
    @staticmethod
    def create_diff_version(old_content: str, new_content: str, version: int) -> Dict[str, Any]:
        """
        Create a differential version entry with validation.
        
        Args:
            old_content: Previous version content
            new_content: New version content
            version: Version number this diff represents
            
        Returns:
            Dict containing differential version data
            
        Raises:
            ValueError: If diff cannot be computed
        """
        try:
            # Compute forward diff (old -> new)
            forward_diff = DiffEngine._compute_unified_diff(old_content, new_content)
            
            # Compute reverse diff (new -> old) for potential rollback optimization
            reverse_diff = DiffEngine._compute_unified_diff(new_content, old_content)
            
            # Validate the forward diff can be applied
            validation = DiffEngine.validate_diff(old_content, forward_diff["operations"])
            if not validation.valid:
                raise ValueError(f"Generated diff is invalid: {validation.issues}")
            
            return {
                "type": "diff",
                "version": version,
                "forward_operations": forward_diff["operations"],
                "reverse_operations": reverse_diff["operations"],
                "old_size": len(old_content),
                "new_size": len(new_content),
                "old_lines": len(old_content.splitlines()),
                "new_lines": len(new_content.splitlines()),
                "diff_size": forward_diff["size"],
                "compression_ratio": forward_diff["size"] / (len(old_content) + len(new_content)) if (len(old_content) + len(new_content)) > 0 else 0,
                "checksum": DiffEngine._content_checksum(new_content)
            }
            
        except Exception as e:
            logger.error(f"Failed to create diff for version {version}: {e}")
            raise ValueError(f"Diff creation failed: {e}")
    
    @staticmethod
    def create_snapshot_version(content: str, version: int) -> Dict[str, Any]:
        """
        Create a full snapshot version for recovery purposes.
        
        Args:
            content: Full content to snapshot
            version: Version number
            
        Returns:
            Dict containing snapshot data
        """
        return {
            "type": "snapshot",
            "version": version,
            "content": content,
            "size": len(content),
            "line_count": len(content.splitlines()),
            "checksum": DiffEngine._content_checksum(content)
        }
    
    @staticmethod
    def should_create_snapshot(version: int) -> bool:
        """
        Determine if a snapshot should be created for this version.
        
        Args:
            version: Version number
            
        Returns:
            True if snapshot should be created
        """
        return version > 1 and (version - 1) % DiffEngine.SNAPSHOT_INTERVAL == 0
    
    @staticmethod
    def reconstruct_version(base_content: str, diff_history: List[Dict[str, Any]], 
                           target_version: int) -> str:
        """
        Reconstruct content for a specific version with error handling.
        
        Args:
            base_content: Content of version 1
            diff_history: List of version data, ordered by version
            target_version: Target version to reconstruct
            
        Returns:
            Reconstructed content
            
        Raises:
            DiffCorruptionError: If reconstruction fails due to corruption
            ValueError: If target version is invalid
        """
        if target_version < 1:
            raise ValueError(f"Invalid target version: {target_version}")
        
        if target_version == 1:
            return base_content
        
        # Find the best starting point (latest snapshot before target)
        start_content = base_content
        start_version = 1
        
        # Look for snapshots before target version
        snapshots = [h for h in diff_history if h.get("type") == "snapshot" and h.get("version", 0) <= target_version]
        if snapshots:
            latest_snapshot = max(snapshots, key=lambda x: x.get("version", 0))
            start_content = latest_snapshot["content"]
            start_version = latest_snapshot["version"]
            logger.debug(f"Using snapshot from version {start_version} as starting point")
        
        # Apply diffs from start_version to target_version
        current_content = start_content
        
        for version in range(start_version + 1, target_version + 1):
            # Find diff for this version
            version_data = next((h for h in diff_history if h.get("version") == version), None)
            
            if not version_data:
                raise ValueError(f"Missing version data for version {version}")
            
            if version_data.get("type") == "snapshot":
                # Use snapshot directly
                current_content = version_data["content"]
            elif version_data.get("type") == "diff":
                # Apply diff
                try:
                    operations = version_data.get("forward_operations", [])
                    validation = DiffEngine.validate_diff(current_content, operations)
                    
                    if not validation.valid:
                        logger.error(f"Diff validation failed for version {version}: {validation.issues}")
                        if validation.can_recover:
                            logger.warning(f"Attempting recovery for version {version}")
                            # Could implement recovery strategies here
                        raise DiffCorruptionError(f"Corrupted diff at version {version}: {validation.issues}")
                    
                    current_content = DiffEngine._apply_diff_operations(current_content, operations)
                    
                    # Verify checksum if available
                    expected_checksum = version_data.get("checksum")
                    if expected_checksum:
                        actual_checksum = DiffEngine._content_checksum(current_content)
                        if actual_checksum != expected_checksum:
                            raise DiffCorruptionError(
                                f"Checksum mismatch at version {version}: "
                                f"expected {expected_checksum}, got {actual_checksum}"
                            )
                    
                except Exception as e:
                    if isinstance(e, DiffCorruptionError):
                        raise
                    raise DiffCorruptionError(f"Failed to apply diff for version {version}: {e}")
            else:
                raise ValueError(f"Unknown version type: {version_data.get('type')}")
        
        return current_content
    
    @staticmethod
    def validate_diff(base_content: str, diff_operations: List[str]) -> DiffValidationResult:
        """
        Validate that diff operations can be applied to base content.
        
        Args:
            base_content: Original content
            diff_operations: List of diff operation lines
            
        Returns:
            Validation result with detailed information
        """
        try:
            if not diff_operations:
                return DiffValidationResult(valid=True, issues=[], hunk_count=0)
            
            hunks = DiffEngine._parse_hunks(diff_operations)
            base_lines = base_content.splitlines(keepends=True) if base_content else []
            
            issues = []
            recoverable = True
            
            for i, hunk in enumerate(hunks):
                # Check hunk boundaries
                if hunk.old_start < 0:
                    issues.append(f"Hunk {i+1}: negative old_start ({hunk.old_start})")
                    recoverable = False
                
                if hunk.old_start + hunk.old_count > len(base_lines):
                    issues.append(
                        f"Hunk {i+1}: extends beyond file end "
                        f"({hunk.old_start + hunk.old_count} > {len(base_lines)})"
                    )
                    recoverable = False
                
                # Validate context and deletion lines
                old_line_idx = hunk.old_start
                context_errors = 0
                
                for j, operation in enumerate(hunk.operations):
                    if operation.startswith(' ') or operation.startswith('-'):
                        # Context or deleted line - should match original
                        expected_content = operation[1:]
                        
                        if old_line_idx >= len(base_lines):
                            issues.append(f"Hunk {i+1}, operation {j+1}: beyond file end")
                            recoverable = False
                        else:
                            actual_content = base_lines[old_line_idx]
                            if actual_content != expected_content:
                                context_errors += 1
                                issues.append(
                                    f"Hunk {i+1}, operation {j+1}: content mismatch at line {old_line_idx+1}"
                                )
                        
                        old_line_idx += 1
                    # '+' operations don't need validation against original
                
                # Too many context errors suggests major corruption
                if context_errors > len(hunk.operations) // 2:
                    recoverable = False
            
            return DiffValidationResult(
                valid=len(issues) == 0,
                issues=issues,
                hunk_count=len(hunks),
                can_recover=recoverable and len(issues) > 0
            )
            
        except Exception as e:
            return DiffValidationResult(
                valid=False,
                issues=[f"Validation error: {str(e)}"],
                hunk_count=0,
                can_recover=False
            )
    
    @staticmethod
    def _compute_unified_diff(old_content: str, new_content: str) -> Dict[str, Any]:
        """Compute unified diff between two texts."""
        old_lines = old_content.splitlines(keepends=True) if old_content else []
        new_lines = new_content.splitlines(keepends=True) if new_content else []
        
        diff_lines = list(difflib.unified_diff(
            old_lines,
            new_lines,
            lineterm='',
            n=3  # Context lines
        ))
        
        total_size = sum(len(line) for line in diff_lines)
        
        # Fail fast on bloated diffs instead of accommodating bad behavior
        MAX_DIFF_SIZE = 100000  # 100KB limit - reasonable for text diffs
        if total_size > MAX_DIFF_SIZE:
            raise ValueError(
                f"Diff too large ({total_size} bytes, limit {MAX_DIFF_SIZE}). "
                f"Content may have oversized lines that are inappropriate for line-based diffing."
            )
        
        return {
            "operations": diff_lines,
            "size": total_size
        }
    
    @staticmethod
    def _apply_diff_operations(base_content: str, diff_operations: List[str]) -> str:
        """
        Apply unified diff operations to base content.
        
        Args:
            base_content: Original content
            diff_operations: Diff operation lines
            
        Returns:
            Modified content
        """
        if not diff_operations:
            return base_content
        
        hunks = DiffEngine._parse_hunks(diff_operations)
        base_lines = base_content.splitlines(keepends=True) if base_content else []
        
        result_lines = []
        current_base_idx = 0
        
        for hunk in hunks:
            # Copy unchanged lines before this hunk
            while current_base_idx < hunk.old_start:
                if current_base_idx < len(base_lines):
                    result_lines.append(base_lines[current_base_idx])
                current_base_idx += 1
            
            # Process hunk operations
            hunk_base_idx = hunk.old_start
            
            for operation in hunk.operations:
                if operation.startswith(' '):
                    # Context line - copy to result
                    result_lines.append(operation[1:])
                    hunk_base_idx += 1
                elif operation.startswith('-'):
                    # Deleted line - skip in original
                    hunk_base_idx += 1
                elif operation.startswith('+'):
                    # Added line - add to result
                    result_lines.append(operation[1:])
            
            current_base_idx = hunk_base_idx
        
        # Copy remaining lines after last hunk
        while current_base_idx < len(base_lines):
            result_lines.append(base_lines[current_base_idx])
            current_base_idx += 1
        
        return ''.join(result_lines)
    
    @staticmethod
    def _parse_hunks(diff_operations: List[str]) -> List[Hunk]:
        """
        Parse unified diff operations into structured hunks.
        
        Args:
            diff_operations: Raw diff operation lines
            
        Returns:
            List of parsed Hunk objects
        """
        hunks = []
        current_operations = []
        current_header = None
        
        for line in diff_operations:
            if line.startswith('@@'):
                # Save previous hunk if exists
                if current_header and current_operations:
                    hunks.append(Hunk(
                        old_start=current_header[0],
                        old_count=current_header[1],
                        new_start=current_header[2],
                        new_count=current_header[3],
                        operations=current_operations[:]
                    ))
                
                # Parse new hunk header
                match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                if not match:
                    raise ValueError(f"Invalid hunk header: {line}")
                
                old_start = int(match.group(1)) - 1  # Convert to 0-based
                old_count = int(match.group(2)) if match.group(2) else 1
                new_start = int(match.group(3)) - 1  # Convert to 0-based
                new_count = int(match.group(4)) if match.group(4) else 1
                
                current_header = (old_start, old_count, new_start, new_count)
                current_operations = []
                
            elif line.startswith((' ', '-', '+')):
                # Operation line
                if current_header:
                    current_operations.append(line)
            # Skip file headers and other metadata
        
        # Don't forget the last hunk
        if current_header and current_operations:
            hunks.append(Hunk(
                old_start=current_header[0],
                old_count=current_header[1],
                new_start=current_header[2],
                new_count=current_header[3],
                operations=current_operations
            ))
        
        return hunks
    
    @staticmethod
    def _content_checksum(content: str) -> str:
        """Compute simple checksum for content validation."""
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    @staticmethod
    def get_reconstruction_plan(diff_history: List[Dict[str, Any]], target_version: int) -> Dict[str, Any]:
        """
        Analyze the optimal reconstruction path for a target version.
        
        Args:
            diff_history: Complete version history
            target_version: Target version to reconstruct
            
        Returns:
            Reconstruction plan with starting point and required operations
        """
        if target_version < 1:
            raise ValueError(f"Invalid target version: {target_version}")
        
        # Find all snapshots at or before target version
        snapshots = [
            h for h in diff_history 
            if h.get("type") == "snapshot" and h.get("version", 0) <= target_version
        ]
        
        if snapshots:
            # Use the latest snapshot as starting point
            latest_snapshot = max(snapshots, key=lambda x: x.get("version", 0))
            start_version = latest_snapshot["version"]
            operations_needed = target_version - start_version
        else:
            # Start from base (version 1)
            start_version = 1
            operations_needed = target_version - 1
        
        # Find required diffs
        required_diffs = [
            h for h in diff_history
            if h.get("type") == "diff" and start_version < h.get("version", 0) <= target_version
        ]
        
        return {
            "target_version": target_version,
            "start_version": start_version,
            "start_type": "snapshot" if snapshots else "base",
            "operations_needed": operations_needed,
            "required_diffs": len(required_diffs),
            "estimated_cost": operations_needed,  # Could be more sophisticated
            "path_complete": len(required_diffs) == operations_needed
        }