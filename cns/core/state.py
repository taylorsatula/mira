"""
Immutable state management for CNS continuums.

Provides immutable state objects and controlled state transitions
for continuum data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict
from uuid import UUID


class ContinuumStateDict(TypedDict):
    """Serialized form of ContinuumState for persistence round-trips."""
    id: str
    user_id: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class ContinuumState:
    """
    Immutable continuum state.

    Represents all continuum data with immutable updates only.
    No direct mutations allowed - use with_* methods for state changes.
    """
    id: UUID
    user_id: str

    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> ContinuumStateDict:
        """Convert state to dictionary for persistence."""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: ContinuumStateDict) -> ContinuumState:
        """Create state from dictionary."""
        return cls(
            id=UUID(data["id"]),
            user_id=data["user_id"],
            metadata=data.get("metadata", {})
        )