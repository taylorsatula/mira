"""
The frozen identity of one continuum: `id`, owning `user_id`, `metadata`.

`ContinuumState` is frozen at the attribute layer only. `metadata` is an
ordinary mutable dict, so freezing stops attribute rebinding and nothing
else, and `to_dict` hands out that same dict rather than a copy — a caller
that persists the result and then mutates the state has mutated the record
it believed it had already written.

`ContinuumState.to_dict`/`from_dict` have no caller today: the repository
builds `ContinuumState` directly from its query rows, and `Continuum`'s own
`to_dict`/`from_dict` reach them only from each other. Treat the pair as
unexercised surface, not as a working round-trip.
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

    # ACCEPTED-RESIDUAL: `to_dict`/`from_dict` have no caller anywhere in the repo;
    # the repository builds `ContinuumState` straight from its query rows.
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