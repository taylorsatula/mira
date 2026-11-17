"""
The frozen identity of one continuum: `id`, owning `user_id`, `metadata`.

`ContinuumState` is frozen at the attribute layer only. `metadata` is an
ordinary mutable dict, so freezing stops attribute rebinding and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True)
class ContinuumState:
    """
    Immutable continuum state.

    Represents all continuum data with immutable updates only.
    No direct mutations allowed.
    """
    id: UUID
    user_id: str

    metadata: dict[str, object] = field(default_factory=dict)
