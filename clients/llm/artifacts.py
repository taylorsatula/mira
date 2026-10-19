"""The one persistence hook a dialect needs to land provider-produced files.

`FileArtifactSink` is a `Protocol`, and dialects take it as a constructor
argument (`from_selection(..., artifact_sink=...)`), so the implementation
never imports this module: `utils.artifact_store.UserArtifactStore` satisfies
it structurally by declaring `save_file_artifact`. The consequence a caller
must hold onto is that `artifact_sink` is legitimately `None` — a dialect that
can produce files then has no destination for them, and what happens next is
that dialect's own choice rather than anything this contract enforces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class FileArtifactSink(Protocol):
    """Stores provider-produced file artifacts for later user retrieval."""

    def save_file_artifact(
        self,
        *,
        file_id: str,
        filename: str,
        mime_type: str,
        size_bytes: int,
        content: bytes,
    ) -> Path:
        """Persist a provider-produced file artifact and return the content path."""
