"""User-scoped artifact storage for generated files.

Each artifact lands under ``data/users/{user_id}/artifacts/{file_id}/`` as a
UUID-named ``.bin`` plus a ``.meta`` JSON holding the sanitized filename and
MIME type. The user comes from the ``user_id`` contextvar, so a caller outside
a request raises before anything is written rather than writing to a default
owner. ``file_id`` must match ``FILE_ID_PATTERN`` — it becomes a directory
name, and an unchecked id would let a caller escape the user's tree.

Both the declared size and the byte length are checked against
``MAX_FILE_ARTIFACT_SIZE`` before the write: the declared size is what the
caller promised, the byte length is what it actually passed, and checking only
the former stores an oversized file under a small claim.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

from utils.user_context import get_current_user_id

MAX_FILE_ARTIFACT_SIZE = 32 * 1024 * 1024
FILE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """Sanitize filename from external source for safe filesystem storage."""

    safe = Path(filename).name
    safe = safe.replace("\x00", "").replace("\n", "").replace("\r", "")
    safe = safe.lstrip(".")
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", safe)
    if len(safe) > max_length:
        stem, sep, ext = safe.rpartition(".")
        if sep and len(ext) + 1 < max_length:
            safe = stem[:max_length - len(ext) - 1] + "." + ext
        else:
            safe = safe[:max_length]
    return safe or "file"


class UserArtifactStore:
    """Persists generated artifacts under the current user's artifact directory."""

    def save_file_artifact(
        self,
        *,
        file_id: str,
        filename: str,
        mime_type: str,
        size_bytes: int,
        content: bytes,
    ) -> Path:
        if not FILE_ID_PATTERN.match(file_id):
            raise ValueError(f"Invalid file artifact ID: {file_id}")
        if size_bytes > MAX_FILE_ARTIFACT_SIZE:
            raise ValueError(f"File too large: {size_bytes} bytes (max {MAX_FILE_ARTIFACT_SIZE})")
        if len(content) > MAX_FILE_ARTIFACT_SIZE:
            raise ValueError(f"File too large: {len(content)} bytes (max {MAX_FILE_ARTIFACT_SIZE})")

        user_id = str(get_current_user_id())
        artifacts_dir = Path("data/users") / user_id / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        file_dir = artifacts_dir / file_id
        file_dir.mkdir(exist_ok=True)

        random_stem = uuid4().hex
        content_path = file_dir / f"{random_stem}.bin"
        content_path.resolve().relative_to(file_dir.resolve())
        content_path.write_bytes(content)

        meta_path = file_dir / f"{random_stem}.meta"
        meta_path.write_text(json.dumps({
            "filename": sanitize_filename(filename),
            "mime_type": mime_type,
        }))
        return content_path
