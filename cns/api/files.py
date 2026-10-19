"""
Artifact serving: downloaded tool files and inline tool images.

Each artifact is a directory under `data/users/{user_id}/artifacts/{file_id}/`
holding a `.bin` body beside a `.meta` sidecar written during response
processing. Four layers stand between a request and those bytes: the auth
dependency scopes the path to the caller's own directory, `file_id` is
regex-validated, the resolved path is re-checked as a child of the artifact
directory, and the stored MIME type is trusted for nothing — `/images` renders
inline only for `INLINE_IMAGE_MIME_TYPES` and sends `nosniff`, while any other
stored type is forced to an octet-stream attachment, so a stored `text/html`
cannot execute as stored XSS. `/files` is always an attachment.

The provider-supplied filename passes through `FILENAME_SAFE_PATTERN` before it
reaches a `Content-Disposition` header, and both routes send
`Cache-Control: no-store`.
"""
import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from auth.api import get_current_user
from auth.types import SessionData, APITokenContext

router = APIRouter()

FILE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

# Anything outside this set is forced to application/octet-stream so a provider text/html cannot execute as stored XSS.
INLINE_IMAGE_MIME_TYPES = frozenset({
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
})

# Strips quotes/newlines from the provider-supplied filename to prevent header injection.
FILENAME_SAFE_PATTERN = re.compile(r'[^\w.()\- ]')


@router.get("/files/{file_id}")
def download_file(
    file_id: str,
    current_user: SessionData | APITokenContext = Depends(get_current_user)
) -> FileResponse:
    """Download a code execution file artifact.

    Files are stored during response processing and persist across segments.
    """
    if not FILE_ID_PATTERN.match(file_id):
        raise HTTPException(status_code=400, detail="Invalid file ID")

    user_id = current_user.user_id
    base_dir = (Path("data/users") / user_id / "artifacts" / file_id).resolve()

    if not base_dir.is_dir():
        raise HTTPException(status_code=404, detail="File not found")

    bin_files = list(base_dir.glob("*.bin"))
    meta_files = list(base_dir.glob("*.meta"))
    if not bin_files or not meta_files:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = bin_files[0].resolve()
    # Path traversal guard — resolved path must be within base_dir
    file_path.relative_to(base_dir)

    meta: dict[str, str] = json.loads(meta_files[0].read_text())
    original_filename = meta.get("filename", "download")

    return FileResponse(
        path=file_path,
        filename=original_filename,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{original_filename}"',
            "Cache-Control": "no-store",
        }
    )


@router.get("/images/{file_id}")
def view_image(
    file_id: str,
    current_user: SessionData | APITokenContext = Depends(get_current_user)
) -> FileResponse:
    """Serve a tool-generated image for inline display.

    Same storage convention as /files/{file_id} (.bin + .meta sidecar)
    but returns with Content-Disposition: inline so browsers render
    the image instead of downloading it.
    """
    if not FILE_ID_PATTERN.match(file_id):
        raise HTTPException(status_code=400, detail="Invalid file ID")

    user_id = current_user.user_id
    base_dir = (Path("data/users") / user_id / "artifacts" / file_id).resolve()

    if not base_dir.is_dir():
        raise HTTPException(status_code=404, detail="Image not found")

    bin_files = list(base_dir.glob("*.bin"))
    meta_files = list(base_dir.glob("*.meta"))
    if not bin_files or not meta_files:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = bin_files[0].resolve()
    file_path.relative_to(base_dir)  # Path traversal guard

    meta: dict[str, str] = json.loads(meta_files[0].read_text())
    mime_type = (meta.get("mime_type") or "").strip().lower()
    # Never trust the stored MIME for inline rendering: only whitelisted image types stay inline; everything else is a forced download.
    safe_filename = FILENAME_SAFE_PATTERN.sub("_", meta.get("filename", "image"))
    if mime_type in INLINE_IMAGE_MIME_TYPES:
        disposition = f'inline; filename="{safe_filename}"'
    else:
        mime_type = "application/octet-stream"
        disposition = f'attachment; filename="{safe_filename}"'

    return FileResponse(
        path=file_path,
        media_type=mime_type,
        headers={
            "Content-Disposition": disposition,
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        }
    )
