"""
LLM traffic tap — a debugging firehose for sniffing all LLM request/response
traffic when you need to investigate an issue but don't know which code path
is generating the problematic call.

Toggle at runtime: kill -USR1 <mira_pid>
Output: llm_requests.jsonl, llm_responses.jsonl (project root, append mode)
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Toggle state — GIL-safe bool, reads from worker threads see consistent values
_active = False

# Serializes file writes from concurrent threads (streaming, background agents, etc.)
_write_lock = threading.Lock()

# Output paths in project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REQUEST_PATH = _PROJECT_ROOT / "llm_requests.jsonl"
_RESPONSE_PATH = _PROJECT_ROOT / "llm_responses.jsonl"


def is_active() -> bool:
    return _active


def toggle(signum=None, frame=None) -> None:
    """Toggle the traffic tap on/off. Compatible as a signal handler (SIGUSR1)."""
    global _active
    _active = not _active
    logger.warning("LLM traffic tap %s", "ENABLED" if _active else "DISABLED")


# ---------------------------------------------------------------------------
# Internal writer
# ---------------------------------------------------------------------------

def _write(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON line. Thread-safe via lock."""
    try:
        line = json.dumps(record, default=str, ensure_ascii=False)
        with _write_lock:
            with open(path, "a") as f:
                f.write(line)
                f.write("\n")
    except Exception:
        logger.debug("Traffic tap write failed", exc_info=True)


# ---------------------------------------------------------------------------
# Public API: requests
# ---------------------------------------------------------------------------

def log_request(
    *,
    provider: str,
    endpoint: str,
    model: str,
    body: Dict[str, Any],
) -> None:
    """Log an outbound LLM request to llm_requests.jsonl."""
    if not _active:
        return
    _write(_REQUEST_PATH, {
        "ts": time.time(),
        "provider": provider,
        "endpoint": endpoint,
        "model": model,
        "body": body,
    })


# ---------------------------------------------------------------------------
# Public API: responses
# ---------------------------------------------------------------------------

def log_response(
    *,
    provider: str,
    model: str,
    response_data: Any,
    endpoint: Optional[str] = None,
) -> None:
    """Log an inbound LLM response to llm_responses.jsonl."""
    if not _active:
        return

    if hasattr(response_data, "model_dump"):
        data = response_data.model_dump()
    elif hasattr(response_data, "__dict__"):
        data = _serialize_generic(response_data)
    else:
        data = response_data

    _write(_RESPONSE_PATH, {
        "ts": time.time(),
        "kind": "response",
        "provider": provider,
        "model": model,
        "endpoint": endpoint,
        "response": data,
    })


def log_stream_chunk(
    *,
    provider: str,
    endpoint: str,
    model: str,
    chunk: Dict[str, Any],
) -> None:
    """Log one raw provider streaming chunk before dialect normalization."""
    if not _active:
        return
    _write(_RESPONSE_PATH, {
        "ts": time.time(),
        "kind": "stream_chunk",
        "provider": provider,
        "model": model,
        "endpoint": endpoint,
        "chunk": chunk,
    })


def _serialize_generic(obj: Any) -> Dict:
    """Recursively convert SimpleNamespace-based objects to plain dicts."""
    result = {}
    for key, val in vars(obj).items():
        if hasattr(val, "__dict__") and not isinstance(val, type):
            result[key] = _serialize_generic(val)
        elif isinstance(val, list):
            result[key] = [
                _serialize_generic(item) if hasattr(item, "__dict__") and not isinstance(item, type) else item
                for item in val
            ]
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# httpx request hook (attached to Anthropic SDK clients)
# ---------------------------------------------------------------------------

def httpx_request_hook(request) -> None:
    """httpx event hook: capture full request body for Anthropic API calls.

    Attached via instrument_anthropic_client() in logging_config.py.
    request.content is always available as bytes before the request is sent.
    """
    if not _active:
        return
    try:
        body = json.loads(request.content)
        log_request(
            provider="anthropic",
            endpoint=str(request.url),
            model=body.get("model", "unknown"),
            body=body,
        )
    except Exception:
        logger.debug("Traffic tap request hook failed", exc_info=True)
