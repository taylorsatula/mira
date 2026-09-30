"""
Performance monitoring instrumentation for MIRA.

All perf logic lives here — no instrumentation code is scattered through business
logic files. Database execute_* methods are wrapped externally via monkey-patching
at startup (install_db_instrumentation), matching the pattern used by APM tools.

Three-tier behavior gated by the ``mira.perf`` logger level:
    DEBUG   — full query-level detail, N+1 detection, per-request breakdowns
    INFO    — single summary line per request (query count + DB time)
    WARNING+— silent, near-zero overhead (wrappers not installed, middleware early-returns)

Usage:
    # In main.py at startup:
    from utils.perf import PerfMiddleware, register_perf_routes, install_db_instrumentation
    app.add_middleware(PerfMiddleware)
    register_perf_routes(app)
    install_db_instrumentation()

    # Optional: decorate suspect methods
    from utils.perf import perf_profile

    @perf_profile
    def slow_method(self, data):
        ...
"""

import cProfile
import contextvars
import functools
import io
import logging
import os
import pstats
import re
import time
import traceback
import threading
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger("mira.perf")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PLUS_ONE_THRESHOLD = 4
SLOW_METHOD_THRESHOLD_MS = 500
HIGH_FREQ_THRESHOLD = 50
HIGH_FREQ_WINDOW_S = 60
ROLLING_WINDOW_MAX = 500
ROLLING_WINDOW_TTL_S = 600  # 10 minutes

# Frames to skip when extracting the business-logic caller.
# Use os.sep-normalized paths for cross-platform correctness.
_INFRA_FILES = frozenset({
    os.path.join("utils", "perf.py"),
    os.path.join("clients", "postgres_client.py"),
    os.path.join("utils", "database_session_manager.py"),
})

# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------
_RE_SINGLE_QUOTED = re.compile(r"'[^']*'")
_RE_NUMERIC = re.compile(r"\b\d+\.?\d*\b")
_RE_IN_LIST = re.compile(r"IN\s*\([^)]+\)", re.IGNORECASE)


def _normalize_query(sql: str) -> str:
    """Normalize SQL for pattern grouping: strip literals, preserve structure."""
    result = _RE_SINGLE_QUOTED.sub("?", sql)
    result = _RE_NUMERIC.sub("?", result)
    result = _RE_IN_LIST.sub("IN (?)", result)
    # Collapse whitespace
    result = " ".join(result.split())
    return result


# ---------------------------------------------------------------------------
# Caller extraction
# ---------------------------------------------------------------------------

def _extract_caller() -> str:
    """Return 'relative/path:line' of the first business-logic frame in the stack.

    Walks the stack backwards, skipping infrastructure frames (this module,
    postgres_client, database_session_manager). Only called at DEBUG level.
    """
    stack = traceback.extract_stack()
    for frame in reversed(stack):
        filename = frame.filename
        # Convert to relative path for readability
        rel = filename
        for prefix in (os.getcwd() + os.sep, ):
            if filename.startswith(prefix):
                rel = filename[len(prefix):]
                break
        # Skip infrastructure frames
        if any(rel.endswith(inf) for inf in _INFRA_FILES):
            continue
        # Skip stdlib / site-packages
        if "site-packages" in filename or "/lib/python" in filename:
            continue
        return f"{rel}:{frame.lineno}"
    return "unknown"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryRecord:
    """Single query recorded during a request (DEBUG level only)."""
    query_preview: str
    duration_ms: float
    caller: str
    normalized_pattern: str


@dataclass
class RequestPerfContext:
    """Per-request accumulator set via contextvar by the middleware."""
    query_count: int = 0
    total_db_ms: float = 0.0
    records: List[QueryRecord] = field(default_factory=list)

    def detect_n_plus_one(self) -> List[Tuple[str, int, float, str]]:
        """Return (pattern, count, total_ms, most_common_caller) for N+1 candidates."""
        if not self.records:
            return []
        pattern_groups: Dict[str, List[QueryRecord]] = defaultdict(list)
        for rec in self.records:
            pattern_groups[rec.normalized_pattern].append(rec)

        results = []
        for pattern, recs in pattern_groups.items():
            if len(recs) >= N_PLUS_ONE_THRESHOLD:
                total_ms = sum(r.duration_ms for r in recs)
                callers = Counter(r.caller for r in recs)
                top_caller = callers.most_common(1)[0][0]
                results.append((pattern, len(recs), total_ms, top_caller))
        return results


_perf_context: contextvars.ContextVar[Optional[RequestPerfContext]] = contextvars.ContextVar(
    "perf_context", default=None
)


# ---------------------------------------------------------------------------
# Rolling stats window (for diagnostic endpoint)
# ---------------------------------------------------------------------------

@dataclass
class RequestStats:
    """Summary of one completed request, stored in the rolling window."""
    endpoint: str
    method: str
    status_code: int
    total_ms: float
    query_count: int
    db_ms: float
    n_plus_one_patterns: List[str]
    query_patterns: List[str]  # normalized patterns, populated at DEBUG only
    timestamp: float


_rolling_stats: deque = deque(maxlen=ROLLING_WINDOW_MAX)
_rolling_lock = threading.Lock()


def _append_rolling_stat(stat: RequestStats) -> None:
    with _rolling_lock:
        _rolling_stats.append(stat)


def _get_rolling_summary() -> Dict[str, Any]:
    """Compute aggregate stats from the rolling window for the diagnostic endpoint."""
    now = time.time()
    cutoff = now - ROLLING_WINDOW_TTL_S

    with _rolling_lock:
        recent = [s for s in _rolling_stats if s.timestamp > cutoff]

    if not recent:
        return {
            "window_info": {"total_requests": 0, "message": "No requests recorded yet"},
            "top_endpoints_by_query_count": [],
            "top_endpoints_by_response_time": [],
            "top_query_patterns": [],
            "n_plus_one_detections": [],
            "pool_stats": _collect_pool_stats(),
        }

    # Aggregate by endpoint
    endpoint_data: Dict[str, List[RequestStats]] = defaultdict(list)
    for s in recent:
        key = f"{s.method} {s.endpoint}"
        endpoint_data[key].append(s)

    # Top by avg query count
    by_query_count = sorted(
        [
            {"endpoint": ep, "avg_queries": sum(s.query_count for s in stats) / len(stats),
             "request_count": len(stats)}
            for ep, stats in endpoint_data.items()
        ],
        key=lambda x: x["avg_queries"],
        reverse=True,
    )[:10]

    # Top by avg response time
    by_response_time = sorted(
        [
            {"endpoint": ep, "avg_ms": round(sum(s.total_ms for s in stats) / len(stats), 1),
             "request_count": len(stats)}
            for ep, stats in endpoint_data.items()
        ],
        key=lambda x: x["avg_ms"],
        reverse=True,
    )[:10]

    # Top query patterns (populated at DEBUG level only)
    all_patterns: List[str] = []
    for s in recent:
        all_patterns.extend(s.query_patterns)
    top_patterns = Counter(all_patterns).most_common(10)

    # N+1 detections
    all_n_plus_one: List[str] = []
    for s in recent:
        all_n_plus_one.extend(s.n_plus_one_patterns)
    n_plus_one_counts = Counter(all_n_plus_one).most_common(10)

    return {
        "window_info": {
            "total_requests": len(recent),
            "window_seconds": round(now - min(s.timestamp for s in recent), 1),
            "oldest": min(s.timestamp for s in recent),
            "newest": max(s.timestamp for s in recent),
        },
        "top_endpoints_by_query_count": by_query_count,
        "top_endpoints_by_response_time": by_response_time,
        "top_query_patterns": [
            {"pattern": p[:120], "count": c} for p, c in top_patterns
        ],
        "n_plus_one_detections": [
            {"pattern": p, "occurrences": c} for p, c in n_plus_one_counts
        ],
        "pool_stats": _collect_pool_stats(),
    }


def _collect_pool_stats() -> Dict[str, Any]:
    """Gather pool stats from both DB clients. Import inline to avoid circular deps."""
    stats: Dict[str, Any] = {}
    try:
        from clients.postgres_client import PostgresClient
        stats["postgres_client"] = PostgresClient.get_pool_stats()
    except Exception:
        stats["postgres_client"] = "unavailable"
    try:
        from utils.database_session_manager import get_shared_session_manager
        stats["session_manager"] = get_shared_session_manager().get_pool_stats()
    except Exception:
        stats["session_manager"] = "unavailable"
    return stats


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class PerfMiddleware(BaseHTTPMiddleware):
    """Request-level performance tracking middleware.

    At WARNING+: early-returns with zero overhead.
    At INFO: logs a single summary line per request.
    At DEBUG: logs full query breakdown with N+1 detection.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not logger.isEnabledFor(logging.INFO):
            return await call_next(request)

        ctx = RequestPerfContext()
        token = _perf_context.set(ctx)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000
            self._log_request(request, response.status_code, duration_ms, ctx)
            return response
        finally:
            _perf_context.reset(token)

    def _log_request(
        self, request: Request, status_code: int, duration_ms: float, ctx: RequestPerfContext
    ) -> None:
        """Log request performance and append to rolling stats."""
        if ctx.query_count == 0:
            return  # Skip requests with no DB activity (health, static, etc.)

        method = request.method
        path = request.url.path
        db_pct = (ctx.total_db_ms / duration_ms * 100) if duration_ms > 0 else 0

        # Summary line (both INFO and DEBUG)
        logger.info(
            "%s %s | %d | %.0fms total | %d queries | %.0fms db (%.0f%%)",
            method, path, status_code, duration_ms,
            ctx.query_count, ctx.total_db_ms, db_pct,
        )

        n_plus_one_patterns: List[str] = []
        query_patterns: List[str] = []

        # Detailed breakdown (DEBUG only)
        if logger.isEnabledFor(logging.DEBUG) and ctx.records:
            query_patterns = [rec.normalized_pattern for rec in ctx.records]
            # Group queries by caller
            caller_groups: Dict[str, List[QueryRecord]] = defaultdict(list)
            for rec in ctx.records:
                caller_groups[rec.caller].append(rec)

            for caller, recs in sorted(caller_groups.items(), key=lambda x: -sum(r.duration_ms for r in x[1])):
                total_ms = sum(r.duration_ms for r in recs)
                preview = recs[0].query_preview[:60]
                logger.debug(
                    "  %-45s | %d calls | %.0fms | %s",
                    caller, len(recs), total_ms, preview,
                )

            # N+1 detection
            for pattern, count, total_ms, caller in ctx.detect_n_plus_one():
                n_plus_one_patterns.append(f"{pattern[:80]} ({caller})")
                logger.warning(
                    "  N+1 DETECTED | %d calls | %.0fms | %s | %s",
                    count, total_ms, pattern[:80], caller,
                )

        # Append to rolling window
        _append_rolling_stat(RequestStats(
            endpoint=path,
            method=method,
            status_code=status_code,
            total_ms=duration_ms,
            query_count=ctx.query_count,
            db_ms=ctx.total_db_ms,
            n_plus_one_patterns=n_plus_one_patterns,
            query_patterns=query_patterns,
            timestamp=time.time(),
        ))


# ---------------------------------------------------------------------------
# Database instrumentation (monkey-patch installer)
# ---------------------------------------------------------------------------

def _make_instrumented_wrapper(original: Callable, method_name: str) -> Callable:
    """Create an instrumented wrapper for a database execute_* method."""

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        # Fast path: if logging is at WARNING+, call original directly.
        # This check is ~50ns. The wrapper itself only exists because
        # install_db_instrumentation() was called at INFO/DEBUG level.
        if not logger.isEnabledFor(logging.INFO):
            return original(self, *args, **kwargs)

        # Extract query string for logging
        if method_name == "execute_transaction":
            # execute_transaction takes List[Tuple[query, params]]
            operations = args[0] if args else kwargs.get("operations", [])
            query_str = f"TRANSACTION ({len(operations)} ops)"
        else:
            query_str = args[0] if args else kwargs.get("query", "")

        start = time.perf_counter()
        result = original(self, *args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000

        ctx = _perf_context.get(None)
        if ctx is not None:
            ctx.query_count += 1
            ctx.total_db_ms += duration_ms

            if logger.isEnabledFor(logging.DEBUG):
                caller = _extract_caller()
                normalized = _normalize_query(query_str)
                ctx.records.append(QueryRecord(
                    query_preview=query_str[:200],
                    duration_ms=duration_ms,
                    caller=caller,
                    normalized_pattern=normalized,
                ))
                logger.debug(
                    "%.1fms | %s | %s",
                    duration_ms, query_str[:80], caller,
                )

        return result

    return wrapper


def install_db_instrumentation() -> None:
    """Wrap all execute_* methods on DB classes with perf instrumentation.

    Called once at startup from main.py. At WARNING+ log level, does nothing —
    the original methods run completely unmodified.
    """
    if not logger.isEnabledFor(logging.INFO):
        logger.debug("Perf instrumentation skipped (log level >= WARNING)")
        return

    from clients.postgres_client import PostgresClient
    from utils.database_session_manager import LTMemorySession, AdminSession

    # (class, [method_names]) — skip execute_single since it delegates to execute_query
    targets = [
        (PostgresClient, [
            "execute_query", "execute_returning", "execute_scalar",
            "execute_insert", "execute_update", "execute_transaction",
        ]),
        (LTMemorySession, [
            "execute_query", "execute_bulk_insert",
            "execute_update", "execute_delete",
        ]),
        (AdminSession, [
            "execute_query", "execute_update", "execute_delete",
        ]),
    ]

    wrapped_count = 0
    for cls, method_names in targets:
        for method_name in method_names:
            original = getattr(cls, method_name)
            wrapped = _make_instrumented_wrapper(original, method_name)
            setattr(cls, method_name, wrapped)
            wrapped_count += 1

    logger.info("DB instrumentation installed: %d methods wrapped", wrapped_count)


# ---------------------------------------------------------------------------
# @perf_profile decorator
# ---------------------------------------------------------------------------

# Track high-frequency calls per function
_freq_tracker: Dict[str, Tuple[int, float]] = {}
_freq_lock = threading.Lock()


def perf_profile(fn: Callable) -> Callable:
    """Decorator for targeted method profiling.

    DEBUG  — cProfile each call, log top 5 internal calls by cumtime
    INFO   — time-only, log if call exceeds SLOW_METHOD_THRESHOLD_MS
    WARNING+ — passes through to original function with zero wrapping overhead
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not logger.isEnabledFor(logging.INFO):
            return fn(*args, **kwargs)

        qualname = fn.__qualname__

        # High-frequency tracking
        _track_call_frequency(qualname)

        if logger.isEnabledFor(logging.DEBUG):
            return _profile_debug(fn, qualname, args, kwargs)

        # INFO: time-only, log slow calls
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        ms = (time.perf_counter() - start) * 1000
        if ms > SLOW_METHOD_THRESHOLD_MS:
            logger.info("SLOW | %.0fms | %s", ms, qualname)
        return result

    return wrapper


def _track_call_frequency(qualname: str) -> None:
    """Track call frequency; warn once per window if threshold exceeded."""
    now = time.time()
    with _freq_lock:
        count, window_start = _freq_tracker.get(qualname, (0, now))
        if now - window_start > HIGH_FREQ_WINDOW_S:
            # Start new window
            _freq_tracker[qualname] = (1, now)
            return
        count += 1
        _freq_tracker[qualname] = (count, window_start)
        if count == HIGH_FREQ_THRESHOLD:
            logger.warning(
                "HIGH FREQUENCY | %s called %d times in %.0fs window",
                qualname, count, now - window_start,
            )


def _profile_debug(fn: Callable, qualname: str, args: tuple, kwargs: dict) -> Any:
    """Run function under cProfile and log top 5 calls by cumulative time."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        result = fn(*args, **kwargs)
    finally:
        profiler.disable()

    # Extract stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")

    # Get top 5 entries
    stats.print_stats(5)
    stats_output = stream.getvalue()

    # Log total time + top calls
    total_ms = sum(
        entry[3] for entry in stats.stats.values()  # entry[3] = cumtime
    ) * 1000 if stats.stats else 0

    logger.debug("PROFILE | %.0fms | %s\n%s", total_ms, qualname, stats_output)
    return result


# ---------------------------------------------------------------------------
# Diagnostic endpoint
# ---------------------------------------------------------------------------

def register_perf_routes(app: 'FastAPI') -> None:
    """Register GET /dev/perf/summary. Only creates the route at DEBUG/INFO level.

    At WARNING+ the route simply doesn't exist (404).
    """
    if not logger.isEnabledFor(logging.INFO):
        return

    from fastapi import Depends
    from fastapi.responses import JSONResponse

    from auth.api import get_current_user
    from auth.types import SessionData, APITokenContext

    @app.get("/dev/perf/summary", tags=["dev"])
    def perf_summary(current_user: SessionData | APITokenContext = Depends(get_current_user)):
        """Rolling performance summary — recent request stats, N+1 detections, pool info."""
        summary = _get_rolling_summary()
        return JSONResponse(content=summary)

    logger.info("Perf diagnostic endpoint registered at /dev/perf/summary")
