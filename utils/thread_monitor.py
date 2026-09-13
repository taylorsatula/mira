"""
Thread monitoring and debugging utilities.

Provides detailed logging and monitoring for thread operations to diagnose
hanging issues in the application.
"""

import threading
import time
import traceback
import logging
import functools
import asyncio
import os
import tempfile
from datetime import datetime
from typing import Dict, Optional, List

from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor, Future
import psutil

from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


class ActiveOperation(TypedDict):
    """Internal record of a running thread operation."""
    operation: str
    thread_name: str
    start_time: datetime
    context: Dict[str, str]
    stack_trace: List[str]
    pid: int


class StuckOperationInfo(TypedDict):
    """Externalized info about a stuck operation."""
    thread_id: int
    thread_name: str
    operation: str
    duration_seconds: float
    start_time: str
    context: Dict[str, str]
    stack_trace: List[str]


class ActiveOperationInfo(TypedDict):
    """Externalized info about an active operation."""
    thread_id: int
    thread_name: str
    operation: str
    duration_seconds: float
    start_time: str
    context: Dict[str, str]


class FutureInfo(TypedDict):
    """Tracking info for a submitted ThreadPool future."""
    operation: str
    submit_time: datetime
    args_preview: str
    thread_prefix: str


# Global registry of active thread operations
_active_operations: Dict[int, ActiveOperation] = {}
_operation_lock = threading.RLock()

# Thread-local storage for operation context
_thread_local = threading.local()


class ThreadMonitor:
    """Monitors and logs thread operations for debugging hanging issues."""

    # Threshold for warning about long-running operations (seconds)
    SLOW_OPERATION_THRESHOLD = 30
    STUCK_OPERATION_THRESHOLD = 300  # 5 minutes

    @classmethod
    def start_operation(cls, operation_name: str, context: Optional[Dict[str, str]] = None) -> None:
        """
        Mark the start of a thread operation.

        Args:
            operation_name: Name/description of the operation
            context: Additional context about the operation
        """
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name

        with _operation_lock:
            _active_operations[thread_id] = {
                'operation': operation_name,
                'thread_name': thread_name,
                'start_time': utc_now(),
                'context': context or {},
                'stack_trace': traceback.format_stack(),
                'pid': os.getpid()
            }

            # Store in thread-local for nested operations
            if not hasattr(_thread_local, 'operation_stack'):
                _thread_local.operation_stack = []
            _thread_local.operation_stack.append(operation_name)

        logger.debug(
            f"Thread {thread_name} (ID: {thread_id}) started: {operation_name}",
            extra={'context': context}
        )

    @classmethod
    def end_operation(cls, operation_name: Optional[str] = None) -> None:
        """
        Mark the end of a thread operation.

        Args:
            operation_name: Name of the operation (for validation)
        """
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name

        with _operation_lock:
            if thread_id in _active_operations:
                op_info = _active_operations[thread_id]
                duration = (utc_now() - op_info['start_time']).total_seconds()

                # Log based on duration
                if duration > cls.STUCK_OPERATION_THRESHOLD:
                    logger.error(
                        f"Thread {thread_name} completed STUCK operation '{op_info['operation']}' "
                        f"after {duration:.2f} seconds",
                        extra={'context': op_info['context'], 'duration': duration}
                    )
                elif duration > cls.SLOW_OPERATION_THRESHOLD:
                    logger.warning(
                        f"Thread {thread_name} completed SLOW operation '{op_info['operation']}' "
                        f"after {duration:.2f} seconds",
                        extra={'context': op_info['context'], 'duration': duration}
                    )
                else:
                    logger.debug(
                        f"Thread {thread_name} completed '{op_info['operation']}' "
                        f"in {duration:.2f} seconds",
                        extra={'duration': duration}
                    )

                del _active_operations[thread_id]

                # Clean up thread-local stack
                if hasattr(_thread_local, 'operation_stack'):
                    if _thread_local.operation_stack and _thread_local.operation_stack[-1] == operation_name:
                        _thread_local.operation_stack.pop()

    @classmethod
    def get_stuck_operations(cls) -> list[StuckOperationInfo]:
        """Get list of operations that appear to be stuck."""
        stuck = []
        current_time = utc_now()

        with _operation_lock:
            for thread_id, op_info in _active_operations.items():
                duration = (current_time - op_info['start_time']).total_seconds()
                if duration > cls.STUCK_OPERATION_THRESHOLD:
                    stuck.append({
                        'thread_id': thread_id,
                        'thread_name': op_info['thread_name'],
                        'operation': op_info['operation'],
                        'duration_seconds': duration,
                        'start_time': op_info['start_time'].isoformat(),
                        'context': op_info['context'],
                        'stack_trace': op_info['stack_trace']
                    })

        return stuck

    @classmethod
    def get_active_operations(cls) -> list[ActiveOperationInfo]:
        """Get list of all active operations."""
        active = []
        current_time = utc_now()

        with _operation_lock:
            for thread_id, op_info in _active_operations.items():
                duration = (current_time - op_info['start_time']).total_seconds()
                active.append({
                    'thread_id': thread_id,
                    'thread_name': op_info['thread_name'],
                    'operation': op_info['operation'],
                    'duration_seconds': duration,
                    'start_time': op_info['start_time'].isoformat(),
                    'context': op_info['context']
                })

        return sorted(active, key=lambda x: x['duration_seconds'], reverse=True)

    @classmethod
    def dump_thread_states(cls) -> str:
        """Generate a detailed dump of all thread states."""
        lines = [
            "=" * 80,
            f"Thread State Dump at {utc_now().isoformat()}",
            "=" * 80,
            ""
        ]

        # System thread info
        process = psutil.Process()
        lines.append(f"Process ID: {process.pid}")
        lines.append(f"Thread Count: {process.num_threads()}")
        lines.append(f"CPU Percent: {process.cpu_percent()}%")
        lines.append(f"Memory RSS: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        lines.append("")

        # Active operations
        stuck_ops = cls.get_stuck_operations()
        if stuck_ops:
            lines.append(f"STUCK OPERATIONS ({len(stuck_ops)}):")
            lines.append("-" * 40)
            for op in stuck_ops:
                lines.append(f"Thread: {op['thread_name']} (ID: {op['thread_id']})")
                lines.append(f"Operation: {op['operation']}")
                lines.append(f"Duration: {op['duration_seconds']:.2f} seconds")
                lines.append(f"Context: {op['context']}")
                lines.append("Stack trace at start:")
                for frame_line in op['stack_trace'][:10]:  # First 10 frames
                    lines.append(f"  {frame_line.strip()}")
                lines.append("")

        # All threads
        lines.append("ALL THREADS:")
        lines.append("-" * 40)
        for thread in threading.enumerate():
            lines.append(f"Thread: {thread.name} (ID: {thread.ident})")
            lines.append(f"  Daemon: {thread.daemon}")
            lines.append(f"  Alive: {thread.is_alive()}")

            # Check if this thread has an active operation
            if thread.ident in _active_operations:
                op = _active_operations[thread.ident]
                duration = (utc_now() - op['start_time']).total_seconds()
                lines.append(f"  Active Operation: {op['operation']}")
                lines.append(f"  Duration: {duration:.2f}s")
            lines.append("")

        return "\n".join(lines)


def monitored_operation(operation_name: Optional[str] = None,
                        include_args: bool = False):
    """
    Decorator to monitor function execution in threads.

    Args:
        operation_name: Name for the operation (defaults to function name)
        include_args: Whether to include function arguments in context
    """
    def decorator(func):
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            context = {}

            if include_args:
                # Include first few args/kwargs for context
                context['args'] = str(args[:3]) if len(args) > 3 else str(args)
                context['kwargs'] = str(dict(list(kwargs.items())[:3]))

            ThreadMonitor.start_operation(name, context)
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(
                    f"Exception in monitored operation '{name}': {e}",
                    exc_info=True
                )
                raise
            finally:
                ThreadMonitor.end_operation(name)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            context = {}

            if include_args:
                context['args'] = str(args[:3]) if len(args) > 3 else str(args)
                context['kwargs'] = str(dict(list(kwargs.items())[:3]))

            ThreadMonitor.start_operation(name, context)
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(
                    f"Exception in async monitored operation '{name}': {e}",
                    exc_info=True
                )
                raise
            finally:
                ThreadMonitor.end_operation(name)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class MonitoredThreadPoolExecutor(ThreadPoolExecutor):
    """
    ThreadPoolExecutor with built-in monitoring and logging.

    Tracks all submitted tasks and logs when they get stuck.
    """

    def __init__(self, max_workers=None, thread_name_prefix='', **kwargs):
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix, **kwargs)
        self._active_futures: Dict[Future, FutureInfo] = {}
        self._futures_lock = threading.RLock()

        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_futures,
            name=f"{thread_name_prefix}-monitor",
            daemon=True
        )
        self._monitor_thread.start()

    def submit(self, fn, *args, **kwargs):
        """Submit a function with monitoring."""
        # Wrap function with monitoring
        operation_name = f"{fn.__module__}.{fn.__name__}" if hasattr(fn, '__name__') else str(fn)

        @functools.wraps(fn)
        def monitored_fn(*args, **kwargs):
            ThreadMonitor.start_operation(
                f"ThreadPool: {operation_name}",
                {'args': str(args[:2]), 'kwargs': str(list(kwargs.keys()))}
            )
            try:
                return fn(*args, **kwargs)
            finally:
                ThreadMonitor.end_operation(f"ThreadPool: {operation_name}")

        future = super().submit(monitored_fn, *args, **kwargs)

        # Track the future
        with self._futures_lock:
            self._active_futures[future] = {
                'operation': operation_name,
                'submit_time': utc_now(),
                'args_preview': str(args[:2]) if args else '',
                'thread_prefix': self._thread_name_prefix
            }

        # Clean up when done
        def cleanup(_):
            with self._futures_lock:
                if future in self._active_futures:
                    info = self._active_futures[future]
                    duration = (utc_now() - info['submit_time']).total_seconds()
                    if duration > ThreadMonitor.SLOW_OPERATION_THRESHOLD:
                        logger.warning(
                            f"ThreadPool task '{operation_name}' completed after {duration:.2f}s"
                        )
                    del self._active_futures[future]

        future.add_done_callback(cleanup)
        return future

    def _monitor_futures(self):
        """Monitor thread that checks for stuck futures."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                current_time = utc_now()

                with self._futures_lock:
                    for future, info in list(self._active_futures.items()):
                        if not future.done():
                            duration = (current_time - info['submit_time']).total_seconds()
                            if duration > ThreadMonitor.STUCK_OPERATION_THRESHOLD:
                                logger.error(
                                    f"ThreadPool task '{info['operation']}' has been "
                                    f"running for {duration:.2f} seconds! "
                                    f"Args: {info['args_preview']}"
                                )
            except Exception as e:
                logger.error(f"Error in thread monitor: {e}")


# Periodic monitoring task
def start_periodic_monitoring(interval_seconds: int = 300):
    """
    Start a background thread that periodically logs thread state.

    Args:
        interval_seconds: How often to check for stuck threads (default: 5 minutes)
    """
    def monitor_loop():
        logger.info("Thread monitoring started")
        while True:
            try:
                time.sleep(interval_seconds)

                stuck_ops = ThreadMonitor.get_stuck_operations()
                if stuck_ops:
                    logger.error(
                        f"THREAD MONITOR: {len(stuck_ops)} stuck operations detected!",
                        extra={'stuck_operations': stuck_ops}
                    )

                    # Dump full state for debugging
                    dump = ThreadMonitor.dump_thread_states()
                    logger.error(f"Thread state dump:\n{dump}")

                    # Write to file for analysis (use tempfile to respect TMPDIR)
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode='w',
                            prefix='thread_dump_',
                            suffix='.txt',
                            delete=False
                        ) as f:
                            f.write(dump)
                            dump_file = f.name
                        logger.error(f"Thread dump written to {dump_file}")
                    except OSError as e:
                        logger.warning(f"Could not write thread dump to file: {e}")

                # Log active operation summary
                active_ops = ThreadMonitor.get_active_operations()
                if active_ops:
                    logger.info(
                        f"Thread monitor: {len(active_ops)} active operations",
                        extra={'active_count': len(active_ops)}
                    )
                    for op in active_ops[:5]:  # Log top 5 longest running
                        if op['duration_seconds'] > 10:
                            logger.info(
                                f"  - {op['operation']}: {op['duration_seconds']:.1f}s "
                                f"(Thread: {op['thread_name']})"
                            )

            except Exception as e:
                logger.error(f"Error in periodic monitor: {e}", exc_info=True)

    monitor_thread = threading.Thread(
        target=monitor_loop,
        name="thread-monitor",
        daemon=True
    )
    monitor_thread.start()
    logger.info(f"Periodic thread monitoring started (interval: {interval_seconds}s)")