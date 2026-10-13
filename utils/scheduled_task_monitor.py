"""
Monitoring wrapper for scheduled tasks to diagnose hanging issues.

Wraps scheduled tasks with detailed logging and timeout detection.
"""

import logging
import functools
import threading
import tempfile
from typing import Callable, Optional, Dict, Union

from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from utils.thread_monitor import ThreadMonitor
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


class JobStats(TypedDict):
    """Execution statistics for a single scheduled job."""
    total_runs: int
    successful_runs: int
    failed_runs: int
    timeout_runs: int
    last_run_time: Optional[str]
    last_duration_seconds: Optional[float]
    max_duration_seconds: float


class ScheduledTaskMonitor:
    """Monitors scheduled task execution for hanging detection."""

    _job_history: Dict[str, JobStats] = {}
    _history_lock = threading.RLock()

    @classmethod
    def wrap_scheduled_job(cls,
                           job_id: str,
                           func: Callable,
                           timeout_seconds: Optional[int] = None,
                           kill_on_timeout: bool = False) -> Callable:
        """
        Wrap a scheduled job function with monitoring and timeout handling.

        Args:
            job_id: Unique identifier for the job
            func: The job function to wrap
            timeout_seconds: Optional timeout in seconds (default: no timeout)
            kill_on_timeout: Whether to forcefully kill stuck jobs

        Returns:
            Wrapped function with monitoring
        """
        @functools.wraps(func)
        def monitored_wrapper(*args, **kwargs):
            """Wrapper that monitors job execution."""
            start_time = utc_now()
            thread_id = threading.current_thread().ident
            thread_name = threading.current_thread().name

            logger.info(
                f"Scheduled job '{job_id}' starting on thread {thread_name} (ID: {thread_id})"
            )

            ThreadMonitor.start_operation(
                f"ScheduledJob: {job_id}",
                {
                    'job_id': job_id,
                    'function': f"{func.__module__}.{func.__name__}",
                    'thread': thread_name,
                    'timeout_seconds': timeout_seconds
                }
            )

            with cls._history_lock:
                if job_id not in cls._job_history:
                    cls._job_history[job_id] = {
                        'total_runs': 0,
                        'successful_runs': 0,
                        'failed_runs': 0,
                        'timeout_runs': 0,
                        'last_run_time': None,
                        'last_duration_seconds': None,
                        'max_duration_seconds': 0
                    }
                cls._job_history[job_id]['total_runs'] += 1
                cls._job_history[job_id]['last_run_time'] = start_time.isoformat()

            # Set only by the monitor's own timeout path, so a job-body
            # TimeoutError is accounted as a failure, not re-raised as
            # "already logged"
            timeout_counted = False

            try:
                if timeout_seconds:
                    # Never wrap this executor in 'with': __exit__ calls
                    # shutdown(wait=True), which blocks forever on a stuck thread
                    executor = ThreadPoolExecutor(max_workers=1)
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        result = future.result(timeout=timeout_seconds)
                        duration = (utc_now() - start_time).total_seconds()

                        logger.info(
                            f"Scheduled job '{job_id}' completed successfully in {duration:.2f}s"
                        )

                        with cls._history_lock:
                            cls._job_history[job_id]['successful_runs'] += 1
                            cls._job_history[job_id]['last_duration_seconds'] = duration
                            cls._job_history[job_id]['max_duration_seconds'] = max(
                                cls._job_history[job_id]['max_duration_seconds'],
                                duration
                            )

                        return result

                    except FuturesTimeoutError:
                        # Python 3.11+ aliases concurrent.futures.TimeoutError to
                        # the built-in TimeoutError, so a TimeoutError raised by
                        # the job body surfaces here too. If the future is done,
                        # the job itself raised it - re-raise so it flows to the
                        # failure accounting below instead of being counted as a
                        # monitor timeout.
                        if future.done() and not future.cancelled():
                            raise

                        duration = (utc_now() - start_time).total_seconds()

                        logger.error(
                            f"TIMEOUT: Scheduled job '{job_id}' exceeded timeout of "
                            f"{timeout_seconds}s (ran for {duration:.2f}s) on thread {thread_name}"
                        )

                        with cls._history_lock:
                            cls._job_history[job_id]['timeout_runs'] += 1
                        timeout_counted = True

                        # Attempt to cancel - this won't stop a running thread,
                        # but will prevent queued work from starting
                        future.cancel()

                        if kill_on_timeout:
                            logger.error(
                                f"Job '{job_id}' thread is stuck - cannot force-kill Python threads. "
                                f"The orphaned thread will continue running until it completes or the process restarts."
                            )
                            dump = ThreadMonitor.dump_thread_states()
                            logger.error(f"Thread dump for stuck job:\n{dump}")

                        raise TimeoutError(
                            f"Job '{job_id}' timed out after {timeout_seconds}s"
                        )

                    finally:
                        # Shutdown WITHOUT waiting - don't block if thread is stuck
                        # cancel_futures=True attempts to cancel pending (not running) work
                        executor.shutdown(wait=False, cancel_futures=True)
                else:
                    result = func(*args, **kwargs)
                    duration = (utc_now() - start_time).total_seconds()

                    logger.info(
                        f"Scheduled job '{job_id}' completed in {duration:.2f}s"
                    )

                    with cls._history_lock:
                        cls._job_history[job_id]['successful_runs'] += 1
                        cls._job_history[job_id]['last_duration_seconds'] = duration
                        cls._job_history[job_id]['max_duration_seconds'] = max(
                            cls._job_history[job_id]['max_duration_seconds'],
                            duration
                        )

                    return result

            except TimeoutError as e:
                if timeout_counted:
                    raise  # Monitor timeout - already logged and counted

                # A TimeoutError raised by the job body itself is a job failure,
                # not a monitor timeout - log and count it like any other error
                duration = (utc_now() - start_time).total_seconds()
                logger.error(
                    f"Scheduled job '{job_id}' failed after {duration:.2f}s: {e}",
                    exc_info=True
                )

                with cls._history_lock:
                    cls._job_history[job_id]['failed_runs'] += 1
                    cls._job_history[job_id]['last_duration_seconds'] = duration

                raise

            except Exception as e:
                duration = (utc_now() - start_time).total_seconds()
                logger.error(
                    f"Scheduled job '{job_id}' failed after {duration:.2f}s: {e}",
                    exc_info=True
                )

                with cls._history_lock:
                    cls._job_history[job_id]['failed_runs'] += 1
                    cls._job_history[job_id]['last_duration_seconds'] = duration

                raise

            finally:
                ThreadMonitor.end_operation(f"ScheduledJob: {job_id}")

                duration = (utc_now() - start_time).total_seconds()
                if duration > 60:
                    logger.warning(
                        f"Scheduled job '{job_id}' took {duration:.2f}s - "
                        f"this may indicate a performance issue"
                    )

                    if duration > 300:
                        dump = ThreadMonitor.dump_thread_states()
                        try:
                            with tempfile.NamedTemporaryFile(
                                mode='w',
                                prefix=f'slow_job_{job_id}_',
                                suffix='.txt',
                                delete=False
                            ) as f:
                                f.write(dump)
                                dump_file = f.name
                            logger.warning(f"Thread dump for slow job written to {dump_file}")
                        except OSError as e:
                            logger.warning(f"Could not write thread dump: {e}")

        return monitored_wrapper

    @classmethod
    def get_job_stats(cls, job_id: Optional[str] = None) -> Union[JobStats, Dict[str, JobStats]]:
        """
        Get execution statistics for scheduled jobs.

        Args:
            job_id: Specific job ID to get stats for (or None for all)

        Returns:
            Dictionary of job statistics
        """
        with cls._history_lock:
            if job_id:
                return cls._job_history.get(job_id, {})
            else:
                return dict(cls._job_history)

    @classmethod
    def log_job_summary(cls):
        """Log a summary of all scheduled job performance."""
        with cls._history_lock:
            if not cls._job_history:
                logger.info("No scheduled job history to report")
                return

            logger.info("=== Scheduled Job Performance Summary ===")
            for job_id, stats in cls._job_history.items():
                total = stats['total_runs']
                if total == 0:
                    continue

                success_rate = (stats['successful_runs'] / total) * 100
                timeout_rate = (stats['timeout_runs'] / total) * 100

                logger.info(
                    f"Job '{job_id}':\n"
                    f"  Total runs: {total}\n"
                    f"  Success rate: {success_rate:.1f}%\n"
                    f"  Timeout rate: {timeout_rate:.1f}%\n"
                    f"  Last duration: {stats.get('last_duration_seconds', 0):.2f}s\n"
                    f"  Max duration: {stats.get('max_duration_seconds', 0):.2f}s"
                )

                if timeout_rate > 10:
                    logger.warning(
                        f"Job '{job_id}' has high timeout rate ({timeout_rate:.1f}%) - "
                        f"investigate for hanging issues"
                    )

                if stats.get('max_duration_seconds', 0) > 300:
                    logger.warning(
                        f"Job '{job_id}' has taken up to {stats['max_duration_seconds']:.2f}s - "
                        f"consider optimization or timeout adjustment"
                    )
