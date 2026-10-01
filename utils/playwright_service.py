"""
Security-hardened Playwright service for rendering JavaScript-heavy webpages.

Provides a singleton headless browser that executes JavaScript and returns
fully-rendered HTML. Implements security controls to prevent SSRF, resource
exhaustion, and other browser-based attacks.

Lifecycle: Chromium launches lazily on first fetch_rendered_html() call and
auto-shuts down after IDLE_TIMEOUT_SECONDS of inactivity. This prevents a
permanently idle browser process from leaking memory on the Droplet.

All Playwright operations are dispatched to a single persistent thread because
Playwright's sync API enforces thread affinity — the browser object can only be
used from the thread that called sync_playwright().start().
"""
import concurrent.futures
import logging
import threading
import time
from typing import Optional
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from utils.url_safety import validate_public_http_url


# Shut down Chromium after 10 minutes of no fetch_rendered_html() calls.
IDLE_TIMEOUT_SECONDS = 600

# Bounded wait for the Playwright node driver's graceful .stop() teardown.
# If it expires, teardown continues; the Chromium kill remains the fallback.
_DRIVER_STOP_TIMEOUT_SECONDS = 10

# Process names Chromium may appear as (varies by OS/install method).
_CHROMIUM_PROCESS_NAMES = frozenset({'chromium', 'chrome', 'headless_shell'})
_NON_NETWORK_SCHEMES = frozenset({'about', 'blob', 'data'})


class _CancelledError(Exception):
    """Internal sentinel raised when caller timeout triggers cancellation."""


class PlaywrightService:
    """
    Singleton headless browser service for JavaScript-rendered webpages.

    Chromium launches lazily on first use and shuts down after idle timeout.
    Browser contexts provide user isolation while sharing the underlying browser.
    Security controls prevent SSRF attacks and resource exhaustion.
    """

    _instance: Optional['PlaywrightService'] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize the Playwright service (browser starts lazily)."""
        self.logger = logging.getLogger("playwright_service")

        # Browser state — guarded by _browser_lock
        self._browser_lock = threading.Lock()
        self._playwright = None
        self._browser = None
        self._browser_pid: Optional[int] = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._last_use: float = 0.0
        self._idle_timer: Optional[threading.Timer] = None

        self.logger.info("PlaywrightService initialized (browser starts on first use)")

    def _ensure_browser(self) -> None:
        """Launch Chromium if not already running. Thread-safe."""
        if self._browser is not None:
            return

        with self._browser_lock:
            if self._browser is not None:
                return

            self.logger.info("Launching Chromium browser (first use)...")

            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="playwright",
            )

            try:
                future = self._executor.submit(self._init_playwright)
                future.result(timeout=30)
                self.logger.info("Chromium browser launched successfully")
            except Exception as e:
                self.logger.error(f"Failed to launch Chromium: {e}")
                raise RuntimeError(f"Playwright initialization failed: {e}") from e
            finally:
                # Covers the launch-failure path: _init_playwright may have
                # already started the Playwright node driver before failing
                # (e.g. chromium.launch raises after sync_playwright().start()
                # succeeded). Stop it on the executor worker before the
                # executor is discarded, or the driver subprocess leaks.
                if self._browser is None:
                    self._stop_playwright_driver()
                    self._playwright = None
                    self._executor.shutdown(wait=False)
                    self._executor = None

    def _stop_playwright_driver(self) -> None:
        """Stop the Playwright node driver on its executor worker thread.

        Playwright's sync API enforces thread affinity, so .stop() must run on
        the same single worker thread that called sync_playwright().start().
        This method's callers (the idle Timer thread, the lifespan thread, the
        _ensure_browser failure path) are not that thread, so the stop is
        dispatched to the executor and awaited with a bounded wait. Best-effort:
        on failure or timeout, teardown continues and the Chromium kill remains
        the fallback.
        """
        playwright = self._playwright
        executor = self._executor
        if playwright is None or executor is None:
            return
        try:
            future = executor.submit(playwright.stop)
            future.result(timeout=_DRIVER_STOP_TIMEOUT_SECONDS)
            self.logger.info("Playwright node driver stopped")
        except Exception as e:
            self.logger.warning(f"Playwright driver stop failed or timed out: {e}")

    def _init_playwright(self) -> None:
        """Initialize browser on the executor thread. Must only be called via _executor."""
        import psutil
        import time as _time

        pids_before = {
            p.pid for p in psutil.process_iter()
            if p.name().lower() in _CHROMIUM_PROCESS_NAMES
        }

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-gpu',
                '--disable-dev-shm-usage',
                '--disable-software-rasterizer',
                '--disable-extensions',
                '--no-sandbox',
                '--disable-setuid-sandbox',
            ]
        )

        _time.sleep(0.5)

        pids_after = {
            p.pid for p in psutil.process_iter()
            if p.name().lower() in _CHROMIUM_PROCESS_NAMES
        }
        new_pids = pids_after - pids_before
        self._browser_pid = min(new_pids) if new_pids else None

        if self._browser_pid:
            self.logger.info(
                f"Chromium process {self._browser_pid} captured for forced shutdown"
            )
        else:
            self.logger.warning(
                "Could not capture Chromium process PID — force shutdown may not work"
            )

    def _touch_and_schedule_idle_shutdown(self) -> None:
        """Record usage time and (re)schedule idle shutdown timer."""
        self._last_use = time.monotonic()

        # Cancel any existing timer
        if self._idle_timer is not None:
            self._idle_timer.cancel()

        self._idle_timer = threading.Timer(
            IDLE_TIMEOUT_SECONDS,
            self._idle_shutdown,
        )
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _idle_shutdown(self) -> None:
        """Shut down the browser if it hasn't been used since the timer was set."""
        elapsed = time.monotonic() - self._last_use
        if elapsed < IDLE_TIMEOUT_SECONDS:
            # Another call happened since the timer was scheduled — reschedule.
            remaining = IDLE_TIMEOUT_SECONDS - elapsed
            self._idle_timer = threading.Timer(remaining, self._idle_shutdown)
            self._idle_timer.daemon = True
            self._idle_timer.start()
            return

        self.logger.info(
            f"Chromium idle for {elapsed:.0f}s — shutting down to free memory"
        )
        self._shutdown_browser()

    @classmethod
    def get_instance(cls) -> 'PlaywrightService':
        """Get or create the singleton PlaywrightService instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def fetch_rendered_html(
        self,
        url: str,
        timeout: int = 30,
        max_size_mb: int = 10
    ) -> str:
        """
        Fetch fully-rendered HTML after JavaScript execution.

        Launches Chromium on first call. Dispatches to the Playwright thread.
        On caller timeout, signals cancellation so the Playwright thread
        finishes promptly and becomes available for new work.

        Args:
            url: Target URL (must already be validated by caller)
            timeout: Max time for page load in seconds
            max_size_mb: Max response size to prevent memory bombs

        Returns:
            Rendered HTML content as string

        Raises:
            TimeoutError: If page load exceeds timeout
            RuntimeError: For other failures (HTTP errors, size limits, etc.)
        """
        self._ensure_browser()
        self._touch_and_schedule_idle_shutdown()

        cancel = threading.Event()
        future = self._executor.submit(
            self._fetch_rendered_html_impl, url, timeout, max_size_mb, cancel
        )
        try:
            return future.result(timeout=timeout + 15)
        except concurrent.futures.TimeoutError:
            # Python 3.11+ aliases concurrent.futures.TimeoutError to the
            # built-in TimeoutError, so a TimeoutError raised by the worker
            # itself surfaces here too. If the future is done, the worker
            # raised it — re-raise so the worker's own "after {timeout}s"
            # label propagates unrelabeled instead of being miscounted as a
            # caller-budget timeout. (Mirror of scheduled_task_monitor.py:124-134.)
            if future.done() and not future.cancelled():
                raise
            cancel.set()
            self.logger.warning(f"Caller timed out for {url} — cancellation signalled")
            if not future.running() and not future.done():
                # The fetch was still queued on the single-worker executor when
                # the caller's budget expired: no page load ever began, so the
                # elapsed time was queue wait behind another fetch. Name that
                # instead of misreporting it as a page-load timeout.
                raise TimeoutError(
                    f"Page fetch timed out after {timeout + 15}s for {url} — "
                    f"queued behind another fetch, page load never started"
                )
            raise TimeoutError(f"Page fetch timed out after {timeout + 15}s for {url}")

    def _fetch_rendered_html_impl(
        self,
        url: str,
        timeout: int,
        max_size_mb: int,
        cancel: threading.Event,
    ) -> str:
        """Actual fetch implementation — runs on the Playwright thread."""
        # Queued-fetch entry check: with a single worker, this fetch may have
        # sat in the executor queue behind another fetch while its caller's
        # timeout expired. If so the caller is gone, and navigating now would
        # hold the only Playwright worker through a full page load nobody will
        # ever read — abort before creating a context or a page.
        if cancel.is_set():
            self.logger.warning(
                f"Fetch cancelled while queued for {url} — skipping navigation"
            )
            raise _CancelledError()

        context = self._browser.new_context(
            # Disable risky browser features
            geolocation=None,
            permissions=[],
            bypass_csp=False,
            java_script_enabled=True,

            # Performance/security limits
            viewport={'width': 1280, 'height': 720},
            ignore_https_errors=False,

            # User agent
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:143.0) Gecko/20100101 Firefox/143.0'
        )

        blocked_requests = []

        def handle_request(route, request):
            """Intercept and validate all network requests."""
            request_url = request.url

            try:
                validate_browser_request_url(request_url)
            except ValueError as e:
                self.logger.warning(f"Blocked browser request {request_url}: {e}")
                blocked_requests.append(request_url)
                route.abort()
                return

            # Block unnecessary resource types for faster loading
            resource_type = request.resource_type
            if resource_type in ['font', 'media', 'websocket']:
                route.abort()
                return

            # Continue with request
            route.continue_()

        try:
            page = context.new_page()

            # Set up request interception
            page.route('**/*', handle_request)

            # Set resource limits
            page.set_default_timeout(timeout * 1000)
            page.set_default_navigation_timeout(timeout * 1000)

            # Navigate with strict timeout
            try:
                response = page.goto(
                    url,
                    wait_until='networkidle',
                    timeout=timeout * 1000
                )

                if cancel.is_set():
                    raise _CancelledError()

            except PlaywrightTimeoutError:
                if cancel.is_set():
                    raise _CancelledError()
                self.logger.warning("Network idle timeout, falling back to domcontentloaded")
                response = page.goto(
                    url,
                    wait_until='domcontentloaded',
                    timeout=timeout * 1000
                )

                if cancel.is_set():
                    raise _CancelledError()

            if cancel.is_set():
                raise _CancelledError()

            # Check response status
            if response and response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")

            # Get rendered HTML
            html = page.content()

            # Check size limits
            html_size_mb = len(html) / (1024 * 1024)
            if html_size_mb > max_size_mb:
                raise RuntimeError(
                    f"Content too large: {html_size_mb:.1f}MB exceeds {max_size_mb}MB limit"
                )

            # Warn if internal requests were blocked
            if blocked_requests:
                self.logger.warning(
                    f"Blocked {len(blocked_requests)} SSRF attempts from {url}"
                )

            return html

        except _CancelledError:
            self.logger.warning(f"Fetch cancelled for {url}")
            raise RuntimeError(f"Fetch cancelled (caller timed out) for {url}")
        except PlaywrightTimeoutError:
            self.logger.error(f"Page load timeout after {timeout}s for {url}")
            raise TimeoutError(f"Page load timeout after {timeout}s")
        except Exception as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            raise
        finally:
            try:
                context.close()
            except Exception as e:
                self.logger.warning(f"Error closing context: {e}")

    def _shutdown_browser(self) -> None:
        """Shut down Chromium and release all resources. Safe to call multiple times."""
        with self._browser_lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None

            # Force-kill the Chromium process tree
            if self._browser_pid:
                try:
                    import psutil
                    process = psutil.Process(self._browser_pid)
                    process.kill()
                    self.logger.info(f"Force-killed Chromium process {self._browser_pid}")
                except Exception as e:
                    self.logger.debug(f"Chromium kill (may already be gone): {e}")

            # Stop the Playwright node driver on its executor worker before
            # dropping the reference and discarding the executor: without it
            # the driver subprocess survives and one leaks per idle-relaunch
            # cycle. Dispatched (thread affinity) and awaited with a bounded
            # wait; the Chromium kill above remains the fallback.
            self._stop_playwright_driver()

            self._browser = None
            self._playwright = None
            self._browser_pid = None

            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None

    def shutdown(self):
        """Full shutdown for process exit. Called from lifespan."""
        self._shutdown_browser()


def validate_browser_request_url(request_url: str) -> None:
    """Allow public HTTP(S) browser requests plus inert browser-local schemes."""
    scheme = urlparse(request_url).scheme.lower()
    if scheme in ("http", "https"):
        validate_public_http_url(request_url)
        return
    if scheme in _NON_NETWORK_SCHEMES:
        return
    raise ValueError(f"Blocked non-HTTP browser request scheme: {scheme}")
