"""
Centralized HTTP client module with automatic retry logic for transient errors.
This module provides drop-in replacements for httpx components with built-in retry support.

Usage:
    from utils import http_client
    
    # Instead of httpx.Client()
    client = http_client.Client()
    
    # Instead of httpx.get()
    response = http_client.get("https://api.example.com/data")
    
    # Exceptions work the same way
    try:
        response = client.get(url)
    except http_client.HTTPStatusError as e:
        handle_error(e)
"""

import logging
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterable, Optional
from contextlib import contextmanager

import httpcore
import httpx

# Re-export httpx exceptions so code doesn't need to change
from httpx import (
    TimeoutException,
    HTTPStatusError,
    RequestError,
    ConnectError,
    ConnectTimeout,
    Response,
)

logger = logging.getLogger("http_client")

# Configuration for retry behavior
RETRYABLE_STATUS_CODES = {429, 502, 503, 504, 529}
BACKOFF_STATUS_CODES = {429, 529}  # Need longer delays
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

# Non-idempotent verbs are never retried on status: the server may have
# already processed the write before returning a retryable status, and a
# re-send would duplicate it.
NON_IDEMPOTENT_METHODS = {"POST", "PATCH"}


def _retry_after_seconds(response: Response) -> Optional[float]:
    """Parse a Retry-After header into seconds, if present and valid.

    Supports both the delta-seconds form and the HTTP-date form.
    """
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


class RetryMixin:
    """Mixin class providing retry logic for HTTP requests."""
    
    def __init__(self, *args, max_retries: Optional[int] = None, retry_non_idempotent: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES
        # Per-caller opt-in: when True, status retries apply to this client's
        # POST/PATCH requests too. Callers may opt in ONLY when every
        # non-idempotent request the client sends is safe to repeat — e.g.
        # pure inference queries where a 429/5xx is returned before any
        # server-side effect. The default (False) keeps POST/PATCH
        # un-retried on status for every existing caller.
        self.retry_non_idempotent = retry_non_idempotent
        
    def _calculate_delay(self, attempt: int, status_code: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        if status_code in BACKOFF_STATUS_CODES:
            # Longer delays for rate limiting: 2, 4, 8 seconds
            base_delay = 2.0
        else:
            # Shorter delays for transient errors: 0.5, 1, 2 seconds
            base_delay = 0.5
            
        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
        return min(delay, 30.0)  # Cap at 30 seconds
    
    def _should_retry(self, status_code: int, attempt: int, method: Optional[str] = None) -> bool:
        """Determine if a request should be retried based on status code and attempt number.

        Status retries apply only to idempotent verbs; a non-idempotent
        verb (POST/PATCH) is never retried on status, because the server
        may have processed the write before returning the retryable status.
        A client constructed with retry_non_idempotent=True opts its own
        requests out of that guard — callers may do so only when their
        requests are safe to repeat.
        """
        if (
            method is not None
            and method.upper() in NON_IDEMPOTENT_METHODS
            and not self.retry_non_idempotent
        ):
            return False
        return status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries
    
    def _execute_with_retry(self, request_func: Callable[..., Response], *args: Any, **kwargs: Any) -> Response:
        """Execute a request function with retry logic.

        httpx returns responses instead of raising for 4xx/5xx, so retryable
        status codes are detected on the returned response. After exhausting
        retries the final response is returned to the caller (httpx semantics).
        """
        last_exception = None
        # The request method is the first positional argument of the wrapped
        # request call (httpx request(method, url, ...)); fall back to the
        # keyword form. Used to block status retries of non-idempotent verbs.
        method = args[0] if args else kwargs.get("method")
        
        for attempt in range(self.max_retries + 1):
            try:
                response = request_func(*args, **kwargs)
                status_code = response.status_code
                
                if self._should_retry(status_code, attempt, method):
                    delay = self._calculate_delay(attempt, status_code)
                    if status_code == 429:
                        retry_after = _retry_after_seconds(response)
                        if retry_after is not None:
                            delay = min(retry_after, 30.0)
                    
                    if status_code == 529:
                        logger.warning(f"Server overloaded (529), attempt {attempt + 1}/{self.max_retries + 1}, retrying in {delay:.1f}s...")
                    elif status_code == 429:
                        logger.warning(f"Rate limited (429), attempt {attempt + 1}/{self.max_retries + 1}, retrying in {delay:.1f}s...")
                    else:
                        logger.warning(f"Server error ({status_code}), attempt {attempt + 1}/{self.max_retries + 1}, retrying in {delay:.1f}s...")
                    
                    time.sleep(delay)
                    continue
                
                return response
                
            except HTTPStatusError as e:
                last_exception = e
                status_code = e.response.status_code if e.response else 0
                
                if self._should_retry(status_code, attempt, method):
                    delay = self._calculate_delay(attempt, status_code)
                    
                    if status_code == 529:
                        logger.warning(f"Server overloaded (529), attempt {attempt + 1}/{self.max_retries + 1}, retrying in {delay:.1f}s...")
                    elif status_code == 429:
                        logger.warning(f"Rate limited (429), attempt {attempt + 1}/{self.max_retries + 1}, retrying in {delay:.1f}s...")
                    else:
                        logger.warning(f"Server error ({status_code}), attempt {attempt + 1}/{self.max_retries + 1}, retrying in {delay:.1f}s...")
                    
                    time.sleep(delay)
                    continue
                else:
                    # Not retryable or max retries exceeded
                    raise
                    
            except (ConnectError, ConnectTimeout) as e:
                # Connection errors might be retryable
                last_exception = e
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt, 503)  # Treat as service unavailable
                    logger.warning(f"Connection error, attempt {attempt + 1}/{self.max_retries + 1}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    raise
                    
            except (TimeoutException, RequestError):
                # Other errors are not retryable
                raise
        
        # If we exhausted retries, raise the last exception
        if last_exception:
            raise last_exception


class Client(RetryMixin, httpx.Client):
    """
    Drop-in replacement for httpx.Client with automatic retry logic.
    
    Automatically retries on:
    - 429 (Rate Limited)
    - 502 (Bad Gateway)
    - 503 (Service Unavailable)
    - 504 (Gateway Timeout)
    - 529 (Server Overloaded)
    - Connection errors

    Status retries apply only to idempotent verbs by default. Construct
    with retry_non_idempotent=True to extend status retries to this
    client's POST/PATCH requests; opt in only when those requests are
    safe to repeat (e.g. pure inference queries with no server-side
    writes — a 429/5xx is then returned pre-execution and re-sending
    cannot duplicate an effect).
    """
    
    def __init__(self, *args, max_retries: Optional[int] = None, retry_non_idempotent: bool = False, **kwargs):
        # Set default timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = DEFAULT_TIMEOUT
        super().__init__(*args, max_retries=max_retries, retry_non_idempotent=retry_non_idempotent, **kwargs)
    
    def request(self, *args, **kwargs):
        """Override request method to add retry logic."""
        return self._execute_with_retry(super().request, *args, **kwargs)
    
    def get(self, *args, **kwargs):
        """GET request with retry logic.

        Delegates to self.request, the single retry-wrapped entry point.
        httpx verb methods call self.request internally, so wrapping
        super().get here would nest two retry loops.
        """
        return self.request("GET", *args, **kwargs)
    
    def post(self, *args, **kwargs):
        """POST request with retry logic."""
        return self.request("POST", *args, **kwargs)
    
    def put(self, *args, **kwargs):
        """PUT request with retry logic."""
        return self.request("PUT", *args, **kwargs)
    
    def patch(self, *args, **kwargs):
        """PATCH request with retry logic."""
        return self.request("PATCH", *args, **kwargs)
    
    def delete(self, *args, **kwargs):
        """DELETE request with retry logic."""
        return self.request("DELETE", *args, **kwargs)
    
    def stream(self, *args, **kwargs):
        """Return the httpx stream context manager without retry.

        The connection is established when the CALLER enters the returned
        context manager, outside this method's frame — no retry here can see
        a ConnectError. Connection retry for streaming lives in the
        module-level `stream()` wrapper, where CM entry is inside the function.
        """
        return super().stream(*args, **kwargs)


# Convenience functions that mirror httpx module-level functions
def get(url: str, **kwargs) -> Response:
    """
    Convenience function for GET requests with automatic retry.
    
    Args:
        url: The URL to request
        **kwargs: Additional arguments passed to httpx.get()
        
    Returns:
        httpx.Response object
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    with Client(max_retries=max_retries) as client:
        return client.get(url, **kwargs)


def post(url: str, **kwargs) -> Response:
    """
    Convenience function for POST requests with automatic retry.
    
    Args:
        url: The URL to request
        **kwargs: Additional arguments passed to httpx.post()
        
    Returns:
        httpx.Response object
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    with Client(max_retries=max_retries) as client:
        return client.post(url, **kwargs)


def put(url: str, **kwargs) -> Response:
    """
    Convenience function for PUT requests with automatic retry.
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    with Client(max_retries=max_retries) as client:
        return client.put(url, **kwargs)


def patch(url: str, **kwargs) -> Response:
    """
    Convenience function for PATCH requests with automatic retry.
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    with Client(max_retries=max_retries) as client:
        return client.patch(url, **kwargs)


def delete(url: str, **kwargs) -> Response:
    """
    Convenience function for DELETE requests with automatic retry.
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    with Client(max_retries=max_retries) as client:
        return client.delete(url, **kwargs)


@contextmanager
def stream(method: str, url: str, **kwargs):
    """Streaming request with retry on connection establishment.

    The context manager is entered inside this function, so ConnectError at
    entry is retryable here. No mid-stream retry: once the response body has
    started yielding, any error propagates to the caller unchanged.

    Usage:
        with http_client.stream('GET', url) as response:
            for line in response.iter_lines():
                process(line)
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    http2 = kwargs.pop('http2', False)
    with Client(max_retries=max_retries, http2=http2) as client:
        for attempt in range(max_retries + 1):
            response_started = False
            try:
                with client.stream(method, url, **kwargs) as response:
                    response_started = True
                    yield response
                return
            except (ConnectError, ConnectTimeout):
                if response_started or attempt >= max_retries:
                    raise
                delay = 0.5 * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    f"Stream connection error, attempt {attempt + 1}/{max_retries + 1}, "
                    f"retrying in {delay:.1f}s..."
                )
                time.sleep(delay)


class _IPPinNetworkBackend(httpcore.NetworkBackend):
    """Network backend that connects to a pre-validated IP instead of re-resolving DNS.

    Closes the DNS-rebinding TOCTOU in SSRF validation: `utils.url_safety` resolves
    and validates the hostname once, then callers pass that exact IP here. The TCP
    connection targets the validated IP while TLS SNI and certificate verification
    still use the URL hostname. Connections for any other hostname fail loudly
    rather than silently falling back to fresh DNS resolution.
    """

    def __init__(self, hostname: str, ip: str):
        self._hostnames = {
            hostname.rstrip(".").lower(),
            hostname.encode("idna").decode("ascii").rstrip(".").lower(),
        }
        self._ip = ip
        self._backend = httpcore.SyncBackend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        if host.rstrip(".").lower() not in self._hostnames:
            raise httpcore.ConnectError(
                f"IP-pinned connection attempted for unvalidated hostname: {host}"
            )
        return self._backend.connect_tcp(
            self._ip,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        return self._backend.connect_unix_socket(path, timeout=timeout, socket_options=socket_options)

    def sleep(self, seconds: float) -> None:
        self._backend.sleep(seconds)


class PinnedHTTPTransport(httpx.HTTPTransport):
    """HTTP transport pinned to a single pre-validated IP address."""

    def __init__(self, hostname: str, ip: str):
        super().__init__()
        self._pool = httpcore.ConnectionPool(
            network_backend=_IPPinNetworkBackend(hostname, ip)
        )


def pinned_request(method: str, url: str, *, hostname: str, ip: str, **kwargs: Any) -> Response:
    """Issue an HTTP request whose TCP connection targets a pre-validated IP.

    `hostname` and `ip` must come from a `utils.url_safety.ValidatedURL` so the
    connection uses the exact address that passed SSRF validation. TLS SNI and
    certificate verification still use the URL hostname.
    """
    max_retries = kwargs.pop('max_retries', DEFAULT_MAX_RETRIES)
    transport = PinnedHTTPTransport(hostname, ip)
    with Client(max_retries=max_retries, transport=transport) as client:
        return client.request(method, url, **kwargs)
