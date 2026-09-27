"""
Centralized MCP (Model Context Protocol) client with automatic retry logic and user-scoped connections.
This module provides MIRA-integrated wrappers for the MCP SDK with built-in authentication support.

Usage:
    from utils import mcp_client

    # Create user-scoped client
    async with mcp_client.create_session("square", "https://mcp.squareup.com/sse") as session:
        tools = await session.list_tools()
        result = await session.call_tool("create_customer", params)

    # Or use convenience functions
    result = await mcp_client.call_tool("square", "create_customer", params)
"""

import logging
import asyncio
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, List, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import random

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import Tool, Resource, CallToolResult, ReadResourceResult

from utils.user_context import get_current_user_id
from utils.user_credentials import UserCredentialService

logger = logging.getLogger("mcp_client")

# Connection pool for user-scoped MCP sessions
_connection_pool: Dict[str, Dict[str, 'MCPConnection']] = {}

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
RETRYABLE_ERROR_TYPES = {ConnectionError, TimeoutError, OSError}


def _unwrap_exception(exc: Exception) -> Exception:
    current: Exception = exc
    visited = set()

    while True:
        if isinstance(current, BaseExceptionGroup):
            exceptions = getattr(current, "exceptions", None)
            if exceptions:
                current = exceptions[0]
                continue

        cause = getattr(current, "__cause__", None)
        if isinstance(cause, Exception) and id(cause) not in visited:
            visited.add(id(cause))
            current = cause
            continue

        context = getattr(current, "__context__", None)
        if isinstance(context, Exception) and id(context) not in visited:
            visited.add(id(context))
            current = context
            continue

        return current


def _is_auth_error(exc: Optional[Exception]) -> bool:
    if exc is None:
        return False

    status = getattr(exc, "status", None)
    if isinstance(status, int) and status in (401, 403):
        return True

    message = str(exc).lower()
    return "401" in message or "403" in message or "unauthorized" in message


def _is_connection_error(exc: Optional[Exception]) -> bool:
    if exc is None:
        return False

    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True

    message = str(exc).lower()
    connection_tokens = [
        "connection refused",
        "cannot connect",
        "connection reset",
        "network is unreachable",
        "name or service not known",
    ]
    return any(token in message for token in connection_tokens)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    url: Optional[str] = None
    transport: str = "sse"  # "sse", "stdio"
    command: Optional[List[str]] = None  # For stdio transport
    oauth_client_id: Optional[str] = None
    custom_headers: Optional[Dict[str, str]] = None


class MCPError(Exception):
    """Base exception for MCP client errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""
    pass


class MCPAuthenticationError(MCPError):
    """Raised when authentication with MCP server fails."""
    pass


class MCPToolNotFoundError(MCPError):
    """Raised when requested tool is not found on server."""
    pass


class MCPResourceNotFoundError(MCPError):
    """Raised when requested resource is not found on server."""
    pass


class MCPConnection:
    """
    Manages a single MCP connection with retry logic and credential handling.
    """

    def __init__(self, server_config: MCPServerConfig, user_id: str):
        self.config = server_config
        self.user_id = user_id
        self.session: Optional[ClientSession] = None
        self.credential_service = UserCredentialService()
        self._tools_cache: Optional[List[Tool]] = None
        self._resources_cache: Optional[List[Resource]] = None
        self._read = None
        self._write = None
        self._transport_context = None
        self._session_context = None  # Tracks if we entered ClientSession context
        self._in_use = False  # Claimed by an active create_session() context

    async def connect(self) -> ClientSession:
        """Establish connection to MCP server with authentication."""
        if self.session:
            return self.session

        try:
            # Get OAuth token if needed
            headers = dict(self.config.custom_headers or {})
            if self.config.oauth_client_id:
                token = self.credential_service.get_credential(
                    "oauth_token",
                    self.config.name
                )
                if not token:
                    raise MCPAuthenticationError(
                        f"No OAuth token found for {self.config.name}. "
                        f"Please authenticate first."
                    )
                headers["Authorization"] = f"Bearer {token}"

            # Create appropriate transport
            if self.config.transport == "sse":
                if not self.config.url:
                    raise ValueError(f"SSE transport requires URL for {self.config.name}")

                self._transport_context = sse_client(self.config.url, headers=headers)
                self._read, self._write = await self._enter_transport_context()
                # Bound every request/response read so a hung server cannot
                # block callers indefinitely
                self.session = ClientSession(
                    self._read, self._write,
                    read_timeout_seconds=timedelta(seconds=DEFAULT_TIMEOUT)
                )
                # CRITICAL: ClientSession must be entered as async context manager
                # to start _receive_loop which dispatches responses to waiting requests
                self._session_context = self.session
                await self._session_context.__aenter__()
                await self.session.initialize()

            elif self.config.transport == "stdio":
                if not self.config.command:
                    raise ValueError(f"Stdio transport requires command for {self.config.name}")

                server_params = StdioServerParameters(
                    command=self.config.command[0],
                    args=self.config.command[1:] if len(self.config.command) > 1 else []
                )
                self._transport_context = stdio_client(server_params)
                self._read, self._write = await self._enter_transport_context()
                # Bound every request/response read so a hung server cannot
                # block callers indefinitely
                self.session = ClientSession(
                    self._read, self._write,
                    read_timeout_seconds=timedelta(seconds=DEFAULT_TIMEOUT)
                )
                # CRITICAL: ClientSession must be entered as async context manager
                # to start _receive_loop which dispatches responses to waiting requests
                self._session_context = self.session
                await self._session_context.__aenter__()
                await self.session.initialize()

            else:
                raise ValueError(f"Unsupported transport: {self.config.transport}")

            logger.info(f"Connected to MCP server: {self.config.name}")
            return self.session

        except Exception as e:
            root_exc = _unwrap_exception(e)
            try:
                await self.disconnect()
            except Exception as cleanup_error:
                logger.debug(
                    "Additional error while cleaning up failed MCP connection to %s: %s",
                    self.config.name,
                    cleanup_error,
                )
            message = str(root_exc) if root_exc else str(e)
            logger.error(f"Failed to connect to {self.config.name}: {message}")

            if _is_auth_error(root_exc):
                raise MCPAuthenticationError(
                    f"Authentication failed for {self.config.name}: {message}"
                ) from root_exc
            elif _is_connection_error(root_exc):
                raise MCPConnectionError(
                    f"Connection failed to {self.config.name}: {message}"
                ) from root_exc
            else:
                raise MCPError(
                    f"Failed to connect to {self.config.name}: {message}"
                ) from root_exc

    async def disconnect(self):
        """Close the MCP connection."""
        # First exit the ClientSession context (stops _receive_loop)
        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error exiting session context for {self.config.name}: {e}")
            finally:
                self._session_context = None
        if self.session:
            try:
                if hasattr(self.session, 'close'):
                    await self.session.close()
            except Exception as e:
                logger.warning(f"Error closing session for {self.config.name}: {e}")
            finally:
                self.session = None
                self._read = None
                self._write = None
                self._tools_cache = None
                self._resources_cache = None
        if self._transport_context:
            try:
                await self._transport_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing transport for {self.config.name}: {e}")
            finally:
                self._transport_context = None

    async def list_tools(self, use_cache: bool = True) -> List[Tool]:
        """List available tools from the MCP server."""
        if use_cache and self._tools_cache is not None:
            return self._tools_cache

        session = await self.connect()
        result = await session.list_tools()
        # MCP SDK returns ListToolsResult with .tools attribute
        self._tools_cache = result.tools
        return self._tools_cache

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Call a tool on the MCP server."""
        session = await self.connect()

        # Verify tool exists
        tools = await self.list_tools()
        if not any(t.name == name for t in tools):
            raise MCPToolNotFoundError(f"Tool '{name}' not found on {self.config.name}")

        return await session.call_tool(name, arguments)

    async def list_resources(self, use_cache: bool = True) -> List[Resource]:
        """List available resources from the MCP server."""
        if use_cache and self._resources_cache is not None:
            return self._resources_cache

        session = await self.connect()
        result = await session.list_resources()
        # MCP SDK returns ListResourcesResult with .resources attribute
        self._resources_cache = result.resources
        return self._resources_cache

    async def get_resource(self, uri: str) -> ReadResourceResult:
        """Get a resource from the MCP server."""
        session = await self.connect()
        return await session.read_resource(uri)

    async def _enter_transport_context(self) -> Any:
        try:
            if self._transport_context is None:
                raise RuntimeError("Transport context not initialized")
            return await self._transport_context.__aenter__()
        except Exception as exc:
            # Ensure context is cleaned up if __aenter__ fails
            await self._cleanup_failed_transport(exc)
            raise

    async def _cleanup_failed_transport(self, original_exc: Exception) -> None:
        if self._transport_context is None:
            return
        try:
            await self._transport_context.__aexit__(type(original_exc), original_exc, original_exc.__traceback__)
        except Exception as cleanup_exc:
            logger.debug(
                "Error cleaning up transport context after failure for %s: %s",
                self.config.name,
                cleanup_exc,
            )
        finally:
            self._transport_context = None


class RetryableMCPConnection(MCPConnection):
    """MCPConnection with automatic retry logic for transient failures."""

    def __init__(self, server_config: MCPServerConfig, user_id: str, max_retries: int = DEFAULT_MAX_RETRIES):
        super().__init__(server_config, user_id)
        self.max_retries = max_retries

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        base_delay = 1.0
        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
        return min(delay, 30.0)  # Cap at 30 seconds

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried."""
        if attempt >= self.max_retries:
            return False

        # Check if it's a retryable error type
        if isinstance(exception, MCPConnectionError):
            return True
        if isinstance(exception, (ConnectionError, TimeoutError, OSError)):
            return True
        if isinstance(exception, MCPError) and "timeout" in str(exception).lower():
            return True

        return False

    async def _execute_with_retry(self, operation: Callable, *args: Any, **kwargs: Any):
        """Execute an operation with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if self._should_retry(e, attempt):
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"MCP operation failed for {self.config.name}, "
                        f"attempt {attempt + 1}/{self.max_retries + 1}, "
                        f"retrying in {delay:.1f}s... Error: {str(e)}"
                    )
                    await asyncio.sleep(delay)

                    # Reset connection for next attempt
                    await self.disconnect()
                    continue
                else:
                    raise

        if last_exception:
            raise last_exception

    async def connect(self) -> ClientSession:
        """Connect with retry logic."""
        return await self._execute_with_retry(super().connect)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Call tool with retry logic."""
        return await self._execute_with_retry(super().call_tool, name, arguments)

    async def get_resource(self, uri: str) -> ReadResourceResult:
        """Get resource with retry logic."""
        return await self._execute_with_retry(super().get_resource, uri)

    async def list_tools(self, use_cache: bool = True) -> List[Tool]:
        """List tools with retry logic."""
        if use_cache and self._tools_cache is not None:
            return self._tools_cache
        return await self._execute_with_retry(super().list_tools, use_cache=False)

    async def list_resources(self, use_cache: bool = True) -> List[Resource]:
        """List resources with retry logic."""
        if use_cache and self._resources_cache is not None:
            return self._resources_cache
        return await self._execute_with_retry(super().list_resources, use_cache=False)


# Module-level functions for convenience

@asynccontextmanager
async def create_session(
    server_name: str,
    server_url: Optional[str] = None,
    transport: str = "sse",
    command: Optional[List[str]] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    oauth_client_id: Optional[str] = None,
    custom_headers: Optional[Dict[str, str]] = None
) -> AsyncIterator[MCPConnection]:
    """
    Create a user-scoped MCP session with automatic cleanup.

    Usage:
        async with mcp_client.create_session("square", "https://mcp.squareup.com/sse") as session:
            result = await session.call_tool("create_customer", {...})

    Args:
        server_name: Unique name for this MCP server
        server_url: URL for SSE transport (required for SSE)
        transport: Transport type ("sse" or "stdio")
        command: Command and args for stdio transport
        max_retries: Number of retry attempts for transient failures
        oauth_client_id: OAuth client ID if authentication is required
        custom_headers: Additional headers to send with requests

    Yields:
        MCPConnection instance
    """
    user_id = get_current_user_id()
    if not user_id:
        raise RuntimeError("No user context set. Ensure authentication is properly initialized.")

    # Check connection pool
    if user_id not in _connection_pool:
        _connection_pool[user_id] = {}

    pooled = _connection_pool[user_id].get(server_name)
    if pooled is not None and not pooled._in_use:
        connection = pooled
    else:
        # The pooled connection (if any) is claimed by another concurrent
        # context - give this call its own connection so one context exiting
        # cannot disconnect another's session
        config = MCPServerConfig(
            name=server_name,
            url=server_url,
            transport=transport,
            command=command,
            oauth_client_id=oauth_client_id,
            custom_headers=custom_headers
        )
        connection = RetryableMCPConnection(config, user_id, max_retries)
        _connection_pool[user_id][server_name] = connection

    connection._in_use = True

    try:
        yield connection
    finally:
        # Disconnect the session when exiting the context
        # SSE connections spawn background tasks that keep the event loop alive,
        # so we must disconnect to allow clean shutdown (especially important
        # when running in a temporary thread with its own event loop)
        try:
            await connection.disconnect()
            # Remove from pool since it's now disconnected - but only if this
            # connection is still the pooled one (a concurrent context may have
            # replaced it with its own connection)
            if (
                user_id in _connection_pool
                and _connection_pool[user_id].get(server_name) is connection
            ):
                del _connection_pool[user_id][server_name]
        except Exception as e:
            logger.warning(f"Error disconnecting {server_name}: {e}")
        finally:
            connection._in_use = False


async def call_tool(
    server_name: str,
    tool_name: str,
    arguments: Dict[str, Any],
    server_url: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> CallToolResult:
    """
    Convenience function to call an MCP tool without managing sessions.

    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to call
        arguments: Tool arguments
        server_url: Optional server URL (required for first connection)
        max_retries: Number of retries for transient failures

    Returns:
        Tool execution result
    """
    async with create_session(server_name, server_url, max_retries=max_retries) as session:
        return await session.call_tool(tool_name, arguments)


async def list_tools(
    server_name: str,
    server_url: Optional[str] = None
) -> List[Tool]:
    """List available tools from an MCP server."""
    async with create_session(server_name, server_url) as session:
        return await session.list_tools()


async def get_resource(
    server_name: str,
    resource_uri: str,
    server_url: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> ReadResourceResult:
    """Get a resource from an MCP server."""
    async with create_session(server_name, server_url, max_retries=max_retries) as session:
        return await session.get_resource(resource_uri)
