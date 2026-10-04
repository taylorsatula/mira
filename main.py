#!/usr/bin/env python3
"""
MIRA - Main Application Entry Point
FastAPI server that wires together the CNS architecture and handles startup/shutdown.
"""

import argparse
import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from utils.logging_config import setup_colored_root_logging, setup_anthropic_sdk_logging
setup_colored_root_logging(log_level=logging.WARNING, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Deployment installs set MIRA_LOG_DIR=/opt/mira/logs (systemd unit, macOS
# launcher, s6 run script). Everywhere else the relative default keeps the
# import pure-Python instead of touching a machine-level install path.
setup_anthropic_sdk_logging(log_dir=os.environ.get("MIRA_LOG_DIR", "logs"))

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import http_exception_handler
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from auth import api as auth_api
from auth.mode import auth_mode
from auth.security_middleware import SecurityHeadersMiddleware
from config.config_manager import config
from config.announcement import load_announcement
from cns.api import data, actions, health, websocket_chat, tool_config, trigger_rules, update, federation as federation_api
from cns.api import chat as chat_api
from cns.api import files as files_api
from cns.api import heartbeat_api
from cns.api import location
from cns.api.base import APIError, create_error_response, generate_request_id
from utils.scheduler_service import scheduler_service
from utils.scheduled_tasks import initialize_all_scheduled_tasks

# Suppress routine APScheduler job execution logs (running/success/debug chatter)
logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)
logging.getLogger('apscheduler.scheduler').setLevel(logging.WARNING)
# logging.getLogger('tools.implementations.imagegen_tool').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    
    # Startup
    logger.info("  Starting MIRA...\n\n\n")
    logger.info("====================")

    # Two-mode identity bootstrap: `single` auto-provisions its local account
    # lazily through GET /v0/auth/local/session on first visit; `multi`
    # bootstraps through public signup and magic links. Neither seeds anything
    # at boot.
    logger.info(f"Auth mode: {auth_mode()}")


    # Configure FastAPI thread pool for synchronous endpoints (per-worker)
    from anyio import to_thread
    thread_limit = config.api_server.sync_endpoint_thread_limit
    to_thread.current_default_thread_limiter().total_tokens = thread_limit
    logger.info(f"FastAPI thread pool configured for {thread_limit} concurrent threads")

    # Pre-initialize expensive singleton resources at startup
    logger.info("Pre-initializing singleton resources...")

    # Preload all Vault secrets into memory cache (prevents token expiration issues)
    from clients.vault_client import preload_secrets
    try:
        preload_secrets()
    except Exception as e:
        logger.critical(f"Failed to preload Vault secrets: {e}")
        raise RuntimeError(f"vault initialization failed - cannot start MIRA: {e}") from e

    # Load announcement config (cached for lifetime of process)
    try:
        load_announcement()
    except Exception as e:
        logger.critical(f"Failed to load announcement config: {e}")
        raise RuntimeError(f"announcement loading failed - cannot start MIRA: {e}") from e

    # Initialize the embeddings provider named by the embedding_config row
    from clients.embeddings_provider import get_embeddings_provider
    try:
        embeddings_provider = get_embeddings_provider()
    except Exception as e:
        logger.critical(f"Failed to initialize embeddings provider: {e}")
        raise RuntimeError(f"embeddings initialization failed - cannot start MIRA: {e}") from e
    logger.info(f"Embeddings provider initialized: {type(embeddings_provider).__name__}")
    
    # Initialize continuum repository (creates DB connection pool)
    from cns.infrastructure.continuum_repository import get_continuum_repository
    try:
        continuum_repo = get_continuum_repository()
    except Exception as e:
        logger.critical(f"Failed to initialize continuum repository: {e}")
        raise RuntimeError(f"continuum repository initialization failed - cannot start MIRA: {e}") from e
    logger.info("Continuum repository initialized with connection pool")

    # Load the fixed model_configs routes from database (fail-fast at startup)
    from utils.user_context import load_model_configs
    try:
        load_model_configs()
    except Exception as e:
        logger.critical(f"Failed to load model_configs routes: {e}")
        raise RuntimeError(f"model_configs loading failed - cannot start MIRA: {e}") from e
    logger.info("model_configs routes loaded from database")

    # No payments surface in mira-OSS; cost visibility lives in utils/cost_accumulator.py.

    # LLMProvider is core infrastructure owned by main: Mira cannot operate
    # without it, so it constructs here rather than nested inside whichever
    # component happened to need it first.
    from clients.llm_provider import get_llm_provider
    try:
        shared_llm_provider = get_llm_provider()
    except Exception as e:
        logger.critical(f"Failed to initialize LLMProvider: {e}")
        raise RuntimeError(f"llm_provider initialization failed - cannot start MIRA: {e}") from e

    # Warm the dialect registry: importing every dialect module (including the
    # anthropic SDK) costs hundreds of milliseconds; doing it lazily would land
    # on the first LLM request of every fresh process instead of here at boot.
    from clients.llm.dialect_registry import get_registry
    get_registry().discover()

    # Initialize lt_memory factory following MIRA's singleton pattern
    logger.info("Initializing lt_memory factory...")
    try:
        from utils.database_session_manager import get_shared_session_manager
        from lt_memory.factory import get_lt_memory_factory

        lt_memory_factory = get_lt_memory_factory(
            session_manager=get_shared_session_manager(),
            embeddings_provider=embeddings_provider,
            llm_provider=shared_llm_provider,
            continuum_repo=continuum_repo
        )
        logger.info("lt_memory factory initialized as singleton")
    except Exception as e:
        logger.critical(f"Failed to initialize lt_memory factory: {e}")
        raise RuntimeError(f"lt_memory initialization failed - cannot start MIRA: {e}") from e

    # Initialize orchestrator as singleton
    logger.info("Initializing continuum orchestrator...")
    from cns.integration.factory import create_cns_orchestrator
    from cns.services.orchestrator import initialize_orchestrator

    try:
        orchestrator = create_cns_orchestrator()
        initialize_orchestrator(orchestrator)
    except Exception as e:
        logger.critical(f"Failed to initialize CNS orchestrator: {e}")
        raise RuntimeError(f"cns_orchestrator initialization failed - cannot start MIRA: {e}") from e
    logger.info("CNS Orchestrator initialized as global singleton")

    # Flush Valkey caches on startup except auth sessions, CSRF tokens paired
    # with them, and rate limiting. `csrf:` must be listed because
    # auth/session.py writes `csrf:<digest>` alongside `session:<digest>` and
    # the session revocation helpers delete both together — flushing CSRF
    # while preserving sessions (O-10) would 403 the first
    # cookie-authenticated write after every restart.
    # The `pending_memories*` prefixes preserve the durable queue of
    # user-confirmed manual memories (memory_tool.create_memory) plus its
    # `pending_memories_done:` / `pending_memories_attempts:` idempotency
    # markers — these exist nowhere else, so a restart must not destroy them.
    logger.info("Flushing Valkey caches (preserving sessions, CSRF tokens, rate limits and pending memories)...")
    from clients.valkey_client import get_valkey_client
    try:
        valkey_client = get_valkey_client()
        flushed_count = valkey_client.flush_except_whitelist(
            preserve_prefixes=[
                "session:", "csrf:", "rate_limit:", "heartbeat:",
                "pending_memories:", "pending_memories_done:", "pending_memories_attempts:",
            ]
        )
    except Exception as e:
        logger.critical(f"Failed to flush Valkey caches on startup: {e}")
        raise RuntimeError(f"valkey initialization failed - cannot start MIRA: {e}") from e
    logger.info(f"Flushed {flushed_count} cache keys from Valkey")

    # PlaywrightService is lazy — Chromium launches on first web_tool fetch,
    # auto-shuts down after idle timeout. Just verify the import works.
    try:
        from utils.playwright_service import PlaywrightService
        PlaywrightService.get_instance()  # Creates singleton, does NOT launch Chromium
        logger.info("PlaywrightService ready (Chromium launches on first use)")
    except ImportError as e:
        logger.warning(f"Playwright not available: {e}")
        logger.warning("web_tool will not be able to render JavaScript-heavy pages")

    # Event bus is synchronous
    logger.info("Event bus initialized (synchronous)")

    try:
        # Initialize all scheduled tasks through central registry
        initialize_all_scheduled_tasks(scheduler_service)

        # Register segment timeout detection job (needs event_bus from orchestrator)
        from utils.scheduled_tasks import register_segment_timeout_job, register_sidebar_dispatcher_job
        register_segment_timeout_job(scheduler_service, orchestrator.event_bus)

        # Register sidebar dispatcher (needs tool_repo + event_bus)
        register_sidebar_dispatcher_job(
            scheduler_service, orchestrator.tool_repo, orchestrator.event_bus
        )

        # Register heartbeat wake cycle (double gate: registration skipped when
        # disabled; the tick re-checks config every time it fires)
        from cns.services.heartbeat_service import register_heartbeat_job
        register_heartbeat_job(scheduler_service)

        scheduler_service.start()
    except Exception as e:
        logger.critical(f"Failed to initialize scheduled task system: {e}")
        raise RuntimeError(f"scheduled_task_system initialization failed - cannot start MIRA: {e}") from e

    # Collapse any segments stale during downtime through the existing event pipeline.
    # check_timeouts() publishes SegmentTimeoutEvent for stale segments, which the
    # collapse handler processes (summary + extraction). The 6-hour extract_unprocessed_segments
    # sweep catches any that fail.
    from cns.services.segment_timeout_service import get_timeout_service
    timeout_service = get_timeout_service(orchestrator.event_bus)
    try:
        timeout_service.check_timeouts()
    except Exception as e:
        logger.critical(f"Startup stale-segment timeout check failed: {e}")
        raise RuntimeError(f"segment_timeout_check failed - cannot start MIRA: {e}") from e
    logger.info("Startup timeout check complete (stale segments will collapse via event pipeline)")

    # Verify Vault connection (non-blocking)
    from clients.vault_client import test_vault_connection
    vault_status = test_vault_connection()
    if vault_status["status"] != "success":
        logger.warning(f"Vault connection issue: {vault_status['message']}")

    # Lattice federation is first-class but opt-in via config.lattice.enabled.
    # When enabled it must work or the service aborts startup loudly; when
    # disabled nothing is imported and the /v0/api/federation router stays unmounted.
    if config.lattice.enabled:
        logger.info("Initializing Lattice federation username resolver...")
        try:
            from lattice.username_resolver import set_username_resolver
            from clients.postgres_client import PostgresClient
            from typing import Optional

            def mira_resolve_username(username: str) -> Optional[str]:
                """Resolve username to user_id for Lattice federation."""
                db = PostgresClient("mira_service")
                result = db.execute_single(
                    "SELECT user_id FROM global_usernames WHERE username = %(username)s AND active = true",
                    {"username": username.lower()}
                )
                return str(result["user_id"]) if result else None

            set_username_resolver(mira_resolve_username)
        except ImportError as e:
            logger.critical(f"Lattice package unavailable but federation is enabled: {e}")
            raise RuntimeError(f"lattice initialization failed - cannot start MIRA: {e}") from e
        except Exception as e:
            logger.critical(f"Failed to register Lattice username resolver: {e}")
            raise RuntimeError(f"lattice initialization failed - cannot start MIRA: {e}") from e
        logger.info("Lattice username resolver registered")
    else:
        logger.info("Lattice federation disabled (enable via config.lattice.enabled=true)")


    # MCP /v0/mcp (mounted in create_app only when config.system.mcp_enabled):
    # mounted sub-apps receive no lifespan events, so the MCP session manager's
    # task group is entered here — once per process — and closed at the very
    # end of shutdown below.
    mcp_stack = AsyncExitStack()
    mcp_manager = getattr(app.state, "mcp_session_manager", None)
    if mcp_manager is not None:
        await mcp_stack.enter_async_context(mcp_manager.run())
        logger.info("MCP /v0/mcp session manager running (check_in tool available)")

    # Self-edit trial boot (utils/self_edit.py): the edited tree just completed
    # startup, so it becomes the new last-good commit. No-op on every
    # non-trial start.
    from utils.self_edit import commit_trial_if_booting
    commit_trial_if_booting()

    logger.info("MIRA startup complete")
    
    
    
    
    ###
    #
    
    yield
    
    #
    ###
    
    
    
    
    
    # Shutdown
    logger.info("Shutting down MIRA...")
    scheduler_service.stop()

    # Close all active WebSocket connections
    from cns.api.websocket_chat import close_all_connections
    await close_all_connections()
    logger.info("WebSocket connections closed")
    
    # Shutdown event bus
    if orchestrator.event_bus:
        orchestrator.event_bus.shutdown()
        logger.info("Event bus shutdown complete")
    
    # Shutdown Valkey client
    from clients.valkey_client import get_valkey_client
    valkey = get_valkey_client()
    if valkey:
        await valkey.shutdown()
    
    # Clean up singleton resources
    logger.info("Cleaning up singleton resources...")

    # Clean up lt_memory factory
    try:
        lt_memory_factory = get_lt_memory_factory()
        if lt_memory_factory:
            lt_memory_factory.cleanup()
            logger.info("LT_Memory factory cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up LT_Memory factory: {e}")

    # Shutdown PlaywrightService
    try:
        from utils.playwright_service import PlaywrightService
        if PlaywrightService._instance:
            PlaywrightService._instance.shutdown()
            logger.info("PlaywrightService shutdown complete")
    except Exception as e:
        logger.warning(f"Error shutting down PlaywrightService: {e}")

    # Clean up UserDataManager SQLite connections
    from utils.userdata_manager import clear_manager_cache
    clear_manager_cache()
    logger.info("UserDataManager cache cleared (SQLite connections closed)")

    # Clean up database connections
    from clients.postgres_client import PostgresClient as ShutdownPostgresClient
    ShutdownPostgresClient.close_all_pools()
    logger.info("PostgreSQL connection pools closed")
    
    from utils.database_session_manager import get_shared_session_manager
    get_shared_session_manager().cleanup()

    # MCP session manager task group closes last of all.
    await mcp_stack.aclose()
    logger.info("MIRA shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="MIRA",
        description="A lil Brain-in-a-Box",
        version=(Path(__file__).resolve().parent / "VERSION").read_text().strip(),
        lifespan=lifespan
    )
    
    # Global exception handlers for consistent error responses
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors."""
        request_id = generate_request_id()
        errors = exc.errors()
        
        # Format validation errors consistently
        formatted_errors = []
        for error in errors:
            formatted_errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        response = create_error_response(
            APIError("VALIDATION_ERROR", "Request validation failed", {"errors": formatted_errors}),
            request_id
        )
        return JSONResponse(
            status_code=422,
            content=response.to_dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError):
        """Handle FastAPI request validation errors."""
        request_id = generate_request_id()
        errors = exc.errors()
        
        # Format validation errors with field details
        formatted_errors = []
        for error in errors:
            formatted_errors.append({
                "loc": error["loc"],
                "msg": error["msg"],
                "type": error["type"]
            })
        
        _ = create_error_response(
            APIError("REQUEST_VALIDATION_ERROR", "Invalid request format", {"detail": formatted_errors}),
            request_id
        )
        # Keep FastAPI's standard validation error format for compatibility
        return JSONResponse(
            status_code=422,
            content={"detail": formatted_errors}
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTPExceptions raised with the standard error envelope as detail.

        The auth dependency ladder raises HTTPException(detail=<flat envelope
        dict>); FastAPI's default wrapper would nest it under "detail", which
        the TUI's error extractor misses. Flatten only when the detail carries
        the success/error envelope keys; string details keep the default
        {"detail": ...} shape for their non-TUI consumers.
        """
        detail = exc.detail
        if isinstance(detail, dict) and "success" in detail and "error" in detail:
            return JSONResponse(
                status_code=exc.status_code,
                headers=exc.headers,
                content=detail,
            )
        return await http_exception_handler(request, exc)
    
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        """Handle custom API errors."""
        request_id = generate_request_id()
        response = create_error_response(exc, request_id)

        status_code = 400  # Default to bad request
        if exc.code == "NOT_FOUND":
            status_code = 404
        elif exc.code == "UNAUTHORIZED":
            status_code = 401
        elif exc.code == "FORBIDDEN":
            status_code = 403
        elif exc.code == "SERVICE_UNAVAILABLE":
            status_code = 503
        elif exc.code == "INTERNAL_ERROR":
            status_code = 500
        elif exc.code == "RATE_LIMIT_EXCEEDED":
            status_code = 429
        
        return JSONResponse(
            status_code=status_code,
            content=response.to_dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all unhandled exceptions."""
        request_id = generate_request_id()
        
        # Log the actual error for debugging
        logger.error(f"Unhandled exception (request_id: {request_id}): {exc}", exc_info=True)
        
        # Return safe error message to client
        response = create_error_response(
            APIError("INTERNAL_ERROR", "An unexpected error occurred", {"request_id": request_id}),
            request_id
        )
        return JSONResponse(
            status_code=500,
            content=response.to_dict()
        )
    
    # Middleware stack (order matters — applied in reverse registration order)
    from utils.perf import PerfMiddleware
    app.add_middleware(PerfMiddleware)
    # CSP posture is governed by MIRA_CSP=off|strict, parsed strictly when
    # this constructor runs (auth/security_middleware.py). It takes no
    # importmap hash: the CRM workspace import-map exception has no
    # counterpart in the retained web UI.
    app.add_middleware(SecurityHeadersMiddleware)

    if config.api_server.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api_server.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
    
    # API routes - v0 versioning (beta signal)
    app.include_router(health.router, prefix="/v0/api", tags=["health"])
    app.include_router(update.router, prefix="/v0/api", tags=["update"])  # Public update check
    # The auth surface mounts unconditionally in both modes; every caller
    # authenticates through the shared credential ladder (session cookie →
    # issued API token).
    app.include_router(auth_api.router, prefix="/v0/auth", tags=["auth"])
    app.include_router(chat_api.router, prefix="/v0/api", tags=["chat"])
    app.include_router(data.router, prefix="/v0/api", tags=["data"])
    app.include_router(actions.router, prefix="/v0/api", tags=["actions"])
    app.include_router(tool_config.router, prefix="/v0/api", tags=["tool_config"])
    app.include_router(trigger_rules.router, prefix="/v0/api", tags=["trigger_rules"])
    app.include_router(files_api.router, prefix="/v0/api", tags=["files"])
    app.include_router(location.router, prefix="/v0/api", tags=["location"])
    app.include_router(heartbeat_api.router, prefix="/v0/api", tags=["heartbeat"])
    app.include_router(websocket_chat.router, prefix="/v0", tags=["websocket"])  # /v0/ws/chat
    if config.lattice.enabled:
        app.include_router(federation_api.router, prefix="/v0/api", tags=["federation"])

    # MCP check-in endpoint (opt-in, config.system.mcp_enabled — default off).
    # Lazy import: disabled mode constructs nothing MCP-related at boot — no
    # SDK import, no app, no session manager. mount() parks the session
    # manager on app.state for the lifespan block below.
    if config.system.mcp_enabled:
        from cns.api import mcp as mcp_api
        mcp_api.mount(app)

    # No payments routes in mira-OSS; cost visibility lives in utils/cost_accumulator.py.

    # Performance monitoring (gated by mira.perf logger level)
    from utils.perf import register_perf_routes, install_db_instrumentation
    register_perf_routes(app)
    install_db_instrumentation()

    # Full web UI — page routes, plus root meta files and the /assets
    # static mount. App pages gate behind get_current_user_for_pages in
    # every mode: under `single` an unauthenticated visit round-trips
    # through /v0/auth/local/session and its cookie, and under `multi` it
    # gets the standard 401 envelope until the deployment ships its own
    # sign-in surface (mira-OSS has no /login/ page). Root meta files
    # and /assets carry no user data and stay public so a gated page can
    # still load its own JS/CSS.
    page_dependencies = [Depends(auth_api.get_current_user_for_pages)]
    if Path("web").exists():
        @app.get("/", include_in_schema=False)
        async def serve_root():
            return RedirectResponse(url="/chat")

        @app.get("/chat", include_in_schema=False, dependencies=page_dependencies)
        @app.get("/chat/", include_in_schema=False, dependencies=page_dependencies)
        async def serve_chat():
            return FileResponse("web/chat/index.html")

        @app.get("/memories", include_in_schema=False, dependencies=page_dependencies)
        @app.get("/memories/", include_in_schema=False, dependencies=page_dependencies)
        async def serve_memories():
            return FileResponse("web/memories/index.html")

        @app.get("/domaindocs", include_in_schema=False, dependencies=page_dependencies)
        @app.get("/domaindocs/", include_in_schema=False, dependencies=page_dependencies)
        async def serve_domaindocs():
            return FileResponse("web/domaindocs/index.html")

        @app.get("/settings", include_in_schema=False, dependencies=page_dependencies)
        @app.get("/settings/", include_in_schema=False, dependencies=page_dependencies)
        async def serve_settings():
            return FileResponse("web/settings/index.html")

        # Browser-expected static files from root
        @app.get("/apple-touch-icon.png", include_in_schema=False)
        async def serve_apple_touch_icon():
            return FileResponse("web/apple-touch-icon.png")

        @app.get("/favicon.ico", include_in_schema=False)
        async def serve_favicon():
            return FileResponse("web/favicon.ico")

        @app.get("/manifest.json", include_in_schema=False)
        async def serve_manifest():
            return FileResponse("web/manifest.json")

        # Static assets (JS/CSS/fonts/images) — mounted after page routes
        app.mount("/assets", StaticFiles(directory="web/assets"), name="assets")

    return app


def main():
    """Main entry point."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MIRA - AI Assistant with persistent memory')
    parser.add_argument('--firehose', action='store_true',
                       help='Enable firehose mode: log all LLM API calls to firehose_output.json for debugging')
    args = parser.parse_args()

    # Firehose: toggle live with kill -USR1 $(systemctl show mira -p MainPID --value)
    if args.firehose:
        from utils.llm_tap import toggle as _toggle_traffic_tap
        _toggle_traffic_tap(None, None)

    try:
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, config.system.log_level.upper(), logging.INFO))
        
        logger.info(f"Starting MIRA on {config.api_server.host}:{config.api_server.port}")
        
        # HTTP/2 is required to prevent connection blocking during streaming
        import hypercorn.asyncio
        from hypercorn import Config
        
        logger.info("Starting with Hypercorn (HTTP/2 enabled)")
        
        hypercorn_config = Config()
        hypercorn_config.bind = [f"{config.api_server.host}:{config.api_server.port}"]
        hypercorn_config.alpn_protocols = ["h2", "http/1.1"]  # Prefer HTTP/2, fallback to HTTP/1.1
        hypercorn_config.log_level = config.api_server.log_level

        # NOTE (recorded topology decision): this assignment is a NO-OP — Hypercorn's
        # Config has never had a `forwarded_allow_ips` attribute, and no proxy-header
        # trust is configured anywhere. X-Forwarded-For is deliberately ignored:
        # no proxy ships with MIRA, so request.client.host is the socket peer. If a
        # reverse proxy is ever put in front of MIRA, add a loopback-gated
        # proxy-header middleware here rather than re-flagging this line.
        hypercorn_config.forwarded_allow_ips = ["127.0.0.1", "::1"]
        
        
        hypercorn_config.workers = config.api_server.workers

        from utils.power_on_self_test import run_pre_server_post_gate
        run_pre_server_post_gate()
        
        # Run the server — Hypercorn manages SIGTERM/SIGINT natively
        asyncio.run(hypercorn.asyncio.serve(create_app(), hypercorn_config))
        
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        sys.exit(1)

    # A graceful SIGTERM returns serve() normally (exit 0), which neither
    # systemd's Restart=on-failure nor launchd's KeepAlive treats as a reason
    # to restart. A self-edit restart must exit non-zero to come back up.
    from utils.self_edit import RESTART_EXIT_CODE, restart_requested
    if restart_requested():
        logger.warning("Exiting with %d so the supervisor restarts MIRA", RESTART_EXIT_CODE)
        sys.exit(RESTART_EXIT_CODE)


if __name__ == "__main__":
    main()
