#!/usr/bin/env python3
"""
MIRA - Main Application Entry Point
FastAPI server that wires together the CNS architecture and handles startup/shutdown.
"""

    #@model it seems like the main.py was modified iterarively over time and the order of code fell by the wayside. it should be shuffled around. even if functionalally it works the same it is good code hygene


import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from utils.logging_config import setup_colored_root_logging, setup_anthropic_sdk_logging
setup_colored_root_logging(log_level=logging.WARNING, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Deployment installs set MIRA_LOG_DIR=/opt/mira/logs (systemd unit, macOS
# launcher, s6 run script). Everywhere else the relative default keeps the
# import pure-Python instead of touching a machine-level install path.
# @model is this a comment worth keeping ^^^
setup_anthropic_sdk_logging(log_dir=os.environ.get("MIRA_LOG_DIR", "logs"))

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from auth.mode import auth_mode
from config.config_manager import config
from config.announcement import load_announcement
from cns.api import data, actions, health, websocket_chat, tool_config, trigger_rules, update, federation as federation_api
from cns.api import chat as chat_api
from cns.api import files as files_api
from cns.api import location
#@model the cns.api imports could be laid out cleaner
from auth import api as auth_api
from cns.api.base import APIError, create_error_response, generate_request_id
#@model shouldt this go up a line to be with the other cns.api?
from auth.security_middleware import SecurityHeadersMiddleware
from utils.scheduler_service import scheduler_service
from utils.scheduled_tasks import initialize_all_scheduled_tasks

# Suppress routine APScheduler job execution logs (running/success/debug chatter)
logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)
logging.getLogger('apscheduler.scheduler').setLevel(logging.WARNING)
# logging.getLogger('tools.implementations.imagegen_tool').setLevel(logging.DEBUG)
#@model was the imagegen tool removed?

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


    # Configure FastAPI thread pool for synchronous endpoints
    from anyio import to_thread
    to_thread.current_default_thread_limiter().total_tokens = 100
    logger.info("FastAPI thread pool configured for 100 concurrent threads")
   
    #@model the thread count should be conditional based on if we're in one user or multi user mode'
    
     
    # Pre-initialize expensive singleton resources at startup
    logger.info("Pre-initializing singleton resources...")

    # Preload all Vault secrets into memory cache (prevents token expiration issues)
    from clients.vault_client import preload_secrets
    preload_secrets()

    # Load announcement config (cached for lifetime of process)
    load_announcement()

    # Initialize embeddings provider (loads mdbr-leaf-ir-asym 768d model)
    from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
    embeddings_provider = get_hybrid_embeddings_provider()
    logger.info(f"Embeddings provider initialized: {type(embeddings_provider).__name__}")
    
    # Initialize continuum repository (creates DB connection pool)
    from cns.infrastructure.continuum_repository import get_continuum_repository
    continuum_repo = get_continuum_repository()
    logger.info("Continuum repository initialized with connection pool")

    # Load the fixed model_configs routes from database (fail-fast at startup)
    from utils.user_context import load_model_configs
    load_model_configs()
    logger.info("model_configs routes loaded from database")

    # The payments subsystem is not part of mira-OSS (decision D7): there
    # is no paid-account pricing module in this tree, and cost visibility
    # lives in utils/cost_accumulator.py against usage_pricing rows seeded
    # by the greenfield schema.
    
    #@model good flag. this needs a proper finalized aporoach because mira is now OSS centric. 

    # Initialize lt_memory factory following MIRA's singleton pattern
    logger.info("Initializing lt_memory factory...")
    try:
        from clients.llm_provider import LLMProvider
        #@model why are we init llmprovider inside memory factory? what if factory is disabled? we'll still need llm provider globally in main and then reference it, no?
        from utils.database_session_manager import get_shared_session_manager
        from lt_memory.factory import get_lt_memory_factory

        lt_memory_llm_provider = LLMProvider()

        lt_memory_factory = get_lt_memory_factory(
            session_manager=get_shared_session_manager(),
            embeddings_provider=embeddings_provider,
            llm_provider=lt_memory_llm_provider,
            conversation_repo=continuum_repo
        )
        logger.info("lt_memory factory initialized as singleton")
    except Exception as e:
        logger.critical(f"Failed to initialize lt_memory factory: {e}")
        raise RuntimeError(f"lt_memory initialization failed - cannot start MIRA: {e}") from e
        #@model we should standardize rhe cannot start messages across all crucial startup components

    # Initialize orchestrator as singleton
    logger.info("Initializing continuum orchestrator...")
    from cns.integration.factory import create_cns_orchestrator
    from cns.services.orchestrator import initialize_orchestrator

    orchestrator = create_cns_orchestrator()
    initialize_orchestrator(orchestrator)
    logger.info("CNS Orchestrator initialized as global singleton")

    # Flush Valkey caches on startup except auth sessions, CSRF tokens paired
    # with them, and rate limiting. `csrf:` must be listed because
    # auth/session.py writes `csrf:<digest>` alongside `session:<digest>` and
    # the session revocation helpers delete both together — flushing CSRF
    # while preserving sessions (O-10) would 403 the first
    # cookie-authenticated write after every restart.
    logger.info("Flushing Valkey caches (preserving sessions, CSRF tokens and rate limits)...")
    from clients.valkey_client import get_valkey_client
    valkey_client = get_valkey_client()
    flushed_count = valkey_client.flush_except_whitelist(
        preserve_prefixes=["session:", "csrf:", "rate_limit:"]
    )
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

    # Initialize all scheduled tasks through central registry
    initialize_all_scheduled_tasks(scheduler_service)

    # Register segment timeout detection job (needs event_bus from orchestrator)
    from utils.scheduled_tasks import register_segment_timeout_job, register_sidebar_dispatcher_job
    register_segment_timeout_job(scheduler_service, orchestrator.event_bus)

    # Register sidebar dispatcher (needs tool_repo + event_bus)
    register_sidebar_dispatcher_job(
        scheduler_service, orchestrator.tool_repo, orchestrator.event_bus
    )

    scheduler_service.start()

    # Collapse any segments stale during downtime through the existing event pipeline.
    # check_timeouts() publishes SegmentTimeoutEvent for stale segments, which the
    # collapse handler processes (summary + extraction). The 6-hour extract_unprocessed_segments
    # sweep catches any that fail.
    from cns.services.segment_timeout_service import get_timeout_service
    timeout_service = get_timeout_service(orchestrator.event_bus)
    timeout_service.check_timeouts()
    logger.info("Startup timeout check complete (stale segments will collapse via event pipeline)")

    # Verify Vault connection (non-blocking)
    from clients.vault_client import test_vault_connection
    vault_status = test_vault_connection()
    if vault_status["status"] != "success":
        logger.warning(f"Vault connection issue: {vault_status['message']}")

    # Register Lattice username resolver for federation
    # This allows Lattice to resolve usernames to user_ids for inbound message delivery
    #@model during this transition to v2 we should make lattice a firsr class citizen even if its disabled for netoskr security reasons. its a useful functionality to bundle
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
        logger.info("Lattice username resolver registered")
    except ImportError:
        logger.warning("Lattice package not available - federation disabled")
    except Exception as e:
        logger.warning(f"Failed to register Lattice username resolver: {e}")


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
        valkey.shutdown()
        logger.info("Valkey client shutdown complete")
    
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
    from clients.postgres_client import PostgresClient
    PostgresClient.close_all_pools()
    logger.info("PostgreSQL connection pools closed")
    
    from utils.database_session_manager import get_shared_session_manager
    get_shared_session_manager().cleanup()
    
    logger.info("MIRA shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="MIRA",
        description="A lil Brain-in-a-Box",
        version="2026.03.07-major",
        #@model this needs to be updated or better yet tied to the VERSION
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
        
        response = create_error_response(
            APIError("REQUEST_VALIDATION_ERROR", "Invalid request format", {"detail": formatted_errors}),
            request_id
        )
        # Keep FastAPI's standard validation error format for compatibility
        return JSONResponse(
            status_code=422,
            content={"detail": formatted_errors}
        )
    
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        """Handle custom API errors."""
        request_id = generate_request_id()
        response = create_error_response(exc, request_id)
        
        # Determine status code based on error code
        #@model some of the commenrs do not add much value... they add to the noise. of course this code determines status codes.
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
    # The multi-user auth surface is mounted in all three modes (plan
    # §6.3.4): `single` authenticates its callers through the union branch
    # of get_current_user, and dev/multi through sessions and API tokens.
    #@model same thing down here. standardixing on single (rename dev mode) and multi. remove v1 holdover. 
    app.include_router(auth_api.router, prefix="/v0/auth", tags=["auth"])
    app.include_router(chat_api.router, prefix="/v0/api", tags=["chat"])
    app.include_router(data.router, prefix="/v0/api", tags=["data"])
    app.include_router(actions.router, prefix="/v0/api", tags=["actions"])
    app.include_router(tool_config.router, prefix="/v0/api", tags=["tool_config"])
    app.include_router(trigger_rules.router, prefix="/v0/api", tags=["trigger_rules"])
    app.include_router(files_api.router, prefix="/v0/api", tags=["files"])
    #@model iirc files api support was removed
    app.include_router(location.router, prefix="/v0/api", tags=["location"])
    app.include_router(websocket_chat.router, prefix="/v0", tags=["websocket"])  # /v0/ws/chat
    app.include_router(federation_api.router, prefix="/v0/api", tags=["federation"])

    # Payments routes are not part of mira-OSS (decision D7): there is no
    # paid-account module in this tree, and account access is not a product
    # surface here. Cost visibility lives in utils/cost_accumulator.py.
    #@model handle this as per above too

    # Performance monitoring (gated by mira.perf logger level)
    from utils.perf import register_perf_routes, install_db_instrumentation
    register_perf_routes(app)
    install_db_instrumentation()

    # Full web UI — page routes, plus root meta files and the /assets
    # static mount. App pages gate behind get_current_user_for_pages in
    # every mode: under `single` an unauthenticated visit round-trips
    # through /v0/auth/local/session and its cookie, and under `multi` it
    # gets the standard 401 envelope until the deployment ships its own
    # sign-in surface (mira-OSS has no /login/ page — D8). Root meta files
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

        # Trust proxy headers from nginx (localhost only)
        # This allows proper client IP logging from X-Forwarded-For header
        hypercorn_config.forwarded_allow_ips = ["127.0.0.1", "::1"]
        
        
        hypercorn_config.workers = config.api_server.workers

        from utils.power_on_self_test import run_pre_server_post_gate
        run_pre_server_post_gate()
        
        # Run the server — Hypercorn manages SIGTERM/SIGINT natively
        asyncio.run(hypercorn.asyncio.serve(create_app(), hypercorn_config))
        
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
