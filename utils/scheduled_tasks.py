"""
Centralized scheduled task registration.

This module provides a single place to configure all scheduled tasks
while keeping the actual scheduling logic with the owning components.

Also provides `get_users_due_for_job()` — a reusable platform function
for use-day-based scheduling via modular arithmetic on cumulative_activity_days.
"""
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


# Registry of modules with scheduled tasks
# Format: (module_path, service_name, needs_init, init_args_factory)
SCHEDULED_TASK_MODULES: List[Tuple[str, str, bool, callable]] = [
    # Segment timeout detection - registered separately (needs event_bus)
    # See register_segment_timeout_job() below
]

# The auth service's cleanup jobs (expired magic links hourly, account GC
# nightly) maintain state that only exists once the multi-user auth stack
# is live. In `single` mode nothing ever creates a magic link or a Valkey
# session, so the entry is not registered — and auth.service is not even
# imported — keeping the single-user boot free of any assumption that the
# hosted-auth layer is constructed (92d768c re-added this registration
# unconditionally; mode-gating it is plan §6.3.4's scheduler row).
# needs_init=True routes through the lazy `get_auth_service()` factory
# rather than a module-level singleton.
AUTH_SERVICE_TASK = ("auth.service", "get_auth_service", True, lambda: {})


def scheduled_task_modules() -> List[Tuple[str, str, bool, callable]]:
    """The registry as it applies to this process's auth mode."""
    from auth.mode import single_user_mode_enabled

    if single_user_mode_enabled():
        return list(SCHEDULED_TASK_MODULES)
    return [*SCHEDULED_TASK_MODULES, AUTH_SERVICE_TASK]



def initialize_all_scheduled_tasks(scheduler_service):
    """
    Initialize and register all scheduled tasks.

    Args:
        scheduler_service: The system scheduler service instance

    Returns:
        int: Number of successfully registered services

    Raises:
        RuntimeError: If any job registration fails
    """
    successful = 0

    # First register standard services from the registry
    registry = scheduled_task_modules()
    for module_path, service_name, needs_init, init_args_factory in registry:
        try:
            # Dynamic import
            module = __import__(module_path, fromlist=[service_name])

            if needs_init:
                # Service needs initialization
                service_class = getattr(module, service_name)
                init_args = init_args_factory() if init_args_factory else {}
                service = service_class(**init_args)
            else:
                # Service is already a global instance
                service = getattr(module, service_name)

            # Register the service's scheduled jobs (raises on failure)
            if hasattr(service, 'register_cleanup_jobs'):
                service.register_cleanup_jobs(scheduler_service)
            elif hasattr(service, 'register_jobs'):
                service.register_jobs(scheduler_service)
            else:
                logger.warning(f"Service {service_name} has no job registration method")
                continue

            logger.info(f"Successfully registered scheduled tasks for {service_name}")
            successful += 1

        except Exception as e:
            logger.error(f"Error loading scheduled tasks from {module_path}: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to register scheduled tasks for {module_path}.{service_name}: {e}"
            ) from e

    # Register LT_Memory jobs using its special pattern
    try:
        from utils.lt_memory_jobs import register_lt_memory_jobs
        from lt_memory.factory import get_lt_memory_factory

        lt_memory_factory = get_lt_memory_factory()
        if not lt_memory_factory:
            raise RuntimeError("LT_Memory factory not initialized")

        register_lt_memory_jobs(scheduler_service, lt_memory_factory)

        logger.info("Successfully registered scheduled tasks for lt_memory")
        successful += 1

    except Exception as e:
        logger.error(f"Error registering LT_Memory scheduled tasks: {e}", exc_info=True)
        raise RuntimeError(f"Failed to register LT_Memory scheduled tasks: {e}") from e

    total_services = len(registry) + 1  # +1 lt_memory
    logger.info(f"Scheduled task initialization complete: {successful}/{total_services} services registered")
    return successful


def register_segment_timeout_job(scheduler_service, event_bus) -> None:
    """
    Register segment timeout detection job (called separately after event_bus initialization).

    Args:
        scheduler_service: System scheduler service
        event_bus: CNS event bus for publishing timeout events

    Raises:
        RuntimeError: If job registration fails
    """
    try:
        from cns.services.segment_timeout_service import register_timeout_job

        register_timeout_job(scheduler_service, event_bus)
        logger.info("Successfully registered segment timeout detection job")

    except Exception as e:
        logger.error(f"Error registering segment timeout job: {e}", exc_info=True)
        raise RuntimeError(f"Failed to register segment timeout job: {e}") from e


def register_sidebar_dispatcher_job(scheduler_service, tool_repo, event_bus) -> None:
    """
    Register sidebar dispatcher job (called separately after tool_repo initialization).

    Args:
        scheduler_service: System scheduler service
        tool_repo: ToolRepository for sidebar agents
        event_bus: CNS event bus for sidebar agents

    Raises:
        RuntimeError: If job registration fails
    """
    try:
        from utils.sidebar_jobs import register_sidebar_jobs

        register_sidebar_jobs(scheduler_service, tool_repo, event_bus)
        logger.info("Successfully registered sidebar dispatcher job")

    except Exception as e:
        logger.error(f"Error registering sidebar dispatcher job: {e}", exc_info=True)
        raise RuntimeError(f"Failed to register sidebar dispatcher job: {e}") from e


def get_users_due_for_job(interval: int) -> list[dict]:
    """
    Get recently-active users whose cumulative_activity_days falls on the interval.

    Uses MOD() for stateless use-day scheduling: a user is "due" when
    MOD(cumulative_activity_days, interval) = 0. The 2-day recency window
    on last_activity_date handles timezone skew and prevents re-processing
    users whose counter is stuck on a multiple.

    Args:
        interval: Use-day interval (e.g., 7 = every 7th activity day)

    Returns:
        List of user dicts with 'id' key
    """
    from utils.database_session_manager import get_shared_session_manager

    session_manager = get_shared_session_manager()
    with session_manager.get_admin_session() as session:
        return session.execute_query("""
            SELECT id FROM users
            WHERE cumulative_activity_days > 0
            AND MOD(cumulative_activity_days, %(interval)s) = 0
            AND last_activity_date >= CURRENT_DATE - INTERVAL '2 days'
            AND memory_manipulation_enabled = TRUE
            AND is_active = TRUE
        """, {'interval': interval})
