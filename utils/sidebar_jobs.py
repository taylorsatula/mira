"""
Sidebar agent scheduler registration.

Registers the SidebarDispatcher as an APScheduler job that polls
registered triggers on a configurable interval.
"""
import logging

logger = logging.getLogger(__name__)


def register_sidebar_jobs(scheduler_service, tool_repo, event_bus) -> None:
    """Register the sidebar dispatcher poll job if enabled."""
    from apscheduler.triggers.interval import IntervalTrigger
    from config.config_manager import config

    sidebar_config = config.sidebar_dispatcher
    if not sidebar_config.enabled:
        logger.info("Sidebar dispatcher disabled, skipping job registration")
        return

    from agents.sidebar import SidebarDispatcher
    from agents.triggers.registry import TRIGGER_CLASSES

    dispatcher = SidebarDispatcher(
        tool_repo=tool_repo,
        event_bus=event_bus,
        max_concurrent_agents=sidebar_config.max_concurrent_agents,
    )

    # Register sidebar triggers (pure discovery — the dispatcher owns dedup).
    # TRIGGER_CLASSES is the single trigger registry: the trigger-rules API
    # derives its valid trigger/field set from the same list
    # (agents/triggers/registry.py), so what dispatches is exactly what is
    # rule-addressable.
    for trigger_class in TRIGGER_CLASSES:
        dispatcher.register_trigger(trigger_class())

    scheduler_service.register_job(
        job_id="sidebar_dispatcher_poll",
        func=dispatcher.poll,
        trigger=IntervalTrigger(minutes=sidebar_config.poll_interval_minutes),
        component="sidebar",
        description="Poll sidebar triggers for new work items",
    )

    logger.info(
        f"Sidebar dispatcher registered: poll every "
        f"{sidebar_config.poll_interval_minutes}m, "
        f"max {sidebar_config.max_concurrent_agents} concurrent agents"
    )
