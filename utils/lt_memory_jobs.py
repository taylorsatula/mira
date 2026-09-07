"""
Scheduled task registration for LT_Memory system.

Registers periodic jobs for memory extraction and maintenance. Day-interval jobs use
modular arithmetic on cumulative_activity_days for stateless, use-day-based
scheduling via get_users_due_for_job().
"""
import logging
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


def register_lt_memory_jobs(scheduler_service, lt_memory_factory) -> None:
    """
    Register all LT_Memory scheduled jobs with the scheduler.

    Jobs registered:
    - Extraction retry sweep (6-hour intervals, calendar-based)
    - Temporal score recalculation (daily tick, use-day gated)
    - Bulk score recalculation (daily tick, use-day gated)

    Args:
        scheduler_service: System scheduler service instance
        lt_memory_factory: LTMemoryFactory instance with all services

    Raises:
        RuntimeError: If job registration fails
    """
    from config import config
    extraction_orchestrator = lt_memory_factory.extraction_orchestrator
    jobs_config = config.scheduled_jobs

    # ================================================================
    # Calendar-based jobs (not use-day gated)
    # ================================================================

    # Extraction retry sweep (6-hour intervals)
    scheduler_service.register_job(
        job_id="lt_memory_extract_unprocessed_segments",
        func=extraction_orchestrator.extract_unprocessed_segments,
        trigger=IntervalTrigger(hours=jobs_config.extraction_retry_hours),
        component="lt_memory",
        description=f"Extract unprocessed collapsed segments every {jobs_config.extraction_retry_hours} hours (safety net)"
    )
    logger.info("Registered extraction retry sweep (%dh interval)", jobs_config.extraction_retry_hours)

    # ================================================================
    # Use-day-gated jobs (daily tick, filtered by modular arithmetic)
    # ================================================================

    # Temporal score recalculation
    def run_temporal_score_recalculation():
        from utils.user_context import set_current_user_id, clear_user_context
        from utils.scheduled_tasks import get_users_due_for_job

        users = get_users_due_for_job(jobs_config.temporal_score_recalc_use_days)
        total_updated = 0
        for user in users:
            user_id = str(user["id"])
            set_current_user_id(user_id)
            try:
                db = lt_memory_factory.db
                updated = db.recalculate_temporal_scores(user_id=user_id, batch_size=1000)
                total_updated += updated
            finally:
                clear_user_context()

        logger.info("Temporal score sweep: updated %d memories across %d due users", total_updated, len(users))
        return {"memories_updated": total_updated}

    scheduler_service.register_job(
        job_id="lt_memory_temporal_score_recalculation",
        func=run_temporal_score_recalculation,
        trigger=IntervalTrigger(days=1),
        component="lt_memory",
        description=f"Recalculate temporal memory scores (every {jobs_config.temporal_score_recalc_use_days} use-days)"
    )
    logger.info("Registered temporal score recalculation (every %d use-days)", jobs_config.temporal_score_recalc_use_days)

    # Bulk score recalculation
    def run_bulk_score_recalculation():
        from utils.user_context import set_current_user_id, clear_user_context
        from utils.scheduled_tasks import get_users_due_for_job

        users = get_users_due_for_job(jobs_config.bulk_score_recalc_use_days)
        total_updated = 0
        for user in users:
            user_id = str(user["id"])
            set_current_user_id(user_id)
            try:
                db = lt_memory_factory.db
                updated = db.bulk_recalculate_scores(user_id=user_id, batch_size=1000)
                total_updated += updated
            finally:
                clear_user_context()

        logger.info("Bulk score recalculation sweep: updated %d stale memories across %d due users", total_updated, len(users))
        return {"memories_updated": total_updated}

    scheduler_service.register_job(
        job_id="lt_memory_bulk_score_recalculation",
        func=run_bulk_score_recalculation,
        trigger=IntervalTrigger(days=1),
        component="lt_memory",
        description=f"Recalculate stale memory scores (every {jobs_config.bulk_score_recalc_use_days} use-days)"
    )
    logger.info("Registered bulk score recalculation (every %d use-days)", jobs_config.bulk_score_recalc_use_days)

    # Entity merge — background LLM-driven dedup of similar entity rows
    def run_entity_merge_for_due_users():
        from utils.user_context import set_current_user_id, clear_user_context
        from utils.scheduled_tasks import get_users_due_for_job
        from lt_memory.entity_merge import run_entity_merge_for_user

        users = get_users_due_for_job(jobs_config.entity_merge_use_days)
        total_merged = 0
        for user in users:
            user_id = str(user["id"])
            set_current_user_id(user_id)
            try:
                stats = run_entity_merge_for_user(user_id)
                total_merged += stats.get("merged", 0)
            finally:
                clear_user_context()

        logger.info("Entity merge sweep: merged %d entities across %d due users", total_merged, len(users))
        return {"entities_merged": total_merged}

    scheduler_service.register_job(
        job_id="lt_memory_entity_merge",
        func=run_entity_merge_for_due_users,
        trigger=IntervalTrigger(days=1),
        component="lt_memory",
        description=f"Dedup/merge similar entities via LLM judge (every {jobs_config.entity_merge_use_days} use-days)"
    )
    logger.info("Registered entity merge (every %d use-days)", jobs_config.entity_merge_use_days)

    logger.info("All LT_Memory scheduled jobs registered successfully")
