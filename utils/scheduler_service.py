
"""
The APScheduler wrapper every scheduled job is registered through.

``register_job`` records the registration and, once the scheduler is started,
adds it to the live scheduler as well. ``start`` adds every already-registered
job before calling APScheduler's ``start()``: adding a job to an already-running
scheduler makes APScheduler recalculate schedules after each addition, and
registering jobs one at a time against a running scheduler is the shape this
ordering exists to avoid.

``stop`` shuts down without waiting, because ``AsyncIOExecutor`` ignores
``wait=True`` anyway; a job stuck in a thread cannot be waited on.

``scheduler_service`` is the process singleton ``main.py`` wires.
"""
import logging
from typing import Dict, Callable
from dataclasses import dataclass

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.base import BaseTrigger

logger = logging.getLogger(__name__)


@dataclass
class JobRegistration:
    job_id: str
    func: Callable
    trigger: BaseTrigger
    component: str
    description: str
    replace_existing: bool = True


class SchedulerService:
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone='UTC')
        self._running = False
        self._registered_jobs: Dict[str, JobRegistration] = {}
        
    
    def register_job(
        self,
        job_id: str,
        func: Callable,
        trigger: BaseTrigger,
        component: str,
        description: str,
        replace_existing: bool = True
    ) -> None:
        """
        Register a scheduled job.

        Raises:
            RuntimeError: If job registration fails
        """
        try:
            job_reg = JobRegistration(
                job_id=job_id,
                func=func,
                trigger=trigger,
                component=component,
                description=description,
                replace_existing=replace_existing
            )

            self._registered_jobs[job_id] = job_reg

            if self._running:
                self.scheduler.add_job(
                    func=func,
                    trigger=trigger,
                    id=job_id,
                    replace_existing=replace_existing
                )
                logger.info(f"Job {job_id} added to running scheduler for component {component}")
            else:
                logger.info(f"Job {job_id} registered for component {component} (will start when scheduler starts)")

        except Exception as e:
            logger.error(f"Error registering job {job_id} for component {component}: {e}")
            raise RuntimeError(f"Failed to register job {job_id} for component {component}: {e}") from e
    
    def start(self) -> None:
        """
        Start the scheduler service.

        Raises:
            RuntimeError: If scheduler fails to start or any job fails to add
        """
        if self._running:
            logger.warning("Scheduler service already running")
            return

        try:
            # Add all jobs BEFORE starting the scheduler
            # This prevents APScheduler from recalculating schedules after each job addition
            for job_id, job_reg in self._registered_jobs.items():
                self.scheduler.add_job(
                    func=job_reg.func,
                    trigger=job_reg.trigger,
                    id=job_id,
                    replace_existing=job_reg.replace_existing
                )
                logger.info(f"Added job {job_id} for component {job_reg.component}")

            self.scheduler.start()
            self._running = True
            logger.info(f"Scheduler service started with {len(self._registered_jobs)} jobs")

        except Exception as e:
            logger.error(f"Error starting scheduler service: {e}")
            self._running = False
            raise RuntimeError(f"Failed to start scheduler service: {e}") from e
    
    def stop(self) -> None:
        if not self._running:
            return

        try:
            logger.info("Stopping scheduler service")

            self.scheduler.shutdown(wait=False)  # AsyncIOExecutor ignores wait=True anyway

            self._running = False
            logger.info("Scheduler service stopped")

        except Exception as e:
            logger.error(f"Error stopping scheduler service: {e}")
            self._running = False
    


scheduler_service = SchedulerService()