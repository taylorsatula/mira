"""Persistent Memory Consolidation Background Service

System-level service that runs memory consolidation for all registered users
independently of active sessions. Uses existing mira_auth database User model.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from db import Database
from utils.timezone_utils import utc_now
from working_memory import WorkingMemory
from lt_memory.managers.memory_manager import MemoryManager
from lt_memory.memory_tasks import MemoryTasks
from conversation import Conversation
from auth.models import User


logger = logging.getLogger(__name__)


@dataclass
class TaskExecution:
    """Simple task execution record for logging."""
    user_id: str
    task_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    attempt_number: int = 1
    duration_seconds: Optional[float] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


class MemoryConsolidationService:
    """Persistent background service for memory consolidation.
    
    Features:
    - Runs independently of user sessions
    - Uses existing mira_auth database User model
    - Skips inactive users automatically
    - Conversation lookup at execution time
    - Minimal resource usage for inactive users
    """
    
    def __init__(self, working_memory: WorkingMemory, memory_manager: MemoryManager, llm_provider: 'LLMProvider', max_retries: int = 2):
        """Initialize memory consolidation service.
        
        Args:
            working_memory: Working memory for alerts
            memory_manager: Memory manager for tasks
            llm_provider: LLM provider for summaries
            max_retries: Maximum retry attempts for failed tasks
        """
        self.scheduler = AsyncIOScheduler()
        self.working_memory = working_memory
        self.memory_manager = memory_manager
        self.llm_provider = llm_provider
        self.max_retries = max_retries
        self.db = Database()
        self._running = False
        
        # Conversation lookup function (set by integration)
        self._conversation_lookup: Optional[Callable[[str], Optional[Conversation]]] = None
        self._activity_checker: Optional[Callable[[str], bool]] = None
    
    def set_conversation_lookup(self, lookup_func: Callable[[str], Optional[Conversation]]):
        """Set the conversation lookup function.
        
        Args:
            lookup_func: Function that takes user_id and returns their active Conversation or None
        """
        self._conversation_lookup = lookup_func
        logger.info("Conversation lookup function registered")
    
    def set_activity_checker(self, activity_func: Callable[[str], bool]):
        """Set the user activity checker function.
        
        Args:
            activity_func: Function that takes user_id and returns True if user has recent activity
        """
        self._activity_checker = activity_func
        logger.info("Activity checker function registered")
    
    def enable_user_consolidation(self, user_id: str):
        """Enable memory consolidation for a user.
        
        Args:
            user_id: User identifier from auth system
        """
        with self.db.get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                user.memory_consolidation_enabled = True
                session.commit()
                logger.info(f"Enabled memory consolidation for user {user_id}")
            else:
                logger.warning(f"User {user_id} not found in auth database")
    
    def disable_user_consolidation(self, user_id: str):
        """Disable memory consolidation for a user.
        
        Args:
            user_id: User identifier from auth system
        """
        with self.db.get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                user.memory_consolidation_enabled = False
                session.commit()
                logger.info(f"Disabled memory consolidation for user {user_id}")
            else:
                logger.warning(f"User {user_id} not found in auth database")
    
    def get_consolidation_enabled_users(self) -> List[str]:
        """Get list of users with memory consolidation enabled.
        
        Returns:
            List of user IDs with consolidation enabled
        """
        with self.db.get_session() as session:
            users = session.query(User).filter_by(
                is_active=True, 
                memory_consolidation_enabled=True
            ).all()
            return [str(user.id) for user in users]
    
    def _should_skip_user(self, user_id: str) -> tuple[bool, str]:
        """Check if user should be skipped for memory consolidation.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (should_skip, reason)
        """
        # Check if user has activity checker
        if self._activity_checker:
            has_activity = self._activity_checker(user_id)
            if not has_activity:
                return True, "No recent activity since last consolidation cycle"
        
        # Check if conversation exists and has content
        if self._conversation_lookup:
            conversation = self._conversation_lookup(user_id)
            if not conversation:
                return True, "No active conversation found"
            
            if not conversation.messages or len(conversation.messages) == 0:
                return True, "No messages in conversation"
        
        return False, ""
    
    async def _execute_memory_task(self, task_name: str, attempt: int = 1):
        """Execute memory consolidation task for all eligible users.
        
        Args:
            task_name: Name of the consolidation task (daily/weekly/monthly)
            attempt: Retry attempt number
        """
        logger.info(f"Starting {task_name} consolidation for all users")
        
        # Get all users with consolidation enabled
        enabled_users = self.get_consolidation_enabled_users()
        if not enabled_users:
            logger.info(f"No users with memory consolidation enabled for {task_name}")
            return
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        for user_id in enabled_users:
            execution = TaskExecution(
                user_id=user_id,
                task_name=task_name,
                started_at=utc_now(),
                attempt_number=attempt
            )
            
            try:
                # Check if user should be skipped
                should_skip, skip_reason = self._should_skip_user(user_id)
                if should_skip:
                    execution.completed_at = utc_now()
                    execution.skipped = True
                    execution.skip_reason = skip_reason
                    execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
                    
                    logger.debug(f"Skipped {task_name} for user {user_id}: {skip_reason}")
                    skipped_count += 1
                    self.db.add(execution)
                    continue
                
                # Get user's conversation
                conversation = self._conversation_lookup(user_id) if self._conversation_lookup else None
                if not conversation:
                    execution.completed_at = utc_now()
                    execution.skipped = True
                    execution.skip_reason = "Could not retrieve user conversation"
                    execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
                    
                    logger.warning(f"Could not get conversation for user {user_id}")
                    skipped_count += 1
                    self.db.add(execution)
                    continue
                
                # Execute memory consolidation task
                memory_tasks = MemoryTasks(self.memory_manager, self.llm_provider, conversation)
                
                if task_name == "daily_consolidation":
                    await memory_tasks.daily_consolidation(user_id)
                elif task_name == "weekly_consolidation":
                    await memory_tasks.weekly_consolidation(user_id)
                elif task_name == "monthly_consolidation":
                    await memory_tasks.monthly_consolidation(user_id)
                elif task_name == "learning_reflection":
                    await memory_tasks.learning_reflection(user_id)
                else:
                    raise ValueError(f"Unknown task: {task_name}")
                
                # Task succeeded
                execution.completed_at = utc_now()
                execution.success = True
                execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
                
                logger.info(f"Completed {task_name} for user {user_id} in {execution.duration_seconds:.2f}s")
                processed_count += 1
                
                # Update last run time in user record
                self._update_user_last_run_time(user_id, task_name)
                
            except Exception as e:
                execution.completed_at = utc_now()
                execution.error_message = str(e)
                execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
                
                logger.error(f"Failed {task_name} for user {user_id}: {e}")
                error_count += 1
                
                # Add alert for failed task
                self.working_memory.add(
                    content=f"# Memory Consolidation Failed\nUser: {user_id}\nTask: {task_name}\nError: {e}",
                    category="task_alerts"
                )
            
            # Log execution
            self.db.add(execution)
        
        logger.info(f"Completed {task_name} consolidation: {processed_count} processed, {skipped_count} skipped, {error_count} errors")
    
    def _update_user_last_run_time(self, user_id: str, task_name: str):
        """Update the last run time for a user's task in auth database.
        
        Args:
            user_id: User identifier
            task_name: Task that was executed
        """
        with self.db.get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                now = utc_now()
                if task_name == "daily_consolidation":
                    user.daily_consolidation_last_run = now
                elif task_name == "weekly_consolidation":
                    user.weekly_consolidation_last_run = now
                elif task_name == "monthly_consolidation":
                    user.monthly_consolidation_last_run = now
                elif task_name == "learning_reflection":
                    user.learning_reflection_last_run = now
                
                user.last_activity_check = now
                session.commit()
    
    def start(self):
        """Start the memory consolidation service."""
        if self._running:
            logger.warning("Memory consolidation service already running")
            return
        
        # Schedule consolidation tasks for all users
        # Daily consolidation at 3 AM
        self.scheduler.add_job(
            func=self._execute_memory_task,
            trigger=CronTrigger.from_crontab("0 3 * * *"),
            args=["daily_consolidation"],
            id="daily_consolidation_all_users",
            replace_existing=True
        )
        
        # Weekly consolidation at 4 AM on Sunday
        self.scheduler.add_job(
            func=self._execute_memory_task,
            trigger=CronTrigger.from_crontab("0 4 * * 0"),
            args=["weekly_consolidation"],
            id="weekly_consolidation_all_users",
            replace_existing=True
        )
        
        # Monthly consolidation at 5 AM on 1st of month
        self.scheduler.add_job(
            func=self._execute_memory_task,
            trigger=CronTrigger.from_crontab("0 5 1 * *"),
            args=["monthly_consolidation"],
            id="monthly_consolidation_all_users",
            replace_existing=True
        )
        
        # Learning reflection at 11 PM on Sunday (after weekly consolidation)
        self.scheduler.add_job(
            func=self._execute_memory_task,
            trigger=CronTrigger.from_crontab("0 23 * * 0"),
            args=["learning_reflection"],
            id="learning_reflection_all_users",
            replace_existing=True
        )
        
        self.scheduler.start()
        self._running = True
        
        logger.info("Memory consolidation service started")
        logger.info(f"Users with consolidation enabled: {len(self.get_consolidation_enabled_users())}")
    
    def stop(self):
        """Stop the memory consolidation service."""
        if not self._running:
            return
            
        self.scheduler.shutdown()
        self._running = False
        logger.info("Memory consolidation service stopped")
    
    def get_user_stats(self, user_id: str) -> Optional[Dict]:
        """Get consolidation statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with user's consolidation history or None if not found
        """
        with self.db.get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                return None
            
            return {
                "user_id": str(user.id),
                "email": user.email,
                "memory_consolidation_enabled": user.memory_consolidation_enabled,
                "daily_consolidation_last_run": user.daily_consolidation_last_run.isoformat() if user.daily_consolidation_last_run else None,
                "weekly_consolidation_last_run": user.weekly_consolidation_last_run.isoformat() if user.weekly_consolidation_last_run else None,
                "monthly_consolidation_last_run": user.monthly_consolidation_last_run.isoformat() if user.monthly_consolidation_last_run else None,
                "last_activity_check": user.last_activity_check.isoformat() if user.last_activity_check else None
            }
    
    def get_service_stats(self) -> Dict:
        """Get overall service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        enabled_users = self.get_consolidation_enabled_users()
        return {
            "running": self._running,
            "total_enabled_users": len(enabled_users),
            "scheduled_jobs": len(self.scheduler.get_jobs()) if self._running else 0,
            "consolidation_schedule": {
                "daily": "3:00 AM UTC daily",
                "weekly": "4:00 AM UTC Sunday", 
                "monthly": "5:00 AM UTC 1st of month",
                "learning_reflection": "11:00 PM UTC Sunday"
            }
        }