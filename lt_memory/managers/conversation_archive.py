"""
Conversation archive manager for temporal storage and retrieval.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta
from pathlib import Path

from sqlalchemy import and_, text, func

from lt_memory.models.base import ArchivedConversation
from errors import error_context, ErrorCode, ToolError
from utils.timezone_utils import utc_now, ensure_utc, format_utc_iso

logger = logging.getLogger(__name__)


class ConversationArchive:
    """
    Manages temporal storage and retrieval of conversation history.
    
    Handles archiving daily conversations to PostgreSQL and retrieving
    them for web interface linking and progressive summarization.
    """
    
    def __init__(self, memory_manager):
        """
        Initialize conversation archive.
        
        Args:
            memory_manager: Parent MemoryManager instance
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
    
    def archive_day(self, conversation, target_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Archive a day's conversation history from live conversation to database.
        
        Args:
            conversation: Live Conversation instance
            target_date: Date to archive (defaults to yesterday)
            
        Returns:
            Archive operation results
            
        Raises:
            ToolError: If archiving fails
        """
        with error_context("conversation_archive", "archive_day", ToolError, ErrorCode.MEMORY_ERROR):
            # Default to yesterday if no date specified
            if target_date is None:
                target_date = (utc_now() - timedelta(days=1)).date()
            
            # Get messages from live conversation
            if not hasattr(conversation, 'messages') or not conversation.messages:
                return {
                    "success": False,
                    "message": "No messages found in conversation",
                    "date": target_date.isoformat(),
                    "message_count": 0
                }
            
            # Filter messages for the target date
            day_messages = self._filter_messages_by_date(conversation.messages, target_date)
            
            if not day_messages:
                return {
                    "success": False,
                    "message": f"No messages found for {target_date}",
                    "date": target_date.isoformat(),
                    "message_count": 0
                }
            
            # Convert Message objects to dict format for storage
            day_messages_dict = self._convert_messages_to_dict(day_messages)
            
            # Generate daily summary
            summary = self._generate_daily_summary(day_messages_dict)
            
            # Archive to database with summary
            archive_id = self._store_conversation(target_date, day_messages_dict, summary)
            
            # Remove archived messages from live conversation
            original_count = len(conversation.messages)
            conversation.messages = self._remove_archived_messages_from_live(conversation.messages, target_date)
            
            self.logger.info(
                f"Archived {len(day_messages)} messages for {target_date} "
                f"(ID: {archive_id}). Removed from live conversation."
            )
            
            return {
                "success": True,
                "message": f"Archived conversation for {target_date}",
                "date": target_date.isoformat(),
                "message_count": len(day_messages),
                "archive_id": archive_id,
                "original_messages": original_count,
                "remaining_messages": len(conversation.messages)
            }
    
    def get_conversation_by_date(self, target_date: date) -> Optional[Dict[str, Any]]:
        """
        Retrieve conversation for a specific date.
        
        Args:
            target_date: Date to retrieve
            
        Returns:
            Conversation data or None if not found
        """
        with self.memory_manager.get_session() as session:
            # Query for conversations on the target date
            start_of_day = ensure_utc(datetime.combine(target_date, datetime.min.time()))
            end_of_day = ensure_utc(datetime.combine(target_date, datetime.max.time()))
            
            archived_conv = session.query(ArchivedConversation).filter(
                and_(
                    ArchivedConversation.conversation_date >= start_of_day,
                    ArchivedConversation.conversation_date <= end_of_day
                )
            ).first()
            
            if archived_conv:
                return {
                    "id": str(archived_conv.id),
                    "date": ensure_utc(archived_conv.conversation_date).date().isoformat(),
                    "messages": archived_conv.messages,
                    "message_count": archived_conv.message_count,
                    "summary": archived_conv.summary,
                    "weekly_summary": archived_conv.weekly_summary,
                    "monthly_summary": archived_conv.monthly_summary,
                    "archived_at": archived_conv.archived_at.isoformat(),
                    "metadata": archived_conv.conversation_metadata
                }
            
            return None
    
    def get_conversations_by_range(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """
        Retrieve conversations within a date range.
        
        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            
        Returns:
            List of conversation data
        """
        with self.memory_manager.get_session() as session:
            start_datetime = ensure_utc(datetime.combine(start_date, datetime.min.time()))
            end_datetime = ensure_utc(datetime.combine(end_date, datetime.max.time()))
            
            archived_convs = session.query(ArchivedConversation).filter(
                and_(
                    ArchivedConversation.conversation_date >= start_datetime,
                    ArchivedConversation.conversation_date <= end_datetime
                )
            ).order_by(ArchivedConversation.conversation_date.asc()).all()
            
            conversations = []
            for conv in archived_convs:
                conversations.append({
                    "id": str(conv.id),
                    "date": ensure_utc(conv.conversation_date).date().isoformat(),
                    "messages": conv.messages,
                    "message_count": conv.message_count,
                    "summary": conv.summary,
                    "weekly_summary": conv.weekly_summary,
                    "monthly_summary": conv.monthly_summary,
                    "archived_at": conv.archived_at.isoformat(),
                    "metadata": conv.conversation_metadata
                })
            
            return conversations
    
    def get_week_conversations(self, target_date: date) -> List[Dict[str, Any]]:
        """
        Get all conversations for the week containing the target date.
        
        Args:
            target_date: Any date within the target week
            
        Returns:
            List of conversation data for the week
        """
        # Find start of week (Monday)
        days_since_monday = target_date.weekday()
        week_start = target_date - timedelta(days=days_since_monday)
        week_end = week_start + timedelta(days=6)
        
        return self.get_conversations_by_range(week_start, week_end)
    
    def get_month_conversations(self, target_date: date) -> List[Dict[str, Any]]:
        """
        Get all conversations for the month containing the target date.
        
        Args:
            target_date: Any date within the target month
            
        Returns:
            List of conversation data for the month
        """
        # First and last day of the month
        month_start = target_date.replace(day=1)
        if target_date.month == 12:
            # Handle December -> January transition
            month_end = target_date.replace(year=target_date.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            # Normal month transition
            month_end = target_date.replace(month=target_date.month + 1, day=1) - timedelta(days=1)
        
        return self.get_conversations_by_range(month_start, month_end)
    
    def _filter_messages_by_date(self, messages, target_date: date):
        """Filter messages to only include those from the target date."""
        day_messages = []
        
        for msg in messages:
            # Handle both Message objects and dict format
            if hasattr(msg, 'created_at'):
                # Message object
                msg_date = msg.created_at
            elif isinstance(msg, dict):
                # Dict format
                msg_date = self._extract_message_date(msg)
            else:
                continue
            
            if msg_date and msg_date.date() == target_date:
                day_messages.append(msg)
        
        return day_messages
    
    def _extract_message_date(self, message: Dict) -> Optional[datetime]:
        """Extract datetime from message."""
        # Try different timestamp fields
        timestamp_fields = ["created_at", "timestamp", "time", "date"]
        
        for field in timestamp_fields:
            if field in message:
                try:
                    timestamp = message[field]
                    if isinstance(timestamp, str):
                        # Parse ISO format datetime
                        return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    elif isinstance(timestamp, (int, float)):
                        # Unix timestamp
                        return ensure_utc(datetime.fromtimestamp(timestamp))
                except:
                    continue
        
        return None
    
    def _convert_messages_to_dict(self, messages) -> List[Dict]:
        """Convert Message objects to dictionary format for storage."""
        dict_messages = []
        
        for msg in messages:
            if hasattr(msg, '__dict__'):
                # Message object - convert to dict
                msg_dict = {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": format_utc_iso(msg.created_at),
                    "metadata": msg.metadata
                }
                dict_messages.append(msg_dict)
            elif isinstance(msg, dict):
                # Already dict format
                dict_messages.append(msg)
        
        return dict_messages
    
    def _generate_daily_summary(self, messages: List[Dict]) -> str:
        """Generate summary for a day's conversation."""
        try:
            summary = self.memory_manager.summarization_engine.summarize(
                messages=messages,
                scope="daily",
                compression_level="detailed"
            )
            return summary
        except Exception as e:
            self.logger.warning(f"Failed to generate summary: {e}")
            # Fallback to simple summary
            return f"Conversation with {len(messages)} messages on this day."
    
    def _store_conversation(self, conversation_date: date, messages: List[Dict], summary: str) -> str:
        """Store conversation in database with summary."""
        with self.memory_manager.get_session() as session:
            # Create archive entry
            archived_conv = ArchivedConversation(
                conversation_date=ensure_utc(datetime.combine(conversation_date, datetime.min.time())),
                messages=messages,
                message_count=len(messages),
                summary=summary,
                conversation_metadata={
                    "archive_source": "daily_archival",
                    "first_message_time": messages[0]["created_at"] if messages else None,
                    "last_message_time": messages[-1]["created_at"] if messages else None
                }
            )
            
            session.add(archived_conv)
            session.commit()
            
            return str(archived_conv.id)
    
    def _remove_archived_messages_from_live(self, messages, archived_date: date):
        """Remove messages from the specified date from live conversation."""
        remaining_messages = []
        
        for msg in messages:
            # Handle Message objects
            if hasattr(msg, 'created_at'):
                msg_date = msg.created_at
            elif isinstance(msg, dict):
                msg_date = self._extract_message_date(msg)
            else:
                continue
            
            if not msg_date or msg_date.date() != archived_date:
                remaining_messages.append(msg)
        
        return remaining_messages
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """Get archive statistics."""
        with self.memory_manager.get_session() as session:
            stats = {
                "total_archived_conversations": session.query(ArchivedConversation).count(),
                "total_archived_messages": session.query(
                    func.sum(ArchivedConversation.message_count)
                ).scalar() or 0,
                "oldest_archive": None,
                "newest_archive": None
            }
            
            # Get date range
            oldest = session.query(ArchivedConversation).order_by(
                ArchivedConversation.conversation_date.asc()
            ).first()
            
            newest = session.query(ArchivedConversation).order_by(
                ArchivedConversation.conversation_date.desc()
            ).first()
            
            if oldest:
                stats["oldest_archive"] = ensure_utc(oldest.conversation_date).date().isoformat()
            if newest:
                stats["newest_archive"] = ensure_utc(newest.conversation_date).date().isoformat()
            
            return stats
    
    def get_weekly_summary(self, target_date: Optional[date] = None) -> Optional[str]:
        """
        Get weekly summary for a specific date.
        
        Args:
            target_date: Date to get weekly summary for (defaults to yesterday)
            
        Returns:
            Weekly summary text or None if not found
        """
        if target_date is None:
            target_date = (utc_now() - timedelta(days=1)).date()
        
        # Calculate the Sunday of the week containing target_date
        days_since_monday = target_date.weekday()
        week_start = target_date - timedelta(days=days_since_monday)  # Monday of the week
        week_sunday = week_start + timedelta(days=6)  # Sunday of the week
        
        # Look for weekly summary on Sunday of the week
        sunday_data = self.get_conversation_by_date(week_sunday)
        if sunday_data and sunday_data.get("weekly_summary"):
            return sunday_data["weekly_summary"]
        
        return None
    
    def get_monthly_summary(self, target_month: date) -> Optional[str]:
        """
        Get monthly summary for a specific month (always references 1st of month).
        
        Args:
            target_month: Any date within the target month
            
        Returns:
            Monthly summary text or None if not found
        """
        # Always reference the 1st of the month
        first_of_month = target_month.replace(day=1)
        
        archived_data = self.get_conversation_by_date(first_of_month)
        if archived_data and archived_data.get("monthly_summary"):
            return archived_data["monthly_summary"]
        
        return None
    
    def get_recent_context_summaries(self, target_date: Optional[date] = None) -> Dict[str, Optional[str]]:
        """
        Get recent context summaries for working memory.
        
        Args:
            target_date: Date to generate context for (defaults to today)
            
        Returns:
            Dictionary with yesterday and weekly summary information
        """
        if target_date is None:
            # Use consistent UTC timezone handling throughout application
            target_date = utc_now().date()
        
        context = {
            "yesterday": None,
            "weekly": None
        }
        
        # Get yesterday's daily summary
        yesterday = target_date - timedelta(days=1)
        yesterday_data = self.get_conversation_by_date(yesterday)
        if yesterday_data:
            context["yesterday"] = {
                "date": yesterday.isoformat(),
                "summary": yesterday_data["summary"]
            }
        
        # Get weekly summary (from yesterday's entry)
        weekly_summary = self.get_weekly_summary(yesterday)
        if weekly_summary:
            week_start = yesterday - timedelta(days=7)
            week_end = yesterday - timedelta(days=1)
            context["weekly"] = {
                "date_range": f"{week_start.isoformat()} to {week_end.isoformat()}",
                "summary": weekly_summary
            }
        
        return context
    
    def generate_weekly_summary(self, target_date: Optional[date] = None) -> Optional[str]:
        """
        Generate and store weekly summary for a given date.
        
        Creates a summary of the week containing the target date,
        using all daily summaries from that week.
        
        Args:
            target_date: Any date within the target week (defaults to yesterday)
            
        Returns:
            Generated weekly summary text or None if no data
        """
        if target_date is None:
            target_date = (utc_now() - timedelta(days=1)).date()
        
        # Get conversations for the entire week
        week_conversations = self.get_week_conversations(target_date)
        
        if not week_conversations:
            self.logger.info(f"No conversations found for week containing {target_date}")
            return None
        
        # Extract daily summaries for the week
        daily_summaries = []
        for conv in week_conversations:
            if conv.get("summary"):
                date_str = conv["date"]
                summary = conv["summary"]
                daily_summaries.append(f"{date_str}: {summary}")
        
        if not daily_summaries:
            self.logger.info(f"No daily summaries found for week containing {target_date}")
            return None
        
        # Generate weekly summary using summarization engine
        try:
            # Create fake messages structure for summarization engine
            summary_messages = [
                {
                    "role": "user",
                    "content": f"Daily summaries for the week:\n\n" + "\n\n".join(daily_summaries)
                }
            ]
            
            weekly_summary = self.memory_manager.summarization_engine.summarize(
                messages=summary_messages,
                scope="weekly",
                compression_level="detailed"
            )
            
            # Store the weekly summary in the database (on the target date's entry)
            self._store_weekly_summary(target_date, weekly_summary)
            
            self.logger.info(f"Generated weekly summary for week containing {target_date}")
            return weekly_summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate weekly summary: {e}")
            return None
    
    def generate_monthly_summary(self, target_month: date) -> Optional[str]:
        """
        Generate and store monthly summary for a given month.
        
        Creates a summary of the entire month using all daily summaries,
        with weekly summaries if available.
        
        Args:
            target_month: Any date within the target month
            
        Returns:
            Generated monthly summary text or None if no data
        """
        # Get conversations for the entire month
        month_conversations = self.get_month_conversations(target_month)
        
        if not month_conversations:
            self.logger.info(f"No conversations found for month {target_month.strftime('%Y-%m')}")
            return None
        
        # Extract daily and weekly summaries
        daily_summaries = []
        weekly_summaries = set()  # Use set to avoid duplicates
        
        for conv in month_conversations:
            if conv.get("summary"):
                date_str = conv["date"]
                daily_summary = conv["summary"]
                daily_summaries.append(f"{date_str}: {daily_summary}")
            
            # Collect unique weekly summaries
            if conv.get("weekly_summary"):
                weekly_summaries.add(conv["weekly_summary"])
        
        if not daily_summaries:
            self.logger.info(f"No summaries found for month {target_month.strftime('%Y-%m')}")
            return None
        
        # Generate monthly summary using summarization engine
        try:
            # Create comprehensive content for monthly summary
            content_parts = [f"Daily summaries for {target_month.strftime('%B %Y')}:"]
            content_parts.extend(daily_summaries)
            
            if weekly_summaries:
                content_parts.append("\nWeekly summaries:")
                content_parts.extend(weekly_summaries)
            
            summary_messages = [
                {
                    "role": "user", 
                    "content": "\n\n".join(content_parts)
                }
            ]
            
            monthly_summary = self.memory_manager.summarization_engine.summarize(
                messages=summary_messages,
                scope="monthly",
                compression_level="detailed"
            )
            
            # Store the monthly summary in the database (on the 1st of the month)
            first_of_month = target_month.replace(day=1)
            self._store_monthly_summary(first_of_month, monthly_summary)
            
            self.logger.info(f"Generated monthly summary for {target_month.strftime('%Y-%m')}")
            return monthly_summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate monthly summary: {e}")
            return None
    
    def _store_weekly_summary(self, target_date: date, weekly_summary: str) -> None:
        """Store weekly summary on the Sunday of the week containing target_date."""
        # Calculate the Sunday of the week containing target_date
        days_since_monday = target_date.weekday()
        week_start = target_date - timedelta(days=days_since_monday)  # Monday of the week
        week_sunday = week_start + timedelta(days=6)  # Sunday of the week
        
        with self.memory_manager.get_session() as session:
            # Find the archived conversation for Sunday of the week
            start_of_day = ensure_utc(datetime.combine(week_sunday, datetime.min.time()))
            end_of_day = ensure_utc(datetime.combine(week_sunday, datetime.max.time()))
            
            archived_conv = session.query(ArchivedConversation).filter(
                and_(
                    ArchivedConversation.conversation_date >= start_of_day,
                    ArchivedConversation.conversation_date <= end_of_day
                )
            ).first()
            
            if archived_conv:
                archived_conv.weekly_summary = weekly_summary
                session.commit()
                self.logger.debug(f"Stored weekly summary for week ending {week_sunday}")
            else:
                self.logger.warning(f"No archived conversation found for Sunday {week_sunday} to store weekly summary")
    
    def _store_monthly_summary(self, first_of_month: date, monthly_summary: str) -> None:
        """Store monthly summary in the first day of month's archived conversation."""
        with self.memory_manager.get_session() as session:
            # Find the archived conversation for first of month
            start_of_day = ensure_utc(datetime.combine(first_of_month, datetime.min.time()))
            end_of_day = ensure_utc(datetime.combine(first_of_month, datetime.max.time()))
            
            archived_conv = session.query(ArchivedConversation).filter(
                and_(
                    ArchivedConversation.conversation_date >= start_of_day,
                    ArchivedConversation.conversation_date <= end_of_day
                )
            ).first()
            
            if archived_conv:
                archived_conv.monthly_summary = monthly_summary
                session.commit()
                self.logger.debug(f"Stored monthly summary for {first_of_month}")
            else:
                self.logger.warning(f"No archived conversation found for {first_of_month} to store monthly summary")