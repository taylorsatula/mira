"""
Bridge between LT_Memory system and working memory.

Provides conversation archive access and content formatting
for working memory integration.

RENAMED CLASSES (2025-06-10):
- ConversationArchiveBridge → ConversationTimelineManager
- Fixture: conversation_bridge → conversation_timeline_manager
- Variables: conversation_archive_bridge → conversation_timeline_manager

Files updated: timeline_manager.py (was bridge.py), test_timeline_manager.py (was test_bridge.py), 
integration.py, working_memory.py, __init__.py, test_mira_conflict_resolution.py
If you see any remaining references to the old names, they need to be updated.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, date, timedelta
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


class MemoryBridge:
    """
    Bridge between working memory and LT_Memory system.
    
    Handles the existing working memory integration and core memory updates.
    """
    
    def __init__(self, working_memory, memory_manager):
        """
        Initialize memory bridge.
        
        Args:
            working_memory: WorkingMemory instance
            memory_manager: LT_Memory MemoryManager instance
        """
        self.working_memory = working_memory
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
        
        # Track memory content ID for core memory updates
        self._memory_id: Optional[str] = None
    
    def update_working_memory(self) -> None:
        """
        Update working memory with core memory content.
        
        Called by WorkingMemory.update_all_managers() before each response.
        """
        # Remove existing core memory content
        if self._memory_id:
            self.working_memory.remove(self._memory_id)
        
        try:
            # Get current core memory blocks
            core_blocks = self.memory_manager.block_manager.get_all_blocks()
            
            if core_blocks:
                # Format core memory for system prompt
                memory_content = "# Core Memory\n\n"
                
                for block in core_blocks:
                    memory_content += f"## {block['label'].title()}\n"
                    memory_content += f"{block['value']}\n\n"
                
                # Add to working memory
                self._memory_id = self.working_memory.add(
                    content=memory_content,
                    category="core_memory"
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to update core memory in working memory: {e}")


class ConversationTimelineManager:
    """
    Manages temporal access to conversation history for working memory.
    
    Provides functionality for retrieving, formatting, and injecting
    specific conversations by date or date range into working memory.
    """
    
    def __init__(self, memory_manager):
        """
        Initialize conversation timeline manager.
        
        Args:
            memory_manager: LT_Memory MemoryManager instance
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
        
        # Track currently injected conversations
        self.injected_dates: Set[str] = set()  # ISO date strings
    
    def inject_conversation(self, target_date: date, include_full_messages: bool = False) -> Dict[str, Any]:
        """
        Inject an archived conversation for working memory access.
        
        Args:
            target_date: Date of conversation to inject
            include_full_messages: Whether to include full message content or just summary
            
        Returns:
            Operation results with formatted content
        """
        try:
            # Get archived conversation
            archived_data = self.memory_manager.conversation_archive.get_conversation_by_date(target_date)
            
            if not archived_data:
                return {
                    "success": False,
                    "message": f"No archived conversation found for {target_date}",
                    "content": None
                }
            
            # Add to injected dates
            date_str = target_date.isoformat()
            self.injected_dates.add(date_str)
            
            # Format content for working memory
            content = self._format_single_conversation(archived_data, include_full_messages)
            
            return {
                "success": True,
                "message": f"Prepared conversation from {date_str} for injection",
                "date": date_str,
                "message_count": archived_data["message_count"],
                "content": content,
                "include_full_messages": include_full_messages
            }
            
        except Exception as e:
            self.logger.error(f"Failed to inject conversation for {target_date}: {e}")
            return {
                "success": False,
                "message": f"Error preparing conversation: {str(e)}",
                "content": None
            }
    
    def remove_conversation(self, target_date: date) -> Dict[str, Any]:
        """
        Remove an injected conversation.
        
        Args:
            target_date: Date of conversation to remove
            
        Returns:
            Operation results
        """
        date_str = target_date.isoformat()
        
        if date_str not in self.injected_dates:
            return {
                "success": False,
                "message": f"Conversation from {date_str} is not currently injected"
            }
        
        # Remove from injected dates
        self.injected_dates.remove(date_str)
        
        return {
            "success": True,
            "message": f"Removed conversation from {date_str}",
            "remaining_count": len(self.injected_dates)
        }
    
    def clear_all_injections(self) -> Dict[str, Any]:
        """
        Clear all injected conversations.
        
        Returns:
            Operation results
        """
        count = len(self.injected_dates)
        self.injected_dates.clear()
        
        return {
            "success": True,
            "message": f"Cleared {count} injected conversations",
            "cleared_count": count
        }
    
    def inject_date_range(self, start_date: date, end_date: date, 
                         summary_only: bool = True) -> Dict[str, Any]:
        """
        Inject multiple conversations from a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range  
            summary_only: Whether to include only summaries
            
        Returns:
            Operation results with formatted content
        """
        try:
            # Get conversations in range
            conversations = self.memory_manager.conversation_archive.get_conversations_by_range(
                start_date, end_date
            )
            
            if not conversations:
                return {
                    "success": False,
                    "message": f"No conversations found in range {start_date} to {end_date}",
                    "content": None
                }
            
            # Add all dates to injected set
            for conv in conversations:
                self.injected_dates.add(conv["date"])
            
            # Format content for working memory
            content = self._format_conversation_range(conversations, not summary_only)
            
            return {
                "success": True,
                "message": f"Prepared {len(conversations)} conversations from {start_date} to {end_date}",
                "injected_count": len(conversations),
                "content": content,
                "summary_only": summary_only
            }
            
        except Exception as e:
            self.logger.error(f"Failed to inject date range {start_date} to {end_date}: {e}")
            return {
                "success": False,
                "message": f"Error preparing date range: {str(e)}",
                "content": None
            }
    
    def get_current_injections_content(self, include_full_messages: bool = False) -> Optional[str]:
        """
        Get formatted content for all currently injected conversations.
        
        Args:
            include_full_messages: Whether to include full message content
            
        Returns:
            Formatted content string or None if no injections
        """
        if not self.injected_dates:
            return None
        
        try:
            # Gather all injected conversations
            conversations = []
            for date_str in sorted(self.injected_dates):
                target_date = date.fromisoformat(date_str)
                archived_data = self.memory_manager.conversation_archive.get_conversation_by_date(target_date)
                if archived_data:
                    conversations.append(archived_data)
            
            if not conversations:
                return None
            
            return self._format_conversation_range(conversations, include_full_messages)
            
        except Exception as e:
            self.logger.error(f"Failed to get current injections content: {e}")
            return None
    
    def get_archive_status(self) -> Dict[str, Any]:
        """
        Get archive system status.
        
        Returns:
            Archive status information
        """
        try:
            archive_stats = self.memory_manager.conversation_archive.get_archive_stats()
            
            return {
                "total_conversations": archive_stats["total_archived_conversations"],
                "total_messages": archive_stats["total_archived_messages"],
                "oldest_archive": archive_stats.get("oldest_archive"),
                "newest_archive": archive_stats.get("newest_archive"),
                "injected_dates": sorted(list(self.injected_dates)),
                "injection_count": len(self.injected_dates)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get archive status: {e}")
            return {
                "total_conversations": 0,
                "total_messages": 0,
                "oldest_archive": None,
                "newest_archive": None,
                "injected_dates": [],
                "injection_count": 0
            }
    
    def suggest_relevant_dates(self, query: str = None, days_back: int = 30) -> List[str]:
        """
        Suggest relevant archived conversation dates.
        
        Args:
            query: Optional query to match against summaries
            days_back: How many days back to look
            
        Returns:
            List of relevant date strings
        """
        try:
            # Get recent conversations
            end_date = utc_now().date()
            start_date = end_date - timedelta(days=days_back)
            
            conversations = self.memory_manager.conversation_archive.get_conversations_by_range(
                start_date, end_date
            )
            
            relevant_dates = []
            
            for conv in conversations:
                # If no query, include all
                if not query:
                    relevant_dates.append(conv["date"])
                else:
                    # Simple keyword matching in summary
                    if query.lower() in conv["summary"].lower():
                        relevant_dates.append(conv["date"])
            
            return sorted(relevant_dates, reverse=True)  # Most recent first
            
        except Exception as e:
            self.logger.error(f"Failed to suggest relevant dates: {e}")
            return []
    
    def _format_single_conversation(self, archived_data: Dict[str, Any], 
                                   include_full_messages: bool = False) -> str:
        """Format a single archived conversation for working memory."""
        content = f"## Referenced Conversation: {archived_data['date']}\n\n"
        content += f"**Messages:** {archived_data['message_count']}\n"
        content += f"**Summary:** {archived_data['summary']}\n\n"
        
        if include_full_messages:
            content += "**Full Conversation:**\n"
            for msg in archived_data['messages']:
                role = msg.get('role', 'unknown').title()
                msg_content = str(msg.get('content', ''))
                # Truncate very long messages
                if len(msg_content) > 500:
                    msg_content = msg_content[:497] + "..."
                content += f"- **{role}:** {msg_content}\n"
            content += "\n"
        else:
            content += "*Full conversation messages available on request.*\n\n"
        
        return content
    
    def _format_conversation_range(self, conversations: List[Dict[str, Any]], 
                                  include_full_messages: bool = False) -> str:
        """Format multiple conversations for working memory."""
        content = "# Referenced Archived Conversations\n\n"
        content += f"**Total conversations:** {len(conversations)}\n"
        content += f"**Date range:** {conversations[0]['date']} to {conversations[-1]['date']}\n\n"
        
        for conv in conversations:
            content += f"## {conv['date']} ({conv['message_count']} messages)\n"
            content += f"**Summary:** {conv['summary']}\n\n"
            
            if include_full_messages:
                content += "**Messages:**\n"
                for msg in conv['messages']:
                    role = msg.get('role', 'unknown').title()
                    msg_content = str(msg.get('content', ''))
                    if len(msg_content) > 300:  # Shorter for ranges
                        msg_content = msg_content[:297] + "..."
                    content += f"- **{role}:** {msg_content}\n"
                content += "\n"
            else:
                content += "*Full messages available on request.*\n\n"
        
        return content