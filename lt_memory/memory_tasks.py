"""Memory consolidation tasks using existing memory managers.

Simple functions that directly call the well-implemented memory managers,
avoiding the complex abstractions of the task_manager.
"""

import logging
from datetime import timedelta
from typing import List

from utils.timezone_utils import utc_now
from lt_memory.managers.memory_manager import MemoryManager
from lt_memory.managers.block_manager import BlockManager
from lt_memory.managers.conversation_archive import ConversationArchive
from api.llm_provider import LLMProvider
from conversation import Conversation


logger = logging.getLogger(__name__)


class MemoryTasks:
    """Memory consolidation tasks using existing memory managers."""
    
    def __init__(self, memory_manager: MemoryManager, llm_provider: LLMProvider, conversation: Conversation):
        """Initialize with memory manager.
        
        Args:
            memory_manager: Memory manager instance
            llm_provider: LLM provider for weekly/monthly summaries only
            conversation: Active conversation for archiving
        """
        self.memory_manager = memory_manager
        self.block_manager = BlockManager(memory_manager)
        self.archive = ConversationArchive(memory_manager)
        self.llm_provider = llm_provider
        self.conversation = conversation
    
    async def daily_consolidation(self, user_id: str):
        """Simple daily memory consolidation.
        
        Uses existing memory managers to:
        1. Archive yesterday's conversations (automatically creates LLM summary)
        2. Update yesterday_summary block with the archive summary
        
        Args:
            user_id: User identifier
        """
        logger.info(f"Starting daily consolidation for user {user_id}")
        
        # Get yesterday's date
        yesterday = (utc_now() - timedelta(days=1)).date()
        
        # Archive yesterday's conversations - this already uses LLM to create summary
        archive_result = self.archive.archive_day(self.conversation, yesterday)
        
        if archive_result["success"] and archive_result["message_count"] > 0:
            # The archive already generated an LLM summary - just use it directly
            daily_summary = archive_result.get("summary", f"Archived {archive_result['message_count']} messages")
            
            # Update yesterday_summary block using existing method
            self.block_manager.memory_rethink(
                label="yesterday_summary",
                new_content=daily_summary,
                actor=f"daily_consolidation_{user_id}"
            )
            
            logger.info(f"Daily consolidation completed for {user_id}: {archive_result['message_count']} messages archived")
        else:
            logger.info(f"Daily consolidation for {user_id}: no messages to archive")
    
    async def weekly_consolidation(self, user_id: str):
        """Simple weekly consolidation.
        
        Uses existing memory managers to:
        1. Get last 7 versions of yesterday_summary using rollback
        2. Create weekly summary using LLM
        3. Update lastweek_summary block
        
        Args:
            user_id: User identifier
        """
        logger.info(f"Starting weekly consolidation for user {user_id}")
        
        # Get current version to calculate rollback targets
        yesterday_block = self.block_manager.get_block("yesterday_summary")
        if not yesterday_block:
            logger.warning(f"No yesterday_summary block found for user {user_id}")
            return
        
        current_version = yesterday_block["version"]
        
        # Collect last 7 daily summaries using existing rollback method
        daily_summaries = []
        for days_back in range(7):
            target_version = current_version - days_back
            if target_version < 1:
                break
                
            try:
                # Use existing rollback_block method
                rolled_back = self.block_manager.rollback_block(
                    label="yesterday_summary",
                    version=target_version,
                    actor=f"weekly_consolidation_{user_id}"
                )
                daily_summaries.append(rolled_back["value"])
            except Exception as e:
                logger.warning(f"Could not rollback to version {target_version}: {e}")
        
        if not daily_summaries:
            logger.info(f"No daily summaries found for weekly consolidation")
            return
        
        # Create weekly summary using LLM
        weekly_summary = await self._create_weekly_summary(daily_summaries)
        
        # Update lastweek_summary block using existing method
        self.block_manager.memory_rethink(
            label="lastweek_summary", 
            new_content=weekly_summary,
            actor=f"weekly_consolidation_{user_id}"
        )
        
        logger.info(f"Weekly consolidation completed for {user_id}: merged {len(daily_summaries)} daily summaries")
    
    async def monthly_consolidation(self, user_id: str):
        """Simple monthly consolidation.
        
        Uses existing memory managers to:
        1. Get last 4 versions of lastweek_summary using rollback
        2. Create monthly summary using LLM
        3. Update lastmonth_summary block
        
        Args:
            user_id: User identifier
        """
        logger.info(f"Starting monthly consolidation for user {user_id}")
        
        # Get current version of weekly summaries
        weekly_block = self.block_manager.get_block("lastweek_summary")
        if not weekly_block:
            logger.warning(f"No lastweek_summary block found for user {user_id}")
            return
        
        current_version = weekly_block["version"]
        
        # Collect last 4 weekly summaries using existing rollback method
        weekly_summaries = []
        for weeks_back in range(4):
            target_version = current_version - weeks_back
            if target_version < 1:
                break
                
            try:
                # Use existing rollback_block method
                rolled_back = self.block_manager.rollback_block(
                    label="lastweek_summary",
                    version=target_version,
                    actor=f"monthly_consolidation_{user_id}"
                )
                weekly_summaries.append(rolled_back["value"])
            except Exception as e:
                logger.warning(f"Could not rollback to version {target_version}: {e}")
        
        if not weekly_summaries:
            logger.info(f"No weekly summaries found for monthly consolidation")
            return
        
        # Create monthly summary using LLM
        monthly_summary = await self._create_monthly_summary(weekly_summaries)
        
        # Update lastmonth_summary block using existing method
        self.block_manager.memory_rethink(
            label="lastmonth_summary",
            new_content=monthly_summary,
            actor=f"monthly_consolidation_{user_id}"
        )
        
        logger.info(f"Monthly consolidation completed for {user_id}: merged {len(weekly_summaries)} weekly summaries")
    
    async def learning_reflection(self, user_id: str):
        """Weekly learning reflection and pattern analysis.
        
        Analyzes conversation patterns to identify areas for improvement
        and updates learning insights in memory blocks.
        
        Args:
            user_id: User identifier
        """
        logger.info(f"Starting learning reflection for user {user_id}")
        
        # Get last week's conversations for pattern analysis
        weekly_conversations = await self._get_recent_conversations(user_id, days=7)
        
        if not weekly_conversations or len(weekly_conversations) < 10:
            logger.info(f"Insufficient conversation data for learning reflection (need 10+ messages, got {len(weekly_conversations) if weekly_conversations else 0})")
            return
        
        # Analyze patterns using LLM
        learning_insights = await self._analyze_learning_patterns(weekly_conversations)
        
        # Update learning insights block
        try:
            self.block_manager.memory_rethink(
                label="learning_insights",
                new_content=learning_insights,
                actor=f"learning_reflection_{user_id}"
            )
            logger.info(f"Learning reflection completed for user {user_id}")
        except Exception as e:
            # If block doesn't exist, create it
            if "not found" in str(e).lower():
                self.block_manager.create_block(
                    label="learning_insights",
                    content=learning_insights,
                    character_limit=4096,
                    actor=f"learning_reflection_{user_id}"
                )
                logger.info(f"Created learning_insights block for user {user_id}")
            else:
                raise
    
    async def _get_recent_conversations(self, user_id: str, days: int = 7) -> List[Dict]:
        """Get recent conversation messages for analysis.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            
        Returns:
            List of recent conversation messages
        """
        # Get messages from the current conversation
        if hasattr(self.conversation, 'messages') and self.conversation.messages:
            # Filter messages from last N days
            cutoff_date = utc_now() - timedelta(days=days)
            recent_messages = []
            
            for message in self.conversation.messages:
                if message.created_at >= cutoff_date:
                    recent_messages.append({
                        "role": message.role,
                        "content": message.content,
                        "created_at": message.created_at.isoformat(),
                        "metadata": message.metadata
                    })
            
            return recent_messages
        
        return []
    
    async def _analyze_learning_patterns(self, conversations: List[Dict]) -> str:
        """Analyze conversation patterns for learning insights.
        
        Args:
            conversations: List of conversation messages
            
        Returns:
            Learning insights and improvement recommendations
        """
        # Prepare conversation data for analysis
        conversation_text = "\n\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversations[-50:]  # Last 50 messages for analysis
        ])
        
        prompt = f"""Analyze the following conversation patterns from the past week and provide learning insights:

{conversation_text}

Please analyze:
1. **Communication Patterns**: How does the user prefer to communicate? (formal/casual, brief/detailed, etc.)
2. **Task Patterns**: What types of tasks does the user frequently need help with?
3. **Knowledge Gaps**: What areas could I improve to be more helpful?
4. **User Preferences**: What response styles or approaches work best?
5. **Recurring Themes**: What topics or challenges come up repeatedly?

Provide actionable insights in this format:

## Communication Style Observations
[Insights about how user communicates and prefers responses]

## Recurring Tasks & Needs
[Common tasks and how to better support them]

## Areas for Improvement
[Specific ways I could be more helpful]

## User Preferences Identified
[Preferences for response style, detail level, etc.]

Keep insights practical and focused on improving future interactions."""

        try:
            response = self.llm_provider.generate_response(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            return self.llm_provider.extract_text_content(response)
            
        except Exception as e:
            logger.error(f"Failed to analyze learning patterns: {e}")
            return f"Learning reflection completed with {len(conversations)} messages analyzed. Pattern analysis temporarily unavailable."
    
    async def _create_weekly_summary(self, daily_summaries: List[str]) -> str:
        """Create weekly summary using LLM.
        
        Args:
            daily_summaries: List of daily summaries from the week
            
        Returns:
            Weekly summary
        """
        summaries_text = "\n\n".join([f"Day {i+1}: {summary}" for i, summary in enumerate(daily_summaries)])
        
        prompt = f"""Create a weekly summary by analyzing the following daily summaries:

{summaries_text}

Please create a comprehensive weekly summary that includes:
1. Major themes and recurring topics
2. Progress on ongoing projects or discussions
3. Key decisions and outcomes
4. Patterns or trends observed
5. Important follow-ups for next week

Focus on synthesis rather than repetition. Highlight what happened over the week as a whole."""

        try:
            response = self.llm_provider.generate_response(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            return self.llm_provider.extract_text_content(response)
            
        except Exception as e:
            logger.error(f"Failed to create weekly summary with LLM: {e}")
            return f"Weekly summary based on {len(daily_summaries)} daily summaries. " + \
                   "Key themes: " + " | ".join(daily_summaries[:2])
    
    async def _create_monthly_summary(self, weekly_summaries: List[str]) -> str:
        """Create monthly summary using LLM.
        
        Args:
            weekly_summaries: List of weekly summaries from the month
            
        Returns:
            Monthly summary
        """
        summaries_text = "\n\n".join([f"Week {i+1}: {summary}" for i, summary in enumerate(weekly_summaries)])
        
        prompt = f"""Create a monthly summary by analyzing the following weekly summaries:

{summaries_text}

Please create a high-level monthly summary that captures:
1. Primary topics and areas of focus this month
2. Major accomplishments and milestones
3. Recurring patterns and themes
4. Evolution of discussions and projects over time
5. Key learnings and insights
6. Areas of consistent attention or concern

Focus on the big picture - what characterized this month as a whole? What were the dominant themes and how did things evolve?"""

        try:
            response = self.llm_provider.generate_response(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            return self.llm_provider.extract_text_content(response)
            
        except Exception as e:
            logger.error(f"Failed to create monthly summary with LLM: {e}")
            return f"Monthly summary based on {len(weekly_summaries)} weekly summaries. " + \
                   "Primary themes: " + " | ".join(weekly_summaries[:2])