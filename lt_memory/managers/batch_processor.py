"""
Batch processor for scheduled conversation processing.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, UTC, timedelta
from pathlib import Path

from errors import error_context, ErrorCode, ToolError

logger = logging.getLogger(__name__)


class BatchConversationProcessor:
    """
    Process conversations in batches for memory extraction.
    
    Handles scheduled processing of conversations to extract
    memories, entities, and relationships.
    """
    
    def __init__(self, memory_manager):
        """
        Initialize batch processor.
        
        Args:
            memory_manager: Parent MemoryManager instance
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
        
        # Track processed conversations
        self.processed_file = Path(
            self.memory_manager.config.paths.data_dir
        ) / "lt_memory" / "processed_conversations.json"
        self.processed_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._load_processed_list()
    
    def _load_processed_list(self):
        """Load list of already processed conversations."""
        if self.processed_file.exists():
            try:
                with open(self.processed_file, 'r') as f:
                    self.processed_conversations = set(json.load(f))
            except Exception as e:
                self.logger.warning(f"Failed to load processed list: {e}")
                self.processed_conversations = set()
        else:
            self.processed_conversations = set()
    
    def _save_processed_list(self):
        """Save list of processed conversations."""
        try:
            with open(self.processed_file, 'w') as f:
                json.dump(list(self.processed_conversations), f)
        except Exception as e:
            self.logger.error(f"Failed to save processed list: {e}")
    
    def process_recent_conversations(self, hours: int = 1) -> Dict[str, int]:
        """
        Process conversations from the last N hours.
        
        Args:
            hours: How many hours back to process
            
        Returns:
            Processing statistics
        """
        results = {
            "conversations_processed": 0,
            "passages_created": 0,
            "entities_extracted": 0,
            "relationships_inferred": 0,
            "errors": 0
        }
        
        with error_context("batch_processor", "process_conversations", ToolError, ErrorCode.MEMORY_ERROR):
            # Get unprocessed conversations
            cutoff = datetime.now(UTC) - timedelta(hours=hours)
            conversations = self._get_unprocessed_conversations(since=cutoff)
            
            self.logger.info(
                f"Found {len(conversations)} conversations to process"
            )
            
            for conv_id, messages in conversations.items():
                try:
                    # Process this conversation
                    conv_results = self._process_conversation(conv_id, messages)
                    
                    # Update results
                    results["passages_created"] += conv_results["passages"]
                    results["entities_extracted"] += conv_results["entities"]
                    results["relationships_inferred"] += conv_results["relationships"]
                    results["conversations_processed"] += 1
                    
                    # Mark as processed
                    self.processed_conversations.add(conv_id)
                    
                except Exception as e:
                    self.logger.error(
                        f"Error processing conversation {conv_id}: {e}"
                    )
                    results["errors"] += 1
            
            # Save updated processed list
            self._save_processed_list()
            
            self.logger.info(f"Batch processing complete: {results}")
            
        return results
    
    def process_all_pending(self) -> Dict[str, int]:
        """
        Process all pending conversations regardless of age.
        
        Returns:
            Processing statistics
        """
        return self.process_recent_conversations(hours=24*365)  # Process up to a year
    
    def _get_unprocessed_conversations(self, since: datetime) -> Dict[str, List[Dict]]:
        """
        Get conversations that haven't been processed yet.
        
        Args:
            since: Cutoff datetime
            
        Returns:
            Dict of conversation_id -> messages
        """
        conversations = {}
        
        # Get conversation history directory
        history_dir = Path(self.memory_manager.config.paths.conversation_history_dir)
        
        if not history_dir.exists():
            return conversations
        
        # Scan for conversation files
        for conv_file in history_dir.glob("*.json"):
            conv_id = conv_file.stem
            
            # Skip if already processed
            if conv_id in self.processed_conversations:
                continue
            
            # Check modification time
            if datetime.fromtimestamp(conv_file.stat().st_mtime, UTC) < since:
                continue
            
            try:
                # Load conversation
                with open(conv_file, 'r') as f:
                    data = json.load(f)
                
                messages = data.get("messages", [])
                if messages:
                    conversations[conv_id] = messages
                    
            except Exception as e:
                self.logger.warning(
                    f"Failed to load conversation {conv_id}: {e}"
                )
        
        return conversations
    
    def _process_conversation(self, conv_id: str, 
                            messages: List[Dict]) -> Dict[str, int]:
        """
        Process a single conversation.
        
        Args:
            conv_id: Conversation ID
            messages: List of messages
            
        Returns:
            Processing statistics
        """
        stats = {
            "passages": 0,
            "entities": 0,
            "relationships": 0
        }
        
        # Archive to passages
        passage_count = self.memory_manager.passage_manager.archive_conversation(
            conv_id, messages
        )
        stats["passages"] = passage_count
        
        # Extract entities and relationships from all messages
        all_entities = set()
        
        for msg in messages:
            if msg.get("role") not in ["user", "assistant"]:
                continue
            
            content = msg.get("content", "")
            if not content:
                continue
            
            # Handle different content formats
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            elif not isinstance(content, str):
                content = str(content)
            
            # Create temporary passage for entity extraction
            temp_passage_id = self.memory_manager.passage_manager.create_passage(
                text=content,
                source="conversation_processing",
                source_id=conv_id,
                importance=0.3,  # Low importance for temporary passages
                context={
                    "message_role": msg.get("role"),
                    "message_id": msg.get("id"),
                    "temporary": True
                }
            )
            
            # Extract entities
            entity_ids = self.memory_manager.entity_manager.extract_entities(
                content, temp_passage_id
            )
            all_entities.update(entity_ids)
            stats["entities"] += len(entity_ids)
            
            # Infer relationships within this message
            if len(entity_ids) >= 2:
                rel_ids = self.memory_manager.entity_manager.infer_relationships(
                    temp_passage_id, entity_ids
                )
                stats["relationships"] += len(rel_ids)
        
        # Infer cross-message relationships
        if len(all_entities) >= 2:
            # Create a synthetic passage representing the whole conversation
            conv_summary = self._create_conversation_summary(messages)
            
            summary_passage_id = self.memory_manager.passage_manager.create_passage(
                text=conv_summary,
                source="conversation_summary",
                source_id=conv_id,
                importance=0.6,
                context={
                    "message_count": len(messages),
                    "entity_count": len(all_entities)
                }
            )
            
            # Infer relationships across the whole conversation
            rel_ids = self.memory_manager.entity_manager.infer_relationships(
                summary_passage_id, list(all_entities)
            )
            stats["relationships"] += len(rel_ids)
        
        self.logger.info(
            f"Processed conversation {conv_id}: "
            f"{stats['passages']} passages, "
            f"{stats['entities']} entities, "
            f"{stats['relationships']} relationships"
        )
        
        return stats
    
    def _create_conversation_summary(self, messages: List[Dict]) -> str:
        """
        Create a high-level summary of a conversation.
        
        Args:
            messages: List of messages
            
        Returns:
            Summary text
        """
        # Extract key points from conversation
        key_points = []
        
        # Get first and last substantive messages
        substantive_messages = [
            msg for msg in messages 
            if msg.get("role") in ["user", "assistant"] and 
            len(str(msg.get("content", ""))) > 20
        ]
        
        if not substantive_messages:
            return "Empty conversation"
        
        # Opening context
        first_msg = substantive_messages[0]
        key_points.append(
            f"Conversation started with {first_msg.get('role')} discussing: "
            f"{str(first_msg.get('content', ''))[:100]}..."
        )
        
        # Look for questions and their answers
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                if "?" in content and i + 1 < len(messages):
                    # Found a question, get the response
                    response = messages[i + 1]
                    if response.get("role") == "assistant":
                        key_points.append(
                            f"User asked about: {content[:100]}... "
                            f"Assistant responded with: {str(response.get('content', ''))[:100]}..."
                        )
        
        # Ending context
        if len(substantive_messages) > 1:
            last_msg = substantive_messages[-1]
            key_points.append(
                f"Conversation ended with {last_msg.get('role')} saying: "
                f"{str(last_msg.get('content', ''))[:100]}..."
            )
        
        # Combine key points
        summary = " ".join(key_points[:5])  # Limit to 5 key points
        
        return summary
    
    def reprocess_conversation(self, conv_id: str) -> Dict[str, int]:
        """
        Force reprocessing of a specific conversation.
        
        Args:
            conv_id: Conversation ID to reprocess
            
        Returns:
            Processing statistics
        """
        # Remove from processed list
        self.processed_conversations.discard(conv_id)
        self._save_processed_list()
        
        # Load conversation
        history_file = Path(
            self.memory_manager.config.paths.conversation_history_dir
        ) / f"{conv_id}.json"
        
        if not history_file.exists():
            raise ValueError(f"Conversation {conv_id} not found")
        
        with open(history_file, 'r') as f:
            data = json.load(f)
        
        messages = data.get("messages", [])
        if not messages:
            raise ValueError(f"Conversation {conv_id} has no messages")
        
        # Process it
        return self._process_conversation(conv_id, messages)
    
    def get_processing_status(self) -> Dict[str, Any]:
        """
        Get current processing status and statistics.
        
        Returns:
            Status information
        """
        history_dir = Path(self.memory_manager.config.paths.conversation_history_dir)
        
        total_conversations = 0
        unprocessed = 0
        
        if history_dir.exists():
            for conv_file in history_dir.glob("*.json"):
                total_conversations += 1
                if conv_file.stem not in self.processed_conversations:
                    unprocessed += 1
        
        return {
            "total_conversations": total_conversations,
            "processed_conversations": len(self.processed_conversations),
            "unprocessed_conversations": unprocessed,
            "processing_enabled": True,
            "batch_size": self.memory_manager.config.memory.batch_process_hours
        }