"""
Manager for archival memory passages with vector search.
"""

import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC

from sqlalchemy import text
import numpy as np

from lt_memory.models.base import MemoryPassage
from errors import error_context, ErrorCode, ToolError

logger = logging.getLogger(__name__)


class PassageManager:
    """
    Manages archival memory passages.
    
    Handles storage and retrieval of long-term memories with
    vector similarity search capabilities.
    """
    
    def __init__(self, memory_manager):
        """
        Initialize passage manager.
        
        Args:
            memory_manager: Parent MemoryManager instance
        """
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
    
    def create_passage(self, text: str, source: str, source_id: str, 
                      importance: float = 0.5, metadata: Optional[Dict] = None) -> str:
        """
        Create a new memory passage.
        
        Args:
            text: Passage text content
            source: Source type (conversation, document, automation)
            source_id: ID of the source
            importance: Initial importance score (0-1)
            metadata: Additional metadata
            
        Returns:
            Passage ID
            
        Raises:
            ToolError: If creation fails
        """
        with error_context("passage_manager", "create_passage", ToolError, ErrorCode.MEMORY_ERROR):
            # Validate inputs
            if not text.strip():
                raise ToolError(
                    "Cannot create passage with empty text",
                    error_code=ErrorCode.INVALID_INPUT
                )
            
            if importance < 0 or importance > 1:
                raise ToolError(
                    f"Importance must be between 0 and 1, got {importance}",
                    error_code=ErrorCode.INVALID_INPUT
                )
            
            # Generate embedding
            embedding = self.memory_manager.generate_embedding(text)
            
            with self.memory_manager.get_session() as session:
                passage = MemoryPassage(
                    text=text,
                    embedding=embedding,
                    source=source,
                    source_id=source_id,
                    importance_score=importance,
                    context=metadata or {}
                )
                session.add(passage)
                session.commit()
                
                passage_id = str(passage.id)
                
            self.logger.info(
                f"Created passage {passage_id} from {source} "
                f"(importance: {importance:.2f})"
            )
            return passage_id
    
    def search_passages(self, query: str, limit: int = 10, 
                       filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search passages using vector similarity.
        
        Args:
            query: Search query text
            limit: Maximum results to return
            filters: Optional filters (source, min_importance, etc.)
            
        Returns:
            List of passage results with similarity scores
        """
        # Generate query embedding
        query_embedding = self.memory_manager.generate_embedding(query)
        
        # Use vector store for search
        search_filters = filters or {}
        search_filters["min_similarity"] = self.memory_manager.config.memory.similarity_threshold
        
        results = self.memory_manager.vector_store.search(
            query_embedding,
            k=limit,
            table="memory_passages",
            filters=search_filters
        )
        
        # Fetch full passage data
        passages = []
        with self.memory_manager.get_session() as session:
            for passage_id, score in results:
                passage = session.query(MemoryPassage).filter_by(
                    id=passage_id
                ).first()
                
                if passage:
                    passages.append({
                        "id": str(passage.id),
                        "text": passage.text,
                        "score": float(score),
                        "source": passage.source,
                        "source_id": passage.source_id,
                        "importance": passage.importance_score,
                        "created_at": passage.created_at.isoformat(),
                        "access_count": passage.access_count,
                        "metadata": passage.context
                    })
        
        self.logger.info(
            f"Found {len(passages)} passages for query "
            f"(threshold: {search_filters['min_similarity']})"
        )
        return passages
    
    def get_passage(self, passage_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific passage by ID.
        
        Args:
            passage_id: Passage ID
            
        Returns:
            Passage data or None if not found
        """
        with self.memory_manager.get_session() as session:
            passage = session.query(MemoryPassage).filter_by(
                id=passage_id
            ).first()
            
            if passage:
                # Update access tracking
                passage.access_count += 1
                passage.last_accessed = datetime.now(UTC)
                session.commit()
                
                return {
                    "id": str(passage.id),
                    "text": passage.text,
                    "source": passage.source,
                    "source_id": passage.source_id,
                    "importance": passage.importance_score,
                    "created_at": passage.created_at.isoformat(),
                    "access_count": passage.access_count,
                    "metadata": passage.context
                }
        
        return None
    
    def update_passage_importance(self, passage_id: str, 
                                 importance: float) -> bool:
        """
        Update the importance score of a passage.
        
        Args:
            passage_id: Passage ID
            importance: New importance score (0-1)
            
        Returns:
            True if updated successfully
        """
        if importance < 0 or importance > 1:
            raise ToolError(
                f"Importance must be between 0 and 1, got {importance}",
                error_code=ErrorCode.INVALID_INPUT
            )
        
        with self.memory_manager.get_session() as session:
            passage = session.query(MemoryPassage).filter_by(
                id=passage_id
            ).first()
            
            if passage:
                old_importance = passage.importance_score
                passage.importance_score = importance
                session.commit()
                
                self.logger.info(
                    f"Updated passage {passage_id} importance: "
                    f"{old_importance:.2f} -> {importance:.2f}"
                )
                return True
        
        return False
    
    def archive_conversation(self, conversation_id: str, 
                           messages: List[Dict]) -> int:
        """
        Archive a conversation to passages.
        
        Args:
            conversation_id: ID of the conversation
            messages: List of message dictionaries
            
        Returns:
            Number of passages created
        """
        if not messages:
            return 0
        
        archived_count = 0
        
        # Chunk messages into logical passages
        chunks = self._chunk_messages(messages)
        
        for i, chunk in enumerate(chunks):
            # Create summary of chunk
            summary = self._summarize_chunk(chunk)
            
            # Calculate importance based on content
            importance = self._calculate_importance(chunk)
            
            # Extract key topics for metadata
            topics = self._extract_topics(chunk)
            
            # Create passage
            self.create_passage(
                text=summary,
                source="conversation",
                source_id=conversation_id,
                importance=importance,
                context={
                    "chunk_index": i,
                    "message_count": len(chunk),
                    "start_time": chunk[0].get("created_at"),
                    "end_time": chunk[-1].get("created_at"),
                    "topics": topics,
                    "participants": list(set(msg.get("role", "unknown") for msg in chunk))
                }
            )
            archived_count += 1
        
        self.logger.info(
            f"Archived {archived_count} passages from conversation {conversation_id} "
            f"({len(messages)} messages)"
        )
        return archived_count
    
    def _chunk_messages(self, messages: List[Dict], 
                       chunk_size: int = 10) -> List[List[Dict]]:
        """
        Chunk messages into logical groups.
        
        Args:
            messages: List of messages
            chunk_size: Target chunk size
            
        Returns:
            List of message chunks
        """
        chunks = []
        current_chunk = []
        
        for msg in messages:
            current_chunk.append(msg)
            
            # Start new chunk on natural boundaries
            if (len(current_chunk) >= chunk_size or
                msg.get("role") == "user" and len(current_chunk) > chunk_size // 2):
                chunks.append(current_chunk)
                current_chunk = []
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _summarize_chunk(self, messages: List[Dict]) -> str:
        """
        Create a summary of a message chunk.
        
        For now, uses simple concatenation. In production,
        this would use LLM-based summarization.
        
        Args:
            messages: List of messages to summarize
            
        Returns:
            Summary text
        """
        summary_parts = []
        
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Handle different content types
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            elif not isinstance(content, str):
                content = str(content)
            
            # Truncate long messages
            if len(content) > 500:
                content = content[:497] + "..."
            
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)
    
    def _calculate_importance(self, messages: List[Dict]) -> float:
        """
        Calculate importance score for a chunk of messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Importance score (0-1)
        """
        base_score = 0.5
        
        # Combine all message content
        all_content = " ".join(
            str(msg.get("content", "")) for msg in messages
        ).lower()
        
        # Check for importance indicators
        important_keywords = [
            "important", "remember", "critical", "always", "never",
            "key", "essential", "must", "vital", "crucial"
        ]
        
        keyword_matches = sum(1 for kw in important_keywords if kw in all_content)
        keyword_bonus = min(0.3, keyword_matches * 0.05)
        
        # Length bonus - longer conversations might be more important
        length_bonus = min(0.2, len(messages) * 0.02)
        
        # User message ratio - more user input might indicate importance
        user_messages = sum(1 for msg in messages if msg.get("role") == "user")
        user_ratio = user_messages / len(messages) if messages else 0
        user_bonus = user_ratio * 0.1
        
        return min(1.0, base_score + keyword_bonus + length_bonus + user_bonus)
    
    def _extract_topics(self, messages: List[Dict]) -> List[str]:
        """
        Extract key topics from messages.
        
        Simple keyword extraction for now. In production,
        would use NLP techniques or LLM.
        
        Args:
            messages: List of messages
            
        Returns:
            List of topic keywords
        """
        import re
        from collections import Counter
        
        # Combine all content
        all_content = " ".join(
            str(msg.get("content", "")) for msg in messages
        )
        
        # Extract words (simple tokenization)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_content.lower())
        
        # Filter common words
        common_words = {
            "that", "this", "with", "from", "have", "been",
            "will", "would", "could", "should", "about", "there",
            "their", "what", "when", "where", "which", "while"
        }
        
        filtered_words = [w for w in words if w not in common_words]
        
        # Get most common words as topics
        word_counts = Counter(filtered_words)
        topics = [word for word, _ in word_counts.most_common(5)]
        
        return topics
    
    def delete_passage(self, passage_id: str) -> bool:
        """
        Delete a passage.
        
        Args:
            passage_id: Passage ID to delete
            
        Returns:
            True if deleted successfully
        """
        with self.memory_manager.get_session() as session:
            passage = session.query(MemoryPassage).filter_by(
                id=passage_id
            ).first()
            
            if passage:
                session.delete(passage)
                session.commit()
                
                self.logger.info(f"Deleted passage {passage_id}")
                return True
        
        return False
    
    def get_recent_passages(self, hours: int = 24, 
                           limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recently created passages.
        
        Args:
            hours: How many hours back to look
            limit: Maximum passages to return
            
        Returns:
            List of recent passages
        """
        from datetime import timedelta
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        
        with self.memory_manager.get_session() as session:
            passages = session.query(MemoryPassage).filter(
                MemoryPassage.created_at > cutoff
            ).order_by(
                MemoryPassage.created_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    "id": str(p.id),
                    "text": p.text[:200] + "..." if len(p.text) > 200 else p.text,
                    "source": p.source,
                    "importance": p.importance_score,
                    "created_at": p.created_at.isoformat()
                }
                for p in passages
            ]