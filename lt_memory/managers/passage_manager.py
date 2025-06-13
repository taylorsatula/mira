"""
Manager for archival memory passages with vector search.
"""

import uuid
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dateutil import parser as date_parser

from sqlalchemy import text
import numpy as np

from lt_memory.models.base import MemoryPassage
from errors import error_context, ErrorCode, ToolError
from utils.timezone_utils import utc_now, format_utc_iso

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
                      importance: float = 0.5, metadata: Optional[Dict] = None,
                      expires_on: Optional[datetime] = None) -> str:
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
                    code=ErrorCode.INVALID_INPUT
                )
            
            if importance < 0 or importance > 1:
                raise ToolError(
                    f"Importance must be between 0 and 1, got {importance}",
                    code=ErrorCode.INVALID_INPUT
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
                    context=metadata or {},
                    expires_on=expires_on,
                    human_verified=metadata.get("human_verified", False) if metadata else False
                )
                session.add(passage)
                session.commit()
                
                passage_id = str(passage.id)
                
            self.logger.info(
                f"Created {'expiring' if expires_on else 'permanent'} passage {passage_id} "
                f"from {source} (importance: {importance:.2f})"
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
        # Only set default threshold if user hasn't provided min_similarity
        if "min_similarity" not in search_filters:
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
                        "created_at": format_utc_iso(passage.created_at),
                        "access_count": passage.access_count,
                        "metadata": passage.context
                    })
        
        self.logger.info(
            f"Found {len(passages)} passages for query "
            f"(threshold: {search_filters['min_similarity']})"
        )
        return passages
    
    def search_passages_by_embedding(self, query_embedding, limit: int = 10, 
                                   filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search passages using a pre-computed embedding vector.
        
        Args:
            query_embedding: Pre-computed embedding vector
            limit: Maximum results to return
            filters: Optional filters (source, min_importance, etc.)
            
        Returns:
            List of passage results with similarity scores
        """
        # Use vector store for search
        search_filters = filters or {}
        # Only set default threshold if user hasn't provided min_similarity
        if "min_similarity" not in search_filters:
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
            for result in results:
                passage = session.get(MemoryPassage, result["id"])
                if passage:
                    passages.append({
                        "id": str(passage.id),
                        "text": passage.text,
                        "similarity": result["score"],
                        "source": passage.source,
                        "source_id": passage.source_id,
                        "importance": passage.importance_score,
                        "created_at": format_utc_iso(passage.created_at),
                        "access_count": passage.access_count,
                        "metadata": passage.context
                    })
        
        self.logger.info(
            f"Found {len(passages)} passages for embedding search "
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
                passage.last_accessed = utc_now()
                session.commit()
                
                return {
                    "id": str(passage.id),
                    "text": passage.text,
                    "source": passage.source,
                    "source_id": passage.source_id,
                    "importance": passage.importance_score,
                    "created_at": format_utc_iso(passage.created_at),
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
                code=ErrorCode.INVALID_INPUT
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
        Archive a conversation by extracting facts as micro-chunks.
        
        Args:
            conversation_id: ID of the conversation
            messages: List of message dictionaries
            
        Returns:
            Number of facts (passages) created
        """
        if not messages:
            return 0
        
        archived_count = 0
        
        # Chunk messages into logical passages
        chunks = self._chunk_messages(messages)
        
        for i, chunk in enumerate(chunks):
            # Extract facts from this chunk
            facts = self._extract_facts(chunk)
            
            # Get metadata for this chunk
            chunk_metadata = {
                "chunk_index": i,
                "chunk_type": "micro_chunk",
                "extraction_version": "1.0",
                "start_time": chunk[0].get("created_at"),
                "end_time": chunk[-1].get("created_at"),
                "source_message_count": len(chunk)
            }
            
            # Check for duplicates and create passages for each fact
            for fact in facts:
                # Check for duplicates before creating
                duplicate_id, should_update = self._check_duplicate_fact(
                    fact["text"], 
                    fact.get("expires_on")
                )
                
                if duplicate_id and should_update:
                    # For temporal updates, we need to replace the fact content entirely
                    # Delete the old fact and create the new one
                    self.delete_passage(duplicate_id)
                    
                    # Create new passage with updated fact content
                    metadata = chunk_metadata.copy()
                    metadata.update({
                        "human_verified": False,
                        "confidence": 0.9,
                        "replaces": duplicate_id  # Track what this replaced
                    })
                    
                    self.create_passage(
                        text=fact["text"],
                        source="conversation",
                        source_id=conversation_id,
                        importance=fact["importance"],
                        metadata=metadata,
                        expires_on=fact.get("expires_on")  # New fact has its own expiration
                    )
                    archived_count += 1
                    self.logger.info(f"Replaced temporal fact: {duplicate_id} with new content")
                elif not duplicate_id:
                    # Create new passage for this fact
                    metadata = chunk_metadata.copy()
                    metadata.update({
                        "human_verified": False,
                        "confidence": 0.9  # Default high confidence for LLM extraction
                    })
                    
                    self.create_passage(
                        text=fact["text"],
                        source="conversation",
                        source_id=conversation_id,
                        importance=fact["importance"],
                        metadata=metadata,
                        expires_on=fact.get("expires_on")
                    )
                    archived_count += 1
        
        self.logger.info(
            f"Archived {archived_count} facts from conversation {conversation_id} "
            f"({len(messages)} messages)"
        )
        return archived_count
    
    def _chunk_messages(self, messages: List[Dict], 
                       max_chunk_size: int = 50) -> List[List[Dict]]:
        """
        Chunk messages using topic boundaries marked by MIRA.
        
        Creates a new chunk whenever:
        - Assistant message contains topic_changed=true in metadata
        - Or we hit the maximum chunk size as fallback
        
        Args:
            messages: List of messages
            max_chunk_size: Maximum chunk size as fallback (default: 50)
            
        Returns:
            List of message chunks grouped by topic boundaries
        """
        chunks = []
        current_chunk = []
        
        for msg in messages:
            current_chunk.append(msg)
            
            # Check if this assistant message marked a topic change
            if (msg.get("role") == "assistant" and 
                msg.get("metadata", {}).get("topic_changed", False)):
                
                # End current chunk and start new one at topic boundary
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
            
            # Fallback: Don't let chunks get too huge
            elif len(current_chunk) >= max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
        
        # Add remaining messages
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _extract_facts(self, messages: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract discrete facts from user messages in a chunk.
        
        Each fact is a micro-chunk of information about the user
        that can be individually stored and searched.
        
        Args:
            messages: List of messages to extract facts from
            
        Returns:
            List of fact dictionaries with text, expires_on, and importance
        """
        # Filter to only user messages
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        
        if not user_messages:
            return []
        
        with error_context("passage_manager", "_extract_facts", ToolError, ErrorCode.MEMORY_ERROR):
            # Load fact extraction prompt
            prompt_path = self.memory_manager.config.app_root / "config" / "prompts" / "fact_extraction" / "extract_facts.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
            
            # Format user messages for fact extraction
            user_content = self._format_user_messages_for_extraction(user_messages)
            
            # Call LLM for fact extraction
            response = self.memory_manager.llm_provider.generate_response(
                messages=[{"role": "user", "content": user_content}],
                system_prompt=system_prompt,
                max_tokens=2000,
                temperature=0.1  # Low temperature for consistent extraction
            )
            
            # Parse JSON response
            response_text = response["content"][0]["text"].strip()
            facts = json.loads(response_text)
            
            # Validate and normalize facts
            validated_facts = []
            for fact in facts:
                if self._validate_fact(fact):
                    # Parse expires_on if it's a string
                    if fact.get("expires_on") and isinstance(fact["expires_on"], str):
                        try:
                            fact["expires_on"] = date_parser.parse(fact["expires_on"])
                        except:
                            fact["expires_on"] = None
                    validated_facts.append(fact)
            
            self.logger.info(
                f"Extracted {len(validated_facts)} facts from {len(user_messages)} user messages"
            )
            return validated_facts
    
    def _format_user_messages_for_extraction(self, user_messages: List[Dict]) -> str:
        """Format user messages for fact extraction."""
        formatted_lines = []
        
        for msg in user_messages:
            content = msg.get("content", "")
            
            # Handle different content types
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            elif not isinstance(content, str):
                content = str(content)
            
            # Skip empty messages
            if content.strip():
                formatted_lines.append(content)
        
        return "\n\n".join(formatted_lines)
    
    def _validate_fact(self, fact: Dict) -> bool:
        """Validate a fact dictionary has required fields."""
        required_fields = ["text", "importance"]
        
        # Check required fields
        for field in required_fields:
            if field not in fact:
                return False
        
        # Validate text is non-empty
        if not fact["text"].strip():
            return False
        
        # Validate importance is between 0 and 1
        try:
            importance = float(fact["importance"])
            if importance < 0 or importance > 1:
                return False
        except (ValueError, TypeError):
            return False
        
        return True
    
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
    
    def _check_duplicate_fact(self, fact_text: str, 
                             expires_on: Optional[datetime]) -> Tuple[Optional[str], bool]:
        """
        Check if a fact is a duplicate using vector similarity and LLM reasoning.
        
        Args:
            fact_text: The fact text to check
            expires_on: Expiration date of the new fact
            
        Returns:
            Tuple of (duplicate_passage_id, should_update_expiration)
        """
        # Search for similar facts
        results = self.search_passages(
            query=fact_text,
            limit=5,
            filters={"min_similarity": 0.7}  # Lower threshold to catch more candidates
        )
        
        if not results:
            return None, False
        
        # Use LLM to analyze if these are truly duplicates or updates
        duplicate_prompt = f"""Analyze if these facts are duplicates, updates, or distinct facts.

New fact: "{fact_text}"
New expiration: {format_utc_iso(expires_on) if expires_on else 'permanent'}

Existing similar facts:
"""
        
        for i, result in enumerate(results[:3]):
            existing_expires = result.get("metadata", {}).get("expires_on")
            duplicate_prompt += f"\n{i+1}. \"{result['text']}\" (expires: {existing_expires or 'permanent'}, similarity: {result['score']:.2f})"
        
        duplicate_prompt += """\n\nRespond with JSON:
{
  "is_duplicate": true/false,
  "duplicate_of_index": 1/2/3 or null,
  "is_temporal_update": true/false,
  "reasoning": "brief explanation"
}"""
        
        try:
            response = self.memory_manager.llm_provider.generate_response(
                messages=[{"role": "user", "content": duplicate_prompt}],
                system_prompt="You are analyzing facts for deduplication. Be precise in identifying true duplicates vs related but distinct facts.",
                max_tokens=200,
                temperature=0.1
            )
            
            analysis = json.loads(response["content"][0]["text"].strip())
            
            if analysis["is_duplicate"] and analysis["duplicate_of_index"]:
                duplicate_idx = analysis["duplicate_of_index"] - 1
                if 0 <= duplicate_idx < len(results):
                    duplicate_id = results[duplicate_idx]["id"]
                    
                    # If it's a temporal update, we need to replace the fact
                    should_update = analysis.get("is_temporal_update", False)
                    
                    self.logger.info(
                        f"Duplicate analysis: {analysis['reasoning']}"
                    )
                    
                    return duplicate_id, should_update
            
        except Exception as e:
            self.logger.warning(f"LLM duplicate analysis failed: {e}")
            # Fall back to simple similarity check
            if results[0]["score"] >= self.memory_manager.config.memory.fact_similarity_threshold:
                return results[0]["id"], False
        
        return None, False
    
    def _update_fact_expiration(self, passage_id: str, 
                               new_expires_on: Optional[datetime]) -> bool:
        """
        Update the expiration date of a fact using LLM reasoning.
        
        NOTE: This is for updating expiration only, not fact content.
        For temporal updates that change the fact itself, the old fact
        should be deleted and replaced with a new one.
        
        Args:
            passage_id: ID of the passage to update
            new_expires_on: New expiration date
            
        Returns:
            True if updated successfully
        """
        with self.memory_manager.get_session() as session:
            passage = session.query(MemoryPassage).filter_by(
                id=passage_id
            ).first()
            
            if not passage:
                return False
            
            old_expires = passage.expires_on
            
            # Use LLM to determine if the expiration update makes sense
            update_prompt = f"""Should we update the expiration date of this fact?

Fact: "{passage.text}"
Current expiration: {format_utc_iso(old_expires) if old_expires else 'permanent'}
Proposed new expiration: {format_utc_iso(new_expires_on) if new_expires_on else 'permanent'}
Fact importance: {passage.importance_score}
Human verified: {passage.human_verified}

Consider:
- Is the new expiration reasonable for this type of fact?
- Should permanent facts remain permanent?
- Does the update make logical sense?

Respond with JSON:
{{
  "should_update": true/false,
  "recommended_expiration": "ISO date or null",
  "reasoning": "brief explanation"
}}"""
            
            try:
                response = self.memory_manager.llm_provider.generate_response(
                    messages=[{"role": "user", "content": update_prompt}],
                    system_prompt="You are managing fact expiration dates. Be thoughtful about which facts should expire and when.",
                    max_tokens=200,
                    temperature=0.1
                )
                
                decision = json.loads(response["content"][0]["text"].strip())
                
                if decision["should_update"]:
                    # Use recommended expiration if provided
                    if decision.get("recommended_expiration"):
                        try:
                            final_expires = date_parser.parse(decision["recommended_expiration"])
                        except:
                            final_expires = new_expires_on
                    else:
                        final_expires = new_expires_on
                    
                    passage.expires_on = final_expires
                    session.commit()
                    
                    self.logger.info(
                        f"Updated passage {passage_id} expiration: "
                        f"{format_utc_iso(old_expires) if old_expires else 'None'} -> "
                        f"{format_utc_iso(final_expires) if final_expires else 'None'} "
                        f"(Reason: {decision['reasoning']})"
                    )
                    return True
                else:
                    self.logger.info(
                        f"Skipped expiration update for {passage_id}: {decision['reasoning']}"
                    )
                    return False
                    
            except Exception as e:
                self.logger.warning(f"LLM expiration analysis failed: {e}")
                # Fall back to simple update
                passage.expires_on = new_expires_on
                session.commit()
                return True
        
        return False
    
    def expire_old_memories(self) -> int:
        """
        Remove facts that have passed their expiration date.
        
        Returns:
            Number of expired facts removed
        """
        if not self.memory_manager.config.memory.auto_expire_enabled:
            return 0
        
        expired_count = 0
        current_time = utc_now()
        
        with self.memory_manager.get_session() as session:
            # Find expired passages
            expired_passages = session.query(MemoryPassage).filter(
                MemoryPassage.expires_on != None,
                MemoryPassage.expires_on < current_time
            ).all()
            
            for passage in expired_passages:
                self.logger.info(
                    f"Expiring fact: {passage.text[:50]}... "
                    f"(expired: {format_utc_iso(passage.expires_on)})"
                )
                session.delete(passage)
                expired_count += 1
            
            session.commit()
        
        if expired_count > 0:
            self.logger.info(f"Expired {expired_count} old facts")
        
        return expired_count
    
    def update_passage_expiration(self, passage_id: str, 
                                 new_expires_on: Optional[datetime]) -> bool:
        """
        Public method to update passage expiration.
        
        Args:
            passage_id: Passage ID to update
            new_expires_on: New expiration date (None for permanent)
            
        Returns:
            True if updated successfully
        """
        return self._update_fact_expiration(passage_id, new_expires_on)
    
    def get_expiring_memories(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Get facts that will expire soon.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of soon-to-expire facts
        """
        cutoff = utc_now() + timedelta(days=days_ahead)
        
        with self.memory_manager.get_session() as session:
            expiring = session.query(MemoryPassage).filter(
                MemoryPassage.expires_on != None,
                MemoryPassage.expires_on <= cutoff,
                MemoryPassage.expires_on > utc_now()
            ).order_by(
                MemoryPassage.expires_on
            ).all()
            
            return [
                {
                    "id": str(p.id),
                    "text": p.text,
                    "expires_on": format_utc_iso(p.expires_on),
                    "importance": p.importance_score,
                    "human_verified": p.human_verified
                }
                for p in expiring
            ]
    
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
        cutoff = utc_now() - timedelta(hours=hours)
        
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
                    "created_at": format_utc_iso(p.created_at)
                }
                for p in passages
            ]