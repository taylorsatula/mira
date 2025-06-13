"""
White-label summarization engine for LT_Memory.

Provides flexible summarization capabilities with configurable scopes
and compression levels using system/user message separation.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from errors import error_context, ErrorCode, ToolError

logger = logging.getLogger(__name__)


class SummarizationEngine:
    """
    White-label summarization engine with scope-based system prompts
    and compression-level user instructions.
    """
    
    def __init__(self, llm_provider, template_dir: Optional[str] = None):
        """
        Initialize summarization engine.
        
        Args:
            llm_provider: LLM provider for text generation
            template_dir: Directory containing scope template files
        """
        self.llm_provider = llm_provider
        
        # Set template directory
        if template_dir:
            self.template_dir = Path(template_dir)
        else:
            # Default to config/prompts/summarization/
            self.template_dir = Path("config/prompts/summarization")
        
        self.template_cache = {}
        self.logger = logging.getLogger(__name__)
        
        # Compression level instructions (hardcoded)
        self.compression_instructions = {
            "ultra_concise": "Extract only the essence of the messages. One sentence maximum.",
            "concise": "Create a concise summary of the content. Three sentences maximum.",
            "detailed": "Create a detailed recap of the messages. Six sentences maximum."
        }
    
    def summarize(self, 
                  messages: List[Dict], 
                  scope: str = "daily",
                  compression_level: str = "detailed") -> str:
        """
        Summarize messages using specified scope and compression level.
        
        Args:
            messages: List of message dictionaries to summarize
            scope: Temporal scope (daily, weekly, monthly)
            compression_level: Level of compression (ultra_concise, concise, detailed)
            
        Returns:
            Summarized text
            
        Raises:
            ToolError: If summarization fails
        """
        with error_context("summarization_engine", "summarize", ToolError, ErrorCode.MEMORY_ERROR):
            if not messages:
                return "No messages to summarize."
            
            # Get system message (scope template)
            system_message = self._get_scope_template(scope)
            
            # Get user message (compression instruction + conversation)
            user_message = self._build_user_message(messages, compression_level)
            
            # Call LLM with system/user message structure
            try:
                response = self.llm_provider.generate_response(
                    messages=[
                        {"role": "user", "content": user_message}
                    ],
                    system_prompt=system_message,
                    max_tokens=self._get_max_tokens(compression_level),
                    temperature=0.1  # Low temperature for consistent summaries
                )
                
                # Extract text from standardized response object
                summary = response["content"][0]["text"].strip()
                
                self.logger.info(
                    f"Generated {scope}/{compression_level} summary: "
                    f"{len(messages)} messages -> {len(summary)} chars"
                )
                
                return summary
                
            except Exception as e:
                raise ToolError(
                    f"LLM summarization failed: {str(e)}",
                    code=ErrorCode.MEMORY_ERROR
                )
    
    def _get_scope_template(self, scope: str) -> str:
        """
        Load system message template for the specified scope.
        
        Args:
            scope: Temporal scope (daily, weekly, monthly)
            
        Returns:
            Template content as string
        """
        # Check cache first
        if scope in self.template_cache:
            template = self.template_cache[scope]
        else:
            # Load template from file
            template_file = self.template_dir / f"{scope}.txt"
            
            if not template_file.exists():
                # Single fallback
                template = "You are summarizing a conversation between a user and AI assistant. Focus on the most important information and outcomes."
            else:
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        template = f.read().strip()
                except (PermissionError, OSError) as e:
                    self.logger.warning(f"Could not read template file {template_file}: {e}")
                    # Fall back to default template
                    template = "You are summarizing a conversation between a user and AI assistant. Focus on the most important information and outcomes."
            
            # Cache for future use
            self.template_cache[scope] = template
        
        return template
    
    def _build_user_message(self, messages: List[Dict], compression_level: str) -> str:
        """
        Build user message with compression instruction and conversation text.
        
        Args:
            messages: List of message dictionaries
            compression_level: Compression level
            
        Returns:
            User message string
        """
        # Get compression instruction
        instruction = self.compression_instructions.get(
            compression_level, 
            self.compression_instructions["detailed"]
        )
        
        # Format conversation
        conversation_text = self._format_conversation(messages)
        
        # Build user message
        user_message = f"{instruction}\n\nConversation to summarize:\n\n{conversation_text}"
        
        return user_message
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        """
        Format messages into readable conversation text.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted conversation string
        """
        formatted_lines = []
        
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Handle different content types
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            elif not isinstance(content, str):
                content = str(content)
            
            # Skip empty messages
            if not content.strip():
                continue
            
            # Format with role prefix
            if role == "user":
                formatted_lines.append(f"User: {content}")
            elif role == "assistant":
                formatted_lines.append(f"Assistant: {content}")
            else:
                formatted_lines.append(f"{role.title()}: {content}")
        
        return "\n".join(formatted_lines)
    
    def _get_max_tokens(self, compression_level: str) -> int:
        """Get maximum tokens for compression level."""
        token_limits = {
            "ultra_concise": 50,
            "concise": 200,
            "detailed": 500
        }
        return token_limits.get(compression_level, 300)
    
    def get_available_scopes(self) -> List[str]:
        """
        Get list of available scope templates.
        
        Returns:
            List of scope names (without .txt extension)
        """
        if not self.template_dir.exists():
            return []
        
        scopes = []
        for file in self.template_dir.glob("*.txt"):
            scopes.append(file.stem)
        
        return sorted(scopes)
    
    def get_compression_levels(self) -> List[str]:
        """
        Get list of available compression levels.
        
        Returns:
            List of compression level names
        """
        return list(self.compression_instructions.keys())
    
    def validate_setup(self) -> Dict[str, Any]:
        """
        Validate summarization engine setup.
        
        Returns:
            Validation results
        """
        results = {
            "template_dir_exists": self.template_dir.exists(),
            "available_scopes": self.get_available_scopes(),
            "compression_levels": self.get_compression_levels(),
            "llm_provider_available": self.llm_provider is not None
        }
        
        return results