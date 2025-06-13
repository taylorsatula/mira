"""
Integration module for adding LT_Memory to MIRA.

This module provides the initialization function to be called from main.py.
"""

import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)


def initialize_lt_memory(config, working_memory, tool_repo, automation_controller, llm_provider) -> Dict[str, Any]:
    """
    Initialize LT_Memory system and integrate with MIRA.
    
    Args:
        config: Application configuration
        working_memory: WorkingMemory instance
        tool_repo: ToolRepository instance
        automation_controller: Automation controller instance
        llm_provider: LLM provider for text generation
        
    Returns:
        Dictionary with LT_Memory components
    """
    logger.info("Initializing LT_Memory system...")
    
    try:
        # Import components
        from lt_memory.managers.memory_manager import MemoryManager
        from lt_memory.timeline_manager import MemoryBridge, ConversationTimelineManager
        from lt_memory.tools.memory_tool import LTMemoryTool
        from lt_memory.automations.memory_automations import register_memory_automations
        
        # Ensure memory configuration exists
        if not hasattr(config, 'memory'):
            logger.info("Adding default memory configuration")
            from config.memory_config import MemoryConfig
            config.memory = MemoryConfig()
        
        
        # Create memory manager
        logger.info("Creating memory manager...")
        memory_manager = MemoryManager(config, llm_provider)
        
        # Run health check
        health = memory_manager.health_check()
        if health["status"] == "unhealthy":
            logger.error(f"Memory system unhealthy: {health['issues']}")
            # Continue anyway - system may recover
        else:
            logger.info(f"Memory system status: {health['status']}")
        
        # Create bridges
        logger.info("Creating memory bridges...")
        memory_bridge = MemoryBridge(working_memory, memory_manager)
        conversation_timeline_manager = ConversationTimelineManager(memory_manager)
        
        # Register memory bridge with working memory
        logger.info("Registering memory bridge with working memory...")
        working_memory.register_manager(memory_bridge)
        
        # Create tool interface
        logger.info("Creating memory tool...")
        memory_tool = LTMemoryTool(memory_manager)
        
        # Register tool with repository
        logger.info("Registering memory tool...")
        tool_repo.register_tool(memory_tool)
        
        # Register automations
        logger.info("Registering memory automations...")
        registered_automations = register_memory_automations(automation_controller)
        logger.info(f"Registered {len(registered_automations)} memory automations")
        
        # Log initial statistics
        stats = memory_manager.get_memory_stats()
        archive_stats = memory_manager.conversation_archive.get_archive_stats()
        logger.info(
            f"Memory system initialized with "
            f"{stats['blocks']['count']} core blocks, "
            f"{stats['passages']['count']} passages, "
            f"{archive_stats['total_archived_conversations']} archived conversations"
        )
        
        return {
            "manager": memory_manager,
            "bridge": memory_bridge,
            "conversation_timeline_manager": conversation_timeline_manager,
            "tool": memory_tool,
            "automations": registered_automations
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize LT_Memory: {e}")
        # Return minimal components so system can still run
        return {
            "manager": None,
            "bridge": None,
            "conversation_timeline_manager": None,
            "tool": None,
            "automations": {}
        }


def check_lt_memory_requirements() -> Dict[str, bool]:
    """
    Check if all requirements for LT_Memory are met.
    
    Returns:
        Dictionary with requirement status
    """
    requirements = {
        "postgresql": False,
        "pgvector": False,
        "openai_api": False,
        "database_url": False
    }
    
    # Check for PostgreSQL connection
    db_url = os.getenv("LT_MEMORY_DATABASE_URL")
    if db_url and db_url.startswith("postgresql://"):
        requirements["database_url"] = True
        
        # Try to connect and check pgvector
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                ).fetchone()
                if result:
                    requirements["pgvector"] = True
                requirements["postgresql"] = True
        except Exception as e:
            logger.warning(f"PostgreSQL check failed: {e}")
    
    # Check for OpenAI API key (required for embeddings)
    openai_key = os.getenv("OAI_EMBEDDINGS_KEY")
    if openai_key and openai_key.strip():
        requirements["openai_api"] = True
    
    return requirements