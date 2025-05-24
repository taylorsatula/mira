"""
Integration module for adding LT_Memory to MIRA.

This module provides the initialization function to be called from main.py.
"""

import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)


def initialize_lt_memory(config, working_memory, tool_repo, automation_controller) -> Dict[str, Any]:
    """
    Initialize LT_Memory system and integrate with MIRA.
    
    Args:
        config: Application configuration
        working_memory: WorkingMemory instance
        tool_repo: ToolRepository instance
        automation_controller: Automation controller instance
        
    Returns:
        Dictionary with LT_Memory components
    """
    logger.info("Initializing LT_Memory system...")
    
    try:
        # Import components
        from lt_memory.managers.memory_manager import MemoryManager
        from lt_memory.bridge import MemoryBridge
        from lt_memory.tools.memory_tool import LTMemoryTool
        from lt_memory.automations.memory_automations import register_memory_automations
        
        # Ensure memory configuration exists
        if not hasattr(config, 'memory'):
            logger.info("Adding default memory configuration")
            from config.config import MemoryConfig
            config.memory = MemoryConfig()
        
        # Create memory manager
        logger.info("Creating memory manager...")
        memory_manager = MemoryManager(config)
        
        # Run health check
        health = memory_manager.health_check()
        if health["status"] == "unhealthy":
            logger.error(f"Memory system unhealthy: {health['issues']}")
            # Continue anyway - system may recover
        else:
            logger.info(f"Memory system status: {health['status']}")
        
        # Create bridge to working memory
        logger.info("Creating memory bridge...")
        memory_bridge = MemoryBridge(working_memory, memory_manager)
        
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
        logger.info(
            f"Memory system initialized with "
            f"{stats['blocks']['count']} core blocks, "
            f"{stats['passages']['count']} passages, "
            f"{stats['entities']['count']} entities"
        )
        
        return {
            "manager": memory_manager,
            "bridge": memory_bridge,
            "tool": memory_tool,
            "automations": registered_automations
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize LT_Memory: {e}")
        # Return minimal components so system can still run
        return {
            "manager": None,
            "bridge": None,
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
        "onnx_model": False,
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
    
    # Check for ONNX model
    onnx_path = os.getenv("LT_MEMORY_ONNX_MODEL", "onnx/model.onnx")
    if os.path.exists(onnx_path):
        requirements["onnx_model"] = True
    
    return requirements