"""
Pre-configured memory management automations.
"""

import logging

logger = logging.getLogger(__name__)

# Memory automation definitions
MEMORY_AUTOMATIONS = [
    {
        "name": "Hourly Conversation Processing",
        "description": "Process recent conversations into memory every hour",
        "type": "simple_task",
        "execution_mode": "direct",
        "frequency": "hourly",
        "enabled": True,
        "tool_name": "lt_memory",
        "tool_params": {
            "operation": "process_recent_conversations",
            "hours": 1
        },
        "metadata": {
            "category": "memory_processing",
            "priority": "high"
        }
    },
    {
        "name": "Daily Memory Consolidation",
        "description": "Consolidate and optimize memories every night",
        "type": "sequence",
        "frequency": "daily",
        "scheduled_time": "02:00",
        "enabled": True,
        "steps": [
            {
                "name": "Process all pending conversations",
                "execution_mode": "direct",
                "tool_name": "lt_memory",
                "tool_params": {
                    "operation": "process_all_pending"
                }
            },
            {
                "name": "Run memory consolidation",
                "execution_mode": "direct",
                "tool_name": "lt_memory",
                "tool_params": {
                    "operation": "consolidate_memories"
                }
            },
            {
                "name": "Update importance scores",
                "execution_mode": "direct",
                "tool_name": "lt_memory",
                "tool_params": {
                    "operation": "update_importance_scores"
                }
            }
        ],
        "metadata": {
            "category": "memory_maintenance",
            "priority": "medium"
        }
    },
    {
        "name": "Weekly Memory Review",
        "description": "Review memories and update user understanding",
        "type": "simple_task",
        "execution_mode": "orchestrated",
        "frequency": "weekly",
        "scheduled_time": "sunday 00:00",
        "enabled": True,
        "task_description": """Review the week's memories and perform the following:

1. Search archival memory for patterns in user behavior and preferences
2. Identify any new insights about the user that should be remembered
3. Update the 'human' core memory block with significant new information
4. Look for entities that have become important this week
5. Check if any old memories can be safely forgotten

Use the lt_memory tool to search memories, update core memory, and get entity information.
Focus on actionable insights that will improve future interactions.""",
        "available_tools": ["lt_memory"],
        "metadata": {
            "category": "memory_analysis",
            "priority": "low",
            "max_thinking_time": 30000
        }
    },
    {
        "name": "Memory Health Check",
        "description": "Check memory system health and alert on issues",
        "type": "conditional",
        "frequency": "every_6_hours",
        "enabled": True,
        "condition": {
            "type": "simple_task",
            "execution_mode": "direct",
            "tool_name": "lt_memory",
            "tool_params": {
                "operation": "get_memory_stats"
            },
            "evaluate": "result.get('health', {}).get('status') != 'healthy'"
        },
        "action": {
            "type": "simple_task",
            "execution_mode": "orchestrated",
            "task_description": "The memory system health check found issues. Analyze the health report and take corrective action if possible.",
            "available_tools": ["lt_memory"]
        },
        "metadata": {
            "category": "memory_monitoring",
            "priority": "high"
        }
    },
    {
        "name": "Entity Relationship Discovery",
        "description": "Discover new relationships between entities",
        "type": "simple_task",
        "execution_mode": "orchestrated",
        "frequency": "daily",
        "scheduled_time": "04:00",
        "enabled": False,  # Disabled by default for performance
        "task_description": """Analyze recently discovered entities and:

1. Search for entities that frequently appear together
2. Identify potential relationships between entities
3. Look for entity disambiguation opportunities (same entity with different names)
4. Update entity importance based on recent activity

Focus on high-value relationships that provide context for understanding user's world.""",
        "available_tools": ["lt_memory"],
        "metadata": {
            "category": "memory_enrichment",
            "priority": "low"
        }
    }
]


def register_memory_automations(automation_controller):
    """
    Register all memory automations with the automation controller.
    
    Args:
        automation_controller: The automation controller instance
        
    Returns:
        Dict mapping automation names to their IDs
    """
    registered = {}
    
    for automation in MEMORY_AUTOMATIONS:
        try:
            # Check if automation already exists
            existing = automation_controller.get_automation(automation["name"])
            if existing:
                logger.info(f"Memory automation already exists: {automation['name']}")
                registered[automation["name"]] = existing.get("id")
                continue
            
            # Create new automation
            result = automation_controller.create_automation(automation)
            if result.get("success"):
                automation_id = result.get("automation_id")
                registered[automation["name"]] = automation_id
                logger.info(
                    f"Registered memory automation: {automation['name']} "
                    f"(ID: {automation_id})"
                )
            else:
                logger.error(
                    f"Failed to register automation {automation['name']}: "
                    f"{result.get('error')}"
                )
                
        except Exception as e:
            logger.error(
                f"Error registering automation {automation['name']}: {e}"
            )
    
    logger.info(
        f"Memory automation registration complete. "
        f"Registered {len(registered)} automations."
    )
    
    return registered


def get_memory_automation_status(automation_controller) -> dict:
    """
    Get status of all memory automations.
    
    Args:
        automation_controller: The automation controller instance
        
    Returns:
        Dict with automation statuses
    """
    status = {}
    
    for automation in MEMORY_AUTOMATIONS:
        name = automation["name"]
        try:
            auto = automation_controller.get_automation(name)
            if auto:
                status[name] = {
                    "exists": True,
                    "enabled": auto.get("enabled", False),
                    "last_run": auto.get("last_run"),
                    "next_run": auto.get("next_run"),
                    "error_count": auto.get("error_count", 0)
                }
            else:
                status[name] = {
                    "exists": False,
                    "enabled": False
                }
        except Exception as e:
            status[name] = {
                "exists": False,
                "error": str(e)
            }
    
    return status