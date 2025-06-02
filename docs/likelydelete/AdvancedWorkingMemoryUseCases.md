# Advanced Working Memory Use Cases

This document explores advanced applications of the `WorkingMemory` system beyond basic prompt management.

## 1. Dynamic Response Adaptation

Automatically adapt response style based on user preferences detected during conversation.

```python
class ResponseAdaptationManager:
    def __init__(self, working_memory):
        self.working_memory = working_memory
        self._adaptation_id = None
        self._preferences = {
            "verbosity": "medium",      # low, medium, high
            "technical_depth": "medium", # low, medium, high
            "examples": "occasional",   # none, occasional, frequent
            "tone": "neutral"           # formal, neutral, casual
        }
    
    def record_feedback(self, response_type, success_indicator):
        """Record if certain response characteristics worked well"""
        # Example: record_feedback("concise_with_examples", True)
        
        if self._adaptation_id:
            self.working_memory.remove(self._adaptation_id)
            
        # Adjust preferences based on feedback
        if "concise" in response_type and success_indicator:
            self._preferences["verbosity"] = "low"
        if "detailed" in response_type and success_indicator:
            self._preferences["verbosity"] = "high"
        if "examples" in response_type and success_indicator:
            self._preferences["examples"] = "frequent"
            
        adaptation_content = f"""# Response Adaptation Guidance
- Preferred verbosity: {self._preferences['verbosity']}
- Technical detail level: {self._preferences['technical_depth']}
- Example frequency: {self._preferences['examples']}
- Communication tone: {self._preferences['tone']}
"""
        
        self._adaptation_id = self.working_memory.add(
            content=adaptation_content,
            category="response_adaptation"
        )
```

### Benefits

- Learns user preferences automatically during conversation
- Creates a more personalized experience
- Improves response usefulness without explicit configuration
- More efficient communication as users get their preferred style

## 2. Debug/Introspection Mode

Expose working memory contents for debugging and transparency.

```python
class WorkingMemory:
    # Add to existing implementation
    
    def get_memory_snapshot(self, formatted=False):
        """Return a snapshot of current working memory for debugging"""
        if not formatted:
            return self._memory_items
            
        snapshot = "# Working Memory Contents\n\n"
        
        # Group by category
        categories = {}
        for item_id, data in self._memory_items.items():
            category = data["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append((item_id, data["content"]))
            
        # Format by category
        for category, items in categories.items():
            snapshot += f"## {category.title()} ({len(items)} items)\n"
            for item_id, content in items:
                # Truncate content if too long
                display_content = content[:100] + "..." if len(content) > 100 else content
                # Replace newlines with spaces for compact display
                display_content = display_content.replace("\n", " ")
                snapshot += f"- {item_id[:8]}: {display_content}\n"
            snapshot += "\n"
            
        return snapshot
    
    def introspect_command(self, command, params=None):
        """Process debug commands for memory introspection"""
        if command == "snapshot":
            return self.get_memory_snapshot(formatted=True)
        elif command == "show_category" and params:
            # Filter snapshot to specific category
            return self._filter_by_category(params)
        elif command == "memory_stats":
            # Show statistics about memory usage
            return self._generate_memory_stats()
        else:
            return "Unknown introspection command"
            
    def _filter_by_category(self, category):
        """Filter memory contents by category"""
        items = [(id, item) for id, item in self._memory_items.items() 
                if item["category"] == category]
        
        if not items:
            return f"No items found in category '{category}'"
            
        result = f"# Items in category '{category}':\n\n"
        for item_id, item in items:
            result += f"## Item {item_id[:8]}:\n{item['content']}\n\n"
        return result
        
    def _generate_memory_stats(self):
        """Generate statistics about memory usage"""
        if not self._memory_items:
            return "Memory is empty"
            
        total_items = len(self._memory_items)
        categories = {}
        total_size = 0
        
        for item in self._memory_items.values():
            cat = item["category"]
            size = len(item["content"])
            total_size += size
            
            if cat not in categories:
                categories[cat] = {"count": 0, "size": 0}
            categories[cat]["count"] += 1
            categories[cat]["size"] += size
            
        stats = f"# Memory Statistics\n\n"
        stats += f"Total items: {total_items}\n"
        stats += f"Total content size: {total_size} characters\n\n"
        stats += "## Categories:\n"
        
        for cat, data in categories.items():
            avg_size = data["size"] / data["count"]
            stats += f"- {cat}: {data['count']} items, {data['size']} chars (avg: {avg_size:.1f})\n"
            
        return stats
```

### Benefits

- Helps developers understand what's in memory
- Enables transparent debugging
- Creates a self-diagnostic capability
- Assists in optimizing memory usage
- Special commands like `/memory snapshot` make introspection easy

## 3. Embedded Knowledge Graphs

Build and maintain relationship maps between entities discussed in conversation.

```python
class KnowledgeGraph:
    def __init__(self, working_memory):
        self.working_memory = working_memory
        self._graph_id = None
        self.entities = {}
        self.relationships = []
        
    def add_entity(self, entity_id, entity_type, properties=None):
        """Add an entity to the knowledge graph"""
        if properties is None:
            properties = {}
            
        self.entities[entity_id] = {
            "type": entity_type,
            "properties": properties
        }
        self._update_graph_in_memory()
        
    def add_relationship(self, source_id, relation_type, target_id, properties=None):
        """Add a relationship between entities"""
        if properties is None:
            properties = {}
            
        # Ensure both entities exist
        if source_id not in self.entities or target_id not in self.entities:
            return False
            
        self.relationships.append({
            "source": source_id,
            "type": relation_type,
            "target": target_id,
            "properties": properties
        })
        self._update_graph_in_memory()
        return True
        
    def query_relationships(self, entity_id):
        """Find all relationships involving an entity"""
        if entity_id not in self.entities:
            return []
            
        results = []
        for rel in self.relationships:
            if rel["source"] == entity_id:
                target = self.entities[rel["target"]]
                results.append(f"{entity_id} {rel['type']} {rel['target']} ({target['type']})")
            elif rel["target"] == entity_id:
                source = self.entities[rel["source"]]
                results.append(f"{rel['source']} ({source['type']}) {rel['type']} {entity_id}")
        
        return results
        
    def _update_graph_in_memory(self):
        """Update the knowledge graph in working memory"""
        if self._graph_id:
            self.working_memory.remove(self._graph_id)
            
        # Generate text representation of the graph
        graph_content = "# Knowledge Graph\n\n"
        
        # List key entities
        graph_content += "## Entities\n"
        for entity_id, data in self.entities.items():
            props_str = ", ".join(f"{k}: {v}" for k, v in data["properties"].items())
            graph_content += f"- {entity_id} ({data['type']}): {props_str}\n"
            
        # List important relationships
        graph_content += "\n## Key Relationships\n"
        for rel in self.relationships:
            source_type = self.entities[rel["source"]]["type"]
            target_type = self.entities[rel["target"]]["type"]
            graph_content += f"- {rel['source']} ({source_type}) {rel['type']} {rel['target']} ({target_type})\n"
            
        self._graph_id = self.working_memory.add(
            content=graph_content,
            category="knowledge_graph"
        )
        
    def extract_project_entities(self, content):
        """Analyze content to extract project entities automatically"""
        # This would use NLP techniques to identify entities and relationships
        # Simplified example:
        
        # Sample patterns for entity detection
        class_pattern = r"class\s+(\w+)"
        function_pattern = r"def\s+(\w+)"
        module_pattern = r"import\s+(\w+)"
        
        # Extract and add entities
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            self.add_entity(class_name, "class")
            
        for match in re.finditer(function_pattern, content):
            function_name = match.group(1)
            self.add_entity(function_name, "function")
            
        # Infer relationships (simplified)
        # e.g., if a function name appears inside a class, it's a method of that class
        # This would be much more sophisticated in practice
```

### Benefits

- Maintains a coherent model of project structures
- Enables more contextually-aware responses
- Helps identify relationships between different parts of a codebase
- Creates persistent understanding of project domains
- Automatically builds up knowledge through conversation

## Integration Example

Here's how these advanced features could be integrated into the response generation pipeline:

```python
# In main.py
working_memory = WorkingMemory()

# Initialize additional components
response_adapter = ResponseAdaptationManager(working_memory)
knowledge_graph = KnowledgeGraph(working_memory)

# In response generation pipeline
def generate_response(user_message):
    # Record feedback when user expresses satisfaction
    if "thanks" in user_message.lower() or "that's helpful" in user_message.lower():
        response_adapter.record_feedback("last_response_style", True)
    
    # Update knowledge graph when discussing code
    if is_code_discussion(user_message):
        knowledge_graph.extract_project_entities(user_message)
    
    # Process introspection commands
    if user_message.startswith("/memory"):
        # Example: "/memory snapshot" or "/memory show_category response_adaptation"
        parts = user_message.split()
        command = parts[1] if len(parts) > 1 else "snapshot"
        params = parts[2] if len(parts) > 2 else None
        return working_memory.introspect_command(command, params)
    
    # Regular response generation using working memory content
    return regular_response_generation(user_message)
```

## 4. Superuser Mode

Provide a conversational interface for advanced system control and debugging.

```python
class SuperuserMode:
    def __init__(self, working_memory):
        self.working_memory = working_memory
        self._superuser_id = None
        self._authorized = False
        self._command_prefix = "@Mira"
        self._superuser_password = "your_secure_password"  # In practice, use env vars or secure storage

    def process_command(self, message):
        """Process potential superuser commands"""
        if not message.startswith(self._command_prefix):
            return None

        # Extract command parts
        parts = message.split(" ", 2)
        command = parts[1].lower() if len(parts) > 1 else "help"
        args = parts[2] if len(parts) > 2 else ""

        # Check authorization
        if not self._authorized and command != "unlock":
            return "Please unlock superuser mode first with '@Mira unlock <password>'"

        # Handle commands
        if command == "unlock":
            return self._handle_unlock(args)
        elif command == "status":
            return self._handle_status()
        elif command == "memory":
            return self._handle_memory()
        elif command == "clear":
            return self._handle_clear(args)
        elif command == "stats":
            return self._handle_stats()
        elif command == "help":
            return self._handle_help()
        else:
            return f"Unknown superuser command: {command}"

    def _handle_unlock(self, password):
        """Handle unlock command"""
        if password == self._superuser_password:
            self._authorized = True

            # Add superuser flag to memory
            if self._superuser_id:
                self.working_memory.remove(self._superuser_id)

            self._superuser_id = self.working_memory.add(
                content="# SUPERUSER MODE ACTIVE\nAdvanced system commands are enabled.",
                category="superuser_active"
            )

            return "Superuser mode activated. Welcome, administrator."
        else:
            return "Invalid password. Access denied."

    def _handle_status(self):
        """Handle status command"""
        status = "Superuser mode is currently "
        status += "ACTIVE" if self._authorized else "INACTIVE"

        if self._authorized:
            status += "\nThe following commands are available:\n"
            status += "- @Mira memory: Show memory contents\n"
            status += "- @Mira clear <category>: Clear memory category\n"
            status += "- @Mira stats: Show memory statistics\n"
            status += "- @Mira help: Show this help message"

        return status

    def _handle_memory(self):
        """Show memory snapshot"""
        # Leverage existing introspection methods
        snapshot = self.working_memory.get_memory_snapshot(formatted=True)
        return snapshot

    def _handle_clear(self, category):
        """Clear a memory category"""
        if not category:
            return "Please specify a category to clear"

        count = self.working_memory.remove_by_category(category)
        return f"Cleared {count} items from category '{category}'"

    def _handle_stats(self):
        """Show memory statistics"""
        return self.working_memory.introspect_command("memory_stats")

    def _handle_help(self):
        """Show help information"""
        help_text = "# Superuser Command Reference\n\n"
        help_text += "## Available Commands\n"
        help_text += "- @Mira status: Check superuser status\n"
        help_text += "- @Mira memory: Show memory contents\n"
        help_text += "- @Mira clear <category>: Clear memory category\n"
        help_text += "- @Mira stats: Show memory statistics\n"
        help_text += "- @Mira help: Show this help message\n"

        return help_text
```

### Integration with Conversation Flow

```python
def process_message(self, user_message):
    """Pre-process message before normal handling"""

    # Check for superuser commands first
    superuser_response = self.superuser_mode.process_command(user_message)
    if superuser_response:
        return superuser_response

    # Otherwise process normally
    return self.regular_message_processing(user_message)
```

### Benefits

- Natural conversational interface for system management
- Secure access to advanced functionality
- Transparent view into system memory and state
- Ability to debug and modify system behavior at runtime
- Integration with existing introspection capabilities
- Hidden from regular users but easily accessible for administrators

## Conclusion

These advanced use cases transform `WorkingMemory` from a simple prompt management tool into a sophisticated system that can:

1. Adapt to user preferences automatically
2. Enable transparent debugging and introspection
3. Build and maintain knowledge about entities and their relationships
4. Provide secure administrative access through natural conversation

By implementing these features, the assistant becomes more personalized, debuggable, and contextually aware, significantly enhancing the user experience beyond what's possible with standard conversational approaches.