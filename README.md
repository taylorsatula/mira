# MIRA: Just Talk Normal

MIRA is an AI agent system that provides natural conversation capabilities with persistent memory, intelligent tool usage, and workflow automation. It combines the conversational abilities of modern language models with practical tools and a sophisticated memory system to create a genuinely useful AI assistant.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Memory System](#memory-system)
- [Tool System](#tool-system)
- [Available Tools](#available-tools)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **Natural Conversation**: Talk to MIRA like you would a human assistant - no special commands or syntax required
- **Continuous Memory**: Maintains one long continuous conversation with daily summarization and context management
- **Intelligent Tool Usage**: Automatically selects and uses appropriate tools based on conversation context
- **Workflow Automation**: Handles multi-step tasks with context-aware workflows
- **Provider Flexibility**: Works with OpenAI, Anthropic (via proxy), or local models through Ollama
- **Extensible Tool System**: Easy to add new capabilities through the modular tool framework

## Architecture

MIRA consists of several key components:

- **Conversation Manager**: Handles message flow, tool execution, and response generation
- **Working Memory**: Manages dynamic context visible throughout conversations
- **Long-Term Memory**: PostgreSQL-based system for persistent knowledge storage with semantic search
- **Tool System**: Modular framework for integrating external services and capabilities

## Installation

MIRA includes an automated setup script that handles all configuration and dependencies:

```bash
git clone https://github.com/yourusername/mira.git
cd mira
chmod +x autodeploy.sh
./autodeploy.sh
```

The script will:
- Set up Python virtual environment
- Install all dependencies
- Configure PostgreSQL with pgvector
- Create required databases
- Generate configuration files
- Initialize the system

Manual installation instructions are available in `docs/MANUAL_SETUP.md`.

## Usage

Start MIRA in interactive mode:

```bash
python main.py
```

Available commands:
- `/exit` - Exit the program
- `/save` - Save conversation
- `/clear` - Clear conversation history
- `/reload_user` - Reload user profile
- `/tokens` - Show token usage
- `/toolfeedback` - Provide tool feedback

## Memory System

MIRA's memory system consists of two complementary components that work together to provide both immediate context and long-term knowledge retention.

### Working Memory

Working memory serves as the dynamic context system that's active throughout every conversation. It functions as a live dashboard that updates automatically before each response, ensuring MIRA always has current information available.

**Key Components:**

- **Content Categories**: Organizes information into logical groups (reminders, datetime, system_status, proactive_memories, archived_conversations)
- **Manager System**: Registered components automatically update their content before each LLM call
- **Trinket Architecture**: Specialized utility classes handle specific types of dynamic content:
  - `TimeManager`: Provides current date/time context
  - `ReminderManager`: Surfaces overdue and upcoming reminders
  - `SystemStatusManager`: Reports system health and notices
  - `ProactiveMemoryTrinket`: Intelligently surfaces relevant long-term memories
  - `ConversationArchiveManager`: Injects relevant historical conversations

**How It Works:**
1. Before generating each response, all registered managers update their content
2. The system builds a formatted context block that's included in the LLM prompt
3. Content is organized by category with clear formatting for easy LLM consumption
4. Items are tracked with UUIDs and can be dynamically added/removed during conversation

### Long-Term Memory

Long-term memory provides persistent knowledge storage with sophisticated retrieval capabilities. It uses PostgreSQL with pgvector for semantic similarity search, enabling MIRA to find and utilize relevant information from past conversations and learned knowledge.

**Core Components:**

**Memory Blocks (Core Memory)**
- Always-visible context that MIRA can self-edit
- Three categories: `persona` (MIRA's personality), `human` (user information), `system` (operational context)
- Version tracking with differential storage for change history
- Character limits ensure focused, relevant content (2048 for persona/human, 1024 for system)

**Memory Passages (Archival Memory)**
- Searchable long-term memories with vector embeddings
- Sources include conversations, documents, and automation systems
- Importance scoring for relevance weighting
- Access tracking to identify frequently used information
- Optional expiration dates for temporal facts
- Human verification flags for trusted information

**Archived Conversations**
- Complete conversation history organized by date
- Pre-generated summaries at multiple time scales (daily, weekly, monthly)
- Efficient temporal indexing for quick retrieval
- Integration with working memory for contextual access

**Memory Snapshots**
- Point-in-time captures of entire memory state
- Used for recovery, auditing, and debugging
- Enables rollback to previous memory states if needed

**Semantic Search Capabilities:**
- Uses OpenAI's text-embedding-3-small model (1024 dimensions)
- pgvector's IVFFlat indexing for efficient similarity search
- Configurable similarity thresholds (default 0.6 for proactive surfacing)
- Multiple filter options (source, date range, importance level)
- Weighted context building that prioritizes recent conversation content

**Integration Between Systems:**
The two memory systems work together through:
- `MemoryBridge`: Automatically injects core memory blocks into working memory
- `ProactiveMemoryTrinket`: Uses semantic search to surface relevant memories based on conversation context
- `ConversationArchiveManager`: Provides access to historical conversations when contextually relevant

This architecture enables MIRA to maintain both immediate conversational awareness and accumulated knowledge over time, creating a truly continuous interaction experience.

## Tool System

MIRA's tool system is designed for both intelligent automation and developer ease-of-use. The system automatically determines which tools are relevant to a conversation and enables them just-in-time, while providing developers with a streamlined path to add new capabilities.

### Intelligent Tool Management

**Tool Relevance Engine**
MIRA uses a sophisticated relevance engine that analyzes conversation context to determine which tools should be available:

- **Embedding-Based Classification**: Uses semantic similarity to match user messages with tool capabilities
- **Context Persistence**: Keeps relevant tools enabled for several message exchanges after initial activation
- **Topic Change Detection**: Automatically adjusts tool availability when conversation topics shift
- **Training Data**: Uses both manual examples and automatically generated synthetic examples for each tool

**Automatic Training**
When a new tool is added:
1. MIRA examines the tool's description and parameters
2. Generates synthetic conversation examples that would require the tool
3. Trains its relevance classifier to recognize similar patterns
4. The tool becomes available for automatic activation in future conversations

### Developer-Friendly Tool Creation

The tool system follows a consistent pattern that makes it easy to add new capabilities using natural language with AI assistance:

**Standardized Tool Pattern**
All tools inherit from a base `Tool` class and implement a simple interface:

```python
class MyTool(Tool):
    name = "my_tool"
    description = "What the tool does"
    
    def run(self, **params):
        # Tool implementation
        return {"result": "success"}
```

**Natural Language Development Workflow**
1. Load MIRA's codebase into an AI assistant like Claude Code
2. Examine existing tools in the `tools/` directory for patterns
3. Reference the included guide for best practices
4. Describe the desired functionality in natural language
5. The AI assistant generates a complete tool implementation following MIRA's patterns
6. Drop the completed tool file into the `tools/` directory
7. MIRA automatically discovers, loads, and trains itself to use the new tool

**Automatic Integration**
- **Discovery**: Tools are automatically discovered and loaded from the `tools/` directory
- **Configuration**: Tools can register their own configuration schemas
- **Error Handling**: Consistent error handling with recovery guidance
- **Documentation**: Self-documenting through description and parameter schemas
- **Testing**: Standard testing patterns ensure reliability

**Example Development Process**
A developer wanting to add Slack integration might say:
> "I need a tool that can send messages to Slack channels, list channels, and get recent messages from a channel. It should use the Slack API and handle authentication with a bot token."

The AI assistant would then:
1. Examine existing tools like `email_tool.py` for patterns
2. Generate a complete `slack_tool.py` implementation
3. Include proper error handling, configuration management, and OpenAI schema
4. Provide the tool ready to drop into the `tools/` directory

This approach makes MIRA highly extensible while maintaining code quality and consistency across all tools.

## Available Tools

MIRA includes several built-in tools that are automatically enabled based on conversation context:

| Tool | Description | Auto-Discovery |
|------|-------------|----------------|
| **Email** | Send and manage emails | ✓ |
| **Calendar** | Manage calendar events | ✓ |
| **Weather** | Get weather forecasts and heat stress calculations | ✓ |
| **Maps** | Geocoding, place details, and directions | ✓ |
| **Web Access** | Search the web and fetch page content | ✓ |
| **Customer Database** | Manage customer information | ✓ |
| **Kasa** | Control Kasa smart home devices | ✓ |
| **Square API** | Business operations via Square | ✓ |
| **Reminders** | Create and manage reminders | ✓ |
| **Automation** | Set up automated workflows | ✓ |

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black .

# Type checking
mypy .

# Linting
flake8
```

### Project Structure

```
mira/
├── config/           # Configuration management
├── lt_memory/        # Long-term memory system
├── tools/            # Tool implementations
├── tests/            # Test suite
├── conversation.py   # Core conversation logic
├── working_memory.py # Working memory system
└── main.py          # CLI entry point
```

## Troubleshooting

### Common Issues

1. **PostgreSQL Connection Errors**
   - Ensure PostgreSQL is running: `sudo service postgresql status`
   - Verify database exists and user has permissions
   - Check connection string in environment variables

2. **Missing pgvector Extension**
   - Install pgvector for your PostgreSQL version
   - Enable it in the lt_memory database: `CREATE EXTENSION vector;`

3. **Tool Errors**
   - Check tool-specific API keys are set
   - Verify external services are accessible
   - Review logs for detailed error messages

### Debug Mode

Enable debug logging:

```bash
export AGENT_LOG_LEVEL=DEBUG
```

## License

[Your License Here]