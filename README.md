# ~ I CANNOT STRESS TO YOU ENOUGH DEAR READER: ~
## THIS CODEBASE IS IN CONSTANT FLUX RIGHT NOW AND FILES SOMETIMES DEADEND OR IT JUST WON'T LOAD OR I'LL DELETE CORE INFRASTRUCTURE SO I CAN REBUILD IT IN A FUTURE COMMIT. I'M IN MY FAST AND LOOSE ERA. CHECK OUT THE CODE.. ITS PRETTY NEAT. BUT DON'T JUDGE ME ON THE CONTENT CONTAINED WITHIN RIGHT NOW. THIS IS MY METHOD AND IT WORKS EVEN THOUGH ITS VERY WUSHI DRUNKEN FIST STYLE.

# =======================

# MIRA: Just Talk Normal

MIRA is an AI agent system that provides natural conversation capabilities with persistent memory, intelligent tool usage, and workflow automation. It combines the conversational abilities of modern language models with practical tools and a sophisticated memory system to create a genuinely useful long-running AI assistant. My vision for this project is to create a little brain-in-a-box. I will not stop till my vision is realized.

I originally started writing what would become MIRA because I wanted to create a bot that could generate recipes tailored to my personal tastes by following a decision tree. A thousand scope creeps later and now it proactively remembers that next week is your mom's birthday and asks if you'd like it to arrange a gardenia delivery to her work address because that is what you got her last year lol.

**I am distributing MIRA 100% unequivocally free for personal and research use.** Download it, modify it, heck,, if you're feeling kind please contribute to the source. But **please don't use it in commmercial applications without getting a license** from me (I'll probably give you one for free for an interesting usecase tbh) or wait for the license to decay into Apache 2.0. 

Thanks! I hope you find value in this project. 

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

- **Natural Conversation**: Talk to MIRA like you would a human  - no special commands or syntax required
- **Continuous Memory**: Maintains one long continuous conversation with daily summarization and context management (this is a huge issue in many long-running AI agents and I think I mitigated it. TL;DR Each day is a standalone event judged against larger trends. No one day has excessive weight and I've integrated decay at multiple strategic points.)
- **Intelligent Tool Usage**: Automatically selects and uses appropriate tools based on conversation context. This just-in-time approach to adding tool definitions to the context window in conjunction with dynamically updating workflows that only show the current step allows MIRA to reliably execute multi-step tool calls on pittifly low parameter LLMs. This was designed so that MIRA can run totally offline on edge devices. Heck, my testbed for this ability is an old Android phone.
- **Workflow Automation**: Handles multi-step tasks with context-aware workflows
- **Provider Flexibility**: Works with Anthropic (via proxy), Hugging Face, Lambda, any provider that follow the OAI API format, or local models through Ollama. llm_provider.py is provider agnostic. 
- **Extensible Tool System**: Easy to add new capabilities through the modular tool framework (more details below. MIRA includes resources to semi-reliably zero-shot new tool creation)

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

### Embedding Models Configuration

MIRA supports flexible embedding model providers for both tool classification and memory operations:

#### Local BGE Models (Default)
MIRA uses BAAI BGE (Bidirectional Generative Embeddings) models for efficient local inference:
- **BGE-large-en-v1.5**: 1024-dimensional embeddings with INT8 quantization for CPU efficiency
- **BGE-reranker-base**: FP16 precision reranker for improved memory search relevance

These models run entirely on your local machine with ONNX Runtime optimization.
embeddings_provider.py goes hand-in-hand with llm_provider.py wherein you can use any embeddings provider you want including Vertex and OpenAI.
My theory was that though using BGM with Reranker produces awesome results not every computer can handle these tasks so MIRA can seamlessly offload heavy lifting to a remote server.

#### Unified Embeddings Architecture
MIRA implements a novel unified embeddings approach that optimizes both performance and relevance:

**Shared Infrastructure with Specialized Usage:**
- Both tool relevance and memory search use the same `EmbeddingsProvider` infrastructure
- Tool relevance embeds the current user message for immediate tool classification
- Memory search embeds weighted conversation context from recent messages for historical relevance
- Each system optimizes what text it embeds for its specific use case

**Differentiated Processing for Optimal Results:**
- Tool relevance uses efficient matrix operations to compare against all tool embeddings and returns a top set
- Memory search applies the same embedding model but includes an additional reranking step to ensure only truly relevant memories are surfaced
- This dual approach balances speed for tool selection with precision for memory retrieval

**Advantages over Traditional Approaches:**
- **Performance**: Eliminates redundant embedding infrastructure and model loading
- **Consistency**: Ensures consistent embedding quality across all system components  
- **Efficiency**: Shared LRU and disk caching & optimization reduces computational overhead
- **Contextual Precision**: Tool relevance focuses on immediate needs while memory search considers conversational flow
- **Scalability**: Single embedding provider scales to support additional AI subsystems

This architecture enables MIRA to make intelligent decisions about both tool activation and memory retrieval while maintaining optimal performance through shared resources.

#### Remote OpenAI Embeddings
For cloud-based embeddings, configure the provider in your environment:
```bash
# Set embeddings provider to remote
export EMBEDDINGS_PROVIDER=remote
export OAI_EMBEDDINGS_KEY=your-openai-api-key
```
If you choose to use local embeddings (preferred):
The system will automatically download and cache BGE models on first use (~1.2GB for base model, ~500MB for reranker).

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
  - `OtherOnes`: Trinkets follow a factory pattern so new ones can be added quickly and reliably

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
- Three categories: `persona` (MIRA's personality/directive), `human` (user information), `system` (operational context)
- Version tracking with differential storage for change history
- Character limits ensure focused, relevant content (2048 for persona/human, 1024 for system)
- There is also a blind core memory called `scratchpad` that allows MIRA to log notes "It is Friday and the user asked me to provide them a report again again this week. Consider proactively providing that in their morning report on Fridays". MIRA can write to this block but cannot read it. Allowing MIRA to read the block would pollute the context. The contents of `scratchpad` are reviwed nightly during the consolidation routine and cleared for the next day.

**Memory Passages (Archival Memory)**
- Searchable long-term memories with vector embeddings
- Sources include conversations, documents, and automation systems
- Importance scoring for relevance weighting
- Access tracking to identify frequently used information
- Optional expiration dates for temporal facts
- Human verification flags for trusted information
- Time and Retrieval Delay for eventually pruning memories that have faded into obscurity

**Archived Conversations**
- Complete conversation history organized by date
- Pre-generated summaries at multiple time scales (daily, weekly, monthly)
- Efficient temporal indexing for quick retrieval
- Integration with working memory for contextual access
- Uses will be able to inject whole snippets into conversation eventually when I get around to coding this but it is low prioriry right now

**Memory Snapshots**
- Point-in-time captures of entire memory state
- Used for recovery, auditing, and debugging
- Enables rollback to previous memory states if needed

**Semantic Search Capabilities:**
- Uses OpenAI's text-embedding-3-small model (1024 dimensions) or BGM-base (1024 dimensions)
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
- **Training Data**: System automatically creates high-quality synthetic examples for each tool that are verified for diversity and token length before finalization (no human in-the-loop)

**Automatic Training**
When a new tool is added:
1. MIRA examines the tool's description and parameters
2. Generates synthetic conversation examples that would require the tool
3. Trains its relevance classifier to recognize similar patterns
4. The tool becomes available for automatic activation after application is reloaded

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
3. '@' Reference the included guide for best practices
4. Describe the desired functionality in natural language
5. The AI assistant generates a complete tool implementation following MIRA's patterns
6. Drop the completed tool file into the `tools/` directory if Claude doesn't put it there automatically
7. MIRA automatically discovers, loads, and trains itself to use the new tool

**Automatic Integration**
- **Discovery**: Tools are automatically discovered and loaded from the `tools/` directory
- **Configuration**: Tools can register their own configuration schemas and system allows for dependency injection
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

A note from the developer: This rocks. It is amazing. You can imagine a tool and for like 40 cents in API calls later your new tool works with MIRA. I was going to give MIRA the ability to build its own tools with headless Claude Code and maybe that'll be a thing in the future but for now I made it dead simple to add tools relevant to *your needs*.

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

This project is licensed under the BSL License - see the LICENSE file for details.

### Third-Party Licenses

#### BAAI BGE Models
The BGE (BAAI General Embedding) models are licensed under the MIT License:
- BGE-large-en-v1.5: https://huggingface.co/BAAI/bge-large-en-v1.5
- BGE-reranker-base: https://huggingface.co/BAAI/bge-reranker-base

These models are developed by the Beijing Academy of Artificial Intelligence (BAAI) and are freely available for both commercial and non-commercial use under the MIT License.
