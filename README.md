# BotWithMemory

A Python-based AI agent system with persistent memory capabilities, asynchronous task processing, and extensible tool architecture.

## Purpose

BotWithMemory is a framework for building conversational AI agents that can:
- Maintain persistent conversation history across sessions
- Execute tasks in the background while continuing conversations
- Access specialized tools for data extraction, persistent storage, and more
- Adapt to user preferences through memory capabilities

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/botwithmemory.git
   cd botwithmemory
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (create a `.env` file):
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

Run the interactive CLI:

```bash
python main.py
```

### Command Line Options

```bash
# Specify a config file
python main.py --config my_config.json

# Continue previous conversation
python main.py --conversation my_conversation_id

# Set log level
python main.py --log-level DEBUG

# Control response streaming
python main.py --stream      # Enable streaming
python main.py --no-stream   # Disable streaming
```

### Interactive Commands

Once running, you can use these commands:
- `/exit` - End the session and save conversation
- `/save` - Save the current conversation
- `/clear` - Clear the conversation history

## Features

### Core Capabilities

- **Persistent Conversations**: Save and resume conversations with unique IDs
- **Streaming Responses**: See AI responses as they are generated 
- **Background Tasks**: Run long-running operations asynchronously
- **Configurable System**: Configure via environment variables, files, or command line

### Tool Architecture

The system includes several powerful tools:

- **Extraction Tool**: Parse and extract specific data from text
- **Persistence Tool**: Store and retrieve data persistently
- **Questionnaire Tool**: Create and process user questionnaires
- **Repository Tool**: Access and search file repositories
- **Asynchronous Task Tools**: Schedule and monitor background tasks

## Architecture

BotWithMemory follows a modular architecture:

- **Main Process**: Handles user interaction and conversation management
- **Background Service**: Processes asynchronous tasks independently
- **LLM Bridge**: Provides a unified interface to the Claude AI model
- **Tool Repository**: Manages discovery and execution of specialized tools
- **Configuration System**: Centralizes configuration from multiple sources

The system uses a file-based persistence mechanism for conversations, background tasks, and tool data storage, enabling persistent memory across sessions.