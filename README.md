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

### Terminal Mode

Run the interactive CLI:

```bash
python main.py
```

#### Command Line Options

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

#### Interactive Commands

Once running, you can use these commands:
- `/exit` - End the session and save conversation
- `/save` - Save the current conversation
- `/clear` - Clear the conversation history

### Web API Mode

Run as a Flask web server:

```bash
# Basic HTTP server on port 443
python app.py

# HTTP server on custom port
python app.py --port 8080

# HTTPS server with SSL
python app.py --ssl --cert /path/to/cert.pem --key /path/to/key.pem
```

This starts a Flask server that you can interact with through REST API endpoints. By default, it runs on port 443 but can be configured with command line options.

#### API Authentication

The API uses a simple, direct API key authentication to secure all endpoints.

1. **Setting the API Key**:
   - Set the `IOS_APP_API_KEY` environment variable or in your config file
   - The server will generate a random key if none is provided (check console output)

2. **Authenticating Requests**:
   - All API calls (except `/api/health`) require the API key
   - Include the API key in the `Authorization` header: `Authorization: Bearer your-api-key-here`
   - The API key should be securely stored in your iOS app

3. **Security Notes**:
   - The API key is a static value and should be treated as a secret
   - In production, consider additional security measures like certificate pinning
   - For debugging purposes, the health endpoint does not require authentication

#### API Endpoints

- **POST /api/chat** - Send a message and get a response
  - Auth Required: Yes
  - Request: `{"message": "Hello", "conversation_id": "optional-id"}`
  - Response: 
    ```json
    {
      "conversation_id": "unique-id",
      "response": "Bot response text",
      "messages": [...full conversation history...],
      "timestamp": 1713267583
    }
    ```

- **GET /api/conversation/{conversation_id}** - Get conversation history
  - Auth Required: Yes
  - Response: 
    ```json
    {
      "conversation_id": "unique-id",
      "messages": [...],
      "exists": true,
      "timestamp": 1713267583
    }
    ```

- **DELETE /api/conversation/{conversation_id}** - Clear conversation history
  - Auth Required: Yes
  - Response: 
    ```json
    {
      "status": "success", 
      "message": "Conversation cleared",
      "conversation_id": "unique-id",
      "timestamp": 1713267583
    }
    ```

- **GET /api/health** - Health check endpoint
  - Auth Required: No
  - Response: 
    ```json
    {
      "status": "ok",
      "version": "1.0.0",
      "timestamp": 1713267583
    }
    ```


#### Conversation Flow

1. **Starting a new conversation**: 
   - Make a POST request to `/api/chat` without a conversation_id
   - The API will create a new conversation and return its ID
   - Store this ID in your client application for future requests

2. **Continuing a conversation**:
   - Include the conversation_id in all subsequent requests
   - The API maintains conversation state between requests

3. **Handling client restarts**:
   - If your client application restarts, it can retrieve the conversation state by:
     - Using a previously saved conversation_id with GET `/api/conversation/{conversation_id}`
     - Or starting a new conversation if needed

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
- **Calendar Tool**: Manage calendar events and appointments
- **Email Tool**: Send and receive emails
- **HTTP Tool**: Make HTTP requests to external services
- **Maps Tool**: Get location and navigation information
- **Kasa Tool**: Control smart home devices

#### Automatic Tool Classifier Examples

The system automatically generates training examples for tool discovery when a tool doesn't have a manually created `classifier_examples.json` file:

- The tool relevance engine checks for classifier examples
- If no examples exist, it generates synthetic examples using LLM-based analysis
- Synthetic examples are saved as `autogen_classifier_examples.json`
- The system still works with manually created examples (preferred) or auto-generated ones

## Architecture

BotWithMemory follows a modular architecture:

- **Main Process**: Handles user interaction and conversation management
- **Background Service**: Processes asynchronous tasks independently
- **LLM Bridge**: Provides a unified interface to the Claude AI model
- **Tool Repository**: Manages discovery and execution of specialized tools
- **Configuration System**: Centralizes configuration from multiple sources

The system uses a file-based persistence mechanism for conversations, background tasks, and tool data storage, enabling persistent memory across sessions.