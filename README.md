# Mira
## Machine Intellegence Resource Assistant
A Python-based AI agent system with persistent memory capabilities, asynchronous task processing, and extensible tool architecture.

## Purpose

Mira is a framework for building conversational AI agents that can:
- (Autonomously) Maintain persistent knowledge history across sessions
- Use and invoke a whole host of tools
  - MIRA can zero-shot generate tools by dropping an open-source API into a special foloder and running a 'create tool' command! full system integration on next startup. zero oversight.
- Adapt to user preferences through memory capabilities
- Generate and curate its own training data
- interact with local SQLite databases
- Do self-directed async tasks (both cron-style and directive-style)


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mira.git
   cd mira
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
- `/reload_user` - Reload user information
- `/tokens` - Show token usage counts
- `/toolfeedback [feedback]` - Save feedback about tool activation with LLM analysis to improve tool classification

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
- **The Ability to Fashion Its Own Tools**: This one is the kicker. The Reminders tool was created for $0.98 in API tokens and a one line prompt describing what I wanted out of the tool. No code edits, no training data, no creating directories or configs,,. just speak a tool into existance (you should review the code eventually tho lol). It returned a well-formed SQLite integrated tool with Read/Write/Update/Delete/Manage. Proper documentation and all. 
- **Tool Feedback Collection**: Capture user feedback about tool activations with AI-powered analysis to improve classification accuracy. The system analyzes feedback, examines related training examples, and provides technical insights for improving tool selection.
- **Configurable System**: Configure via environment variables, files, or command line

### Tool Architecture

The system includes several powerful tools:

- **Reminders Tool**: Create intellegent reminders using natural language
- **Calendar Tool**: Manage calendar events and appointments
- **Email Tool**: Send and receive emails, process inboxes, summerize, flag, move.
- **Maps Tool**: Get location information using unstructured spacial queries
- **Questionnaire Tool**: Create and process questionnaires for gathering information.
- **HTTP Tool**: Make HTTP requests to external services

- **Kasa Tool**: Control smart home devices
- **Square Tool**: Interact with the full Square API
- **Customer Database Tool** Interact with a local SQLite database of customer imformation. Use this data within other tools.

#### Automatic Tool Classifier Examples

The system automatically generates training examples for tool discovery when a tool doesn't have a manually created `classifier_examples.json` file:

- The tool relevance engine checks for classifier examples
- If no examples exist, it generates synthetic examples using LLM-based analysis
- Synthetic examples are saved as `autogen_classifier_examples.json`
- The new classifier data triggers a relearn and the tool integrates into convo
- The system still works with manually created examples (preferred) or auto-generated ones

#### Tool Classification Feedback and Analysis

The system includes an AI-powered feedback mechanism to improve tool classification over time:

- Users can provide feedback on tool activations using the `/toolfeedback` command
- The system captures context including recent messages and active tools
- An LLM analyzes the feedback and provides technical insights including:
  - Root cause analysis of classification issues
  - Pattern recognition in training examples vs. user requests
  - Specific suggestions for improving classification accuracy
  - Implementation plans for addressing the issue
- Feedback and analysis are stored for future tool training improvements
- This helps refine the classification system through user interaction

## Architecture

Mira follows a modular architecture:

- **Main Process**: Handles user interaction and conversation management
- **LLM Bridge**: Provides a unified interface to the Anthropic API
- **Tool Repository**: Manages discovery and execution of specialized tools
- **ToolRelevanceEngine**: Manages (add/remove) the tools in the context window autonomously
- **Tool Feedback System**: Captures and stores user feedback about tool activations with contextual information
- **Configuration System**: Centralizes configuration from multiple sources

The system uses a file-based persistence mechanism for conversationn tool data storage, enabling persistent memory across sessions.

## Neat Stuff It Can Do
- "Mira, whats the email address of the customer I just drove by?"
- "Mira, which customer lives by Southerland Photo south of the hospital?"
- "Mira, next week I'll need you to hound me every day till I initiate the return of those eBay items. The return codes are in an email in the Orders mailbox."
- "Mira, please summerize all my notification emails from this week and show me emails from real people that came in today"
- "Mira, set aside the email from John till we're done going through the others,, I want to get back to him but we've gotta work through this inbox first. Are any of the remaining emails particularly important or time-sensitive?"

## It blows my mind that I've built it in a way where all of these requests reliably complete using a cpu-classifier and Haiku.
