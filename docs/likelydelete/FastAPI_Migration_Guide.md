# FastAPI Migration Guide: Multi-User Support with Anthropic & Ollama

This guide outlines a complete migration plan from the existing Flask server to a FastAPI implementation that supports both Anthropic API and local Ollama, with multi-user capabilities and improved scalability.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Implementation Steps](#implementation-steps)
4. [Core Components](#core-components)
5. [Configuration Changes](#configuration-changes)
6. [Testing & Validation](#testing--validation)
7. [Deployment](#deployment)
8. [Performance Considerations](#performance-considerations)
9. [Appendix: Example Files](#appendix-example-files)

## Architecture Overview

The updated architecture maintains dual provider support through abstraction layers:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   FastAPI API   │────▶│  LLM Providers   │────▶│  Core System    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        ▼                       ▼                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Authentication  │     │ Anthropic Bridge │     │  Conversation   │
│ & Session Mgmt  │     │  Ollama Bridge   │     │     Handler     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        │                       ▼                        │
        │              ┌──────────────────┐             │
        └─────────────▶│  Request Queue   │◀────────────┘
                       │  (Ollama only)   │
                       └──────────────────┘
```

Key improvements:
- Asynchronous request handling with FastAPI
- Multi-user session management
- Prioritized request queue for Ollama
- Streaming support for both providers
- Simplified API endpoints with automatic documentation

## Prerequisites

Before beginning migration, ensure these dependencies are available:

```
fastapi>=0.95.0
uvicorn[standard]>=0.21.0
sse-starlette>=1.6.1
python-multipart>=0.0.5
```

## Implementation Steps

### 1. Create FastAPI Application Structure

Create the following files:
- `/api/fastapi_app.py` - Main FastAPI application
- `/run_api.py` - FastAPI launcher script

### 2. Implement Request Models

Define Pydantic models for request/response validation:

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    priority: Optional[int] = Field(5, ge=1, le=10)  # 1 is highest

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    request_id: Optional[str] = None
```

### 3. Implement FastAPI Endpoints

Create the core API endpoints:
- `/health` - Status check
- `/api/chat` - Standard chat endpoint
- `/api/chat/stream` - Streaming endpoint
- `/api/status` - Server status
- `/api/queue/status` - Check queue position (Ollama)
- `/api/queue/cancel` - Cancel pending request (Ollama)

### 4. Implement User Session Management

Add user session tracking:

```python
# User session tracking
user_sessions = {}

# Track and retrieve user sessions
async def get_user_session(user_id: str) -> str:
    session_key = f"user:{user_id}"
    if session_key in user_sessions:
        return user_sessions[session_key]
    
    # Create new session ID
    session_id = str(uuid.uuid4())
    user_sessions[session_key] = session_id
    return session_id
```

### 5. Implement Authentication

Maintain the existing token-based authentication:

```python
# Security scheme
security = HTTPBearer()

# Define token authentication dependency (optional)
async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    api_token = os.environ.get('IOS_APITOKEN')
    if not api_token:
        return None  # No authentication required if no token configured
        
    if not credentials or credentials.credentials != api_token:
        raise HTTPException(status_code=401, detail="Invalid API token")
    return credentials.credentials
```

### 6. Implement Streaming Support

Add Server-Sent Events streaming support:

```python
from sse_starlette.sse import EventSourceResponse

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, token: Optional[str] = Depends(verify_token)):
    """Streaming chat endpoint using Server-Sent Events."""
    async def event_generator():
        # Stream tokens logic
        yield {"event": "message", "data": token}
        
    return EventSourceResponse(event_generator())
```

## Core Components

### FastAPI Application (api/fastapi_app.py)

The main FastAPI application should:
1. Initialize the system components (as in main.py)
2. Create FastAPI routes and handlers
3. Implement authentication and user sessions
4. Handle both providers via the bridge pattern
5. Support streaming responses
6. Include queue management for Ollama

### Launcher Script (run_api.py)

The launcher should configure and start the FastAPI application:

```python
import uvicorn
from api.fastapi_app import create_app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=2)
```

## Configuration Changes

Update config.py to include FastAPI specific settings:

```yaml
# config/config.py
# API server configuration
api_server:
  host: "0.0.0.0"
  port: 8000
  workers: 2
  log_level: "info"
  enable_cors: true
  cors_origins: ["*"]
  request_timeout: 300
```

## Testing & Validation

Test the new implementation with these scenarios:

1. **Basic Functionality**
   - Make regular API calls to /api/chat
   - Test command handling (/clear, /reload_user, etc.)
   - Test conversation loading and saving

2. **Provider Switching**
   - Test with Anthropic API configuration
   - Test with Ollama configuration
   - Verify tool calls work with both providers

3. **Multi-user Support**
   - Test concurrent requests with different user IDs
   - Test conversation persistence across sessions
   - Test priority settings in the queue

4. **Streaming Support**
   - Test streaming with both providers
   - Test cancellation during streaming
   - Measure latency for first token

5. **Error Handling**
   - Test invalid requests
   - Test API timeouts
   - Test authentication failures
   - Test queue limits

## Deployment

### System Requirements

For production deployment:
- 2+ CPU cores
- 4+ GB RAM
- Sufficient disk space for conversation history
- Docker (recommended) or Python 3.8+

### Docker Deployment

Create a Dockerfile for containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run_api.py"]
```

### Systemd Service (Linux)

For direct server deployment:

```ini
[Unit]
Description=FastAPI AI Agent
After=network.target

[Service]
User=appuser
WorkingDirectory=/path/to/app
ExecStart=/path/to/venv/bin/python run_api.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Performance Considerations

For optimal performance:
- Use multiple uvicorn workers (# of cores + 1)
- Implement request queue tuning for Ollama
- Consider adding API response caching
- Monitor memory usage with larger conversation histories
- Implement proper token tracking for both providers
- Use timeout settings appropriate for your hardware

## Appendix: Example Files

### FastAPI Application (api/fastapi_app.py)

```python
"""
FastAPI application for multi-user AI agent system with dual LLM provider support.
"""
import json
import logging
import os
import uuid
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Project imports
from main import initialize_system, parse_arguments, save_conversation
from conversation import Conversation
from errors import error_context, AgentError, handle_error, ErrorCode
from config import config

# Create FastAPI app
app = create_app()

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    logger = logging.getLogger("fastapi_app")
    app = FastAPI(
        title="Multi-User AI Agent API", 
        description="API for the AI agent system with multi-user support"
    )
    
    # CORS configuration
    if getattr(config.api_server, "enable_cors", True):
        origins = getattr(config.api_server, "cors_origins", ["*"])
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Initialize the system
    args = parse_arguments()
    system = initialize_system(args)
    
    # Get the request queue for Ollama
    request_queue = None
    if hasattr(system['llm_bridge'], 'request_queue'):
        request_queue = system['llm_bridge'].request_queue
        logger.info("Request queue available for multi-user support")
    
    # User session tracking
    user_sessions = {}
    
    # Security scheme
    security = HTTPBearer(auto_error=False)
    
    # Helper function to validate token
    async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
        api_token = os.environ.get('IOS_APITOKEN')
        if not api_token:
            return None  # No authentication required if no token configured
            
        if not credentials or credentials.credentials != api_token:
            logger.warning("Invalid API token in request")
            raise HTTPException(status_code=401, detail="Invalid API token")
        return credentials.credentials
    
    # API Routes
    @app.get("/health")
    async def health_check():
        """Health check endpoint - no auth required."""
        return {"status": "ok"}
    
    @app.get("/api/status")
    async def get_api_status(token: Optional[str] = Depends(verify_token)):
        """Return status information about the API."""
        try:
            from config import config
            provider = getattr(config.api, 'provider', 'anthropic')
            
            # If using Ollama, include queue stats
            queue_stats = {}
            if provider.lower() == 'ollama' and request_queue:
                try:
                    queue_stats = request_queue.get_queue_stats()
                except Exception as e:
                    logger.error(f"Error getting queue stats: {e}")
            
            return {
                "status": "ok",
                "provider": provider,
                "queue_stats": queue_stats,
                "active_users": len(user_sessions)
            }
        except Exception as e:
            logger.error(f"Error in status endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    # Helper function to get or create a conversation
    async def get_conversation(conversation_id: Optional[str], user_id: Optional[str]) -> Conversation:
        """Get existing conversation or create a new one."""
        # Implementation details...
    
    # Chat endpoint
    @app.post("/api/chat")
    async def chat(request_data: dict, token: Optional[str] = Depends(verify_token)):
        """Chat endpoint implementation."""
        # Implementation details...
    
    # Streaming endpoint
    @app.post("/api/chat/stream")
    async def chat_stream(request_data: dict, token: Optional[str] = Depends(verify_token)):
        """Streaming chat endpoint implementation."""
        # Implementation details...
    
    # Queue status endpoint
    @app.post("/api/queue/status")
    async def queue_status(request_data: dict, token: Optional[str] = Depends(verify_token)):
        """Check queue position for a request."""
        # Implementation details...
    
    # Cancel request endpoint
    @app.post("/api/queue/cancel")
    async def cancel_request(request_data: dict, token: Optional[str] = Depends(verify_token)):
        """Cancel a pending request."""
        # Implementation details...
    
    return app
```

### Launcher Script (run_api.py)

```python
"""
Launcher for the FastAPI application.
"""
import os
import uvicorn
from config import config

def main():
    """Start the FastAPI server."""
    # Get configuration
    host = getattr(config.api_server, "host", "0.0.0.0")
    port = getattr(config.api_server, "port", 8000)
    workers = getattr(config.api_server, "workers", 1)
    log_level = getattr(config.api_server, "log_level", "info")
    
    # Configure environment variables (if needed)
    if not os.environ.get('IOS_APITOKEN') and hasattr(config, 'IOS_APITOKEN'):
        os.environ['IOS_APITOKEN'] = config.IOS_APITOKEN
    
    # Start server
    print(f"Starting FastAPI server on {host}:{port} with {workers} workers")
    uvicorn.run(
        "api.fastapi_app:app",
        host=host,
        port=port, 
        workers=workers,
        log_level=log_level
    )

if __name__ == "__main__":
    main()
```