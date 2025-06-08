"""
FastAPI application for MIRA AI agent system.
Provides REST API endpoints for chat interactions with Anthropic and Ollama support.
"""
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Project imports
from main import initialize_system, parse_arguments, save_conversation
from conversation import Conversation
from errors import error_context, AgentError, handle_error, ErrorCode
from config import config
from auth import auth_router, get_current_user, get_current_user_optional, User
from auth.security_middleware import SecurityHeadersMiddleware
from auth.csrf_middleware import CSRFMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fastapi_app")

# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    priority: Optional[int] = Field(5, ge=1, le=10)  # 1 is highest

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    request_id: Optional[str] = None

class QueueStatusRequest(BaseModel):
    request_id: str

class QueueStatusResponse(BaseModel):
    request_id: str
    position: int
    estimated_wait: Optional[float] = None
    status: str

class CancelRequest(BaseModel):
    request_id: str

class StatusResponse(BaseModel):
    status: str
    provider: str
    queue_stats: Optional[Dict[str, Any]] = None
    active_users: int

# Global variables for system components
system_components = None
conversations = {}
user_sessions = {}
request_queue = None

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    global system_components, request_queue
    
    app = FastAPI(
        title="MIRA AI Agent API",
        description="API for the MIRA AI agent system with multi-provider support",
        version="1.0.0"
    )
    
    # Security middleware - order matters, security headers should be first
    app.add_middleware(SecurityHeadersMiddleware)
    
    # CSRF protection
    app.add_middleware(CSRFMiddleware)
    
    # Trusted host middleware
    allowed_hosts = os.environ.get("ALLOWED_HOSTS", "localhost").split(",")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    
    # CORS configuration - restrict origins for security
    allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-CSRF-Token"],
    )
    
    # Add authentication router
    app.include_router(auth_router)
    
    # Initialize the system
    args = parse_arguments()
    system_components = initialize_system(args)
    
    # Get the request queue for Ollama if available
    if hasattr(system_components['llm_bridge'], 'request_queue'):
        request_queue = system_components['llm_bridge'].request_queue
        logger.info("Request queue available for Ollama multi-user support")
    
    
    # Helper functions
    async def get_or_create_conversation(conversation_id: Optional[str], user_id: Optional[str]) -> tuple[Conversation, str]:
        """Get existing conversation or create a new one."""
        # If conversation_id is provided, try to load it
        if conversation_id:
            if conversation_id in conversations:
                return conversations[conversation_id], conversation_id
            
            # Try to load from disk
            try:
                conv = Conversation(
                    conversation_id=conversation_id,
                    system_prompt=config.get_system_prompt("main_system_prompt"),
                    llm_bridge=system_components['llm_bridge'],
                    tool_repo=system_components['tool_repo'],
                    tool_relevance_engine=system_components['tool_relevance_engine'],
                    workflow_manager=system_components['workflow_manager'],
                    working_memory=system_components['working_memory']
                )
                await conv.load_conversation()
                conversations[conversation_id] = conv
                return conv, conversation_id
            except Exception as e:
                logger.warning(f"Failed to load conversation {conversation_id}: {e}")
        
        # Create new conversation
        conv_id = str(uuid.uuid4())
        conv = Conversation(
            conversation_id=conv_id,
            system_prompt=config.get_system_prompt("main_system_prompt"),
            llm_bridge=system_components['llm_bridge'],
            tool_repo=system_components['tool_repo'],
            tool_relevance_engine=system_components['tool_relevance_engine'],
            workflow_manager=system_components['workflow_manager'],
            working_memory=system_components['working_memory']
        )
        conversations[conv_id] = conv
        
        # Track user session if provided
        if user_id:
            user_sessions[f"user:{user_id}"] = conv_id
        
        return conv, conv_id
    
    async def handle_special_commands(message: str, conversation: Conversation) -> Optional[str]:
        """Handle special slash commands."""
        message_lower = message.lower().strip()
        
        if message_lower == "/clear":
            conversation.messages = []
            return "Conversation cleared."
        elif message_lower == "/reload_user":
            config.reload_config()
            return "User configuration reloaded."
        elif message_lower.startswith("/toolfeedback"):
            parts = message.split(maxsplit=1)
            if len(parts) > 1:
                from tools.tool_feedback import handle_tool_feedback
                feedback_response = handle_tool_feedback(parts[1])
                return feedback_response
            else:
                return "Please provide feedback after /toolfeedback command."
        
        return None
    
    # API Routes
    @app.get("/health")
    async def health_check():
        """Health check endpoint - no auth required."""
        return {"status": "ok"}
    
    @app.get("/api/status", response_model=StatusResponse)
    async def get_api_status(current_user: User = Depends(get_current_user)):
        """Return status information about the API - requires authentication."""
        try:
            provider = getattr(config.api, 'provider', 'anthropic')
            
            # If using Ollama, include queue stats
            queue_stats = {}
            if provider.lower() == 'ollama' and request_queue:
                try:
                    queue_stats = request_queue.get_queue_stats()
                except Exception as e:
                    logger.error(f"Error getting queue stats: {e}")
            
            return StatusResponse(
                status="ok",
                provider=provider,
                queue_stats=queue_stats if queue_stats else None,
                active_users=len(user_sessions)
            )
        except Exception as e:
            logger.error(f"Error in status endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(
        request: ChatRequest,
        current_user: User = Depends(get_current_user)
    ):
        """Chat endpoint for single response - requires authentication."""
        try:
            # Get or create conversation using authenticated user's ID
            conversation, conv_id = await get_or_create_conversation(
                request.conversation_id, 
                str(current_user.id)
            )
            
            # Handle special commands
            special_response = await handle_special_commands(request.message, conversation)
            if special_response:
                return ChatResponse(response=special_response, conversation_id=conv_id)
            
            # Process regular chat
            request_id = str(uuid.uuid4()) if request_queue else None
            
            # For Ollama with queue
            if request_queue:
                result = await request_queue.process_request(
                    conversation=conversation,
                    user_message=request.message,
                    request_id=request_id,
                    priority=request.priority
                )
                response = result['response']
            else:
                # Direct processing for Anthropic
                response = conversation.generate_response(request.message)
            
            # Save conversation
            save_conversation(conversation)
            
            return ChatResponse(
                response=response,
                conversation_id=conv_id,
                request_id=request_id
            )
            
        except Exception as e:
            logger.error(f"Error in chat endpoint: {e}", exc_info=True)
            error_message = handle_error(e)
            raise HTTPException(
                status_code=500,
                detail=error_message
            )
    
    @app.post("/api/chat/stream")
    async def chat_stream(
        request: ChatRequest,
        current_user: User = Depends(get_current_user)
    ):
        """Streaming chat endpoint using Server-Sent Events - requires authentication."""
        try:
            # Get or create conversation using authenticated user's ID
            conversation, conv_id = await get_or_create_conversation(
                request.conversation_id,
                str(current_user.id)
            )
            
            # Handle special commands
            special_response = await handle_special_commands(request.message, conversation)
            if special_response:
                async def special_generator():
                    yield {
                        "event": "message",
                        "data": json.dumps({"text": special_response})
                    }
                    yield {
                        "event": "done",
                        "data": json.dumps({"conversation_id": conv_id})
                    }
                return EventSourceResponse(special_generator())
            
            # Stream regular chat
            request_id = str(uuid.uuid4()) if request_queue else None
            
            async def event_generator():
                try:
                    full_response = ""
                    
                    # For Ollama with queue
                    if request_queue:
                        async for chunk in request_queue.process_streaming_request(
                            conversation=conversation,
                            user_message=request.message,
                            request_id=request_id,
                            priority=request.priority
                        ):
                            if chunk['type'] == 'token':
                                full_response += chunk['text']
                                yield {
                                    "event": "message",
                                    "data": json.dumps({"text": chunk['text']})
                                }
                            elif chunk['type'] == 'error':
                                yield {
                                    "event": "error",
                                    "data": json.dumps({"error": chunk['error']})
                                }
                                return
                    else:
                        # Direct streaming for Anthropic
                        tokens = []
                        
                        def stream_callback(token):
                            tokens.append(token)
                        
                        # Generate response with streaming
                        response = conversation.generate_response(
                            request.message,
                            stream=True,
                            stream_callback=stream_callback
                        )
                        
                        # Yield tokens as they were collected
                        for token in tokens:
                            full_response += token
                            yield {
                                "event": "message",
                                "data": json.dumps({"text": token})
                            }
                    
                    # Save conversation after streaming completes
                    save_conversation(conversation)
                    
                    yield {
                        "event": "done",
                        "data": json.dumps({
                            "conversation_id": conv_id,
                            "request_id": request_id
                        })
                    }
                    
                except Exception as e:
                    logger.error(f"Error in streaming: {e}", exc_info=True)
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": str(e)})
                    }
            
            return EventSourceResponse(event_generator())
            
        except Exception as e:
            logger.error(f"Error in stream endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/queue/status", response_model=QueueStatusResponse)
    async def queue_status(
        request: QueueStatusRequest,
        current_user: User = Depends(get_current_user)
    ):
        """Check queue position for a request (Ollama only) - requires authentication."""
        if not request_queue:
            raise HTTPException(
                status_code=400,
                detail="Queue management not available (Anthropic mode)"
            )
        
        try:
            position = request_queue.get_queue_position(request.request_id)
            if position is None:
                raise HTTPException(
                    status_code=404,
                    detail="Request not found in queue"
                )
            
            return QueueStatusResponse(
                request_id=request.request_id,
                position=position,
                estimated_wait=position * 30.0,  # Rough estimate
                status="queued" if position > 0 else "processing"
            )
        except Exception as e:
            logger.error(f"Error checking queue status: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/queue/cancel")
    async def cancel_request(
        request: CancelRequest,
        current_user: User = Depends(get_current_user)
    ):
        """Cancel a pending request (Ollama only) - requires authentication."""
        if not request_queue:
            raise HTTPException(
                status_code=400,
                detail="Queue management not available (Anthropic mode)"
            )
        
        try:
            success = request_queue.cancel_request(request.request_id)
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail="Request not found or already processing"
                )
            
            return {"status": "cancelled", "request_id": request.request_id}
        except Exception as e:
            logger.error(f"Error cancelling request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

# Create the FastAPI app instance
app = create_app()