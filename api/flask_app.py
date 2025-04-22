"""
Flask API for the AI agent system.

This module provides a Flask API for the AI agent system,
allowing for communication via HTTP endpoints.
"""
import json
import logging
import os
import time
import uuid
import secrets
from functools import wraps
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from config import config
from conversation import Conversation
from tools.repo import ToolRepository
from api.llm_bridge import LLMBridge
from errors import handle_error, APIError, ErrorCode, error_context


# Initialize Flask app
app = Flask(__name__)
# Disable CORS entirely - this will block browser access but allow iOS app access
# iOS apps don't use CORS as they're not subject to same-origin policy
# NOTE: If web browser access is needed in the future, replace with specific allowed origins
CORS(app, origins=[], methods=['GET', 'POST', 'DELETE'], 
     allow_headers=['Content-Type', 'Authorization'])

# Set up logging
logger = logging.getLogger("flask_api")

# Global components
conversations: Dict[str, Conversation] = {}
llm_bridge = None
tool_repo = None

# Memory management constants
MAX_CONVERSATIONS = 100  # Maximum number of conversations to store in memory
MAX_CONVERSATION_AGE = 60 * 60 * 24 * 7  # 7 days in seconds

def cleanup_old_conversations():
    """Remove old conversations to prevent memory leaks."""
    if len(conversations) <= MAX_CONVERSATIONS:
        return
    
    # Get conversation IDs sorted by last access time
    now = time.time()
    sorted_conversations = sorted(
        conversations.items(),
        key=lambda x: now - x[1].messages[-1].created_at if x[1].messages else 0,
        reverse=True
    )
    
    # Remove oldest conversations beyond the limit
    for i in range(MAX_CONVERSATIONS, len(sorted_conversations)):
        conv_id, conv = sorted_conversations[i]
        logger.info(f"Removing old conversation {conv_id} to manage memory")
        del conversations[conv_id]

# Authentication settings
IOS_APP_API_KEY = os.environ.get('IOS_APP_API_KEY') or config.api.key if hasattr(config, 'api') and hasattr(config.api, 'key') else None
if not IOS_APP_API_KEY:
    # Generate a random key if none is provided - NOT RECOMMENDED FOR PRODUCTION
    IOS_APP_API_KEY = secrets.token_hex(32)
    print("\n" + "!" * 80)
    print("!!! SECURITY WARNING: No API key found. Generated temporary API key: !!!")
    print(f"!!! {IOS_APP_API_KEY}")
    print("!!! IMPORTANT: For production, set IOS_APP_API_KEY in environment variables !!!")
    print("!!! or in your config file. This key will change on server restart.   !!!")
    print("!" * 80 + "\n")

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get the token from the Authorization header
        auth_header = request.headers.get('Authorization')
        
        # Check if header exists and has correct format
        if not auth_header or not auth_header.startswith('Bearer '):
            error_response = {
                "error": "Missing or invalid authorization header",
                "timestamp": int(time.time())
            }
            return Response(
                json.dumps(error_response, ensure_ascii=False),
                status=401,
                mimetype='application/json'
            )
        
        # Extract the token
        token = auth_header.split('Bearer ')[1].strip()
        
        # Validate the token directly against the API key
        if token != IOS_APP_API_KEY:
            error_response = {
                "error": "Invalid API key",
                "timestamp": int(time.time())
            }
            return Response(
                json.dumps(error_response, ensure_ascii=False),
                status=401,
                mimetype='application/json'
            )
            
        # Token is valid, proceed
        return f(*args, **kwargs)
    return decorated_function


def initialize_system():
    """Initialize the system components."""
    global llm_bridge, tool_repo
    
    logger.info("Initializing AI agent system components")
    
    with error_context(
        component_name="API",
        operation="initialization",
        error_class=APIError,
        error_code=ErrorCode.UNKNOWN_ERROR,
        logger=logger
    ):
        # Initialize data directory
        os.makedirs(config.paths.data_dir, exist_ok=True)
        
        # Initialize LLM bridge
        llm_bridge = LLMBridge()
        
        # Initialize tool repository
        tool_repo = ToolRepository()
        
        # Discover and register tools
        tool_repo.discover_tools()
        
        # Enable tools from config
        tool_repo.enable_tools_from_config()
        
        logger.info("System components initialized successfully")


def get_or_create_conversation(conversation_id: Optional[str] = None) -> Conversation:
    """
    Get an existing conversation or create a new one.
    
    Args:
        conversation_id: Optional conversation ID
        
    Returns:
        Conversation object
    """
    global conversations, llm_bridge, tool_repo
    
    # If no conversation ID provided, generate a new one
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Check if conversation exists
    if conversation_id not in conversations:
        # Clean up old conversations if we're about to hit the limit
        if len(conversations) >= MAX_CONVERSATIONS:
            cleanup_old_conversations()
            
        # Create a new conversation
        conversations[conversation_id] = Conversation(
            conversation_id=conversation_id,
            llm_bridge=llm_bridge,
            tool_repo=tool_repo
        )
        logger.info(f"Created new conversation: {conversation_id}")
    
    return conversations[conversation_id]


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    response_data = {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": int(time.time())
    }
    return Response(
        json.dumps(response_data, ensure_ascii=False),
        mimetype='application/json'
    )




@app.route('/api/chat', methods=['POST'])
@require_api_key
def chat():
    """Chat endpoint for sending messages and receiving responses."""
    try:
        # Get request data
        data = request.json
        if not data:
            error_response = {
                "error": "No data provided",
                "timestamp": int(time.time())
            }
            return Response(
                json.dumps(error_response, ensure_ascii=False),
                status=400,
                mimetype='application/json'
            )
        
        # Extract required parameters
        user_input = data.get('message')
        conversation_id = data.get('conversation_id')
        
        # Extract optional location data if provided
        location = data.get('location')
        if location and isinstance(location, dict):
            latitude = location.get('latitude')
            longitude = location.get('longitude')
            
            # Store location in the conversation if valid coordinates
            if latitude is not None and longitude is not None:
                try:
                    lat_float = float(latitude)
                    lng_float = float(longitude)
                    
                    # Create conversation if it doesn't exist yet
                    conversation = get_or_create_conversation(conversation_id)
                    
                    # Store the location in conversation metadata
                    if not hasattr(conversation, 'metadata'):
                        conversation.metadata = {}
                    
                    # Update location data
                    conversation.metadata['location'] = {
                        'latitude': lat_float,
                        'longitude': lng_float,
                        'accuracy': float(location.get('accuracy', 0)),
                        'timestamp': location.get('timestamp', int(time.time()))
                    }
                    
                    logger.debug(f"Stored location data for conversation {conversation_id}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid location data: {e}")
        
        # Validate input
        if not user_input or not isinstance(user_input, str) or user_input.strip() == "":
            error_response = {
                "error": "Message is required",
                "timestamp": int(time.time())
            }
            return Response(
                json.dumps(error_response, ensure_ascii=False),
                status=400,
                mimetype='application/json'
            )
        
        # Get or create conversation
        conversation = get_or_create_conversation(conversation_id)
        
        # Generate response
        with error_context(
            component_name="API",
            operation="generating response",
            error_class=APIError,
            logger=logger
        ):
            # Extract optional parameters
            temperature = data.get('temperature')
            max_tokens = data.get('max_tokens')
            
            # Generate the response
            response = conversation.generate_response(
                user_input=user_input,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # We don't need to format messages since we're not returning them
            # This improves performance by avoiding unnecessary processing
            
            # Create JSON response - only include the final response, not tool call details
            response_data = {
                "conversation_id": conversation.conversation_id,
                "response": response,  # Contains markdown that client can render
                "timestamp": int(time.time())
            }
            
            # Return the response with the correct content type
            return Response(
                json.dumps(response_data, ensure_ascii=False),
                mimetype='application/json'
            )
        
    except Exception as e:
        # Handle errors
        error_message = handle_error(e)
        logger.error(f"Error in chat endpoint: {error_message}")
        error_response = {
            "error": error_message,
            "timestamp": int(time.time())
        }
        return Response(
            json.dumps(error_response, ensure_ascii=False),
            status=500,
            mimetype='application/json'
        )


@app.route('/api/conversation/<conversation_id>', methods=['GET'])
@require_api_key
def get_conversation(conversation_id):
    """Get conversation history."""
    try:
        # Check if conversation exists
        if conversation_id not in conversations:
            # Return an empty conversation instead of 404 to allow clients to start fresh
            response_data = {
                "conversation_id": conversation_id,
                "messages": [],
                "exists": False,
                "timestamp": int(time.time())
            }
            return Response(
                json.dumps(response_data, ensure_ascii=False),
                mimetype='application/json'
            )
        
        # Get conversation
        conversation = conversations[conversation_id]
        
        # We don't need to format messages since we're not returning them
        
        # Create response data - only include conversation ID and timestamp
        # Client should make a separate request to /api/chat to get responses
        response_data = {
            "conversation_id": conversation.conversation_id,
            "exists": True,
            "timestamp": int(time.time())
        }
        
        # Return formatted response
        return Response(
            json.dumps(response_data, ensure_ascii=False),
            mimetype='application/json'
        )
        
    except Exception as e:
        # Handle errors
        error_message = handle_error(e)
        logger.error(f"Error getting conversation: {error_message}")
        error_response = {
            "error": error_message,
            "timestamp": int(time.time())
        }
        return Response(
            json.dumps(error_response, ensure_ascii=False),
            status=500,
            mimetype='application/json'
        )


@app.route('/api/conversation/<conversation_id>', methods=['DELETE'])
@require_api_key
def clear_conversation(conversation_id):
    """Clear conversation history."""
    try:
        # Check if conversation exists
        if conversation_id not in conversations:
            # Create a new empty conversation instead of returning 404
            conversation = get_or_create_conversation(conversation_id)
            response_data = {
                "status": "success", 
                "message": "New conversation created",
                "conversation_id": conversation_id,
                "timestamp": int(time.time())
            }
            return Response(
                json.dumps(response_data, ensure_ascii=False),
                mimetype='application/json'
            )
        
        # Clear conversation
        conversations[conversation_id].clear_history()
        
        # Return success
        response_data = {
            "status": "success", 
            "message": "Conversation cleared",
            "conversation_id": conversation_id,
            "timestamp": int(time.time())
        }
        return Response(
            json.dumps(response_data, ensure_ascii=False),
            mimetype='application/json'
        )
        
    except Exception as e:
        # Handle errors
        error_message = handle_error(e)
        logger.error(f"Error clearing conversation: {error_message}")
        error_response = {
            "error": error_message,
            "timestamp": int(time.time())
        }
        return Response(
            json.dumps(error_response, ensure_ascii=False),
            status=500,
            mimetype='application/json'
        )


def setup_app():
    """Set up the Flask application."""
    # Initialize system components
    initialize_system()
    
    # Set up logging
    level = getattr(logging, config.system.log_level)
    logging.basicConfig(
        level=level,
        format='%(levelname)s [%(name)s] %(message)s'
    )
    
    # Disable noisy HTTP library loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return app


if __name__ == '__main__':
    # Set up the application
    app = setup_app()
    
    # Run the Flask application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)