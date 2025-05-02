"""
Flask application for exposing the AI agent system via a web API.

This module creates a minimal Flask adapter that reuses the same core system
as the CLI interface, ensuring durability and consistency.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, Response, stream_with_context, abort
from functools import wraps

# Project imports - reusing the same code as the CLI
from main import initialize_system, parse_arguments, save_conversation
from conversation import Conversation
from errors import error_context, AgentError, handle_error, ErrorCode, FileOperationError
from tools.tool_feedback import save_tool_feedback


def create_app() -> Flask:
    """
    Create and configure the Flask application.
    
    Returns:
        Configured Flask application.
    """
    logger = logging.getLogger("flask_app")
    app = Flask(__name__)
    
    # Set default configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY'),
        IOS_APITOKEN=os.environ.get('IOS_APITOKEN'),
    )
    
    # Initialize the system exactly as the CLI does
    # This ensures we're using the same core components
    args = parse_arguments()
    system = initialize_system(args)
    
    # Verify critical configuration
    if not app.config.get('SECRET_KEY'):
        logger.error("Flask SECRET_KEY not configured in .env file")
        raise RuntimeError("Flask SECRET_KEY must be configured in the .env file for security")
    
    # Define token authentication decorator
    def require_api_token(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_token = app.config.get('IOS_APITOKEN')
            # If token is not configured, abort all requests
            if not api_token:
                logger.error("API token not configured in .env file")
                abort(401, "API token not configured")
                
            # Check if the token is in the request header
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer ') or auth_header[7:] != api_token:
                logger.warning("Invalid or missing API token in request")
                abort(401, "Invalid API token")
            return f(*args, **kwargs)
        return decorated_function
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    @require_api_token
    def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        return jsonify({"status": "ok"})
    
    # Chat endpoint for single response
    @app.route('/api/chat', methods=['POST'])
    @require_api_token
    def chat() -> Dict[str, Any]:
        """
        Chat endpoint for getting a single response.
        
        Expected JSON body:
        {
            "message": "User message here",
            "conversation_id": "optional-conversation-id"
        }
        """
        try:
            data = request.json
            
            if not data:
                return jsonify({"error": "Invalid JSON body"}), 400
                
            message = data.get('message', '')
            conversation_id = data.get('conversation_id')
            
            if not message:
                return jsonify({"error": "No message provided"}), 400
            
            # Get the conversation object from the system
            conversation = system['conversation']
            
            # If a different conversation ID was requested, we need to load it
            if conversation_id and conversation_id != conversation.conversation_id:
                # Process in the same way main.py does for the --conversation flag
                from config import config
                # Direct file operations
                
                conversation_dir = Path(config.paths.conversation_history_dir)
                os.makedirs(conversation_dir, exist_ok=True)
                file_path = conversation_dir / f"conversation_{conversation_id}.json"
                
                try:
                    if not file_path.exists():
                        raise FileOperationError(
                            f"File not found: {file_path}",
                            ErrorCode.FILE_NOT_FOUND
                        )
                    
                    try:
                        with open(file_path, 'r') as f:
                            conversation_data = json.load(f)
                    except json.JSONDecodeError as e:
                        raise FileOperationError(
                            f"Invalid JSON in file {file_path}: {e}",
                            ErrorCode.INVALID_JSON
                        )
                    except Exception as e:
                        raise FileOperationError(
                            f"Error reading file {file_path}: {e}",
                            ErrorCode.FILE_READ_ERROR
                        )
                    # Create a new conversation with the loaded data
                    conversation = Conversation.from_dict(
                        conversation_data,
                        llm_bridge=system['llm_bridge'],
                        tool_repo=system['tool_repo'],
                        tool_relevance_engine=system['tool_relevance_engine'],
                        workflow_manager=system['workflow_manager']
                    )
                    # Update the system's conversation reference
                    system['conversation'] = conversation
                    logger.info(f"Loaded conversation: {conversation.conversation_id}")
                except Exception as e:
                    logger.warning(f"Failed to load conversation, creating new one: {e}")
                    conversation = Conversation(
                        conversation_id=conversation_id,
                        llm_bridge=system['llm_bridge'],
                        tool_repo=system['tool_repo'],
                        tool_relevance_engine=system['tool_relevance_engine'],
                        workflow_manager=system['workflow_manager']
                    )
                    system['conversation'] = conversation
            
            # Process special commands the same way as the CLI
            if message.lower() == '/clear':
                conversation.clear_history()
                save_conversation(system['file_ops'], conversation)
                return jsonify({
                    "response": "Conversation history cleared.",
                    "conversation_id": conversation.conversation_id
                })
                
            elif message.lower() == '/reload_user':
                conversation.reload_user_information()
                save_conversation(system['file_ops'], conversation)
                return jsonify({
                    "response": "User information reloaded.",
                    "conversation_id": conversation.conversation_id
                })
                
            elif message.lower().startswith('/toolfeedback'):
                feedback_text = message[len('/toolfeedback'):].strip()
                if not feedback_text:
                    return jsonify({
                        "response": "Please provide feedback text after /toolfeedback",
                        "conversation_id": conversation.conversation_id
                    })
                
                # Save tool feedback using the same function as the CLI
                success, analysis = save_tool_feedback(system, feedback_text, conversation)
                
                if success:
                    response_text = "Tool feedback saved successfully."
                    if analysis:
                        response_text += f" Analysis: {analysis}"
                    return jsonify({
                        "response": response_text,
                        "conversation_id": conversation.conversation_id
                    })
                else:
                    return jsonify({
                        "response": "Error saving tool feedback.",
                        "conversation_id": conversation.conversation_id
                    }), 500
            
            # Generate normal response
            with error_context(
                component_name="WebAPI",
                operation="generating response",
                error_class=AgentError,
                logger=logger
            ):
                response = conversation.generate_response(message)
            
            # Save conversation using the same function as the CLI
            save_conversation(conversation)
            
            return jsonify({
                "response": response,
                "conversation_id": conversation.conversation_id
            })
            
        except Exception as e:
            logger.error(f"Error in chat endpoint: {e}", exc_info=True)
            error_message = handle_error(e)
            return jsonify({"error": error_message}), 500
    
    # Chat streaming endpoint
    @app.route('/api/chat/stream', methods=['POST'])
    @require_api_token
    def chat_stream() -> Response:
        """
        Streaming chat endpoint.
        
        Expected JSON body:
        {
            "message": "User message here",
            "conversation_id": "optional-conversation-id"
        }
        
        Returns a stream of server-sent events.
        """
        try:
            data = request.json
            
            if not data:
                return jsonify({"error": "Invalid JSON body"}), 400
                
            message = data.get('message', '')
            conversation_id = data.get('conversation_id')
            
            if not message:
                return jsonify({"error": "No message provided"}), 400
            
            # Get the conversation object from the system
            conversation = system['conversation']
            
            # If a different conversation ID was requested, we need to load it
            if conversation_id and conversation_id != conversation.conversation_id:
                # Process in the same way main.py does for the --conversation flag
                from config import config
                # Direct file operations
                
                conversation_dir = Path(config.paths.conversation_history_dir)
                os.makedirs(conversation_dir, exist_ok=True)
                file_path = conversation_dir / f"conversation_{conversation_id}.json"
                
                try:
                    if not file_path.exists():
                        raise FileOperationError(
                            f"File not found: {file_path}",
                            ErrorCode.FILE_NOT_FOUND
                        )
                    
                    try:
                        with open(file_path, 'r') as f:
                            conversation_data = json.load(f)
                    except json.JSONDecodeError as e:
                        raise FileOperationError(
                            f"Invalid JSON in file {file_path}: {e}",
                            ErrorCode.INVALID_JSON
                        )
                    except Exception as e:
                        raise FileOperationError(
                            f"Error reading file {file_path}: {e}",
                            ErrorCode.FILE_READ_ERROR
                        )
                    # Create a new conversation with the loaded data
                    conversation = Conversation.from_dict(
                        conversation_data,
                        llm_bridge=system['llm_bridge'],
                        tool_repo=system['tool_repo'],
                        tool_relevance_engine=system['tool_relevance_engine'],
                        workflow_manager=system['workflow_manager']
                    )
                    # Update the system's conversation reference
                    system['conversation'] = conversation
                    logger.info(f"Loaded conversation: {conversation.conversation_id}")
                except Exception as e:
                    logger.warning(f"Failed to load conversation, creating new one: {e}")
                    conversation = Conversation(
                        conversation_id=conversation_id,
                        llm_bridge=system['llm_bridge'],
                        tool_repo=system['tool_repo'],
                        tool_relevance_engine=system['tool_relevance_engine'],
                        workflow_manager=system['workflow_manager']
                    )
                    system['conversation'] = conversation
            
            # Process special commands the same way as the CLI
            if message.lower() in ['/clear', '/reload_user'] or message.lower().startswith('/toolfeedback'):
                # For commands, we'll just handle them in the regular non-streaming endpoint
                # and return a simple response
                resp = chat()
                return resp
            
            def generate():
                # Container for responses
                response_tokens = []
                
                # Define a proper callback that appends to our container
                def stream_callback(token):
                    response_tokens.append(token)
                    return token
                
                try:
                    # Generate response with streaming, using the same method as CLI
                    conversation.generate_response(
                        message,
                        stream=True,
                        stream_callback=stream_callback
                    )
                    
                    # Now yield each token as a server-sent event
                    for token in response_tokens:
                        yield f"data: {token}\n\n"
                    
                    # Save conversation after streaming completes
                    save_conversation(conversation)
                    
                    # Final event to indicate completion
                    yield f"event: done\n"
                    yield f"data: {{\n"
                    yield f"data: \"conversation_id\": \"{conversation.conversation_id}\"\n"
                    yield f"data: }}\n\n"
                
                except Exception as e:
                    logger.error(f"Error in streaming: {e}", exc_info=True)
                    error_message = handle_error(e)
                    yield f"event: error\n"
                    yield f"data: {{\n"
                    yield f"data: \"error\": \"{error_message}\"\n"
                    yield f"data: }}\n\n"
            
            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream'
            )
            
        except Exception as e:
            logger.error(f"Error in chat stream endpoint: {e}", exc_info=True)
            error_message = handle_error(e)
            return jsonify({"error": error_message}), 500
            
    return app


# This allows direct testing of the Flask app
if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run the application
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=True)