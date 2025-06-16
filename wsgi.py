"""
WSGI/ASGI entry point for the AI agent system.

This module creates a WSGI-compatible application object using FastAPI with a2wsgi adapter.
"""
import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import from the app
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Set this environment variable before importing any libraries that might use tokenizers
# This prevents deadlocks when the process is forked
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Use FastAPI with a2wsgi adapter for WSGI compatibility
from a2wsgi import ASGIMiddleware
from api.fastapi_app import app as fastapi_app

# Create WSGI-compatible wrapper
application = ASGIMiddleware(fastapi_app)
app = application

# This allows the file to be run directly for testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.fastapi_app:app", host='0.0.0.0', port=8000, reload=True)