"""
WSGI entry point for the AI agent system.

This module creates a WSGI-compatible application object that web servers
can use to communicate with the application.
"""
import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import from the app
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Set this environment variable before importing any libraries that might use tokenizers
# This prevents deadlocks when the process is forked
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create a WSGI application
from api.flask_app import create_app

# Create the application instance
application = create_app()

# For WSGI servers that expect the 'app' variable instead of 'application'
app = application

# This allows the file to be run directly for testing
if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8000, debug=True)