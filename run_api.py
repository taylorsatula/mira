#!/usr/bin/env python3
"""
Launcher for the FastAPI application.
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Start the FastAPI server."""
    # Default configuration
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    workers = int(os.environ.get("API_WORKERS", "1"))
    log_level = os.environ.get("API_LOG_LEVEL", "info")
    reload = os.environ.get("API_RELOAD", "false").lower() == "true"
    
    # Validate required environment variables
    if not os.environ.get("SECRET_KEY"):
        print("Error: SECRET_KEY environment variable is required")
        sys.exit(1)
    
    # Optional: Check for API token
    if not os.environ.get("IOS_APITOKEN"):
        print("Warning: IOS_APITOKEN not set. API will run without authentication.")
    
    print(f"Starting FastAPI server on {host}:{port}")
    print(f"Workers: {workers}, Log level: {log_level}, Reload: {reload}")
    
    # Start server
    if workers > 1 and not reload:
        # Multi-worker mode (production)
        uvicorn.run(
            "api.fastapi_app:app",
            host=host,
            port=port,
            workers=workers,
            log_level=log_level
        )
    else:
        # Single worker mode (development)
        uvicorn.run(
            "api.fastapi_app:app",
            host=host,
            port=port,
            log_level=log_level,
            reload=reload
        )

if __name__ == "__main__":
    main()