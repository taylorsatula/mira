"""
Passenger WSGI entry point for the AI agent system.

This module creates a WSGI-compatible application object that Passenger
can use to communicate with the application. This allows the app to run
on hosting providers that use Passenger.
"""
import os
import sys
import imp

# Add the project root to the Python path
# Use os.path instead of Path for better compatibility
import os.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure thread limitations for shared hosting environment
# Limit number of threads for numerical libraries to avoid resource issues
os.environ["OMP_NUM_THREADS"] = "1"             # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"        # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"             # MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"      # Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"         # Numexpr
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # HuggingFace tokenizers

# Configure PyTorch to use fewer threads if available
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass  # PyTorch not installed yet

# Simple minimal WSGI application for when everything else fails
def minimal_error_app(environ, start_response, error_msg):
    """Create a minimal WSGI application that displays an error message."""
    status = '500 Internal Server Error'
    output = error_msg.encode('utf-8')
    response_headers = [('Content-type', 'text/plain'),
                       ('Content-Length', str(len(output)))]
    start_response(status, response_headers)
    return [output]

# For A2Hosting compatibility - create application from our wsgi.py
# This is necessary because A2Hosting has specific requirements for passenger_wsgi.py
try:
    # Import the WSGI application from our actual wsgi.py file
    # Use os.path instead of Path for better compatibility
    import os.path
    wsgi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wsgi.py')
    wsgi = imp.load_source('wsgi', wsgi_path)
    application = wsgi.application
except Exception as e1:
    error1 = str(e1)
    sys.stderr.write(f"Error loading wsgi application: {error1}\n")
    
    # Try direct import as fallback
    try:
        from api.flask_app import create_app
        application = create_app()
    except Exception as e2:
        error2 = str(e2)
        sys.stderr.write(f"Error creating application directly: {error2}\n")
        
        # Create a closure that captures the error messages
        def application(environ, start_response):
            return minimal_error_app(
                environ, 
                start_response, 
                f"Error loading application: {error1}, Fallback error: {error2}"
            )