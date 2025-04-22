"""
Flask application entry point for the AI agent system.

This module provides a clean entry point for running the Flask application.
"""
import os
import sys
import argparse
from api.flask_app import setup_app

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI Agent System API Server')
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=443,
        help='Port to run the server on (default: 443)'
    )
    parser.add_argument(
        '--ssl', '-s',
        action='store_true',
        help='Enable SSL/HTTPS'
    )
    parser.add_argument(
        '--cert', '-c',
        type=str,
        default='cert.pem',
        help='Path to SSL certificate (default: cert.pem)'
    )
    parser.add_argument(
        '--key', '-k',
        type=str,
        default='key.pem',
        help='Path to SSL key (default: key.pem)'
    )
    return parser.parse_args()

# Set up the Flask application
app = setup_app()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Get port from args or environment
    port = int(os.environ.get('PORT', args.port))
    
    # Prepare SSL context
    ssl_context = None
    
    # Force SSL when using port 443 (standard HTTPS port)
    if port == 443 and not args.ssl:
        print("Warning: Running on port 443 requires SSL. Enabling --ssl flag automatically.")
        args.ssl = True
    
    if args.ssl:
        if os.path.exists(args.cert) and os.path.exists(args.key):
            ssl_context = (args.cert, args.key)
            print(f"Using SSL with certificate: {args.cert} and key: {args.key}")
        else:
            print(f"Error: SSL certificate or key not found but SSL is required.")
            print(f"Please provide valid certificate and key files or use a different port.")
            print(f"Certificate path: {args.cert}")
            print(f"Key path: {args.key}")
            sys.exit(1)
    
    # Start message
    protocol = "HTTPS" if ssl_context else "HTTP"
    print(f"Starting Flask server on port {port} ({protocol})...")
    
    # Run the Flask application
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=False,
        ssl_context=ssl_context
    )