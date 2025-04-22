"""
WSGI configuration for Passenger web server integration on A2Hosting.
This file connects the Passenger web server with our Flask application.
"""
import os
import sys

# Add the virtual environment's site-packages to the path
venv_path = '/home/rocketc5/virtualenv/webapps/bot/3.12'
site_packages = os.path.join(venv_path, 'lib', 'python3.12', 'site-packages')
sys.path.insert(0, site_packages)

# Add the app directory to the Python path
app_path = '/home/rocketc5/webapps/bot'
sys.path.insert(0, app_path)

# Import the Flask app from our existing application
from api.flask_app import setup_app

# Initialize the application
application = setup_app()

# Fallback route for testing if Passenger can't find our app routes
@application.route('/passenger_wsgi_test')
def passenger_test():
    return 'Flask application is running via Passenger WSGI!'