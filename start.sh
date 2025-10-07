#!/bin/bash
# Install dependencies
pip install -r requirements.txt
pip install gunicorn

# Run migrations if you have any
# python manage.py db upgrade

# Start the application
echo "Starting the application..."
gunicorn --config gunicorn_config.py app:app
