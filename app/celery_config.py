"""
Celery configuration
"""
import os

# Force Docker service names
broker_url = f"redis://{os.getenv('REDIS_HOST', 'redis')}:6379/0"
result_backend = f"redis://{os.getenv('REDIS_HOST', 'redis')}:6379/0"

# Task settings
task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True