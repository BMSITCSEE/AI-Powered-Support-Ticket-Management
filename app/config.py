"""
Application configuration settings
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    """Configuration class for application settings"""
    
    # Application settings
    APP_NAME = os.getenv("APP_NAME", "AI Support Ticket Manager")
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8501))
    
    # Database settings
    # Postgres settings
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "ticket_system")
    
    # Connection pool settings
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", 5))
    DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", 30))
    
    # Redis settings
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    
    # Model settings
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/saved/bert_classifier.pt")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0")
    MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", 128))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
    MODEL_CACHE_TTL = int(os.getenv("MODEL_CACHE_TTL", 3600))  # 1 hour
    
    # Celery settings
    CELERY_BROKER_URL = os.getenv(
        "CELERY_BROKER_URL", 
        f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    )
    CELERY_RESULT_BACKEND = os.getenv(
        "CELERY_RESULT_BACKEND",
        f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    )
    CELERY_TASK_TIME_LIMIT = int(os.getenv("CELERY_TASK_TIME_LIMIT", 300))  # 5 minutes
    CELERY_TASK_SOFT_TIME_LIMIT = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", 240))
    
    # Security settings
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
    JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", 24))
    SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", 30))
    PASSWORD_HASH_ROUNDS = int(os.getenv("PASSWORD_HASH_ROUNDS", 12))
    
    # Email settings
    EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "True").lower() == "true"
    EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
    EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
    EMAIL_USER = os.getenv("EMAIL_USER", "")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
    EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "True").lower() == "true"
    EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "Support Team")
    EMAIL_FROM_ADDRESS = os.getenv("EMAIL_FROM_ADDRESS", EMAIL_USER)
    
    # Notification settings
    NOTIFICATION_BATCH_SIZE = int(os.getenv("NOTIFICATION_BATCH_SIZE", 50))
    NOTIFICATION_RETRY_LIMIT = int(os.getenv("NOTIFICATION_RETRY_LIMIT", 3))
    
    # Ticket categories and priorities
    TICKET_CATEGORIES = [
        "Technical",
        "Billing",
        "Account",
        "Feedback",
        "General"
    ]
    
    URGENCY_LEVELS = [
        "Critical",
        "High",
        "Medium",
        "Low"
    ]
    
    TICKET_STATUSES = [
        "Open",
        "In Progress",
        "Resolved",
        "Closed",
        "Escalated",
        "Archived"
    ]
    
    # SLA settings (in hours)
    SLA_RESPONSE_TIME = {
        "Critical": 4,
        "High": 24,
        "Medium": 48,
        "Low": 72
    }
    
    # File upload settings
    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 50))
    ALLOWED_UPLOAD_EXTENSIONS = ['.csv', '.xlsx', '.xls']
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
    LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 10485760))  # 10MB
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))
    
    # Cache settings
        # Cache settings
    CACHE_TTL = int(os.getenv("CACHE_TTL", 300))  # 5 minutes
    CACHE_KEY_PREFIX = os.getenv("CACHE_KEY_PREFIX", "ticket_system:")
    
    # Performance settings
    QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", 30))
    CONNECTION_TIMEOUT = int(os.getenv("CONNECTION_TIMEOUT", 10))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 60))
    
    # Feature flags
    ENABLE_EMAIL_NOTIFICATIONS = os.getenv("ENABLE_EMAIL_NOTIFICATIONS", "True").lower() == "true"
    ENABLE_SMS_NOTIFICATIONS = os.getenv("ENABLE_SMS_NOTIFICATIONS", "False").lower() == "true"
    ENABLE_SLACK_NOTIFICATIONS = os.getenv("ENABLE_SLACK_NOTIFICATIONS", "False").lower() == "true"
    ENABLE_AUTO_ASSIGNMENT = os.getenv("ENABLE_AUTO_ASSIGNMENT", "True").lower() == "true"
    ENABLE_CUSTOMER_PORTAL = os.getenv("ENABLE_CUSTOMER_PORTAL", "False").lower() == "true"
    
    # External service settings
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
    
    # Monitoring settings
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "True").lower() == "true"
    METRICS_PORT = int(os.getenv("METRICS_PORT", 9090))
    
    # Rate limiting
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 60))
    RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", 1000))
    
    @classmethod
    def get_db_uri(cls):
        return f"postgresql+psycopg2://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DATABASE}"
    
    @classmethod
    def get_redis_url(cls):
        """Get Redis connection URL"""
        if cls.REDIS_PASSWORD:
            return f"redis://:{cls.REDIS_PASSWORD}@{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
        return f"redis://{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        # Check required settings
        if not cls.SECRET_KEY or cls.SECRET_KEY == "your-secret-key-here-change-in-production":
            errors.append("SECRET_KEY must be set to a secure value")
        
        if cls.EMAIL_ENABLED and not cls.EMAIL_USER:
            errors.append("EMAIL_USER must be set when EMAIL_ENABLED is True")
        
        if cls.ENVIRONMENT == "production":
            if cls.DEBUG:
                errors.append("DEBUG should be False in production")
            if not cls.EMAIL_PASSWORD:
                errors.append("EMAIL_PASSWORD must be set in production")
        
        return errors

# Validate configuration on import
config_errors = Config.validate()
if config_errors and Config.ENVIRONMENT == "production":
    raise ValueError(f"Configuration errors: {', '.join(config_errors)}")
