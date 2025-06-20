"""
Configuration for authentication system.
"""

import os
from typing import Optional

class AuthConfig:
    """Simple configuration using environment variables."""
    
    # Database
    DATABASE_URL: str = os.environ["DATABASE_URL"]
    
    # JWT Settings
    JWT_SECRET_KEY: str = os.environ["JWT_SECRET_KEY"]
    JWT_ALGORITHM: str = "HS256"
    
    # Token Expiration (in seconds)
    MAGIC_LINK_EXPIRY: int = 600  # 10 minutes
    JWT_ACCESS_TOKEN_EXPIRY: int = 900  # 15 minutes
    REFRESH_TOKEN_EXPIRY: int = 86400 * 7  # 7 days
    SESSION_EXPIRY: int = 86400 * 7  # 7 days (backward compatibility)
    
    # Email Settings
    SMTP_HOST: str = os.environ["SMTP_HOST"]
    SMTP_PORT: int = int(os.environ["SMTP_PORT"])
    SMTP_USERNAME: str = os.environ["SMTP_USERNAME"]
    SMTP_PASSWORD: str = os.environ["SMTP_PASSWORD"]
    SMTP_USE_TLS: bool = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
    FROM_EMAIL: str = os.environ["FROM_EMAIL"]
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 5
    RATE_LIMIT_WINDOW: int = 300  # 5 minutes
    REDIS_URL: str = os.environ["REDIS_URL"]
    
    # Refresh Token Settings
    REFRESH_TOKEN_ROTATION: bool = True  # Enable refresh token rotation
    MAX_REFRESH_TOKENS_PER_USER: int = 5  # Limit concurrent sessions
    
    # WebAuthn/TouchID
    WEBAUTHN_RP_ID: str = os.environ["WEBAUTHN_RP_ID"]
    WEBAUTHN_RP_NAME: str = os.environ["WEBAUTHN_RP_NAME"]
    WEBAUTHN_ORIGIN: str = os.environ["WEBAUTHN_ORIGIN"]
    
    # Application
    APP_URL: str = os.environ["APP_URL"]
    
    @classmethod
    def validate(cls):
        """Validate required environment variables."""
        required = [
            "DATABASE_URL",
            "JWT_SECRET_KEY", 
            "SMTP_HOST",
            "SMTP_PORT",
            "SMTP_USERNAME",
            "SMTP_PASSWORD",
            "FROM_EMAIL",
            "REDIS_URL",
            "WEBAUTHN_RP_ID",
            "WEBAUTHN_RP_NAME", 
            "WEBAUTHN_ORIGIN",
            "APP_URL"
        ]
        
        missing = [var for var in required if not os.environ.get(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Create instance
config = AuthConfig()