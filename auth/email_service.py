"""
Simple email service for magic links.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import logging
from .config import config

logger = logging.getLogger(__name__)

class EmailService:
    """Send authentication emails."""
    
    @staticmethod
    def send_magic_link(email: str, token: str) -> None:
        """
        Send magic link email.
        
        Args:
            email: Recipient email
            token: Magic link token
            
        Raises:
            Exception: If email sending fails
        """
        try:
            magic_link = f"{config.APP_URL}/auth/verify?token={token}"
            
            msg = MIMEMultipart()
            msg['From'] = config.FROM_EMAIL
            msg['To'] = email
            msg['Subject'] = "Sign in to MIRA"
            
            body = f"""
            <html>
            <body>
                <h2>Sign in to MIRA</h2>
                <p>Click the link below to sign in:</p>
                <p><a href="{magic_link}">Sign in to MIRA</a></p>
                <p>This link expires in 10 minutes.</p>
                <p>If you didn't request this, please ignore this email.</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
                if config.SMTP_USE_TLS:
                    server.starttls()
                server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
                server.send_message(msg)
            
            logger.info(f"Magic link sent to {email}")
            
        except Exception as e:
            logger.error(f"Failed to send email to {email}: {e}")
            # Fail closed - raise exception for email sending failures
            raise Exception(f"Email sending failed: {e}")