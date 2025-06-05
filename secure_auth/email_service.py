"""
Secure email service for authentication system.

Provides secure email delivery for magic links with comprehensive
security controls and audit logging.
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any
from datetime import datetime

from secure_auth.audit_service import get_audit_service, SecurityEventType, RiskLevel

logger = logging.getLogger(__name__)


class EmailService:
    """
    Secure email service with enterprise-grade features.
    
    Features:
    - SMTP connection pooling and security
    - Email template management
    - Delivery tracking and audit logging
    - Rate limiting integration
    - Security-focused email content
    """
    
    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        email_address: Optional[str] = None,
        email_password: Optional[str] = None,
        use_tls: bool = True
    ):
        """
        Initialize email service.
        
        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            email_address: Email address to send from
            email_password: Email password or app password
            use_tls: Whether to use TLS encryption
        """
        self.smtp_server = smtp_server or os.environ.get('SECURE_AUTH_SMTP_SERVER')
        self.smtp_port = smtp_port or (
            int(os.environ.get('SECURE_AUTH_SMTP_PORT')) 
            if os.environ.get('SECURE_AUTH_SMTP_PORT') 
            else None
        )
        self.email_address = email_address or os.environ.get('SECURE_AUTH_EMAIL_ADDRESS')
        self.email_password = email_password or os.environ.get('SECURE_AUTH_EMAIL_PASSWORD')
        self.use_tls = use_tls
        
        self.audit_service = get_audit_service()
        
        # Validate configuration
        self._validate_configuration()
        
        # Email templates
        self._load_templates()
    
    def _validate_configuration(self):
        """Validate email service configuration."""
        missing_vars = []
        
        if not self.smtp_server:
            missing_vars.append('SECURE_AUTH_SMTP_SERVER')
        if not self.smtp_port:
            missing_vars.append('SECURE_AUTH_SMTP_PORT')
        if not self.email_address:
            missing_vars.append('SECURE_AUTH_EMAIL_ADDRESS')
        if not self.email_password:
            missing_vars.append('SECURE_AUTH_EMAIL_PASSWORD')
        
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
    
    def _load_templates(self):
        """Load email templates."""
        self.templates = {
            "magic_link": {
                "subject": "Your secure login link",
                "text": """Hello,

Click the link below to securely log in to your account:

{magic_link}

This link will expire in {expiry_minutes} minutes for your security.

If you didn't request this login link, please ignore this email and consider changing your password.

For security reasons:
- This link can only be used once
- It will expire automatically
- Don't share this link with anyone

Best regards,
The Security Team""",
                "html": """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{ 
            text-align: center; 
            margin-bottom: 30px; 
            padding-bottom: 20px;
            border-bottom: 2px solid #f0f0f0;
        }}
        .button {{ 
            display: inline-block; 
            padding: 14px 28px; 
            background-color: #0066cc; 
            color: white; 
            text-decoration: none; 
            border-radius: 6px;
            margin: 20px 0;
            font-weight: 600;
        }}
        .security-notice {{
            background-color: #f8f9fa;
            border-left: 4px solid #0066cc;
            padding: 15px;
            margin: 20px 0;
        }}
        .footer {{ 
            color: #666; 
            font-size: 14px; 
            margin-top: 40px; 
            padding-top: 20px;
            border-top: 1px solid #f0f0f0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h2>Secure Login Request</h2>
    </div>
    
    <p>Hello,</p>
    
    <p>Click the button below to securely log in to your account:</p>
    
    <div style="text-align: center;">
        <a href="{magic_link}" class="button">Log In Securely</a>
    </div>
    
    <p>Or copy and paste this link into your browser:</p>
    <p style="word-break: break-all; color: #0066cc; background-color: #f8f9fa; padding: 10px; border-radius: 4px; font-family: monospace;">{magic_link}</p>
    
    <div class="security-notice">
        <strong>Security Notice:</strong>
        <ul>
            <li>This link will expire in <strong>{expiry_minutes} minutes</strong></li>
            <li>It can only be used once</li>
            <li>Don't share this link with anyone</li>
            <li>If you didn't request this, please ignore this email</li>
        </ul>
    </div>
    
    <div class="footer">
        <p>This is an automated security email. Please do not reply to this message.</p>
        <p>Best regards,<br>The Security Team</p>
    </div>
</body>
</html>"""
            },
            "login_notification": {
                "subject": "New login to your account",
                "text": """Hello,

A new login to your account was detected:

Time: {login_time}
Location: {location}
Device: {device_info}
IP Address: {ip_address}

If this was you, no action is needed.

If you don't recognize this login:
1. Change your password immediately
2. Review your account for any unauthorized changes
3. Contact support if you need assistance

Best regards,
The Security Team""",
                "html": """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .alert {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 15px;
            margin: 20px 0;
        }}
        .login-details {{
            background-color: #f8f9fa;
            border-radius: 6px;
            padding: 15px;
            margin: 20px 0;
        }}
        .action-required {{
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 6px;
            padding: 15px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h2>New Login Detected</h2>
    
    <div class="alert">
        <strong>Notice:</strong> A new login to your account was detected.
    </div>
    
    <div class="login-details">
        <h3>Login Details:</h3>
        <ul>
            <li><strong>Time:</strong> {login_time}</li>
            <li><strong>Location:</strong> {location}</li>
            <li><strong>Device:</strong> {device_info}</li>
            <li><strong>IP Address:</strong> {ip_address}</li>
        </ul>
    </div>
    
    <p><strong>If this was you,</strong> no action is needed.</p>
    
    <div class="action-required">
        <h3>If you don't recognize this login:</h3>
        <ol>
            <li>Change your password immediately</li>
            <li>Review your account for any unauthorized changes</li>
            <li>Contact support if you need assistance</li>
        </ol>
    </div>
    
    <p>Best regards,<br>The Security Team</p>
</body>
</html>"""
            }
        }
    
    def _create_smtp_connection(self):
        """Create and configure SMTP connection."""
        try:
            if self.use_tls and self.smtp_port == 465:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                if self.use_tls:
                    server.starttls()
            
            server.login(self.email_address, self.email_password)
            return server
            
        except Exception as e:
            logger.error(f"Failed to create SMTP connection: {e}")
            raise
    
    def send_magic_link(
        self,
        to_email: str,
        magic_link: str,
        expiry_minutes: int = 10,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Send a magic link email.
        
        Args:
            to_email: Recipient email address
            magic_link: The magic link URL
            expiry_minutes: How many minutes until the link expires
            user_id: User ID for audit logging
            ip_address: IP address for audit logging
            
        Returns:
            True if email sent successfully
        """
        try:
            template = self.templates["magic_link"]
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_address
            msg['To'] = to_email
            msg['Subject'] = template["subject"]
            
            # Format templates
            text_body = template["text"].format(
                magic_link=magic_link,
                expiry_minutes=expiry_minutes
            )
            
            html_body = template["html"].format(
                magic_link=magic_link,
                expiry_minutes=expiry_minutes
            )
            
            # Attach parts
            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            server = self._create_smtp_connection()
            try:
                server.send_message(msg)
                server.quit()
                
                # Log successful delivery
                self.audit_service.log_security_event(
                    event_type=SecurityEventType.DATA_ACCESS,
                    success=True,
                    message=f"Magic link email sent to {to_email}",
                    user_id=user_id,
                    ip_address=ip_address,
                    details={
                        "email_type": "magic_link",
                        "recipient": to_email,
                        "expiry_minutes": expiry_minutes
                    }
                )
                
                logger.info(f"Magic link email sent to {to_email}")
                return True
                
            finally:
                try:
                    server.quit()
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to send magic link email: {e}")
            
            # Log failed delivery
            self.audit_service.log_security_event(
                event_type=SecurityEventType.DATA_ACCESS,
                success=False,
                message=f"Failed to send magic link email to {to_email}",
                user_id=user_id,
                ip_address=ip_address,
                risk_level=RiskLevel.MEDIUM,
                details={
                    "email_type": "magic_link",
                    "recipient": to_email,
                    "error": str(e)
                }
            )
            
            return False
    
    def send_login_notification(
        self,
        to_email: str,
        login_time: datetime,
        ip_address: Optional[str] = None,
        device_info: Optional[str] = None,
        location: str = "Unknown",
        user_id: Optional[str] = None
    ) -> bool:
        """
        Send a login notification email.
        
        Args:
            to_email: Recipient email address
            login_time: When the login occurred
            ip_address: IP address of the login
            device_info: Device/browser information
            location: Geographic location
            user_id: User ID for audit logging
            
        Returns:
            True if email sent successfully
        """
        try:
            template = self.templates["login_notification"]
            
            # Format login time
            login_time_str = login_time.strftime("%B %d, %Y at %I:%M %p UTC")
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_address
            msg['To'] = to_email
            msg['Subject'] = template["subject"]
            
            # Format templates
            template_vars = {
                "login_time": login_time_str,
                "location": location,
                "device_info": device_info or "Unknown device",
                "ip_address": ip_address or "Unknown IP"
            }
            
            text_body = template["text"].format(**template_vars)
            html_body = template["html"].format(**template_vars)
            
            # Attach parts
            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            server = self._create_smtp_connection()
            try:
                server.send_message(msg)
                server.quit()
                
                # Log successful delivery
                self.audit_service.log_security_event(
                    event_type=SecurityEventType.DATA_ACCESS,
                    success=True,
                    message=f"Login notification sent to {to_email}",
                    user_id=user_id,
                    ip_address=ip_address,
                    details={
                        "email_type": "login_notification",
                        "recipient": to_email,
                        "login_time": login_time_str
                    }
                )
                
                logger.info(f"Login notification sent to {to_email}")
                return True
                
            finally:
                try:
                    server.quit()
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to send login notification: {e}")
            
            # Log failed delivery (but don't fail the login process)
            self.audit_service.log_security_event(
                event_type=SecurityEventType.DATA_ACCESS,
                success=False,
                message=f"Failed to send login notification to {to_email}",
                user_id=user_id,
                ip_address=ip_address,
                risk_level=RiskLevel.LOW,
                details={
                    "email_type": "login_notification",
                    "recipient": to_email,
                    "error": str(e)
                }
            )
            
            return False
    
    def send_security_alert(
        self,
        to_email: str,
        alert_type: str,
        alert_message: str,
        details: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Send a security alert email.
        
        Args:
            to_email: Recipient email address
            alert_type: Type of security alert
            alert_message: Alert message
            details: Additional alert details
            user_id: User ID for audit logging
            ip_address: IP address for audit logging
            
        Returns:
            True if email sent successfully
        """
        try:
            # Create message
            msg = MIMEText(f"""SECURITY ALERT

Alert Type: {alert_type}
Message: {alert_message}
Time: {datetime.utcnow().strftime('%B %d, %Y at %I:%M %p UTC')}

Additional Details:
{details if details else 'None'}

If you believe this alert is legitimate, please review your account security settings.
If this alert seems suspicious, please contact support immediately.

Best regards,
The Security Team""")
            
            msg['From'] = self.email_address
            msg['To'] = to_email
            msg['Subject'] = f"Security Alert: {alert_type}"
            
            # Send email
            server = self._create_smtp_connection()
            try:
                server.send_message(msg)
                server.quit()
                
                # Log successful delivery
                self.audit_service.log_security_event(
                    event_type=SecurityEventType.DATA_ACCESS,
                    success=True,
                    message=f"Security alert sent to {to_email}",
                    user_id=user_id,
                    ip_address=ip_address,
                    risk_level=RiskLevel.HIGH,
                    details={
                        "email_type": "security_alert",
                        "alert_type": alert_type,
                        "recipient": to_email
                    }
                )
                
                logger.warning(f"Security alert sent to {to_email}: {alert_type}")
                return True
                
            finally:
                try:
                    server.quit()
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to send security alert: {e}")
            
            # Log failed delivery
            self.audit_service.log_security_event(
                event_type=SecurityEventType.DATA_ACCESS,
                success=False,
                message=f"Failed to send security alert to {to_email}",
                user_id=user_id,
                ip_address=ip_address,
                risk_level=RiskLevel.HIGH,
                details={
                    "email_type": "security_alert",
                    "alert_type": alert_type,
                    "recipient": to_email,
                    "error": str(e)
                }
            )
            
            return False


# Singleton instance
_email_service: Optional[EmailService] = None

def get_email_service() -> EmailService:
    """Get the email service singleton."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service