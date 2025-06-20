"""
Token generation and validation with JWT + Redis refresh tokens.
"""

import secrets
import hashlib
import json
import logging
from datetime import timedelta
from typing import Optional, Dict, Any, Tuple, List
import jwt
from .config import config
from .redis_client import get_redis
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


class TokenService:
    """Handle JWT access tokens and Redis refresh tokens."""
    
    @staticmethod
    def generate_magic_link_token() -> Tuple[str, str]:
        """
        Generate magic link token.
        
        Returns:
            Tuple of (token, token_hash)
        """
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        return token, token_hash
    
    @staticmethod
    def generate_access_token(user_id: str, user_data: Dict[str, Any]) -> str:
        """
        Generate short-lived JWT access token (15 minutes).
        
        Args:
            user_id: User ID
            user_data: Additional user data to include
            
        Returns:
            JWT access token
        """
        payload = {
            "user_id": user_id,
            "email": user_data.get("email"),
            "tenant_id": user_data.get("tenant_id"),
            "exp": utc_now() + timedelta(seconds=config.JWT_ACCESS_TOKEN_EXPIRY),
            "iat": utc_now()
        }
        
        return jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)
    
    @staticmethod
    def generate_refresh_token() -> str:
        """
        Generate cryptographically secure refresh token.
        
        Returns:
            Refresh token string
        """
        return secrets.token_urlsafe(64)
    
    @staticmethod
    def generate_device_fingerprint(ip_address: str, user_agent: str) -> str:
        """
        Generate device fingerprint for tracking sessions.
        
        Args:
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Device fingerprint hash
        """
        fingerprint_data = f"{ip_address}:{user_agent}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
    
    @staticmethod
    async def store_refresh_token(
        refresh_token: str,
        user_id: str,
        device_fingerprint: str,
        ip_address: str,
        user_agent: str
    ) -> bool:
        """
        Store refresh token in Redis with metadata.
        
        Args:
            refresh_token: The refresh token
            user_id: User ID
            device_fingerprint: Device fingerprint
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            True if stored successfully
        """
        try:
            redis = await get_redis()
            
            token_data = {
                "user_id": user_id,
                "device_fingerprint": device_fingerprint,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "created_at": utc_now().isoformat(),
                "last_used_at": utc_now().isoformat()
            }
            
            # Store refresh token
            key = f"refresh:{refresh_token}"
            success = await redis.set_with_expiry(
                key, 
                json.dumps(token_data), 
                config.REFRESH_TOKEN_EXPIRY
            )
            
            if success:
                # Maintain user token index
                user_key = f"user_tokens:{user_id}"
                user_tokens = await redis.get(user_key)
                tokens = json.loads(user_tokens) if user_tokens else []
                
                tokens.append({
                    "token": refresh_token,
                    "device_fingerprint": device_fingerprint,
                    "created_at": utc_now().isoformat()
                })
                
                # Limit concurrent sessions
                if len(tokens) > config.MAX_REFRESH_TOKENS_PER_USER:
                    oldest_token = tokens.pop(0)
                    await redis.delete(f"refresh:{oldest_token['token']}")
                
                await redis.set_with_expiry(
                    user_key,
                    json.dumps(tokens),
                    config.REFRESH_TOKEN_EXPIRY
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing refresh token: {e}")
            # Fail closed - authentication fails if we can't store the token
            raise Exception(f"Authentication failed: Unable to store session data - {e}")
    
    @staticmethod
    async def validate_refresh_token(refresh_token: str) -> Optional[Dict[str, Any]]:
        """
        Validate refresh token and return metadata.
        
        Args:
            refresh_token: The refresh token to validate
            
        Returns:
            Token metadata if valid, None otherwise
        """
        try:
            redis = await get_redis()
            key = f"refresh:{refresh_token}"
            
            token_data_str = await redis.get(key)
            if not token_data_str:
                return None
            
            token_data = json.loads(token_data_str)
            
            # Update last used timestamp
            token_data["last_used_at"] = utc_now().isoformat()
            await redis.set_with_expiry(
                key,
                json.dumps(token_data),
                config.REFRESH_TOKEN_EXPIRY
            )
            
            return token_data
            
        except Exception as e:
            logger.error(f"Error validating refresh token: {e}")
            # Fail closed - deny access if we can't validate the token
            return None
    
    @staticmethod
    async def revoke_refresh_token(refresh_token: str) -> bool:
        """
        Revoke a specific refresh token.
        
        Args:
            refresh_token: The refresh token to revoke
            
        Returns:
            True if revoked successfully
        """
        try:
            redis = await get_redis()
            
            # Get token data to update user index
            key = f"refresh:{refresh_token}"
            token_data_str = await redis.get(key)
            
            if token_data_str:
                token_data = json.loads(token_data_str)
                user_id = token_data["user_id"]
                
                # Remove from user index
                user_key = f"user_tokens:{user_id}"
                user_tokens = await redis.get(user_key)
                if user_tokens:
                    tokens = json.loads(user_tokens)
                    tokens = [t for t in tokens if t["token"] != refresh_token]
                    await redis.set_with_expiry(
                        user_key,
                        json.dumps(tokens),
                        config.REFRESH_TOKEN_EXPIRY
                    )
            
            return await redis.delete(key)
            
        except Exception as e:
            logger.error(f"Error revoking refresh token: {e}")
            # Fail closed - raise exception to indicate logout failed
            raise Exception(f"Logout failed: Unable to revoke session - {e}")
    
    @staticmethod
    async def revoke_all_user_tokens(user_id: str) -> int:
        """
        Revoke all refresh tokens for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of tokens revoked
        """
        try:
            redis = await get_redis()
            user_key = f"user_tokens:{user_id}"
            
            user_tokens = await redis.get(user_key)
            if not user_tokens:
                return 0
            
            tokens = json.loads(user_tokens)
            revoked_count = 0
            
            for token_info in tokens:
                token_key = f"refresh:{token_info['token']}"
                if await redis.delete(token_key):
                    revoked_count += 1
            
            await redis.delete(user_key)
            return revoked_count
            
        except Exception as e:
            logger.error(f"Error revoking all user tokens: {e}")
            # Fail closed - raise exception to indicate logout-all failed
            raise Exception(f"Logout-all failed: Unable to revoke sessions - {e}")
    
    @staticmethod
    async def rotate_refresh_token(
        old_token: str,
        user_data: Dict[str, Any],
        device_fingerprint: str,
        ip_address: str,
        user_agent: str
    ) -> Optional[Tuple[str, str]]:
        """
        Rotate refresh token (if rotation enabled).
        
        Args:
            old_token: Current refresh token
            user_data: User data for new access token
            device_fingerprint: Device fingerprint
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Tuple of (new_access_token, new_refresh_token) if successful
        """
        try:
            # Validate old token
            token_data = await TokenService.validate_refresh_token(old_token)
            if not token_data:
                return None
            
            user_id = token_data["user_id"]
            
            # Generate new tokens
            new_access_token = TokenService.generate_access_token(user_id, user_data)
            new_refresh_token = TokenService.generate_refresh_token()
            
            # Store new refresh token
            if await TokenService.store_refresh_token(
                new_refresh_token,
                user_id,
                device_fingerprint,
                ip_address,
                user_agent
            ):
                # Revoke old token
                await TokenService.revoke_refresh_token(old_token)
                return new_access_token, new_refresh_token
            
            return None
            
        except Exception as e:
            logger.error(f"Error rotating refresh token: {e}")
            # Fail closed - token refresh fails if Redis is unavailable
            return None
    
    @staticmethod
    def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode access token.
        
        Args:
            token: JWT access token
            
        Returns:
            Decoded payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token, 
                config.JWT_SECRET_KEY, 
                algorithms=[config.JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    @staticmethod
    def hash_token(token: str) -> str:
        """Hash a token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()