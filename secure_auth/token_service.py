"""
Secure token generation and validation service.

Provides cryptographically secure token operations with enhanced entropy,
proper salt handling, and timing attack protection.
"""

import os
import hmac
import time
import hashlib
import secrets
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
import logging

logger = logging.getLogger(__name__)


class TokenType:
    """Token type constants for different use cases."""
    MAGIC_LINK = "magic_link"
    SESSION = "session"
    CSRF = "csrf"
    API_KEY = "api_key"
    REFRESH = "refresh"


class SecurityTokenService:
    """
    Enhanced cryptographic token service with enterprise-grade security.
    
    Features:
    - Cryptographically secure random token generation
    - PBKDF2 key derivation with high iteration counts
    - Timing attack protection
    - Token metadata and context binding
    - Configurable entropy levels
    """
    
    def __init__(self):
        """Initialize the token service with secure defaults."""
        self._master_key = self._get_master_key()
        self._default_iterations = 100000  # OWASP recommended minimum
        self._salt_length = 32  # 256 bits
        self._token_length = 64  # 512 bits
        
    def _get_master_key(self) -> bytes:
        """
        Get or generate the master key for token operations.
        
        Returns:
            Master key bytes
            
        Raises:
            ValueError: If AUTH_MASTER_KEY environment variable is missing
        """
        key_hex = os.environ.get('AUTH_MASTER_KEY')
        if not key_hex:
            raise ValueError(
                "AUTH_MASTER_KEY environment variable is required. "
                "Generate with: python -c 'import secrets; print(secrets.token_hex(32))'"
            )
        
        try:
            return bytes.fromhex(key_hex)
        except ValueError:
            raise ValueError("AUTH_MASTER_KEY must be a valid hex string (64 characters)")
    
    def generate_token(
        self,
        token_type: str = TokenType.MAGIC_LINK,
        entropy_bytes: int = 64,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Generate a cryptographically secure token with salt.
        
        Args:
            token_type: Type of token being generated
            entropy_bytes: Number of random bytes for entropy (default 64 = 512 bits)
            metadata: Optional metadata to bind to token
            
        Returns:
            Tuple of (token, salt_hex)
        """
        # Generate high-entropy token
        token_bytes = secrets.token_bytes(entropy_bytes)
        
        # Add metadata binding if provided
        if metadata:
            metadata_bytes = str(sorted(metadata.items())).encode('utf-8')
            token_bytes = hashlib.sha256(token_bytes + metadata_bytes).digest()
        
        # Convert to URL-safe base64
        token = secrets.token_urlsafe(len(token_bytes))
        
        # Generate salt for hashing
        salt = secrets.token_bytes(self._salt_length)
        salt_hex = salt.hex()
        
        logger.debug(f"Generated {token_type} token with {entropy_bytes * 8} bits entropy")
        
        return token, salt_hex
    
    def hash_token(
        self,
        token: str,
        salt_hex: str,
        iterations: Optional[int] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Hash a token using PBKDF2 with salt and optional context binding.
        
        Args:
            token: Plain text token
            salt_hex: Salt in hex format
            iterations: Number of PBKDF2 iterations (default: 100000)
            additional_context: Additional context for binding
            
        Returns:
            Hashed token as hex string
        """
        if iterations is None:
            iterations = self._default_iterations
            
        salt = bytes.fromhex(salt_hex)
        token_bytes = token.encode('utf-8')
        
        # Add context binding if provided
        if additional_context:
            context_bytes = additional_context.encode('utf-8')
            token_bytes = hashlib.sha256(token_bytes + context_bytes).digest()
        
        # Use PBKDF2 with master key as password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        
        # Derive key from master key and token
        derived_key = kdf.derive(self._master_key + token_bytes)
        
        return derived_key.hex()
    
    def verify_token(
        self,
        token: str,
        stored_hash: str,
        salt_hex: str,
        iterations: Optional[int] = None,
        additional_context: Optional[str] = None
    ) -> bool:
        """
        Verify a token against its stored hash with timing attack protection.
        
        Args:
            token: Plain text token to verify
            stored_hash: Stored hash to compare against
            salt_hex: Salt used for hashing
            iterations: Number of PBKDF2 iterations used
            additional_context: Additional context used for binding
            
        Returns:
            True if token is valid
        """
        try:
            # Always compute hash to prevent timing attacks
            computed_hash = self.hash_token(
                token, salt_hex, iterations, additional_context
            )
            
            # Use constant-time comparison
            return hmac.compare_digest(computed_hash, stored_hash)
            
        except Exception as e:
            logger.warning(f"Token verification failed: {e}")
            return False
    
    def generate_session_context_hash(
        self,
        ip_address: Optional[str],
        user_agent: Optional[str],
        device_fingerprint: Optional[str] = None,
        additional_factors: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a context hash for session binding.
        
        Args:
            ip_address: Client IP address
            user_agent: Client user agent
            device_fingerprint: Device fingerprint
            additional_factors: Additional context factors
            
        Returns:
            Context hash as hex string
        """
        context_parts = []
        
        if ip_address:
            context_parts.append(f"ip:{ip_address}")
        if user_agent:
            context_parts.append(f"ua:{user_agent}")
        if device_fingerprint:
            context_parts.append(f"fp:{device_fingerprint}")
            
        if additional_factors:
            for key, value in sorted(additional_factors.items()):
                context_parts.append(f"{key}:{value}")
        
        context_string = "|".join(context_parts)
        
        # Hash with master key for additional security
        hmac_obj = hmac.new(
            self._master_key,
            context_string.encode('utf-8'),
            hashlib.sha256
        )
        
        return hmac_obj.hexdigest()
    
    def generate_device_fingerprint(
        self,
        user_agent: Optional[str],
        screen_resolution: Optional[str] = None,
        timezone: Optional[str] = None,
        language: Optional[str] = None,
        additional_attributes: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a device fingerprint from browser characteristics.
        
        Args:
            user_agent: Browser user agent
            screen_resolution: Screen resolution (e.g., "1920x1080")
            timezone: Client timezone
            language: Client language preference
            additional_attributes: Additional fingerprint attributes
            
        Returns:
            Device fingerprint as hex string
        """
        fingerprint_parts = []
        
        if user_agent:
            fingerprint_parts.append(f"ua:{user_agent}")
        if screen_resolution:
            fingerprint_parts.append(f"res:{screen_resolution}")
        if timezone:
            fingerprint_parts.append(f"tz:{timezone}")
        if language:
            fingerprint_parts.append(f"lang:{language}")
            
        if additional_attributes:
            for key, value in sorted(additional_attributes.items()):
                fingerprint_parts.append(f"{key}:{value}")
        
        fingerprint_string = "|".join(fingerprint_parts)
        
        # Use SHA-256 for device fingerprint
        return hashlib.sha256(fingerprint_string.encode('utf-8')).hexdigest()
    
    def encrypt_sensitive_data(
        self,
        data: str,
        additional_data: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Encrypt sensitive data using AES-GCM.
        
        Args:
            data: Data to encrypt
            additional_data: Additional authenticated data (AAD)
            
        Returns:
            Tuple of (encrypted_data_hex, nonce_hex)
        """
        # Generate random nonce
        nonce = secrets.token_bytes(12)  # 96 bits for GCM
        
        # Derive encryption key from master key
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=10000,  # Lower iterations for performance
            backend=default_backend()
        )
        key = kdf.derive(self._master_key)
        
        # Encrypt using AES-GCM
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        if additional_data:
            encryptor.authenticate_additional_data(additional_data.encode('utf-8'))
        
        ciphertext = encryptor.update(data.encode('utf-8')) + encryptor.finalize()
        
        # Combine salt, nonce, tag, and ciphertext
        encrypted_data = salt + nonce + encryptor.tag + ciphertext
        
        return encrypted_data.hex(), nonce.hex()
    
    def decrypt_sensitive_data(
        self,
        encrypted_data_hex: str,
        additional_data: Optional[str] = None
    ) -> Optional[str]:
        """
        Decrypt sensitive data encrypted with encrypt_sensitive_data.
        
        Args:
            encrypted_data_hex: Encrypted data as hex string
            additional_data: Additional authenticated data (AAD) used during encryption
            
        Returns:
            Decrypted data or None if decryption fails
        """
        try:
            encrypted_data = bytes.fromhex(encrypted_data_hex)
            
            # Extract components
            salt = encrypted_data[:16]
            nonce = encrypted_data[16:28]
            tag = encrypted_data[28:44]
            ciphertext = encrypted_data[44:]
            
            # Derive decryption key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=10000,
                backend=default_backend()
            )
            key = kdf.derive(self._master_key)
            
            # Decrypt using AES-GCM
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            if additional_data:
                decryptor.authenticate_additional_data(additional_data.encode('utf-8'))
            
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return plaintext.decode('utf-8')
            
        except Exception as e:
            logger.warning(f"Data decryption failed: {e}")
            return None
    
    def generate_csrf_token_pair(self) -> Tuple[str, str]:
        """
        Generate a CSRF token pair (value and hash) for double-submit cookie pattern.
        
        Returns:
            Tuple of (token_value, token_hash)
        """
        token_value = secrets.token_urlsafe(32)
        
        # Hash with HMAC for verification
        token_hash = hmac.new(
            self._master_key,
            token_value.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return token_value, token_hash
    
    def verify_csrf_token_pair(self, token_value: str, token_hash: str) -> bool:
        """
        Verify a CSRF token pair with timing attack protection.
        
        Args:
            token_value: Token value from form/header
            token_hash: Token hash from cookie/storage
            
        Returns:
            True if CSRF token pair is valid
        """
        try:
            expected_hash = hmac.new(
                self._master_key,
                token_value.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(expected_hash, token_hash)
            
        except Exception as e:
            logger.warning(f"CSRF token verification failed: {e}")
            return False
    
    def get_token_expiry(
        self,
        token_type: str,
        remember_device: bool = False
    ) -> datetime:
        """
        Get appropriate expiry time for different token types.
        
        Args:
            token_type: Type of token
            remember_device: Whether device should be remembered
            
        Returns:
            Expiry datetime
        """
        now = datetime.utcnow()
        
        if token_type == TokenType.MAGIC_LINK:
            return now + timedelta(minutes=10)
        elif token_type == TokenType.SESSION:
            if remember_device:
                return now + timedelta(days=30)
            else:
                return now + timedelta(hours=8)
        elif token_type == TokenType.CSRF:
            return now + timedelta(hours=12)
        elif token_type == TokenType.API_KEY:
            return now + timedelta(days=365)
        elif token_type == TokenType.REFRESH:
            return now + timedelta(days=7)
        else:
            return now + timedelta(hours=1)


# Singleton instance
_token_service: Optional[SecurityTokenService] = None

def get_token_service() -> SecurityTokenService:
    """Get the token service singleton."""
    global _token_service
    if _token_service is None:
        _token_service = SecurityTokenService()
    return _token_service