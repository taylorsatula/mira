"""
WebAuthn/TouchID support.
"""

import json
import base64
import logging
from typing import Dict, Any, Optional
from webauthn import generate_registration_options, verify_registration_response
from webauthn import generate_authentication_options, verify_authentication_response
from webauthn.helpers.structs import PublicKeyCredentialDescriptor
from .config import config

logger = logging.getLogger(__name__)

class WebAuthnService:
    """Handle WebAuthn/TouchID authentication."""
    
    @staticmethod
    def generate_registration_options(user_id: str, email: str) -> Dict[str, Any]:
        """
        Generate WebAuthn registration options.
        
        Args:
            user_id: User ID
            email: User email
            
        Returns:
            Registration options for client
        """
        options = generate_registration_options(
            rp_id=config.WEBAUTHN_RP_ID,
            rp_name=config.WEBAUTHN_RP_NAME,
            user_id=user_id.encode(),
            user_name=email,
            user_display_name=email,
            attestation="none",
            authenticator_attachment="platform",  # Prefer platform authenticators (TouchID)
            resident_key="required"
        )
        
        # Convert to JSON-serializable format
        return {
            "challenge": base64.b64encode(options.challenge).decode(),
            "rp": {"id": options.rp.id, "name": options.rp.name},
            "user": {
                "id": base64.b64encode(options.user.id).decode(),
                "name": options.user.name,
                "displayName": options.user.display_name
            },
            "pubKeyCredParams": [
                {"type": "public-key", "alg": param.alg}
                for param in options.pub_key_cred_params
            ],
            "authenticatorSelection": {
                "authenticatorAttachment": "platform",
                "userVerification": "required"
            },
            "attestation": "none"
        }
    
    @staticmethod
    def verify_registration(
        credential: Dict[str, Any],
        challenge: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Verify registration response.
        
        Args:
            credential: Credential from client
            challenge: Expected challenge
            user_id: User ID
            
        Returns:
            Credential data to store or None if invalid
        """
        try:
            verification = verify_registration_response(
                credential=credential,
                expected_challenge=base64.b64decode(challenge),
                expected_origin=config.WEBAUTHN_ORIGIN,
                expected_rp_id=config.WEBAUTHN_RP_ID
            )
            
            if verification.verified:
                return {
                    "credential_id": base64.b64encode(
                        verification.credential_id
                    ).decode(),
                    "public_key": base64.b64encode(
                        verification.credential_public_key
                    ).decode(),
                    "sign_count": verification.sign_count,
                    "transports": credential.get("transports", [])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"WebAuthn registration verification failed: {e}")
            return None
    
    @staticmethod
    def generate_authentication_options(
        credentials: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate authentication options.
        
        Args:
            credentials: User's stored credentials
            
        Returns:
            Authentication options for client
        """
        allow_credentials = []
        for cred_id, cred_data in credentials.items():
            allow_credentials.append(
                PublicKeyCredentialDescriptor(
                    id=base64.b64decode(cred_id),
                    transports=cred_data.get("transports", [])
                )
            )
        
        options = generate_authentication_options(
            rp_id=config.WEBAUTHN_RP_ID,
            allow_credentials=allow_credentials,
            user_verification="required"
        )
        
        return {
            "challenge": base64.b64encode(options.challenge).decode(),
            "allowCredentials": [
                {
                    "type": "public-key",
                    "id": base64.b64encode(cred.id).decode(),
                    "transports": cred.transports
                }
                for cred in options.allow_credentials
            ],
            "userVerification": "required"
        }
    
    @staticmethod
    def verify_authentication(
        credential: Dict[str, Any],
        challenge: str,
        stored_credential: Dict[str, Any]
    ) -> bool:
        """
        Verify authentication response.
        
        Args:
            credential: Credential from client
            challenge: Expected challenge
            stored_credential: Stored credential data
            
        Returns:
            True if authentication is valid
        """
        try:
            verification = verify_authentication_response(
                credential=credential,
                expected_challenge=base64.b64decode(challenge),
                expected_origin=config.WEBAUTHN_ORIGIN,
                expected_rp_id=config.WEBAUTHN_RP_ID,
                credential_public_key=base64.b64decode(
                    stored_credential["public_key"]
                ),
                credential_current_sign_count=stored_credential["sign_count"]
            )
            
            if verification.verified:
                # Update sign count
                stored_credential["sign_count"] = verification.new_sign_count
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"WebAuthn authentication verification failed: {e}")
            return False