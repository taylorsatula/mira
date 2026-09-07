"""
WebAuthn service for biometric authentication.
"""

import json
import logging
import secrets
from typing import Optional, Dict, Any, List

from webauthn import (
    generate_registration_options,
    verify_registration_response,
    generate_authentication_options,
    verify_authentication_response,
    options_to_json
)
from webauthn.helpers import base64url_to_bytes, bytes_to_base64url
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    UserVerificationRequirement,
    PublicKeyCredentialDescriptor,
    AuthenticatorTransport,
    AuthenticatorAttachment,
    ResidentKeyRequirement
)

from utils.timezone_utils import utc_now
from utils.user_context import get_current_user_id
from clients.valkey_client import get_valkey
from .config import config
from .database import AuthDatabase
from .exceptions import AuthError

logger = logging.getLogger(__name__)


class WebAuthnService:
    """Service for WebAuthn biometric authentication."""

    def __init__(self):
        self.db = AuthDatabase()
        self.rp_id = self._get_rp_id()
        self.rp_name = "MIRA"
        self.origin = config.APP_URL
        self.challenge_prefix = "webauthn_challenge:"
        self.challenge_ttl = 300  # 5 minutes

    def _get_rp_id(self) -> str:
        """Extract RP ID from app URL."""
        # For development, use localhost
        if "localhost" in config.APP_URL or "127.0.0.1" in config.APP_URL:
            return "localhost"

        # For production, extract domain
        from urllib.parse import urlparse
        parsed = urlparse(config.APP_URL)
        if not parsed.hostname:
            raise ValueError(f"Cannot extract hostname from APP_URL: {config.APP_URL}")
        return parsed.hostname

    def _challenge_key(self, user_id: str, operation: str) -> str:
        """Build the Valkey challenge key for a user and operation."""
        return f"{self.challenge_prefix}{user_id}:{operation}"

    def _discoverable_challenge_key(self, challenge_id: str) -> str:
        """Build the Valkey challenge key for a discoverable (emailless) login ceremony."""
        return f"{self.challenge_prefix}discoverable:{challenge_id}"

    def _store_challenge(
        self,
        challenge: bytes,
        operation: str,
        user_id: Optional[str] = None
    ) -> None:
        """Store challenge in Valkey with expiration."""
        try:
            challenge_user_id = user_id or get_current_user_id()
            valkey = get_valkey()
            data = {
                "challenge": challenge.hex(),
                "created_at": utc_now().isoformat()
            }
            valkey.json_set_with_expiry(
                self._challenge_key(challenge_user_id, operation),
                "$",
                data,
                self.challenge_ttl
            )
        except Exception as e:
            logger.error(f"Failed to store challenge: {e}", exc_info=True)
            raise AuthError("internal_error", "Failed to store challenge")

    def _get_challenge(self, operation: str, user_id: Optional[str] = None) -> Optional[bytes]:
        """
        Retrieve challenge from Valkey.

        Returns None if challenge doesn't exist or is corrupted (legitimate cases).
        Raises exception if infrastructure fails.
        """
        challenge_user_id = user_id or get_current_user_id()
        valkey = get_valkey()
        key = self._challenge_key(challenge_user_id, operation)
        data = valkey.json_get(key, "$")

        if not data or len(data) == 0:
            return None

        challenge_hex = data[0].get("challenge")
        if not challenge_hex:
            return None

        # Delete challenge after retrieval (one-time use)
        valkey.delete(key)

        return bytes.fromhex(challenge_hex)

    def generate_registration_options(self, email: str) -> Dict[str, Any]:
        """Generate WebAuthn registration options."""
        try:
            user_id = get_current_user_id()
            # Check if user exists
            user = self.db.get_user_by_id(user_id)
            if not user:
                raise AuthError("user_not_found", "User not found")

            # Get existing credentials to exclude. Credentials are keyed by
            # their base64url credential ID — the exact encoding browsers send
            # back in ceremonies.
            existing_creds = []
            if user.webauthn_credentials:
                for cred_id, cred_data in user.webauthn_credentials.items():
                    existing_creds.append(
                        PublicKeyCredentialDescriptor(
                            id=base64url_to_bytes(cred_id),
                            transports=[AuthenticatorTransport.INTERNAL]
                        )
                    )

            # Generate registration options
            options = generate_registration_options(
                rp_id=self.rp_id,
                rp_name=self.rp_name,
                user_id=user_id.encode('utf-8'),
                user_name=email,
                user_display_name=email,
                authenticator_selection=AuthenticatorSelectionCriteria(
                    authenticator_attachment=AuthenticatorAttachment.PLATFORM,
                    resident_key=ResidentKeyRequirement.REQUIRED,
                    user_verification=UserVerificationRequirement.REQUIRED
                ),
                exclude_credentials=existing_creds
            )

            # Store challenge
            self._store_challenge(options.challenge, "register")

            # Convert to JSON-serializable format
            return json.loads(options_to_json(options))

        except AuthError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate registration options: {e}", exc_info=True)
            raise AuthError("internal_error", "Failed to generate registration options")

    def verify_registration(
        self,
        credential_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify registration response and store credential."""
        try:
            user_id = get_current_user_id()
            # Get stored challenge
            expected_challenge = self._get_challenge("register")
            if not expected_challenge:
                raise AuthError("invalid_challenge", "Challenge expired or not found")

            # Verify registration
            verification = verify_registration_response(
                credential=credential_json,
                expected_challenge=expected_challenge,
                expected_origin=self.origin,
                expected_rp_id=self.rp_id
            )

            # Get current user data
            user = self.db.get_user_by_id(user_id)
            current_creds = user.webauthn_credentials or {}

            # Prepare credential data for storage, keyed by the base64url
            # credential ID the browser returns in every ceremony.
            credential_id = bytes_to_base64url(verification.credential_id)
            credential_data = {
                "public_key": verification.credential_public_key.hex(),
                "credential_id": credential_id,
                "sign_count": verification.sign_count,
                "aaguid": verification.aaguid,
                "created_at": utc_now().isoformat(),
                "credential_device_type": verification.credential_device_type.value,
                "credential_backed_up": verification.credential_backed_up,
                "name": f"Biometric Device {credential_id[:8]}"
            }

            # Add new credential
            current_creds[credential_data["credential_id"]] = credential_data

            # Update user credentials
            self.db.update_webauthn_credentials(user_id, current_creds)

            return {
                "verified": True,
                "credential_id": credential_data["credential_id"]
            }

        except AuthError:
            raise
        except Exception as e:
            logger.error(f"Failed to verify registration: {e}", exc_info=True)
            raise AuthError("internal_error", "Failed to verify registration")

    def generate_authentication_options(self, email: str) -> Dict[str, Any]:
        """Generate WebAuthn authentication options."""
        try:
            # Get user by email
            user = self.db.get_user_by_email(email)
            if not user:
                raise AuthError("user_not_found", "User not found")

            # Get user's credentials
            user_creds = user.webauthn_credentials or {}
            if not user_creds:
                raise AuthError("no_credentials", "No biometric credentials registered")

            # Build allowed credentials list
            allow_credentials = []
            for cred_id, cred_data in user_creds.items():
                allow_credentials.append(
                    PublicKeyCredentialDescriptor(
                        id=base64url_to_bytes(cred_id),
                        transports=[AuthenticatorTransport.INTERNAL]
                    )
                )

            # Generate authentication options
            options = generate_authentication_options(
                rp_id=self.rp_id,
                allow_credentials=allow_credentials,
                user_verification=UserVerificationRequirement.REQUIRED
            )

            # Store challenge under the authenticating user because login occurs
            # before request context exists.
            self._store_challenge(options.challenge, "authenticate", str(user.id))

            # Convert to JSON-serializable format
            return json.loads(options_to_json(options))

        except AuthError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate authentication options: {e}", exc_info=True)
            raise AuthError("internal_error", "Failed to generate authentication options")

    def generate_discoverable_authentication_options(self) -> Dict[str, Any]:
        """
        Generate WebAuthn authentication options for a discoverable login.

        No email is supplied: the browser offers every passkey registered for
        this RP on the device, and the response's userHandle identifies the
        user. The options dict carries an extra ``challenge_id`` the client
        must echo back on completion, because no user is known yet to key the
        challenge by.
        """
        try:
            options = generate_authentication_options(
                rp_id=self.rp_id,
                user_verification=UserVerificationRequirement.REQUIRED
            )

            challenge_id = secrets.token_hex(16)
            self._store_discoverable_challenge(options.challenge, challenge_id)

            result = json.loads(options_to_json(options))
            result["challenge_id"] = challenge_id
            return result

        except AuthError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate discoverable authentication options: {e}", exc_info=True)
            raise AuthError("internal_error", "Failed to generate authentication options")

    def _store_discoverable_challenge(self, challenge: bytes, challenge_id: str) -> None:
        """Store a discoverable-login challenge in Valkey keyed by its opaque ID."""
        try:
            valkey = get_valkey()
            data = {
                "challenge": challenge.hex(),
                "created_at": utc_now().isoformat()
            }
            valkey.json_set_with_expiry(
                self._discoverable_challenge_key(challenge_id),
                "$",
                data,
                self.challenge_ttl
            )
        except Exception as e:
            logger.error(f"Failed to store discoverable challenge: {e}", exc_info=True)
            raise AuthError("internal_error", "Failed to store challenge")

    def _get_discoverable_challenge(self, challenge_id: str) -> Optional[bytes]:
        """
        Retrieve a discoverable-login challenge by its opaque ID.

        Returns None when the challenge expired or never existed (legitimate
        cases). Raises on infrastructure failure.
        """
        # Challenge IDs are hex; reject other shapes before building a key.
        try:
            bytes.fromhex(challenge_id)
        except ValueError:
            return None

        valkey = get_valkey()
        key = self._discoverable_challenge_key(challenge_id)
        data = valkey.json_get(key, "$")

        if not data or len(data) == 0:
            return None

        challenge_hex = data[0].get("challenge")
        if not challenge_hex:
            return None

        # Delete challenge after retrieval (one-time use)
        valkey.delete(key)

        return bytes.fromhex(challenge_hex)

    def _verify_credential_response(
        self,
        user,
        credential_json: Dict[str, Any],
        expected_challenge: bytes
    ) -> None:
        """
        Verify one authentication response against the user's stored
        credentials and record the new sign count. Shared by the email-keyed
        and discoverable login paths.
        """
        credential_id = credential_json.get("id", "")

        user_creds = user.webauthn_credentials or {}
        stored_cred = user_creds.get(credential_id)

        if not stored_cred:
            raise AuthError("unknown_credential", "Unknown credential")

        verification = verify_authentication_response(
            credential=credential_json,
            expected_challenge=expected_challenge,
            expected_origin=self.origin,
            expected_rp_id=self.rp_id,
            credential_public_key=bytes.fromhex(stored_cred["public_key"]),
            credential_current_sign_count=stored_cred["sign_count"]
        )

        stored_cred["sign_count"] = verification.new_sign_count
        stored_cred["last_used_at"] = utc_now().isoformat()
        user_creds[credential_id] = stored_cred

        self.db.update_webauthn_credentials(str(user.id), user_creds)

    def verify_authentication(
        self,
        email: str,
        credential_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify authentication response."""
        try:
            # Get user by email
            user = self.db.get_user_by_email(email)
            if not user:
                raise AuthError("user_not_found", "User not found")

            user_id = str(user.id)

            # Get stored challenge
            expected_challenge = self._get_challenge("authenticate", user_id)
            if not expected_challenge:
                raise AuthError("invalid_challenge", "Challenge expired or not found")

            self._verify_credential_response(user, credential_json, expected_challenge)

            # Update last login
            self.db.update_user_login(user_id)

            return {
                "verified": True,
                "user_id": user_id,
                "email": email
            }

        except AuthError:
            raise
        except Exception as e:
            logger.error(f"Failed to verify authentication: {e}", exc_info=True)
            raise AuthError("internal_error", "Failed to verify authentication")

    def verify_discoverable_authentication(
        self,
        challenge_id: str,
        credential_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify a discoverable authentication response. The user is identified
        by the credential's userHandle rather than a supplied email.
        """
        try:
            expected_challenge = self._get_discoverable_challenge(challenge_id)
            if not expected_challenge:
                raise AuthError("invalid_challenge", "Challenge expired or not found")

            response = credential_json.get("response")
            user_handle = response.get("userHandle") if isinstance(response, dict) else None
            if not user_handle:
                raise AuthError("unknown_credential", "Credential did not disclose a user handle")

            user_id = base64url_to_bytes(user_handle).decode("utf-8")

            user = self.db.get_user_by_id(user_id)
            if not user:
                raise AuthError("user_not_found", "User not found")

            self._verify_credential_response(user, credential_json, expected_challenge)

            # Update last login
            self.db.update_user_login(user_id)

            return {
                "verified": True,
                "user_id": user_id,
                "email": user.email
            }

        except AuthError:
            raise
        except Exception as e:
            logger.error(f"Failed to verify discoverable authentication: {e}", exc_info=True)
            raise AuthError("internal_error", "Failed to verify authentication")

    def remove_credential(self, credential_id: str) -> bool:
        """Remove a WebAuthn credential."""
        try:
            user_id = get_current_user_id()
            # Get current user data
            user = self.db.get_user_by_id(user_id)
            if not user:
                raise AuthError("user_not_found", "User not found")

            current_creds = user.webauthn_credentials or {}

            # Check if credential exists
            if credential_id not in current_creds:
                raise AuthError("credential_not_found", "Credential not found")

            # Remove credential
            del current_creds[credential_id]

            # Update user credentials
            self.db.update_webauthn_credentials(user_id, current_creds)

            return True

        except AuthError:
            raise
        except Exception as e:
            logger.error(f"Failed to remove credential: {e}", exc_info=True)
            raise AuthError("internal_error", "Failed to remove credential")

    def list_credentials(self) -> List[Dict[str, Any]]:
        """List user's WebAuthn credentials."""
        user_id = get_current_user_id()
        user = self.db.get_user_by_id(user_id)
        if not user:
            raise AuthError("user_not_found", f"Authenticated user {user_id} not found in database")

        user_creds = user.webauthn_credentials or {}

        credentials = []
        for cred_id, cred_data in user_creds.items():
            credentials.append({
                "id": cred_id,
                "name": cred_data.get("name", "Biometric Device"),
                "created_at": cred_data.get("created_at"),
                "last_used_at": cred_data.get("last_used_at"),
                "device_type": cred_data.get("credential_device_type", "platform"),
                "backed_up": cred_data.get("credential_backed_up", False)
            })

        return credentials
