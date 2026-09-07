"""
Email transport seam and the stdlib SMTP backend (decision E-8).

`MailSender` is the seam between an auth flow that must put an email in
someone's inbox and whatever mechanism actually delivers it. One backend
ships: `SmtpMailSender`, built on the standard library's `smtplib` and
`email.message`, adding no dependency and no vendor. It talks to any relay
an operator already has — Postfix, an SES SMTP endpoint, Mailgun, a Gmail
app password.

Conventions mirror `auth/provisioning.py` deliberately, and only
deliberately: primitives in, `bool` out, `None` for "did it", raise for
"could not do it". A send returns `None` on success and raises on failure;
there is no `ensure` analogue for email — a mail either went or it did not,
and retrying a delivery silently would be worse than reporting the failure —
so none is invented. Like `AccountProvisioner`, the protocol is **not**
`@runtime_checkable`: a mis-shaped implementation fails at the call, and the
seam exists as the minimal seam, not a validation framework.

How an HTTP backend gets added
------------------------------
Implement `MailSender` structurally (any class with a matching `send`) and
inject it: `AuthService(mailer=ResendMailSender(api_key))`, or compose the
service yourself at your composition root. Nothing in this module needs to
change, and no vendor implementation ships here — mira-OSS must not acquire
a default dependency on a third-party email service (plan §0.1).

Configuration
-------------
Deliberately owned by this module, not by `auth/config.py`:
`auth/config.py` exposes only `APP_URL` and plain constants, and the private
deployment's email-gateway Vault key names are scrubbed from this
codebase (plan §11) — they map a security topology that is not shipped.
Each setting resolves **environment first, then Vault**
(`secret/mira/services`), matching this deployment's Vault-only-credentials
posture while keeping a container or an interactive dev shell usable with
no Vault write:

    MIRA_SMTP_HOST       / smtp_host        Relay hostname. Unset everywhere
                                            => no mailer is configured.
    MIRA_SMTP_PORT       / smtp_port        Default 587 (submission+STARTTLS).
    MIRA_SMTP_USER       / smtp_user        Optional; relay without auth.
    MIRA_SMTP_PASSWORD   / smtp_password    Optional; never logged.
    MIRA_SMTP_FROM       / smtp_from        Required once a host is set.
    MIRA_SMTP_STARTTLS   / smtp_starttls    Default enabled; strictly parsed
                                            ("1"/"true"/"yes" or
                                            "0"/"false"/"no", anything else
                                            raises — a typo in a transport
                                            security toggle must not
                                            silently downgrade it).

No mode reads any of this at import or at startup. `single` and `dev` boot
and serve with nothing configured (`get_mail_sender()` simply returns
`None`); the failure for a misconfigured `multi` install surfaces at the
point a send is attempted — or earlier if the operator wants it at boot,
which `utils/power_on_self_test.py` does in `multi` mode only.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

SMTP_TIMEOUT_SECONDS = 10


class MailSender(Protocol):
    """Everything an auth flow needs from an email transport.

    Structural typing only — callers annotate this and call `send`, and no
    `isinstance` check is possible against a non-runtime `Protocol`.
    """

    def send(self, recipient: str, subject: str, body: str) -> None:
        """Deliver one plain-text email. Raise on failure; return None on success."""
        ...


def _first_configured(env_name: str, vault_field: str) -> Optional[str]:
    """Resolve one setting: environment first, then Vault service config.

    An empty string counts as unset. A Vault read that raises is *not*
    swallowed — callers only reach this function when the environment did
    not answer, and "Vault is down" is a different failure from "not
    configured", with a different fix. A missing *field* in an existing
    `mira/services` secret (KeyError) is "not configured".
    """
    value = os.environ.get(env_name)
    if value:
        return value
    from clients.vault_client import get_service_config

    try:
        vault_value = get_service_config(vault_field)
    except KeyError:
        return None
    return vault_value or None


_TRUE_VALUES = {"1", "true", "yes"}
_FALSE_VALUES = {"0", "false", "no"}


def _parse_starttls(raw: Optional[str]) -> bool:
    """Strictly parse the STARTTLS toggle. Unset means enabled."""
    if raw is None:
        return True
    lowered = raw.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(
        "MIRA_SMTP_STARTTLS must be one of 1, true, yes, 0, false, no "
        f"(or unset); got {raw!r}"
    )


@dataclass(frozen=True)
class SmtpSettings:
    """Resolved configuration for one SMTP relay."""

    host: str
    port: int
    username: Optional[str]
    password: Optional[str]
    from_address: str
    starttls: bool

    @classmethod
    def from_environment(cls) -> Optional["SmtpSettings"]:
        """Resolve the relay configuration, or None when nothing is configured.

        "Configured" means a host is present in the environment or Vault. A
        present-but-invalid configuration raises rather than degrading to
        None: an operator who set MIRA_SMTP_PORT=abc wants that named, not a
        silent "no mailer configured" on the next magic link.
        """
        host = _first_configured("MIRA_SMTP_HOST", "smtp_host")
        if not host:
            return None

        raw_port = _first_configured("MIRA_SMTP_PORT", "smtp_port")
        try:
            port = int(raw_port) if raw_port else 587
        except ValueError as error:
            raise ValueError(f"MIRA_SMTP_PORT is not an integer: {raw_port!r}") from error
        if not 1 <= port <= 65535:
            raise ValueError(f"MIRA_SMTP_PORT out of range: {port}")

        from_address = _first_configured("MIRA_SMTP_FROM", "smtp_from")
        if not from_address:
            raise ValueError(
                "MIRA_SMTP_FROM (or Vault smtp_from) is required once an "
                "SMTP host is configured"
            )

        username = _first_configured("MIRA_SMTP_USER", "smtp_user")
        password = _first_configured("MIRA_SMTP_PASSWORD", "smtp_password")
        starttls = _parse_starttls(_first_configured("MIRA_SMTP_STARTTLS", "smtp_starttls"))

        return cls(
            host=host,
            port=port,
            username=username,
            password=password,
            from_address=from_address,
            starttls=starttls,
        )


def _recipient_digest(recipient: str) -> str:
    """Pseudonymise a recipient for log lines, same as the security logger does."""
    import hashlib

    return hashlib.sha256(recipient.encode()).hexdigest()[:16]


class SmtpMailSender:
    """`MailSender` over a plain SMTP relay using only the standard library."""

    def __init__(self, settings: SmtpSettings) -> None:
        self._settings = settings

    def send(self, recipient: str, subject: str, body: str) -> None:
        """Deliver one plain-text email through the configured relay.

        Raises:
            RuntimeError: On any connection, TLS, authentication or
                acceptance failure. The message text is generic — relay
                errors can carry the credential exchange — and the detail
                goes to the log under the recipient digest.
        """
        message = EmailMessage()
        message["From"] = self._settings.from_address
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)

        digest = _recipient_digest(recipient)
        try:
            with smtplib.SMTP(
                self._settings.host, self._settings.port, timeout=SMTP_TIMEOUT_SECONDS
            ) as client:
                if self._settings.starttls:
                    client.starttls(context=ssl.create_default_context())
                if self._settings.username:
                    client.login(self._settings.username, self._settings.password or "")
                client.send_message(message)
        except Exception as error:
            logger.error(
                "SMTP send to %s failed: %s: %s",
                digest,
                type(error).__name__,
                error,
                exc_info=True,
            )
            raise RuntimeError("Email sending failed") from error

        logger.info("Email sent to %s via %s:%d", digest, self._settings.host, self._settings.port)


def get_mail_sender() -> Optional[SmtpMailSender]:
    """Return the configured sender, or None when no relay is configured.

    Cheap enough to call per send: the environment lookup is a dict probe and
    the Vault read behind `_first_configured` caches successful results.
    """
    settings = SmtpSettings.from_environment()
    if settings is None:
        return None
    return SmtpMailSender(settings)
