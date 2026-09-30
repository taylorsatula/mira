"""
Configuration for the lean auth system.

Vault-backed values resolve lazily, on first attribute access. Importing this module
performs no network, filesystem or environment access, and `config = AuthConfig()`
reads nothing.

That property is load-bearing: `auth.session` and `auth.rate_limiter` import this
module for the constants below, which are not Vault-backed. Constructing the fields at
import would make every auth consumer require a reachable, fully seeded Vault —
including `MIRA_AUTH_MODE=single` installs that have no email transport and no reason
to contact one (see `auth/mode.py`). A missing or unreachable secret therefore raises
from `clients/vault_client.py` at the moment a caller asks for the value.

Failed lookups are not cached here. `get_service_config` caches successful reads, so a
warm Vault costs one dict lookup per access and a transiently broken Vault is retried
rather than having its failure pinned into this object for the life of the process.

`APP_URL` is seeded into `secret/mira/services` by both `deploy/postgresql.sh` and
`deploy/docker/scripts/init-mira.sh`, so a normal install resolves it with no extra
configuration. It is required when a `WebAuthnService` is constructed — i.e. when
passkeys are actually used — not at import and not at startup.
"""

from urllib.parse import urlparse

from clients.vault_client import get_service_config


class AuthConfig:
    """Auth configuration whose Vault-backed fields resolve on first use."""

    # Not Vault-backed: identical in every deployment, and read on the hot path of
    # session creation and rate limiting.
    MAGIC_LINK_EXPIRY: int = 600  # 10 minutes
    SESSION_IDLE_TIMEOUT: int = 86400 * 14  # 14 days idle timeout
    SESSION_MAX_LIFETIME: int = 86400 * 45  # 45 days max lifetime
    RATE_LIMIT_REQUESTS: int = 5
    RATE_LIMIT_WINDOW: int = 300  # 5 minutes

    @property
    def APP_URL(self) -> str:
        """Public origin of this deployment, in canonical origin form
        (scheme + host + port, no path, no trailing slash). Basis for the
        WebAuthn rp_id and expected_origin, and for the link a magic-link
        email tells the user to open. Normalized once here so every consumer
        sees the same value: a Vault `app_url` written with a trailing slash
        or a path must not diverge between passkey ceremonies and links."""
        raw = get_service_config('app_url')
        parsed = urlparse(raw)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(
                f"Invalid app_url {raw!r}: expected http(s)://host[:port]"
            )
        return f"{parsed.scheme}://{parsed.netloc}"


# Single instance, mirroring how auth/session.py and auth/rate_limiter.py reach
# configuration. Constructing AuthConfig reads nothing.
config = AuthConfig()
