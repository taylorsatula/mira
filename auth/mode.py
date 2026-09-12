"""
The authentication mode decision point.

`MIRA_AUTH_MODE` selects which identity model a process serves. It exists because
mira-OSS ships to single-user installs and to multi-user installs from one tree,
and the two need different bootstrap.

    single  (DEFAULT) One local account auto-provisioned by the local-session
                      endpoint (`GET /v0/auth/local/session`) on first visit.
                      Session-stack auth (Valkey sessions, CSRF) with no public
                      account creation; no email transport needed.
    multi             Full multi-user: signup, magic links, sessions, API tokens,
                      WebAuthn, RLS on `users`. Requires an email transport at
                      boot (verified by the power-on self-test).

`single` is the default for two reasons that are load-bearing, not cosmetic. Most
mira-OSS installs are single-user, and `single` is the only mode that needs no
email transport — so a fresh clone with no Vault `mira/services` expansion and no
mailer still starts. Making any other value the default would mean an install that
cannot boot until a third-party service is configured.

Parsing is strict, mirroring the feature-flag loader in `config/config_manager.py`
(`_load_system_feature_flag_overrides`, which rejects anything but the literal "0"
and "1" rather than coercing truthily). A typo in `MIRA_AUTH_MODE` is a
misconfiguration that changes the security model of the process — `MIRA_AUTH_MODE=Sinlge`
must not quietly become single-user, and `MIRA_AUTH_MODE=mulit` must not quietly
become single-user either. So the value must be exactly one of the two literals,
case-sensitive, or this raises. An unset variable takes the default; a variable set
to the empty string does not, and raises.

The mode is read from the environment on every call rather than captured at import,
so `auth.mode` has no import-time side effect and a process may set the variable
before `create_app()` without depending on import order.
"""

import os
from typing import Literal, get_args

AuthMode = Literal["single", "multi"]

#: Environment variable naming the mode.
AUTH_MODE_ENVIRONMENT_FIELD = "MIRA_AUTH_MODE"

#: The mode assumed when the variable is unset.
DEFAULT_AUTH_MODE: AuthMode = "single"

#: The complete set of accepted values, derived from the annotation so the type and
#: the parser cannot drift apart.
AUTH_MODES: tuple[str, ...] = get_args(AuthMode)


def auth_mode() -> AuthMode:
    """Return the configured authentication mode.

    Raises:
        ValueError: If `MIRA_AUTH_MODE` is set to anything other than exactly
            "single" or "multi".
    """
    raw_value = os.getenv(AUTH_MODE_ENVIRONMENT_FIELD)
    if raw_value is None:
        return DEFAULT_AUTH_MODE
    if raw_value not in AUTH_MODES:
        raise ValueError(
            f"{AUTH_MODE_ENVIRONMENT_FIELD} must be exactly one of "
            f"{', '.join(AUTH_MODES)}: {raw_value!r}"
        )
    return raw_value  # type: ignore[return-value]
