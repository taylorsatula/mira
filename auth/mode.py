"""
The authentication mode decision point.

`MIRA_AUTH_MODE` selects which identity model a process serves. It exists because
mira-OSS ships to single-user installs and to multi-user installs from one tree, and
the two need different bootstrap — see plan §6.3.4 and decision D3.

    single  (DEFAULT) One shared bearer key against one hardcoded user row. The 1.x
                      behaviour, retained verbatim: `ensure_single_user()` runs,
                      `/oss-auth/token` is mounted, and no email transport is needed.
    dev               Cookie sessions over plain HTTP on localhost, one user in the
                      database, the full multi-user stack otherwise live. Gated by
                      `auth/dev_mode.py:development_mode_enabled()` (`MIRA_DEV`) at
                      the request surface; this mode names the bootstrap shape.
    multi             Full multi-user: magic links, sessions, API tokens, RLS on
                      `users`. Requires an email transport.

`single` is the default for two reasons that are load-bearing, not cosmetic. Most
mira-OSS installs are single-user, and `single` is the only mode that needs no email
transport — so a fresh clone with no Vault `mira/services` expansion and no mailer
still starts. Making any other value the default would mean an install that cannot
boot until a third-party service is configured.

Parsing is strict, mirroring the feature-flag loader in `config/config_manager.py`
(`_load_system_feature_flag_overrides`, which rejects anything but the literal "0"
and "1" rather than coercing truthily). A typo in `MIRA_AUTH_MODE` is a
misconfiguration that changes the security model of the process — `MIRA_AUTH_MODE=Sinlge`
must not quietly become single-user, and `MIRA_AUTH_MODE=mulit` must not quietly
become single-user either. So the value must be exactly one of the three literals,
case-sensitive, or this raises. An unset variable takes the default; a variable set
to the empty string does not, and raises.

The mode is read from the environment on every call rather than captured at import,
so `auth.mode` has no import-time side effect and a process may set the variable
before `create_app()` without depending on import order.
"""

import os
from typing import Literal, get_args

AuthMode = Literal["single", "dev", "multi"]

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
            "single", "dev" or "multi".
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


def single_user_mode_enabled() -> bool:
    """Return whether this process serves the single-user identity model."""
    return auth_mode() == "single"
