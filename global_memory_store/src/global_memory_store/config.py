"""Configuration for the store: one TOML file, every field required.

No environment variables, no defaults. A missing or invalid field raises
ConfigError naming the field; the process refuses to start rather than run
with a guess.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    """The config file is missing, unreadable, or fails validation."""


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str  # "local" | "remote"
    model: str
    endpoint_url: str
    api_key_file: str
    dimensions: int


@dataclass(frozen=True)
class Config:
    bind_host: str
    bind_port: int
    database_url_file: str
    token_file: str
    embedding: EmbeddingConfig


def _required_str(table: dict, key: str, where: str, allow_empty: bool = False) -> str:
    if key not in table:
        raise ConfigError(f"{where}: missing required field {key!r}")
    value = table[key]
    if not isinstance(value, str):
        raise ConfigError(f"{where}: field {key!r} must be a string, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise ConfigError(f"{where}: field {key!r} must not be empty")
    return value


def _required_int(table: dict, key: str, where: str, low: int, high: int) -> int:
    if key not in table:
        raise ConfigError(f"{where}: missing required field {key!r}")
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: field {key!r} must be an integer, got {value!r}")
    if not low <= value <= high:
        raise ConfigError(f"{where}: field {key!r} must be between {low} and {high}, got {value}")
    return value


def load_config(path: str) -> Config:
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        raise ConfigError(f"cannot read config file {path}: {e}") from e
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise ConfigError(f"config file {path} is not valid TOML: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a TOML table")

    where = f"config file {path}"
    bind_host = _required_str(data, "bind_host", where)
    bind_port = _required_int(data, "bind_port", where, 1, 65535)
    database_url_file = _required_str(data, "database_url_file", where)
    token_file = _required_str(data, "token_file", where)

    embedding_table = data.get("embedding")
    if not isinstance(embedding_table, dict):
        raise ConfigError(f"{where}: missing required [embedding] table")
    ewhere = f"{where} [embedding]"
    provider = _required_str(embedding_table, "provider", ewhere)
    if provider not in ("local", "remote"):
        raise ConfigError(f"{ewhere}: field 'provider' must be 'local' or 'remote', got {provider!r}")
    model = _required_str(embedding_table, "model", ewhere)
    # Present for both providers; unused (may be empty) for local.
    endpoint_url = _required_str(embedding_table, "endpoint_url", ewhere, allow_empty=True)
    api_key_file = _required_str(embedding_table, "api_key_file", ewhere, allow_empty=True)
    dimensions = _required_int(embedding_table, "dimensions", ewhere, 1, 16000)
    if provider == "remote":
        if not endpoint_url.startswith(("http://", "https://")):
            raise ConfigError(f"{ewhere}: field 'endpoint_url' must be an http(s) URL, got {endpoint_url!r}")
        if not api_key_file.strip():
            raise ConfigError(f"{ewhere}: field 'api_key_file' must not be empty for the remote provider")

    return Config(
        bind_host=bind_host,
        bind_port=bind_port,
        database_url_file=database_url_file,
        token_file=token_file,
        embedding=EmbeddingConfig(
            provider=provider,
            model=model,
            endpoint_url=endpoint_url,
            api_key_file=api_key_file,
            dimensions=dimensions,
        ),
    )


def read_secret_file(path: str, what: str) -> str:
    """Read a secret from a file; the store keeps no secrets in its config."""
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError as e:
        raise ConfigError(f"cannot read {what} from {path}: {e}") from e
    if not value:
        raise ConfigError(f"{what} file {path} is empty")
    return value
