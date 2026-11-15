"""Entry point: `python -m global_memory_store init|serve --config <path>`."""

import argparse
import asyncio
import sys

from .api import create_app
from .config import ConfigError, load_config, read_secret_file
from .storage import Store, StoreError


def _load(args: argparse.Namespace):
    config = load_config(args.config)
    database_url = read_secret_file(config.database_url_file, "database URL")
    token = read_secret_file(config.token_file, "bearer token")
    api_key = (
        read_secret_file(config.embedding.api_key_file, "embeddings API key")
        if config.embedding.provider == "remote"
        else None
    )
    return config, database_url, token, api_key


def _cmd_init(args: argparse.Namespace) -> int:
    config, database_url, _token, _api_key = _load(args)
    store = Store(database_url, config.embedding.dimensions)
    try:
        store.init_schema(config.embedding.model)
    finally:
        store.close()
    print(f"initialized: model={config.embedding.model} dimensions={config.embedding.dimensions}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    config, database_url, token, api_key = _load(args)
    store = Store(database_url, config.embedding.dimensions)
    try:
        store.verify_ready(config.embedding.model)
    except StoreError:
        store.close()
        raise
    app = create_app(config, store, api_key, token)

    from hypercorn.asyncio import serve
    from hypercorn.config import Config as HypercornConfig

    hypercorn_config = HypercornConfig()
    hypercorn_config.bind = [f"{config.bind_host}:{config.bind_port}"]
    try:
        asyncio.run(serve(app, hypercorn_config))
    finally:
        store.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="global_memory_store")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("init", "create the tables and record the embedding identity"),
                            ("serve", "verify the schema and serve the HTTP API")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, help="path to the TOML config file")
    args = parser.parse_args()
    try:
        if args.command == "init":
            return _cmd_init(args)
        return _cmd_serve(args)
    except (ConfigError, StoreError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
