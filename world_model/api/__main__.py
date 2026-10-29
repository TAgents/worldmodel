"""Serve the API and explorer on loopback using an already configured database."""

import argparse
import os

import uvicorn

from world_model.storage.store import PostgresStore
from .app import create_app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--schema", default="world_model")
    parser.add_argument("--research-directory", default="artifacts")
    args = parser.parse_args(argv)
    dsn = os.environ.get("WORLD_MODEL_DATABASE_URL")
    if not dsn:
        parser.error("set WORLD_MODEL_DATABASE_URL; or use scripts/run_explorer.py for local setup")
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    uvicorn.run(create_app(PostgresStore(dsn, schema=args.schema), research_directory=args.research_directory), host="127.0.0.1", port=args.port, proxy_headers=False)


if __name__ == "__main__":
    main()
