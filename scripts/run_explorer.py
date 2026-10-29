"""Start the local database, register the example model, and serve the explorer."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from importlib.resources import files

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    env = os.environ.copy()
    if not env.get("WORLD_MODEL_DATABASE_URL"):
        subprocess.run([sys.executable, str(ROOT / "scripts/local_postgres.py"), "start"], cwd=ROOT, check=True)
        env["WORLD_MODEL_DATABASE_URL"] = subprocess.run(
            [sys.executable, str(ROOT / "scripts/local_postgres.py"), "dsn"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    from world_model.core import ModelArtifact
    from world_model.storage.database import migrate
    from world_model.storage.store import PostgresStore

    dsn = env["WORLD_MODEL_DATABASE_URL"]
    migrate(dsn)
    PostgresStore(dsn).register_model(ModelArtifact.load(ROOT / "docs/fixtures/binary-world-v1.json"))
    PostgresStore(dsn).register_model(ModelArtifact(files("world_model.adapters").joinpath("fixtures/software-delivery-v1.json").read_bytes()))
    print(f"World Model Explorer: http://127.0.0.1:{args.port}", flush=True)
    try:
        return subprocess.run([sys.executable, "-m", "world_model.api", "--port", str(args.port)], cwd=ROOT, env=env).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
