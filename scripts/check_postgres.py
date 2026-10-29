"""Run PostgreSQL acceptance tests, failing explicitly if no server can start.

Use WORLD_MODEL_TEST_DATABASE_URL for an existing dedicated test database.
Otherwise start the checkout's private Unix-socket-only local cluster.
"""

import os
from pathlib import Path
import subprocess
import sys

import psycopg

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = os.environ.copy()
    if not env.get("WORLD_MODEL_TEST_DATABASE_URL"):
        subprocess.run([sys.executable, str(ROOT / "scripts/local_postgres.py"), "start"], cwd=ROOT, check=True)
        env["WORLD_MODEL_TEST_DATABASE_URL"] = subprocess.run(
            [sys.executable, str(ROOT / "scripts/local_postgres.py"), "dsn"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    with psycopg.connect(env["WORLD_MODEL_TEST_DATABASE_URL"], connect_timeout=5) as connection:
        connection.execute("SELECT 1")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=ROOT, env=env,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
