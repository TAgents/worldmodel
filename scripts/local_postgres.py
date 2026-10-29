"""Manage a private, Unix-socket-only PostgreSQL cluster for this checkout."""

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".local" / "postgres"
DATA = STATE / "data"
SOCKET = Path("/private/tmp" if Path("/private/tmp").exists() else "/tmp") / (
    f"wme-pg-{os.getuid()}-{hashlib.sha256(str(ROOT).encode()).hexdigest()[:8]}"
)
PORT = "55432"


def binary(name):
    candidates = []
    if os.environ.get("WORLD_MODEL_POSTGRES_BIN"):
        candidates.append(Path(os.environ["WORLD_MODEL_POSTGRES_BIN"]))
    for executable in ("postgres", "pg_ctl"):
        found = shutil.which(executable)
        if found:
            candidates.append(Path(found).resolve().parent)
    bundled = importlib.util.find_spec("pgserver")
    if bundled is not None:
        candidates.append(Path(bundled.origin).parent / "pginstall" / "bin")
    for directory in candidates:
        if (directory / "postgres").is_file() and (directory / name).is_file():
            return str(directory / name)
    raise RuntimeError("A full PostgreSQL installation is required; install the local-postgres extra or set WORLD_MODEL_POSTGRES_BIN")


def dsn():
    return f"host={SOCKET} port={PORT} dbname=world_model user=world_model"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "stop", "status", "dsn"))
    args = parser.parse_args()
    if args.command == "dsn":
        print(dsn())
        return 0
    ctl = binary("pg_ctl")
    if args.command in {"status", "stop"}:
        command = [ctl, "-D", str(DATA), args.command]
        if args.command == "stop":
            command += ["-m", "fast", "-w"]
        return subprocess.run(command, check=False).returncode
    STATE.mkdir(parents=True, exist_ok=True)
    SOCKET.mkdir(mode=0o700, exist_ok=True)
    if SOCKET.stat().st_uid != os.getuid():
        raise RuntimeError("the local socket directory is owned by another user")
    SOCKET.chmod(0o700)
    if not (DATA / "PG_VERSION").exists():
        subprocess.run([
            binary("initdb"), "-D", str(DATA), "-U", "world_model", "-A", "trust",
            "--encoding=UTF8", "--no-locale",
        ], check=True)
    running = subprocess.run([ctl, "-D", str(DATA), "status"], capture_output=True).returncode == 0
    if not running:
        subprocess.run([
            ctl, "-D", str(DATA), "-l", str(STATE / "server.log"), "-o",
            shlex.join(["-k", str(SOCKET), "-h", "", "-p", PORT]), "-w", "start",
        ], check=True)
    connection = ["-h", str(SOCKET), "-p", PORT, "-U", "world_model"]
    exists = subprocess.run([
        binary("psql"), *connection, "-d", "postgres", "-tAc",
        "SELECT 1 FROM pg_database WHERE datname = 'world_model'",
    ], check=True, capture_output=True, text=True).stdout.strip()
    if exists != "1":
        subprocess.run([binary("createdb"), *connection, "world_model"], check=True)
    print(dsn())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
