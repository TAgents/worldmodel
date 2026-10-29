"""Connections and transactional, checksummed schema migrations."""

from hashlib import sha256
from importlib.resources import files
import re

import psycopg
from psycopg import sql
from psycopg.rows import dict_row


def connect(dsn: str, schema: str = "world_model"):
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", schema):
        raise ValueError("schema must be a lowercase PostgreSQL identifier")
    return psycopg.connect(
        dsn, row_factory=dict_row, connect_timeout=5,
        options=f"-csearch_path={schema},pg_catalog -ctimezone=UTC",
        application_name="world_model",
    )


def migrate(dsn: str, schema: str = "world_model") -> tuple[int, ...]:
    applied = []
    with connect(dsn, schema) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"world_model:migrations:{schema}",))
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
            version integer PRIMARY KEY, checksum text NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
        )""")
        existing = {row["version"]: row["checksum"] for row in conn.execute("SELECT version, checksum FROM schema_migrations")}
        migrations = sorted(files("world_model.storage").joinpath("migrations").iterdir(), key=lambda path: path.name)
        versions = set()
        for path in migrations:
            if not path.name.endswith(".sql"):
                continue
            version = int(path.name.split("_", 1)[0])
            versions.add(version)
            content = path.read_bytes()
            checksum = sha256(content).hexdigest()
            if version in existing:
                if existing[version] != checksum:
                    raise ValueError(f"migration {version} has changed after application")
                continue
            conn.execute(content.decode("utf-8"), prepare=False)
            conn.execute("INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)", (version, checksum))
            applied.append(version)
        if set(existing) - versions:
            raise ValueError("database has migrations newer than this application")
    return tuple(applied)
