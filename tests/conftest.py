from pathlib import Path
import os
from uuid import uuid4

import pytest
from psycopg import sql

from world_model.core import ModelArtifact
from world_model.storage.database import connect, migrate

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "docs" / "fixtures"


@pytest.fixture
def artifact():
    return ModelArtifact.load(FIXTURES / "binary-world-v1.json")


@pytest.fixture
def database():
    dsn = os.environ.get("WORLD_MODEL_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("set WORLD_MODEL_TEST_DATABASE_URL or run scripts/check_postgres.py")
    schema = "wme_test_" + uuid4().hex
    migrate(dsn, schema)
    try:
        yield dsn, schema
    finally:
        # Only remove the namespace created by this test.
        with connect(dsn, schema) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
