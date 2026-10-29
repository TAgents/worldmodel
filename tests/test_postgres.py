"""Real-database acceptance tests. Every test owns a randomly named schema."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import json
import os
import subprocess
import sys

import psycopg
from psycopg import sql
import pytest

from world_model.core import ModelArtifact, ObservationPayload
from world_model.storage.contracts import ConflictError, NotFoundError, ProjectionError
from world_model.storage.database import connect, migrate
from world_model.storage.store import PostgresStore

from .conftest import FIXTURES, ROOT

pytestmark = pytest.mark.postgres


class Clock:
    def __init__(self):
        self.value = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self):
        return self.value

    def advance(self, seconds=60):
        self.value += timedelta(seconds=seconds)


@pytest.fixture
def stored(database, artifact):
    dsn, schema = database
    clock = Clock()
    store = PostgresStore(dsn, schema=schema, clock=clock)
    store.register_model(artifact)
    created = store.create_run(artifact.content_hash, epoch=clock(), target_state="degraded", run_id="run-1")
    return store, clock, created


def reading(measurement="m0", *, tick=0, source="cheap", value=True, supersedes=None):
    return ObservationPayload(measurement_id=measurement, tick=tick, source_id=source, value=value, supersedes_id=supersedes)


def counts(store):
    tables = ("events", "observations", "snapshots", "snapshot_members", "revisions", "forecasts", "forecast_closures", "run_heads")
    with store._connect() as conn:
        return {table: conn.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(sql.Identifier(table))).fetchone()["n"] for table in tables}


def test_migrations_are_idempotent_and_check_applied_content(database):
    dsn, schema = database
    assert migrate(dsn, schema) == ()
    with connect(dsn, schema) as conn:
        conn.execute("UPDATE schema_migrations SET checksum = 'modified' WHERE version = 1")
    with pytest.raises(ValueError, match="changed after application"):
        migrate(dsn, schema)


def test_ingestion_publishes_all_records_and_preserves_provenance(stored):
    store, clock, created = stored
    assert counts(store) == {"events": 1, "observations": 0, "snapshots": 1, "snapshot_members": 0, "revisions": 1, "forecasts": 1, "forecast_closures": 0, "run_heads": 1}
    result = store.ingest("run-1", reading())
    assert result.created
    assert result.publication.belief.distribution["degraded"] == pytest.approx(4 / 13, abs=1e-12)
    assert result.publication.forecast.probability == pytest.approx(189 / 650, abs=1e-12)
    assert store.get_observation("run-1", result.observation.observation_id) == result.observation
    assert result.publication.evidence.observations[0].observation_id == result.observation.observation_id
    assert result.publication.belief.previous_revision_id == created.publication.belief.revision_id
    assert store.replay("run-1").events_checked == 2


def test_retries_are_stable_and_do_not_add_events(stored):
    store, clock, _ = stored
    payload = reading()
    original = store.ingest("run-1", payload)
    store.ingest("run-1", reading("precise", source="precise"))
    before = counts(store)
    retry = store.ingest("run-1", payload)
    assert not retry.created
    assert retry.publication == original.publication
    assert retry.observation == original.observation
    assert counts(store) == before


def test_creation_and_model_registration_are_idempotent(stored, artifact):
    store, clock, created = stored
    registration = store.register_model(artifact)
    again = store.register_model(artifact)
    assert again == registration
    retry = store.create_run(artifact.content_hash, epoch=clock(), target_state="degraded", run_id="run-1")
    assert not retry.created and retry.publication == created.publication
    with pytest.raises(ConflictError):
        store.create_run(artifact.content_hash, epoch=clock(), target_state="healthy", run_id="run-1")
    with pytest.raises(ConflictError):
        store.register_model(ModelArtifact(json.dumps(json.loads(artifact.content)).encode()))


@pytest.mark.parametrize("payload", [reading(tick=1), reading(source="unknown")])
def test_invalid_evidence_writes_nothing(stored, payload):
    store, _, _ = stored
    before = counts(store)
    with pytest.raises(ValueError):
        store.ingest("run-1", payload)
    assert counts(store) == before


def test_conflicts_corrections_and_branch_rejection(stored):
    store, _, _ = stored
    original = store.ingest("run-1", reading())
    for payload in (reading(value=False), reading("another")):
        before = counts(store)
        with pytest.raises(ConflictError):
            store.ingest("run-1", payload)
        assert counts(store) == before
    correction = store.ingest("run-1", reading("corrected", value=False, supersedes=original.observation.observation_id))
    assert correction.publication.belief.distribution["degraded"] == pytest.approx(1 / 37, abs=1e-12)
    assert [ref.observation_id for ref in correction.publication.evidence.observations] == [correction.observation.observation_id]
    with pytest.raises(ConflictError):
        store.ingest("run-1", reading("fork", value=True, supersedes=original.observation.observation_id))
    assert store.replay("run-1").events_checked == 3


def test_run_isolation_applies_to_reads_and_corrections(stored, artifact):
    store, clock, _ = stored
    original = store.ingest("run-1", reading())
    store.create_run(artifact.content_hash, epoch=clock(), target_state="degraded", run_id="run-2")
    with pytest.raises(NotFoundError):
        store.get_observation("run-2", original.observation.observation_id)
    with pytest.raises(ConflictError):
        store.ingest("run-2", reading("foreign", supersedes=original.observation.observation_id))
    assert store.belief("run-2").belief.distribution["degraded"] == 0.1


def test_late_evidence_preserves_history_and_frozen_forecasts(stored):
    store, clock, _ = stored
    first = store.ingest("run-1", reading())
    first_bytes = first.publication.model_dump_json()
    clock.advance()
    advanced = store.advance_tick("run-1", 1)
    retry = store.advance_tick("run-1", 1)
    assert not retry.created and retry.publication == advanced.publication
    store.ingest("run-1", reading("m1", tick=1, value=False))
    clock.advance()
    store.advance_tick("run-1", 2)
    frozen = tuple(view.model_dump_json() for view in store.forecasts("run-1") if view.closed_at)
    clock.advance(1)
    late = store.ingest("run-1", reading("late", source="precise", value=False))
    assert late.publication.belief.distribution["degraded"] == pytest.approx(49391 / 1697350, abs=1e-12)
    assert store.belief("run-1", as_of=first.publication.belief.published_at).model_dump_json() == first_bytes
    assert store.belief("run-1", sequence_cutoff=1).model_dump_json() == first_bytes
    assert tuple(view.model_dump_json() for view in store.forecasts("run-1") if view.closed_at) == frozen
    assert all(view.closed_at is None for view in store.forecasts("run-1", sequence_cutoff=1))
    assert store.replay("run-1", sequence_cutoff=1).publication == first.publication
    assert store.replay("run-1").publication == late.publication


def test_backwards_clock_cannot_backdate_new_evidence(stored):
    store, clock, created = stored
    first = store.ingest("run-1", reading())
    clock.value -= timedelta(days=1)
    later = store.ingest("run-1", reading("precise", source="precise"))
    assert created.publication.belief.published_at < first.publication.belief.published_at < later.publication.belief.published_at
    assert store.belief("run-1", as_of=first.publication.belief.published_at) == first.publication


def test_tick_skips_and_environment_time_in_future_are_rejected(stored):
    store, _, created = stored
    for tick in (1, 2):
        with pytest.raises(ValueError):
            store.advance_tick("run-1", tick)
    assert store.belief("run-1") == created.publication


def test_before_creation_and_naive_time_cutoffs_do_not_leak_data(stored):
    store, clock, _ = stored
    with pytest.raises(NotFoundError):
        store.belief("run-1", as_of=clock() - timedelta(seconds=1))
    with pytest.raises(ValueError):
        store.belief("run-1", as_of=datetime(2026, 1, 1))


def test_concurrent_retries_create_one_publication(stored):
    store, _, _ = stored
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: store.ingest("run-1", reading()), range(6)))
    assert sum(result.created for result in results) == 1
    assert len({result.observation.observation_id for result in results}) == 1
    assert store.replay("run-1").events_checked == 2


def test_concurrent_sources_do_not_lose_an_update(stored):
    store, _, _ = stored
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda payload: store.ingest("run-1", payload), [reading(), reading("p0", source="precise")]))
    assert all(result.created for result in results)
    assert store.belief("run-1").belief.distribution["degraded"] == pytest.approx(76 / 85, abs=1e-12)
    assert store.replay("run-1").events_checked == 3


def test_concurrent_corrections_cannot_create_a_branch(stored):
    store, _, _ = stored
    parent = store.ingest("run-1", reading()).observation.observation_id
    def correct(name):
        try:
            store.ingest("run-1", reading(name, value=False, supersedes=parent))
            return True
        except ConflictError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(correct, ["c1", "c2"])) == [False, True]
    assert store.replay("run-1").events_checked == 3


def test_mid_update_exception_rolls_back_all_records(stored, monkeypatch):
    store, _, _ = stored
    before = counts(store)
    original_publish = store._publish
    def fail(conn, *args):
        assert conn.execute("SELECT count(*) AS n FROM observations").fetchone()["n"] == 1
        raise RuntimeError("simulated inference worker failure")
    monkeypatch.setattr(store, "_publish", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        store.ingest("run-1", reading())
    assert counts(store) == before
    monkeypatch.setattr(store, "_publish", original_publish)
    assert store.ingest("run-1", reading()).observation.event_sequence == 1


def test_terminated_transaction_does_not_publish_partial_evidence(stored, monkeypatch):
    store, _, _ = stored
    before = counts(store)
    def terminate(conn, *args):
        with store._connect() as killer:
            assert killer.execute("SELECT pg_terminate_backend(%s) AS terminated", (conn.info.backend_pid,)).fetchone()["terminated"]
        conn.execute("SELECT 1")
    monkeypatch.setattr(store, "_publish", terminate)
    with pytest.raises(psycopg.OperationalError):
        store.ingest("run-1", reading())
    assert counts(store) == before


@pytest.mark.parametrize("statement", [
    "UPDATE observations SET value = false",
    "DELETE FROM revisions",
    "TRUNCATE snapshot_members",
    "UPDATE models SET model_version = 'changed'",
])
def test_database_rejects_mutating_history(stored, statement):
    store, _, _ = stored
    store.ingest("run-1", reading())
    with pytest.raises(psycopg.errors.CheckViolation, match="append-only"):
        with store._connect() as conn:
            conn.execute(statement)
    assert store.replay("run-1").events_checked == 2


def test_projection_can_be_rebuilt_without_changing_history(stored):
    store, _, _ = stored
    result = store.ingest("run-1", reading())
    original = tuple(item.model_dump_json() for item in store.history("run-1"))
    with store._connect() as conn:
        conn.execute("DELETE FROM run_heads WHERE run_id = %s", ("run-1",))
    assert store.belief("run-1") == result.publication
    with pytest.raises(ProjectionError):
        store.ingest("run-1", reading("p0", source="precise"))
    assert store.rebuild_projection("run-1").publication == result.publication
    assert tuple(item.model_dump_json() for item in store.history("run-1")) == original
    assert not store.ingest("run-1", reading()).created


def test_new_process_connection_reconstructs_the_same_state(stored, database):
    store, _, _ = stored
    result = store.ingest("run-1", reading())
    dsn, schema = database
    restarted = PostgresStore(dsn, schema=schema)
    assert restarted.belief("run-1") == result.publication
    assert restarted.replay("run-1").publication == result.publication


def test_cli_demo_persists_and_replays_seven_events(database):
    dsn, schema = database
    env = os.environ.copy()
    env["WORLD_MODEL_DATABASE_URL"] = dsn
    process = subprocess.run([
        sys.executable, "-m", "world_model.storage", "--schema", schema, "demo",
        "--model", str(FIXTURES / "binary-world-v1.json"),
    ], env=env, cwd=ROOT, check=True, capture_output=True, text=True, timeout=30)
    result = json.loads(process.stdout)
    assert result["historical_publication_unchanged"]
    assert result["replay_events_verified"] == 7
    assert result["frozen_forecast_count"] == 2
    assert result["after_late_evidence"] == pytest.approx(49391 / 1697350, abs=1e-12)
