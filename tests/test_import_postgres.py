from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import json

from fastapi.testclient import TestClient
import psycopg
import pytest

from world_model.agents.contracts import InvestigationConfig
from world_model.api import create_app
from world_model.adapters.github_actions import prepare_import
from world_model.core import ObservationPayload, Reading
from world_model.learning.calibration import fit_calibration
from world_model.learning.promotion import calibrated_artifact
from world_model.learning.synthetic import neutral_structure
from world_model.storage.contracts import ConflictError, NotFoundError, ReplayIntegrityError
from world_model.storage.store import PostgresStore
from .oracle import enumerate_distribution
from .test_github_actions import actions

pytestmark = pytest.mark.postgres
NOW = datetime(2026, 2, 1, tzinfo=UTC)


@pytest.fixture
def store(database):
    return PostgresStore(database[0], schema=database[1], clock=lambda: NOW)


def test_import_infers_from_source_preserves_time_and_replays(store, actions):
    model, content, config = actions
    result = store.import_github_actions(*actions, run_id="imported", target_state="unstable")
    assert result.created
    assert result.publication.belief.sequence_cutoff == 15
    assert result.publication.belief.state_tick == 6
    inspection = store.inspect("imported")
    assert len(inspection.imported.observations) == 9
    assert inspection.imported.imported_at == NOW
    assert store.get_import_source("imported") == content
    for publication in store.history("imported"):
        historical = store.inspect("imported", sequence_cutoff=publication.belief.sequence_cutoff)
        readings = [Reading(tick=o.payload.tick, source_id=o.payload.source_id, value=o.payload.value) for o in historical.observations]
        expected = enumerate_distribution(model.content, readings, publication.belief.state_tick)
        assert dict(publication.belief.distribution) == pytest.approx({state:float(value) for state, value in expected.items()}, abs=1e-12)
        assert all(o.event_sequence <= publication.belief.sequence_cutoff for o in historical.imported.observations)
        assert all(o.ingested_at >= NOW for o in historical.observations)
        assert all(o.observed_at == historical.run.epoch + timedelta(days=o.payload.tick) for o in historical.observations)
        assert store.inspect("imported", as_of=publication.belief.published_at) == historical
    with pytest.raises(NotFoundError):
        store.inspect("imported", as_of=NOW - timedelta(seconds=1))
    restarted = PostgresStore(store._dsn, schema=store.schema)
    report = restarted.replay("imported")
    assert report.events_checked == 16
    assert report.imported_observations_checked == 9
    assert report.import_plan_hash_checked == prepare_import(*actions).content_hash
    assert len(report.closed_forecast_ids) == 6
    assert restarted.replay("imported", sequence_cutoff=1).imported_observations_checked == 1


def test_concurrent_retries_return_original_receipt_and_conflicts_cannot_adopt_run(store, actions):
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda _: store.import_github_actions(*actions, run_id="imported", target_state="unstable"), range(3)))
    assert sum(r.created for r in results) == 1
    assert all(r.publication == results[0].publication for r in results)
    with pytest.raises(ConflictError):
        store.import_github_actions(*actions, run_id="imported", target_state="stable")
    model, content, config = actions
    with pytest.raises(ConflictError):
        store.import_github_actions(model, content + b" ", config, run_id="imported", target_state="unstable")
    plan = prepare_import(*actions)
    original = store.create_run(model.content_hash, epoch=plan.epoch, target_state="unstable", run_id="manual")
    with pytest.raises(ConflictError, match="new run"):
        store.import_github_actions(*actions, run_id="manual", target_state="unstable")
    assert store.belief("manual") == original.publication


def test_import_owns_evidence_and_clock_even_for_an_empty_window(store, actions):
    model, _, config = actions
    empty = b'{"total_count":0,"workflow_runs":[]}'
    store.import_github_actions(model, empty, config.model_copy(update={"windows":1}), run_id="empty", target_state="unstable")
    assert store.belief("empty").belief.sequence_cutoff == 0
    assert store.inspect("empty").imported.observations == ()
    assert store.replay("empty").import_plan_hash_checked is not None
    with pytest.raises(ConflictError, match="snapshot"):
        store.ingest("empty", ObservationPayload(source_id="unit_tests", tick=0, value=True, measurement_id="manual"))
    with pytest.raises(ConflictError, match="snapshot"):
        store.advance_tick("empty", 1)
    with pytest.raises(ConflictError, match="snapshot"):
        store.configure_investigation("empty", InvestigationConfig(allowed_sources=("unit_tests",)))


def test_import_rolls_back_everything_on_failure_or_future_window(store, actions, monkeypatch):
    model, content, config = actions
    future = config.model_copy(update={"window_start":NOW})
    with pytest.raises(ValueError, match="closed"):
        store.import_github_actions(model, content, future, run_id="future", target_state="unstable")
    original = store._publish
    def fail(conn, run, artifact, tick, sequence, timestamp, previous_id):
        if sequence == 2:
            raise RuntimeError("injected publication failure")
        return original(conn, run, artifact, tick, sequence, timestamp, previous_id)
    monkeypatch.setattr(store, "_publish", fail)
    with pytest.raises(RuntimeError, match="injected"):
        store.import_github_actions(*actions, run_id="failed", target_state="unstable")
    assert store.list_models() == store.list_runs() == ()
    with store._connect() as conn:
        assert conn.execute("SELECT count(*) AS n FROM run_imports").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM observations").fetchone()["n"] == 0


def test_import_supports_an_already_registered_calibrated_model(store, actions):
    model, content, config = actions
    structure = neutral_structure(model)
    revision = fit_calibration(structure, [], as_of=config.window_start)
    calibrated, _ = calibrated_artifact(structure, revision)
    store.register_calibrated_model(structure, revision)
    store.import_github_actions(calibrated, content, config, run_id="calibrated-import", target_state="unstable")
    report = store.replay("calibrated-import")
    assert report.calibration_revision_checked == revision.revision_id
    assert report.imported_observations_checked == 9


@pytest.mark.parametrize("statement", ["DELETE FROM run_imports", "UPDATE run_imports SET source_content = ''::bytea", "TRUNCATE run_imports"])
def test_source_archive_is_append_only(store, actions, statement):
    store.import_github_actions(*actions, run_id="imported", target_state="unstable")
    with pytest.raises(psycopg.errors.CheckViolation):
        with store._connect() as conn:
            conn.execute(statement)


@pytest.mark.parametrize("damage", ["source", "plan", "missing"])
def test_replay_detects_corrupt_or_missing_import_source(store, actions, damage):
    store.import_github_actions(*actions, run_id="imported", target_state="unstable")
    with store._connect() as conn:
        conn.execute("ALTER TABLE run_imports DISABLE TRIGGER immutable_history")
        if damage == "source":
            conn.execute("UPDATE run_imports SET source_content = source_content || %s", (b" ",))
        elif damage == "plan":
            conn.execute("UPDATE run_imports SET plan_content = plan_content || %s", (b" ",))
        else:
            conn.execute("DELETE FROM run_imports")
        conn.execute("ALTER TABLE run_imports ENABLE TRIGGER immutable_history")
    with pytest.raises(ReplayIntegrityError):
        store.replay("imported")
    with pytest.raises(ReplayIntegrityError):
        store.inspect("imported")


def test_http_preview_and_import_preserve_exact_content_and_retry(store, actions):
    model, content, config = actions
    store.register_model(model)
    body = {"model_hash":model.content_hash, "export_json":content.decode(), "config":config.model_dump(mode="json")}
    with TestClient(create_app(store), base_url="http://127.0.0.1") as client:
        preview = client.post("/imports/github-actions/preview", json=body)
        assert preview.status_code == 200
        request = body | {"run_id":"http-import", "target_state":"unstable", "expected_plan_hash":preview.json()["plan_hash"]}
        stale = client.post("/imports/github-actions/runs", json=request | {"export_json":content.decode()+" "})
        assert stale.status_code == 409
        assert store.list_runs() == ()
        created = client.post("/imports/github-actions/runs", json=request)
        assert created.status_code == 201
        repeated = client.post("/imports/github-actions/runs", json=request)
        assert repeated.status_code == 200
        assert repeated.json()["publication"] == created.json()["publication"]
        assert client.get("/runs/http-import/import/source").content == content
        assert client.get("/runs/http-import/replay").json()["imported_observations_checked"] == 9
        early = client.get("/runs/http-import/inspection?sequence_cutoff=1").json()
        assert len(early["imported"]["observations"]) == 1
