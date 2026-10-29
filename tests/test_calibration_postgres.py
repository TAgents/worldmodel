"""Durable promotion, atomic failure, concurrent retry, and independent replay."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from fractions import Fraction
import shutil

from fastapi.testclient import TestClient
import psycopg
import pytest

from world_model.api import create_app
from world_model.agents.contracts import InvestigationConfig
from world_model.core import ObservationPayload
from world_model.learning.calibration import fit_calibration
from world_model.learning.catalog import ResearchCatalog
from world_model.storage.contracts import ConflictError, ReplayIntegrityError
from world_model.storage.store import PostgresStore
from .learning_fixtures import write_learning_fixture
from .test_calibration_promotion import calibration

pytestmark = pytest.mark.postgres
NOW = datetime(2026, 1, 2, tzinfo=UTC)
SETTINGS = {"run_id": "calibrated", "target_state": "mode_a", "epoch": NOW - timedelta(hours=1)}


@pytest.fixture
def store(database):
    dsn, schema = database
    return PostgresStore(dsn, schema=schema, clock=lambda: NOW)


def test_promoted_http_run_updates_and_replays_after_archive_removal(store, tmp_path):
    folder, data = write_learning_fixture(tmp_path)
    structure, revision = ResearchCatalog(tmp_path).calibration(folder.name, data["revisions"][-1]["revision_id"])
    with TestClient(create_app(store, research_directory=tmp_path), base_url="http://127.0.0.1") as client:
        body = SETTINGS | {"epoch": SETTINGS["epoch"].isoformat(), "calibration_revision": revision.revision_id}
        response = client.post(f"/research/{folder.name}/runs", json=body)
        assert response.status_code == 201
        created = response.json()
        model_hash = created["publication"]["belief"]["model_hash"]
        assert created["publication"]["belief"]["distribution"] == dict(structure.spec.initial_distribution)
        assert created["publication"]["evidence"]["observations"] == []
        assert client.post(f"/research/{folder.name}/runs", json=body).status_code == 200
        shutil.rmtree(folder)  # only this test's temporary archive
        saved = client.get(f"/models/{model_hash}/calibration")
        assert saved.status_code == 200
        assert saved.json()["revision"] == revision.model_dump(mode="json")
        payload = {"measurement_id": "m0", "source_id": "probe_a", "tick": 0, "value": True}
        first = client.post("/runs/calibrated/observations", json=payload).json()
        weights = {c.state: Fraction(int(c.alpha), int(c.alpha + c.beta)) for c in revision.cells if c.source_id == "probe_a"}
        expected = {state: float(weight / sum(weights.values())) for state, weight in weights.items()}
        assert first["publication"]["belief"]["distribution"] == pytest.approx(expected)
        original = client.get("/runs/calibrated/inspection").json()
        assert client.post("/runs/calibrated/ticks", json={"tick": 1}).status_code == 201
        corrected = client.post("/runs/calibrated/observations", json=payload | {
            "measurement_id": "correction", "value": False, "supersedes_id": first["observation"]["observation_id"]})
        assert corrected.status_code == 201
        assert client.get("/runs/calibrated/inspection?sequence_cutoff=1").json() == original
        assert client.get("/runs/calibrated/inspection", params={"as_of": original["publication"]["belief"]["published_at"]}).json() == original
    restarted = PostgresStore(store._dsn, schema=store.schema)
    report = restarted.replay("calibrated")
    assert report.events_checked == 4
    assert report.calibration_revision_checked == revision.revision_id
    assert report.closed_forecast_ids == (first["publication"]["forecast"]["forecast_id"],)
    assert restarted.replay("calibrated", sequence_cutoff=1).publication.model_dump(mode="json") == first["publication"]


def test_concurrent_promotion_retries_return_original_publication(store, calibration):
    structure, revision = calibration
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: store.create_calibrated_run(structure, revision, **SETTINGS), range(4)))
    assert sum(result.created for result in results) == 1
    assert all(result.publication == results[0].publication for result in results)
    store.ingest("calibrated", ObservationPayload(measurement_id="m0", source_id="probe_a", tick=0, value=True))
    assert store.create_calibrated_run(structure, revision, **SETTINGS).publication == results[0].publication
    with store._connect() as conn:
        assert conn.execute("SELECT count(*) AS n FROM models").fetchone()["n"] == 1
        assert conn.execute("SELECT count(*) AS n FROM model_calibrations").fetchone()["n"] == 1
    revised = fit_calibration(structure, revision.evidence, as_of=revision.as_of + timedelta(seconds=1))
    with pytest.raises(ConflictError):
        store.create_calibrated_run(structure, revised, **SETTINGS)
    assert len(store.list_models()) == 1


def test_new_calibration_does_not_rewrite_prior_runs(store, calibration):
    structure, revision = calibration
    first = store.create_calibrated_run(structure, revision, **SETTINGS)
    assert store.register_calibrated_model(structure, revision).content_hash == first.publication.belief.model_hash
    historical = store.inspect("calibrated")
    prior = fit_calibration(structure, [], as_of=revision.as_of)
    second = store.create_calibrated_run(structure, prior, **(SETTINGS | {"run_id": "prior"}))
    assert first.publication.belief.model_hash != second.publication.belief.model_hash
    assert store.inspect("calibrated") == historical
    assert store.replay("calibrated").calibration_revision_checked == revision.revision_id
    assert store.replay("prior").calibration_revision_checked == prior.revision_id


def test_promoted_model_supports_budgeted_investigation_and_replay(store, calibration):
    structure, revision = calibration
    store.create_calibrated_run(structure, revision, **SETTINGS)
    store.configure_investigation("calibrated", InvestigationConfig(allowed_sources=tuple(structure.spec.observation_models),
        horizon=2, budget_credits=2, max_reads=2), seed=123)
    receipts = [store.step_investigation("calibrated", f"step-{i}") for i in range(5)]
    assert receipts[-1].step.decision.kind == "stop"
    assert receipts[-1].step.spent_credits == 2
    report = store.replay("calibrated")
    assert report.calibration_revision_checked == revision.revision_id
    assert report.investigation_steps_checked == 5


def test_invalid_or_failed_creation_rolls_back_model_and_evidence(store, calibration, monkeypatch):
    structure, revision = calibration
    with pytest.raises(ValueError, match="target"):
        store.create_calibrated_run(structure, revision, **(SETTINGS | {"target_state": "unknown"}))
    future = fit_calibration(structure, revision.evidence, as_of=NOW + timedelta(days=1))
    with pytest.raises(ValueError, match="already be available"):
        store.create_calibrated_run(structure, future, **SETTINGS)
    def fail(*args):
        raise RuntimeError("publication failed")
    monkeypatch.setattr(store, "_publish", fail)
    with pytest.raises(RuntimeError, match="publication failed"):
        store.create_calibrated_run(structure, revision, **SETTINGS)
    assert store.list_models() == store.list_runs() == ()
    with store._connect() as conn:
        assert conn.execute("SELECT count(*) AS n FROM model_calibrations").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM events").fetchone()["n"] == 0


@pytest.mark.parametrize("statement", ["DELETE FROM model_calibrations", "UPDATE model_calibrations SET revision_content = ''::bytea", "TRUNCATE model_calibrations"])
def test_calibration_history_is_append_only(store, calibration, statement):
    store.create_calibrated_run(*calibration, **SETTINGS)
    with pytest.raises(psycopg.errors.CheckViolation):
        with store._connect() as conn:
            conn.execute(statement)
    assert store.replay("calibrated").calibration_revision_checked == calibration[1].revision_id


@pytest.mark.parametrize("damage", ["missing", "revision", "structure"])
def test_replay_detects_missing_or_tampered_calibration(store, calibration, damage):
    store.create_calibrated_run(*calibration, **SETTINGS)
    # Simulate administrative corruption only in this test's isolated schema.
    with store._connect() as conn:
        conn.execute("ALTER TABLE model_calibrations DISABLE TRIGGER immutable_history")
        if damage == "missing":
            conn.execute("DELETE FROM model_calibrations")
        elif damage == "revision":
            conn.execute("UPDATE model_calibrations SET revision_content = revision_content || %s", (b" ",))
        else:
            conn.execute("UPDATE model_calibrations SET structure_content = %s", (b"{}",))
        conn.execute("ALTER TABLE model_calibrations ENABLE TRIGGER immutable_history")
    with pytest.raises(ReplayIntegrityError):
        store.replay("calibrated")
