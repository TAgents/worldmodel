from datetime import UTC, datetime
from unittest.mock import Mock

from fastapi.testclient import TestClient
import psycopg
import pytest

from world_model.api import create_app
from world_model.core import ModelRegistration, Observation
from world_model.storage.contracts import (
    ConflictError, ForecastView, Inspection, ModelView, NotFoundError, ProjectionError,
    ReplayIntegrityError, ReplayReport, RevisionView, WriteResult,
)
from world_model.storage.store import PostgresStore

from .test_replay import history_fixture
from .investigation_fixtures import investigation_snapshots
from world_model.agents.contracts import StepReceipt


def inspection_fixture(artifact, cutoff=6):
    """Independent exact-oracle history; browser/HTTP fixtures, not a replacement store."""
    run, frames = history_fixture(artifact)
    publication = frames[cutoff][1]
    by_id = {event["payload"]["observation_id"]: Observation.model_validate(event["payload"]) for event, _ in frames if event["kind"] == "observation_ingested"}
    closures = {event["payload"]["frozen_forecast_id"]: event for event, _ in frames[:cutoff + 1] if event["kind"] == "tick_advanced"}
    return Inspection(
        run=run,
        model=ModelView(spec=artifact.spec, registration=ModelRegistration(content_hash=artifact.content_hash, model_id=artifact.spec.model_id, model_version=artifact.spec.model_version, registered_at=run.created_at)),
        publication=publication,
        observations=[by_id[ref.observation_id] for ref in publication.evidence.observations],
        history=[RevisionView(belief=pub.belief, event_kind=event["kind"]) for event, pub in frames[:cutoff + 1]],
        forecasts=[ForecastView(forecast=pub.forecast, closed_at=closures[pub.forecast.forecast_id]["recorded_at"] if pub.forecast.forecast_id in closures else None, closed_event_sequence=closures[pub.forecast.forecast_id]["sequence"] if pub.forecast.forecast_id in closures else None) for _, pub in frames[:cutoff + 1]],
    )


@pytest.fixture
def http(artifact):
    data = inspection_fixture(artifact)
    store = Mock(spec=PostgresStore)
    store.inspect.return_value = data
    store.list_runs.return_value = (data.run,)
    store.list_models.return_value = (data.model.registration,)
    store.get_model.return_value = data.model
    store.belief.return_value = data.publication
    store.history.return_value = tuple(pub for _, pub in history_fixture(artifact)[1])
    store.forecasts.return_value = data.forecasts
    store.get_observation.return_value = data.observations[-1]
    store.replay.return_value = ReplayReport(run_id=data.run.run_id, model_hash=artifact.content_hash, events_checked=7, publication=data.publication, closed_forecast_ids=("f1", "f3"))
    with TestClient(create_app(store), base_url="http://127.0.0.1") as client:
        yield client, store, data


def test_inspection_serializes_exact_publication_and_evidence(http):
    client, store, data = http
    response = client.get("/runs/run-1/inspection?sequence_cutoff=5&as_of=2026-01-01T00:02:01Z")
    assert response.status_code == 200
    assert response.json() == data.model_dump(mode="json")
    assert response.headers["cache-control"] == "no-store"
    store.inspect.assert_called_once_with("run-1", as_of=datetime(2026, 1, 1, 0, 2, 1, tzinfo=UTC), sequence_cutoff=5)


@pytest.mark.parametrize("suffix, method", [("belief", "belief"), ("belief/history", "history"), ("forecasts", "forecasts"), ("replay", "replay")])
def test_temporal_reads_pass_both_cutoffs(http, suffix, method):
    client, store, _ = http
    response = client.get(f"/runs/run-1/{suffix}?sequence_cutoff=0&as_of=2026-01-01T00:00:00Z")
    assert response.status_code == 200
    getattr(store, method).assert_called_once_with("run-1", as_of=datetime(2026, 1, 1, tzinfo=UTC), sequence_cutoff=0)


def test_write_receipts_preserve_retry_status_and_original_publication(http):
    client, store, data = http
    observation = data.observations[-1]
    for created, status in ((True, 201), (False, 200)):
        receipt = WriteResult(created=created, publication=data.publication, observation=observation)
        store.ingest.return_value = receipt
        response = client.post("/runs/run-1/observations", json=observation.payload.model_dump(mode="json"))
        assert response.status_code == status
        assert response.json() == receipt.model_dump(mode="json")
        assert store.ingest.call_args.args == ("run-1", observation.payload)
    store.create_run.return_value = WriteResult(created=True, publication=data.publication)
    response = client.post("/runs", json={"run_id": "run-1", "epoch": data.run.epoch.isoformat(), "model_hash": data.run.model_hash, "target_state": "degraded"})
    assert response.status_code == 201
    assert store.create_run.call_args.kwargs["epoch"] == data.run.epoch
    store.advance_tick.return_value = WriteResult(created=False, publication=data.publication)
    assert client.post("/runs/run-1/ticks", json={"tick": 2}).status_code == 200
    store.advance_tick.assert_called_once_with("run-1", 2)


@pytest.mark.parametrize("path", ["/runs/run-1/inspection?sequence_cutoff=-1", "/runs/run-1/belief?sequence_cutoff=1.5", "/runs/run-1/belief?as_of=2026-01-01", "/runs?limit=101", "/runs?offset=-1", "/models/invalid-hash"])
def test_invalid_queries_are_rejected_before_storage(http, path):
    client, store, _ = http
    assert client.get(path).status_code == 422
    assert not store.mock_calls


@pytest.mark.parametrize("change", [{"value": "false"}, {"tick": True}, {"tick": -1}, {"source_id": "unknown source"}, {"ingested_at": "2026-01-01T00:00:00Z"}])
def test_untrusted_evidence_cannot_forge_types_or_engine_time(http, change):
    client, store, data = http
    body = data.observations[-1].payload.model_dump(mode="json") | change
    assert client.post("/runs/run-1/observations", json=body).status_code == 422
    store.ingest.assert_not_called()


@pytest.mark.parametrize("error, status, detail", [
    (NotFoundError("run not found"), 404, "run not found"),
    (ConflictError("already recorded"), 409, "already recorded"),
    (ValueError("future tick"), 422, "future tick"),
    (ProjectionError("rebuild needed"), 503, "rebuild needed"),
    (ReplayIntegrityError("internal details"), 500, "Stored history failed verification."),
    (psycopg.OperationalError("password=secret host=private"), 503, "Database unavailable. Check the local database service."),
    (psycopg.ProgrammingError("internal SQL"), 500, "Database operation failed."),
])
def test_errors_are_useful_without_leaking_database_configuration(http, error, status, detail):
    client, store, _ = http
    store.inspect.side_effect = error
    response = client.get("/runs/run-1/inspection")
    assert response.status_code == status
    assert response.json() == {"detail": detail}


def test_local_origin_and_json_boundary(http):
    client, store, data = http
    payload = data.observations[-1].payload.model_dump(mode="json")
    assert client.post("/runs/run-1/observations", json=payload, headers={"Origin": "https://unrelated.example"}).status_code == 403
    assert client.get("/runs", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get("/runs", headers={"Host": "unrelated.example"}).status_code == 400
    assert client.post("/runs/run-1/observations", content='{"value":true}', headers={"Content-Type": "text/plain"}).status_code == 415
    assert not store.mock_calls
    assert client.get("/runs", headers={"Origin": "http://127.0.0.1"}).status_code == 200


def test_catalog_static_assets_openapi_and_health(http):
    client, store, data = http
    assert client.get("/runs").json() == [data.run.model_dump(mode="json")]
    assert client.get("/models").json() == [data.model.registration.model_dump(mode="json")]
    assert client.get(f"/models/{data.run.model_hash}").json() == data.model.model_dump(mode="json")
    assert client.get("/runs/run-1/observations/corrected0").json() == data.observations[-1].model_dump(mode="json")
    assert client.get("/health").json() == {"status": "ok"}
    page = client.get("/")
    assert "World Model Explorer" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/assets/explorer.js").status_code == 200
    assert client.get("/assets/explorer.css").status_code == 200
    spec = client.get("/openapi.json").json()
    assert {"200", "201", "422"} <= spec["paths"]["/runs"]["post"]["responses"].keys()


def test_investigation_http_contract_and_idempotent_step(http, artifact):
    client, store, _ = http
    snapshots, _, _ = investigation_snapshots(artifact)
    view = snapshots[-1].investigation
    store.configure_investigation.return_value = snapshots[1].investigation
    response = client.post("/runs/agent-run/investigation", json=view.config.model_dump(mode="json"))
    assert response.status_code == 200
    assert response.json()["steps"] == []
    store.configure_investigation.assert_called_once_with("agent-run", view.config)
    for created, status in ((True, 201), (False, 200)):
        store.step_investigation.return_value = StepReceipt(created=created, step=view.steps[0])
        response = client.post("/runs/agent-run/investigation/steps", json={"request_id": "action-0"})
        assert response.status_code == status
        assert response.json()["step"] == view.steps[0].model_dump(mode="json")
    store.investigation.return_value = view
    assert client.get("/runs/agent-run/investigation?sequence_cutoff=4").json() == view.model_dump(mode="json")
    store.investigation.assert_called_once_with("agent-run", as_of=None, sequence_cutoff=4)
    assert client.post("/runs/agent-run/investigation/steps", json={"request_id": "action-0", "source_id": "precise"}).status_code == 422
    assert client.post("/runs/agent-run/investigation", json=view.config.model_dump(mode="json") | {"adapter_seed": 1}).status_code == 422
