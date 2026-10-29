from datetime import UTC, datetime

from fastapi.testclient import TestClient
import pytest

from world_model.api import create_app
from world_model.storage.store import PostgresStore

pytestmark = pytest.mark.postgres


@pytest.fixture
def live_api(database, artifact):
    dsn, schema = database
    clock = datetime(2026, 1, 2, tzinfo=UTC)
    store = PostgresStore(dsn, schema=schema, clock=lambda: clock)
    store.register_model(artifact)
    with TestClient(create_app(store), base_url="http://127.0.0.1") as client:
        yield client, store


def start(client, artifact, run_id="api-run"):
    response = client.post("/runs", json={"run_id": run_id, "epoch": "2026-01-01T00:00:00Z", "model_hash": artifact.content_hash, "target_state": "degraded"})
    assert response.status_code == 201
    return response.json()


def test_http_pipeline_corrections_replay_and_historical_inspection(live_api, artifact):
    client, store = live_api
    start(client, artifact)
    payload = {"measurement_id": "m0", "tick": 0, "source_id": "cheap", "value": True}
    result = client.post("/runs/api-run/observations", json=payload)
    assert result.status_code == 201
    original = result.json()
    assert original["publication"]["belief"]["distribution"]["degraded"] == pytest.approx(4 / 13)
    before = client.get("/runs/api-run/inspection").json()
    assert len(before["observations"]) == 1
    assert client.post("/runs/api-run/ticks", json={"tick": 1}).status_code == 201
    correction = payload | {"measurement_id": "m1", "value": False, "supersedes_id": original["observation"]["observation_id"]}
    assert client.post("/runs/api-run/observations", json=correction).status_code == 201
    latest = client.get("/runs/api-run/inspection").json()
    assert latest["publication"]["belief"]["distribution"]["degraded"] == pytest.approx(.02 + .88 / 37)
    assert latest["observations"][0]["payload"]["value"] is False
    assert len(latest["history"]) == 4
    frozen = [f for f in latest["forecasts"] if f["closed_at"]]
    assert [f["forecast"]["forecast_id"] for f in frozen] == [original["publication"]["forecast"]["forecast_id"]]
    assert client.get("/runs/api-run/inspection?sequence_cutoff=1").json() == before
    assert client.get("/runs/api-run/inspection", params={"as_of": before["publication"]["belief"]["published_at"]}).json() == before
    retry = client.post("/runs/api-run/observations", json=payload)
    assert retry.status_code == 200
    assert retry.json()["publication"] == original["publication"]
    assert client.get("/runs/api-run/replay").json()["events_checked"] == 4
    assert client.get("/runs/api-run/observations/" + original["observation"]["observation_id"]).json() == original["observation"]
    assert store.replay("api-run").events_checked == 4


def test_inspection_catalog_run_isolation_and_missing_history(live_api, artifact):
    client, _ = live_api
    first = start(client, artifact, "first")
    second = start(client, artifact, "second")
    # The injected clock is fixed; equal timestamps are ordered by run ID.
    assert client.get("/runs?limit=1").json()[0]["run_id"] == "first"
    assert client.get("/runs?limit=1&offset=1").json()[0]["run_id"] == "second"
    assert client.get("/models").json()[0]["content_hash"] == artifact.content_hash
    assert client.get(f"/models/{artifact.content_hash}").json()["spec"] == artifact.spec.model_dump(mode="json")
    assert client.get("/runs/first/inspection").json()["publication"] == first["publication"]
    assert client.get("/runs/second/inspection").json()["publication"] == second["publication"]
    assert client.get("/runs/missing/inspection").status_code == 404
    assert client.get("/runs/first/inspection?as_of=2025-12-31T00:00:00Z").status_code == 404
    assert client.get("/runs/first/inspection?sequence_cutoff=0").json()["observations"] == []


def test_http_rejections_leave_no_partial_publication(live_api, artifact):
    client, _ = live_api
    start(client, artifact)
    before = client.get("/runs/api-run/inspection").json()
    assert client.post("/runs/api-run/ticks", json={"tick": 2}).status_code == 422
    assert client.post("/runs/api-run/observations", json={"measurement_id": "future", "tick": 1, "source_id": "cheap", "value": True}).status_code == 422
    assert client.get("/runs/api-run/inspection").json() == before
    body = {"measurement_id": "m0", "tick": 0, "source_id": "cheap", "value": True}
    assert client.post("/runs/api-run/observations", json=body).status_code == 201
    accepted = client.get("/runs/api-run/inspection").json()
    assert client.post("/runs/api-run/observations", json=body | {"value": False}).status_code == 409
    assert client.get("/runs/api-run/inspection").json() == accepted
