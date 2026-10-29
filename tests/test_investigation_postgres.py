from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
import psycopg
import pytest

from world_model.agents.contracts import InvestigationConfig
from world_model.api import create_app
from world_model.core import ObservationPayload
from world_model.storage.contracts import ConflictError
from world_model.storage.store import PostgresStore

pytestmark = pytest.mark.postgres


@pytest.fixture
def investigating(database, artifact):
    dsn, schema = database
    now = datetime(2026, 1, 2, tzinfo=UTC)
    store = PostgresStore(dsn, schema=schema, clock=lambda: now)
    store.register_model(artifact)
    store.create_run(artifact.content_hash, epoch=now - timedelta(days=1), target_state="degraded", run_id="agent")
    config = InvestigationConfig(allowed_sources=("cheap", "precise"), horizon=3, budget_credits=3, max_reads=3)
    store.configure_investigation("agent", config, seed=0)
    return store, config


def test_atomic_steps_budgets_stop_and_original_retry(investigating):
    store, config = investigating
    first = store.step_investigation("agent", "first")
    assert first.created
    assert first.step.decision.kind == "read"
    assert first.step.spent_credits == 1
    assert first.step.outcome.observation_id == store.inspect("agent").observations[0].observation_id
    assert first.step.action.dispatched_at < first.step.outcome.completed_at
    assert store.investigation("agent", sequence_cutoff=0) is None
    snapshot = store.inspect("agent", sequence_cutoff=1)
    assert len(snapshot.investigation.steps) == 1
    for index in range(1, 7):
        store.step_investigation("agent", f"step-{index}")
    view = store.investigation("agent")
    assert view.stopped
    assert view.spent_credits == view.dispatched_reads == 3
    assert [s.decision.kind for s in view.steps] == ["read", "advance_tick"] * 3 + ["stop"]
    report = store.replay("agent")
    assert report.events_checked == 7
    assert report.investigation_steps_checked == 7
    assert store.step_investigation("agent", "first").step == first.step
    assert not store.step_investigation("agent", "after-stop").created
    assert len(store.investigation("agent").steps) == 7
    assert store.inspect("agent", sequence_cutoff=1) == snapshot
    assert not store.investigation("agent", sequence_cutoff=6).stopped
    assert store.investigation("agent", sequence_cutoff=1).steps == (first.step,)
    assert store.investigation("agent", as_of=first.step.recorded_at).steps == (first.step,)


def test_concurrent_retries_cannot_duplicate_dispatch_or_cost(investigating):
    store, _ = investigating
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: store.step_investigation("agent", "same-request"), range(8)))
    assert sum(r.created for r in results) == 1
    assert len({r.step.outcome.observation_id for r in results}) == 1
    assert store.investigation("agent").spent_credits == 1
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda n: store.step_investigation("agent", f"different-{n}"), range(12)))
    view = store.investigation("agent")
    assert view.spent_credits == 3
    assert view.stopped
    assert len(view.steps) == 7


def test_configuration_is_immutable_and_manual_ingestion_is_blocked(investigating):
    store, config = investigating
    assert store.configure_investigation("agent", config, seed=0) == store.investigation("agent")
    with pytest.raises(ConflictError):
        store.configure_investigation("agent", config.model_copy(update={"budget_credits": 2}))
    with pytest.raises(ConflictError):
        store.ingest("agent", ObservationPayload(measurement_id="manual", source_id="cheap", tick=0, value=True))
    with pytest.raises(ConflictError):
        store.advance_tick("agent", 1)
    assert store.belief("agent").belief.sequence_cutoff == 0
    for table in ("investigations", "investigation_steps"):
        with pytest.raises(psycopg.errors.CheckViolation, match="append-only"):
            with store._connect() as conn:
                conn.execute(f"DELETE FROM {table}")


def test_failed_read_is_charged_once_and_slot_is_not_retried(investigating, monkeypatch):
    store, _ = investigating
    class FailedSensor:
        def read(self, **kwargs):
            raise OSError("private adapter details")
    monkeypatch.setattr(store, "_sensor_adapter", lambda *args: FailedSensor())
    result = store.step_investigation("agent", "failed-read")
    assert result.step.outcome.status == "failed"
    assert result.step.outcome.error == "sensor read failed"
    assert result.step.spent_credits == 1
    assert result.step.before_sequence == result.step.after_sequence == 0
    assert store.step_investigation("agent", "failed-read").step == result.step
    assert store.step_investigation("agent", "next").step.decision.kind == "advance_tick"
    assert store.inspect("agent").observations == ()
    assert store.replay("agent").investigation_steps_checked == 2


def test_rollback_keeps_cost_and_evidence_atomic(investigating, monkeypatch):
    store, _ = investigating
    original = store._publish
    def failure(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected failure before action commit")
    monkeypatch.setattr(store, "_publish", failure)
    with pytest.raises(RuntimeError, match="injected"):
        store.step_investigation("agent", "retry")
    assert store.investigation("agent").spent_credits == 0
    assert store.inspect("agent").observations == ()
    assert store.replay("agent").events_checked == 1
    monkeypatch.setattr(store, "_publish", original)
    assert store.step_investigation("agent", "retry").step.spent_credits == 1


def test_investigation_api_hides_environment_seed_and_protects_history(investigating):
    store, _ = investigating
    with TestClient(create_app(store), base_url="http://127.0.0.1") as client:
        response = client.post("/runs/agent/investigation/steps", json={"request_id": "step"})
        assert response.status_code == 201
        assert client.post("/runs/agent/investigation/steps", json={"request_id": "step"}).status_code == 200
        inspection = client.get("/runs/agent/inspection").json()
        assert inspection["investigation"]["spent_credits"] == 1
        assert "adapter_seed" not in str(inspection)
        assert "states" not in inspection["investigation"]
        assert client.get("/runs/agent/investigation?sequence_cutoff=0").json() is None
        assert client.post("/runs/agent/investigation/steps", json={"request_id": "step", "value": True}).status_code == 422
