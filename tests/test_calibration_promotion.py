from datetime import timedelta
import json
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from world_model.api import create_app
from world_model.core import ModelArtifact
from world_model.core.serialization import canonical_json
from world_model.learning.calibration import CalibrationRevision, fit_calibration
from world_model.learning.catalog import ResearchCatalog
from world_model.learning.promotion import calibrated_artifact, verify_calibrated_artifact
from world_model.storage.contracts import NotFoundError, ReplayIntegrityError, WriteResult
from world_model.storage.store import PostgresStore
from .learning_fixtures import learning_fixture, promoted_inspections, write_learning_fixture
from .test_api import inspection_fixture


@pytest.fixture
def calibration():
    data, structure = learning_fixture()
    return structure, CalibrationRevision.model_validate(data["revisions"][-1])


def test_promoted_bytes_bind_means_and_complete_calibration(calibration):
    structure, revision = calibration
    artifact, content = calibrated_artifact(structure, revision)
    view = verify_calibrated_artifact(artifact, structure.content, content)
    assert view.revision == revision
    assert view.derivation.calibration_revision == revision.revision_id
    assert view.derivation.evidence_count == 72
    assert artifact.spec.initial_distribution == structure.spec.initial_distribution
    assert artifact.spec.transition_probabilities == structure.spec.transition_probabilities
    assert all(artifact.spec.observation_models[c.source_id].alert_probability[c.state] == c.mean for c in revision.cells)
    # A provenance change changes the promoted model even when the fitted means do not.
    changed = fit_calibration(structure, revision.evidence, as_of=revision.as_of + timedelta(seconds=1))
    other, _ = calibrated_artifact(structure, changed)
    assert other.content_hash != artifact.content_hash
    assert other.spec.observation_models == artifact.spec.observation_models
    with pytest.raises(ValueError, match="calibrated registration"):
        PostgresStore("unused").register_model(artifact)


def test_promotion_detects_evidence_structure_and_likelihood_tampering(calibration):
    structure, revision = calibration
    artifact, content = calibrated_artifact(structure, revision)
    with pytest.raises(ValueError, match="evidence bytes"):
        verify_calibrated_artifact(artifact, structure.content, content + b" ")
    other_structure = json.loads(structure.content)
    other_structure["tick_duration_seconds"] += 1
    with pytest.raises(ValueError):
        verify_calibrated_artifact(artifact, canonical_json(other_structure), content)
    raw = json.loads(artifact.content)
    raw["observation_models"]["probe_a"]["alert_probability"]["mode_a"] = .5
    with pytest.raises(ValueError, match="does not match"):
        verify_calibrated_artifact(ModelArtifact(canonical_json(raw)), structure.content, content)
    with pytest.raises(ValueError, match="underived"):
        calibrated_artifact(artifact, revision)


def test_archive_promotion_selects_exact_snapshot_and_validates_before_writes(tmp_path, artifact):
    folder, expected = write_learning_fixture(tmp_path)
    store = Mock(spec=PostgresStore)
    publication = inspection_fixture(artifact).publication
    body = {"run_id": "calibrated", "calibration_revision": expected["revisions"][0]["revision_id"],
            "epoch": "2026-01-01T00:00:00Z", "target_state": "mode_a"}
    with TestClient(create_app(store, research_directory=tmp_path), base_url="http://127.0.0.1") as client:
        for created, status in ((True, 201), (False, 200)):
            store.create_calibrated_run.return_value = WriteResult(created=created, publication=publication)
            response = client.post(f"/research/{folder.name}/runs", json=body)
            assert response.status_code == status
            assert response.json()["created"] is created
            assert store.create_calibrated_run.call_args.args[1].evidence == ()
        store.reset_mock()
        assert client.post(f"/research/{folder.name}/runs", json=body | {"calibration_revision": "0" * 64}).status_code == 404
        assert client.post(f"/research/{folder.name}/runs", json=body | {"registered_at": body["epoch"]}).status_code == 422
        assert client.post(f"/research/{folder.name}/runs", json=body, headers={"Origin": "https://elsewhere.example"}).status_code == 403
        (folder / "calibration-revisions.json").write_text("[]")
        assert client.post(f"/research/{folder.name}/runs", json=body).status_code == 422
        store.create_calibrated_run.assert_not_called()


def test_saved_calibration_http_read_is_independent_of_research_catalog(calibration, tmp_path):
    structure, revision = calibration
    artifact, content = calibrated_artifact(structure, revision)
    view = verify_calibrated_artifact(artifact, structure.content, content)
    store = Mock(spec=PostgresStore)
    store.get_calibration.return_value = view
    with TestClient(create_app(store, research_directory=tmp_path), base_url="http://127.0.0.1") as client:
        assert client.get(f"/models/{artifact.content_hash}/calibration").json() == view.model_dump(mode="json")
        store.get_calibration.assert_called_once_with(artifact.content_hash)
        store.get_calibration.side_effect = NotFoundError("model has no calibration evidence")
        assert client.get(f"/models/{artifact.content_hash}/calibration").status_code == 404


def test_promoted_publications_replay_against_independent_oracle(calibration):
    from world_model.storage.replay import ReplayState
    artifact, _ = calibrated_artifact(*calibration)
    snapshots = promoted_inspections()
    state = ReplayState(artifact, snapshots[0].run, "mode_a")
    for snapshot in snapshots:
        sequence = snapshot.publication.belief.sequence_cutoff
        state.apply({"run_id":snapshot.run.run_id, "sequence":sequence,
            "recorded_at":snapshot.publication.belief.published_at,
            "kind":"observation_ingested" if sequence else "run_created",
            "payload":snapshot.observations[0].model_dump(mode="json") if sequence else {"run":snapshot.run.model_dump(mode="json"), "target_state":"mode_a"}}, snapshot.publication)
    assert state.publication == snapshots[1].publication


def test_stored_calibration_checks_availability_and_missing_evidence(calibration):
    structure, revision = calibration
    artifact, content = calibrated_artifact(structure, revision)
    connection = Mock()
    connection.execute.return_value.fetchone.return_value = {"structure_content":structure.content, "revision_content":content}
    store = PostgresStore("unused")
    assert store._calibration(connection, artifact, revision.as_of).revision == revision
    with pytest.raises(ReplayIntegrityError, match="provenance"):
        store._calibration(connection, artifact, revision.as_of - timedelta(seconds=1))
    connection.execute.return_value.fetchone.return_value = None
    with pytest.raises(ReplayIntegrityError, match="missing"):
        store._calibration(connection, artifact, revision.as_of)


def test_interval_library_roundoff_does_not_change_persistent_replay(calibration, monkeypatch):
    from scipy.stats import beta
    structure, revision = calibration
    artifact, content = calibrated_artifact(structure, revision)
    original_ppf = beta.ppf
    monkeypatch.setattr(beta, "ppf", lambda *args, **kwargs: original_ppf(*args, **kwargs) + 1e-14)
    assert verify_calibrated_artifact(artifact, structure.content, content).revision == revision
    monkeypatch.setattr(beta, "ppf", lambda *args, **kwargs: original_ppf(*args, **kwargs) + 1e-5)
    with pytest.raises(ValueError, match="does not match"):
        verify_calibrated_artifact(artifact, structure.content, content)


def test_promote_cli_preserves_snapshot_and_retry_inputs(calibration, tmp_path, monkeypatch, capsys):
    from world_model.learning.__main__ import main
    structure, revision = calibration
    (tmp_path / "structure.json").write_bytes(structure.content)
    (tmp_path / "revision.json").write_text(revision.model_dump_json())
    store = Mock(spec=PostgresStore)
    store.create_calibrated_run.return_value = WriteResult(created=True, publication=promoted_inspections()[0].publication)
    monkeypatch.setattr("world_model.storage.store.PostgresStore", lambda *args, **kwargs: store)
    monkeypatch.setattr("sys.argv", ["world-model-learn", "promote", "--structure", str(tmp_path / "structure.json"),
        "--revision", str(tmp_path / "revision.json"), "--run-id", "learned-run", "--target-state", "mode_a",
        "--epoch", "2026-01-02T00:00:00Z", "--dsn", "test-only"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["publication"]["belief"]["run_id"] == "learned-run"
    assert store.create_calibrated_run.call_args.args == (structure, revision)
    assert store.create_calibrated_run.call_args.kwargs["run_id"] == "learned-run"
