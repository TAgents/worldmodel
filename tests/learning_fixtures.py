"""Small labeled-calibration archive for HTTP and DOM checks (scores are fixtures)."""

from hashlib import sha256
import json

from world_model.core import ModelArtifact
from world_model.core.serialization import canonical_json
from world_model.learning.calibration import fit_calibration
from world_model.learning.evaluation import parameter_metrics
from world_model.learning.synthetic import calibration_cutoff, generate_calibration, neutral_structure
from .conftest import FIXTURES, ROOT


def learning_fixture():
    truth = ModelArtifact.load(FIXTURES / "complementary-world-v1.json")
    structure = neutral_structure(truth)
    training = generate_calibration(truth, seed=30000, samples_per_state=8)
    validation = generate_calibration(truth, seed=33000, samples_per_state=8)
    revisions = [fit_calibration(structure, training, as_of=calibration_cutoff(structure, n)) for n in (0, 8)]
    protocol = json.loads((ROOT / "docs/reliability-protocol-v1.json").read_bytes())
    protocol.update(calibration_samples_per_state=8, calibration_curve_sizes=[0, 8])
    report = {"protocol": protocol["protocol"], "protocol_sha256": sha256(canonical_json(protocol)).hexdigest(),
        "structure_hash": structure.content_hash, "calibration_revision": revisions[-1].revision_id,
        "split": "development", "episode_count": 2, "origin_ticks_per_episode": 4,
        "states": list(structure.spec.states),
        "particle_count": 64, "acquisition_budget": 4, "inference_version": "beta-mixture-filter-v1",
        "parameter_recovery_gate_passed": True, "numerical_ess_gate_passed": True, "heldout_acquisition_gate_passed": None,
        "adaptive_minus_baseline": {"learned_fixed_probe_a": {"brier": {"mean_difference": -.1, "lower_95": -.15, "upper_95": -.05}}},
        "methods": {"learned_adaptive": {"mean_brier": .3, "mean_log_loss": .6, "credits_per_episode": [4, 4], "minimum_effective_particles": 48.},
                    "learned_fixed_probe_a": {"mean_brier": .4, "mean_log_loss": .8, "credits_per_episode": [4, 4], "minimum_effective_particles": 50.}},
        "learning_curve": [{"samples_per_sensor_state": n, "revision_id": r.revision_id, **parameter_metrics(truth, r, validation)} for n, r in zip((0, 8), revisions, strict=True)],
        "calibration_cost": {"sensor_readings": 72, "sensor_credits": 72, "verified_labels": 24, "label_credits": 24}}
    return {"name": "example-reliability", "report": report, "protocol": protocol,
            "revisions": [r.model_dump(mode="json") for r in revisions], "runtime": {"python": "fixture"}}, structure


def write_learning_fixture(root):
    data, structure = learning_fixture()
    folder = root / data["name"]
    folder.mkdir(parents=True)
    contents = {"report.json": canonical_json(data["report"]), "protocol.json": canonical_json(data["protocol"]),
                "structure.json": structure.content, "calibration-revisions.json": canonical_json(data["revisions"])}
    for name, content in contents.items():
        (folder / name).write_bytes(content)
    (folder / "manifest.json").write_bytes(canonical_json({"protocol": "sensor-reliability-v1", "runtime": data["runtime"],
        "files": {name: sha256(content).hexdigest() for name, content in contents.items()}}))
    return folder, data


def promoted_inspections():
    """Two saved publications using independent rational inference for a learned model."""
    from datetime import UTC, datetime, timedelta
    from world_model.core import BeliefRevision, EvidenceSnapshot, Forecast, ModelRegistration, Observation, Reading, RunSpec
    from world_model.learning.calibration import CalibrationRevision
    from world_model.learning.promotion import calibrated_artifact
    from world_model.storage.contracts import ForecastView, Inspection, ModelView, Publication, RevisionView
    from .oracle import enumerate_distribution
    data, structure = learning_fixture()
    revision = CalibrationRevision.model_validate(data["revisions"][-1])
    artifact, _ = calibrated_artifact(structure, revision)
    epoch = datetime(2026, 1, 2, tzinfo=UTC)
    run = RunSpec(run_id="learned-run", model_hash=artifact.content_hash, epoch=epoch, created_at=epoch)
    model = ModelView(spec=artifact.spec, registration=ModelRegistration(model_id=artifact.spec.model_id,
        model_version=artifact.spec.model_version, content_hash=artifact.content_hash, registered_at=epoch))
    observation = Observation(observation_id="learned-o1", run_id=run.run_id, event_sequence=1,
        observed_at=epoch, ingested_at=epoch + timedelta(seconds=1),
        payload={"measurement_id":"m0", "tick":0, "source_id":"probe_a", "value":True})
    snapshots, history, forecasts = [], [], []
    for sequence in (0, 1):
        readings = [Reading(tick=0, source_id="probe_a", value=True)] if sequence else []
        evidence = EvidenceSnapshot(snapshot_id=f"s{sequence}", run_id=run.run_id, sequence_cutoff=sequence,
            observations=[{"observation_id":observation.observation_id, "event_sequence":1}] if sequence else [])
        belief = BeliefRevision(revision_id=f"b{sequence}", run_id=run.run_id, variable=artifact.spec.variable,
            state_tick=0, published_at=epoch + timedelta(seconds=sequence), sequence_cutoff=sequence,
            distribution={k:float(v) for k,v in enumerate_distribution(artifact.content, readings, 0).items()},
            model_hash=artifact.content_hash, evidence_snapshot_id=evidence.snapshot_id,
            inference_version="exact-discrete-v1", previous_revision_id="b0" if sequence else None)
        forecast = Forecast(forecast_id=f"f{sequence}", run_id=run.run_id, origin_tick=0, target_tick=1,
            target_state="mode_a", probability=float(enumerate_distribution(artifact.content, readings, 1)["mode_a"]),
            issued_at=belief.published_at, belief_revision_id=belief.revision_id, model_hash=artifact.content_hash,
            evidence_snapshot_id=evidence.snapshot_id)
        publication = Publication(belief=belief, evidence=evidence, forecast=forecast)
        history.append(RevisionView(belief=belief, event_kind="observation_ingested" if sequence else "run_created"))
        forecasts.append(ForecastView(forecast=forecast))
        snapshots.append(Inspection(run=run, model=model, publication=publication,
            observations=[observation] if sequence else [], history=tuple(history), forecasts=tuple(forecasts)))
    return snapshots
