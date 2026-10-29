import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from world_model.core import (
    Action, BeliefRevision, EvidenceSnapshot, Forecast, ModelArtifact,
    ModelRegistration, ModelSpec, Observation, Outcome, Reading, RunSpec,
)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), "0.2", True])
def test_invalid_probabilities_are_rejected(artifact, value):
    raw = json.loads(artifact.content)
    raw["observation_models"]["cheap"]["alert_probability"]["healthy"] = value
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(raw)


@pytest.mark.parametrize("mutation", [
    lambda raw: raw["initial_distribution"].update(healthy=0.8),
    lambda raw: raw["transition_probabilities"]["healthy"].update(degraded=0.3),
    lambda raw: raw["transition_probabilities"].pop("degraded"),
    lambda raw: raw["initial_distribution"].update(unknown=0.0),
    lambda raw: raw["observation_models"]["cheap"]["alert_probability"].pop("healthy"),
    lambda raw: raw["states"].append("healthy"),
    lambda raw: raw["observation_models"].clear(),
    lambda raw: raw["observation_models"]["cheap"].update(read_cost_credits=0),
    lambda raw: raw.update(schema_version=True),
    lambda raw: raw.update(initial_tick=1),
    lambda raw: raw.update(forecast_horizon_ticks=2),
    lambda raw: raw.update(unrecognized="field"),
])
def test_inconsistent_models_fail_validation(artifact, mutation):
    raw = json.loads(artifact.content)
    mutation(raw)
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(raw)


def test_original_model_bytes_determine_identity(artifact):
    assert artifact.content_hash == hashlib.sha256(artifact.content).hexdigest()
    reformatted = ModelArtifact(json.dumps(json.loads(artifact.content), separators=(",", ":")).encode())
    assert reformatted.spec == artifact.spec
    assert reformatted.content_hash != artifact.content_hash


def test_duplicate_json_keys_cannot_silently_replace_model_values(artifact):
    content = artifact.content.replace(b'"schema_version": 1', b'"schema_version": 2, "schema_version": 1')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        ModelArtifact(content)


def test_nested_model_data_is_immutable_and_serializable(artifact):
    model = artifact.spec
    with pytest.raises(TypeError):
        model.transition_probabilities["healthy"]["degraded"] = 0.9
    with pytest.raises(TypeError):
        model.observation_models["cheap"].alert_probability["healthy"] = 0.9
    with pytest.raises(ValidationError):
        model.variable = "changed"
    assert ModelSpec.model_validate_json(model.model_dump_json()) == model
    assert json.loads(model.model_dump_json()) == json.loads(artifact.content)


def test_mutating_the_input_dictionary_cannot_change_a_validated_model(artifact):
    raw = json.loads(artifact.content)
    model = ModelSpec.model_validate(raw)
    raw["transition_probabilities"]["healthy"]["degraded"] = 1
    assert model.transition_probabilities["healthy"]["degraded"] == 0.02


@pytest.mark.parametrize("value", [0, 1, "false", "true", None])
def test_reading_values_must_be_json_booleans(value):
    with pytest.raises(ValidationError):
        Reading(tick=0, source_id="cheap", value=value)


@pytest.mark.parametrize("tick", [-1, True, "0", 0.5])
def test_reading_ticks_must_be_nonnegative_integers(tick):
    with pytest.raises(ValidationError):
        Reading(tick=tick, source_id="cheap", value=True)


def observation_payload():
    return {
        "observation_id": "obs-1", "run_id": "run-1",
        "payload": {"measurement_id": "measurement-1", "tick": 0, "source_id": "cheap", "value": True},
        "observed_at": "2026-10-27T12:00:00+03:00",
        "ingested_at": "2026-10-27T09:01:00Z",
        "event_sequence": 1,
    }


def test_observation_normalizes_time_to_utc_and_preserves_payload():
    record = Observation.model_validate(observation_payload())
    assert record.observed_at == datetime(2026, 10, 27, 9, tzinfo=UTC)
    assert record.observed_at.tzinfo == UTC
    assert record.payload.measurement_id == "measurement-1"
    assert Observation.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize("mutation", [
    lambda raw: raw.update(observed_at="2026-10-27T09:00:00"),
    lambda raw: raw.update(ingested_at="2026-10-27T08:00:00Z"),
    lambda raw: raw["payload"].update(supersedes_id="obs-1"),
    lambda raw: raw.update(event_sequence=0),
    lambda raw: raw["payload"].update(ground_truth=True),
])
def test_invalid_observation_envelopes_are_rejected(mutation):
    raw = observation_payload()
    mutation(raw)
    with pytest.raises(ValidationError):
        Observation.model_validate(raw)


@pytest.mark.parametrize("references", [
    [{"observation_id": "o1", "event_sequence": 3}],
    [{"observation_id": "o1", "event_sequence": 1}, {"observation_id": "o1", "event_sequence": 2}],
    [{"observation_id": "o1", "event_sequence": 2}, {"observation_id": "o2", "event_sequence": 1}],
    [{"observation_id": "o1", "event_sequence": 1}, {"observation_id": "o2", "event_sequence": 1}],
])
def test_snapshot_membership_cannot_duplicate_or_exceed_cutoff(references):
    with pytest.raises(ValidationError):
        EvidenceSnapshot(snapshot_id="s1", run_id="r1", sequence_cutoff=2, observations=references)


def test_contracts_support_json_schema_and_round_trips(artifact):
    timestamp = datetime(2026, 10, 27, tzinfo=UTC)
    records = [
        ModelRegistration(model_id="binary-world", model_version="1.0.0", content_hash=artifact.content_hash, registered_at=timestamp),
        RunSpec(run_id="r1", epoch=timestamp, created_at=timestamp, model_hash=artifact.content_hash),
        EvidenceSnapshot(snapshot_id="s1", run_id="r1", sequence_cutoff=0, observations=[]),
        BeliefRevision(revision_id="b1", run_id="r1", variable="state", state_tick=0, published_at=timestamp, sequence_cutoff=0, distribution={"healthy": 0.9, "degraded": 0.1}, model_hash=artifact.content_hash, evidence_snapshot_id="s1", inference_version="exact-discrete-v1"),
        Forecast(forecast_id="f1", run_id="r1", origin_tick=0, target_tick=1, target_state="degraded", probability=0.108, issued_at=timestamp, belief_revision_id="b1", model_hash=artifact.content_hash, evidence_snapshot_id="s1"),
        Action(action_id="a1", run_id="r1", tick=0, kind="read", source_id="cheap", reserved_cost_credits=1, created_at=timestamp),
        Outcome(outcome_id="o1", action_id="a1", run_id="r1", status="failed", charged_cost_credits=1, completed_at=timestamp, error="timeout"),
    ]
    for record in records:
        assert type(record).model_validate_json(record.model_dump_json()) == record
        assert type(record).model_json_schema()["type"] == "object"


@pytest.mark.parametrize("overrides", [
    {"kind": "read", "source_id": None},
    {"reserved_cost_credits": 0},
    {"kind": "advance_tick"},
    {"dispatched_at": "2026-10-26T00:00:00Z"},
])
def test_invalid_action_costs_and_dispatch_times(overrides):
    raw = {"action_id": "a1", "run_id": "r1", "tick": 0, "kind": "read", "source_id": "cheap", "reserved_cost_credits": 1, "created_at": "2026-10-27T00:00:00Z"}
    raw.update(overrides)
    with pytest.raises(ValidationError):
        Action.model_validate(raw)


@pytest.mark.parametrize("overrides", [
    {"status": "rejected", "charged_cost_credits": 1},
    {"observation_id": "o1"},
    {"error": None},
    {"error": " "},
    {"status": "succeeded"},
])
def test_invalid_outcomes_do_not_imply_observations_or_free_rejections(overrides):
    raw = {"outcome_id": "o1", "action_id": "a1", "run_id": "r1", "status": "failed", "charged_cost_credits": 1, "completed_at": "2026-10-27T00:00:00Z", "error": "timeout"}
    raw.update(overrides)
    with pytest.raises(ValidationError):
        Outcome.model_validate(raw)
