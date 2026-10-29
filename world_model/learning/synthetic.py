"""Evaluator-owned calibration experiments; only labeled readings cross the boundary."""

from datetime import UTC, datetime, timedelta

from world_model.core import ModelArtifact
from world_model.core.serialization import canonical_json
from world_model.synthetic.generator import _uniform
from .calibration import CalibrationObservation

CALIBRATION_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def neutral_structure(truth: ModelArtifact):
    raw = truth.spec.model_dump(mode="json")
    raw["model_id"] += "-structure"
    for sensor in raw["observation_models"].values():
        sensor["alert_probability"] = {state: .5 for state in truth.spec.states}
    return ModelArtifact(canonical_json(raw))


def calibration_cutoff(artifact, count):
    return CALIBRATION_EPOCH + timedelta(seconds=len(artifact.spec.states) * count)


def generate_calibration(truth, *, seed, samples_per_state):
    if type(seed) is not int or seed < 0 or type(samples_per_state) is not int or samples_per_state < 0:
        raise ValueError("calibration seed and sample count must be nonnegative integers")
    records = []
    for index in range(samples_per_state):
        for state_index, state in enumerate(truth.spec.states):
            observed = CALIBRATION_EPOCH + timedelta(seconds=index * len(truth.spec.states) + state_index)
            for source, sensor in sorted(truth.spec.observation_models.items()):
                records.append(CalibrationObservation(measurement_id=f"cal-{seed}-{state}-{index}",
                    source_id=source, state=state,
                    value=_uniform(seed, f"labeled-calibration-v1:{source}:{state}", index) < sensor.alert_probability[state],
                    observed_at=observed, available_at=observed + timedelta(seconds=1),
                    source_ref=f"synthetic-calibration-v1/{seed}/{state}/{index}/{source}"))
    return tuple(records)
