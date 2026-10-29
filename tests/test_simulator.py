import hashlib
import json

import pytest
from pydantic import ValidationError

from world_model.core import ModelArtifact
from world_model.synthetic.generator import Dataset, GENERATOR_VERSION, generate_dataset


def test_generator_is_frozen_by_version_and_reference_digest(artifact):
    data = generate_dataset(artifact, seeds=[0, 1], horizon=4)
    assert GENERATOR_VERSION == "synthetic-sha256-v1"
    assert hashlib.sha256(data.to_bytes()).hexdigest() == "75141df286ff024fcf8c65328c4059eff1730f8a9fda1e2a8adde45a674979b3"
    assert generate_dataset(artifact, seeds=[0, 1], horizon=4).to_bytes() == data.to_bytes()
    assert Dataset.model_validate_json(data.to_bytes()) == data


def test_sampling_order_and_extra_sensors_do_not_change_existing_streams(artifact):
    original = generate_dataset(artifact, seeds=[0, 1], horizon=12)
    reordered = generate_dataset(artifact, seeds=[1, 0], horizon=12)
    assert reordered.episodes[1] == original.episodes[0]
    raw = json.loads(artifact.content)
    raw["observation_models"]["additional"] = raw["observation_models"]["cheap"].copy()
    extended = generate_dataset(ModelArtifact(json.dumps(raw).encode()), seeds=[0, 1], horizon=12)
    for old, new in zip(original.episodes, extended.episodes, strict=True):
        assert old.states == new.states
        assert old.readings == tuple(reading for reading in new.readings if reading.source_id != "additional")


def test_deterministic_model_recovers_known_states_and_sensor_values(artifact):
    raw = json.loads(artifact.content)
    raw["initial_distribution"] = {"healthy": 1.0, "degraded": 0.0}
    raw["transition_probabilities"] = {
        "healthy": {"healthy": 0.0, "degraded": 1.0},
        "degraded": {"healthy": 1.0, "degraded": 0.0},
    }
    raw["observation_models"]["cheap"]["alert_probability"] = {"healthy": 0.0, "degraded": 1.0}
    raw["observation_models"]["precise"]["alert_probability"] = {"healthy": 1.0, "degraded": 0.0}
    data = generate_dataset(ModelArtifact(json.dumps(raw).encode()), seeds=[0, 1], horizon=4)
    for episode in data.episodes:
        assert episode.states == ("healthy", "degraded", "healthy", "degraded", "healthy")
        for reading in episode.readings:
            expected = bool(reading.tick % 2)
            assert reading.value == (expected if reading.source_id == "cheap" else not expected)


def test_longer_episodes_preserve_the_existing_prefix(artifact):
    short = generate_dataset(artifact, seeds=[3], horizon=3).episodes[0]
    long = generate_dataset(artifact, seeds=[3], horizon=6).episodes[0]
    assert short.states == long.states[:4]
    assert short.readings == long.readings[:6]


@pytest.mark.parametrize("seeds", [[], [-1], [True], ["1"], [1, 1]])
def test_invalid_seed_sequences_are_rejected(artifact, seeds):
    with pytest.raises(ValueError):
        generate_dataset(artifact, seeds=seeds, horizon=2)


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5, 201])
def test_invalid_horizons_are_rejected(artifact, horizon):
    with pytest.raises(ValueError):
        generate_dataset(artifact, seeds=[0], horizon=horizon)


@pytest.mark.parametrize("mutation", [
    lambda raw: raw["episodes"][0]["states"].pop(),
    lambda raw: raw["episodes"][0]["readings"].pop(),
    lambda raw: raw["episodes"][0]["readings"].reverse(),
    lambda raw: raw["episodes"].append(raw["episodes"][0]),
    lambda raw: raw["source_ids"].reverse(),
])
def test_malformed_archives_cannot_silently_drop_or_reorder_ticks(artifact, mutation):
    raw = generate_dataset(artifact, seeds=[0], horizon=2).model_dump(mode="json")
    mutation(raw)
    with pytest.raises(ValidationError):
        Dataset.model_validate(raw)
