"""Sensor-only boundary; latent states never leave the adapter through its API."""

from typing import Protocol

from world_model.core import ModelArtifact, Reading
from .generator import generate_dataset


class SensorAdapter(Protocol):
    def read(self, *, tick: int, source_id: str) -> Reading: ...


class SyntheticSensorAdapter:
    def __init__(self, artifact: ModelArtifact, *, seed: int, horizon: int):
        episode = generate_dataset(artifact, seeds=(seed,), horizon=horizon).episodes[0]
        # Retain only readings; inference and the policy never receive this map.
        self._readings = {(r.tick, r.source_id): r for r in episode.readings}

    def read(self, *, tick: int, source_id: str) -> Reading:
        try:
            return self._readings[tick, source_id]
        except KeyError as error:
            raise ValueError("sensor request outside the adapter's episode") from error
