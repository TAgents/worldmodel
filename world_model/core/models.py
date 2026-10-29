"""A finite-state model and its exact, content-addressed source artifact."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from .types import (
    ContentHash,
    Contract,
    Distribution,
    FrozenMap,
    Identifier,
    PositiveInt,
    NonNegativeInt,
    Probability,
    UTCDateTime,
)


class ObservationModel(Contract):
    value_type: Literal["boolean"]
    alert_probability: FrozenMap[Identifier, Probability]
    read_cost_credits: PositiveInt


class AcquisitionLimits(Contract):
    episode_origin_ticks: PositiveInt
    episode_budget_credits: PositiveInt
    max_dispatches_per_sensor_per_tick: Annotated[int, Field(strict=True, ge=1, le=1)]
    max_reads_per_episode: PositiveInt


class ModelDerivation(Contract):
    method: Literal["beta-posterior-mean-v1"]
    calibration_revision: ContentHash
    calibration_content_hash: ContentHash
    structure_hash: ContentHash
    as_of: UTCDateTime
    evidence_count: NonNegativeInt


class ModelSpec(Contract):
    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]
    model_id: Identifier
    model_version: Identifier
    variable: Identifier
    states: Annotated[tuple[Identifier, ...], Field(min_length=2)]
    initial_tick: Annotated[int, Field(strict=True, ge=0, le=0)]
    tick_duration_seconds: PositiveInt
    initial_distribution: Distribution
    transition_probabilities: FrozenMap[Identifier, Distribution]
    observation_models: FrozenMap[Identifier, ObservationModel]
    forecast_horizon_ticks: Annotated[int, Field(strict=True, ge=1, le=1)]
    acquisition_limits: AcquisitionLimits
    # Omit absent metadata so existing model transformations keep their bytes.
    derivation: ModelDerivation | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_state_support(self) -> Self:
        states = set(self.states)
        if len(states) != len(self.states):
            raise ValueError("state labels must be unique")
        if set(self.initial_distribution) != states:
            raise ValueError("the prior must contain exactly the declared states")
        if set(self.transition_probabilities) != states:
            raise ValueError("the transition matrix must contain one row per state")
        if any(set(row) != states for row in self.transition_probabilities.values()):
            raise ValueError("each transition row must contain exactly the declared states")
        if not self.observation_models:
            raise ValueError("at least one observation model is required")
        for source, sensor in self.observation_models.items():
            if set(sensor.alert_probability) != states:
                raise ValueError(f"observation model {source!r} must cover exactly the declared states")
        return self


class ModelRegistration(Contract):
    model_id: Identifier
    model_version: Identifier
    content_hash: ContentHash
    registered_at: UTCDateTime


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class ModelArtifact:
    """Hash the original bytes, not a reserialized approximation of the file."""

    content: bytes = field(repr=False)
    spec: ModelSpec = field(init=False)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise TypeError("model artifact content must be immutable bytes")
        decoded = json.loads(self.content, object_pairs_hook=_unique_object)
        object.__setattr__(self, "spec", ModelSpec.model_validate(decoded))
        object.__setattr__(self, "content_hash", hashlib.sha256(self.content).hexdigest())

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cls(Path(path).read_bytes())
