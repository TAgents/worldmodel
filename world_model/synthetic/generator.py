"""Versioned finite-state simulation with independent, counter-based streams."""

from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import fsum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from world_model.core import ModelArtifact, Reading
from world_model.core.serialization import canonical_json
from world_model.core.types import ContentHash, Contract, Identifier, NonNegativeInt, PositiveInt

GENERATOR_VERSION = "synthetic-sha256-v1"


class Episode(Contract):
    seed: NonNegativeInt
    states: tuple[Identifier, ...]
    readings: tuple[Reading, ...]


class Dataset(Contract):
    schema_version: Literal[1] = 1
    generator_version: Literal["synthetic-sha256-v1"] = GENERATOR_VERSION
    model_hash: ContentHash
    horizon: PositiveInt
    source_ids: tuple[Identifier, ...]
    episodes: Annotated[tuple[Episode, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_episodes(self) -> Self:
        if not self.source_ids or tuple(sorted(set(self.source_ids))) != self.source_ids:
            raise ValueError("sources must be nonempty, unique, and sorted")
        if len({episode.seed for episode in self.episodes}) != len(self.episodes):
            raise ValueError("episode seeds must be unique")
        expected = [(tick, source) for tick in range(self.horizon) for source in self.source_ids]
        for episode in self.episodes:
            if len(episode.states) != self.horizon + 1:
                raise ValueError("each episode needs one more state than origin ticks")
            if [(reading.tick, reading.source_id) for reading in episode.readings] != expected:
                raise ValueError("potential readings must cover every tick/source exactly once in order")
        return self

    def to_bytes(self) -> bytes:
        return canonical_json(self.model_dump(mode="json"))


def _uniform(seed: int, stream: str, tick: int) -> float:
    # This algorithm does not depend on Python's random module or sampling order.
    digest = sha256(canonical_json([GENERATOR_VERSION, seed, stream, tick])).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / (1 << 53)


def _sample(distribution: Mapping[str, float], support: tuple[str, ...], u: float) -> str:
    threshold = u * fsum(distribution.values())
    cumulative = 0.0
    last_possible = support[0]
    for state in support:
        probability = distribution[state]
        if probability > 0:
            last_possible = state
        cumulative += probability
        if threshold < cumulative:
            return state
    return last_possible  # Only reachable through floating-point summation error.


def generate_dataset(
    artifact: ModelArtifact, *, seeds: Sequence[int], horizon: int | None = None
) -> Dataset:
    """Generate potential observations; querying them later incurs acquisition cost.

    Hidden states and seeds belong exclusively to this dataset and its evaluator.
    A sensor's draws cannot affect state transitions or another sensor's draws.
    """
    model = artifact.spec
    horizon = model.acquisition_limits.episode_origin_ticks if horizon is None else horizon
    if type(horizon) is not int or not 1 <= horizon <= model.acquisition_limits.episode_origin_ticks:
        raise ValueError("horizon must be a positive integer within the model's episode limit")
    seeds = tuple(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonempty nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    sources = tuple(sorted(model.observation_models))
    episodes = []
    for seed in seeds:
        states = [_sample(model.initial_distribution, model.states, _uniform(seed, "state", 0))]
        for tick in range(1, horizon + 1):
            states.append(_sample(
                model.transition_probabilities[states[-1]], model.states, _uniform(seed, "state", tick)
            ))
        readings = tuple(
            Reading(
                tick=tick,
                source_id=source,
                value=_uniform(seed, f"sensor:{source}", tick)
                < model.observation_models[source].alert_probability[states[tick]],
            )
            for tick in range(horizon)
            for source in sources
        )
        episodes.append(Episode(seed=seed, states=tuple(states), readings=readings))
    return Dataset(
        model_hash=artifact.content_hash, horizon=horizon, source_ids=sources, episodes=episodes
    )
