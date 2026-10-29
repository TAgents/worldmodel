"""Exact finite-state Bayesian filtering, with no clock, IO, or hidden truth.

The caller supplies the active readings for its evidence snapshot. Resolving
availability cutoffs and correction chains is a separate persistence concern.
"""

from collections.abc import Iterable, Mapping
from math import exp, fsum, inf, log
from typing import Self

from pydantic import model_validator

from world_model.core import ModelArtifact, ModelSpec, Reading
from world_model.core.types import ContentHash, Contract, Distribution, Identifier, NonNegativeInt

INFERENCE_VERSION = "exact-discrete-v1"


class InvalidEvidenceError(ValueError):
    """The supplied readings cannot form a valid evidence snapshot."""


class ConflictingReadingError(InvalidEvidenceError):
    """A sensor/tick contains conflicting readings; resolve the correction first."""


class ImpossibleObservationError(InvalidEvidenceError):
    """The observation has zero predictive probability under this model."""


class FilterResult(Contract):
    model_hash: ContentHash
    inference_version: Identifier = INFERENCE_VERSION
    variable: Identifier
    state_tick: NonNegativeInt
    posterior: Distribution
    forecast_tick: NonNegativeInt
    predictive: Distribution
    readings: tuple[Reading, ...]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.forecast_tick != self.state_tick + 1:
            raise ValueError("the forecast must be one tick ahead")
        if set(self.posterior) != set(self.predictive):
            raise ValueError("posterior and predictive distributions must share state support")
        return self


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    total = fsum(weights.values())
    return {state: weight / total for state, weight in weights.items()}


def _predict(model: ModelSpec, distribution: Mapping[str, float]) -> dict[str, float]:
    return _normalize({
        target: fsum(
            distribution[source] * model.transition_probabilities[source][target]
            for source in model.states
        )
        for target in model.states
    })


def _condition(
    model: ModelSpec, distribution: Mapping[str, float], reading: Reading
) -> dict[str, float]:
    sensor = model.observation_models[reading.source_id]
    log_weights = {}
    for state in model.states:
        likelihood = sensor.alert_probability[state]
        if not reading.value:
            likelihood = 1.0 - likelihood
        prior = distribution[state]
        log_weights[state] = log(prior) + log(likelihood) if prior and likelihood else -inf
    largest = max(log_weights.values())
    if largest == -inf:
        raise ImpossibleObservationError(
            f"reading from {reading.source_id!r} at tick {reading.tick} "
            "has zero probability under the model"
        )
    return _normalize({state: exp(weight - largest) for state, weight in log_weights.items()})


def filter_state(
    artifact: ModelArtifact, readings: Iterable[Reading], *, state_tick: int
) -> FilterResult:
    """Replay selected readings and predict one tick ahead.

    Identical copies of a sensor/tick reading are counted once. Conflicting
    values fail explicitly: this function never guesses which one is a correction.
    Input order has no effect; evidence is processed in tick/source order.
    """
    if type(state_tick) is not int or state_tick < artifact.spec.initial_tick:
        raise ValueError("state_tick must be an integer at or after the initial tick")
    model = artifact.spec
    unique: dict[tuple[int, str], Reading] = {}
    for reading in readings:
        if not isinstance(reading, Reading):
            raise TypeError("readings must be validated Reading objects")
        if reading.source_id not in model.observation_models:
            raise InvalidEvidenceError(f"unknown observation source {reading.source_id!r}")
        if reading.tick > state_tick:
            raise InvalidEvidenceError("future-tick evidence cannot be used for filtering")
        key = (reading.tick, reading.source_id)
        previous = unique.get(key)
        if previous is not None and previous.value != reading.value:
            raise ConflictingReadingError(f"conflicting readings for sensor/tick {key!r}")
        # Store a plain Reading, without any observation envelope or other metadata.
        unique[key] = Reading(tick=reading.tick, source_id=reading.source_id, value=reading.value)

    ordered = tuple(unique[key] for key in sorted(unique))
    posterior = dict(model.initial_distribution)
    tick = model.initial_tick
    for reading in ordered:
        while tick < reading.tick:
            posterior = _predict(model, posterior)
            tick += 1
        posterior = _condition(model, posterior, reading)
    while tick < state_tick:
        posterior = _predict(model, posterior)
        tick += 1

    return FilterResult(
        model_hash=artifact.content_hash,
        variable=model.variable,
        state_tick=state_tick,
        posterior=posterior,
        forecast_tick=state_tick + 1,
        predictive=_predict(model, posterior),
        readings=ordered,
    )
