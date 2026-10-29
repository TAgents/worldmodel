import json
import random
from fractions import Fraction

import pytest

from world_model.core import ModelArtifact, Reading
from world_model.inference import (
    ConflictingReadingError,
    FilterResult,
    ImpossibleObservationError,
    InvalidEvidenceError,
    filter_state,
)

from .conftest import FIXTURES
from .oracle import enumerate_distribution

CASES = json.loads((FIXTURES / "posterior-cases-v1.json").read_text())["cases"]


def assert_distribution(actual, expected):
    assert set(actual) == set(expected)
    for state, probability in expected.items():
        assert actual[state] == pytest.approx(float(probability), abs=1e-12, rel=0)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_reference_cases_against_exact_enumeration(artifact, case):
    readings = [Reading.model_validate(item) for item in case["observations"]]
    result = filter_state(artifact, readings, state_tick=case["state_tick"])
    posterior = enumerate_distribution(artifact.content, readings, case["state_tick"])
    predictive = enumerate_distribution(artifact.content, readings, case["state_tick"] + 1)
    assert posterior["degraded"] == Fraction(case["posterior_degraded"])
    assert predictive["degraded"] == Fraction(case["next_tick_degraded"])
    assert_distribution(result.posterior, posterior)
    assert_distribution(result.predictive, predictive)
    assert result.model_hash == artifact.content_hash
    assert result.forecast_tick == case["state_tick"] + 1


@pytest.mark.parametrize("state_tick", range(9))
def test_generated_sequences_with_missing_and_multiple_readings(artifact, state_tick):
    # Unit-test observations, not development or reserved benchmark episodes.
    rng = random.Random(7000 + state_tick)
    for _ in range(4):
        readings = [
            Reading(tick=tick, source_id=source, value=bool(rng.getrandbits(1)))
            for tick in range(state_tick + 1)
            for source in artifact.spec.observation_models
            if rng.random() < 0.6
        ]
        rng.shuffle(readings)
        result = filter_state(artifact, readings, state_tick=state_tick)
        assert_distribution(result.posterior, enumerate_distribution(artifact.content, readings, state_tick))
        assert_distribution(result.predictive, enumerate_distribution(artifact.content, readings, state_tick + 1))


def test_identical_copies_and_order_do_not_change_results(artifact):
    a = Reading(tick=0, source_id="cheap", value=True)
    b = Reading(tick=0, source_id="precise", value=False)
    c = Reading(tick=2, source_id="cheap", value=False)
    expected = filter_state(artifact, [a, b, c], state_tick=3)
    assert filter_state(artifact, [c, b, a, c, a, b], state_tick=3) == expected
    assert len(expected.readings) == 3


def test_conflicting_readings_require_explicit_correction_resolution(artifact):
    with pytest.raises(ConflictingReadingError):
        filter_state(artifact, [
            Reading(tick=0, source_id="cheap", value=True),
            Reading(tick=0, source_id="cheap", value=False),
        ], state_tick=0)


def test_future_evidence_is_rejected(artifact):
    with pytest.raises(InvalidEvidenceError, match="future"):
        filter_state(artifact, [Reading(tick=1, source_id="cheap", value=True)], state_tick=0)


def test_unknown_sensor_is_rejected(artifact):
    with pytest.raises(InvalidEvidenceError, match="unknown"):
        filter_state(artifact, [Reading(tick=0, source_id="invented", value=True)], state_tick=0)


@pytest.mark.parametrize("tick", [-1, True, 1.5, "1"])
def test_invalid_query_ticks_are_rejected(artifact, tick):
    with pytest.raises(ValueError, match="state_tick"):
        filter_state(artifact, [], state_tick=tick)


def test_unvalidated_readings_are_rejected(artifact):
    with pytest.raises(TypeError, match="validated"):
        filter_state(artifact, [{"tick": 0, "source_id": "cheap", "value": True}], state_tick=0)


def test_no_evidence_still_advances_every_tick(artifact):
    result = filter_state(artifact, [], state_tick=100)
    # Closed-form solution of this two-state chain, independent of filter code.
    expected = 1 / 6 + (0.1 - 1 / 6) * 0.88**100
    assert result.posterior["degraded"] == pytest.approx(expected, abs=1e-12, rel=0)


def test_impossible_reading_is_not_replaced_with_a_fabricated_posterior(artifact):
    raw = json.loads(artifact.content)
    raw["observation_models"]["cheap"]["alert_probability"] = {"healthy": 0, "degraded": 0}
    model = ModelArtifact(json.dumps(raw).encode())
    with pytest.raises(ImpossibleObservationError):
        filter_state(model, [Reading(tick=0, source_id="cheap", value=True)], state_tick=0)


def test_log_conditioning_preserves_tiny_nonzero_likelihoods(artifact):
    raw = json.loads(artifact.content)
    raw["initial_distribution"] = {"healthy": 1e-200, "degraded": 1.0}
    raw["observation_models"]["cheap"]["alert_probability"] = {"healthy": 1e-200, "degraded": 1e-300}
    model = ModelArtifact(json.dumps(raw).encode())
    result = filter_state(model, [Reading(tick=0, source_id="cheap", value=True)], state_tick=0)
    assert result.posterior["healthy"] == pytest.approx(1e-100, rel=1e-10, abs=0)


def test_three_states_and_different_domain_labels(artifact):
    raw = json.loads(artifact.content)
    raw.update({
        "variable": "temperature_band",
        "states": ["cold", "warm", "hot"],
        "initial_distribution": {"cold": 0.2, "warm": 0.3, "hot": 0.5},
        "transition_probabilities": {
            "cold": {"cold": 0.7, "warm": 0.2, "hot": 0.1},
            "warm": {"cold": 0.1, "warm": 0.6, "hot": 0.3},
            "hot": {"cold": 0.2, "warm": 0.2, "hot": 0.6},
        },
        "observation_models": {
            "probe": {"value_type": "boolean", "alert_probability": {"cold": 0.1, "warm": 0.5, "hot": 0.9}, "read_cost_credits": 2}
        },
    })
    model = ModelArtifact(json.dumps(raw).encode())
    readings = [Reading(tick=0, source_id="probe", value=True), Reading(tick=2, source_id="probe", value=False)]
    result = filter_state(model, readings, state_tick=3)
    assert result.variable == "temperature_band"
    assert_distribution(result.posterior, enumerate_distribution(model.content, readings, 3))
    assert_distribution(result.predictive, enumerate_distribution(model.content, readings, 4))


def test_results_are_immutable_and_round_trip_json(artifact):
    result = filter_state(artifact, [], state_tick=0)
    with pytest.raises(TypeError):
        result.posterior["healthy"] = 0.1
    assert FilterResult.model_validate_json(result.model_dump_json()) == result


def test_later_calls_cannot_mutate_previous_results(artifact):
    original = filter_state(artifact, [Reading(tick=0, source_id="cheap", value=True)], state_tick=1)
    saved = original.model_dump_json()
    changed = filter_state(artifact, [
        Reading(tick=0, source_id="cheap", value=True),
        Reading(tick=0, source_id="precise", value=False),
    ], state_tick=2)
    assert original.model_dump_json() == saved
    assert changed.posterior != original.posterior
