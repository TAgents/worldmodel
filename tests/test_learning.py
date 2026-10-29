from copy import deepcopy
from datetime import timedelta
from fractions import Fraction
from itertools import product
import json

import numpy as np
import pytest

from world_model.agents.contracts import InvestigationConfig
from world_model.core import ModelArtifact, Reading
from world_model.core.serialization import canonical_json
from world_model.learning.calibration import BetaPrior, CalibrationObservation, fit_calibration, mean_artifact, verify_revision
from world_model.learning.client import LearnedProcess
from world_model.learning.evaluation import evaluate_reliability
from world_model.learning.inference import ParameterFilter, parameter_bank
from world_model.learning.session import LearnedSession
from world_model.learning.synthetic import CALIBRATION_EPOCH, calibration_cutoff, generate_calibration, neutral_structure
from world_model.learning.worker import REQUEST
from world_model.synthetic.generator import Dataset, generate_dataset
from .conftest import FIXTURES, ROOT


@pytest.fixture
def calibrated():
    truth = ModelArtifact.load(FIXTURES / "complementary-world-v1.json")
    structure = neutral_structure(truth)
    observations = generate_calibration(truth, seed=30000, samples_per_state=8)
    revision = fit_calibration(structure, observations, as_of=calibration_cutoff(structure, 8))
    return truth, structure, observations, revision


def test_beta_updates_intervals_and_measurement_deduplication(artifact):
    structure = neutral_structure(artifact)
    observations = [CalibrationObservation(measurement_id=f"m{i}", source_id="cheap", state="degraded", value=value,
        observed_at=CALIBRATION_EPOCH, available_at=CALIBRATION_EPOCH, source_ref=f"test/{i}") for i, value in enumerate((True, True, True, False))]
    revision = fit_calibration(structure, observations * 2, as_of=CALIBRATION_EPOCH)
    cell = next(c for c in revision.cells if c.source_id == "cheap" and c.state == "degraded")
    assert (cell.alpha, cell.beta, cell.alerts, cell.non_alerts) == (4., 2., 3, 1)
    assert cell.mean == pytest.approx(2 / 3)
    assert cell.variance == pytest.approx(2 / 63)
    assert cell.lower_95 < cell.mean < cell.upper_95
    unseen = next(c for c in revision.cells if c.source_id == "precise")
    assert unseen.lower_95 == pytest.approx(.025)
    assert unseen.upper_95 == pytest.approx(.975)
    assert len(revision.evidence) == 4
    assert fit_calibration(structure, reversed(observations), as_of=CALIBRATION_EPOCH) == revision
    with pytest.raises(ValueError, match="conflicting"):
        fit_calibration(structure, [*observations, observations[0].model_copy(update={"value": False})], as_of=CALIBRATION_EPOCH)


def test_historical_calibration_uses_only_available_labels(calibrated):
    _, structure, observations, _ = calibrated
    cutoff = calibration_cutoff(structure, 2)
    early = fit_calibration(structure, observations, as_of=cutoff)
    changed_future = [o if o.available_at <= cutoff else o.model_copy(update={"value": not o.value, "state": "unknown_future_label"}) for o in observations]
    assert fit_calibration(structure, changed_future, as_of=cutoff) == early
    assert len(early.evidence) == 2 * 3 * 3
    assert all(o.available_at <= cutoff for o in early.evidence)
    assert fit_calibration(structure, observations, as_of=cutoff + timedelta(seconds=3)).revision_id != early.revision_id
    with pytest.raises(ValueError, match="before"):
        CalibrationObservation.model_validate(observations[0].model_dump() | {"available_at": CALIBRATION_EPOCH - timedelta(seconds=1)})


def test_label_integrity_and_revision_provenance(calibrated):
    truth, structure, observations, revision = calibrated
    with pytest.raises(ValueError, match="masked"):
        fit_calibration(truth, observations, as_of=revision.as_of)
    corrupt = revision.model_copy(update={"cells": (revision.cells[0].model_copy(update={"alpha": 100.}), *revision.cells[1:])})
    with pytest.raises(ValueError, match="does not match"):
        verify_revision(structure, corrupt)
    mislabeled = list(observations)
    mislabeled[1] = mislabeled[1].model_copy(update={"state": "mode_b"})
    with pytest.raises(ValueError, match="verified label"):
        fit_calibration(structure, mislabeled, as_of=revision.as_of)
    assert all(mean_artifact(structure, revision).spec.observation_models[c.source_id].alert_probability[c.state] == c.mean for c in revision.cells)
    assert parameter_bank(structure, revision).flags.writeable is False
    assert np.array_equal(parameter_bank(structure, revision), parameter_bank(structure, revision))


def finite_bank_oracle(artifact, bank, readings, tick):
    """Enumerate parameter atoms and complete state paths with rational arithmetic."""
    model = json.loads(artifact.content, parse_float=Fraction)
    states, sources = model["states"], sorted(model["observation_models"])
    mass = {s: Fraction(0) for s in states}
    for particle in bank:
        for path in product(states, repeat=tick + 1):
            weight = Fraction(model["initial_distribution"][path[0]]) / len(bank)
            for t in range(1, tick + 1):
                weight *= model["transition_probabilities"][path[t - 1]][path[t]]
            for reading in readings:
                p = Fraction(str(particle[sources.index(reading.source_id)][states.index(path[reading.tick])]))
                weight *= p if reading.value else 1 - p
            mass[path[-1]] += weight
    total = sum(mass.values())
    return {s: float(p / total) for s, p in mass.items()}


def test_parameter_state_mixture_matches_independent_path_enumeration(artifact):
    structure = neutral_structure(artifact)
    revision = fit_calibration(structure, [], as_of=CALIBRATION_EPOCH)
    bank = np.array([[[.1, .8], [.3, .6]], [[.4, .7], [.05, .95]], [[.2, .95], [.15, .7]]])
    filter = ParameterFilter(structure, revision, samples=bank)
    readings = []
    for tick, source, value in ((0, "cheap", True), (1, "precise", False), (2, "cheap", True)):
        while filter.tick < tick:
            filter.advance()
        reading = Reading(tick=tick, source_id=source, value=value)
        filter.condition(reading)
        readings.append(reading)
        posterior, predictive, effective, ess = filter.state()
        assert posterior == pytest.approx(finite_bank_oracle(structure, bank, readings, tick), abs=1e-12)
        assert predictive == pytest.approx(finite_bank_oracle(structure, bank, readings, tick + 1), abs=1e-12)
        assert 1 <= ess <= len(bank) + 1e-12
        # The effective likelihood must produce the correct state posterior
        # for a hypothetical next reading, including state/parameter dependence.
        probe = Reading(tick=tick, source_id="precise" if source == "cheap" else "cheap", value=True)
        expected = {s: posterior[s] * effective[probe.source_id][s] for s in posterior}
        expected = {s: p / sum(expected.values()) for s, p in expected.items()}
        assert expected == pytest.approx(finite_bank_oracle(structure, bank, [*readings, probe], tick), abs=1e-12)


def test_static_parameter_uncertainty_is_not_replaced_by_mean_likelihood(artifact):
    raw = neutral_structure(artifact).spec.model_dump(mode="json")
    raw["initial_distribution"] = {"healthy": .5, "degraded": .5}
    raw["transition_probabilities"] = {s: {t: float(s == t) for t in raw["states"]} for s in raw["states"]}
    structure = ModelArtifact(canonical_json(raw))
    revision = fit_calibration(structure, [], as_of=CALIBRATION_EPOCH)
    bank = np.array([[[.1, .2], [.5, .5]], [[.9, .8], [.5, .5]]])
    filter = ParameterFilter(structure, revision, samples=bank)
    filter.condition(Reading(tick=0, source_id="cheap", value=True))
    assert filter.state()[0]["healthy"] == pytest.approx(.5)
    filter.advance()
    filter.condition(Reading(tick=1, source_id="cheap", value=True))
    assert filter.state()[0]["healthy"] == pytest.approx(.41 / (.41 + .34))
    before = filter.state()
    filter.condition(Reading(tick=1, source_id="cheap", value=True))
    assert filter.state() == before
    with pytest.raises(ValueError, match="conflicting"):
        filter.condition(Reading(tick=1, source_id="cheap", value=False))
    assert filter.state() == before


def test_learned_worker_boundary_and_budget(calibrated):
    truth, structure, _, revision = calibrated
    config = InvestigationConfig(allowed_sources=tuple(structure.spec.observation_models), horizon=2, budget_credits=2, max_reads=2)
    with pytest.raises(RuntimeError, match="masked"):
        with LearnedProcess(truth):
            pass
    for key in ("hidden_state", "environment_seed", "true_likelihoods"):
        with pytest.raises(ValueError):
            REQUEST.validate_python({"operation": "accept", "reading": None, key: "forbidden"})
    with LearnedProcess(structure) as worker:
        frame = worker.configure(revision, config, particles=32)
        assert frame.calibration_revision == revision.revision_id
        assert frame.decision.kind == "read"
        updated = worker.accept(Reading(tick=0, source_id=frame.decision.source_id, value=True))
        assert updated.spent_credits == 1
        assert updated.decision.kind == "advance_tick"
        assert worker.accept().decision.tick == 1
    session = LearnedSession(structure, revision, config, particles=32)
    before = session.frame()
    with pytest.raises(ValueError, match="requested"):
        session.accept(Reading(tick=1, source_id=before.decision.source_id, value=True))
    assert session.frame() == before


def test_learning_evaluation_cannot_read_labels_or_future_observations(calibrated):
    truth, _, _, revision = calibrated
    protocol = json.loads((ROOT / "docs/reliability-protocol-v1.json").read_bytes())
    protocol.update(horizon=4, budget_credits=4, max_reads=4, particle_count=32, bootstrap_resamples=20)
    dataset = generate_dataset(truth, seeds=(31000, 31001), horizon=4)
    result = evaluate_reliability(truth, dataset, revision, protocol)
    assert len(result.predictions) == 2 * 4 * 8
    assert all(m["credits_per_episode"] == [4, 4] for m in result.report["methods"].values())
    raw = dataset.model_dump(mode="json")
    for episode in raw["episodes"]:
        episode["states"] = ["mode_b" if s == "mode_a" else "mode_a" for s in episode["states"]]
    other = evaluate_reliability(truth, Dataset.model_validate(raw), revision, protocol)
    assert result.actions == other.actions
    assert result.predictions == other.predictions
    raw = deepcopy(dataset.model_dump(mode="json"))
    for episode in raw["episodes"]:
        for reading in episode["readings"]:
            if reading["tick"] >= 2:
                reading["value"] = not reading["value"]
    later = evaluate_reliability(truth, Dataset.model_validate(raw), revision, protocol)
    assert [p for p in result.predictions if p["origin_tick"] < 2] == [p for p in later.predictions if p["origin_tick"] < 2]
    assert result.report["heldout_acquisition_gate_passed"] is None
    with pytest.raises(ValueError, match="locked"):
        heldout = generate_dataset(truth, seeds=(32000, 32001), horizon=4)
        evaluate_reliability(truth, heldout, revision, protocol, split="heldout")
