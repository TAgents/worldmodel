from copy import deepcopy
import json
from math import log2

import pytest

from world_model.agents.client import AgentProcess
from world_model.agents.contracts import InvestigationConfig
from world_model.agents.policy import choose, entropy, information_gain
from world_model.agents.session import InvestigationSession
from world_model.agents.worker import REQUEST
from world_model.core import ModelArtifact, Reading
from world_model.synthetic.adapter import SyntheticSensorAdapter
from world_model.synthetic.generator import Dataset, generate_dataset
from world_model.evaluation.acquisition import evaluate_acquisition
from world_model.evaluation.runner import archive_result
from world_model.agents.audit import verify_trace
from .investigation_fixtures import investigation_snapshots


def config(**kwargs):
    return InvestigationConfig(**({"allowed_sources": ("cheap", "precise"), "horizon": 3, "budget_credits": 3, "max_reads": 3} | kwargs))


@pytest.mark.parametrize("p", [0.0, 0.01, .1, .5, .9, .99, 1.0])
def test_information_gain_matches_independent_expected_kl(artifact, p):
    prior = {"healthy": 1 - p, "degraded": p}
    for sensor in artifact.spec.observation_models.values():
        expected = 0.0
        for alert in (False, True):
            likelihood = {s: sensor.alert_probability[s] if alert else 1 - sensor.alert_probability[s] for s in prior}
            marginal = sum(prior[s] * likelihood[s] for s in prior)
            if marginal:
                for state in prior:
                    posterior = prior[state] * likelihood[state] / marginal
                    if posterior:
                        expected += marginal * posterior * log2(posterior / prior[state])
        assert information_gain(prior, sensor.alert_probability) == pytest.approx(expected, abs=1e-14)


def test_budget_pacing_permissions_repeat_limits_and_stopping(artifact):
    prior = artifact.spec.initial_distribution
    initial = choose(artifact, config(), prior, tick=0, spent=0, reads=0)
    assert initial.source_id == "cheap"
    assert initial.available_credits == initial.released_credits == 1
    assert initial.candidates[1].reason == "insufficient released credits"
    assert choose(artifact, config(), prior, tick=0, spent=1, reads=1, used_sources=("cheap",)).kind == "advance_tick"
    assert choose(artifact, config(entropy_threshold_bits=1.), prior, tick=0, spent=0, reads=0).kind == "advance_tick"
    assert choose(artifact, config(min_gain_bits_per_credit=1.), prior, tick=0, spent=0, reads=0).kind == "advance_tick"
    assert choose(artifact, config(max_reads=0), prior, tick=0, spent=0, reads=0).kind == "advance_tick"
    assert choose(artifact, config(), prior, tick=3, spent=3, reads=3).kind == "stop"
    permitted = choose(artifact, config(allowed_sources=("precise",)), prior, tick=0, spent=0, reads=0)
    assert [c.source_id for c in permitted.candidates] == ["precise"]
    assert permitted.kind == "advance_tick"
    with pytest.raises(ValueError, match="released"):
        choose(artifact, config(), prior, tick=0, spent=2, reads=1)


@pytest.mark.parametrize("change", [{"allowed_sources": ("unknown",)}, {"budget_credits": 201}, {"horizon": 201}, {"max_reads": 201}, {"allowed_sources": ("cheap", "cheap")}, {"policy": "fixed"}, {"budget_credits": True}])
def test_configuration_cannot_bypass_model_limits(artifact, change):
    with pytest.raises(ValueError):
        choose(artifact, config(**change), artifact.spec.initial_distribution, tick=0, spent=0, reads=0)


def test_policy_choice_changes_with_belief_in_a_three_state_model(artifact):
    raw = json.loads(artifact.content)
    raw["states"] = ["a", "b", "c"]
    raw["initial_distribution"] = {"a": .8, "b": .1, "c": .1}
    raw["transition_probabilities"] = {s: {t: float(s == t) for t in raw["states"]} for s in raw["states"]}
    raw["observation_models"] = {s: {"value_type": "boolean", "alert_probability": {t: .95 if s == t else .5 for t in raw["states"]}, "read_cost_credits": 1} for s in ("a", "b")}
    model = ModelArtifact(json.dumps(raw).encode())
    settings = config(allowed_sources=("a", "b"))
    first = choose(model, settings, {"a": .8, "b": .1, "c": .1}, tick=0, spent=0, reads=0)
    second = choose(model, settings, {"a": .1, "b": .8, "c": .1}, tick=0, spent=0, reads=0)
    assert first.source_id == "a"
    assert second.source_id == "b"


def test_session_and_adapter_enforce_a_causal_read_boundary(artifact):
    session = InvestigationSession(artifact, config())
    sensor = SyntheticSensorAdapter(artifact, seed=0, horizon=3)
    initial = session.frame()
    with pytest.raises(ValueError, match="requested"):
        session.accept(Reading(tick=1, source_id="cheap", value=True))
    assert session.frame() == initial
    frame = session.accept(sensor.read(tick=0, source_id="cheap"))
    assert frame.spent_credits == frame.dispatched_reads == 1
    assert frame.decision.kind == "advance_tick"
    with pytest.raises(ValueError, match="advancement"):
        session.accept(sensor.read(tick=0, source_id="cheap"))
    assert session.frame() == frame
    assert session.accept().decision.tick == 1
    with pytest.raises(ValueError, match="outside"):
        sensor.read(tick=3, source_id="cheap")


def test_worker_protocol_rejects_hidden_environment_fields(artifact):
    for field in ("seed", "hidden_state", "future_readings", "dataset"):
        raw = {"operation": "configure", "config": config().model_dump(mode="json"), field: 1}
        with pytest.raises(ValueError):
            REQUEST.validate_python(raw)
    with AgentProcess(artifact) as agent:
        frame = agent.configure(config())
        assert frame.decision.kind == "read"
        with pytest.raises(RuntimeError, match="requested"):
            agent.accept(Reading(tick=1, source_id="cheap", value=True))


def test_random_and_fixed_policies_save_for_unaffordable_sensors(artifact):
    fixed = config(policy="fixed", fixed_source="precise", horizon=10, budget_credits=10, max_reads=10)
    for tick in range(4):
        assert choose(artifact, fixed, artifact.spec.initial_distribution, tick=tick, spent=0, reads=0).kind == "advance_tick"
    decision = choose(artifact, fixed, artifact.spec.initial_distribution, tick=4, spent=0, reads=0)
    assert decision.source_id == "precise"
    assert decision.cost_credits == 5
    random = config(policy="random", horizon=10, budget_credits=10)
    choices = [choose(artifact, random, artifact.spec.initial_distribution, tick=t, spent=0, reads=0) for t in range(10)]
    assert any(c.source_id == "precise" for c in choices)
    assert any(c.kind == "advance_tick" for c in choices[:4])


def test_acquisition_evaluation_costs_isolation_and_archive(artifact, tmp_path):
    dataset = generate_dataset(artifact, seeds=(0, 1), horizon=5)
    result = evaluate_acquisition(artifact, dataset, budget_credits=5, bootstrap_resamples=20)
    assert len(result.predictions) == 2 * 5 * 4
    methods = result.report["methods"]
    assert methods["information_gain"]["credits_per_episode"] == [5, 5]
    assert methods["fixed_precise"]["credits_per_episode"] == [5, 5]
    assert methods["information_gain"]["mean_brier"] == methods["fixed_cheap"]["mean_brier"]
    assert result.report["heldout_acquisition_gate_passed"] is None
    raw = dataset.model_dump(mode="json")
    for episode in raw["episodes"]:
        episode["states"] = ["healthy" if s == "degraded" else "degraded" for s in episode["states"]]
    changed = evaluate_acquisition(artifact, Dataset.model_validate(raw), budget_credits=5, bootstrap_resamples=20)
    assert changed.predictions == result.predictions
    assert changed.actions == result.actions
    future = deepcopy(dataset.model_dump(mode="json"))
    for episode in future["episodes"]:
        for reading in episode["readings"]:
            if reading["tick"] >= 3:
                reading["value"] = not reading["value"]
    later = evaluate_acquisition(artifact, Dataset.model_validate(future), budget_credits=5, bootstrap_resamples=20)
    assert [p for p in result.predictions if p.origin_tick < 3] == [p for p in later.predictions if p.origin_tick < 3]
    destination = archive_result(tmp_path / "archive", artifact, dataset, result)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert "actions.jsonl" in manifest["files"]
    with pytest.raises(FileExistsError):
        archive_result(destination, artifact, dataset, result)
    with pytest.raises(ValueError, match="prespecified"):
        heldout = generate_dataset(artifact, seeds=(2000, 2001), horizon=5)
        evaluate_acquisition(artifact, heldout, split="heldout")


def test_trace_replay_validates_decisions_costs_and_publication_links(artifact):
    snapshots, publications, observations = investigation_snapshots(artifact)
    final = snapshots[-1].investigation
    assert final.stopped
    assert verify_trace(artifact, final.config, final.steps, publications, observations,
        created_at=final.created_at, final_sequence=4) == 5
    for index in range(2, len(snapshots)):
        current = snapshots[index]
        assert verify_trace(artifact, final.config, current.investigation.steps, publications, observations,
            created_at=final.created_at, final_sequence=current.publication.belief.sequence_cutoff) == index - 1
    for field, value in (("spent_credits", 0), ("after_sequence", 0), ("before_sequence", 1), ("ordinal", 1)):
        modified = final.steps[0].model_copy(update={field: value})
        with pytest.raises(ValueError):
            verify_trace(artifact, final.config, (modified, *final.steps[1:]), publications, observations,
                created_at=final.created_at, final_sequence=4)
    corrupted_decision = final.steps[0].decision.model_copy(update={"reason": "unrecorded policy"})
    with pytest.raises(ValueError, match="decision"):
        verify_trace(artifact, final.config, (final.steps[0].model_copy(update={"decision": corrupted_decision}), *final.steps[1:]), publications, observations, created_at=final.created_at, final_sequence=4)
