from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from world_model.core import (
    BeliefRevision, EvidenceReference, EvidenceSnapshot, Forecast, Observation,
    ObservationPayload, Reading, RunSpec,
)
from world_model.storage.contracts import Publication, ReplayIntegrityError
from world_model.storage.replay import ReplayState

from .oracle import enumerate_distribution


def history_fixture(artifact):
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    run = RunSpec(run_id="run-1", epoch=epoch, created_at=epoch, model_hash=artifact.content_hash)
    def observation(name, sequence, tick, source, value, seconds, supersedes=None):
        return Observation(
            observation_id=name, run_id=run.run_id,
            payload=ObservationPayload(measurement_id=name, tick=tick, source_id=source, value=value, supersedes_id=supersedes),
            observed_at=epoch + timedelta(seconds=60 * tick), ingested_at=epoch + timedelta(seconds=seconds), event_sequence=sequence,
        )
    cheap0 = observation("cheap0", 1, 0, "cheap", True, 1)
    cheap1 = observation("cheap1", 3, 1, "cheap", False, 61)
    precise = observation("precise0", 5, 0, "precise", False, 121)
    corrected = observation("corrected0", 6, 0, "precise", True, 122, precise.observation_id)
    definitions = [
        (0, 0, "run_created", {"run": run.model_dump(mode="json"), "target_state": "degraded"}, []),
        (0, 1, "observation_ingested", cheap0.model_dump(mode="json"), [cheap0]),
        (1, 60, "tick_advanced", {"tick": 1, "frozen_forecast_id": "f1"}, [cheap0]),
        (1, 61, "observation_ingested", cheap1.model_dump(mode="json"), [cheap0, cheap1]),
        (2, 120, "tick_advanced", {"tick": 2, "frozen_forecast_id": "f3"}, [cheap0, cheap1]),
        (2, 121, "observation_ingested", precise.model_dump(mode="json"), [cheap0, cheap1, precise]),
        (2, 122, "observation_ingested", corrected.model_dump(mode="json"), [cheap0, cheap1, corrected]),
    ]
    frames = []
    for sequence, (tick, seconds, kind, payload, observations) in enumerate(definitions):
        timestamp = epoch + timedelta(seconds=seconds)
        readings = [Reading(tick=item.payload.tick, source_id=item.payload.source_id, value=item.payload.value) for item in observations]
        posterior = enumerate_distribution(artifact.content, readings, tick)
        predictive = enumerate_distribution(artifact.content, readings, tick + 1)
        snapshot = EvidenceSnapshot(snapshot_id=f"s{sequence}", run_id=run.run_id, sequence_cutoff=sequence, observations=[EvidenceReference(observation_id=item.observation_id, event_sequence=item.event_sequence) for item in observations])
        belief = BeliefRevision(revision_id=f"b{sequence}", run_id=run.run_id, variable="state", state_tick=tick, published_at=timestamp, sequence_cutoff=sequence, distribution={state: float(value) for state, value in posterior.items()}, model_hash=artifact.content_hash, evidence_snapshot_id=snapshot.snapshot_id, inference_version="exact-discrete-v1", previous_revision_id=f"b{sequence-1}" if sequence else None)
        forecast = Forecast(forecast_id=f"f{sequence}", run_id=run.run_id, origin_tick=tick, target_tick=tick + 1, target_state="degraded", probability=float(predictive["degraded"]), issued_at=timestamp, belief_revision_id=belief.revision_id, model_hash=artifact.content_hash, evidence_snapshot_id=snapshot.snapshot_id)
        frames.append((
            {"run_id": run.run_id, "sequence": sequence, "recorded_at": timestamp, "kind": kind, "payload": payload},
            Publication(belief=belief, evidence=snapshot, forecast=forecast),
        ))
    return run, frames


def test_replay_matches_independent_exact_inference_and_corrections(artifact):
    run, frames = history_fixture(artifact)
    state = ReplayState(artifact, run, "degraded")
    old_bytes = frames[1][1].model_dump_json()
    for event, publication in frames:
        state.apply(event, publication)
    assert state.sequence == 6
    assert state.tick == 2
    assert state.closed_forecasts == ["f1", "f3"]
    assert [ref.observation_id for ref in state.publication.evidence.observations] == ["cheap0", "cheap1", "corrected0"]
    assert frames[1][1].model_dump_json() == old_bytes
    assert frames[5][1].belief.distribution["degraded"] == pytest.approx(49391 / 1697350, abs=1e-12)


@pytest.mark.parametrize("cutoff", range(7))
def test_each_historical_prefix_has_only_its_available_evidence(artifact, cutoff):
    run, frames = history_fixture(artifact)
    state = ReplayState(artifact, run, "degraded")
    for event, publication in frames[:cutoff + 1]:
        state.apply(event, publication)
    assert state.publication == frames[cutoff][1]
    assert all(ref.event_sequence <= cutoff for ref in state.publication.evidence.observations)


@pytest.mark.parametrize("mutation", [
    lambda event: event.update(run_id="another-run"),
    lambda event: event.update(sequence=7),
    lambda event: event.update(recorded_at=datetime(2026, 1, 1, tzinfo=UTC)),
    lambda event: event["payload"]["payload"].update(tick=3),
    lambda event: event["payload"]["payload"].update(source_id="unknown"),
    lambda event: event["payload"]["payload"].update(measurement_id="cheap0", source_id="cheap"),
    lambda event: event["payload"]["payload"].update(supersedes_id="cheap0"),
])
def test_invalid_late_evidence_cannot_enter_replay(artifact, mutation):
    run, frames = history_fixture(artifact)
    state = ReplayState(artifact, run, "degraded")
    for event, publication in frames[:5]:
        state.apply(event, publication)
    event = deepcopy(frames[5][0])
    mutation(event)
    with pytest.raises(ReplayIntegrityError):
        state.apply(event, frames[5][1])


@pytest.mark.parametrize("mutation", [
    lambda raw: raw["belief"].update(distribution={"healthy": 0.5, "degraded": 0.5}),
    lambda raw: raw["belief"].update(inference_version="unavailable-version"),
    lambda raw: raw["belief"].update(previous_revision_id="wrong-predecessor"),
    lambda raw: raw["evidence"].update(observations=[]),
    lambda raw: raw["forecast"].update(probability=0.5),
    lambda raw: raw["forecast"].update(model_hash="0" * 64),
])
def test_saved_publications_are_verified_not_trusted(artifact, mutation):
    run, frames = history_fixture(artifact)
    state = ReplayState(artifact, run, "degraded")
    state.apply(*frames[0])
    raw = frames[1][1].model_dump(mode="json")
    mutation(raw)
    with pytest.raises(ReplayIntegrityError):
        state.apply(frames[1][0], Publication.model_validate(raw))


def test_tick_must_close_the_last_forecast_and_cannot_skip_time(artifact):
    run, frames = history_fixture(artifact)
    for change in ({"tick": 2}, {"frozen_forecast_id": "f0"}):
        state = ReplayState(artifact, run, "degraded")
        for frame in frames[:2]:
            state.apply(*frame)
        event = deepcopy(frames[2][0])
        event["payload"].update(change)
        with pytest.raises(ReplayIntegrityError):
            state.apply(event, frames[2][1])
