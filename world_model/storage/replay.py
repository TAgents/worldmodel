"""Validate and reduce recorded events without storage, sensors, or new facts."""

from datetime import timedelta
from math import isclose

from pydantic import TypeAdapter

from world_model.core import EvidenceReference, ModelArtifact, Observation, Reading, RunSpec
from world_model.core.types import ContentHash, UTCDateTime
from world_model.inference import INFERENCE_VERSION, filter_state

from .contracts import Publication, ReplayIntegrityError


class ReplayState:
    def __init__(self, artifact: ModelArtifact, run: RunSpec, target_state: str):
        if run.model_hash != artifact.content_hash or target_state not in artifact.spec.states:
            raise ReplayIntegrityError("run and replay model are inconsistent")
        self.artifact = artifact
        self.run = run
        self.target_state = target_state
        self.tick = 0
        self.sequence = -1
        self.recorded_at = None
        self.publication = None
        self.active: dict[tuple[str, int], Observation] = {}
        self.identities = set()
        self.closed_forecasts: list[str] = []

    def apply(self, event: dict, publication: Publication) -> None:
        sequence = event["sequence"]
        timestamp = TypeAdapter(UTCDateTime).validate_python(event["recorded_at"])
        if event["run_id"] != self.run.run_id or type(sequence) is not int or sequence != self.sequence + 1:
            raise ReplayIntegrityError("event history must be contiguous and belong to one run")
        if self.recorded_at is not None and timestamp <= self.recorded_at:
            raise ReplayIntegrityError("knowledge timestamps must strictly increase per run")
        kind, payload = event["kind"], event["payload"]
        if sequence == 0:
            expected = {"run": self.run.model_dump(mode="json"), "target_state": self.target_state}
            if "import_plan_hash" in payload:
                expected["import_plan_hash"] = TypeAdapter(ContentHash).validate_python(payload["import_plan_hash"])
            if kind != "run_created" or payload != expected or timestamp != self.run.created_at:
                raise ReplayIntegrityError("invalid run creation event")
        elif kind == "tick_advanced":
            expected = {"tick": self.tick + 1, "frozen_forecast_id": self.publication.forecast.forecast_id}
            if payload != expected:
                raise ReplayIntegrityError("tick event does not close the preceding forecast")
            if timestamp < self.run.epoch + timedelta(seconds=(self.tick + 1) * self.artifact.spec.tick_duration_seconds):
                raise ReplayIntegrityError("tick was published before its environment time")
            self.closed_forecasts.append(self.publication.forecast.forecast_id)
            self.tick += 1
        elif kind == "observation_ingested":
            observation = Observation.model_validate(payload)
            reading = observation.payload
            if (
                observation.run_id != self.run.run_id or observation.event_sequence != sequence
                or observation.ingested_at != timestamp or reading.tick > self.tick
                or reading.source_id not in self.artifact.spec.observation_models
                or observation.observed_at != self.run.epoch + timedelta(seconds=reading.tick * self.artifact.spec.tick_duration_seconds)
            ):
                raise ReplayIntegrityError("observation does not match its event or model")
            identity = (reading.source_id, reading.measurement_id)
            if identity in self.identities:
                raise ReplayIntegrityError("duplicate measurement identity in event history")
            slot = (reading.source_id, reading.tick)
            previous = self.active.get(slot)
            if reading.supersedes_id != (previous.observation_id if previous else None):
                raise ReplayIntegrityError("correction must replace the active observation in its slot")
            self.identities.add(identity)
            self.active[slot] = observation
        else:
            raise ReplayIntegrityError("unsupported event kind")

        observations = sorted(self.active.values(), key=lambda item: item.event_sequence)
        result = filter_state(self.artifact, [Reading(
            tick=item.payload.tick, source_id=item.payload.source_id, value=item.payload.value
        ) for item in observations], state_tick=self.tick)
        belief, snapshot, forecast = publication.belief, publication.evidence, publication.forecast
        expected_refs = tuple(EvidenceReference(observation_id=item.observation_id, event_sequence=item.event_sequence) for item in observations)
        if (
            belief.run_id != self.run.run_id or belief.state_tick != self.tick
            or belief.published_at != timestamp or belief.sequence_cutoff != sequence
            or belief.variable != self.artifact.spec.variable or belief.model_hash != self.artifact.content_hash
            or belief.inference_version != INFERENCE_VERSION
            or belief.previous_revision_id != (self.publication.belief.revision_id if self.publication else None)
            or snapshot.run_id != self.run.run_id or snapshot.sequence_cutoff != sequence
            or snapshot.snapshot_id != belief.evidence_snapshot_id or snapshot.observations != expected_refs
        ):
            raise ReplayIntegrityError("belief metadata or evidence snapshot diverges from replay")
        if set(belief.distribution) != set(result.posterior) or any(
            not isclose(belief.distribution[state], result.posterior[state], rel_tol=0, abs_tol=1e-12)
            for state in result.posterior
        ):
            raise ReplayIntegrityError("posterior diverges from replay")
        if (
            forecast.run_id != self.run.run_id or forecast.origin_tick != self.tick
            or forecast.target_state != self.target_state or forecast.issued_at != timestamp
            or forecast.belief_revision_id != belief.revision_id or forecast.model_hash != belief.model_hash
            or forecast.evidence_snapshot_id != snapshot.snapshot_id
            or not isclose(forecast.probability, result.predictive[self.target_state], rel_tol=0, abs_tol=1e-12)
        ):
            raise ReplayIntegrityError("forecast diverges from replay")
        self.sequence, self.recorded_at, self.publication = sequence, timestamp, publication
