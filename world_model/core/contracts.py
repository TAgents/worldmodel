"""Evidence, belief, forecast, and acquisition records for the first sprint.

Validation here is structural. Cross-record relationships, clock assignment,
correction resolution, and transactional publication belong to the storage layer.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, model_validator

from .types import (
    ContentHash,
    Contract,
    Distribution,
    Identifier,
    NonNegativeInt,
    PositiveInt,
    Probability,
    UTCDateTime,
)


class RunSpec(Contract):
    run_id: Identifier
    epoch: UTCDateTime
    created_at: UTCDateTime
    model_hash: ContentHash


class Reading(Contract):
    """One already-resolved measurement supplied to the pure inference layer."""

    tick: NonNegativeInt
    source_id: Identifier
    value: StrictBool


class ObservationPayload(Reading):
    measurement_id: Identifier
    supersedes_id: Identifier | None = None


class Observation(Contract):
    observation_id: Identifier
    run_id: Identifier
    payload: ObservationPayload
    observed_at: UTCDateTime
    ingested_at: UTCDateTime
    event_sequence: PositiveInt

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if self.ingested_at < self.observed_at:
            raise ValueError("ingestion cannot precede observation time")
        if self.payload.supersedes_id == self.observation_id:
            raise ValueError("an observation cannot supersede itself")
        return self


class EvidenceReference(Contract):
    observation_id: Identifier
    event_sequence: PositiveInt


class EvidenceSnapshot(Contract):
    snapshot_id: Identifier
    run_id: Identifier
    sequence_cutoff: NonNegativeInt
    observations: tuple[EvidenceReference, ...]

    @model_validator(mode="after")
    def validate_membership(self) -> Self:
        ids = [ref.observation_id for ref in self.observations]
        sequences = [ref.event_sequence for ref in self.observations]
        if len(set(ids)) != len(ids):
            raise ValueError("an evidence snapshot cannot contain duplicate observation IDs")
        if sequences != sorted(set(sequences)):
            raise ValueError("evidence references must have strictly increasing event sequences")
        if sequences and sequences[-1] > self.sequence_cutoff:
            raise ValueError("evidence cannot exceed the snapshot's sequence cutoff")
        return self


class BeliefRevision(Contract):
    revision_id: Identifier
    run_id: Identifier
    variable: Identifier
    state_tick: NonNegativeInt
    published_at: UTCDateTime
    sequence_cutoff: NonNegativeInt
    distribution: Distribution
    model_hash: ContentHash
    evidence_snapshot_id: Identifier
    inference_version: Identifier
    previous_revision_id: Identifier | None = None

    @model_validator(mode="after")
    def validate_previous_revision(self) -> Self:
        if self.previous_revision_id == self.revision_id:
            raise ValueError("a belief revision cannot be its own predecessor")
        return self


class Forecast(Contract):
    forecast_id: Identifier
    run_id: Identifier
    origin_tick: NonNegativeInt
    target_tick: PositiveInt
    target_state: Identifier
    probability: Probability
    issued_at: UTCDateTime
    belief_revision_id: Identifier
    model_hash: ContentHash
    evidence_snapshot_id: Identifier

    @model_validator(mode="after")
    def validate_horizon(self) -> Self:
        if self.target_tick != self.origin_tick + 1:
            raise ValueError("this model version forecasts exactly one tick ahead")
        return self


class Action(Contract):
    action_id: Identifier
    run_id: Identifier
    tick: NonNegativeInt
    kind: Literal["read", "advance_tick"]
    source_id: Identifier | None = None
    reserved_cost_credits: NonNegativeInt
    created_at: UTCDateTime
    dispatched_at: UTCDateTime | None = None

    @model_validator(mode="after")
    def validate_action(self) -> Self:
        if self.kind == "read":
            if self.source_id is None or self.reserved_cost_credits == 0:
                raise ValueError("a read requires a source and positive reserved cost")
        elif self.source_id is not None or self.reserved_cost_credits != 0:
            raise ValueError("advancing time has no sensor or credit cost")
        if self.dispatched_at is not None and self.dispatched_at < self.created_at:
            raise ValueError("dispatch cannot precede action creation")
        return self


class Outcome(Contract):
    outcome_id: Identifier
    action_id: Identifier
    run_id: Identifier
    status: Literal["succeeded", "failed", "rejected"]
    charged_cost_credits: NonNegativeInt
    completed_at: UTCDateTime
    observation_id: Identifier | None = None
    error: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.status == "succeeded":
            if self.error is not None:
                raise ValueError("a successful outcome cannot include an error")
        elif self.error is None or not self.error.strip() or self.observation_id is not None:
            raise ValueError("failed and rejected outcomes require an error and no observation")
        if self.status == "rejected" and self.charged_cost_credits != 0:
            raise ValueError("a rejected action cannot consume credits")
        return self
