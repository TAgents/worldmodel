from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from world_model.core import Action, Outcome
from world_model.core.types import Contract, Identifier, NonNegativeInt, PositiveInt, UTCDateTime

Bits = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Policy = Literal["information_gain", "fixed", "random"]
POLICY_VERSION = "entropy-per-credit-v1"


class InvestigationConfig(Contract):
    policy: Policy = "information_gain"
    allowed_sources: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    horizon: Annotated[int, Field(strict=True, ge=1, le=200)] = 20
    budget_credits: NonNegativeInt = 20
    max_reads: Annotated[int, Field(strict=True, ge=0, le=1000)] = 200
    min_gain_bits_per_credit: Bits = 0.0
    entropy_threshold_bits: Bits = 0.0
    fixed_source: Identifier | None = None
    policy_seed: NonNegativeInt = 0

    @model_validator(mode="after")
    def valid_sources(self) -> Self:
        if len(set(self.allowed_sources)) != len(self.allowed_sources):
            raise ValueError("allowed sensors must be unique")
        if self.policy == "fixed" and self.fixed_source not in self.allowed_sources:
            raise ValueError("fixed policy requires an allowed fixed_source")
        if self.policy != "fixed" and self.fixed_source is not None:
            raise ValueError("fixed_source only applies to the fixed policy")
        return self


class Candidate(Contract):
    source_id: Identifier
    cost_credits: PositiveInt
    information_gain_bits: Bits
    gain_bits_per_credit: Bits
    eligible: bool
    reason: str


class Decision(Contract):
    policy_version: Literal["entropy-per-credit-v1"] = POLICY_VERSION
    tick: NonNegativeInt
    kind: Literal["read", "advance_tick", "stop"]
    source_id: Identifier | None = None
    cost_credits: NonNegativeInt = 0
    reason: str
    entropy_bits: Bits
    released_credits: NonNegativeInt
    available_credits: NonNegativeInt
    candidates: tuple[Candidate, ...]


class InvestigationStep(Contract):
    request_id: Identifier
    ordinal: NonNegativeInt
    run_id: Identifier
    decision: Decision
    action: Action | None
    outcome: Outcome | None
    before_sequence: NonNegativeInt
    after_sequence: NonNegativeInt
    spent_credits: NonNegativeInt
    dispatched_reads: NonNegativeInt
    recorded_at: UTCDateTime


class StepReceipt(Contract):
    created: bool
    step: InvestigationStep


class InvestigationView(Contract):
    run_id: Identifier
    config: InvestigationConfig
    created_at: UTCDateTime
    spent_credits: NonNegativeInt
    dispatched_reads: NonNegativeInt
    stopped: bool
    steps: tuple[InvestigationStep, ...]
