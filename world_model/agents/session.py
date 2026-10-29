"""Pure causal investigation state machine, also used in an isolated evaluator worker."""

from world_model.core import ModelArtifact, Reading
from world_model.core.types import Contract, Distribution, NonNegativeInt
from world_model.evaluation.protocol import readings_hash
from world_model.inference import filter_state
from .contracts import Decision, InvestigationConfig
from .policy import choose, validate_config


class Frame(Contract):
    decision: Decision
    posterior: Distribution
    predictive: Distribution
    spent_credits: NonNegativeInt
    dispatched_reads: NonNegativeInt
    acquired_readings_hash: str


class InvestigationSession:
    def __init__(self, artifact: ModelArtifact, config: InvestigationConfig):
        validate_config(artifact, config)
        self.artifact, self.config = artifact, config
        self.tick = self.spent = self.reads = 0
        self.readings = []
        self.used = []

    def frame(self):
        result = filter_state(self.artifact, self.readings, state_tick=self.tick)
        return Frame(decision=choose(self.artifact, self.config, result.posterior,
            tick=self.tick, spent=self.spent, reads=self.reads, used_sources=tuple(self.used)),
            posterior=result.posterior, predictive=result.predictive,
            spent_credits=self.spent, dispatched_reads=self.reads,
            acquired_readings_hash=readings_hash(self.readings))

    def accept(self, reading: Reading | None = None):
        decision = self.frame().decision
        if decision.kind == "stop":
            raise ValueError("the episode has stopped")
        if decision.kind == "read":
            if reading is None or (reading.tick, reading.source_id) != (decision.tick, decision.source_id):
                raise ValueError("only the requested current reading may be returned")
            # Validate likelihood before mutating the causal state.
            history = [*self.readings, reading]
            filter_state(self.artifact, history, state_tick=self.tick)
            self.readings = history
            self.used.append(reading.source_id)
            self.spent += decision.cost_credits
            self.reads += 1
        else:
            if reading is not None:
                raise ValueError("time advancement cannot introduce a reading")
            self.tick += 1
            self.used = []
        return self.frame()
