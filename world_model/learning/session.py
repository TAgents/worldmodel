from typing import Annotated, Literal

from pydantic import Field

from world_model.agents.contracts import InvestigationConfig
from world_model.agents.policy import choose, validate_config
from world_model.agents.session import Frame
from world_model.core import Reading
from world_model.core.types import ContentHash, PositiveInt
from world_model.evaluation.protocol import readings_hash
from .inference import INFERENCE_VERSION, ParameterFilter


class LearnedFrame(Frame):
    inference_version: Literal["beta-mixture-filter-v1"] = INFERENCE_VERSION
    calibration_revision: ContentHash
    structure_hash: ContentHash
    particle_count: PositiveInt
    effective_particle_count: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    parameter_draws_hash: ContentHash


class LearnedSession:
    def __init__(self, artifact, revision, config: InvestigationConfig, *, particles=256, seed=20261028):
        validate_config(artifact, config)
        self.filter = ParameterFilter(artifact, revision, particles=particles, seed=seed)
        self.artifact, self.config = artifact, config
        self.spent = self.reads = 0
        self.readings, self.used = [], []

    def frame(self):
        posterior, predictive, likelihoods, ess = self.filter.state()
        return LearnedFrame(decision=choose(self.artifact, self.config, posterior, tick=self.filter.tick,
            spent=self.spent, reads=self.reads, used_sources=tuple(self.used), sensor_likelihoods=likelihoods),
            posterior=posterior, predictive=predictive, spent_credits=self.spent,
            dispatched_reads=self.reads, acquired_readings_hash=readings_hash(self.readings),
            calibration_revision=self.filter.revision.revision_id, structure_hash=self.artifact.content_hash,
            particle_count=self.filter.particles, effective_particle_count=ess, parameter_draws_hash=self.filter.draws_hash)

    def accept(self, reading=None):
        decision = self.frame().decision
        if decision.kind == "stop":
            raise ValueError("episode is complete")
        if decision.kind == "read":
            reading = Reading.model_validate(reading)
            if (reading.tick, reading.source_id) != (decision.tick, decision.source_id):
                raise ValueError("return only the requested current reading")
            self.filter.condition(reading)
            self.readings.append(reading)
            self.used.append(reading.source_id)
            self.spent += decision.cost_credits
            self.reads += 1
        else:
            if reading is not None:
                raise ValueError("advancement cannot introduce evidence")
            self.filter.advance()
            self.used = []
        return self.frame()
