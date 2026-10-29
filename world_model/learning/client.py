from world_model.agents.worker import Accept
from world_model.evaluation.client import PredictorProcess
from .session import LearnedFrame
from .worker import Configure


class LearnedProcess(PredictorProcess):
    worker_module = "world_model.learning.worker"

    def configure(self, revision, config, *, particles=256, particle_seed=20261028):
        return LearnedFrame.model_validate(self._exchange(Configure(revision=revision, config=config, particles=particles, particle_seed=particle_seed)))

    def accept(self, reading=None):
        return LearnedFrame.model_validate(self._exchange(Accept(reading=reading)))
