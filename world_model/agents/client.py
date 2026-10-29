from world_model.evaluation.client import PredictorProcess
from .session import Frame
from .worker import Accept, Configure


class AgentProcess(PredictorProcess):
    worker_module = "world_model.agents.worker"

    def configure(self, config):
        return Frame.model_validate(self._exchange(Configure(config=config)))

    def accept(self, reading=None):
        return Frame.model_validate(self._exchange(Accept(reading=reading)))
