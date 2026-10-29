"""Read causal requests on stdin and publish predictions before seeing labels.

This module never imports the simulator or the evaluation runner.
"""

import base64
import json
import sys

from world_model.core import ModelArtifact, Reading
from world_model.inference import filter_state

from .protocol import REQUEST, PredictionBatch, ResetRequest, StartRequest, StepRequest, readings_hash


class ForecastWorker:
    def __init__(self, artifact: ModelArtifact):
        self.artifact = artifact
        self.reset()

    def reset(self) -> None:
        self._tick = 0
        self._readings: list[Reading] = []

    def step(self, request: StepRequest) -> PredictionBatch:
        if request.tick != self._tick:
            raise ValueError("steps must begin at zero and advance exactly one tick")
        current = sorted(request.readings, key=lambda reading: reading.source_id)
        history = self._readings + current
        # Ablations use the same exact inference implementation with different
        # evidence retention, so model-fitting differences cannot confound them.
        selected = {"filter": history, "transition_only": [], "memoryless": current}
        predictions = {
            name: filter_state(self.artifact, readings, state_tick=request.tick).predictive
            for name, readings in selected.items()
        }
        batch = PredictionBatch(
            model_hash=self.artifact.content_hash,
            origin_tick=request.tick,
            target_tick=request.tick + 1,
            acquired_reading_count=len(history),
            acquired_readings_hash=readings_hash(history),
            predictions=predictions,
        )
        self._readings = history
        self._tick += 1
        return batch


def main() -> int:
    worker = None
    for line in sys.stdin:
        try:
            request = REQUEST.validate_json(line)
            if isinstance(request, StartRequest):
                if worker is not None:
                    raise ValueError("the worker model is immutable; start it only once")
                content = base64.b64decode(request.model_content_base64, validate=True)
                worker = ForecastWorker(ModelArtifact(content))
                reply = {"ok": True}
            elif worker is None:
                raise ValueError("start the worker before submitting an episode")
            elif isinstance(request, ResetRequest):
                worker.reset()
                reply = {"ok": True}
            else:
                reply = worker.step(request).model_dump(mode="json")
            print(json.dumps(reply, allow_nan=False), flush=True)
        except (ValueError, TypeError) as error:
            print(json.dumps({"error": str(error)}), flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
