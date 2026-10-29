"""A fresh subprocess receives model bytes and causal readings, never a dataset."""

import base64
import json
import selectors
import subprocess
import sys
import tempfile

from world_model.core import ModelArtifact

from .protocol import PredictionBatch, ResetRequest, StartRequest, StepRequest


class PredictorProcess:
    worker_module = "world_model.evaluation.worker"

    def __init__(self, artifact: ModelArtifact, timeout_seconds: float = 30):
        self.artifact = artifact
        self.timeout_seconds = timeout_seconds
        self._process = None
        self._stderr = None

    def __enter__(self):
        self._stderr = tempfile.TemporaryFile(mode="w+t")
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", self.worker_module],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                text=True, encoding="utf-8", bufsize=1,
            )
            self._ack(StartRequest(model_content_base64=base64.b64encode(self.artifact.content).decode("ascii")))
        except BaseException:
            self.close()
            raise
        return self

    def _exchange(self, request):
        if self._process is None:
            raise RuntimeError("the predictor process is not running")
        try:
            self._process.stdin.write(request.model_dump_json() + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError("the predictor process closed its input") from error
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout_seconds):
                raise TimeoutError("the predictor process did not answer before its deadline")
        line = self._process.stdout.readline()
        if not line:
            self._stderr.seek(0)
            raise RuntimeError(f"predictor exited without a response: {self._stderr.read(4096)}")
        reply = json.loads(line)
        if not isinstance(reply, dict):
            raise RuntimeError("predictor response must be a JSON object")
        if "error" in reply:
            raise RuntimeError(f"predictor rejected the request: {reply['error']}")
        return reply

    def _ack(self, request):
        if self._exchange(request) != {"ok": True}:
            raise RuntimeError("unexpected predictor acknowledgement")

    def reset(self) -> None:
        self._ack(ResetRequest())

    def step(self, request: StepRequest) -> PredictionBatch:
        batch = PredictionBatch.model_validate(self._exchange(request))
        if batch.origin_tick != request.tick or batch.model_hash != self.artifact.content_hash:
            raise RuntimeError("predictor returned a different origin tick or model artifact")
        if any(set(distribution) != set(self.artifact.spec.states) for distribution in batch.predictions.values()):
            raise RuntimeError("predictor returned unexpected state labels")
        return batch

    def close(self) -> None:
        if self._process is not None:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            finally:
                self._process.stdout.close()
                self._process = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
