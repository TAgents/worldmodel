"""Calibration labels are explicit training data; episode labels are not inputs."""

import base64
import json
import sys
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from world_model.agents.contracts import InvestigationConfig
from world_model.agents.worker import Accept
from world_model.core import ModelArtifact
from world_model.core.types import Contract, NonNegativeInt
from world_model.evaluation.protocol import StartRequest
from .calibration import CalibrationRevision, validate_structure
from .session import LearnedSession


class Configure(Contract):
    operation: Literal["configure"] = "configure"
    revision: CalibrationRevision
    config: InvestigationConfig
    particles: Annotated[int, Field(strict=True, ge=2, le=4096)] = 256
    particle_seed: NonNegativeInt = 20261028


REQUEST = TypeAdapter(Annotated[StartRequest | Configure | Accept, Field(discriminator="operation")])


def main():
    artifact = session = None
    for line in sys.stdin:
        try:
            request = REQUEST.validate_json(line)
            if isinstance(request, StartRequest):
                if artifact is not None:
                    raise ValueError("structure is immutable")
                artifact = ModelArtifact(base64.b64decode(request.model_content_base64, validate=True))
                validate_structure(artifact)
                response = {"ok": True}
            elif artifact is None:
                raise ValueError("supply the neutral model structure first")
            elif isinstance(request, Configure):
                session = LearnedSession(artifact, request.revision, request.config, particles=request.particles, seed=request.particle_seed)
                response = session.frame().model_dump(mode="json")
            elif session is None:
                raise ValueError("configure the calibrated episode")
            else:
                response = session.accept(request.reading).model_dump(mode="json")
            print(json.dumps(response, allow_nan=False), flush=True)
        except (ValueError, TypeError) as error:
            print(json.dumps({"error": str(error)}), flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
