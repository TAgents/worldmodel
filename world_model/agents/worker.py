"""Worker protocol has no environment seed, labels, future data, or adapter."""

import base64
import json
import sys
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from world_model.core import ModelArtifact, Reading
from world_model.core.types import Contract
from world_model.evaluation.protocol import StartRequest
from .contracts import InvestigationConfig
from .session import InvestigationSession


class Configure(Contract):
    operation: Literal["configure"] = "configure"
    config: InvestigationConfig


class Accept(Contract):
    operation: Literal["accept"] = "accept"
    reading: Reading | None = None


REQUEST = TypeAdapter(Annotated[StartRequest | Configure | Accept, Field(discriminator="operation")])


def main():
    artifact = session = None
    for line in sys.stdin:
        try:
            request = REQUEST.validate_json(line)
            if isinstance(request, StartRequest):
                if artifact is not None:
                    raise ValueError("model already configured")
                artifact = ModelArtifact(base64.b64decode(request.model_content_base64, validate=True))
                response = {"ok": True}
            elif artifact is None:
                raise ValueError("start with a model")
            elif isinstance(request, Configure):
                session = InvestigationSession(artifact, request.config)
                response = session.frame().model_dump(mode="json")
            elif session is None:
                raise ValueError("configure the episode")
            else:
                response = session.accept(request.reading).model_dump(mode="json")
            print(json.dumps(response, allow_nan=False), flush=True)
        except (TypeError, ValueError) as error:
            print(json.dumps({"error": str(error)}), flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
