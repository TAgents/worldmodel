"""The inference worker's complete input protocol deliberately excludes truth."""

from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from world_model.core import Reading
from world_model.core.serialization import canonical_json
from world_model.core.types import ContentHash, Contract, Distribution, FrozenMap, NonNegativeInt

METHODS = ("filter", "transition_only", "memoryless")
Method = Literal["filter", "transition_only", "memoryless"]


class StartRequest(Contract):
    operation: Literal["start"] = "start"
    model_content_base64: str


class ResetRequest(Contract):
    operation: Literal["reset"] = "reset"


class StepRequest(Contract):
    operation: Literal["step"] = "step"
    tick: NonNegativeInt
    readings: tuple[Reading, ...]

    @model_validator(mode="after")
    def validate_readings(self) -> Self:
        if any(reading.tick != self.tick for reading in self.readings):
            raise ValueError("step requests may contain only the current tick's readings")
        if len({reading.source_id for reading in self.readings}) != len(self.readings):
            raise ValueError("a step may contain at most one reading per sensor")
        return self


REQUEST = TypeAdapter(Annotated[StartRequest | ResetRequest | StepRequest, Field(discriminator="operation")])


def readings_hash(readings: tuple[Reading, ...] | list[Reading]) -> str:
    return sha256(canonical_json([reading.model_dump(mode="json") for reading in readings])).hexdigest()


class PredictionBatch(Contract):
    model_hash: ContentHash
    origin_tick: NonNegativeInt
    target_tick: NonNegativeInt
    acquired_reading_count: NonNegativeInt
    acquired_readings_hash: ContentHash
    predictions: FrozenMap[Method, Distribution]

    @model_validator(mode="after")
    def validate_batch(self) -> Self:
        if self.target_tick != self.origin_tick + 1:
            raise ValueError("predictions must be one tick ahead")
        if set(self.predictions) != set(METHODS):
            raise ValueError("each batch must include the filter and both fixed baselines")
        return self
