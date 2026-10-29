"""Shared validated primitives with immutable nested collections."""

from collections.abc import Mapping
from datetime import UTC
from math import fsum, isclose
from types import MappingProxyType
from typing import Annotated, TypeVar

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    WrapSerializer,
)

K = TypeVar("K")
V = TypeVar("V")

# frozen=True alone does not protect dictionaries nested in a Pydantic model.
FrozenMap = Annotated[
    Mapping[K, V],
    AfterValidator(lambda value: MappingProxyType(dict(value))),
    WrapSerializer(lambda value, handler: handler(dict(value))),
]

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
ContentHash = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
UTCDateTime = Annotated[AwareDatetime, AfterValidator(lambda value: value.astimezone(UTC))]


def validate_distribution(value: Mapping[str, float]) -> Mapping[str, float]:
    if not value or not isclose(fsum(value.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("a probability distribution must be nonempty and sum to one")
    return value


Distribution = Annotated[
    FrozenMap[Identifier, Probability], AfterValidator(validate_distribution)
]


class Contract(BaseModel):
    """JSON-compatible contracts reject unknown fields and in-place mutation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
    )
