"""Independent Beta–Bernoulli updates using explicit, labeled evidence only."""

from hashlib import sha256
from math import isclose
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, TypeAdapter, model_validator
from scipy.stats import beta as beta_distribution

from world_model.core import ModelArtifact
from world_model.core.serialization import canonical_json
from world_model.core.types import ContentHash, Contract, Identifier, NonNegativeInt, Probability, UTCDateTime

Shape = Annotated[float, Field(gt=0, le=1000000, allow_inf_nan=False)]


class BetaPrior(Contract):
    alpha: Shape = 1.0
    beta: Shape = 1.0


class CalibrationObservation(Contract):
    measurement_id: Identifier
    source_id: Identifier
    state: Identifier
    value: StrictBool
    observed_at: UTCDateTime
    available_at: UTCDateTime
    source_ref: Annotated[str, Field(min_length=1, max_length=1024)]

    @model_validator(mode="after")
    def valid_time(self) -> Self:
        if self.available_at < self.observed_at:
            raise ValueError("a calibration label cannot be available before its observation")
        return self


class BetaCell(Contract):
    source_id: Identifier
    state: Identifier
    alerts: NonNegativeInt
    non_alerts: NonNegativeInt
    alpha: Shape
    beta: Shape
    mean: Probability
    variance: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    lower_95: Probability
    upper_95: Probability


class CalibrationRevision(Contract):
    algorithm: Literal["beta-calibration-v1"] = "beta-calibration-v1"
    revision_id: ContentHash
    structure_hash: ContentHash
    as_of: UTCDateTime
    prior: BetaPrior
    evidence: tuple[CalibrationObservation, ...]
    cells: tuple[BetaCell, ...]


def validate_structure(artifact):
    if any(p != .5 for sensor in artifact.spec.observation_models.values() for p in sensor.alert_probability.values()):
        raise ValueError("learning requires a neutral structure with all sensor likelihoods masked to 0.5")


def fit_calibration(artifact: ModelArtifact, observations, *, as_of, prior=None) -> CalibrationRevision:
    """As-of revisions ignore unavailable labels; conflicting copies fail explicitly."""
    validate_structure(artifact)
    prior = BetaPrior() if prior is None else BetaPrior.model_validate(prior)
    cutoff = TypeAdapter(UTCDateTime).validate_python(as_of)
    unique, labels = {}, {}
    for item in observations:
        item = CalibrationObservation.model_validate(item)
        if item.available_at > cutoff:
            continue
        if item.source_id not in artifact.spec.observation_models or item.state not in artifact.spec.states:
            raise ValueError("calibration sensor and state must be declared in the structure")
        key = item.source_id, item.measurement_id
        if key in unique and unique[key] != item:
            raise ValueError("conflicting copies of a calibration measurement")
        if item.measurement_id in labels and labels[item.measurement_id] != item.state:
            raise ValueError("calibration sensors disagree about an example's verified label")
        unique[key], labels[item.measurement_id] = item, item.state
    evidence = tuple(sorted(unique.values(), key=lambda o: (o.available_at, o.measurement_id, o.source_id)))
    cells = []
    for source in sorted(artifact.spec.observation_models):
        for state in artifact.spec.states:
            selected = [o.value for o in evidence if o.source_id == source and o.state == state]
            alerts, non_alerts = sum(selected), len(selected) - sum(selected)
            alpha, beta = prior.alpha + alerts, prior.beta + non_alerts
            lower, upper = beta_distribution.ppf([.025, .975], alpha, beta)
            cells.append(BetaCell(source_id=source, state=state, alerts=alerts, non_alerts=non_alerts,
                alpha=alpha, beta=beta, mean=alpha / (alpha + beta),
                variance=alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1)),
                lower_95=float(lower), upper_95=float(upper)))
    body = {"algorithm": "beta-calibration-v1", "structure_hash": artifact.content_hash,
        "as_of": cutoff.isoformat(), "prior": prior.model_dump(mode="json"),
        "evidence": [o.model_dump(mode="json") for o in evidence]}
    # Identity is based on evidence and exact prior, independent of the numeric
    # library's inverse-CDF rounding used only for displayed intervals.
    return CalibrationRevision(revision_id=sha256(canonical_json(body)).hexdigest(),
        structure_hash=artifact.content_hash, as_of=cutoff, prior=prior, evidence=evidence, cells=tuple(cells))


def verify_revision(artifact, revision, *, allow_interval_roundoff=False):
    revision = CalibrationRevision.model_validate(revision)
    fitted = fit_calibration(artifact, revision.evidence, as_of=revision.as_of, prior=revision.prior)
    if fitted != revision:
        # Persistent fixed-mean models retain the original interval bytes. Tiny
        # inverse-CDF differences across SciPy releases must not break replay;
        # evidence, identity, counts, means, and variances still compare exactly.
        exclude_intervals = {"cells": {"__all__": {"lower_95", "upper_95"}}}
        same_parameters = allow_interval_roundoff and fitted.model_dump(exclude=exclude_intervals) == revision.model_dump(exclude=exclude_intervals)
        same_intervals = same_parameters and all(
            isclose(getattr(a, bound), getattr(b, bound), rel_tol=0, abs_tol=1e-12)
            for a, b in zip(fitted.cells, revision.cells, strict=True) for bound in ("lower_95", "upper_95"))
        if not same_intervals:
            raise ValueError("calibration revision does not match its evidence and structure")
    return revision


def mean_artifact(structure, revision):
    """An explicitly named point-estimate ablation, not the uncertain model."""
    verify_revision(structure, revision)
    raw = structure.spec.model_dump(mode="json")
    raw["model_id"] = "calibrated-" + structure.content_hash[:24]
    raw["model_version"] = revision.revision_id[:32]
    for cell in revision.cells:
        raw["observation_models"][cell.source_id]["alert_probability"][cell.state] = cell.mean
    return ModelArtifact(canonical_json(raw))
