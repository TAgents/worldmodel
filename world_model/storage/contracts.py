"""Publication and replay results; database writes are exposed as one unit."""

from typing import Literal
from world_model.agents.contracts import InvestigationView
from world_model.adapters.github_actions import ImportView

from world_model.core import BeliefRevision, EvidenceSnapshot, Forecast, Observation, RunSpec
from world_model.core.models import ModelRegistration, ModelSpec
from world_model.core.types import ContentHash, Contract, Identifier, NonNegativeInt, UTCDateTime


class Publication(Contract):
    belief: BeliefRevision
    evidence: EvidenceSnapshot
    forecast: Forecast


class WriteResult(Contract):
    created: bool
    publication: Publication
    observation: Observation | None = None


class ForecastView(Contract):
    forecast: Forecast
    closed_at: UTCDateTime | None = None
    closed_event_sequence: NonNegativeInt | None = None


class ReplayReport(Contract):
    run_id: Identifier
    model_hash: ContentHash
    events_checked: NonNegativeInt
    publication: Publication
    closed_forecast_ids: tuple[Identifier, ...]
    investigation_steps_checked: NonNegativeInt = 0
    calibration_revision_checked: ContentHash | None = None
    imported_observations_checked: NonNegativeInt = 0
    import_plan_hash_checked: ContentHash | None = None


class ModelView(Contract):
    registration: ModelRegistration
    spec: ModelSpec


class RevisionView(Contract):
    belief: BeliefRevision
    event_kind: Literal["run_created", "tick_advanced", "observation_ingested"]


class Inspection(Contract):
    """One consistent read, bounded by the selected publication's sequence."""

    run: RunSpec
    model: ModelView
    publication: Publication
    observations: tuple[Observation, ...]
    history: tuple[RevisionView, ...]
    forecasts: tuple[ForecastView, ...]
    investigation: InvestigationView | None = None
    imported: ImportView | None = None


class NotFoundError(LookupError):
    pass


class ConflictError(ValueError):
    pass


class ProjectionError(RuntimeError):
    pass


class ReplayIntegrityError(RuntimeError):
    pass
