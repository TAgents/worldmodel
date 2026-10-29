"""HTTP contracts and a same-origin explorer; storage owns all inference and writes."""

from importlib.resources import files
from pathlib import Path
from typing import Annotated
import json

from pydantic import Field

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import psycopg
from starlette.middleware.trustedhost import TrustedHostMiddleware

from world_model.core import ModelArtifact, Observation, ObservationPayload, RunSpec
from world_model.adapters.github_actions import ActionsConfig, ImportPreview, MAX_BYTES
from world_model.agents.contracts import InvestigationConfig, InvestigationView, StepReceipt
from world_model.core.models import ModelRegistration
from world_model.core.types import ContentHash, Contract, Identifier, PositiveInt, UTCDateTime
from world_model.storage.contracts import (
    ConflictError, ForecastView, Inspection, ModelView, NotFoundError,
    ProjectionError, Publication, ReplayIntegrityError, ReplayReport, WriteResult,
)
from world_model.storage.store import PostgresStore
from world_model.learning.promotion import CalibrationView


class CreateRun(Contract):
    run_id: Identifier
    model_hash: ContentHash
    epoch: UTCDateTime
    target_state: Identifier


class AdvanceTick(Contract):
    tick: PositiveInt


class CalibratedRun(Contract):
    run_id: Identifier
    calibration_revision: ContentHash
    epoch: UTCDateTime
    target_state: Identifier


class InvestigationRequest(Contract):
    request_id: Identifier


class ActionsExport(Contract):
    model_hash: ContentHash
    export_json: Annotated[str, Field(min_length=1, max_length=MAX_BYTES)]
    config: ActionsConfig


class ActionsImport(ActionsExport):
    run_id: Identifier
    target_state: Identifier
    expected_plan_hash: ContentHash


Cutoff = Annotated[int | None, Query(ge=0)]
Limit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0)]


def create_app(store: PostgresStore, *, research_directory=None) -> FastAPI:
    from world_model.learning.catalog import ResearchCatalog
    research = ResearchCatalog(Path.cwd() / "artifacts" if research_directory is None else research_directory)
    app = FastAPI(
        title="World Model Engine", version="0.1.0", redoc_url=None,
        description="Local research API. Probabilities come from the registered finite-state model. "
        "Use /inspection to read a consistent evidence and belief snapshot. "
        "Historical cutoffs refer to when the engine learned information.",
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

    @app.middleware("http")
    async def local_browser_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        expected = f"{request.url.scheme}://{request.url.netloc}"
        if (origin and origin != expected) or request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Use the explorer and API from the same local origin."}, status_code=403)
        if request.method in {"POST", "PUT", "PATCH"} and request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            return JSONResponse({"detail": "An application/json request body is required."}, status_code=415)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path in {"/", "/learning", "/imports"} or request.url.path.startswith("/assets/"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
            )
        return response

    async def domain_error(request, error):
        if isinstance(error, NotFoundError):
            status, detail = 404, str(error)
        elif isinstance(error, ConflictError):
            status, detail = 409, str(error)
        elif isinstance(error, ProjectionError):
            status, detail = 503, str(error)
        elif isinstance(error, ReplayIntegrityError):
            status, detail = 500, "Stored history failed verification."
        elif isinstance(error, (psycopg.OperationalError, psycopg.InterfaceError)):
            status, detail = 503, "Database unavailable. Check the local database service."
        elif isinstance(error, psycopg.Error):
            status, detail = 500, "Database operation failed."
        else:
            status, detail = 422, str(error)
        return JSONResponse({"detail": detail}, status_code=status)

    for error_type in (NotFoundError, ConflictError, ProjectionError, ReplayIntegrityError, psycopg.Error, ValueError):
        app.add_exception_handler(error_type, domain_error)

    @app.get("/health")
    def health():
        store.list_models(limit=1)
        return {"status": "ok"}

    @app.get("/models")
    def models(limit: Limit = 100, offset: Offset = 0) -> tuple[ModelRegistration, ...]:
        return store.list_models(limit=limit, offset=offset)

    @app.get("/models/{content_hash}")
    def model(content_hash: ContentHash) -> ModelView:
        return store.get_model(content_hash)

    @app.get("/models/{content_hash}/calibration")
    def model_calibration(content_hash: ContentHash) -> CalibrationView:
        return store.get_calibration(content_hash)

    @app.get("/runs")
    def runs(limit: Limit = 100, offset: Offset = 0) -> tuple[RunSpec, ...]:
        return store.list_runs(limit=limit, offset=offset)

    def write_response(result, response):
        response.status_code = 201 if result.created else 200
        return result

    @app.post("/runs", status_code=201, responses={200: {"model": WriteResult}})
    def create_run(body: CreateRun, response: Response) -> WriteResult:
        return write_response(store.create_run(**body.model_dump()), response)

    @app.post("/runs/{run_id}/ticks", status_code=201, responses={200: {"model": WriteResult}})
    def advance(run_id: Identifier, body: AdvanceTick, response: Response) -> WriteResult:
        return write_response(store.advance_tick(run_id, body.tick), response)

    @app.post("/runs/{run_id}/observations", status_code=201, responses={200: {"model": WriteResult}})
    def ingest(run_id: Identifier, body: ObservationPayload, response: Response) -> WriteResult:
        return write_response(store.ingest(run_id, body), response)

    @app.get("/runs/{run_id}/observations/{observation_id}")
    def observation(run_id: Identifier, observation_id: Identifier) -> Observation:
        return store.get_observation(run_id, observation_id)

    @app.get("/runs/{run_id}/belief")
    def belief(run_id: Identifier, as_of: UTCDateTime | None = None, sequence_cutoff: Cutoff = None) -> Publication:
        return store.belief(run_id, as_of=as_of, sequence_cutoff=sequence_cutoff)

    @app.get("/runs/{run_id}/belief/history")
    def history(run_id: Identifier, as_of: UTCDateTime | None = None, sequence_cutoff: Cutoff = None) -> tuple[Publication, ...]:
        return store.history(run_id, as_of=as_of, sequence_cutoff=sequence_cutoff)

    @app.get("/runs/{run_id}/forecasts")
    def forecasts(run_id: Identifier, as_of: UTCDateTime | None = None, sequence_cutoff: Cutoff = None) -> tuple[ForecastView, ...]:
        return store.forecasts(run_id, as_of=as_of, sequence_cutoff=sequence_cutoff)

    @app.get("/runs/{run_id}/inspection")
    def inspection(run_id: Identifier, as_of: UTCDateTime | None = None, sequence_cutoff: Cutoff = None) -> Inspection:
        return store.inspect(run_id, as_of=as_of, sequence_cutoff=sequence_cutoff)

    @app.get("/runs/{run_id}/replay")
    def replay(run_id: Identifier, as_of: UTCDateTime | None = None, sequence_cutoff: Cutoff = None) -> ReplayReport:
        return store.replay(run_id, as_of=as_of, sequence_cutoff=sequence_cutoff)

    @app.post("/runs/{run_id}/investigation")
    def configure_investigation(run_id: Identifier, body: InvestigationConfig) -> InvestigationView:
        return store.configure_investigation(run_id, body)

    @app.get("/runs/{run_id}/investigation")
    def investigation(run_id: Identifier, as_of: UTCDateTime | None = None, sequence_cutoff: Cutoff = None) -> InvestigationView | None:
        return store.investigation(run_id, as_of=as_of, sequence_cutoff=sequence_cutoff)

    @app.post("/runs/{run_id}/investigation/steps", status_code=201, responses={200: {"model": StepReceipt}})
    def investigation_step(run_id: Identifier, body: InvestigationRequest, response: Response) -> StepReceipt:
        return write_response(store.step_investigation(run_id, body.request_id), response)

    static = files("world_model.api").joinpath("static")
    app.mount("/assets", StaticFiles(directory=str(static)), name="assets")

    @app.get("/", include_in_schema=False)
    def explorer():
        return FileResponse(str(static.joinpath("index.html")))

    @app.get("/research")
    def research_catalog():
        return research.list()

    @app.get("/research/{name}")
    def research_report(name: Identifier):
        return research.load(name)

    @app.post("/research/{name}/runs", status_code=201, responses={200: {"model": WriteResult}})
    def create_calibrated_run(name: Identifier, body: CalibratedRun, response: Response) -> WriteResult:
        structure, revision = research.calibration(name, body.calibration_revision)
        return write_response(store.create_calibrated_run(structure, revision,
            epoch=body.epoch, target_state=body.target_state, run_id=body.run_id), response)

    @app.post("/imports/github-actions/preview")
    def preview_actions(body: ActionsExport) -> ImportPreview:
        plan = store.preview_github_actions(body.model_hash, body.export_json.encode("utf-8"), body.config)
        return ImportPreview(plan=plan, plan_hash=plan.content_hash)

    @app.post("/imports/github-actions/runs", status_code=201, responses={200: {"model": WriteResult}})
    def import_actions(body: ActionsImport, response: Response) -> WriteResult:
        return write_response(store.import_github_actions(body.model_hash, body.export_json.encode("utf-8"), body.config,
            run_id=body.run_id, target_state=body.target_state, expected_plan_hash=body.expected_plan_hash), response)

    @app.get("/runs/{run_id}/import/source")
    def import_source(run_id: Identifier):
        return Response(store.get_import_source(run_id), media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="github-actions-source.json"'})

    @app.get("/imports/github-actions/example")
    def actions_example():
        fixtures = files("world_model.adapters").joinpath("fixtures")
        artifact = ModelArtifact(fixtures.joinpath("software-delivery-v1.json").read_bytes())
        return {"model_hash": artifact.content_hash,
            "export_json": fixtures.joinpath("github-actions-example.json").read_text(),
            "config": json.loads(fixtures.joinpath("github-actions-config.json").read_text()),
            "notice": "Illustrative fixture, not real repository data. Example model parameters are assumptions."}

    @app.get("/imports", include_in_schema=False)
    def imports_page():
        return FileResponse(str(static.joinpath("imports.html")))

    @app.get("/learning", include_in_schema=False)
    def calibration_lab():
        return FileResponse(str(static.joinpath("learning.html")))

    return app
