"""Deterministic, offline translation of GitHub Actions REST exports.

One reading means any observed failure in a closed workflow/time window.
No readings are invented for absent or inconclusive data. Source updated_at is
retained as a source timestamp, never used as the engine's ingestion clock.
"""

from datetime import timedelta
from hashlib import sha256
import json
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from world_model.core import ModelArtifact, ObservationPayload
from world_model.core.serialization import canonical_json
from world_model.core.types import ContentHash, Contract, FrozenMap, Identifier, NonNegativeInt, UTCDateTime

VERSION = "github-actions-windows-v1"
MAX_BYTES = 8 * 1024 * 1024
ProviderID = Annotated[int, Field(strict=True, gt=0, le=2**53 - 1)]
Repository = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", max_length=200)]


class ActionsConfig(Contract):
    repository: Repository
    branch: Annotated[str, Field(min_length=1, max_length=250)]
    event: Identifier = "push"
    window_start: UTCDateTime
    windows: Annotated[int, Field(strict=True, ge=1, le=200)]
    workflow_sources: FrozenMap[Identifier, ProviderID]

    @model_validator(mode="after")
    def unique_workflows(self) -> Self:
        values = list(self.workflow_sources.values())
        if not 1 <= len(values) <= 4 or len(values) != len(set(values)):
            raise ValueError("map one to four distinct workflow IDs to model sensors")
        return self


class RepositoryRecord(BaseModel):
    full_name: Repository


class WorkflowRun(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    id: ProviderID
    workflow_id: ProviderID
    run_attempt: ProviderID
    repository: RepositoryRecord
    head_branch: str | None
    head_sha: Annotated[str, Field(pattern=r"^[a-fA-F0-9]{40,64}$")]
    event: str
    status: str
    conclusion: str | None
    created_at: UTCDateTime
    updated_at: UTCDateTime

    @model_validator(mode="after")
    def timestamps(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("workflow update cannot precede creation")
        return self


class SourceRun(Contract):
    run_id: ProviderID
    attempt: ProviderID
    head_sha: str
    updated_at: UTCDateTime
    status: str
    conclusion: str | None
    url: str


class WindowReading(Contract):
    payload: ObservationPayload
    window_start: UTCDateTime
    window_end: UTCDateTime
    runs: tuple[SourceRun, ...]


class SkippedRun(Contract):
    run_id: ProviderID
    reason: str


class ImportPlan(Contract):
    adapter_version: Literal["github-actions-windows-v1"] = VERSION
    model_hash: ContentHash
    source_hash: ContentHash
    config: ActionsConfig
    epoch: UTCDateTime
    final_tick: NonNegativeInt
    readings: tuple[WindowReading, ...]
    skipped: tuple[SkippedRun, ...]
    unique_run_count: NonNegativeInt
    duplicate_count: NonNegativeInt
    incomplete_windows: tuple[str, ...]

    @property
    def content(self):
        return canonical_json(self.model_dump(mode="json"))

    @property
    def content_hash(self):
        return sha256(self.content).hexdigest()


class ObservationSource(Contract):
    observation_id: Identifier
    event_sequence: NonNegativeInt
    reading: WindowReading


class ImportView(Contract):
    adapter_version: Literal["github-actions-windows-v1"] = VERSION
    source_hash: ContentHash
    plan_hash: ContentHash
    repository: Repository
    branch: str
    imported_at: UTCDateTime
    observations: tuple[ObservationSource, ...]


class ImportPreview(Contract):
    plan: ImportPlan
    plan_hash: ContentHash


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate export JSON key: {key}")
        result[key] = value
    return result


def parse_export(content: bytes):
    if not content or len(content) > MAX_BYTES:
        raise ValueError("supply a nonempty GitHub export no larger than 8 MiB")
    def invalid_constant(value):
        raise ValueError("nonfinite JSON values are not supported")
    raw = json.loads(content, object_pairs_hook=_unique, parse_constant=invalid_constant)
    pages = raw if isinstance(raw, list) else [raw]
    if not 1 <= len(pages) <= 100:
        raise ValueError("supply one REST response or up to 100 paginated responses")
    records, digests, totals, duplicates = {}, {}, set(), 0
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("workflow_runs"), list) or type(page.get("total_count")) is not int:
            raise ValueError("expected GitHub REST total_count and workflow_runs, or an array of those pages")
        totals.add(page["total_count"])
        if not 0 <= page["total_count"] <= 10000:
            raise ValueError("exports support at most 10000 workflow runs")
        for raw_record in page["workflow_runs"]:
            record = WorkflowRun.model_validate(raw_record)
            digest = sha256(canonical_json(raw_record)).hexdigest()
            if record.id in records:
                if digests[record.id] != digest:
                    raise ValueError("conflicting snapshots of the same workflow run; export a stable snapshot")
                duplicates += 1
            records[record.id], digests[record.id] = record, digest
            if len(records) > 10000:
                raise ValueError("exports support at most 10000 workflow runs")
    if len(totals) != 1 or next(iter(totals)) != len(records):
        raise ValueError("incomplete or changing export: unique runs must match every page's total_count")
    return tuple(records[key] for key in sorted(records)), duplicates


def prepare_import(artifact: ModelArtifact, content: bytes, config: ActionsConfig) -> ImportPlan:
    config = ActionsConfig.model_validate(config)
    if not set(config.workflow_sources) <= set(artifact.spec.observation_models):
        raise ValueError("workflow mappings must name sensors in the registered model")
    records, duplicates = parse_export(content)
    duration = timedelta(seconds=artifact.spec.tick_duration_seconds)
    end = config.window_start + duration * config.windows
    by_workflow = {workflow: source for source, workflow in config.workflow_sources.items()}
    groups, skipped = {}, []
    for run in records:
        if run.repository.full_name.casefold() != config.repository.casefold():
            raise ValueError("export contains a different repository")
        reason = ("different branch" if run.head_branch != config.branch else
                  "different event" if run.event != config.event else
                  "unmapped workflow" if run.workflow_id not in by_workflow else
                  "outside requested windows" if not config.window_start <= run.updated_at < end else None)
        if reason:
            skipped.append(SkippedRun(run_id=run.id, reason=reason))
            continue
        tick = (run.updated_at - config.window_start) // duration
        source = by_workflow[run.workflow_id]
        groups.setdefault((tick, source), []).append(SourceRun(run_id=run.id, attempt=run.run_attempt,
            head_sha=run.head_sha, updated_at=run.updated_at, status=run.status, conclusion=run.conclusion,
            url=f"https://github.com/{config.repository}/actions/runs/{run.id}/attempts/{run.run_attempt}"))
    readings, incomplete = [], []
    for (tick, source), runs in sorted(groups.items()):
        failures = [r for r in runs if r.status == "completed" and r.conclusion in {"failure", "timed_out"}]
        unknown = [r for r in runs if r.status != "completed" or r.conclusion not in {"success", "failure", "timed_out"}]
        if unknown:
            incomplete.append(f"{source}:{tick}")
        if not failures and unknown:
            skipped.extend(SkippedRun(run_id=r.run_id, reason="inconclusive workflow window") for r in runs)
            continue
        start = config.window_start + tick * duration
        measurement_id = "gha-" + sha256(canonical_json([source, tick])).hexdigest()
        readings.append(WindowReading(payload=ObservationPayload(measurement_id=measurement_id,
            source_id=source, tick=tick, value=bool(failures)), window_start=start, window_end=start + duration,
            runs=tuple(sorted(runs, key=lambda r: r.run_id))))
    return ImportPlan(model_hash=artifact.content_hash, source_hash=sha256(content).hexdigest(), config=config,
        epoch=config.window_start + duration, final_tick=config.windows - 1, readings=tuple(readings),
        skipped=tuple(skipped), unique_run_count=len(records), duplicate_count=duplicates,
        incomplete_windows=tuple(incomplete))


def operations(plan):
    """An ordered event plan using only ordinary ticks and observation payloads."""
    by_tick = {}
    for reading in plan.readings:
        by_tick.setdefault(reading.payload.tick, []).append(reading)
    result = []
    for tick in range(plan.final_tick + 1):
        if tick:
            result.append(("tick_advanced", tick))
        result.extend(("observation_ingested", reading) for reading in by_tick.get(tick, ()))
    return tuple(result)
