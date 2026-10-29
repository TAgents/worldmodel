"""Atomic PostgreSQL ingestion with immutable publications and deterministic replay."""

from collections.abc import Callable
from datetime import datetime, timedelta
from uuid import uuid4

from pydantic import TypeAdapter
from psycopg.types.json import Jsonb

from world_model.core import (
    BeliefRevision, EvidenceReference, EvidenceSnapshot, Forecast, ModelArtifact,
    ModelRegistration, Observation, ObservationPayload, Reading, RunSpec,
)
from world_model.core.types import ContentHash, Identifier, NonNegativeInt, PositiveInt, UTCDateTime
from world_model.inference import INFERENCE_VERSION, filter_state

from .contracts import (
    ConflictError, ForecastView, Inspection, ModelView, NotFoundError, ProjectionError, Publication,
    ReplayIntegrityError, ReplayReport, RevisionView, WriteResult,
)
from .database import connect
from .replay import ReplayState
from .investigations import InvestigationStoreMixin
from .calibrations import CalibrationStoreMixin
from .imports import ImportStoreMixin


def _id() -> str:
    return uuid4().hex


def _run(row) -> RunSpec:
    return RunSpec(**{key: row[key] for key in ("run_id", "epoch", "created_at", "model_hash")})


def _observation(row) -> Observation:
    return Observation(
        observation_id=row["observation_id"], run_id=row["run_id"],
        payload=ObservationPayload(
            tick=row["observed_tick"], source_id=row["source_id"], value=row["value"],
            measurement_id=row["measurement_id"], supersedes_id=row["supersedes_id"],
        ),
        observed_at=row["observed_at"], ingested_at=row["ingested_at"], event_sequence=row["event_sequence"],
    )


def _cutoff(time_column, sequence_column, as_of, sequence_cutoff):
    clauses, values = [], []
    if as_of is not None:
        clauses.append(f"{time_column} <= %s")
        values.append(TypeAdapter(UTCDateTime).validate_python(as_of))
    if sequence_cutoff is not None:
        clauses.append(f"{sequence_column} <= %s")
        values.append(TypeAdapter(NonNegativeInt).validate_python(sequence_cutoff))
    return (" AND " + " AND ".join(clauses) if clauses else ""), values


class PostgresStore(ImportStoreMixin, CalibrationStoreMixin, InvestigationStoreMixin):
    def __init__(self, dsn: str, *, schema: str = "world_model", clock: Callable[[], datetime] | None = None):
        self._dsn, self.schema, self._clock = dsn, schema, clock

    def _connect(self):
        return connect(self._dsn, self.schema)

    def _now(self, conn, after=None):
        raw = self._clock() if self._clock else conn.execute("SELECT clock_timestamp() AS now").fetchone()["now"]
        timestamp = TypeAdapter(UTCDateTime).validate_python(raw)
        return max(timestamp, after + timedelta(microseconds=1)) if after else timestamp

    @staticmethod
    def _read_transaction(conn):
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")

    def register_model(self, artifact: ModelArtifact) -> ModelRegistration:
        if artifact.spec.derivation is not None:
            raise ValueError("use calibrated registration to preserve calibration evidence")
        with self._connect() as conn:
            return self._register_model(conn, artifact, self._now(conn))

    @staticmethod
    def _register_model(conn, artifact, timestamp):
        conn.execute(
            """INSERT INTO models (content_hash, model_id, model_version, content, registered_at)
               VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
            (artifact.content_hash, artifact.spec.model_id, artifact.spec.model_version, artifact.content, timestamp),
        )
        row = conn.execute(
            "SELECT content_hash, model_id, model_version, registered_at FROM models WHERE model_id = %s AND model_version = %s",
            (artifact.spec.model_id, artifact.spec.model_version),
        ).fetchone()
        if row is None or row["content_hash"] != artifact.content_hash:
            raise ConflictError("a model version cannot be registered with different artifact bytes")
        return ModelRegistration.model_validate(row)

    def _model(self, conn, content_hash):
        row = conn.execute("SELECT * FROM models WHERE content_hash = %s", (content_hash,)).fetchone()
        if row is None:
            raise NotFoundError("model artifact not found")
        artifact = ModelArtifact(bytes(row["content"]))
        if artifact.content_hash != content_hash or (artifact.spec.model_id, artifact.spec.model_version) != (row["model_id"], row["model_version"]):
            raise ReplayIntegrityError("stored model does not match its registration")
        self._calibration(conn, artifact, row["registered_at"])
        return artifact, row["registered_at"]

    @staticmethod
    def _run_row(conn, run_id, *, lock=False):
        row = conn.execute("SELECT * FROM runs WHERE run_id = %s" + (" FOR UPDATE" if lock else ""), (run_id,)).fetchone()
        if row is None:
            raise NotFoundError("run not found")
        return row

    @staticmethod
    def _head(conn, run_id):
        head = conn.execute("SELECT * FROM run_heads WHERE run_id = %s", (run_id,)).fetchone()
        latest = conn.execute("SELECT max(sequence) AS sequence FROM events WHERE run_id = %s", (run_id,)).fetchone()["sequence"]
        if head is None or head["sequence"] != latest:
            raise ProjectionError("current-state projection is missing or stale; rebuild it from history")
        return head

    @staticmethod
    def _event(conn, run_id, sequence, kind, timestamp, payload):
        conn.execute(
            "INSERT INTO events (run_id, sequence, kind, recorded_at, payload) VALUES (%s, %s, %s, %s, %s)",
            (run_id, sequence, kind, timestamp, Jsonb(payload)),
        )

    @staticmethod
    def _active(conn, run_id, sequence):
        rows = conn.execute("""
            SELECT o.* FROM observations o
            WHERE o.run_id = %s AND o.event_sequence <= %s
              AND NOT EXISTS (
                SELECT 1 FROM observations c
                WHERE c.run_id = o.run_id AND c.supersedes_id = o.observation_id AND c.event_sequence <= %s
              )
            ORDER BY o.event_sequence
        """, (run_id, sequence, sequence)).fetchall()
        return tuple(_observation(row) for row in rows)

    def _publish(self, conn, run_row, artifact, tick, sequence, timestamp, previous_id):
        run_id = run_row["run_id"]
        observations = self._active(conn, run_id, sequence)
        result = filter_state(artifact, [Reading(
            tick=item.payload.tick, source_id=item.payload.source_id, value=item.payload.value
        ) for item in observations], state_tick=tick)
        snapshot = EvidenceSnapshot(
            snapshot_id=_id(), run_id=run_id, sequence_cutoff=sequence,
            observations=tuple(EvidenceReference(observation_id=item.observation_id, event_sequence=item.event_sequence) for item in observations),
        )
        belief = BeliefRevision(
            revision_id=_id(), run_id=run_id, variable=artifact.spec.variable, state_tick=tick,
            published_at=timestamp, sequence_cutoff=sequence, distribution=result.posterior,
            model_hash=artifact.content_hash, evidence_snapshot_id=snapshot.snapshot_id,
            inference_version=INFERENCE_VERSION, previous_revision_id=previous_id,
        )
        forecast = Forecast(
            forecast_id=_id(), run_id=run_id, origin_tick=tick, target_tick=tick + 1,
            target_state=run_row["target_state"], probability=result.predictive[run_row["target_state"]],
            issued_at=timestamp, belief_revision_id=belief.revision_id,
            model_hash=artifact.content_hash, evidence_snapshot_id=snapshot.snapshot_id,
        )
        conn.execute("INSERT INTO snapshots (run_id, snapshot_id, sequence_cutoff) VALUES (%s, %s, %s)", (run_id, snapshot.snapshot_id, sequence))
        for reference in snapshot.observations:
            conn.execute("INSERT INTO snapshot_members (run_id, snapshot_id, observation_id, event_sequence) VALUES (%s, %s, %s, %s)", (run_id, snapshot.snapshot_id, reference.observation_id, reference.event_sequence))
        conn.execute("""INSERT INTO revisions (
            run_id, revision_id, variable, state_tick, published_at, sequence_cutoff,
            distribution, model_hash, evidence_snapshot_id, inference_version, previous_revision_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", (
            run_id, belief.revision_id, belief.variable, tick, timestamp, sequence,
            Jsonb(dict(belief.distribution)), artifact.content_hash, snapshot.snapshot_id, INFERENCE_VERSION, previous_id,
        ))
        conn.execute("""INSERT INTO forecasts (
            run_id, forecast_id, origin_tick, target_tick, target_state, probability, issued_at,
            belief_revision_id, model_hash, evidence_snapshot_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", (
            run_id, forecast.forecast_id, tick, tick + 1, forecast.target_state, forecast.probability,
            timestamp, belief.revision_id, artifact.content_hash, snapshot.snapshot_id,
        ))
        self._set_head(conn, belief)
        return Publication(belief=belief, evidence=snapshot, forecast=forecast)

    @staticmethod
    def _set_head(conn, belief):
        conn.execute("""INSERT INTO run_heads (run_id, state_tick, sequence, revision_id) VALUES (%s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET state_tick = EXCLUDED.state_tick,
            sequence = EXCLUDED.sequence, revision_id = EXCLUDED.revision_id""",
            (belief.run_id, belief.state_tick, belief.sequence_cutoff, belief.revision_id))

    @staticmethod
    def _publication(conn, run_id, sequence):
        row = conn.execute("SELECT * FROM revisions WHERE run_id = %s AND sequence_cutoff = %s", (run_id, sequence)).fetchone()
        if row is None:
            raise NotFoundError("belief revision not found")
        belief = BeliefRevision.model_validate(row)
        snapshot_row = conn.execute("SELECT * FROM snapshots WHERE run_id = %s AND snapshot_id = %s", (run_id, belief.evidence_snapshot_id)).fetchone()
        members = conn.execute("SELECT observation_id, event_sequence FROM snapshot_members WHERE run_id = %s AND snapshot_id = %s ORDER BY event_sequence", (run_id, belief.evidence_snapshot_id)).fetchall()
        snapshot = EvidenceSnapshot(**snapshot_row, observations=members)
        forecast_row = conn.execute("SELECT * FROM forecasts WHERE run_id = %s AND belief_revision_id = %s", (run_id, belief.revision_id)).fetchone()
        if forecast_row is None:
            raise ReplayIntegrityError("belief revision has no forecast")
        return Publication(belief=belief, evidence=snapshot, forecast=Forecast.model_validate(forecast_row))

    def create_run(self, model_hash: str, *, epoch: datetime, target_state: str, run_id: str | None = None) -> WriteResult:
        with self._connect() as conn:
            return self._create_run(conn, model_hash, epoch=epoch, target_state=target_state, run_id=run_id)

    def _create_run(self, conn, model_hash, *, epoch, target_state, run_id=None, import_plan_hash=None):
        model_hash = TypeAdapter(ContentHash).validate_python(model_hash)
        epoch = TypeAdapter(UTCDateTime).validate_python(epoch)
        run_id = TypeAdapter(Identifier).validate_python(_id() if run_id is None else run_id)
        artifact, registered_at = self._model(conn, model_hash)
        timestamp = self._now(conn)
        if target_state not in artifact.spec.states:
            raise ValueError("forecast target is not a state in the model")
        if epoch > timestamp or registered_at > timestamp:
            raise ValueError("run epoch and model registration must already be available")
        inserted = conn.execute("""INSERT INTO runs (run_id, model_hash, target_state, epoch, created_at)
            VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING *""",
            (run_id, model_hash, target_state, epoch, timestamp)).fetchone()
        if inserted is None:
            existing = self._run_row(conn, run_id)
            if (existing["model_hash"], existing["target_state"], existing["epoch"]) != (model_hash, target_state, epoch):
                raise ConflictError("run ID is already associated with different settings")
            return WriteResult(created=False, publication=self._publication(conn, run_id, 0))
        run = _run(inserted)
        payload = {"run": run.model_dump(mode="json"), "target_state": target_state}
        if import_plan_hash is not None:
            payload["import_plan_hash"] = TypeAdapter(ContentHash).validate_python(import_plan_hash)
        self._event(conn, run_id, 0, "run_created", timestamp, payload)
        return WriteResult(created=True, publication=self._publish(conn, inserted, artifact, 0, 0, timestamp, None))

    def ingest(self, run_id: str, payload: ObservationPayload) -> WriteResult:
        payload = ObservationPayload.model_validate(payload)
        with self._connect() as conn:
            run = self._run_row(conn, run_id, lock=True)
            self._require_manual_run(conn, run_id)
            return self._ingest_locked(conn, run, payload)

    def _ingest_locked(self, conn, run, payload, *, after=None):
        run_id = run["run_id"]
        head = self._head(conn, run_id)
        existing = conn.execute("SELECT * FROM observations WHERE run_id = %s AND source_id = %s AND measurement_id = %s", (run_id, payload.source_id, payload.measurement_id)).fetchone()
        if existing is not None:
            observation = _observation(existing)
            if observation.payload != payload:
                raise ConflictError("measurement identity already has a different payload")
            return WriteResult(created=False, observation=observation, publication=self._publication(conn, run_id, observation.event_sequence))
        artifact, _ = self._model(conn, run["model_hash"])
        if payload.source_id not in artifact.spec.observation_models:
            raise ValueError("unknown observation source")
        if payload.tick > head["state_tick"]:
            raise ValueError("advance the run before ingesting a future-tick observation")
        active = next((item for item in self._active(conn, run_id, head["sequence"]) if (item.payload.source_id, item.payload.tick) == (payload.source_id, payload.tick)), None)
        if payload.supersedes_id != (active.observation_id if active else None):
            raise ConflictError("correction must reference the active observation in the same run, source, and tick")
        previous = self._publication(conn, run_id, head["sequence"])
        timestamp = self._now(conn, after=max(previous.belief.published_at, after) if after else previous.belief.published_at)
        sequence = head["sequence"] + 1
        observation = Observation(
            observation_id=_id(), run_id=run_id, payload=payload,
            observed_at=run["epoch"] + timedelta(seconds=payload.tick * artifact.spec.tick_duration_seconds),
            ingested_at=timestamp, event_sequence=sequence,
        )
        self._event(conn, run_id, sequence, "observation_ingested", timestamp, observation.model_dump(mode="json"))
        conn.execute("""INSERT INTO observations (
            run_id, observation_id, source_id, measurement_id, observed_tick, value,
            observed_at, ingested_at, event_sequence, supersedes_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", (
            run_id, observation.observation_id, payload.source_id, payload.measurement_id, payload.tick, payload.value,
            observation.observed_at, timestamp, sequence, payload.supersedes_id,
        ))
        publication = self._publish(conn, run, artifact, head["state_tick"], sequence, timestamp, head["revision_id"])
        return WriteResult(created=True, observation=observation, publication=publication)

    def advance_tick(self, run_id: str, tick: int) -> WriteResult:
        tick = TypeAdapter(PositiveInt).validate_python(tick)
        with self._connect() as conn:
            run = self._run_row(conn, run_id, lock=True)
            self._require_manual_run(conn, run_id)
            return self._advance_locked(conn, run, tick)

    def _advance_locked(self, conn, run, tick, *, after=None):
        run_id = run["run_id"]
        head = self._head(conn, run_id)
        if tick <= head["state_tick"]:
            existing = conn.execute("SELECT sequence FROM events WHERE run_id = %s AND kind = 'tick_advanced' AND payload @> %s", (run_id, Jsonb({"tick": tick}))).fetchone()
            if existing is None:
                raise ReplayIntegrityError("advanced tick has no clock event")
            return WriteResult(created=False, publication=self._publication(conn, run_id, existing["sequence"]))
        if tick != head["state_tick"] + 1:
            raise ValueError("advance exactly one tick at a time")
        artifact, _ = self._model(conn, run["model_hash"])
        previous = self._publication(conn, run_id, head["sequence"])
        timestamp = self._now(conn, after=max(previous.belief.published_at, after) if after else previous.belief.published_at)
        if timestamp < run["epoch"] + timedelta(seconds=tick * artifact.spec.tick_duration_seconds):
            raise ValueError("the environment tick is ahead of the engine's knowledge clock")
        sequence = head["sequence"] + 1
        self._event(conn, run_id, sequence, "tick_advanced", timestamp, {"tick": tick, "frozen_forecast_id": previous.forecast.forecast_id})
        conn.execute("INSERT INTO forecast_closures (run_id, origin_tick, forecast_id, closed_event_sequence) VALUES (%s, %s, %s, %s)", (run_id, head["state_tick"], previous.forecast.forecast_id, sequence))
        publication = self._publish(conn, run, artifact, tick, sequence, timestamp, head["revision_id"])
        return WriteResult(created=True, publication=publication)

    def get_observation(self, run_id: str, observation_id: str) -> Observation:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM observations WHERE run_id = %s AND observation_id = %s", (run_id, observation_id)).fetchone()
            if row is None:
                raise NotFoundError("observation not found in this run")
            return _observation(row)

    def list_runs(self, *, limit: int = 100, offset: int = 0) -> tuple[RunSpec, ...]:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("limit must be 1..100 and offset must be nonnegative")
        with self._connect() as conn:
            return tuple(_run(row) for row in conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, run_id LIMIT %s OFFSET %s", (limit, offset),
            ))

    def list_models(self, *, limit: int = 100, offset: int = 0) -> tuple[ModelRegistration, ...]:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("limit must be 1..100 and offset must be nonnegative")
        with self._connect() as conn:
            return tuple(ModelRegistration.model_validate(row) for row in conn.execute(
                "SELECT content_hash, model_id, model_version, registered_at FROM models ORDER BY registered_at DESC, content_hash LIMIT %s OFFSET %s", (limit, offset),
            ))

    @staticmethod
    def _model_view(artifact, registered_at):
        return ModelView(spec=artifact.spec, registration=ModelRegistration(
            content_hash=artifact.content_hash, model_id=artifact.spec.model_id,
            model_version=artifact.spec.model_version, registered_at=registered_at,
        ))

    def get_model(self, content_hash: str) -> ModelView:
        with self._connect() as conn:
            self._read_transaction(conn)
            return self._model_view(*self._model(conn, content_hash))

    def inspect(self, run_id: str, *, as_of=None, sequence_cutoff=None) -> Inspection:
        condition, params = _cutoff("published_at", "sequence_cutoff", as_of, sequence_cutoff)
        with self._connect() as conn:
            self._read_transaction(conn)
            run_row = self._run_row(conn, run_id)
            run = _run(run_row)
            row = conn.execute("SELECT sequence_cutoff FROM revisions WHERE run_id = %s" + condition + " ORDER BY sequence_cutoff DESC LIMIT 1", (run_id, *params)).fetchone()
            if row is None:
                raise NotFoundError("no belief was published by the requested cutoff")
            sequence = row["sequence_cutoff"]
            publication = self._publication(conn, run_id, sequence)
            observations = tuple(_observation(item) for item in conn.execute("""
                SELECT o.* FROM observations o JOIN snapshot_members m
                  ON m.run_id = o.run_id AND m.observation_id = o.observation_id
                  AND m.event_sequence = o.event_sequence
                WHERE m.run_id = %s AND m.snapshot_id = %s ORDER BY o.event_sequence
            """, (run_id, publication.evidence.snapshot_id)))
            if tuple(EvidenceReference(observation_id=o.observation_id, event_sequence=o.event_sequence) for o in observations) != publication.evidence.observations:
                raise ReplayIntegrityError("snapshot evidence is incomplete")
            history = []
            for item in conn.execute("""
                SELECT r.*, e.kind AS event_kind FROM revisions r JOIN events e
                  ON e.run_id = r.run_id AND e.sequence = r.sequence_cutoff
                WHERE r.run_id = %s AND r.sequence_cutoff <= %s ORDER BY r.sequence_cutoff
            """, (run_id, sequence)):
                kind = item.pop("event_kind")
                history.append(RevisionView(belief=BeliefRevision.model_validate(item), event_kind=kind))
            artifact, registered_at = self._model(conn, run.model_hash)
            return Inspection(
                run=run, model=self._model_view(artifact, registered_at),
                publication=publication, observations=observations, history=history,
                forecasts=self._forecasts(conn, run_id, sequence_cutoff=sequence),
                investigation=self._inspection_investigation(conn, run_id, publication,
                    historical=as_of is not None or sequence_cutoff is not None),
                imported=self._import_view(conn, artifact, run_row, sequence),
            )

    def history(self, run_id: str, *, as_of=None, sequence_cutoff=None) -> tuple[Publication, ...]:
        condition, params = _cutoff("published_at", "sequence_cutoff", as_of, sequence_cutoff)
        with self._connect() as conn:
            self._read_transaction(conn)
            self._run_row(conn, run_id)
            rows = conn.execute("SELECT sequence_cutoff FROM revisions WHERE run_id = %s" + condition + " ORDER BY sequence_cutoff", (run_id, *params)).fetchall()
            return tuple(self._publication(conn, run_id, row["sequence_cutoff"]) for row in rows)

    def belief(self, run_id: str, *, as_of=None, sequence_cutoff=None) -> Publication:
        condition, params = _cutoff("published_at", "sequence_cutoff", as_of, sequence_cutoff)
        with self._connect() as conn:
            self._read_transaction(conn)
            self._run_row(conn, run_id)
            row = conn.execute("SELECT sequence_cutoff FROM revisions WHERE run_id = %s" + condition + " ORDER BY sequence_cutoff DESC LIMIT 1", (run_id, *params)).fetchone()
            if row is None:
                raise NotFoundError("no belief was published by the requested cutoff")
            return self._publication(conn, run_id, row["sequence_cutoff"])

    def forecasts(self, run_id: str, *, as_of=None, sequence_cutoff=None) -> tuple[ForecastView, ...]:
        with self._connect() as conn:
            self._read_transaction(conn)
            self._run_row(conn, run_id)
            return self._forecasts(conn, run_id, as_of, sequence_cutoff)

    @staticmethod
    def _forecasts(conn, run_id, as_of=None, sequence_cutoff=None):
        condition, params = _cutoff("r.published_at", "r.sequence_cutoff", as_of, sequence_cutoff)
        closure_condition, closure_params = _cutoff("e.recorded_at", "e.sequence", as_of, sequence_cutoff)
        rows = conn.execute("SELECT f.* FROM forecasts f JOIN revisions r ON r.run_id = f.run_id AND r.revision_id = f.belief_revision_id WHERE f.run_id = %s" + condition + " ORDER BY r.sequence_cutoff", (run_id, *params)).fetchall()
        closures = conn.execute("SELECT c.forecast_id, c.closed_event_sequence, e.recorded_at FROM forecast_closures c JOIN events e ON e.run_id = c.run_id AND e.sequence = c.closed_event_sequence WHERE c.run_id = %s" + closure_condition, (run_id, *closure_params)).fetchall()
        closed = {item["forecast_id"]: item for item in closures}
        return tuple(ForecastView(
            forecast=Forecast.model_validate(row), closed_at=closed[row["forecast_id"]]["recorded_at"] if row["forecast_id"] in closed else None,
            closed_event_sequence=closed[row["forecast_id"]]["closed_event_sequence"] if row["forecast_id"] in closed else None,
        ) for row in rows)

    def _replay(self, conn, run_id, as_of=None, sequence_cutoff=None):
        row = self._run_row(conn, run_id)
        run = _run(row)
        artifact, registered_at = self._model(conn, run.model_hash)
        if registered_at > run.created_at:
            raise ReplayIntegrityError("model was not registered when the run began")
        condition, params = _cutoff("recorded_at", "sequence", as_of, sequence_cutoff)
        events = conn.execute("SELECT * FROM events WHERE run_id = %s" + condition + " ORDER BY sequence", (run_id, *params)).fetchall()
        if not events:
            raise NotFoundError("no events available by the requested cutoff")
        state = ReplayState(artifact, run, row["target_state"])
        for event in events:
            if event["kind"] == "observation_ingested":
                expected = Observation.model_validate(event["payload"])
                stored = conn.execute("SELECT * FROM observations WHERE run_id = %s AND observation_id = %s", (run_id, expected.observation_id)).fetchone()
                if stored is None or _observation(stored) != expected:
                    raise ReplayIntegrityError("observation record diverges from its source event")
            state.apply(event, self._publication(conn, run_id, event["sequence"]))
        closures = conn.execute("SELECT forecast_id FROM forecast_closures WHERE run_id = %s AND closed_event_sequence <= %s ORDER BY closed_event_sequence", (run_id, state.sequence)).fetchall()
        if [item["forecast_id"] for item in closures] != state.closed_forecasts:
            raise ReplayIntegrityError("frozen forecasts diverge from the clock events")
        steps_checked = self._verify_investigation(conn, artifact, run_id, state.publication,
            historical=as_of is not None or sequence_cutoff is not None)
        imported_checked, import_hash = self._verify_import(conn, artifact, row, state.publication)
        return ReplayReport(run_id=run_id, model_hash=artifact.content_hash, events_checked=len(events), publication=state.publication,
            closed_forecast_ids=tuple(state.closed_forecasts), investigation_steps_checked=steps_checked,
            calibration_revision_checked=artifact.spec.derivation.calibration_revision if artifact.spec.derivation else None,
            imported_observations_checked=imported_checked, import_plan_hash_checked=import_hash)

    def replay(self, run_id: str, *, as_of=None, sequence_cutoff=None) -> ReplayReport:
        with self._connect() as conn:
            self._read_transaction(conn)
            return self._replay(conn, run_id, as_of, sequence_cutoff)

    def rebuild_projection(self, run_id: str) -> ReplayReport:
        with self._connect() as conn:
            self._run_row(conn, run_id, lock=True)
            report = self._replay(conn, run_id)
            self._set_head(conn, report.publication.belief)
            return report
