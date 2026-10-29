"""Atomic offline imports using the ordinary run, clock, and evidence pipeline."""

from datetime import timedelta
from hashlib import sha256

from pydantic import TypeAdapter

from world_model.adapters.github_actions import (
    VERSION, ImportPlan, ImportView, ObservationSource, operations, prepare_import,
)
from world_model.core import Observation
from world_model.core.types import Identifier
from .contracts import ConflictError, NotFoundError, ReplayIntegrityError, WriteResult


class ImportStoreMixin:
    def preview_github_actions(self, model_hash, content, config):
        with self._connect() as conn:
            self._read_transaction(conn)
            artifact, _ = self._model(conn, model_hash)
            plan = prepare_import(artifact, content, config)
            if plan.epoch + timedelta(seconds=plan.final_tick * artifact.spec.tick_duration_seconds) > self._now(conn):
                raise ValueError("all imported time windows must be closed before the import")
            return plan

    def import_github_actions(self, artifact, content, config, *, run_id, target_state, expected_plan_hash=None):
        run_id = TypeAdapter(Identifier).validate_python(run_id)
        if isinstance(artifact, str):
            with self._connect() as conn:
                self._read_transaction(conn)
                artifact, _ = self._model(conn, artifact)
        plan = prepare_import(artifact, content, config)
        if expected_plan_hash is not None and plan.content_hash != expected_plan_hash:
            raise ConflictError("import preview changed; preview the current export and settings again")
        with self._connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"world_model:import:{self.schema}:{run_id}",))
            timestamp = self._now(conn)
            if plan.epoch + timedelta(seconds=plan.final_tick * artifact.spec.tick_duration_seconds) > timestamp:
                raise ValueError("all imported time windows must be closed before the import")
            existing = conn.execute("SELECT * FROM run_imports WHERE run_id = %s", (run_id,)).fetchone()
            if existing:
                run = self._run_row(conn, run_id)
                if existing["plan_hash"] != plan.content_hash or run["target_state"] != target_state:
                    raise ConflictError("run ID is already associated with a different import")
                self._model(conn, artifact.content_hash)
                self._read_import(conn, artifact, run)
                return WriteResult(created=False, publication=self._publication(conn, run_id, existing["completed_sequence"]))
            if artifact.spec.derivation is None:
                self._register_model(conn, artifact, timestamp)
            else:
                # Calibrated models must already have their full calibration in storage.
                self._model(conn, artifact.content_hash)
            created = self._create_run(conn, artifact.content_hash, epoch=plan.epoch,
                target_state=target_state, run_id=run_id, import_plan_hash=plan.content_hash)
            if not created.created:
                raise ConflictError("an import requires a new run ID")
            run = self._run_row(conn, run_id, lock=True)
            result = created
            for kind, item in operations(plan):
                result = self._advance_locked(conn, run, item) if kind == "tick_advanced" else self._ingest_locked(conn, run, item.payload)
            conn.execute("""INSERT INTO run_imports (run_id, adapter_version, source_hash, plan_hash,
                source_content, plan_content, completed_sequence) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (run_id, VERSION, plan.source_hash, plan.content_hash, content, plan.content, result.publication.belief.sequence_cutoff))
            return WriteResult(created=True, publication=result.publication)

    @staticmethod
    def _require_not_imported(conn, run_id):
        if conn.execute("SELECT 1 FROM run_imports WHERE run_id = %s", (run_id,)).fetchone():
            raise ConflictError("this run preserves an imported snapshot; import a new snapshot into a new run")

    @staticmethod
    def _read_import(conn, artifact, run):
        row = conn.execute("SELECT * FROM run_imports WHERE run_id = %s", (run["run_id"],)).fetchone()
        event = conn.execute("SELECT payload FROM events WHERE run_id = %s AND sequence = 0", (run["run_id"],)).fetchone()
        marker = event["payload"].get("import_plan_hash") if event else None
        if row is None:
            if marker is not None:
                raise ReplayIntegrityError("imported run is missing its source archive")
            return None
        try:
            if row["adapter_version"] != VERSION or marker != row["plan_hash"]:
                raise ValueError("import registration does not match the creation event")
            content, plan_content = bytes(row["source_content"]), bytes(row["plan_content"])
            if sha256(content).hexdigest() != row["source_hash"] or sha256(plan_content).hexdigest() != row["plan_hash"]:
                raise ValueError("import content hash mismatch")
            saved = ImportPlan.model_validate_json(plan_content)
            plan = prepare_import(artifact, content, saved.config)
            if plan.content != plan_content or row["completed_sequence"] != len(operations(plan)):
                raise ValueError("import translation changed")
            if run["epoch"] != plan.epoch or run["model_hash"] != plan.model_hash:
                raise ValueError("import run uses a different clock or model")
            if plan.epoch + timedelta(seconds=plan.final_tick * artifact.spec.tick_duration_seconds) > run["created_at"]:
                raise ValueError("import included a future window")
            return row, plan
        except (ValueError, TypeError, KeyError) as error:
            raise ReplayIntegrityError("stored import failed provenance verification") from error

    def _import_view(self, conn, artifact, run, sequence):
        data = self._read_import(conn, artifact, run)
        if data is None:
            return None
        row, plan = data
        planned = {(r.payload.source_id, r.payload.measurement_id): r for r in plan.readings}
        linked = []
        from .store import _observation
        for stored in conn.execute("SELECT * FROM observations WHERE run_id = %s AND event_sequence <= %s ORDER BY event_sequence", (run["run_id"], sequence)):
            observation = _observation(stored)
            reading = planned.get((observation.payload.source_id, observation.payload.measurement_id))
            if reading is None or reading.payload != observation.payload:
                raise ReplayIntegrityError("imported observation does not match its source window")
            linked.append(ObservationSource(observation_id=observation.observation_id,
                event_sequence=observation.event_sequence, reading=reading))
        return ImportView(source_hash=plan.source_hash, plan_hash=plan.content_hash,
            repository=plan.config.repository, branch=plan.config.branch,
            imported_at=run["created_at"], observations=tuple(linked))

    def _verify_import(self, conn, artifact, run, publication):
        data = self._read_import(conn, artifact, run)
        if data is None:
            return 0, None
        row, plan = data
        sequence = publication.belief.sequence_cutoff
        if sequence > row["completed_sequence"]:
            raise ReplayIntegrityError("imported run has unexpected subsequent events")
        expected = operations(plan)[:sequence]
        events = conn.execute("SELECT * FROM events WHERE run_id = %s AND sequence > 0 AND sequence <= %s ORDER BY sequence", (run["run_id"], sequence)).fetchall()
        if len(events) != len(expected):
            raise ReplayIntegrityError("import event history is incomplete")
        checked = 0
        for event, (kind, item) in zip(events, expected, strict=True):
            if event["kind"] != kind:
                raise ReplayIntegrityError("import operation order changed")
            if kind == "tick_advanced":
                if event["payload"]["tick"] != item:
                    raise ReplayIntegrityError("import tick differs from its window")
            else:
                if Observation.model_validate(event["payload"]).payload != item.payload:
                    raise ReplayIntegrityError("imported evidence differs from its source")
                checked += 1
        return checked, plan.content_hash

    def get_import_source(self, run_id):
        with self._connect() as conn:
            self._read_transaction(conn)
            run = self._run_row(conn, run_id)
            artifact, _ = self._model(conn, run["model_hash"])
            data = self._read_import(conn, artifact, run)
            if data is None:
                raise NotFoundError("run has no imported source archive")
            return bytes(data[0]["source_content"])
