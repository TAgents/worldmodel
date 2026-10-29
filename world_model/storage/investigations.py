"""A serialized, atomic executor for the deterministic local synthetic adapter.

No external side effects occur inside these transactions. Remote tools will need
a durable reservation/outbox and reconciliation protocol before being supported.
"""

import secrets

from psycopg.types.json import Jsonb
from pydantic import TypeAdapter

from world_model.agents.contracts import InvestigationConfig, InvestigationStep, InvestigationView, StepReceipt
from world_model.agents.policy import choose, validate_config
from world_model.core import Action, ObservationPayload, Outcome, Reading
from world_model.core.types import Identifier
from world_model.synthetic.adapter import SyntheticSensorAdapter
from world_model.synthetic.generator import GENERATOR_VERSION
from .contracts import ConflictError, NotFoundError, ReplayIntegrityError


class InvestigationStoreMixin:
    def _require_manual_run(self, conn, run_id):
        self._require_not_imported(conn, run_id)
        if conn.execute("SELECT 1 FROM investigations WHERE run_id = %s", (run_id,)).fetchone():
            raise ConflictError("this run is controlled by its synthetic investigation; use investigation steps")

    @staticmethod
    def _investigation_row(conn, run_id):
        row = conn.execute("SELECT * FROM investigations WHERE run_id = %s", (run_id,)).fetchone()
        if row is None:
            raise NotFoundError("no investigation configured for this run")
        if row["adapter_version"] != GENERATOR_VERSION:
            raise ReplayIntegrityError("the recorded synthetic adapter version is unavailable")
        return row

    @staticmethod
    def _steps(conn, run_id, *, cutoff=None):
        query = "SELECT * FROM investigation_steps WHERE run_id = %s"
        args = [run_id]
        if cutoff is not None:
            query += " AND recorded_at <= %s"
            args.append(cutoff)
        steps = []
        for row in conn.execute(query + " ORDER BY ordinal", args):
            step = InvestigationStep.model_validate(row["data"])
            if any(row[key] != getattr(step, key) for key in ("run_id", "request_id", "ordinal", "before_sequence", "after_sequence", "recorded_at")):
                raise ReplayIntegrityError("acquisition receipt differs from its indexed metadata")
            steps.append(step)
        return tuple(steps)

    @staticmethod
    def _investigation_view(row, steps):
        return InvestigationView(run_id=row["run_id"], config=InvestigationConfig.model_validate(row["config"]),
            created_at=row["created_at"], steps=steps,
            spent_credits=steps[-1].spent_credits if steps else 0,
            dispatched_reads=steps[-1].dispatched_reads if steps else 0,
            stopped=bool(steps and steps[-1].decision.kind == "stop"))

    def configure_investigation(self, run_id, config: InvestigationConfig, *, seed=None) -> InvestigationView:
        config = InvestigationConfig.model_validate(config)
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("synthetic seed must be a nonnegative signed 64-bit integer")
        with self._connect() as conn:
            run = self._run_row(conn, run_id, lock=True)
            self._require_not_imported(conn, run_id)
            existing = conn.execute("SELECT * FROM investigations WHERE run_id = %s", (run_id,)).fetchone()
            if existing:
                if InvestigationConfig.model_validate(existing["config"]) != config or (seed is not None and seed != existing["adapter_seed"]):
                    raise ConflictError("investigation settings are immutable; create a new run")
                return self._investigation_view(existing, self._steps(conn, run_id))
            head = self._head(conn, run_id)
            if head["sequence"] != 0:
                raise ConflictError("configure an investigation on a fresh run before supplying observations or advancing time")
            artifact, _ = self._model(conn, run["model_hash"])
            validate_config(artifact, config)
            publication = self._publication(conn, run_id, 0)
            timestamp = self._now(conn, after=publication.belief.published_at)
            row = conn.execute("""INSERT INTO investigations (run_id, config, adapter_seed, adapter_version, created_at)
                VALUES (%s, %s, %s, %s, %s) RETURNING *""", (run_id, Jsonb(config.model_dump(mode="json")),
                secrets.randbits(63) if seed is None else seed, GENERATOR_VERSION, timestamp)).fetchone()
            return self._investigation_view(row, ())

    def investigation(self, run_id, *, sequence_cutoff=None, as_of=None) -> InvestigationView | None:
        # With a historical publication selector, action history is additionally
        # bounded by that publication's timestamp. Later failed/stopping actions
        # can share its sequence and must not leak into the earlier snapshot.
        with self._connect() as conn:
            self._read_transaction(conn)
            self._run_row(conn, run_id)
            row = conn.execute("SELECT * FROM investigations WHERE run_id = %s", (run_id,)).fetchone()
            if row is None:
                return None
            cutoff = None
            if sequence_cutoff is not None or as_of is not None:
                from .store import _cutoff
                condition, values = _cutoff("published_at", "sequence_cutoff", as_of, sequence_cutoff)
                publication = conn.execute("SELECT published_at FROM revisions WHERE run_id = %s" + condition + " ORDER BY sequence_cutoff DESC LIMIT 1", (run_id, *values)).fetchone()
                if publication is None:
                    raise NotFoundError("no belief was published by the requested cutoff")
                cutoff = publication["published_at"]
                if row["created_at"] > cutoff:
                    return None
            return self._investigation_view(row, self._steps(conn, run_id, cutoff=cutoff))

    def _sensor_adapter(self, artifact, row, config):
        return SyntheticSensorAdapter(artifact, seed=row["adapter_seed"], horizon=config.horizon)

    def _inspection_investigation(self, conn, run_id, publication, *, historical):
        row = conn.execute("SELECT * FROM investigations WHERE run_id = %s", (run_id,)).fetchone()
        cutoff = publication.belief.published_at if historical else None
        if row is None or (cutoff is not None and row["created_at"] > cutoff):
            return None
        return self._investigation_view(row, self._steps(conn, run_id, cutoff=cutoff))

    def _verify_investigation(self, conn, artifact, run_id, publication, *, historical):
        from world_model.agents.audit import verify_trace
        from .store import _observation
        view = self._inspection_investigation(conn, run_id, publication, historical=historical)
        if view is None:
            return 0
        sequences = {0, *(step.before_sequence for step in view.steps), *(step.after_sequence for step in view.steps)}
        publications = {seq: self._publication(conn, run_id, seq) for seq in sequences}
        observations = {row["observation_id"]: _observation(row) for row in conn.execute(
            "SELECT * FROM observations WHERE run_id = %s AND event_sequence <= %s", (run_id, publication.belief.sequence_cutoff))}
        try:
            return verify_trace(artifact, view.config, view.steps, publications, observations,
                created_at=view.created_at, final_sequence=publication.belief.sequence_cutoff)
        except (ValueError, KeyError) as error:
            raise ReplayIntegrityError("investigation trace failed policy and accounting verification") from error

    def step_investigation(self, run_id, request_id: str) -> StepReceipt:
        request_id = TypeAdapter(Identifier).validate_python(request_id)
        with self._connect() as conn:
            run = self._run_row(conn, run_id, lock=True)
            row = self._investigation_row(conn, run_id)
            config = InvestigationConfig.model_validate(row["config"])
            steps = self._steps(conn, run_id)
            previous = next((step for step in steps if step.request_id == request_id), None)
            if previous:
                return StepReceipt(created=False, step=previous)
            if steps and steps[-1].decision.kind == "stop":
                return StepReceipt(created=False, step=steps[-1])
            head = self._head(conn, run_id)
            publication = self._publication(conn, run_id, head["sequence"])
            artifact, _ = self._model(conn, run["model_hash"])
            spent = steps[-1].spent_credits if steps else 0
            reads = steps[-1].dispatched_reads if steps else 0
            used = tuple(step.decision.source_id for step in steps if step.decision.kind == "read" and step.decision.tick == head["state_tick"])
            decision = choose(artifact, config, publication.belief.distribution,
                tick=head["state_tick"], spent=spent, reads=reads, used_sources=used)
            timestamp = self._now(conn, after=max(publication.belief.published_at, steps[-1].recorded_at if steps else row["created_at"]))
            action, outcome = None, None
            after_sequence = head["sequence"]
            if decision.kind != "stop":
                action = Action(action_id=request_id, run_id=run_id, tick=decision.tick,
                    kind=decision.kind, source_id=decision.source_id, reserved_cost_credits=decision.cost_credits,
                    created_at=timestamp, dispatched_at=timestamp)
                observation_id, error = None, None
                if decision.kind == "read":
                    spent += decision.cost_credits
                    reads += 1
                    # Failure consumes the reserved cost and sensor slot. A
                    # programming/integrity error aborts the entire transaction.
                    try:
                        reading = Reading.model_validate(self._sensor_adapter(artifact, row, config).read(tick=decision.tick, source_id=decision.source_id))
                    except OSError:
                        error = "sensor read failed"
                    if error is None:
                        if (reading.tick, reading.source_id) != (decision.tick, decision.source_id):
                            raise ReplayIntegrityError("sensor returned a different requested tick or source")
                        result = self._ingest_locked(conn, run, ObservationPayload(
                            **reading.model_dump(), measurement_id=request_id), after=timestamp)
                        observation_id = result.observation.observation_id
                else:
                    result = self._advance_locked(conn, run, decision.tick + 1, after=timestamp)
                if error is None:
                    after_sequence = result.publication.belief.sequence_cutoff
                    timestamp = result.publication.belief.published_at
                outcome = Outcome(outcome_id=request_id, action_id=request_id, run_id=run_id,
                    status="failed" if error else "succeeded", charged_cost_credits=decision.cost_credits,
                    completed_at=timestamp, observation_id=observation_id, error=error)
            step = InvestigationStep(request_id=request_id, ordinal=len(steps), run_id=run_id,
                decision=decision, action=action, outcome=outcome, before_sequence=head["sequence"],
                after_sequence=after_sequence, spent_credits=spent, dispatched_reads=reads, recorded_at=timestamp)
            conn.execute("""INSERT INTO investigation_steps
                (run_id, request_id, ordinal, before_sequence, after_sequence, recorded_at, data)
                VALUES (%s, %s, %s, %s, %s, %s, %s)""", (run_id, request_id, step.ordinal,
                step.before_sequence, step.after_sequence, timestamp, Jsonb(step.model_dump(mode="json"))))
            return StepReceipt(created=True, step=step)
