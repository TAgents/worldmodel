"""Exact-oracle publications plus real policy decisions for trace and UI checks."""

from datetime import UTC, datetime, timedelta

from world_model.agents.contracts import InvestigationConfig, InvestigationStep, InvestigationView
from world_model.agents.policy import choose
from world_model.core import Action, BeliefRevision, EvidenceReference, EvidenceSnapshot, Forecast, ModelRegistration, Observation, ObservationPayload, Outcome, Reading, RunSpec
from world_model.storage.contracts import ForecastView, Inspection, ModelView, Publication, RevisionView
from .oracle import enumerate_distribution


def investigation_snapshots(artifact):
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    run = RunSpec(run_id="agent-run", epoch=epoch, created_at=epoch + timedelta(days=1), model_hash=artifact.content_hash)
    config = InvestigationConfig(allowed_sources=("cheap", "precise"), horizon=2, budget_credits=2, max_reads=2)
    configured_at = run.created_at + timedelta(seconds=1)
    model = ModelView(spec=artifact.spec, registration=ModelRegistration(content_hash=artifact.content_hash, model_id=artifact.spec.model_id, model_version=artifact.spec.model_version, registered_at=epoch))
    publications, observations, steps, history, forecasts, snapshots = {}, {}, [], [], [], []
    tick = sequence = spent = reads = 0
    used = []

    def publish(kind, time):
        selected = [Reading(tick=o.payload.tick, source_id=o.payload.source_id, value=o.payload.value) for o in observations.values()]
        snapshot = EvidenceSnapshot(snapshot_id=f"s{sequence}", run_id=run.run_id, sequence_cutoff=sequence, observations=[EvidenceReference(observation_id=o.observation_id, event_sequence=o.event_sequence) for o in observations.values()])
        posterior = enumerate_distribution(artifact.content, selected, tick)
        predictive = enumerate_distribution(artifact.content, selected, tick + 1)
        belief = BeliefRevision(revision_id=f"b{sequence}", run_id=run.run_id, variable=artifact.spec.variable, state_tick=tick, published_at=time, sequence_cutoff=sequence, distribution={s: float(p) for s, p in posterior.items()}, model_hash=artifact.content_hash, evidence_snapshot_id=snapshot.snapshot_id, inference_version="exact-discrete-v1", previous_revision_id=f"b{sequence - 1}" if sequence else None)
        forecast = Forecast(forecast_id=f"f{sequence}", run_id=run.run_id, origin_tick=tick, target_tick=tick + 1, target_state="degraded", probability=float(predictive["degraded"]), issued_at=time, belief_revision_id=belief.revision_id, model_hash=artifact.content_hash, evidence_snapshot_id=snapshot.snapshot_id)
        publication = Publication(belief=belief, evidence=snapshot, forecast=forecast)
        publications[sequence] = publication
        history.append(RevisionView(belief=belief, event_kind=kind))
        forecasts.append(ForecastView(forecast=forecast))

    def snapshot(enabled=True):
        view = InvestigationView(run_id=run.run_id, config=config, created_at=configured_at, spent_credits=spent, dispatched_reads=reads, stopped=bool(steps and steps[-1].decision.kind == "stop"), steps=tuple(steps)) if enabled else None
        return Inspection(run=run, model=model, publication=publications[sequence], observations=tuple(observations.values()), history=tuple(history), forecasts=tuple(forecasts), investigation=view)

    publish("run_created", run.created_at)
    snapshots.append(snapshot(False))
    snapshots.append(snapshot())
    for ordinal in range(5):
        before = sequence
        decision = choose(artifact, config, publications[sequence].belief.distribution, tick=tick, spent=spent, reads=reads, used_sources=tuple(used))
        dispatched = configured_at + timedelta(seconds=2 * ordinal + 1)
        recorded = dispatched + timedelta(seconds=1)
        request_id = f"action-{ordinal}"
        action = outcome = None
        if decision.kind != "stop":
            action = Action(action_id=request_id, run_id=run.run_id, tick=tick, kind=decision.kind, source_id=decision.source_id, reserved_cost_credits=decision.cost_credits, created_at=dispatched, dispatched_at=dispatched)
            sequence += 1
            observation_id = None
            if decision.kind == "read":
                spent += decision.cost_credits
                reads += 1
                used.append(decision.source_id)
                observation_id = f"o{sequence}"
                observations[observation_id] = Observation(observation_id=observation_id, run_id=run.run_id, payload=ObservationPayload(tick=tick, source_id=decision.source_id, value=tick == 0, measurement_id=request_id), observed_at=epoch + timedelta(seconds=60 * tick), ingested_at=recorded, event_sequence=sequence)
                kind = "observation_ingested"
            else:
                forecasts[-1] = ForecastView(forecast=forecasts[-1].forecast, closed_at=recorded, closed_event_sequence=sequence)
                tick += 1
                used = []
                kind = "tick_advanced"
            publish(kind, recorded)
            outcome = Outcome(outcome_id=request_id, action_id=request_id, run_id=run.run_id, status="succeeded", charged_cost_credits=decision.cost_credits, completed_at=recorded, observation_id=observation_id)
        steps.append(InvestigationStep(request_id=request_id, ordinal=ordinal, run_id=run.run_id, decision=decision, action=action, outcome=outcome, before_sequence=before, after_sequence=sequence, spent_credits=spent, dispatched_reads=reads, recorded_at=recorded))
        snapshots.append(snapshot())
    return snapshots, publications, observations
