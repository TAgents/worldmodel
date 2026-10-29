"""Independent exact-oracle publications for the import UI and replay tests."""

from datetime import UTC, datetime, timedelta
from importlib.resources import files

from world_model.adapters.github_actions import ActionsConfig, ImportView, ObservationSource, operations, prepare_import
from world_model.core import BeliefRevision, EvidenceSnapshot, Forecast, ModelArtifact, ModelRegistration, Observation, Reading, RunSpec
from world_model.storage.contracts import ForecastView, Inspection, ModelView, Publication, RevisionView
from .oracle import enumerate_distribution


def import_fixture():
    root = files("world_model.adapters").joinpath("fixtures")
    model = ModelArtifact(root.joinpath("software-delivery-v1.json").read_bytes())
    content = root.joinpath("github-actions-example.json").read_bytes()
    config = ActionsConfig.model_validate_json(root.joinpath("github-actions-config.json").read_bytes())
    plan = prepare_import(model, content, config)
    now = datetime(2026, 2, 1, tzinfo=UTC)
    run = RunSpec(run_id="imported-run", model_hash=model.content_hash, epoch=plan.epoch, created_at=now)
    view = ModelView(spec=model.spec, registration=ModelRegistration(content_hash=model.content_hash,
        model_id=model.spec.model_id, model_version=model.spec.model_version, registered_at=now))
    observations, source_links, history, snapshots, events, forecasts, closures = [], [], [], [], [], [], {}
    tick = 0
    for sequence, (kind, item) in enumerate((("run_created", None), *operations(plan))):
        timestamp = now + timedelta(microseconds=sequence)
        if kind == "run_created":
            payload = {"run":run.model_dump(mode="json"), "target_state":"unstable", "import_plan_hash":plan.content_hash}
        elif kind == "tick_advanced":
            tick = item
            payload = {"tick":tick,"frozen_forecast_id":forecasts[-1].forecast_id}
            closures[forecasts[-1].forecast_id] = (timestamp, sequence)
        else:
            observation = Observation(observation_id=f"o{sequence}", run_id=run.run_id, payload=item.payload,
                observed_at=item.window_end, ingested_at=timestamp, event_sequence=sequence)
            observations.append(observation)
            source_links.append(ObservationSource(observation_id=observation.observation_id, event_sequence=sequence, reading=item))
            payload = observation.model_dump(mode="json")
        events.append({"run_id":run.run_id, "sequence":sequence, "recorded_at":timestamp, "kind":kind, "payload":payload})
        readings = [Reading(tick=o.payload.tick, source_id=o.payload.source_id, value=o.payload.value) for o in observations]
        evidence = EvidenceSnapshot(snapshot_id=f"s{sequence}", run_id=run.run_id, sequence_cutoff=sequence,
            observations=[{"observation_id":o.observation_id,"event_sequence":o.event_sequence} for o in observations])
        belief = BeliefRevision(revision_id=f"b{sequence}", run_id=run.run_id, variable=model.spec.variable, state_tick=tick,
            published_at=timestamp, sequence_cutoff=sequence, distribution={k:float(v) for k,v in enumerate_distribution(model.content, readings, tick).items()},
            model_hash=model.content_hash, evidence_snapshot_id=evidence.snapshot_id, inference_version="exact-discrete-v1",
            previous_revision_id=f"b{sequence-1}" if sequence else None)
        forecast = Forecast(forecast_id=f"f{sequence}", run_id=run.run_id, origin_tick=tick, target_tick=tick+1,
            target_state="unstable", probability=float(enumerate_distribution(model.content, readings, tick+1)["unstable"]),
            issued_at=timestamp, belief_revision_id=belief.revision_id, model_hash=model.content_hash, evidence_snapshot_id=evidence.snapshot_id)
        forecasts.append(forecast)
        history.append(RevisionView(belief=belief, event_kind=kind))
        snapshots.append(Inspection(run=run, model=view, publication=Publication(belief=belief, evidence=evidence, forecast=forecast),
            observations=tuple(observations), history=tuple(history),
            forecasts=[ForecastView(forecast=f, closed_at=closures.get(f.forecast_id,(None,None))[0], closed_event_sequence=closures.get(f.forecast_id,(None,None))[1]) for f in forecasts],
            imported=ImportView(source_hash=plan.source_hash, plan_hash=plan.content_hash, repository=config.repository,
                branch=config.branch, imported_at=now, observations=tuple(source_links))))
    return model, content, plan, events, snapshots
