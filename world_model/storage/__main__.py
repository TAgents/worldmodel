"""Migrate PostgreSQL, demonstrate persistent updates, or inspect/replay a run."""

import argparse
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path

import psycopg

from world_model.core import ModelArtifact, ObservationPayload

from .contracts import NotFoundError, ProjectionError, ReplayIntegrityError
from .database import migrate
from .store import PostgresStore


def demonstrate(store, artifact):
    if not {"cheap", "precise"}.issubset(artifact.spec.observation_models) or "degraded" not in artifact.spec.states:
        raise ValueError("the demo requires the first sprint's cheap/precise sensors and degraded state")
    store.register_model(artifact)
    created = store.create_run(artifact.content_hash, epoch=datetime.now(UTC) - timedelta(days=1), target_state="degraded")
    run_id = created.publication.belief.run_id
    first = store.ingest(run_id, ObservationPayload(measurement_id="cheap-0", tick=0, source_id="cheap", value=True))
    original_bytes = first.publication.model_dump_json()
    store.advance_tick(run_id, 1)
    store.ingest(run_id, ObservationPayload(measurement_id="cheap-1", tick=1, source_id="cheap", value=False))
    before_late = store.advance_tick(run_id, 2)
    late = store.ingest(run_id, ObservationPayload(measurement_id="precise-0", tick=0, source_id="precise", value=False))
    correction = store.ingest(run_id, ObservationPayload(
        measurement_id="precise-0-correction", tick=0, source_id="precise", value=True,
        supersedes_id=late.observation.observation_id,
    ))
    historical = store.belief(run_id, as_of=first.publication.belief.published_at)
    if historical.model_dump_json() != original_bytes:
        raise ReplayIntegrityError("historical publication changed after late evidence")
    replay = store.replay(run_id)
    return {
        "run_id": run_id,
        "first_alert_posterior": first.publication.belief.distribution["degraded"],
        "before_late_evidence": before_late.publication.belief.distribution["degraded"],
        "after_late_evidence": late.publication.belief.distribution["degraded"],
        "after_correction": correction.publication.belief.distribution["degraded"],
        "historical_publication_unchanged": True,
        "replay_events_verified": replay.events_checked,
        "frozen_forecast_count": len(replay.closed_forecast_ids),
        "current_publication": replay.publication.model_dump(mode="json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=os.environ.get("WORLD_MODEL_DATABASE_URL"), help="defaults to WORLD_MODEL_DATABASE_URL")
    parser.add_argument("--schema", default="world_model")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate")
    demo = commands.add_parser("demo")
    demo.add_argument("--model", type=Path, required=True)
    for name in ("belief", "history", "replay", "rebuild-projection"):
        command = commands.add_parser(name)
        command.add_argument("--run-id", required=True)
        if name != "rebuild-projection":
            command.add_argument("--as-of")
            command.add_argument("--sequence-cutoff", type=int)
    args = parser.parse_args()
    if not args.dsn:
        parser.error("set WORLD_MODEL_DATABASE_URL or pass --dsn")
    store = PostgresStore(args.dsn, schema=args.schema)
    try:
        if args.command == "migrate":
            result = {"applied_migrations": migrate(args.dsn, args.schema)}
        elif args.command == "demo":
            result = demonstrate(store, ModelArtifact.load(args.model))
        elif args.command == "rebuild-projection":
            result = store.rebuild_projection(args.run_id).model_dump(mode="json")
        else:
            method = getattr(store, args.command)
            data = method(args.run_id, as_of=args.as_of, sequence_cutoff=args.sequence_cutoff)
            result = [item.model_dump(mode="json") for item in data] if isinstance(data, tuple) else data.model_dump(mode="json")
    except (OSError, ValueError, NotFoundError, ProjectionError, ReplayIntegrityError, psycopg.Error) as error:
        parser.exit(2, f"world-model-store: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
