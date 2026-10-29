"""Fit labeled sensor calibration, or run its prespecified synthetic evaluation."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

from world_model.core import ModelArtifact
from world_model.core.serialization import canonical_json
from .calibration import CalibrationObservation, CalibrationRevision, fit_calibration
from .evaluation import run_reliability_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--protocol", default="docs/reliability-protocol-v1.json")
    evaluate.add_argument("--split", choices=("development", "heldout"), default="development")
    evaluate.add_argument("--output", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--structure", required=True)
    fit.add_argument("--observations", required=True)
    fit.add_argument("--as-of", required=True)
    fit.add_argument("--output", required=True)
    promote = commands.add_parser("promote", help="register calibration and create a new persistent run atomically")
    promote.add_argument("--structure", required=True)
    promote.add_argument("--revision", required=True)
    promote.add_argument("--run-id", required=True)
    promote.add_argument("--epoch", required=True)
    promote.add_argument("--target-state", required=True)
    promote.add_argument("--dsn", default=os.environ.get("WORLD_MODEL_DATABASE_URL"))
    promote.add_argument("--schema", default="world_model")
    args = parser.parse_args()
    if args.command == "evaluate":
        result = run_reliability_benchmark(args.protocol, destination=args.output, split=args.split)
        print(json.dumps({"archive": args.output, "parameter_recovery_gate_passed": result.report["parameter_recovery_gate_passed"],
            "numerical_ess_gate_passed": result.report["numerical_ess_gate_passed"], "heldout_acquisition_gate_passed": result.report["heldout_acquisition_gate_passed"],
            "methods": {name: {k: values[k] for k in ("mean_brier", "mean_log_loss", "minimum_effective_particles")} for name, values in result.report["methods"].items()}}, indent=2))
    elif args.command == "promote":
        if not args.dsn:
            parser.error("set WORLD_MODEL_DATABASE_URL or pass --dsn")
        from world_model.storage.store import PostgresStore
        result = PostgresStore(args.dsn, schema=args.schema).create_calibrated_run(
            ModelArtifact.load(args.structure), CalibrationRevision.model_validate_json(Path(args.revision).read_bytes()),
            epoch=datetime.fromisoformat(args.epoch), target_state=args.target_state, run_id=args.run_id)
        print(result.model_dump_json(indent=2))
    else:
        observations = [CalibrationObservation.model_validate(item) for item in json.loads(Path(args.observations).read_bytes())]
        revision = fit_calibration(ModelArtifact.load(args.structure), observations, as_of=datetime.fromisoformat(args.as_of))
        with Path(args.output).open("xb") as file:
            file.write(canonical_json(revision.model_dump(mode="json")))
        print(revision.revision_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
