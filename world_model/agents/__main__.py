"""Run a bounded synthetic investigation or compare acquisition policies."""

import argparse
import json
import os
from uuid import uuid4

from world_model.core import ModelArtifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", choices=("development", "heldout"), default="development")
    run = sub.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--schema", default="world_model")
    args = parser.parse_args()
    if args.command == "evaluate":
        from world_model.evaluation.acquisition import run_acquisition_benchmark
        result = run_acquisition_benchmark(ModelArtifact.load(args.model), destination=args.output, split=args.split)
        print(json.dumps(result.report, indent=2))
    else:
        from world_model.storage.store import PostgresStore
        dsn = os.environ.get("WORLD_MODEL_DATABASE_URL")
        if not dsn:
            parser.error("set WORLD_MODEL_DATABASE_URL")
        store = PostgresStore(dsn, schema=args.schema)
        view = store.investigation(args.run_id)
        if view is None:
            parser.error("configure an investigation in the explorer or API first")
        for _ in range(view.config.horizon + view.config.max_reads + 1):
            receipt = store.step_investigation(args.run_id, uuid4().hex)
            print(receipt.model_dump_json(), flush=True)
            if receipt.step.decision.kind == "stop":
                break
        else:
            raise RuntimeError("investigation failed to stop within its action bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
