"""Preview or atomically import a GitHub Actions REST export."""

import argparse
from importlib.resources import files
import json
import os
from pathlib import Path

from world_model.core import ModelArtifact
from .github_actions import ActionsConfig, ImportPreview, prepare_import


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "import"))
    fixtures = files("world_model.adapters").joinpath("fixtures")
    parser.add_argument("--model", default=str(fixtures.joinpath("software-delivery-v1.json")))
    parser.add_argument("--export", dest="export_file", default=str(fixtures.joinpath("github-actions-example.json")))
    parser.add_argument("--config", default=str(fixtures.joinpath("github-actions-config.json")))
    parser.add_argument("--run-id")
    parser.add_argument("--target-state", default="unstable")
    parser.add_argument("--expected-plan-hash")
    parser.add_argument("--dsn", default=os.environ.get("WORLD_MODEL_DATABASE_URL"))
    parser.add_argument("--schema", default="world_model")
    args = parser.parse_args()
    artifact = ModelArtifact.load(args.model)
    content = Path(args.export_file).read_bytes()
    config = ActionsConfig.model_validate_json(Path(args.config).read_bytes())
    if args.command == "preview":
        plan = prepare_import(artifact, content, config)
        print(ImportPreview(plan=plan, plan_hash=plan.content_hash).model_dump_json(indent=2))
    else:
        if not args.dsn or not args.run_id:
            parser.error("import requires --run-id and WORLD_MODEL_DATABASE_URL or --dsn")
        from world_model.storage.store import PostgresStore
        result = PostgresStore(args.dsn, schema=args.schema).import_github_actions(artifact, content, config,
            run_id=args.run_id, target_state=args.target_state, expected_plan_hash=args.expected_plan_hash)
        print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
