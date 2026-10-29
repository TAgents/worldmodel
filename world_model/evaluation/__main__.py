"""Run a reproducible synthetic benchmark and preserve its complete artifacts."""

import argparse
import json
from pathlib import Path

from world_model.core import ModelArtifact

from .runner import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new archive directory")
    parser.add_argument("--split", choices=("development", "reserved"), default="development")
    parser.add_argument("--source", default="cheap")
    parser.add_argument("--target-state", default="degraded")
    args = parser.parse_args()
    try:
        artifact = ModelArtifact.load(args.model)
        result = run_benchmark(
            artifact, destination=args.output, split=args.split,
            source_id=args.source, target_state=args.target_state,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"world-model-evaluate: {error}\n")
    report = result.report
    print(json.dumps({
        "report": str(args.output / "report.json"),
        "split": report["split"], "episode_count": report["episode_count"],
        "forecasts_per_method": report["forecasts_per_method"],
        "mean_brier": {name: values["mean_brier"] for name, values in report["methods"].items()},
        "filter_minus_baseline": report["filter_minus_baseline"],
        "reserved_predictive_gate_passed": report["reserved_predictive_gate_passed"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
