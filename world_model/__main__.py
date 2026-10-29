"""Run the pure filter on a model artifact and already-selected readings."""

import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from world_model.core import ModelArtifact, Reading
from world_model.inference import filter_state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="model JSON artifact")
    parser.add_argument("--readings", type=Path, required=True, help="JSON array of tick/source_id/value records")
    parser.add_argument("--tick", type=int, required=True, help="state tick to estimate")
    args = parser.parse_args()
    try:
        artifact = ModelArtifact.load(args.model)
        readings = TypeAdapter(list[Reading]).validate_json(args.readings.read_bytes())
        result = filter_state(artifact, readings, state_tick=args.tick)
    except (OSError, ValueError) as error:
        parser.exit(2, f"world-model: {error}\n")
    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
