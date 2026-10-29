"""Stable bytes for generated artifacts and evidence digests."""

import json


def canonical_json(value: object) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ) + "\n").encode("utf-8")
