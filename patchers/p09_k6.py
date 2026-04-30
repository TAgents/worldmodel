"""p09: rollout K=4 -> K=6 on top of accepted p07.

Calm-lr regime (1e-4 from p07) might absorb the longer compounding that K=8
couldn't at lr=3e-4. K=6 is a midpoint between p04's working K=4 (0.040 with
the lr drop) and K=8's noise floor.

Single-coherent change: one default in Config.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"


def main() -> int:
    src = TRAIN.read_text()
    old = "    rollout_k: int = 4\n"
    new = "    rollout_k: int = 6\n"
    if old not in src:
        sys.exit("p09: rollout_k=4 default not found")
    TRAIN.write_text(src.replace(old, new, 1))
    print("p09: rollout_k 4 -> 6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
