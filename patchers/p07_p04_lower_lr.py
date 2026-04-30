"""p07: p04's multi-step K=4 + lr 3e-4 -> 1e-4.

p04 (multi-step K=4, end-state target) scored 0.020 — best so far, near miss
on the +0.02 gate. Through K=4 predictor calls the gradient is effectively
~4× larger than a single-step setup; with the original lr=3e-4, this likely
overshoots. Drop lr to 1e-4 to recover stable optimization.

Single coherent change: re-apply p04's machinery, then flip lr.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"


def _apply_p04() -> None:
    spec = importlib.util.spec_from_file_location("p04", ROOT / "patchers" / "p04_multistep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = TRAIN.read_text()
    TRAIN.write_text(mod.patch(src))


def main() -> int:
    _apply_p04()
    src = TRAIN.read_text()
    old = "    lr: float = 3e-4\n"
    new = "    lr: float = 1e-4\n"
    if old not in src:
        print("p07: lr=3e-4 not found", file=sys.stderr)
        return 1
    TRAIN.write_text(src.replace(old, new, 1))
    print("p07: applied p04 (multi-step K=4) + lr 3e-4 -> 1e-4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
