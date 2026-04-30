"""p05: multi-step rollout with horizon K=8.

p04 (K=4) scored 0.020, just under the +0.02 accept threshold but a real 3×
improvement over p01's 0.007 floor. Multi-step composition is the correct axis
— push it harder. Eval rolls out 25 steps; training at K=8 closes more of that
gap. Trade-off: wall-clock grows ~linearly in K (more predictor passes per
training step), so we're spending more of the 5-min budget per experiment.

Single-coherent change: edit one default in p04's machinery (rollout_k 4 -> 8).
This patcher must be applied AFTER p04 has been re-applied, since the latter is
not yet committed. To make it standalone, p05 first applies p04's patch then
flips K.
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
    old = "    rollout_k: int = 4\n"
    new = "    rollout_k: int = 8\n"
    if old not in src:
        print("p05: rollout_k=4 default not found after p04 apply", file=sys.stderr)
        return 1
    TRAIN.write_text(src.replace(old, new, 1))
    print("p05: applied p04 (multi-step machinery), bumped K=4 -> K=8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
