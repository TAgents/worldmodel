"""p02: residual predictor.

Attacks the predictor identity-collapse observed in the baseline. Instead of
predicting z_{t+1} directly, predict the delta and add it to z_t:

    z_pred = z_t + MLP(concat(z_t, a_t))

This is mathematically equivalent in capacity but changes the optimization
landscape: predicting a near-zero delta is easy by initialization, and any
gradient signal pulls the MLP toward modeling the *dynamics* (action-conditioned
shift), not the (much larger) absolute position. Single-coherent change at the
predictor's forward pass — eval.py's contract is unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"

OLD = "    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:\n        return self.net(torch.cat([z, a], dim=-1))\n"
NEW = "    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:\n        return z + self.net(torch.cat([z, a], dim=-1))\n"


def main() -> int:
    src = TRAIN.read_text()
    if OLD not in src:
        print("p02: target line not found (already residual or refactored?)", file=sys.stderr)
        return 1
    TRAIN.write_text(src.replace(OLD, NEW, 1))
    print("p02: predictor now returns z + MLP(z, a) (residual)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
