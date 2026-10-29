"""p03: cosine-similarity prediction loss.

Replaces MSE with `1 - cos_sim(z_pred, z_target)`. Scale-invariant: tiny
per-step shifts in absolute MSE produce full direction-error gradient. Stays
within the "single prediction term" constraint (still exactly one prediction
loss + sigreg).

Why MSE was the bottleneck: with LayerNorm-bounded latents, ||z|| ≈ √D and
||Δz|| is small relative to ||z||. MSE(z_pred, z_target) is dominated by the
identity component ||z_pred - z_t||, so the trivial minimum z_pred ≈ z_t hides
the actual dynamics signal. Cosine on absolute z is closer to direction-of-z,
but combined with stop-grad target it should bias the predictor away from
pure-identity collapse.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"

OLD = "            loss_pred = F.mse_loss(z_pred, z_tp1_target)\n"
NEW = "            loss_pred = (1.0 - F.cosine_similarity(z_pred, z_tp1_target, dim=-1)).mean()\n"


def main() -> int:
    src = TRAIN.read_text()
    if OLD not in src:
        print("p03: target line not found (loss already changed?)", file=sys.stderr)
        return 1
    TRAIN.write_text(src.replace(OLD, NEW, 1))
    print("p03: prediction loss MSE -> 1 - cosine_similarity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
