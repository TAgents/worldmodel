"""First-move patcher: program.md hypothesis.

Reduce latent_dim 192 -> 64 and warm up SIGReg lambda from 0 -> target over 1000
steps so the predictor learns structure before the regularizer pulls.

This is a deterministic, single-coherent-change patcher. Real Phase 2 should use
an LLM patcher that reads leaderboard.md and program.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"


def patch(text: str) -> str:
    text = re.sub(r"latent_dim:\s*int\s*=\s*\d+", "latent_dim: int = 64", text, count=1)
    # Match 4th element of encoder_widths tuple to the new latent_dim so the
    # final feature map dim matches what `Encoder.head` projects from.
    text = re.sub(
        r"encoder_widths:\s*tuple\s*=\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)",
        "encoder_widths: tuple = (32, 64, 96, 128)",
        text, count=1,
    )
    text = re.sub(r"lambda_warmup_steps:\s*int\s*=\s*\d+", "lambda_warmup_steps: int = 1000", text, count=1)
    return text


def main() -> int:
    src = TRAIN.read_text()
    new = patch(src)
    if new == src:
        print("p01: no change applied (regex miss?)", file=sys.stderr)
        return 1
    TRAIN.write_text(new)
    print("p01: latent_dim 192->64, encoder widths (32,64,128,192)->(32,64,96,128), lambda_warmup_steps 0->1000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
