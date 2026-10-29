"""p08: bigger predictor (4 layers @ 512 -> 6 @ 1024) on top of accepted p07.

Predictor capacity bump. Now that gradient flow is calm (lr=1e-4) and signal is
strong (multi-step K=4), give the predictor more layers/width to model
action-conditioned dynamics.

Operates on the current train.py state (which already has p07's accepted
changes); does not re-apply earlier patchers.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"


def _replace(src: str, old: str, new: str) -> str:
    if old not in src:
        sys.exit(f"p08: not found: {old.strip()[:60]}")
    return src.replace(old, new, 1)


def main() -> int:
    src = TRAIN.read_text()
    src = _replace(src, "    pred_hidden: int = 512\n", "    pred_hidden: int = 1024\n")
    src = _replace(src, "    pred_layers: int = 4\n", "    pred_layers: int = 6\n")
    TRAIN.write_text(src)
    print("p08: predictor 4@512 -> 6@1024")
    return 0


if __name__ == "__main__":
    sys.exit(main())
