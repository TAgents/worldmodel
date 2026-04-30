"""p10: action embedding (2-d -> 64-d via MLP).

Two parameters in concat([64-d z, 2-d a]) gives the action 3% of the predictor's
input dimensionality. Project the action through a small MLP first so it gets
~half the input slots and the predictor can no longer learn to ignore it.

Single coherent change: extend Predictor with a 2-layer action embedder; concat
with z. Latent dim and predictor depth/width unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"

OLD = '''class Predictor(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, layers: int, action_dim: int = 2):
        super().__init__()
        dims = [latent_dim + action_dim] + [hidden] * (layers - 1) + [latent_dim]
        seq = []
        for i, (di, do) in enumerate(zip(dims[:-1], dims[1:])):
            seq.append(nn.Linear(di, do))
            if i < len(dims) - 2:
                seq.append(nn.GELU())
        self.net = nn.Sequential(*seq)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))
'''

NEW = '''class Predictor(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, layers: int, action_dim: int = 2):
        super().__init__()
        emb_dim = max(latent_dim // 1, 64)
        self.action_emb = nn.Sequential(
            nn.Linear(action_dim, emb_dim), nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        dims = [latent_dim + emb_dim] + [hidden] * (layers - 1) + [latent_dim]
        seq = []
        for i, (di, do) in enumerate(zip(dims[:-1], dims[1:])):
            seq.append(nn.Linear(di, do))
            if i < len(dims) - 2:
                seq.append(nn.GELU())
        self.net = nn.Sequential(*seq)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, self.action_emb(a)], dim=-1))
'''


def main() -> int:
    src = TRAIN.read_text()
    if OLD not in src:
        sys.exit("p10: Predictor block not found (already patched?)")
    TRAIN.write_text(src.replace(OLD, NEW, 1))
    print("p10: action embedding 2 -> 64 via 2-layer MLP, concat with z")
    return 0


if __name__ == "__main__":
    sys.exit(main())
