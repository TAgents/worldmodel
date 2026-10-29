"""p04: multi-step rollout training (horizon K=4).

Per-step prediction has no signal: ||Δz||² is 22× smaller than the predictor's
representational floor, so MSE can be minimized by predicting near-identity.
Multi-step composition makes errors compound, blowing up the predictor's
trivial solutions and forcing it to actually model dynamics.

Single-coherent change, contained to:
  1. TransitionDataset.sample_batch -> returns (o_t, a_seq[K], o_{t+K})
  2. train loop -> rolls out predictor K times before computing MSE
  3. Config -> adds rollout_k

Eval.py contract (predict(z, a) -> z_next) is unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"

OLD_CFG = "    seed: int = 0\n    data_path: str = \"data/twoor.npz\"\n"
NEW_CFG = "    rollout_k: int = 4\n    seed: int = 0\n    data_path: str = \"data/twoor.npz\"\n"

OLD_DS = '''class TransitionDataset:
    def __init__(self, path: str):
        d = np.load(path)
        self.obs = d["obs"]              # (N, 64, 64, 3) uint8
        self.act = d["act"]              # (N, 2) float32
        self.ep_starts = d["ep_starts"]  # (E+1,) int64
        # valid (t, t+1) pairs: within each episode, indices [s, s+1, ..., e-2]
        valid = []
        for s, e in zip(self.ep_starts[:-1], self.ep_starts[1:]):
            if e - s >= 2:
                valid.append(np.arange(s, e - 1, dtype=np.int64))
        self.valid_idx = np.concatenate(valid) if valid else np.array([], dtype=np.int64)

    def sample_batch(self, rng: np.random.Generator, batch: int):
        i = rng.choice(self.valid_idx, size=batch, replace=False)
        return self.obs[i], self.act[i], self.obs[i + 1]
'''

NEW_DS = '''class TransitionDataset:
    def __init__(self, path: str, horizon: int = 1):
        d = np.load(path)
        self.obs = d["obs"]              # (N, 64, 64, 3) uint8
        self.act = d["act"]              # (N, 2) float32
        self.ep_starts = d["ep_starts"]  # (E+1,) int64
        self.horizon = horizon
        # valid t such that [t..t+K] all live inside the same episode.
        valid = []
        for s, e in zip(self.ep_starts[:-1], self.ep_starts[1:]):
            if e - s > horizon:
                valid.append(np.arange(s, e - horizon, dtype=np.int64))
        self.valid_idx = np.concatenate(valid) if valid else np.array([], dtype=np.int64)

    def sample_batch(self, rng: np.random.Generator, batch: int):
        i = rng.choice(self.valid_idx, size=batch, replace=False)
        K = self.horizon
        a_seq = np.stack([self.act[i + k] for k in range(K)], axis=1)  # (B, K, 2)
        return self.obs[i], a_seq, self.obs[i + K]
'''

OLD_TRAIN_DS_INIT = "    ds = TransitionDataset(cfg.data_path)\n"
NEW_TRAIN_DS_INIT = "    ds = TransitionDataset(cfg.data_path, horizon=cfg.rollout_k)\n"

OLD_LOOP = '''        ot, at, otp1 = ds.sample_batch(rng, cfg.batch_size)
        ot_t = torch.from_numpy(ot).to(device, non_blocking=True)
        at_t = torch.from_numpy(at).to(device, non_blocking=True)
        otp1_t = torch.from_numpy(otp1).to(device, non_blocking=True)

        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else _nullctx()
        with ctx:
            z_t = model.encode(ot_t)
            with torch.no_grad():
                z_tp1_target = model.encode(otp1_t)  # stop-grad on target
            z_pred = model.predict(z_t, at_t)
            loss_pred = F.mse_loss(z_pred, z_tp1_target)
            loss_reg = sigreg(z_t)
            lam = lambda_at(step, cfg)
            loss = loss_pred + lam * loss_reg
'''

NEW_LOOP = '''        ot, a_seq, otpK = ds.sample_batch(rng, cfg.batch_size)  # a_seq: (B,K,2)
        ot_t = torch.from_numpy(ot).to(device, non_blocking=True)
        a_seq_t = torch.from_numpy(a_seq).to(device, non_blocking=True)
        otpK_t = torch.from_numpy(otpK).to(device, non_blocking=True)

        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else _nullctx()
        with ctx:
            z_t = model.encode(ot_t)
            with torch.no_grad():
                z_tpK_target = model.encode(otpK_t)  # stop-grad on K-step target
            z = z_t
            for k in range(cfg.rollout_k):
                z = model.predict(z, a_seq_t[:, k])
            loss_pred = F.mse_loss(z, z_tpK_target)
            loss_reg = sigreg(z_t)
            lam = lambda_at(step, cfg)
            loss = loss_pred + lam * loss_reg
'''


def patch(text: str) -> str:
    for old, new in [(OLD_CFG, NEW_CFG), (OLD_DS, NEW_DS), (OLD_TRAIN_DS_INIT, NEW_TRAIN_DS_INIT), (OLD_LOOP, NEW_LOOP)]:
        if old not in text:
            raise SystemExit(f"p04: target block not found:\n{old[:80]}...")
        text = text.replace(old, new, 1)
    return text


def main() -> int:
    src = TRAIN.read_text()
    new = patch(src)
    TRAIN.write_text(new)
    print("p04: multi-step rollout training, horizon K=4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
