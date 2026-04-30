"""p06: teacher-forced multi-step K=4 with per-step supervision.

p04 (K=4, single end-state target) scored 0.020. p05 (K=8, end-state) scored
0.000 — too far. Stay at K=4 but train the predictor on K independent 1-step
problems, teacher-forced from the stop-grad target sequence:

    targets[k] = sg(encode(o_{t+k+1}))  for k in 0..K-1
    losses[0] = MSE(predict(encode(o_t), a_0), targets[0])
    losses[k] = MSE(predict(targets[k-1], a_k), targets[k])  for k in 1..K-1
    loss_pred = mean(losses)

K× more predictor gradient signal per training step, no compounding gradient
through the encoder (step 0 carries the encoder grad; step k>0 inputs are
stop-grad). Single coherent change to the prediction-loss term (still exactly
one prediction term + sigreg).

Requires the dataset to return all K+1 frames, not just the endpoints. So this
patcher rewrites both the dataset and the loop, replacing p04's machinery.
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
        valid = []
        for s, e in zip(self.ep_starts[:-1], self.ep_starts[1:]):
            if e - s > horizon:
                valid.append(np.arange(s, e - horizon, dtype=np.int64))
        self.valid_idx = np.concatenate(valid) if valid else np.array([], dtype=np.int64)

    def sample_batch(self, rng: np.random.Generator, batch: int):
        i = rng.choice(self.valid_idx, size=batch, replace=False)
        K = self.horizon
        a_seq = np.stack([self.act[i + k] for k in range(K)], axis=1)        # (B, K, 2)
        o_target_seq = np.stack([self.obs[i + k + 1] for k in range(K)], axis=1)  # (B, K, 64, 64, 3)
        return self.obs[i], a_seq, o_target_seq
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

NEW_LOOP = '''        ot, a_seq, o_target_seq = ds.sample_batch(rng, cfg.batch_size)
        ot_t = torch.from_numpy(ot).to(device, non_blocking=True)
        a_seq_t = torch.from_numpy(a_seq).to(device, non_blocking=True)
        # (B, K, 64, 64, 3) -> encode each step as a stop-grad target
        K = cfg.rollout_k
        o_targets = torch.from_numpy(o_target_seq).to(device, non_blocking=True)

        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else _nullctx()
        with ctx:
            z_t = model.encode(ot_t)
            with torch.no_grad():
                z_targets = torch.stack(
                    [model.encode(o_targets[:, k]) for k in range(K)], dim=1
                )  # (B, K, D), stop-grad
            losses = []
            for k in range(K):
                z_in = z_t if k == 0 else z_targets[:, k - 1]  # teacher-force, sg
                z_pred = model.predict(z_in, a_seq_t[:, k])
                losses.append(F.mse_loss(z_pred, z_targets[:, k]))
            loss_pred = torch.stack(losses).mean()
            loss_reg = sigreg(z_t)
            lam = lambda_at(step, cfg)
            loss = loss_pred + lam * loss_reg
'''


def patch(text: str) -> str:
    for old, new in [(OLD_CFG, NEW_CFG), (OLD_DS, NEW_DS), (OLD_TRAIN_DS_INIT, NEW_TRAIN_DS_INIT), (OLD_LOOP, NEW_LOOP)]:
        if old not in text:
            raise SystemExit(f"p06: target block not found:\n{old[:80]}...")
        text = text.replace(old, new, 1)
    return text


def main() -> int:
    src = TRAIN.read_text()
    new = patch(src)
    TRAIN.write_text(new)
    print("p06: multi-step K=4 with per-step intermediate supervision (mean MSE over K targets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
