"""LeWM baseline trainer. The autoresearch ratchet edits THIS file.

Constraints (program.md):
- end-to-end from pixels, no pretrained encoders
- stop-grad on target latent (no EMA)
- exactly two loss terms: prediction MSE + SIGReg
- AdamW only, ≤10k steps, bf16, ≤5 min wall-clock
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- config (the ratchet typically tweaks this block) -----------------------

@dataclass
class Config:
    latent_dim: int = 64
    pred_hidden: int = 512
    pred_layers: int = 4
    encoder_widths: tuple = (32, 64, 96, 128)
    lambda_sigreg: float = 1.0
    lambda_warmup_steps: int = 1000  # 0 = no warmup
    grad_clip: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.95)
    batch_size: int = 256
    steps: int = 10_000
    warmup_steps: int = 500
    seed: int = 0
    data_path: str = "data/twoor.npz"
    out_path: str = "checkpoints/baseline.pt"
    device: str = "auto"


# --- model ------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, latent_dim: int, widths=(32, 64, 128, 192)):
        super().__init__()
        c0, c1, c2, c3 = widths
        self.net = nn.Sequential(
            nn.Conv2d(3, c0, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c0, c1, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.GELU(),
        )
        self.head = nn.Linear(c3, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x_uint8_nhwc: torch.Tensor) -> torch.Tensor:
        x = x_uint8_nhwc.to(torch.float32).div_(255.0).permute(0, 3, 1, 2).contiguous()
        h = self.net(x)
        h = h.mean(dim=(2, 3))
        return self.norm(self.head(h))


class Predictor(nn.Module):
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


class LeWM(nn.Module):
    def __init__(self, cfg: Config | None = None):
        super().__init__()
        cfg = cfg or Config()
        self.cfg = cfg
        self.encoder = Encoder(cfg.latent_dim, widths=cfg.encoder_widths)
        self.predictor = Predictor(cfg.latent_dim, cfg.pred_hidden, cfg.pred_layers)

    def encode(self, obs_uint8_nhwc: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs_uint8_nhwc)

    def predict(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.predictor(z, a)


def sigreg(z: torch.Tensor) -> torch.Tensor:
    """Std-hinge anti-collapse regularizer (VICReg-style).

    Per-latent-dim std should be at least 1; only collapse below 1 is penalized.
    Hinge form: mean(relu(1 - std(z, dim=0))). Single regularizer term.
    """
    std = z.std(dim=0, unbiased=False).clamp_min(1e-6)
    return F.relu(1.0 - std).mean()


# --- data -------------------------------------------------------------------

class TransitionDataset:
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


# --- train ------------------------------------------------------------------

def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * p))


def lambda_at(step: int, cfg: Config) -> float:
    if cfg.lambda_warmup_steps <= 0:
        return cfg.lambda_sigreg
    return cfg.lambda_sigreg * min(1.0, step / cfg.lambda_warmup_steps)


def train(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = pick_device(cfg.device)
    use_bf16 = device == "cuda"

    ds = TransitionDataset(cfg.data_path)
    model = LeWM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)

    log_every = max(1, cfg.steps // 20)
    metrics = []
    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, cfg.steps, cfg.warmup_steps, cfg.lr)
        ot, at, otp1 = ds.sample_batch(rng, cfg.batch_size)
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

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % log_every == 0 or step == cfg.steps - 1:
            metrics.append({"step": step, "loss": float(loss.item()), "pred": float(loss_pred.item()),
                            "reg": float(loss_reg.item()), "lambda": lam})
            print(f"step {step:5d}  loss={loss.item():.4f}  pred={loss_pred.item():.4f}  "
                  f"reg={loss_reg.item():.4f}  λ={lam:.4f}")

    save_model(model, cfg, cfg.out_path)
    return {"final_loss": metrics[-1]["loss"], "metrics": metrics}


# --- save / load ------------------------------------------------------------

def save_model(model: LeWM, cfg: Config, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, path)


def load_model(path: str, device: str = "cpu") -> LeWM:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**blob["cfg"])
    model = LeWM(cfg).to(device)
    model.load_state_dict(blob["state_dict"])
    return model


# --- misc -------------------------------------------------------------------

class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/twoor.npz")
    p.add_argument("--out", type=str, default="checkpoints/baseline.pt")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = Config(data_path=args.data, out_path=args.out, steps=args.steps, seed=args.seed)
    train(cfg)


if __name__ == "__main__":
    _main()
