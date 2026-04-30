# Leaderboard

Append-only log of accepted experiments. Format:

```
## <run_id> — score=<X.XXX> (Δ +<X.XXX>)
**Change:** <one sentence summarizing the diff>
**Why:** <hypothesis-grounded reason it worked>
```

---

## baseline — score=0.000
**Change:** initial LeWM (CNN→192-d latent + LayerNorm, 4-layer MLP predictor, MSE + std-hinge SIGReg, AdamW 3e-4 cosine, 10k steps, grad-clip 1.0; bf16 on CUDA, fp32 on MPS/CPU).
**Why:** Reproduces the Two-Room failure the LeWM paper reports. Encoder is healthy (std 0.32 across positions, ||Δz|| 19.7 end-to-end) but predictor is identity-collapsed (per-step pred err 22× the actual encoder shift). Wall-clock 3:55 on M-series MPS for 10k steps. This is the floor the ratchet ratchets from.

## 87eede52 — score=0.007 (Δ +0.007)
**Change:** p01: latent_dim 192→64, encoder widths shrunk, λ warmup 0→1000 steps (program.md first hypothesis: small latent + warmup against over-regularization)
**Why:** TODO — agent should fill this in.

## 70a66f8e — score=0.040 (Δ +0.033)
**Change:** p07: p04 multi-step K=4 + lr 3e-4 → 1e-4 (compounded grad through K calls is 4× too strong at original lr)
**Why:** TODO — agent should fill this in.
