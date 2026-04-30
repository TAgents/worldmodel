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
