# program.md — agent brief

You are an autoresearch agent optimizing a Latent World Model (LeWM) on the Two-Room 2D navigation task. You edit `train.py`. The outer ratchet runs your edits, scores them with the frozen `eval.py`, and keeps changes that improve the score.

## Goal

Maximize **mean planning success** on Two-Room over 50 fixed (start, goal) pairs averaged over 3 seeds.

## Hard constraints (do not violate)

1. End-to-end from pixels. No pretrained encoders.
2. No EMA on the target encoder. Use **stop-grad only** on the target latent.
3. Exactly two loss terms: prediction MSE + SIGReg.
4. Do not touch `eval.py` or the dataset on disk.
5. Each experiment must finish in **≤5 min** wall-clock on a single GPU. Use bf16 mixed precision.
6. Train for ≤10k optimizer steps. AdamW only.

## Hypothesis (first move)

Two-Room has very low intrinsic dimensionality (agent xy in a small arena). The default 192-dim latent + standard SIGReg likely *over-regularizes* against a near-degenerate prior, causing the predictor to collapse. Attack:

- Reduce latent dim: try {32, 64, 96}.
- Warm up λ (SIGReg weight) from 0 → target over the first 1–2k steps so the predictor learns structure before the regularizer pulls.

**Observed baseline failure mode** (run on M-series MPS, 10k steps, 2k episodes):
- Encoder is healthy: ||Δz|| across the arena ≈ 19.7, per-position std ≈ 0.32.
- **Predictor is identity-collapsed**: per-step encoder shift ≈ 9e-4 but predictor error ≈ 1.9e-2 — **22× larger than the shift it's meant to predict**.
- Result: CEM rollouts are dominated by predictor noise → 0/150 planning success on Two-Room.
- Reading: the predictor minimizes MSE by approximating identity because per-step latent deltas are tiny relative to absolute latent magnitude. Likely fixes: scale up the action effect (FiLM, larger-magnitude action embedding), use cosine loss on the *delta* (z_pred − z_t) vs (z_t+1 − z_t), or train on multi-step targets.

## Exploration priorities (in rough order)

1. **λ schedule:** constant vs linear warmup vs cosine warmup; target λ ∈ {0.01, 0.05, 0.1, 0.5}.
2. **Latent dim:** {32, 64, 96, 128, 192}.
3. **Predictor architecture:** depth ∈ {3, 4, 6}, width ∈ {256, 512, 1024}, residual vs plain.
4. **Action conditioning:** concat vs FiLM vs cross-attn over a per-token action embedding.
5. **Prediction loss:** MSE vs cosine vs smoothed L1 (still a single term).
6. **Encoder:** small CNN depth/width; stride pattern.
7. **Optimizer:** AdamW betas, weight decay, cosine vs constant LR, warmup steps.

## Reporting format (each experiment)

After every run, append one line to `runs/log.jsonl`:

```json
{"id": "<uuid>", "ts": "<iso8601>", "diff": "<short summary>", "score": 0.42, "delta": 0.03, "accepted": true, "wall_s": 287}
```

If accepted, also append to `leaderboard.md`:

```
## <id> — <score> (Δ +<delta>)
**Change:** one sentence.
**Why it worked:** one sentence (hypothesis, not just "score went up").
```

## What counts as one experiment

A *single* coherent change to `train.py` (e.g. "set latent_dim=64 and warm λ from 0→0.1 over 1k steps"). Mixing unrelated changes makes the ratchet's accept/reject signal worthless.
