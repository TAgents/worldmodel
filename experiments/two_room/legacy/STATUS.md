# Status

Phase 1 (scaffolding) + Phase 2 dry-run (10 ratchet iterations on MacBook Pro M-series MPS). Day-zero gate green, ratchet validated, two improvements committed. Production overnight run wants a CUDA host.

## Where the score lives now

| | score | Δ vs prior |
|---|---|---|
| Random init | 0.000 | — |
| `baseline.pt` (vanilla LeWM, faithful reproduction) | 0.000 | reproduces the paper's Two-Room failure mode |
| `p01` accept (latent_dim=64, λ warmup 1000) | 0.007 | +0.007 (set the floor) |
| **`p07` accept (multi-step K=4 + lr 3e-4 → 1e-4)** | **0.040** | **+0.033** |

Plan's done-criterion is `final best ≥ 0.05`. Current state is **+4pp** of the +5pp goal.

## Iteration table

| iter | summary | score | accepted | notes |
|---|---|---|---|---|
| p01 | latent_dim 192→64, encoder widths shrunk, λ warmup 0→1000 | 0.007 | ✅ | Set floor (best was −∞). |
| p02 | residual predictor (z_pred = z + MLP(z, a)) | 0.000 | ❌ | Lowered initial loss → weaker gradient signal. |
| p03 | prediction loss MSE → 1 − cosine_similarity | 0.000 | ❌ | Direction-only signal didn't anchor magnitude. |
| p04 | multi-step rollout K=4, end-state target | 0.020 | ❌ near-miss | Real signal (3× over floor); held by min_delta=0.02. |
| p05 | multi-step K=8 | 0.000 | ❌ | At K=8, target ≈ noise under random-policy data. |
| p06 | teacher-forced K=4, per-step supervision | 0.000 | ❌ | Stop-grad inputs starve the encoder of multi-step grad. |
| **p07** | p04 + lr 3e-4 → 1e-4 | **0.040** | ✅ | New floor. K-fold compounded grad needs calmer optimizer. |
| p08 | predictor 4@512 → 6@1024 | 0.000 | ❌ | Capacity wasn't the bottleneck; undertrained at 10k. Also burst the 5-min budget. |
| p09 | K=4 → K=6 | 0.007 | ❌ | K=4 is the genuine sweet spot, not an artifact of the lr it was tuned with. |
| p10 | action embedding 2 → 64 via 2-layer MLP | 0.007 | ❌ | Extra params undertrained. |

## What works

- **Multi-step rollout, K=4, end-state target.** Single largest lever. Without it the predictor identity-collapses.
- **lr=1e-4 once K>1.** Compounded gradient through K predictor calls is effectively K× larger; the original 3e-4 overshoots.
- **VICReg-style std-hinge SIGReg + LayerNorm on the encoder head + grad-clip 1.0.** Required to keep MPS stable; without these the original baseline NaN'd.

## What we know is *not* the bottleneck

- Predictor capacity (p08 worse).
- Predictor architecture (residual p02 and bigger p08 both worse).
- K above 4 (p05, p09 both worse).
- Direction-only loss (p03 worse).

## Wall-clock on M-series MPS

| stage | time |
|---|---|
| Collect 2k episodes | ~30s |
| Train 10k steps (baseline) | 3:55 |
| Train 10k steps (multi-step K=4) | ~4:00 |
| Train 10k steps (multi-step K=6) | 4:13 |
| Train 10k steps (predictor 6@1024) | 6:59 — out of budget |
| Eval (50 pairs × 3 seeds, CEM 3×100×25) | 35s |

## Known frozen-ness

- `eval.py` and `data/twoor.npz` (2000 episodes, seed=0, random policy) untouched across all 10 iterations.
- All accepted train.py edits respect "exactly two losses (prediction + SIGReg)", "stop-grad target only (no EMA)", "AdamW only", "≤10k steps".

## What to run next, on a CUDA host

```bash
git clone https://github.com/TAgents/worldmodel && cd worldmodel
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m envs.two_room --collect data/twoor.npz --episodes 2000 --seed 0
# Sanity:
.venv/bin/python eval.py --random            # expect 0.000
.venv/bin/python train.py --steps 10000      # ~30s on H100
.venv/bin/python eval.py --checkpoint checkpoints/baseline.pt
# Phase 2 overnight:
# Replace the placeholder patcher with an LLM patcher (Claude Code one-shot
# CLI, OpenAI function-calling, etc.) that reads program.md + leaderboard.md
# + runs/log.jsonl + the current train.py and emits a single coherent diff.
.venv/bin/python ratchet.py \
  --patcher <your-llm-patcher> \
  --iters 100 \
  --min-delta 0.02
```

On CUDA the 5-min budget loosens to ~30s/iter, so 100 experiments fits in ~1h
instead of ~8h, and the bf16 path in `train.py` actually engages.

## Why an LLM patcher should outperform regex patchers

By iteration 8 the floor was set and the next gains needed *combinations* of
levers (e.g. p07's K=4 + lr=1e-4 with p10's action embedding rebalanced for
the smaller train budget). Regex patchers don't read prior accepts/rejects and
can't compose. An LLM patcher reads `leaderboard.md` and `runs/log.jsonl` and
proposes the *next* coherent change conditioned on what already worked.

Concretely, the patcher should be invoked with:

- `train.py` (current state, after all accepted patches)
- `program.md` (constraints + hypotheses)
- `leaderboard.md` (accepts, in order)
- `runs/log.jsonl` (accepts AND rejects, in order)

and emit a unified diff or a rewritten `train.py`.

## Open questions for the next attempt

1. Does cosine-on-delta (predict `z_t+1 − z_t`, supervise with cosine sim of the predicted vs actual delta) work where p03's cosine-on-absolute didn't? Direction-of-change is a more meaningful signal than direction-of-z.
2. Does data quality matter more than architecture? Random-policy rollouts rarely cross the doorway; goal-conditioned data collection (still allowed by the constraints — only the *eval* dataset is frozen) might lift the floor before any architectural change.
3. Is 10k steps enough? With longer cosine schedule (smaller batch ÷ more steps, same compute) the same architecture might converge further.
