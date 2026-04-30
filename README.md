# worldmodel

LeWM (Latent World Model) + Karpathy-style autoresearch ratchet on Two-Room 2D nav.

## Layout

- `program.md` — agent brief: goal, hard constraints, exploration priorities.
- `envs/two_room.py` — 2D top-down nav env, 64x64 RGB obs, continuous 2D velocity action.
- `eval.py` — **frozen**. 50 seed-locked (start, goal) pairs, CEM-MPC planner. Success = within 0.05 of goal in ≤100 steps.
- `train.py` — LeWM baseline. The autoresearch agent edits this file.
- `ratchet.py` — outer loop: agent edits `train.py`, run with timeout, score via `eval.py`, keep if score > best+min_delta else `git revert`.
- `leaderboard.md` — accepted improvements (append-only).
- `data/` — generated rollouts (gitignored).
- `checkpoints/` — model weights (gitignored).
- `runs/` — per-experiment logs (gitignored).

## Workflow

1. `python -m envs.two_room --collect data/twoor.npz --episodes 2000` — collect random-policy rollouts.
2. `python eval.py --random` — sanity check: random-init model should score ~0%.
3. `python train.py` — train baseline. Should hit 30–50% planning success on Two-Room.
4. `python ratchet.py` — overnight loop. Edits `train.py`, accepts/reverts, appends to `leaderboard.md`.

## Constraints (frozen)

- End-to-end from pixels. No pretrained encoders.
- No EMA on target encoder — stop-grad only.
- Exactly two loss terms: prediction MSE + SIGReg.
- `eval.py` and the dataset are frozen across experiments.
