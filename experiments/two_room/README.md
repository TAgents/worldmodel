# Two-Room navigation experiment

The original pixel-based navigation project is preserved byte for byte in
`legacy/`, from commit `dc0cafea9ed58224bb2cb2fb4117fe9ff83592d6`.
`legacy-manifest.json` records the original Git blob IDs and SHA-256 file hashes.
The [`legacy/two-room-v1` preservation branch](https://github.com/TAgents/worldmodel/tree/legacy/two-room-v1)
retains the original repository layout and history. A local
`legacy-two-room-v1` tag also points to that commit; publishing the tag remains
a terminal step because the connected GitHub tool cannot create tags.

The engine lives at the repository root. This experiment remains separate:
it uses image observations and action-conditioned neural dynamics, while the
engine currently implements finite-state filtering with boolean observations.
No neural embeddings are presented as calibrated probabilities.

## Check the preserved simulator

From the repository root after installing the engine's development dependencies:

```sh
.venv/bin/python -m pytest -q tests/test_legacy_two_room.py
```

These checks run in the full CI suite. They verify source preservation,
deterministic replay, wall collisions, doorway traversal, and rollout boundaries.
They do not train a network or verify its planning success rate.

## Run a standalone neural experiment

Use a separate environment to keep PyTorch dependencies out of the engine:

```sh
python3.12 -m venv experiments/two_room/.venv
experiments/two_room/.venv/bin/python -m pip install -r experiments/two_room/legacy/requirements.txt
cd experiments/two_room/legacy
../.venv/bin/python -m envs.two_room --collect data/twoor.npz --episodes 2000 --seed 0
../.venv/bin/python train.py --steps 10000
../.venv/bin/python eval.py --checkpoint checkpoints/baseline.pt
```

These commands generate local data and weights. The migrated checkout does not
include the old training data, checkpoints, or experiment logs. The old
`STATUS.md` reports 4% planning success; migration does not reproduce that result.
The original README's 30–50% expectation is historical and was not an observed
result in the recorded status.

The archived `ratchet.py` and patchers assume the old repository root and mutate
Git state. To reproduce their original behavior, use a disposable clone checked
out at commit `dc0cafea9ed58224bb2cb2fb4117fe9ff83592d6`. They are historical code, not the new engine's
experiment executor.

## Next integration gate

1. Freeze separate development and holdout environments before proposing models.
2. Define a versioned interface for action-conditioned dynamics and image
   evidence, including model weights, runtime versions, and reproducible seeds.
3. Run candidates in isolated directories/processes, recording every proposal,
   resource cost, result, and artifact hash. Failures retain the incumbent model.
4. Compare candidates against an established baseline with repeated seeds and
   uncertainty estimates. Consult the holdout only at the promotion gate.
5. Expose validated experiment records through the Explorer before enabling
   automatic model promotion.

This is a synthetic navigation domain. It does not satisfy the project's
separate requirement for validation in two real-world domains.
