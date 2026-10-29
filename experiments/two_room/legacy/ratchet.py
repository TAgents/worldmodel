"""Outer-loop autoresearch ratchet.

Each iteration:
    1. Snapshot current best score and current train.py.
    2. Invoke a *patcher* command that mutates train.py (any executable; the
       agent reads program.md and the leaderboard, writes a new train.py).
    3. Run training with a wall-clock timeout.
    4. Run frozen eval.py to score the new model.
    5. If score > best + min_delta, accept (git commit, append leaderboard);
       else revert (git checkout -- train.py) and keep best.

Always appends a one-line JSON record to runs/log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(ROOT, "train.py")
LOG_PATH = os.path.join(ROOT, "runs", "log.jsonl")
LEADERBOARD = os.path.join(ROOT, "leaderboard.md")
TRAIN_TIMEOUT_S = 6 * 60
EVAL_TIMEOUT_S = 12 * 60


def sh(cmd: list[str], cwd: str = ROOT, timeout: float | None = None, env: dict | None = None) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env or os.environ)
    return p.returncode, p.stdout, p.stderr


def git_head_train() -> str:
    rc, out, _ = sh(["git", "rev-parse", "HEAD:train.py"])
    return out.strip() if rc == 0 else ""


def git_revert_train() -> None:
    sh(["git", "checkout", "--", "train.py"])


def git_commit_train(message: str) -> None:
    sh(["git", "add", "train.py", "leaderboard.md", "runs/log.jsonl"])
    sh(["git", "commit", "-m", message])


def parse_score_from_eval(stdout: str) -> float | None:
    # eval.py prints: "mean=0.123  per_seed=[..]  pairs=50"
    for tok in stdout.split():
        if tok.startswith("mean="):
            try:
                return float(tok.split("=", 1)[1])
            except ValueError:
                return None
    return None


def run_train(python: str, data_path: str, ckpt: str) -> tuple[bool, str]:
    t0 = time.time()
    try:
        rc, out, err = sh([python, "train.py", "--data", data_path, "--out", ckpt],
                          timeout=TRAIN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, f"train timeout after {TRAIN_TIMEOUT_S}s"
    return rc == 0, f"train rc={rc} ({time.time()-t0:.0f}s)\n--stdout--\n{out[-2000:]}\n--stderr--\n{err[-1000:]}"


def run_eval(python: str, ckpt: str) -> tuple[float | None, str]:
    try:
        rc, out, err = sh([python, "eval.py", "--checkpoint", ckpt], timeout=EVAL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, f"eval timeout after {EVAL_TIMEOUT_S}s"
    if rc != 0:
        return None, f"eval rc={rc}\n{err[-1000:]}"
    return parse_score_from_eval(out), out + err


def append_log(record: dict) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def append_leaderboard(record: dict, summary: str) -> None:
    line = (
        f"\n## {record['id']} — score={record['score']:.3f} "
        f"(Δ +{record['delta']:.3f})\n"
        f"**Change:** {summary}\n"
        f"**Why:** TODO — agent should fill this in.\n"
    )
    with open(LEADERBOARD, "a") as f:
        f.write(line)


def current_best() -> float:
    """Best accepted score so far. -inf if none."""
    if not os.path.exists(LOG_PATH):
        return float("-inf")
    best = float("-inf")
    with open(LOG_PATH) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("accepted") and r.get("score") is not None:
                    best = max(best, float(r["score"]))
            except json.JSONDecodeError:
                continue
    return best


def iterate(patcher: list[str], python: str, data_path: str, ckpt: str, min_delta: float, summary: str) -> dict:
    rid = uuid.uuid4().hex[:8]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    best = current_best()

    rc, out, err = sh(patcher)
    if rc != 0:
        rec = {"id": rid, "ts": ts, "stage": "patcher", "rc": rc, "err": err[-500:],
               "score": None, "delta": None, "accepted": False}
        append_log(rec)
        return rec

    train_ok, train_msg = run_train(python, data_path, ckpt)
    if not train_ok:
        git_revert_train()
        rec = {"id": rid, "ts": ts, "stage": "train", "score": None, "delta": None,
               "accepted": False, "msg": train_msg[-1500:]}
        append_log(rec)
        return rec

    score, eval_msg = run_eval(python, ckpt)
    if score is None:
        git_revert_train()
        rec = {"id": rid, "ts": ts, "stage": "eval", "score": None, "delta": None,
               "accepted": False, "msg": eval_msg[-1500:]}
        append_log(rec)
        return rec

    delta = score - (best if best != float("-inf") else 0.0)
    accepted = score > best + min_delta
    rec = {"id": rid, "ts": ts, "stage": "done", "score": score, "delta": delta,
           "accepted": accepted, "summary": summary}
    append_log(rec)

    if accepted:
        append_leaderboard(rec, summary)
        git_commit_train(f"ratchet: accept {rid} score={score:.3f} Δ+{delta:.3f}")
    else:
        git_revert_train()
    return rec


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--patcher", nargs="+", required=True,
                   help="Command that mutates train.py in place (e.g. an LLM CLI).")
    p.add_argument("--summary", default="(no summary)", help="Human-readable summary of the change.")
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--data", default="data/twoor.npz")
    p.add_argument("--ckpt", default="checkpoints/run.pt")
    p.add_argument("--min-delta", type=float, default=0.02)
    args = p.parse_args()

    for i in range(args.iters):
        rec = iterate(args.patcher, args.python, args.data, args.ckpt, args.min_delta, args.summary)
        print(f"[{i+1}/{args.iters}] {rec['stage']} score={rec.get('score')} accepted={rec.get('accepted')}")


if __name__ == "__main__":
    _main()
