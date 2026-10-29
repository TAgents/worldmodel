"""Two-Room 2D top-down navigation env.

Arena [0,1]x[0,1] split by a vertical wall at x=0.5 with a single doorway.
Circular agent, continuous 2D velocity action, 64x64 RGB observation.
Episode ends on success (within `goal_radius` of goal) or after `max_steps`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

OBS_SIZE = 64
AGENT_RADIUS = 0.04
GOAL_RADIUS = 0.05
WALL_X = 0.5
WALL_THICKNESS = 0.04
DOOR_Y = 0.5
DOOR_HALF_WIDTH = 0.10
MAX_VEL = 0.05
MAX_STEPS = 100


@dataclass
class StepResult:
    obs: np.ndarray
    success: bool
    done: bool
    info: dict


class TwoRoom:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.t = 0

    # --- public API ---

    def reset(self, start: np.ndarray | None = None, goal: np.ndarray | None = None) -> np.ndarray:
        self.pos = self._sample_free_point() if start is None else np.asarray(start, dtype=np.float32).copy()
        self.goal = self._sample_free_point(other_room_than=self.pos) if goal is None else np.asarray(goal, dtype=np.float32).copy()
        self.t = 0
        return self._render()

    def step(self, action: np.ndarray) -> StepResult:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0) * MAX_VEL
        new_pos = self.pos + a
        new_pos = self._resolve_collision(self.pos, new_pos)
        new_pos = np.clip(new_pos, AGENT_RADIUS, 1.0 - AGENT_RADIUS).astype(np.float32)
        self.pos = new_pos
        self.t += 1
        success = bool(np.linalg.norm(self.pos - self.goal) < GOAL_RADIUS)
        done = success or self.t >= MAX_STEPS
        return StepResult(obs=self._render(), success=success, done=done, info={"t": self.t, "pos": self.pos.copy()})

    # --- helpers ---

    def _sample_free_point(self, other_room_than: np.ndarray | None = None) -> np.ndarray:
        for _ in range(200):
            p = self.rng.uniform(AGENT_RADIUS, 1.0 - AGENT_RADIUS, size=2).astype(np.float32)
            if self._in_wall(p):
                continue
            if other_room_than is not None and (p[0] < WALL_X) == (other_room_than[0] < WALL_X):
                continue
            return p
        return np.array([0.1, 0.5], dtype=np.float32)

    @staticmethod
    def _in_wall(p: np.ndarray) -> bool:
        if abs(p[0] - WALL_X) > WALL_THICKNESS / 2 + AGENT_RADIUS:
            return False
        if abs(p[1] - DOOR_Y) < DOOR_HALF_WIDTH - AGENT_RADIUS:
            return False
        return True

    @staticmethod
    def _resolve_collision(old: np.ndarray, new: np.ndarray) -> np.ndarray:
        # Sub-step along the segment to avoid tunneling through the wall.
        steps = 8
        cur = old.copy()
        for i in range(1, steps + 1):
            t = i / steps
            cand = old + (new - old) * t
            if TwoRoom._in_wall(cand):
                return cur
            cur = cand
        return cur

    def _render(self) -> np.ndarray:
        img = np.full((OBS_SIZE, OBS_SIZE, 3), 240, dtype=np.uint8)  # arena bg
        ys = (np.arange(OBS_SIZE) + 0.5) / OBS_SIZE
        xs = (np.arange(OBS_SIZE) + 0.5) / OBS_SIZE
        gx, gy = np.meshgrid(xs, ys)  # gx[i,j] = x at col j

        wall_mask = (np.abs(gx - WALL_X) < WALL_THICKNESS / 2) & (np.abs(gy - DOOR_Y) > DOOR_HALF_WIDTH)
        img[wall_mask] = (90, 90, 90)

        goal_mask = (gx - self.goal[0]) ** 2 + (gy - self.goal[1]) ** 2 < GOAL_RADIUS ** 2
        img[goal_mask] = (40, 200, 80)

        agent_mask = (gx - self.pos[0]) ** 2 + (gy - self.pos[1]) ** 2 < AGENT_RADIUS ** 2
        img[agent_mask] = (60, 100, 230)
        return img

    # --- dataset collection ---

    @staticmethod
    def collect(out_path: str, episodes: int = 2000, seed: int = 0) -> None:
        env = TwoRoom(seed=seed)
        rng = np.random.default_rng(seed + 1)
        obs_buf, act_buf, ep_starts = [], [], [0]
        for _ in range(episodes):
            env.reset()
            for _ in range(MAX_STEPS):
                a = rng.uniform(-1, 1, size=2).astype(np.float32)
                obs_buf.append(env._render())
                act_buf.append(a)
                r = env.step(a)
                if r.done:
                    break
            ep_starts.append(len(obs_buf))
        np.savez_compressed(
            out_path,
            obs=np.stack(obs_buf),
            act=np.stack(act_buf),
            ep_starts=np.array(ep_starts, dtype=np.int64),
        )
        print(f"wrote {out_path}: {len(obs_buf)} transitions, {episodes} episodes")


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--collect", type=str, help="output .npz path")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.collect:
        TwoRoom.collect(args.collect, episodes=args.episodes, seed=args.seed)
    else:
        p.print_help()


if __name__ == "__main__":
    _main()
