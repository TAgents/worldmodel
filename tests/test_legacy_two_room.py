"""Preservation and simulator checks; neural training is a separate experiment."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments" / "two_room"
LEGACY = EXPERIMENT / "legacy"


@pytest.fixture
def environment():
    name = "_worldmodel_legacy_two_room"
    spec = importlib.util.spec_from_file_location(name, LEGACY / "envs" / "two_room.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def test_legacy_snapshot_matches_original_file_hashes():
    manifest = json.loads((EXPERIMENT / "legacy-manifest.json").read_text())
    assert manifest["commit"] == "dc0cafea9ed58224bb2cb2fb4117fe9ff83592d6"
    for name, expected in manifest["files"].items():
        content = (LEGACY / name).read_bytes()
        assert len(content) == expected["bytes"], name
        assert hashlib.sha256(content).hexdigest() == expected["sha256"], name
        blob = b"blob " + str(len(content)).encode() + b"\0" + content
        assert hashlib.sha1(blob).hexdigest() == expected["git_blob_sha1"], name


def test_seeded_navigation_replays_identical_observations(environment):
    first, second = environment.TwoRoom(42), environment.TwoRoom(42)
    initial = first.reset()
    assert initial.shape == (64, 64, 3)
    assert initial.dtype == np.uint8
    np.testing.assert_array_equal(initial, second.reset())
    for action in np.random.default_rng(101).uniform(-1, 1, size=(30, 2)):
        left, right = first.step(action), second.step(action)
        np.testing.assert_array_equal(left.obs, right.obs)
        np.testing.assert_array_equal(left.info["pos"], right.info["pos"])
        assert (left.success, left.done) == (right.success, right.done)


def test_wall_blocks_motion_but_doorway_allows_goal_reaching(environment):
    env = environment.TwoRoom()
    env.reset(start=np.array([0.4, 0.2]), goal=np.array([0.8, 0.2]))
    for _ in range(10):
        result = env.step(np.array([1.0, 0.0]))
    assert result.info["pos"][0] < 0.44
    assert not result.success
    env.reset(start=np.array([0.4, 0.5]), goal=np.array([0.8, 0.5]))
    for _ in range(10):
        result = env.step(np.array([1.0, 0.0]))
        if result.done:
            break
    assert result.success and result.done
    assert result.info["pos"][0] > environment.WALL_X


def test_rollout_collection_keeps_episode_boundaries(environment, tmp_path):
    path = tmp_path / "rollouts.npz"
    environment.TwoRoom.collect(str(path), episodes=3, seed=17)
    with np.load(path) as data:
        starts = data["ep_starts"]
        assert starts.shape == (4,)
        assert starts[0] == 0
        assert np.all((np.diff(starts) > 0) & (np.diff(starts) <= environment.MAX_STEPS))
        assert data["obs"].shape == (starts[-1], 64, 64, 3)
        assert data["act"].shape == (starts[-1], 2)
        assert np.all(np.abs(data["act"]) <= 1)
