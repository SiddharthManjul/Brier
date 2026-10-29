from types import SimpleNamespace
from typing import Any

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import pytest
from brier_sim.telemetry import snapshot, summarize


def test_snapshot_captures_real_pusht_position_and_distance() -> None:
    env = gym.make("gym_pusht/PushT-v0", max_episode_steps=1)
    try:
        env.reset(seed=4)
        result = snapshot(env)
        position = np.asarray(env.unwrapped.block.position)
        assert result["object_position"] == [*position, 0.0]
        assert result["goal_distance"] == pytest.approx(
            np.linalg.norm(position - env.unwrapped.goal_pose[:2])
        )
        assert result["collision"] is False
    finally:
        env.close()


@pytest.mark.parametrize(
    "names,expected",
    [(("red_box", "table"), False),
     (("red_box", "vx300s_left/10_left_gripper_finger"), False),
     (("table", "vx300s_left/10_left_gripper_finger"), True),
     (("vx300s_right/4_upper_forearm", "table"), True),
     ((None, "table"), False)],
)
def test_aloha_snapshot_distinguishes_robot_table_collisions(
    names: tuple[str | None, str | None], expected: bool,
) -> None:
    env = gym.Env()
    env.spec = gym.envs.registration.EnvSpec("gym_aloha/AlohaTransferCube-v0")
    physics = SimpleNamespace(
        named=SimpleNamespace(data=SimpleNamespace(geom_xpos={
            "red_box": np.array([1.0, 2.0, 0.08]),
            "vx300s_left/10_left_gripper_finger": np.array([4.0, 6.0, 0.08]),
        })),
        model=SimpleNamespace(id2name=lambda index, _: names[index]),
        data=SimpleNamespace(ncon=1, contact=[SimpleNamespace(geom1=0, geom2=1)]),
    )
    env._env = SimpleNamespace(physics=physics)
    assert snapshot(env) == {
        "object_position": [1.0, 2.0, 0.08], "goal_distance": 5.0,
        "collision": expected,
    }


def test_summarize_reports_trajectory_and_action_statistics() -> None:
    actions = [np.array([value, 0.0]) for value in (0.0, 1.0, 0.0, 2.0)]
    snapshots = [
        {"object_position": [0.0, 0.0, height], "goal_distance": distance,
         "collision": collision}
        for height, distance, collision in zip(
            (0.05, 0.1, 0.15, 0.12, 0.08), (5.0, 4.0, 3.0, 1.0, 2.0),
            (True, False, True, False, True), strict=True,
        )
    ]
    assert summarize(actions, snapshots) == {
        "collision_steps": 2, "action_direction_changes": 2,
        "action_transitions": 2, "object_max_height": 0.15,
        "object_final_height": 0.08, "goal_distance_min": 1.0,
        "goal_distance_final": 2.0,
    }


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_stationary_and_short_actions_have_no_direction_changes(count: int) -> None:
    state = {"object_position": [0.0, 0.0, 0.0], "goal_distance": 1.0, "collision": False}
    result = summarize([np.zeros(2)] * count, [state] * (count + 1))
    assert result["action_direction_changes"] == 0
    assert result["action_transitions"] == max(count - 2, 0)


@pytest.mark.parametrize("field,value", [
    ("object_position", [0.0, 0.0, float("inf")]),
    ("object_position", [0.0, 0.0]), ("goal_distance", -1.0),
    ("goal_distance", float("nan")), ("collision", 1),
])
def test_summarize_rejects_invalid_snapshots(field: str, value: Any) -> None:
    state = {"object_position": [0.0, 0.0, 0.0], "goal_distance": 1.0, "collision": False}
    state[field] = value
    with pytest.raises(ValueError):
        summarize([], [state])


def test_telemetry_rejects_unknown_environment_counts_and_bad_actions() -> None:
    with pytest.raises(ValueError, match="Unsupported telemetry env"):
        snapshot(gym.Env())
    with pytest.raises(ValueError, match="one reset"):
        summarize([], [])
    state = {"object_position": [0.0, 0.0, 0.0], "goal_distance": 1.0, "collision": False}
    with pytest.raises(ValueError, match="finite numeric"):
        summarize([np.array([float("nan")])], [state, state])
    with pytest.raises(ValueError, match="matching shapes"):
        summarize([np.zeros(2), np.zeros(3)], [state] * 3)
    with pytest.raises(ValueError, match="products must be finite"):
        summarize([np.array([1e308]), np.array([-1e308]), np.array([1e308])], [state] * 4)
