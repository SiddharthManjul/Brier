from copy import deepcopy
from typing import Any

import gym_aloha  # noqa: F401
import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import pytest
from brier_sim.scenarios import apply_scenario, scenarios

PUSHT = "gym_pusht/PushT-v0"
ALOHA = "gym_aloha/AlohaTransferCube-v0"


@pytest.mark.parametrize("env", [PUSHT, ALOHA])
def test_scenarios_cover_balanced_parameters_and_fixed_seeds(env: str) -> None:
    definitions = scenarios(env)
    assert definitions == scenarios(env) and len(definitions) == 16
    assert len({item["scenario_id"] for item in definitions}) == 16
    assert len({seed for item in definitions for seed in item["seeds"]}) == 800
    combinations = {
        (p["pose_bin"], p["lighting"], p["camera_jitter"])
        for p in (item["params"] for item in definitions)
    }
    assert len(combinations) == 16
    for item in definitions:
        assert item["env"] == env and len(item["seeds"]) == 50
        assert all(type(seed) is int and seed >= 0 for seed in item["seeds"])
    if env == ALOHA:
        assert sum(item["params"]["physics_jitter"] for item in definitions) == 8
    definitions[0]["params"]["pose_bin"] = 99
    assert scenarios(env)[0]["params"]["pose_bin"] == 0


def test_scenarios_reject_unknown_environment() -> None:
    with pytest.raises(ValueError, match="Unsupported env"):
        scenarios("unknown")


@pytest.mark.parametrize("pose_bin", range(4))
def test_apply_scenario_changes_real_pusht_pose_region(pose_bin: int) -> None:
    definition = scenarios(PUSHT)[pose_bin * 4]
    env = apply_scenario(
        gym.make(PUSHT, obs_type="pixels_agent_pos", max_episode_steps=2),
        definition,
        "cheap",
    )
    try:
        observation, info = env.reset(seed=definition["seeds"][0])
        expected_low = 150 + np.array([pose_bin % 2, pose_bin // 2]) * 100
        position = np.asarray(env.unwrapped.block.position)
        assert np.all(position >= expected_low) and np.all(
            position < expected_low + 100
        )
        np.testing.assert_array_equal(
            observation["pixels"], env.unwrapped.get_obs()["pixels"]
        )
        assert info["is_success"] is False
        result = env.step(np.array([256.0, 256.0]))
        assert len(result) == 5 and result[3] is False
        assert env.step(np.array([256.0, 256.0]))[3] is True
    finally:
        env.close()


@pytest.mark.parametrize("tier", ["cheap", "expensive", "baseline"])
def test_scenario_reset_and_step_are_reproducible(tier: str) -> None:
    definition = scenarios(PUSHT)[3]
    env = apply_scenario(gym.make(PUSHT, obs_type="pixels_agent_pos"), definition, tier)
    try:
        first, _ = env.reset(seed=definition["seeds"][0])
        mass = env.unwrapped.block.mass
        first_step = env.step(np.array([240.0, 320.0]))
        second, _ = env.reset(seed=definition["seeds"][0])
        second_step = env.step(np.array([240.0, 320.0]))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
            np.testing.assert_array_equal(first_step[0][key], second_step[0][key])
        assert mass == env.unwrapped.block.mass
        assert first_step[1:4] == second_step[1:4]
        assert mass == 1.0 if tier == "cheap" else mass != 1.0
    finally:
        env.close()


def test_low_lighting_and_camera_jitter_modify_images_only() -> None:
    for index in (1, 2):
        definition = scenarios(PUSHT)[index]
        env = apply_scenario(
            gym.make(PUSHT, obs_type="pixels_agent_pos"), definition, "cheap"
        )
        try:
            transformed, _ = env.reset(seed=definition["seeds"][0])
            raw = env.unwrapped.get_obs()
            np.testing.assert_array_equal(transformed["agent_pos"], raw["agent_pos"])
            if index == 2:
                np.testing.assert_array_equal(
                    transformed["pixels"], (raw["pixels"] * 0.6).astype(np.uint8)
                )
            else:
                assert not np.array_equal(transformed["pixels"], raw["pixels"])
        finally:
            env.close()


@pytest.mark.parametrize("tier", ["expensive", "baseline"])
def test_full_tiers_add_observation_and_physics_noise(tier: str) -> None:
    definition = scenarios(PUSHT)[0]
    env = apply_scenario(gym.make(PUSHT, obs_type="pixels_agent_pos"), definition, tier)
    try:
        transformed, _ = env.reset(seed=definition["seeds"][0])
        raw = env.unwrapped.get_obs()
        assert not np.array_equal(transformed["pixels"], raw["pixels"])
        assert not np.array_equal(transformed["agent_pos"], raw["agent_pos"])
        assert env.unwrapped.block.mass != 1.0
    finally:
        env.close()


def test_aloha_pose_and_physics_apply_without_graphics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "disable")
    definition = scenarios(ALOHA)[5]
    raw_env = gym.make(
        ALOHA,
        obs_type="pixels_agent_pos",
        max_episode_steps=2,
        observation_width=8,
        observation_height=8,
    )
    task = raw_env.unwrapped._env.task

    def observation(physics: Any) -> dict[str, Any]:
        return {
            "qpos": task.get_qpos(physics),
            "images": {"top": np.full((8, 8, 3), 200, np.uint8)},
        }

    monkeypatch.setattr(task, "get_observation", observation)
    env = apply_scenario(raw_env, definition, "expensive")
    try:
        physics = env.unwrapped._env.physics
        body = physics.model.name2id("box", "body")
        geom = physics.model.name2id("red_box", "geom")
        initial_mass = float(physics.model.body_mass[body])
        initial_friction = physics.model.geom_friction[geom].copy()
        first, _ = env.reset(seed=definition["seeds"][0])
        position = physics.data.qpos[-7:-5].copy()
        assert 0.1 <= position[0] <= 0.2 and 0.4 <= position[1] <= 0.5
        assert physics.model.body_mass[body] != initial_mass
        assert not np.array_equal(physics.model.geom_friction[geom], initial_friction)
        mass = float(physics.model.body_mass[body])
        second, _ = env.reset(seed=definition["seeds"][0])
        assert physics.model.body_mass[body] == mass
        np.testing.assert_array_equal(first["pixels"]["top"], second["pixels"]["top"])
        np.testing.assert_array_equal(position, physics.data.qpos[-7:-5])
        assert len(env.step(np.zeros(14))) == 5
    finally:
        env.close()


@pytest.mark.parametrize("mutation", ["env", "extra", "params", "seeds", "bool"])
def test_invalid_scenarios_fail_loudly(mutation: str) -> None:
    definition = deepcopy(scenarios(PUSHT)[0])
    if mutation == "env":
        definition["env"] = "unknown"
    elif mutation == "extra":
        definition["unused"] = True
    elif mutation == "params":
        definition["params"]["unused"] = True
    elif mutation == "seeds":
        definition["seeds"][0] = -1
    else:
        definition["params"]["camera_jitter"] = 0
    env = gym.make(PUSHT)
    try:
        with pytest.raises(ValueError):
            apply_scenario(env, definition, "cheap")
    finally:
        env.close()


def test_invalid_tier_seed_options_and_repeated_wrapper_fail() -> None:
    definition = scenarios(PUSHT)[0]
    env = gym.make(PUSHT)
    try:
        with pytest.raises(ValueError, match="tier"):
            apply_scenario(env, definition, "invalid")
        wrapped = apply_scenario(env, definition, "cheap")
        with pytest.raises(ValueError, match="already"):
            apply_scenario(wrapped, definition, "cheap")
        with pytest.raises(ValueError, match="fixed seeds"):
            wrapped.reset(seed=17)
        with pytest.raises(ValueError, match="reset options"):
            wrapped.reset(seed=definition["seeds"][0], options={"reset_to_state": []})
    finally:
        env.close()
