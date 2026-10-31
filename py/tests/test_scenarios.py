import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import gym_aloha  # noqa: F401
import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import pytest
import torch
from brier_sim.scenarios import (
    apply_scenario,
    build_scenarios,
    scenario_obs_hook,
    scenario_reset_hook,
    scenarios,
    select_seeds,
    tier_obs_noise_hook,
)

PUSHT = "gym_pusht/PushT-v0"
ALOHA = "gym_aloha/AlohaTransferCube-v0"


def _environment(env: str) -> gym.Env:
    return gym.make(env, obs_type="pixels_agent_pos", max_episode_steps=2)


def _rng(identifier: str, seed: int, tier: str) -> np.random.Generator:
    salt = "cheap" if tier == "cheap" else "full"
    digest = hashlib.sha256(f"{identifier}:{seed}:{salt}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


@pytest.mark.parametrize("name,env", [("pusht", PUSHT), ("aloha", ALOHA)])
def test_build_scenarios_balances_native_bins_and_disjoint_seeds(name: str, env: str) -> None:
    definitions = build_scenarios(name)
    assert definitions == build_scenarios(env) and len(definitions) == 16
    assert [item["scenario_id"] for item in definitions] == [f"{name}-s{i:02}" for i in range(1, 17)]
    assert len({seed for item in definitions for seed in item["seeds"]}) == 800
    dimension = "camera_jitter" if name == "pusht" else "physics_jitter"
    assert len({(d["params"]["pose_bin"], d["params"]["lighting"], d["params"][dimension]) for d in definitions}) == 16
    for item in definitions:
        assert set(item) == {"scenario_id", "params", "seeds"}
        assert len(item["seeds"]) == 50 and len(set(item["seeds"])) == 50
        params = item["params"]
        assert set(params) == ({"pose_bin", "lighting", "camera_jitter", "physics_jitter"} if name == "aloha" else {"pose_bin", "lighting", "camera_jitter"})
        if name == "aloha":
            assert params["camera_jitter"] == ((params["pose_bin"] + int(params["lighting"] == "low")) % 2 == 1)
        else:
            assert "physics_jitter" not in params
    definitions[0]["params"]["pose_bin"] = 99
    assert build_scenarios(name)[0]["params"]["pose_bin"] == 1


@pytest.mark.parametrize("name,env", [("pusht", PUSHT), ("aloha", ALOHA)])
def test_select_seeds_rechecks_five_actual_native_resets(name: str, env: str) -> None:
    bins = select_seeds(name)
    assert set(bins) == {1, 2, 3, 4}
    assert all(len(values) == 200 for values in bins.values())
    assert len(set().union(*map(set, bins.values()))) == 800
    selected = [(pose, seed) for pose, seeds in bins.items() for seed in seeds]
    environment = _environment(env)
    try:
        for index in np.random.default_rng(2026).choice(len(selected), size=5, replace=False):
            expected, seed = selected[index]
            environment.reset(seed=seed)
            raw = environment.unwrapped
            position = raw.block.position if name == "pusht" else raw._env.physics.data.qpos[-7:-5]
            midpoint = (256, 256) if name == "pusht" else (0.1, 0.5)
            actual = 1 + int(position[0] >= midpoint[0]) + 2 * int(position[1] >= midpoint[1])
            assert actual == expected
    finally:
        environment.close()
    bins[1][0] = -1
    assert select_seeds(name)[1][0] >= 0


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_select_seeds_rejects_invalid_count(count: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        select_seeds("pusht", count)


@pytest.mark.parametrize("function", [select_seeds, build_scenarios, scenarios])
def test_scenarios_reject_unknown_environment(function: Any) -> None:
    with pytest.raises(ValueError, match="Unsupported env"):
        function("unknown")


@pytest.mark.parametrize("env", [PUSHT, ALOHA])
def test_legacy_scenarios_adapt_only_ids_and_env(env: str) -> None:
    name = "pusht" if env == PUSHT else "aloha"
    definitions = scenarios(env)
    for legacy, canonical in zip(definitions, build_scenarios(name), strict=True):
        assert legacy["env"] == env
        assert legacy["params"] == canonical["params"]
        assert legacy["seeds"] == canonical["seeds"]
        expected = canonical["scenario_id"].replace("aloha-", "aloha_transfer_cube-")
        assert legacy["scenario_id"] == expected


@pytest.mark.parametrize("pose_bin", range(1, 5))
def test_apply_scenario_preserves_native_pusht_pose(pose_bin: int) -> None:
    definition = scenarios(PUSHT)[(pose_bin - 1) * 4]
    environment = _environment(PUSHT)
    seed = definition["seeds"][0]
    environment.reset(seed=seed)
    original = np.asarray(environment.unwrapped.block.position).copy()
    wrapped = apply_scenario(environment, definition, "cheap")
    try:
        observation, info = wrapped.reset(seed=seed)
        np.testing.assert_array_equal(wrapped.unwrapped.block.position, original)
        np.testing.assert_array_equal(observation["pixels"], wrapped.unwrapped.get_obs()["pixels"])
        assert info["is_success"] is False
        assert wrapped.step(np.array([256.0, 256.0]))[3] is False
        assert wrapped.step(np.array([256.0, 256.0]))[3] is True
    finally:
        wrapped.close()


@pytest.mark.parametrize("env", [PUSHT, ALOHA])
def test_reset_hook_restores_original_physics_and_lighting_exactly(env: str) -> None:
    name = "pusht" if env == PUSHT else "aloha"
    normal, changed = build_scenarios(name)[0], build_scenarios(name)[3]
    environment = _environment(env)
    try:
        raw = environment.unwrapped
        if env == ALOHA:
            model = raw._env.physics.model
            original = {key: getattr(model, key).copy() for key in
                        ("body_mass", "geom_friction", "light_diffuse", "light_specular")}
        else:
            environment.reset(seed=normal["seeds"][0])
            original = {"mass": raw.block.mass, "k_p": raw.k_p, "k_v": raw.k_v}
        for definition, tier in ((changed, "expensive"), (changed, "baseline"), (normal, "cheap")):
            seed = definition["seeds"][0]
            environment.reset(seed=seed)
            before = np.asarray(raw.block.position).copy() if env == PUSHT else raw._env.physics.data.qpos.copy()
            scenario_reset_hook(raw, definition, seed, tier, _rng(definition["scenario_id"], seed, tier))
            after = raw.block.position if env == PUSHT else raw._env.physics.data.qpos
            np.testing.assert_array_equal(before, after)
            if env == ALOHA:
                if tier != "cheap":
                    np.testing.assert_array_equal(model.light_diffuse, original["light_diffuse"] * 0.45)
                    np.testing.assert_array_equal(model.light_specular, original["light_specular"] * 0.45)
                    assert not np.array_equal(model.body_mass, original["body_mass"])
                    assert not np.array_equal(model.geom_friction, original["geom_friction"])
                else:
                    for key, value in original.items():
                        np.testing.assert_array_equal(getattr(model, key), value)
            elif tier == "cheap":
                assert raw.block.mass == original["mass"]
                assert raw.k_p == original["k_p"] and raw.k_v == original["k_v"]
            else:
                assert 0.85 <= raw.block.mass / original["mass"] <= 1.15
                assert 0.9 <= raw.k_p / original["k_p"] <= 1.1
                assert 0.9 <= raw.k_v / original["k_v"] <= 1.1
    finally:
        environment.close()


@pytest.mark.parametrize("env", [PUSHT, ALOHA])
def test_baseline_and_expensive_world_and_noise_pair_exactly(env: str) -> None:
    definition = scenarios(env)[3]
    environment = _environment(env)
    results = []
    try:
        for tier in ("expensive", "baseline", "expensive"):
            wrapped = apply_scenario(environment, definition, tier)
            first, _ = wrapped.reset(seed=definition["seeds"][0])
            second = wrapped.step(np.array([240.0, 320.0]) if env == PUSHT else np.zeros(14))[0]
            results.append((first, second))
        for first, second in results[1:]:
            for observed, expected in ((first, results[0][0]), (second, results[0][1])):
                np.testing.assert_array_equal(observed["agent_pos"], expected["agent_pos"])
                image = observed["pixels"] if env == PUSHT else observed["pixels"]["top"]
                expected_image = expected["pixels"] if env == PUSHT else expected["pixels"]["top"]
                np.testing.assert_array_equal(image, expected_image)
    finally:
        environment.close()


def test_scenario_obs_hook_has_fixed_translation_and_pusht_only_lighting() -> None:
    pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    observation = {"pixels": {"top": pixels}, "agent_pos": np.array([12.0, 34.0])}
    state = {"aloha": False, "params": {"lighting": "low"}, "offset": (1, -2)}
    first = scenario_obs_hook(state, observation)
    expected = (np.pad(pixels * 0.55, ((2, 2), (2, 2), (0, 0)), mode="edge")[3:11, 0:8]).astype(np.uint8)
    np.testing.assert_array_equal(first["pixels"]["top"], expected)
    np.testing.assert_array_equal(first["pixels"]["top"], scenario_obs_hook(state, observation)["pixels"]["top"])
    np.testing.assert_array_equal(first["agent_pos"], observation["agent_pos"])
    assert not np.shares_memory(first["agent_pos"], observation["agent_pos"])
    state.update(aloha=True, offset=(0, 0))
    np.testing.assert_array_equal(scenario_obs_hook(state, observation)["pixels"]["top"], pixels)


@pytest.mark.parametrize("tier", ["cheap", "expensive", "baseline"])
def test_tier_obs_noise_hook_uses_exact_documented_sigmas(tier: str) -> None:
    observation = {"pixels": np.full((12, 12, 3), 100, np.uint8), "agent_pos": np.array([1.0, 2.0])}
    actual = tier_obs_noise_hook(observation, tier, np.random.default_rng(42))
    expected_rng = np.random.default_rng(42)
    if tier == "cheap":
        np.testing.assert_array_equal(actual["pixels"], observation["pixels"])
        np.testing.assert_array_equal(actual["agent_pos"], observation["agent_pos"])
    else:
        pixels = np.clip(100 + expected_rng.normal(0, 0.02 * 255, (12, 12, 3)), 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(actual["pixels"], pixels)
        np.testing.assert_array_equal(actual["agent_pos"], observation["agent_pos"] + expected_rng.normal(0, 0.002, 2))


def test_preprocessed_tensor_hooks_preserve_bchw_precision_and_raw_state() -> None:
    image = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 100
    observation = {"observation.images.top": image, "observation.state": torch.tensor([[12.0, 34.0]]), "unrelated_statistics": torch.tensor([0.5])}
    state = {"aloha": False, "params": {"lighting": "low"}, "offset": (1, -2)}
    transformed = scenario_obs_hook(state, observation)
    expected = image[:, :, torch.tensor([1, 2, 3, 3]), :][:, :, :, torch.tensor([0, 0, 0, 1])] * 0.55
    torch.testing.assert_close(transformed["observation.images.top"], expected)
    torch.testing.assert_close(transformed["observation.state"], observation["observation.state"])
    result = tier_obs_noise_hook(transformed, "expensive", np.random.default_rng(123))
    expected_rng = np.random.default_rng(123)
    expected_image = np.clip(expected.numpy() + expected_rng.normal(0, 0.02, expected.shape), 0, 1)
    torch.testing.assert_close(result["observation.images.top"], torch.tensor(expected_image, dtype=torch.float32))
    expected_state = observation["observation.state"].numpy() + expected_rng.normal(0, 0.002, (1, 2))
    torch.testing.assert_close(result["observation.state"], torch.tensor(expected_state, dtype=torch.float32))
    torch.testing.assert_close(result["unrelated_statistics"], observation["unrelated_statistics"])
    for key, value in result.items():
        assert value.dtype == observation[key].dtype and value.shape == observation[key].shape
        assert value.device == observation[key].device and isinstance(value, torch.Tensor)


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
    environment = _environment(PUSHT)
    try:
        with pytest.raises(ValueError):
            apply_scenario(environment, definition, "cheap")
    finally:
        environment.close()


def test_invalid_tier_seed_options_and_repeated_wrapper_fail() -> None:
    definition = scenarios(PUSHT)[0]
    environment = _environment(PUSHT)
    try:
        with pytest.raises(ValueError, match="tier"):
            apply_scenario(environment, definition, "invalid")
        wrapped = apply_scenario(environment, definition, "cheap")
        with pytest.raises(ValueError, match="already"):
            apply_scenario(wrapped, definition, "cheap")
        with pytest.raises(ValueError, match="fixed seeds"):
            wrapped.reset(seed=-1)
        with pytest.raises(ValueError, match="reset options"):
            wrapped.reset(seed=definition["seeds"][0], options={"reset_to_state": []})
        with pytest.raises(ValueError, match="tier"):
            scenario_reset_hook(environment.unwrapped, definition, 0, "invalid", np.random.default_rng())
        with pytest.raises(ValueError, match="fixed seeds"):
            scenario_reset_hook(environment.unwrapped, definition, -1, "cheap", np.random.default_rng())
        with pytest.raises(ValueError, match="tier"):
            tier_obs_noise_hook({}, "invalid", np.random.default_rng())
    finally:
        environment.close()


def test_cli_writes_only_local_manifest_from_brier_data(tmp_path: Path) -> None:
    command = [sys.executable, "-m", "brier_sim.scenarios", "--env", "pusht", "--write"]
    result = subprocess.run(command, check=False, env={**os.environ, "BRIER_DATA": str(tmp_path)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "manifests/scenarios_pusht.json").read_text()) == build_scenarios("pusht")
    rejected = subprocess.run(command, check=False, env={**os.environ, "BRIER_DATA": "gs://example"}, capture_output=True, text=True)
    assert rejected.returncode != 0 and "BRIER_DATA must be a local directory" in rejected.stderr
