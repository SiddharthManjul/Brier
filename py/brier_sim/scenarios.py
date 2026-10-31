from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

_ENVS = {"pusht": "gym_pusht/PushT-v0", "aloha": "gym_aloha/AlohaTransferCube-v0"}


def _name(env: str) -> str:
    for name, identifier in _ENVS.items():
        if env in (name, identifier):
            return name
    raise ValueError(f"Unsupported env: {env}")


@cache
def _selected(name: str, count: int) -> tuple[tuple[int, ...], ...]:
    environment = gym.make(f"gym_{name}:{_ENVS[name]}")
    bins: list[list[int]] = [[] for _ in range(4)]
    try:
        seed = 0
        while any(len(values) < count * 4 for values in bins):
            environment.reset(seed=seed)
            raw = environment.unwrapped
            position = raw.block.position if name == "pusht" else raw._env.physics.data.qpos[-7:-5]
            midpoint = (256, 256) if name == "pusht" else (0.1, 0.5)
            index = int(position[0] >= midpoint[0]) + 2 * int(position[1] >= midpoint[1])
            if len(bins[index]) < count * 4:
                bins[index].append(seed)
            seed += 1
    finally:
        environment.close()
    return tuple(tuple(values) for values in bins)


def select_seeds(env: str, per_scenario: int = 50) -> dict[int, list[int]]:
    """Scan actual native resets into four disjoint, one-based pose bins."""
    if type(per_scenario) is not int or per_scenario <= 0:
        raise ValueError("per_scenario must be a positive integer")
    return {index + 1: list(values) for index, values in enumerate(_selected(_name(env), per_scenario))}


@lru_cache(maxsize=2)
def _catalog(name: str) -> str:
    seeds, definitions = select_seeds(name), []
    for pose in range(1, 5):
        for light, lighting in enumerate(("normal", "low")):
            for enabled in (False, True):
                offset = (light * 2 + int(enabled)) * 50
                definitions.append({
                    "scenario_id": f"{name}-s{len(definitions) + 1:02}",
                    "params": {"pose_bin": pose, "lighting": lighting,
                               "camera_jitter": enabled if name == "pusht" else (pose + light) % 2 == 1,
                               **({"physics_jitter": enabled} if name == "aloha" else {})},
                    "seeds": seeds[pose][offset:offset + 50],
                })
    return json.dumps(definitions)


def build_scenarios(env: str) -> list[dict[str, Any]]:
    """Return fresh copies of the frozen native-seed scenario catalog."""
    return json.loads(_catalog(_name(env)))


def scenarios(env: str) -> list[dict[str, Any]]:
    """Adapt the same catalog to the existing full-Gym-ID runner contract."""
    name = _name(env)
    definitions = build_scenarios(name)
    for item in definitions:
        item["env"] = _ENVS[name]
        if name == "aloha":
            item["scenario_id"] = item["scenario_id"].replace("aloha-", "aloha_transfer_cube-")
    return definitions


def _rng(scenario: dict[str, Any], seed: int, tier: str) -> np.random.Generator:
    identifier = scenario["scenario_id"].replace("aloha_transfer_cube-", "aloha-")
    salt = "cheap" if tier == "cheap" else "full"
    digest = hashlib.sha256(f"{identifier}:{seed}:{salt}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def scenario_reset_hook(raw_env: gym.Env, scenario: dict[str, Any], seed: int, tier: str, rng: np.random.Generator) -> dict[str, Any]:
    """Restore physics, then apply paired world changes without moving the native pose."""
    if tier not in ("cheap", "expensive", "baseline"):
        raise ValueError("tier must be cheap, expensive, or baseline")
    if type(seed) is not int or seed not in scenario["seeds"]:
        raise ValueError("seed must be one of the scenario's fixed seeds")
    aloha = scenario["scenario_id"].startswith("aloha")
    params, full = scenario["params"], tier != "cheap"
    if not hasattr(raw_env, "_brier_originals"):
        if aloha:
            model = raw_env._env.physics.model
            original = {key: getattr(model, key).copy() for key in
                        ("body_mass", "geom_friction", "light_diffuse", "light_specular")}
        else:
            original = {"mass": raw_env.block.mass, "k_p": raw_env.k_p, "k_v": raw_env.k_v}
        raw_env._brier_originals = original
    original = raw_env._brier_originals
    if aloha:
        model = raw_env._env.physics.model
        for key, value in original.items():
            getattr(model, key)[:] = value
        if params["lighting"] == "low":
            model.light_diffuse[:] *= 0.45
            model.light_specular[:] *= 0.45
        body, geom = model.name2id("box", "body"), model.name2id("red_box", "geom")
        for key, index in (("body_mass", body), ("geom_friction", geom)):
            scale = rng.uniform(0.7, 1.3) if params["physics_jitter"] else 1
            scale *= rng.uniform(0.85, 1.15) if full else 1
            getattr(model, key)[index] *= scale
        raw_env._env.physics.forward()
    else:
        raw_env.block.mass = original["mass"] * (rng.uniform(0.85, 1.15) if full else 1)
        for key in ("k_p", "k_v"):
            setattr(raw_env, key, original[key] * (rng.uniform(0.9, 1.1) if full else 1))
    radius = 12 if aloha else 3
    offset = tuple(rng.integers(-radius, radius + 1, 2)) if params["camera_jitter"] else (0, 0)
    return {"aloha": aloha, "params": params, "offset": offset}


def _map_observation(observation: Any, transform: Any, key: str = "") -> Any:
    if isinstance(observation, dict):
        return {key: _map_observation(value, transform, key) for key, value in observation.items()}
    tensor = hasattr(observation, "detach")
    value = observation.detach().cpu().numpy().copy() if tensor else np.asarray(observation).copy()
    result = transform(value, tensor, key).astype(value.dtype)
    return observation.new_tensor(result) if tensor else result


def scenario_obs_hook(state: dict[str, Any], observation: Any) -> Any:
    """Apply fixed camera translation and lighting to raw or preprocessed images."""
    def transform(value: np.ndarray, tensor: bool, key: str) -> np.ndarray:
        if value.ndim >= 3:
            if not state["aloha"] and state["params"]["lighting"] == "low":
                value = value * 0.55
            axes = (value.ndim - 2, value.ndim - 1) if tensor else (0, 1)
            for axis, offset in zip(axes, state["offset"], strict=True):
                indices = np.clip(np.arange(value.shape[axis]) + offset, 0, value.shape[axis] - 1)
                value = np.take(value, indices, axis=axis)
        return value
    return _map_observation(observation, transform)


def tier_obs_noise_hook(observation: Any, tier: str, rng: np.random.Generator) -> Any:
    """Add full-tier Gaussian noise, preserving normalized image precision and types."""
    if tier not in ("cheap", "expensive", "baseline"):
        raise ValueError("tier must be cheap, expensive, or baseline")
    def transform(value: np.ndarray, tensor: bool, key: str) -> np.ndarray:
        if tier != "cheap":
            if value.ndim >= 3:
                scale = 255 if value.dtype == np.uint8 else 1
                value = np.clip(value + rng.normal(0, 0.02 * scale, value.shape), 0, scale)
            elif key in ("", "agent_pos", "observation.state") and np.issubdtype(value.dtype, np.floating):
                value += rng.normal(0, 0.002, value.shape)
        return value
    return _map_observation(observation, transform)


class _ScenarioEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, definition: dict[str, Any], tier: str) -> None:
        super().__init__(env)
        self._definition, self._tier = definition, tier

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if options:
            raise ValueError("scenario reset does not accept custom reset options")
        if type(seed) is not int or seed not in self._definition["seeds"]:
            raise ValueError("seed must be one of the scenario's fixed seeds")
        self._rng = _rng(self._definition, seed, self._tier)
        observation, info = self.env.reset(seed=seed)
        self._state = scenario_reset_hook(self.unwrapped, self._definition, seed, self._tier, self._rng)
        if self._state["aloha"]:
            raw = self.unwrapped
            observation = raw._format_raw_obs(raw._env.task.get_observation(raw._env.physics))
        return self._observation(observation), info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(observation), reward, terminated, truncated, info

    def _observation(self, value: Any) -> Any:
        return tier_obs_noise_hook(scenario_obs_hook(self._state, value), self._tier, self._rng)


def apply_scenario(environment: gym.Env, scenario: dict[str, Any], tier: str) -> gym.Env:
    """Wrap a native environment with a validated fixed scenario and evaluation tier."""
    if tier not in ("cheap", "expensive", "baseline"):
        raise ValueError("tier must be cheap, expensive, or baseline")
    if isinstance(environment, _ScenarioEnv):
        raise ValueError("environment already has a scenario wrapper")  # noqa: TRY004
    if not isinstance(scenario, dict) or json.dumps(scenario, sort_keys=True) not in (
        json.dumps(item, sort_keys=True) for item in scenarios(scenario.get("env", ""))
    ):
        raise ValueError("scenario parameters and seeds must match a fixed definition")
    if environment.spec is None or environment.spec.id != scenario["env"]:
        raise ValueError("environment must match the scenario env")
    return _ScenarioEnv(environment, scenario, tier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=tuple(_ENVS), required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        root = os.environ.get("BRIER_DATA", "")
        if not root or "://" in root:
            parser.error("BRIER_DATA must be a local directory for manifest writing")
        destination = Path(root) / "manifests" / f"scenarios_{args.env}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(build_scenarios(args.env), indent=2) + "\n")
