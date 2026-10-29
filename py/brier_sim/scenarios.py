from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


def scenarios(env: str) -> list[dict[str, Any]]:
    """Return sixteen fixed pose, lighting, camera and seed configurations."""
    if env == "gym_pusht/PushT-v0":
        prefix, offset = "pusht", 10000
    elif env == "gym_aloha/AlohaTransferCube-v0":
        prefix, offset = "aloha_transfer_cube", 20000
    else:
        raise ValueError(f"Unsupported env: {env}")
    definitions = []
    for pose_bin in range(4):
        for lighting in ("normal", "low"):
            for camera_jitter in (False, True):
                index = len(definitions) + 1
                params = {
                    "pose_bin": pose_bin,
                    "lighting": lighting,
                    "camera_jitter": camera_jitter,
                }
                if prefix == "aloha_transfer_cube":
                    params["physics_jitter"] = bool(pose_bin % 2) ^ camera_jitter
                definitions.append(
                    {
                        "scenario_id": f"{prefix}-s{index:02}",
                        "env": env,
                        "params": params,
                        "seeds": list(
                            range(offset + index * 100, offset + index * 100 + 50)
                        ),
                    }
                )
    return definitions


class _ScenarioEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, definition: dict[str, Any], tier: str) -> None:
        super().__init__(env)
        self._definition = definition
        self._params = definition["params"]
        self._extra = tier != "cheap"
        self._aloha = definition["env"] == "gym_aloha/AlohaTransferCube-v0"
        self._rng = np.random.default_rng()
        if self._aloha:
            physics = self.unwrapped._env.physics
            self._body = physics.model.name2id("box", "body")
            self._geom = physics.model.name2id("red_box", "geom")
            self._mass = float(physics.model.body_mass[self._body])
            self._inertia = physics.model.body_inertia[self._body].copy()
            self._friction = physics.model.geom_friction[self._geom].copy()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if type(seed) is not int or seed not in self._definition["seeds"]:
            raise ValueError("seed must be one of the scenario's fixed seeds")
        if options:
            raise ValueError("scenario reset does not accept custom reset options")
        self._rng = np.random.default_rng(seed)
        _, info = self.env.reset(seed=seed)
        pose = self._params["pose_bin"]
        region = np.array([pose % 2, pose // 2])
        mass_scale = friction_scale = 1.0
        if self._aloha and self._params["physics_jitter"]:
            mass_scale, friction_scale = 1.1, 0.9
        if self._extra:
            mass_scale *= self._rng.uniform(0.95, 1.05)
            friction_scale *= self._rng.uniform(0.95, 1.05)
        if self._aloha:
            physics = self.unwrapped._env.physics
            physics.model.body_mass[self._body] = self._mass * mass_scale
            physics.model.body_inertia[self._body] = self._inertia * mass_scale
            physics.model.geom_friction[self._geom] = self._friction * friction_scale
            with physics.reset_context():
                physics.data.qpos[-7:-5] = (
                    np.array([0.0, 0.4]) + region * 0.1 + self._rng.uniform(0.0, 0.1, 2)
                )
            observation = self.unwrapped._format_raw_obs(
                self.unwrapped._env.task.get_observation(physics)
            )
        else:
            raw = self.unwrapped
            raw.block.position = (
                150 + region * 100 + self._rng.uniform(0.0, 100.0, 2)
            ).tolist()
            raw.space.reindex_shapes_for_body(raw.block)
            raw.block.mass *= mass_scale
            raw.block.moment *= mass_scale
            for shape in raw.block.shapes:
                shape.friction *= friction_scale
            observation = raw.get_obs()
            info = raw._get_info()
            info["is_success"] = False
        return self._observation(observation), info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(observation), reward, terminated, truncated, info

    def _observation(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._observation(item) for key, item in value.items()}
        result = np.asarray(value).copy()
        if result.dtype == np.uint8 and result.ndim == 3:
            image = result.astype(np.float32)
            if self._params["lighting"] == "low":
                image *= 0.6
            if self._params["camera_jitter"]:
                dy, dx = self._rng.integers(-2, 3, 2)
                height, width = image.shape[:2]
                image = np.pad(image, ((2, 2), (2, 2), (0, 0)))[
                    2 + dy : 2 + dy + height, 2 + dx : 2 + dx + width
                ]
            if self._extra:
                image += self._rng.normal(0.0, 1.275, image.shape)
            return np.clip(image, 0, 255).astype(np.uint8)
        if self._extra and np.issubdtype(result.dtype, np.floating):
            sigma = 0.002 if self._aloha else 0.5
            result += self._rng.normal(0.0, sigma, result.shape)
        return result


def apply_scenario(
    environment: gym.Env, scenario: dict[str, Any], tier: str
) -> gym.Env:
    """Wrap an environment with a validated scenario and seeded tier randomization."""
    if tier not in ("cheap", "expensive", "baseline"):
        raise ValueError("tier must be cheap, expensive, or baseline")
    if not isinstance(scenario, dict) or set(scenario) != {
        "scenario_id",
        "env",
        "params",
        "seeds",
    }:
        raise ValueError(
            "scenario must contain exactly scenario_id, env, params, seeds"
        )
    definitions = scenarios(scenario["env"])
    params, seeds = scenario["params"], scenario["seeds"]
    if (
        not isinstance(params, dict)
        or type(params.get("pose_bin")) is not int
        or type(params.get("camera_jitter")) is not bool
        or ("physics_jitter" in params and type(params["physics_jitter"]) is not bool)
        or not isinstance(seeds, list)
        or any(type(seed) is not int for seed in seeds)
        or scenario not in definitions
    ):
        raise ValueError("scenario parameters and seeds must match a fixed definition")
    if environment.spec is None or environment.spec.id != scenario["env"]:
        raise ValueError("environment must match the scenario env")
    if isinstance(environment, _ScenarioEnv):
        raise ValueError("environment already has a scenario wrapper")  # noqa: TRY004
    definition = next(
        item for item in definitions if item["scenario_id"] == scenario["scenario_id"]
    )
    return _ScenarioEnv(environment, definition, tier)
