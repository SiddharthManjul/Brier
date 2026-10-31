from __future__ import annotations

import json
import re
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_model

_POLICIES = {
    "pusht": "lerobot/diffusion_pusht",
    "aloha": "lerobot/act_aloha_sim_transfer_cube_human",
}
_ObsHook = Callable[[dict[str, Any]], dict[str, Any]]
_ActionHook = Callable[[np.ndarray, gym.Env], np.ndarray]


def _load(env_name: str) -> tuple[gym.vector.SyncVectorEnv, Any]:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import AlohaEnv, PushtEnv
    from lerobot.envs.factory import make_env
    from lerobot.policies.factory import make_policy

    pins = json.loads(Path(__file__).with_name("pins.json").read_text())
    if version("lerobot") != pins["lerobot_version"]:
        raise RuntimeError("installed LeRobot does not match pins.json")
    pin = pins["policies"][env_name]
    if pin["repo_id"] != _POLICIES[env_name] or not re.fullmatch(
        r"[0-9a-f]{40}", pin["revision"]
    ):
        raise ValueError("policy pins must name the expected repository and immutable revision")
    checkpoint = snapshot_download(
        pin["repo_id"], revision=pin["revision"], allow_patterns=["*.json", "*.safetensors"]
    )
    env_cfg = PushtEnv() if env_name == "pusht" else AlohaEnv(task="AlohaTransferCube-v0")
    env = make_env(env_cfg, n_envs=1, use_async_envs=False)
    try:
        assert isinstance(env, gym.vector.SyncVectorEnv) and env.num_envs == 1
        config = PreTrainedConfig.from_pretrained(checkpoint, cli_overrides=["--device=cpu"])
        config.pretrained_path = Path(checkpoint)
        policy = make_policy(cfg=config, env_cfg=env_cfg)
        load_model(policy, str(Path(checkpoint) / "model.safetensors"), strict=True, device="cpu")
        policy.eval()
        return env, policy
    except BaseException:
        env.close()
        raise


def _run_episode(
    loaded: tuple[Any, ...], seed: int, obs_hook: _ObsHook | None, action_hook: _ActionHook | None
) -> dict[str, Any]:
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.policies.utils import get_device_from_parameters

    env, policy = loaded
    raw_env = env.envs[0].unwrapped
    device = get_device_from_parameters(policy)
    policy.reset()
    observation, _ = env.reset(seed=[seed])
    max_steps = env.call("_max_episode_steps")[0]
    success = False
    for steps in range(1, max_steps + 1):
        observation = preprocess_observation(observation)
        if obs_hook is not None:
            observation = obs_hook(observation)
        observation = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in observation.items()
        }
        observation = add_envs_task(env, observation)
        with torch.inference_mode():
            action = policy.select_action(observation)
        action_numpy = action.to("cpu").numpy()
        assert action_numpy.ndim == 2 and action_numpy.shape[0] == 1
        if action_hook is not None:
            action_numpy = action_hook(action_numpy, raw_env)
        observation, _, terminated, truncated, info = env.step(action_numpy)
        step_success = (
            info["final_info"][0]["is_success"]
            if "final_info" in info else info["is_success"][0]
        )
        success = bool(step_success) or success
        if bool(terminated[0] or truncated[0]):
            break
    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()
    return {"seed": seed, "success": success, "steps": steps}


def run_plain_episodes(
    env_name: str,
    n: int,
    seeds: list[int],
    *,
    obs_hook: _ObsHook | None = None,
    action_hook: _ActionHook | None = None,
) -> list[dict[str, Any]]:
    """Run pinned CPU policies with LeRobot's native normalization and optional hooks."""
    if env_name not in _POLICIES:
        raise ValueError("env_name must be pusht or aloha")
    if type(n) is not int or n <= 0 or len(seeds) != n:
        raise ValueError("n must be positive and match the number of seeds")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("episode seeds must be nonnegative integers")
    from lerobot.utils.random_utils import set_seed

    torch.set_num_threads(1)
    set_seed(0)
    loaded = _load(env_name)
    try:
        return [_run_episode(loaded, seed, obs_hook, action_hook) for seed in seeds]
    finally:
        loaded[0].close()
