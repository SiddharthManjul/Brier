from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import gymnasium as gym
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lerobot.envs.utils import preprocess_observation

from brier_sim.changes import apply_change
from brier_sim.scenarios import apply_scenario, scenarios

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy

_SCHEMA = pa.schema(
    [
        pa.field("env", pa.string(), nullable=False),
        pa.field("policy", pa.string(), nullable=False),
        pa.field("change_id", pa.string(), nullable=False),
        pa.field("scenario_id", pa.string(), nullable=False),
        pa.field("tier", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("success", pa.bool_(), nullable=False),
        pa.field("steps", pa.int64(), nullable=False),
        pa.field("terminal_state", pa.string(), nullable=False),
        pa.field("reward_sum", pa.float64(), nullable=False),
        pa.field("keyframes_uri", pa.string()),
        pa.field("wall_seconds", pa.float64(), nullable=False),
    ]
)


def _load(env: str, policy: str, tier: str) -> tuple[gym.Env, PreTrainedPolicy]:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    if env == "gym_pusht/PushT-v0":
        import gym_pusht  # noqa: F401

        horizon = 300
        policy_class = DiffusionPolicy
    else:
        import gym_aloha  # noqa: F401

        horizon = 400
        policy_class = ACTPolicy
    gym_env = gym.make(
        env,
        obs_type="pixels_agent_pos",
        max_episode_steps=horizon // 2 if tier == "cheap" else horizon,
    )
    try:
        config = PreTrainedConfig.from_pretrained(
            policy, cli_overrides=["--device=cpu"]
        )
        config.pretrained_backbone_weights = None
        pretrained_policy = policy_class.from_pretrained(
            policy, config=config, strict=True
        )
        pretrained_policy.to("cpu").eval()
        return gym_env, pretrained_policy
    except BaseException:
        gym_env.close()
        raise


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Cannot serialize terminal-state value of type {type(value).__name__}"
    )


def run_episodes(
    env: str,
    policy: str,
    episodes: int,
    seed: int,
    change_id: str,
    scenario_id: str,
    tier: str,
    *,
    change: dict[str, Any] | None = None,
    scenario: dict[str, Any] | None = None,
) -> Path:
    """Run seeded pretrained-policy episodes and save their raw records to Parquet."""
    if env == "gym_pusht/PushT-v0":
        expected_policy, scenario_prefix = "lerobot/diffusion_pusht", "pusht"
    elif env == "gym_aloha/AlohaTransferCube-v0":
        expected_policy = "lerobot/act_aloha_sim_transfer_cube_human"
        scenario_prefix = "aloha_transfer_cube"
    else:
        raise ValueError(f"Unsupported env: {env}")
    if policy != expected_policy:
        raise ValueError(f"policy for {env} must be {expected_policy}")
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if tier not in ("cheap", "expensive", "baseline"):
        raise ValueError("tier must be cheap, expensive, or baseline")
    if scenario is not None and (
        scenario not in scenarios(env)
        or scenario["scenario_id"] != scenario_id
        or seed != scenario["seeds"][0]
        or episodes > len(scenario["seeds"])
    ):
        raise ValueError(
            "scenario, first seed and episode count must match its fixed definition"
        )
    if (
        not isinstance(change_id, str)
        or not re.fullmatch(r"c[0-9]{3,}", change_id)
        or int(change_id[1:]) == 0
    ):
        raise ValueError(
            "change_id must be c followed by at least three digits, starting at c001"
        )
    if (
        not isinstance(scenario_id, str)
        or not re.fullmatch(rf"{scenario_prefix}-s[0-9]{{2,}}", scenario_id)
        or int(scenario_id.rsplit("s", 1)[1]) == 0
    ):
        raise ValueError(
            f"scenario_id must be {scenario_prefix}-s followed by at least two digits"
        )
    data_root = os.environ.get("BRIER_DATA", "")
    if not data_root.strip() or "://" in data_root:
        raise ValueError(
            "BRIER_DATA must be a local directory; remote storage is not supported"
        )
    data_dir = Path(data_root).expanduser().resolve()
    if not data_dir.is_dir():
        raise ValueError("BRIER_DATA must name an existing local directory")

    gym_env = None
    rows = []
    try:
        gym_env, pretrained_policy = _load(env, policy, tier)
        if scenario is not None:
            gym_env = apply_scenario(gym_env, scenario, tier)
        if change is not None:
            pretrained_policy = apply_change(
                pretrained_policy, change, seed=int(change_id[1:])
            )
        for index in range(episodes):
            episode_seed = (
                seed + index if scenario is None else scenario["seeds"][index]
            )
            started = time.perf_counter()
            torch.manual_seed(episode_seed)
            pretrained_policy.reset()
            observation, info = gym_env.reset(seed=episode_seed)
            success = False
            steps, reward_sum = 0, 0.0
            terminated = truncated = False
            with torch.inference_mode():
                while not (terminated or truncated):
                    action = (
                        pretrained_policy.select_action(
                            preprocess_observation(observation)
                        )
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
                    observation, reward, terminated, truncated, info = gym_env.step(
                        action
                    )
                    success = bool(info["is_success"]) or success
                    reward_sum += float(reward)
                    steps += 1
            terminal_state = dict(
                info, terminated=bool(terminated), truncated=bool(truncated)
            )
            if env == "gym_aloha/AlohaTransferCube-v0":
                physics = gym_env.unwrapped._env.physics
                terminal_state.update(
                    qpos=physics.data.qpos.copy(), qvel=physics.data.qvel.copy()
                )
            rows.append(
                {
                    "env": env,
                    "policy": policy,
                    "change_id": change_id,
                    "scenario_id": scenario_id,
                    "tier": tier,
                    "seed": episode_seed,
                    "success": success,
                    "steps": steps,
                    "terminal_state": json.dumps(
                        terminal_state, default=_json_value, allow_nan=False
                    ),
                    "reward_sum": reward_sum,
                    "keyframes_uri": None,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
    finally:
        if gym_env is not None:
            gym_env.close()

    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    output_dir = data_dir / "episodes"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{uuid4()}.parquet"
    output_file = output_path.open("xb")
    try:
        with output_file:
            pq.write_table(table, output_file)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return output_path
