from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import gymnasium as gym
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from brier_explain.keyframes import save_keyframes
from lerobot.envs.utils import add_envs_task, preprocess_observation
from PIL import Image

from brier_sim import envs, storage
from brier_sim.changes import action_hook, apply_change, obs_hook
from brier_sim.scenarios import (
    _rng,
    apply_scenario,
    scenario_obs_hook,
    scenario_reset_hook,
    scenarios,
    tier_obs_noise_hook,
)
from brier_sim.telemetry import snapshot, summarize

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy

_SCHEMA = pa.schema([
        *[pa.field(name, pa.string(), nullable=False) for name in
          ("env", "policy", "change_id", "scenario_id", "tier")],
        pa.field("seed", pa.int64(), nullable=False), pa.field("success", pa.bool_(), nullable=False),
        pa.field("steps", pa.int64(), nullable=False), pa.field("terminal_state", pa.string(), nullable=False),
        pa.field("reward_sum", pa.float64(), nullable=False), pa.field("keyframes_uri", pa.string()),
        pa.field("wall_seconds", pa.float64(), nullable=False),
    ])


def _load(env: str, policy: str, tier: str, *, revision: str | None = None) -> tuple[gym.Env, PreTrainedPolicy]:
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("policy revision must be an immutable 40-hex commit")
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    if env == "gym_pusht/PushT-v0":
        import gym_pusht  # noqa: F401

        horizon, policy_class = 300, DiffusionPolicy
    else:
        import gym_aloha  # noqa: F401

        horizon, policy_class = 400, ACTPolicy
    gym_env = gym.make(env, obs_type="pixels_agent_pos", max_episode_steps=horizon // 2 if tier == "cheap" else horizon)
    try:
        pin = {"revision": revision} if revision is not None else {}
        config = PreTrainedConfig.from_pretrained(policy, cli_overrides=["--device=cpu"], **pin)
        config.pretrained_backbone_weights = None
        pretrained_policy = policy_class.from_pretrained(policy, config=config, strict=True, **pin)
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
    raise TypeError(f"Cannot serialize terminal-state value of type {type(value).__name__}")


def _frame(observation: dict[str, Any], env: str) -> np.ndarray:
    pixels = observation["pixels"]
    return (pixels["top"] if env == "gym_aloha/AlohaTransferCube-v0" else pixels).copy()


def run_episodes(
    env: str, policy: str, episodes: int, seed: int,
    change_id: str, scenario_id: str, tier: str, *,
    change: dict[str, Any] | None = None, scenario: dict[str, Any] | None = None, revision: str | None = None,
) -> Path:
    """Run seeded pretrained-policy episodes and save their raw records to Parquet."""
    if env == "gym_pusht/PushT-v0":
        expected_policy, scenario_prefix = "lerobot/diffusion_pusht", "pusht"
    elif env == "gym_aloha/AlohaTransferCube-v0":
        expected_policy, scenario_prefix = "lerobot/act_aloha_sim_transfer_cube_human", "aloha_transfer_cube"
    else:
        raise ValueError(f"Unsupported env: {env}")
    if policy != expected_policy:
        raise ValueError(f"policy for {env} must be {expected_policy}")
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("policy revision must be an immutable 40-hex commit")
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if tier not in ("cheap", "expensive", "baseline"):
        raise ValueError("tier must be cheap, expensive, or baseline")
    if scenario is not None and (
        scenario not in scenarios(env) or scenario["scenario_id"] != scenario_id
        or seed != scenario["seeds"][0] or episodes > len(scenario["seeds"])
    ):
        raise ValueError("scenario, first seed and episode count must match its fixed definition")
    if not isinstance(change_id, str) or not re.fullmatch(r"c[0-9]{3,}", change_id) or int(change_id[1:]) == 0:
        raise ValueError("change_id must be c followed by at least three digits, starting at c001")
    if (
        not isinstance(scenario_id, str) or not re.fullmatch(rf"{scenario_prefix}-s[0-9]{{2,}}", scenario_id)
        or int(scenario_id.rsplit("s", 1)[1]) == 0
    ):
        raise ValueError(f"scenario_id must be {scenario_prefix}-s followed by at least two digits")
    data_root = os.environ.get("BRIER_DATA", "")
    if not data_root.strip() or "://" in data_root:
        raise ValueError("BRIER_DATA must be a local directory; remote storage is not supported")
    data_dir = Path(data_root).expanduser().resolve()
    if not data_dir.is_dir():
        raise ValueError("BRIER_DATA must name an existing local directory")

    gym_env, rows = None, []
    frame_directories: list[Path] = []
    try:
        pin = {"revision": revision} if revision is not None else {}
        gym_env, pretrained_policy = _load(env, policy, tier, **pin)
        if scenario is not None:
            gym_env = apply_scenario(gym_env, scenario, tier)
        if change is not None:
            pretrained_policy = apply_change(pretrained_policy, change, seed=int(change_id[1:]))
        for index in range(episodes):
            episode_seed = seed + index if scenario is None else scenario["seeds"][index]
            started = time.perf_counter()
            torch.manual_seed(episode_seed)
            pretrained_policy.reset()
            observation, info = gym_env.reset(seed=episode_seed)
            frames = [_frame(observation, env)] if tier == "expensive" else []
            snapshots, actions = [snapshot(gym_env)], []
            success = False
            steps, reward_sum = 0, 0.0
            terminated = truncated = False
            with torch.inference_mode():
                while not (terminated or truncated):
                    action = pretrained_policy.select_action(preprocess_observation(observation)).squeeze(0).cpu().numpy()
                    observation, reward, terminated, truncated, info = gym_env.step(action)
                    if tier == "expensive":
                        frames.append(_frame(observation, env))
                    actions.append(action.copy())
                    snapshots.append(snapshot(gym_env))
                    success = bool(info["is_success"]) or success
                    reward_sum += float(reward)
                    steps += 1
            terminal_state = dict(info, terminated=bool(terminated), truncated=bool(truncated), trajectory=summarize(actions, snapshots))
            if revision is not None:
                terminal_state["checkpoint_revision"] = revision
            if env == "gym_aloha/AlohaTransferCube-v0":
                physics = gym_env.unwrapped._env.physics
                terminal_state.update(qpos=physics.data.qpos.copy(), qvel=physics.data.qvel.copy())
            keyframes_uri = None
            if tier == "expensive" and not success:
                relative = Path("public/keyframes") / scenario_prefix / change_id / scenario_id / str(episode_seed) / str(uuid4())
                directory = save_keyframes(frames, data_dir / relative, success=success, tier=tier)
                frame_directories.append(directory)
                keyframes_uri = relative.as_posix()
            elif tier == "cheap":
                frame = _frame(observation, env)
                if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8 or 0 in frame.shape:
                    raise ValueError("cheap keyframe must be a nonempty uint8 RGB observation")
                relative = Path("public/keyframes") / scenario_prefix / change_id / scenario_id / str(episode_seed) / str(uuid4())
                directory = data_dir / relative
                directory.mkdir(parents=True, exist_ok=False)
                frame_directories.append(directory)
                with (directory / f"frame-{steps:04d}.png").open("xb") as stream:
                    Image.fromarray(frame).save(stream, format="PNG")
                keyframes_uri = relative.as_posix()
            rows.append({
                "env": env, "policy": policy, "change_id": change_id, "scenario_id": scenario_id,
                "tier": tier, "seed": episode_seed, "success": success, "steps": steps,
                "terminal_state": json.dumps(terminal_state, default=_json_value, allow_nan=False),
                "reward_sum": reward_sum, "keyframes_uri": keyframes_uri, "wall_seconds": time.perf_counter() - started,
            })
    except BaseException:
        for directory in frame_directories:
            shutil.rmtree(directory)
        raise
    finally:
        if gym_env is not None:
            gym_env.close()

    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    output_dir = data_dir / "episodes"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{uuid4()}.parquet"
    output_file = None
    try:
        output_file = output_path.open("xb")
        with output_file:
            pq.write_table(table, output_file)
    except BaseException:
        if output_file is not None:
            output_path.unlink(missing_ok=True)
        for directory in frame_directories:
            shutil.rmtree(directory)
        raise
    return output_path


def _guide_frame(observation: dict[str, Any], env: str) -> np.ndarray:
    value = observation["pixels"]["top"] if env == "aloha" else observation["pixels"]
    image = Image.fromarray(value)
    if env == "pusht":
        image = image.resize((384, 384), Image.Resampling.NEAREST)
    return np.asarray(image).copy()


def _guide_terminal(raw: Any, env: str) -> dict[str, Any]:
    if env == "pusht":
        return {"agent_pos": list(raw.agent.position), "block_pose": [*raw.block.position, raw.block.angle]}
    physics = raw._env.physics
    qpos = raw._env.task.get_qpos(physics)
    return {"box_xyz": physics.named.data.geom_xpos["red_box"].tolist(),
            "gripper_qpos": [float(qpos[6]), float(qpos[13])]}


def _table_contact(raw: Any) -> bool:
    physics = raw._env.physics
    for contact in physics.data.contact[:physics.data.ncon]:
        pair = [physics.model.id2name(index, "geom") for index in (contact.geom1, contact.geom2)]
        if "table" in pair and any(isinstance(name, str) and "gripper_finger" in name for name in pair):
            return True
    return False


def _guide_episode(loaded: tuple, env: str, scenario: dict, tier: str, seed: int) -> tuple[dict, list]:
    vector, policy, change_state = loaded
    child, raw = vector.envs[0], vector.envs[0].unwrapped
    horizon = child._max_episode_steps // (2 if tier == "cheap" else 1)
    if horizon < 1:
        raise ValueError("tier max_steps must be positive")
    full_tier = "expensive" if tier == "pilot" else tier
    world_rng, change_rng = _rng(scenario, seed, full_tier), _rng(scenario, seed, full_tier)
    observation, _ = child.reset(seed=seed)
    world = scenario_reset_hook(raw, scenario, seed, full_tier, world_rng)
    if env == "aloha":
        observation = raw._format_raw_obs(raw._env.task.get_observation(raw._env.physics))
    torch.manual_seed(seed)
    policy.reset() if change_state is None else change_state.reset()
    start = np.array(raw.block.position) if env == "pusht" else None
    frame_steps = [round(index * (horizon - 1) / 7) for index in range(8)]
    frames, actions, rewards, coverages, contacts, success = {}, [], [], [], 0, False
    for step in range(horizon):
        if tier == "expensive" and step in frame_steps:
            frames[step] = _guide_frame(observation, env)
        batch = scenario_obs_hook(world, preprocess_observation(observation))
        batch = tier_obs_noise_hook(batch, full_tier, world_rng)
        if change_state is not None:
            batch = obs_hook(change_state, batch, change_rng)
        with torch.inference_mode():
            action = policy.select_action(add_envs_task(vector, batch)).cpu().numpy()
        if action.ndim != 2 or action.shape[0] != 1:
            raise ValueError("policy action must have shape (1, action_dim)")
        if change_state is not None:
            action = action_hook(change_state, action, raw, change_rng)
        observation, reward, terminated, truncated, info = child.step(action[0])
        actions.append(action[0].copy())
        rewards.append(float(reward))
        success = bool(info["is_success"]) or success
        if env == "pusht":
            coverages.append(float(info["coverage"]))
        else:
            contacts += int(_table_contact(raw))
        if terminated or truncated:
            break
    if tier == "expensive" and len(actions) < horizon:
        terminal = _guide_frame(observation, env)
        for index in frame_steps:
            if index >= len(actions):
                frames[index] = terminal
        frames[frame_steps[-1]] = terminal
    deltas = np.diff(np.asarray(actions), axis=0)
    flips = (np.sum(deltas[1:] * deltas[:-1], axis=1) < 0) & (np.linalg.norm(deltas[1:], axis=1) > 1e-6)
    row = {"seed": seed, "success": success, "steps": len(actions), "max_steps": horizon,
           "reward_sum": sum(rewards), "max_reward": max(rewards), "final_reward": rewards[-1],
           "coverage_final": coverages[-1] if env == "pusht" else None,
           "coverage_max": max(coverages) if env == "pusht" else None,
           "block_disp_px": float(np.linalg.norm(np.asarray(raw.block.position) - start)) if env == "pusht" else None,
           "touched_right": max(rewards) >= 1 if env == "aloha" else None,
           "touched_left": max(rewards) >= 3 if env == "aloha" else None,
           "lifted": max(rewards) >= 2 if env == "aloha" else None,
           "table_contact_steps": contacts if env == "aloha" else None,
           "action_flip_rate": float(flips.sum() / len(flips)) if len(flips) else 0.,
           "terminal_state_json": json.dumps(_guide_terminal(raw, env), allow_nan=False), "keyframes_uri": None}
    return row, [frames[index] for index in frame_steps] if frames else []


def run_key(env: str, change: dict | None, scenario: dict, tier: str, policy_cache: dict) -> tuple[list[dict], list[str]]:
    """Persist one native key; return rows and uploaded folders (empty on skip).

    None is base. Caller owns cache tuples (vector_env, policy, change_state) and
    must close every cached vector_env at exit.
    """
    change_id = "base" if change is None else change["change_id"]
    uri = storage.key_uri(env, change_id, scenario["scenario_id"], tier)
    if storage.exists(uri):
        return [], []
    count = {"cheap": 3, "expensive": 20, "baseline": 50, "pilot": 10}[tier]
    seeds = scenario["seeds"][:count]
    if len(seeds) != count or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != count:
        raise ValueError("scenario must contain enough distinct nonnegative fixed seeds")
    if change is not None and change["env"] != env:
        raise ValueError("change env must match rollout env")
    cache_key = (env, change_id)
    if cache_key not in policy_cache:
        torch.set_num_threads(1)
        vector, policy = envs._load(env)
        try:
            state = apply_change(policy, change) if change is not None else None
            policy_cache[cache_key] = (vector, policy, state)
        except BaseException:
            vector.close()
            raise
    pins = json.loads(Path(__file__).with_name("pins.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    rows, folders = [], []
    for index, seed in enumerate(seeds):
        started = time.perf_counter()
        row, frames = _guide_episode(policy_cache[cache_key], env, scenario, tier, seed)
        if tier == "expensive" and not row["success"]:
            folder = uri.rsplit("/episodes/", 1)[0] + f"/public/keyframes/{env}/{change_id}/{scenario['scenario_id']}/{index}"
            row["keyframes_uri"] = storage.upload_keyframes(folder, frames)
            folders.append(row["keyframes_uri"])
        del frames
        row.update(env=env, policy_repo=pins["policies"][env]["repo_id"],
                   policy_revision=pins["policies"][env]["revision"], change_id=change_id,
                   scenario_id=scenario["scenario_id"], tier=tier, episode_idx=index,
                   wall_seconds=time.perf_counter() - started, lerobot_version=pins["lerobot_version"],
                   code_commit=commit, host=socket.gethostname())
        rows.append(row)
    storage.write_parquet(rows, uri)
    return rows, folders
