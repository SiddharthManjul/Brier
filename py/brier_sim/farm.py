from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pyarrow.parquet as pq

from brier_sim.runner import _SCHEMA, run_episodes
from brier_sim.scenarios import scenarios

_ENVS = {
    "pusht": ("gym_pusht/PushT-v0", "lerobot/diffusion_pusht"),
    "aloha_transfer_cube": (
        "gym_aloha/AlohaTransferCube-v0",
        "lerobot/act_aloha_sim_transfer_cube_human",
    ),
}
_LOG = logging.getLogger(__name__)


def catalog() -> dict[str, Any]:
    """Return the deterministic 80-change-per-environment rollout catalog."""
    definitions: dict[str, Any] = {"changes": [], "scenarios": []}
    for env_index, (env, _) in enumerate(_ENVS.values()):
        rng = np.random.default_rng(42)
        benign = {0, *rng.choice(np.arange(1, 80), size=27, replace=False).tolist()}
        for index in range(80):
            choices = [
                ("obs_noise", {"sigma": float(rng.choice([0.01, 0.03, 0.05, 0.1]))}),
                ("action_latency", {"k": int(rng.choice([1, 2, 4]))}),
                ("action_scale", {"scale": float(rng.choice([0.8, 0.9, 1.1, 1.2]))}),
                ("action_bias", {"drift": float(rng.choice([-0.002, 0.001, 0.003]))}),
                ("camera_shift", {"dx": 4, "dy": -3, "angle": 5.0}),
                ("occlusion", {"x": 24, "y": 24, "width": 24, "height": 24}),
                ("brightness", {"gain": 0.7, "contrast": 1.2}),
                ("frame_drop", {"p": 0.2}),
                ("chunk_len", {"n_action_steps": 4}),
                ("weight_noise", {"sigma": 0.0001}),
            ]
            if index in benign:
                families = [{"family": "benign", "params": {}}]
            else:
                selected = rng.choice(len(choices), size=1 + index % 2, replace=False)
                families = [
                    {"family": choices[int(i)][0], "params": dict(choices[int(i)][1])}
                    for i in selected
                ]
            definitions["changes"].append(
                {
                    "change_id": f"c{env_index * 80 + index + 1:03}",
                    "env": env,
                    "card": {
                        "description": "Evaluate "
                        + ", ".join(f["family"] for f in families),
                        "families": families,
                    },
                }
            )
        definitions["scenarios"].extend(scenarios(env))
    return definitions


def _copy(source: str, destination: str) -> None:
    subprocess.run(
        ["gcloud", "storage", "cp"]
        + (["--no-clobber"] if destination.startswith("gs://") else [])
        + [source, destination],
        check=True,
        capture_output=True,
        text=True,
    )


def _existing(uri: str, local: Path) -> bool:
    result = subprocess.run(
        ["gcloud", "storage", "objects", "describe", uri, "--format=json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        if "404" in result.stderr or "not found" in result.stderr.lower():
            return False
        raise RuntimeError(f"Cannot inspect {uri}: {result.stderr.strip()}")
    with tempfile.NamedTemporaryFile(
        dir=local.parent, prefix=f".{local.name}-", delete=False
    ) as download:
        downloaded = Path(download.name)
    try:
        _copy(uri, str(downloaded))
        downloaded.replace(local)
    finally:
        downloaded.unlink(missing_ok=True)
    return True


def _validate(
    path: Path, change: dict[str, Any], scenario: dict[str, Any], tier: str
) -> None:
    count = {"cheap": 3, "expensive": 20, "baseline": 50}[tier]
    table = pq.read_table(path)
    if table.schema != _SCHEMA:
        raise ValueError(f"Invalid completed rollout schema: {path}")
    rows = table.to_pylist()
    if len(rows) != count or any(
        row["change_id"] != change["change_id"]
        or row["scenario_id"] != scenario["scenario_id"]
        or row["env"] != change["env"]
        or row["tier"] != tier
        or row["policy"] != next(p for e, p in _ENVS.values() if e == change["env"])
        or row["seed"] != scenario["seeds"][i]
        or type(row["success"]) is not bool
        or type(row["steps"]) is not int
        or row["steps"] <= 0
        or not np.isfinite(row["wall_seconds"])
        or row["wall_seconds"] < 0
        or not np.isfinite(row["reward_sum"])
        for i, row in enumerate(rows)
    ):
        raise ValueError(f"Invalid completed rollout key: {path}")
    for row in rows:
        try:
            terminal = json.loads(row["terminal_state"])
            json.dumps(terminal, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid completed rollout terminal JSON: {path}"
            ) from error
        if not isinstance(terminal, dict):
            raise ValueError(f"Invalid completed rollout terminal object: {path}")  # noqa: TRY004
        if row["success"] or tier == "cheap":
            continue
        trajectory = terminal.get("trajectory")
        metrics = (
            "collision_steps",
            "action_direction_changes",
            "action_transitions",
            "object_max_height",
            "object_final_height",
            "goal_distance_min",
            "goal_distance_final",
        )
        if not isinstance(trajectory, dict) or any(
            type(trajectory.get(key)) not in (int, float)
            or not np.isfinite(trajectory[key])
            or trajectory[key] < 0
            for key in metrics
        ):
            raise ValueError(f"Invalid completed rollout failure telemetry: {path}")
        if (
            trajectory["action_direction_changes"] > trajectory["action_transitions"]
            or trajectory["goal_distance_min"]
            > trajectory["goal_distance_final"] + 1e-9
        ):
            raise ValueError(
                f"Inconsistent completed rollout failure telemetry: {path}"
            )


def run_shard(
    env: str, shard: int, shards: int, *, baseline: bool = False
) -> list[str]:
    """Run one change shard and preserve validated, completed rollout keys."""
    if env not in _ENVS or type(shards) is not int or shards <= 0:
        raise ValueError("env must be supported and shards must be a positive integer")
    if type(shard) is not int or not 0 <= shard < shards:
        raise ValueError("shard must be in [0, shards)")
    if type(baseline) is not bool:
        raise ValueError("baseline must be a boolean")
    raw_root = os.environ.get("BRIER_DATA", "")
    parsed = urlsplit(raw_root)
    remote = parsed.scheme == "gs"
    if (
        not raw_root.strip()
        or (parsed.scheme and not remote)
        or (
            remote
            and (
                not raw_root.startswith("gs://")
                or not parsed.netloc
                or parsed.query
                or parsed.fragment
                or "@" in parsed.netloc
                or ":" in parsed.netloc
            )
        )
    ):
        raise ValueError("BRIER_DATA must be a local directory or gs:// bucket prefix")
    root = raw_root.rstrip("/") if remote else raw_root
    definitions = catalog()
    completed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="brier-rollouts-") as temporary:
        local = Path(temporary) if remote else Path(root).expanduser().resolve()
        local.mkdir(parents=True, exist_ok=True)
        metadata = local / "catalog.json"
        content = json.dumps(definitions, sort_keys=True, indent=2) + "\n"
        if remote:
            exists = _existing(f"{root}/catalog.json", metadata)
        else:
            exists = metadata.exists()
        if not exists:
            metadata.write_text(content)
            if remote:
                _copy(str(metadata), f"{root}/catalog.json")
                if not _existing(f"{root}/catalog.json", metadata):
                    raise RuntimeError("Uploaded rollout catalog is missing")
        if metadata.read_text() != content:
            raise ValueError(
                "Existing catalog differs from deterministic rollout definitions"
            )
        previous = os.environ["BRIER_DATA"]
        os.environ["BRIER_DATA"] = str(local)
        try:
            env_id, policy = _ENVS[env]
            changes = [c for c in definitions["changes"] if c["env"] == env_id]
            for change in changes[:1] if baseline else changes[shard::shards]:
                selected = (
                    scenarios(env_id)[shard::shards] if baseline else scenarios(env_id)
                )
                for scenario in selected:
                    tiers = (
                        (("baseline", 50),)
                        if baseline
                        else (("cheap", 3), ("expensive", 20))
                    )
                    for tier, count in tiers:
                        key = f"episodes/{env}/{change['change_id']}/{scenario['scenario_id']}/{tier}.parquet"
                        destination = local / key
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        uri = f"{root}/{key}"
                        exists = (
                            _existing(uri, destination)
                            if remote
                            else destination.exists()
                        )
                        if exists:
                            _validate(destination, change, scenario, tier)
                        else:
                            result = run_episodes(
                                env_id,
                                policy,
                                count,
                                scenario["seeds"][0],
                                change["change_id"],
                                scenario["scenario_id"],
                                tier,
                                change=change["card"],
                                scenario=scenario,
                            )
                            _validate(result, change, scenario, tier)
                            result.replace(destination)
                            if remote:
                                _copy(str(destination), uri)
                                if not _existing(uri, destination):
                                    raise RuntimeError(
                                        f"Uploaded rollout is missing: {uri}"
                                    )
                                _validate(destination, change, scenario, tier)
                        completed.append(uri)
                        _LOG.info("Completed %s", uri)
        finally:
            os.environ["BRIER_DATA"] = previous
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=_ENVS)
    parser.add_argument("--shard", required=True, type=int)
    parser.add_argument("--shards", required=True, type=int)
    parser.add_argument("--baseline", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_shard(
        arguments.env, arguments.shard, arguments.shards, baseline=arguments.baseline
    )
