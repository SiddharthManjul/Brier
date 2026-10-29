from __future__ import annotations

import argparse
import json
import math
import platform
import socket
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from brier_sim.runner import run_episodes
from brier_sim.scenarios import scenarios


def benchmark(env: str, output: Path) -> dict[str, Any]:
    """Measure cheap and expensive episode throughput from actual rollout records."""
    if env == "gym_pusht/PushT-v0":
        policy, change_id = "lerobot/diffusion_pusht", "c001"
    elif env == "gym_aloha/AlohaTransferCube-v0":
        policy, change_id = "lerobot/act_aloha_sim_transfer_cube_human", "c081"
    else:
        raise ValueError(f"Unsupported env: {env}")
    if not isinstance(output, Path):
        raise TypeError("output must be a pathlib.Path")
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError("Benchmark output parent directory must exist")
    scenario = scenarios(env)[0]
    result: dict[str, Any] = {
        "env": env,
        "policy": policy,
        "scenario_id": scenario["scenario_id"],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "total_episodes": 0,
        "tiers": {},
    }
    for tier, count in (("cheap", 3), ("expensive", 20)):
        path = run_episodes(
            env,
            policy,
            count,
            scenario["seeds"][0],
            change_id,
            scenario["scenario_id"],
            tier,
            change={
                "description": "Evaluate benign",
                "families": [{"family": "benign", "params": {}}],
            },
            scenario=scenario,
        )
        rows = pq.read_table(path).to_pylist()
        if len(rows) != count:
            raise ValueError(
                f"{tier} benchmark requires exactly {count} episode records"
            )
        durations = []
        for index, row in enumerate(rows):
            duration = row.get("wall_seconds")
            if (
                row.get("env") != env
                or row.get("policy") != policy
                or row.get("change_id") != change_id
                or row.get("scenario_id") != scenario["scenario_id"]
                or row.get("tier") != tier
                or row.get("seed") != scenario["seeds"][index]
                or type(duration) not in (int, float)
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ValueError(
                    f"Invalid {tier} benchmark episode record at index {index}"
                )
            durations.append(duration)
        total_seconds = math.fsum(durations)
        result["tiers"][tier] = {
            "episodes": len(rows),
            "total_seconds": total_seconds,
            "seconds_per_episode": total_seconds / len(rows),
            "parquet_path": str(path),
        }
        result["total_episodes"] += len(rows)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        required=True,
        choices=("gym_pusht/PushT-v0", "gym_aloha/AlohaTransferCube-v0"),
    )
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()
    benchmark(arguments.env, arguments.out)


if __name__ == "__main__":
    _main()
