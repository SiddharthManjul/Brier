from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pyarrow.parquet as pq
import torch

from brier_sim import envs, storage
from brier_sim.runner import _SCHEMA, _guide_episode, run_episodes
from brier_sim.scenarios import build_scenarios, scenarios

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


def _upload_keyframes(parquet: Path, local: Path, remote_root: str) -> None:
    directories = {(row["keyframes_uri"], row["tier"]) for row in pq.read_table(parquet).to_pylist()
                   if row["keyframes_uri"] is not None}
    for value, tier in sorted(directories):
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("public", "keyframes"):
            raise ValueError("keyframes_uri must be a relative directory under public/keyframes")
        directory = (local / str(relative)).resolve()
        if not directory.is_relative_to(local.resolve()) or not directory.is_dir():
            raise ValueError("keyframes_uri directory is missing or outside BRIER_DATA")
        frames = sorted(directory.glob("frame-*.png"))
        expected = 1 if tier == "cheap" else 8
        if tier not in ("cheap", "expensive") or len(frames) != expected or any(not frame.is_file() or not frame.resolve().is_relative_to(directory)
                                   for frame in frames):
            raise ValueError("cheap episodes require one PNG; failed expensive episodes require eight recorded PNG keyframes")
        for frame in frames:
            _copy(str(frame), f"{remote_root}/{relative}/{frame.name}")


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
                                _upload_keyframes(destination, local, root)
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


def _timing_json(payload: dict, uri: str) -> None:
    destination = storage._location(uri)
    parent = destination.parent if isinstance(destination, Path) else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        source = Path(directory) / "timing.json"
        source.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
        storage._publish(source, uri)


def _timing_root() -> str:
    root = os.environ.get("BRIER_DATA", "").strip().rstrip("/")
    if not root:
        raise ValueError("BRIER_DATA is required for timing and sizing publication")
    return root


def _timing(env: str, out: str | None = None) -> dict:
    if env not in ("pusht", "aloha"):
        raise ValueError("timing env must be pusht or aloha")
    host = socket.gethostname()
    uri = out if out is not None else f"{_timing_root()}/timing/timing_{env}_{host}.json"
    if out is not None and not isinstance(storage._location(out), Path):
        raise ValueError("--out must be a local JSON file")
    pins = hashlib.sha256(Path(__file__).with_name("pins.json").read_bytes()).hexdigest()
    scenario = build_scenarios(env)[0]
    torch.set_num_threads(1)
    vector, policy = envs._load(env)
    samples: dict[str, list[float]] = {"cheap": [], "expensive": []}
    try:
        for tier, values in samples.items():
            for seed in scenario["seeds"][:3]:
                started = time.perf_counter()
                _guide_episode((vector, policy, None), env, scenario, tier, seed)
                values.append(time.perf_counter() - started)
    finally:
        vector.close()
    record = {"env": env, "host": host, "nproc": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
              "pins_sha256": pins, "github_actions": os.environ.get("GITHUB_ACTIONS") == "true",
              "samples_seconds": samples, "median_seconds": {tier: float(np.median(values)) for tier, values in samples.items()}}
    _timing_json(record, uri)
    return record


def _sizing() -> dict:
    root = _timing_root()
    location = storage._location(f"{root}/timing/")
    if isinstance(location, Path):
        records = [json.loads(path.read_text()) for path in sorted(location.glob("timing_*.json"))]
    else:
        blobs = storage._client().list_blobs(location[0], prefix=location[1].rstrip("/") + "/", retry=None)
        records = [json.loads(blob.download_as_bytes(retry=None)) for blob in sorted(blobs, key=lambda item: item.name)
                   if Path(blob.name).match("timing_*.json")]
    pins = hashlib.sha256(Path(__file__).with_name("pins.json").read_bytes()).hexdigest()
    measurements: dict[str, list[dict]] = {"pusht": [], "aloha": []}
    for record in records:
        if isinstance(record, dict) and record.get("github_actions") is False:
            continue
        if (not isinstance(record, dict) or record.get("github_actions") is not True or not isinstance(record.get("env"), str) or record["env"] not in measurements
                or record.get("pins_sha256") != pins or not isinstance(record.get("host"), str) or not record["host"]
                or type(record.get("nproc")) is not int or record["nproc"] < 1
                or not isinstance(record.get("samples_seconds"), dict) or not isinstance(record.get("median_seconds"), dict)):
            raise ValueError("invalid GitHub timing provenance or pins; re-time on GitHub Actions")
        for tier in ("cheap", "expensive"):
            values, median = record.get("samples_seconds", {}).get(tier), record.get("median_seconds", {}).get(tier)
            if (not isinstance(values, list) or len(values) != 3
                    or any(type(value) not in (int, float) or not 0 < value <= sys.float_info.max for value in values)
                    or type(median) not in (int, float) or not 0 < median <= sys.float_info.max
                    or not np.isclose(float(median), np.median(values), rtol=1e-12, atol=0)):
                raise ValueError("timing requires three positive finite samples and their consistent median per tier")
        measurements[record["env"]].append(record["median_seconds"])
    if any(not values for values in measurements.values()):
        raise ValueError("sizing requires actual GitHub Actions timing for both pusht and aloha")
    medians = {env: {tier: float(np.median([sample[tier] for sample in samples])) for tier in ("cheap", "expensive")}
               for env, samples in measurements.items()}
    if medians["pusht"]["expensive"] > 30:
        raise ValueError("PushT exceeds 30 s per expensive episode; adjust and record policy pins, then re-time all settings before sizing")
    def projection(changes: int, scenarios: int) -> dict:
        return {env: {"changes": changes, "scenarios": scenarios, "pairs": changes * scenarios, "median_seconds": times,
                      "cpu_hours": (changes * scenarios * (3 * times["cheap"] + 20 * times["expensive"]) + 16 * 50 * times["expensive"]) / 3600}
                for env, times in medians.items()}
    def wall(environments: dict) -> float:
        return round(sum(value["cpu_hours"] for value in environments.values()) / 80, 12)
    target = wall(projection(80, 16))
    changes, scenarios, notes = 80, 16, []
    if target > 20:
        changes, scenarios = 40, 12
        notes.append("Minimum viable data selected: 40 changes at 35% benign can cover at most eight retained families four times; pilot must resolve coverage before generation.")
    elif target > 10:
        changes = next((count for count in range(76, 47, -4) if wall(projection(count, 16)) <= 10), 48)
        if wall(projection(changes, 16)) > 10:
            notes.append("The guide's 48-change floor exceeds 10 hours; the projection retains that floor explicitly.")
    result = {"pins_sha256": pins, "capacity": {"github_actions_processes": 80, "spot_processes": 0, "total_processes": 80},
              "environments": projection(changes, scenarios), "target_wall_hours": target,
              "wall_hours": wall(projection(changes, scenarios)), "notes": notes}
    _timing_json(result, f"{root}/manifests/sizing.json")
    return result


def _timing_cli(arguments: list[str]) -> bool:
    if not arguments or arguments[0] not in ("timing", "sizing"):
        return False
    parser = argparse.ArgumentParser(prog=f"{__package__}.farm {arguments[0]}")
    if arguments[0] == "timing":
        parser.add_argument("--env", required=True, choices=("pusht", "aloha"))
        parser.add_argument("--out")
    args = parser.parse_args(arguments[1:])
    result = _timing(args.env, args.out) if arguments[0] == "timing" else _sizing()
    print(json.dumps(result["median_seconds"] if arguments[0] == "timing" else result, sort_keys=True))
    return True


if __name__ == "__main__" and not _timing_cli(sys.argv[1:]):
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
