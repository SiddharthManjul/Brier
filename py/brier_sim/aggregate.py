from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq

from brier_sim import storage


def _read(uri: str) -> bytes:
    location = storage._location(uri)
    return location.read_bytes() if isinstance(location, Path) else storage._blob(location).download_as_bytes(retry=None)


def _plan(root: str) -> tuple[list[dict], list[dict], list[tuple], dict]:
    pins_bytes = Path(__file__).with_name("pins.json").read_bytes()
    pins, sizing = json.loads(pins_bytes), json.loads(_read(f"{root}/manifests/sizing.json"))
    if sizing.get("pins_sha256") != hashlib.sha256(pins_bytes).hexdigest():
        raise ValueError("sizing must use the current policy pins")
    changes, scenarios, keys = [], [], []
    for env in ("pusht", "aloha"):
        selected = sizing["environments"][env]
        if (type(selected.get("changes")) is not int or not 1 <= selected["changes"] <= 80
                or type(selected.get("scenarios")) is not int or not 1 <= selected["scenarios"] <= 16):
            raise ValueError("sizing requires positive change and scenario counts within the frozen library")
        cards = [json.loads(line) for line in _read(f"{root}/manifests/changes_{env}.jsonl").decode().splitlines() if line.strip()]
        start = 1 if env == "pusht" else 81
        expected = {f"c{index:03}" for index in range(start, start + selected["changes"])}
        if (len(cards) != selected["changes"] or any(not isinstance(card, dict) or card.get("env") != env
                                                  or not isinstance(card.get("change_id"), str) for card in cards)
                or {card.get("change_id") for card in cards} != expected):
            raise ValueError("change manifest must match the sizing count, environment and unique production IDs")
        definitions = json.loads(_read(f"{root}/manifests/scenarios_{env}.json"))
        if (not isinstance(definitions, list) or len(definitions) != 16
                or any(not isinstance(item, dict) or not isinstance(item.get("scenario_id"), str) for item in definitions)
                or {item.get("scenario_id") for item in definitions} != {f"{env}-s{index:02}" for index in range(1, 17)}):
            raise ValueError("scenario manifest must contain all 16 unique frozen scenario IDs")
        definitions.sort(key=lambda item: item["scenario_id"])
        if (any(not isinstance(item.get("params"), dict) or not isinstance(item.get("seeds"), list)
                or len(item["seeds"]) != 50 for item in definitions)):
            raise ValueError("scenario manifests require params and 50 disjoint nonnegative seeds per scenario")
        seeds = [seed for scenario in definitions for seed in scenario["seeds"]]
        if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != 800:
            raise ValueError("scenario manifests require params and 50 disjoint nonnegative seeds per scenario")
        cards.sort(key=lambda item: item["change_id"])
        changes.extend(cards)
        scenarios.extend({**scenario, "env": env} for scenario in definitions)
        keys.extend((env, "base", scenario, "baseline") for scenario in definitions)
        keys.extend((env, card["change_id"], scenario, tier) for card in cards
                    for scenario in definitions[:selected["scenarios"]] for tier in ("cheap", "expensive"))
    return changes, scenarios, keys, pins


def _rows(uri: str, key: tuple, pins: dict) -> list[dict]:
    env, change, scenario, tier = key
    table = pq.ParquetFile(BytesIO(_read(uri))).read()
    if not table.schema.equals(storage._SCHEMA, check_metadata=False):
        raise ValueError(f"episode schema differs from the 28-column contract: {uri}")
    rows = sorted(table.to_pylist(), key=lambda item: item["episode_idx"])
    count = {"baseline": 50, "cheap": 3, "expensive": 20}[tier]
    policy = pins["policies"][env]
    if (len(rows) != count or [row["episode_idx"] for row in rows] != list(range(count))
            or [row["seed"] for row in rows] != scenario["seeds"][:count]
            or any(row["env"] != env or row["change_id"] != change or row["scenario_id"] != scenario["scenario_id"]
                   or row["tier"] != tier or row["policy_repo"] != policy["repo_id"]
                   or row["policy_revision"] != policy["revision"] or row["lerobot_version"] != pins["lerobot_version"]
                   or any(row[field.name] is None for field in storage._SCHEMA if not field.nullable) for row in rows)):
        raise ValueError(f"episode key has incomplete or mismatched rows, seeds or provenance: {uri}")
    pusht = ("coverage_final", "coverage_max", "block_disp_px")
    aloha = ("touched_right", "touched_left", "lifted", "table_contact_steps")
    required, absent = (pusht, aloha) if env == "pusht" else (aloha, pusht)
    for row in rows:
        if (any(row[name] is None for name in required) or any(row[name] is not None for name in absent)
                or any(row[field.name] is not None and not math.isfinite(row[field.name]) for field in storage._SCHEMA
                       if field.type == storage._TYPES["f"])
                or not 1 <= row["steps"] <= row["max_steps"] or row["wall_seconds"] < 0
                or not 0 <= row["action_flip_rate"] <= 1
                or not 0 <= row["final_reward"] <= row["max_reward"] <= (1 if env == "pusht" else 4)
                or row["reward_sum"] < row["max_reward"]
                or env == "pusht" and (not 0 <= row["coverage_final"] <= row["coverage_max"] <= 1 or row["block_disp_px"] < 0)
                or env == "aloha" and not 0 <= row["table_contact_steps"] <= row["steps"]):
            raise ValueError(f"episode key has invalid scalar ranges, finite values or environment nulls: {uri}")
    return rows


def _aggregate(out: Path) -> None:
    root = os.environ.get("BRIER_DATA", "").strip().rstrip("/")
    if not root:
        raise ValueError("BRIER_DATA is required")
    changes, scenarios, keys, pins = _plan(root)
    planned = [(storage.key_uri(env, change, scenario["scenario_id"], tier), (env, change, scenario, tier))
               for env, change, scenario, tier in keys]
    missing = [uri for uri, _ in planned if not storage.exists(uri)]
    if missing:
        raise FileNotFoundError("Missing planned episode keys:\n" + "\n".join(missing))
    with tempfile.TemporaryDirectory() as directory:
        staging, rates = Path(directory), {}
        with (staging / "episodes.jsonl").open("w") as stream:
            for uri, key in planned:
                rows = _rows(uri, key, pins)
                if key[3] == "baseline":
                    rates[key[2]["scenario_id"]] = sum(row["success"] for row in rows) / 50
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        for name, rows in (("changes", changes), ("scenarios", [{**item, "baseline_success": rates[item["scenario_id"]]} for item in scenarios])):
            with (staging / f"{name}.jsonl").open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        out.mkdir(parents=True, exist_ok=True)
        for name in ("episodes", "changes", "scenarios"):
            destination = out / f"{name}.jsonl"
            shutil.copyfile(staging / destination.name, destination)


def _main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate complete rollout keys for brier label")
    parser.add_argument("--out", type=Path, default=Path("data"))
    _aggregate(parser.parse_args(arguments).out)


if __name__ == "__main__":
    _main()
