from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

_PARAMETERS = {
    "obs_noise": ("sigma",),
    "action_latency": ("k",),
    "action_scale": ("scale",),
    "action_bias": ("drift",),
    "camera_shift": ("dx", "dy", "angle"),
    "occlusion": ("x", "y", "width", "height"),
    "brightness": ("gain", "contrast"),
    "frame_drop": ("p",),
    "chunk_len": ("n_action_steps",),
    "weight_noise": ("sigma",),
    "benign": (),
}
_CHANGE_FEATURES = [f"family_{family}" for family in _PARAMETERS] + [
    f"{family}_{parameter}"
    for family, parameters in _PARAMETERS.items()
    for parameter in parameters
]
FEATURE_NAMES = _CHANGE_FEATURES + [
    "env_is_aloha", "pose_bin", "lighting_low", "camera_jitter", "physics_jitter",
    "cheap_success", "cheap_mean_reward", "baseline_success",
    "scenario_fragility", "similar_regressed_rate",
]


class Dataset(TypedDict):
    ids: list[str]
    changes: list[str]
    splits: list[str]
    feature_names: list[str]
    features: NDArray[np.float64]
    labels: NDArray[np.int64]


def _number(value: Any, name: str) -> float:
    if type(value) not in (float, int) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _change_features(card: dict[str, Any]) -> list[float]:
    if set(card) != {"description", "families"} or not isinstance(card["description"], str):
        raise ValueError("change card must contain description and families")
    entries = card["families"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 2:
        raise ValueError("change card must have one or two families")
    values = dict.fromkeys(_CHANGE_FEATURES, 0.0)
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"family", "params"}:
            raise ValueError("invalid change family entry")
        family, params = entry["family"], entry["params"]
        if family not in _PARAMETERS or family in seen:
            raise ValueError("change families must be known and distinct")
        if not isinstance(params, dict) or set(params) != set(_PARAMETERS[family]):
            raise ValueError(f"invalid parameters for {family}")
        seen.add(family)
        values[f"family_{family}"] = 1.0
        for name, value in params.items():
            number = _number(value, name)
            values[f"{family}_{name}"] = abs(
                number - (1.0 if name in {"scale", "gain", "contrast"} else 0.0)
            )
    if "benign" in seen and len(seen) != 1:
        raise ValueError("benign must be the only change family")
    return list(values.values())


def _scenario_features(env: str, params: dict[str, Any]) -> list[float]:
    if env not in {"gym_pusht/PushT-v0", "gym_aloha/AlohaTransferCube-v0"}:
        raise ValueError(f"unsupported scenario environment: {env}")
    aloha = env == "gym_aloha/AlohaTransferCube-v0"
    expected = {"pose_bin", "lighting", "camera_jitter"}
    if aloha or "physics_jitter" in params:
        expected.add("physics_jitter")
    if (
        set(params) != expected
        or type(params["pose_bin"]) is not int
        or params["pose_bin"] not in range(4)
        or params["lighting"] not in {"normal", "low"}
        or type(params["camera_jitter"]) is not bool
        or ("physics_jitter" in params and type(params["physics_jitter"]) is not bool)
    ):
        raise ValueError("invalid scenario parameters")
    return [
        float(aloha), float(params["pose_bin"]), float(params["lighting"] == "low"),
        float(params["camera_jitter"]), float(params.get("physics_jitter", False)),
    ]


def _examples(compiled: Path, labels: dict[tuple[str, str], sqlite3.Row]) -> list[dict[str, Any]]:
    result, seen_ids, change_splits = [], set(), {}
    for line_number, line in enumerate(compiled.read_text().splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {
            "id", "split", "question", "options", "prompt", "label"
        }:
            raise ValueError(f"compiled line {line_number} has invalid fields")
        if not isinstance(row["id"], str) or len(row["id"].split("/")) != 4:
            raise ValueError(f"compiled line {line_number} has invalid id")
        change, scenario, question, evidence = row["id"].split("/")
        if (
            question != row["question"]
            or question not in {"regressed", "failure_mode", "success_bucket"}
            or evidence not in {"cheap", "no_cheap"}
            or row["split"] not in {"train", "calibration", "test", "ood"}
            or not isinstance(row["prompt"], str) or not row["prompt"]
            or row["id"] in seen_ids
        ):
            raise ValueError(f"compiled line {line_number} is invalid or duplicate")
        seen_ids.add(row["id"])
        if change in change_splits and change_splits[change] != row["split"]:
            raise ValueError(f"change {change} appears in multiple splits")
        change_splits[change] = row["split"]
        label = labels.get((change, scenario))
        if label is None or label["regressed"] not in (0, 1):
            raise ValueError(f"missing or invalid ledger label for {change}/{scenario}")
        options = row["options"]
        expected_values = {
            "regressed": {"yes", "no"},
            "failure_mode": {"none", "stall/timeout", "overshoot/misalignment",
                             "grasp or drop failure", "collision", "oscillation"},
            "success_bucket": {"1", "2", "3", "4", "5"},
        }[question]
        expected_label = (
            "yes" if label["regressed"] else "no"
        ) if question == "regressed" else str(label[question])
        if (
            not isinstance(options, dict) or not isinstance(row["label"], str)
            or set(options.values()) != expected_values
            or len(options) != len(expected_values)
            or any(key not in "ABCDEF" or len(key) != 1 for key in options)
            or options.get(row["label"]) != expected_label
        ):
            raise ValueError(f"compiled options/label disagree with ledger: {row['id']}")
        if question == "regressed" and evidence == "cheap":
            result.append({**row, "change": change, "scenario": scenario,
                           "target": int(label["regressed"])})
    if not result:
        raise ValueError("compiled dataset contains no regressed/cheap examples")
    return result


def load_dataset(ledger: Path, compiled: Path) -> Dataset:
    """Read authoritative labels and numeric features with training-only history."""
    with sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        changes = {r["change_id"]: r for r in connection.execute("SELECT * FROM changes")}
        scenarios = {r["scenario_id"]: r for r in connection.execute("SELECT * FROM scenarios")}
        labels = {(r["change_id"], r["scenario_id"]): r
                  for r in connection.execute("SELECT * FROM labels")}
        runs: dict[tuple[str, str], list[sqlite3.Row]] = {}
        for run in connection.execute("SELECT * FROM runs WHERE tier = 'cheap'"):
            runs.setdefault((run["change_id"], run["scenario_id"]), []).append(run)
        examples = _examples(compiled, labels)
    base, targets = [], []
    for example in examples:
        change, scenario = example["change"], example["scenario"]
        if change not in changes or scenario not in scenarios or (change, scenario) not in runs:
            raise ValueError(f"missing ledger change/scenario/cheap run for {example['id']}")
        c, s = changes[change], scenarios[scenario]
        if c["env"] != s["env"]:
            raise ValueError(f"change/scenario environment mismatch for {example['id']}")
        baseline = _number(s["baseline_success"], "baseline_success")
        if not 0 <= baseline <= 1:
            raise ValueError("baseline_success must be a probability")
        episodes = successes = 0
        rewards = 0.0
        for run in runs[(change, scenario)]:
            n, count = run["episodes"], run["successes"]
            if type(n) is not int or type(count) is not int or n <= 0 or not 0 <= count <= n:
                raise ValueError("cheap runs require valid episode and success counts")
            episodes += n
            successes += count
            rewards += n * _number(run["mean_reward"], "mean_reward")
        card = json.loads(c["card_json"])
        change_features = _change_features(card)
        if any(entry["family"] in {"occlusion", "chunk_len"} for entry in card["families"]) and example["split"] != "ood":
            raise ValueError("occlusion and chunk_len changes must be held out as ood")
        base.append(change_features +
                    _scenario_features(s["env"], json.loads(s["params_json"])) +
                    [successes / episodes, rewards / episodes, baseline])
        targets.append(example["target"])
    features, y = np.asarray(base, dtype=np.float64), np.asarray(targets, dtype=np.int64)
    history = []
    for index, example in enumerate(examples):
        eligible = [i for i, other in enumerate(examples)
                    if other["split"] == "train" and other["scenario"] == example["scenario"]
                    and other["change"] != example["change"]]
        if len(eligible) < 5:
            raise ValueError(f"five other training changes required for {example['id']}")
        candidates = features[eligible, :len(_CHANGE_FEATURES)]
        scale = np.maximum(np.max(np.abs(candidates), axis=0), 1.0)
        distances = np.sum(((candidates - features[index, :len(_CHANGE_FEATURES)]) / scale) ** 2, axis=1)
        neighbors = sorted(range(len(eligible)),
                           key=lambda i: (distances[i], examples[eligible[i]]["change"]))[:5]
        history.append([float(y[eligible].mean()),
                        float(y[[eligible[i] for i in neighbors]].mean())])
    return {"ids": [row["id"] for row in examples],
            "changes": [row["change"] for row in examples],
            "splits": [row["split"] for row in examples],
            "feature_names": FEATURE_NAMES.copy(),
            "features": np.column_stack([features, history]), "labels": y}
