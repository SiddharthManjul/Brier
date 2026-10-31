from __future__ import annotations

import hashlib
import json
import math
import re
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
_VALUES = {
    "regressed": ("yes", "no"),
    "failure_mode": ("none", "stall/timeout", "overshoot/misalignment",
                     "grasp or drop failure", "collision", "oscillation"),
    "success_bucket": ("1", "2", "3", "4", "5"),
}
_GUIDE_VALUES = {**_VALUES,
    "failure_mode": ("none", "stall or timeout", "overshoot or misalignment",
                     "grasp or drop", "collision", "oscillation"),
    "success_bucket": ("0-19%", "20-39%", "40-59%", "60-79%", "80-100%")}


def _compiled_rows(compiled: Path) -> list[dict[str, Any]]:
    rows, identities, splits = [], set(), {}
    fields = {"id", "split", "question", "options", "prompt", "label"}
    for number, line in enumerate(compiled.read_text().splitlines(), 1):
        row = json.loads(line)
        guide = isinstance(row, dict) and "pair_id" in row
        expected = fields | ({"pair_id", "change_id", "scenario_id", "env", "variant"} if guide else set())
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError(f"compiled line {number} has invalid fields")
        identity, question = row["id"], row["question"]
        pattern = (r"c[0-9]{3,}:(?:pusht|aloha(?:_transfer_cube)?)-s[0-9]{2,}:(?:regressed|failure_mode|success_bucket):(?:ev|noev)"
                   if guide else r"c[0-9]{3,}/(?:pusht|aloha|aloha_transfer_cube)-s[0-9]{2,}/(?:regressed|failure_mode|success_bucket)/(?:cheap|no_cheap)")
        if not isinstance(identity, str) or not re.fullmatch(pattern, identity):
            raise ValueError(f"compiled line {number} has invalid id")
        change, scenario, kind, variant = identity.split(":" if guide else "/")
        if guide and (row["pair_id"] != f"{change}:{scenario}" or row["change_id"] != change
                      or row["scenario_id"] != scenario or row["env"] != scenario.split("-s")[0].removesuffix("_transfer_cube")
                      or row["variant"] != variant or int(change[1:]) == 0
                      or int(scenario.rsplit("s", 1)[1]) == 0):
            raise ValueError(f"compiled metadata disagrees with id: {identity}")
        if (identity in identities or question != kind or not isinstance(row["split"], str)
                or row["split"] not in {"train", "calibration", "test", "ood"}
                or not isinstance(row["prompt"], str) or not row["prompt"]):
            raise ValueError(f"compiled line {number} is invalid or duplicate")
        if change in splits and splits[change] != row["split"]:
            raise ValueError(f"change {change} appears in multiple splits")
        values = (_GUIDE_VALUES if guide else _VALUES)[kind]
        options = row["options"]
        if guide:
            bucket = int.from_bytes(hashlib.sha256(change.encode()).digest()[:4], "big") % 10
            split = "train" if bucket < 6 else "calibration" if bucket < 8 else "test"
            if (not isinstance(options, list) or len(options) != len(values) or not all(isinstance(v, str) for v in options)
                    or (row["split"] != "ood" and row["split"] != split)
                    or (row["split"] != "train" and options != list(values))):
                raise ValueError(f"invalid compiled guide options/split: {identity}")
            options = dict(zip("ABCDEF", options))
            if (not row["prompt"].startswith("<system> ") or not row["prompt"].endswith("<answer>")
                    or any(marker not in row["prompt"] for marker in ("\n<state>\n", "\n<question> ", "\n<options> "))
                    or row["prompt"].split("\n<options> ")[1].split("\n")[0]
                    != "  ".join(f"{letter}) {text}" for letter, text in options.items())):
                raise ValueError(f"invalid compiled guide prompt: {identity}")
        if (not isinstance(options, dict) or not all(isinstance(v, str) for v in options.values())
                or set(options) != set("ABCDEF"[:len(values)]) or set(options.values()) != set(values)
                or not isinstance(row["label"], str) or row["label"] not in options):
            raise ValueError(f"compiled options/label disagree with contract: {identity}")
        identities.add(identity)
        splits[change] = row["split"]
        rows.append({**row, "options": options, "change": change, "scenario": scenario,
                     "evidence": {"ev": "cheap", "noev": "no_cheap"}.get(variant, variant), "guide": guide})
    if not rows:
        raise ValueError("compiled dataset contains no examples")
    return rows


class Dataset(TypedDict):
    ids: list[str]
    changes: list[str]
    splits: list[str]
    feature_names: list[str]
    features: NDArray[np.float64]
    labels: NDArray[np.int64]


def _number(value: Any, name: str) -> float:
    try:
        finite = type(value) in (float, int) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _change_features(card: dict[str, Any]) -> list[float]:
    if not isinstance(card, dict):
        raise TypeError("change card must be an object")
    guide = "desc" in card
    expected = {"change_id", "env", "families", "desc", "seed"} if guide else {"description", "families"}
    if set(card) != expected or not isinstance(card["desc" if guide else "description"], str):
        raise ValueError("change card must contain description and families")
    if guide and (not re.fullmatch(r"c[0-9]{3,}", str(card["change_id"]))
                  or int(card["change_id"][1:]) == 0 or card["env"] not in {"pusht", "aloha"}
                  or type(card["seed"]) is not int or card["seed"] < 0
                  or not card["desc"].strip() or card["desc"].splitlines() != [card["desc"]]):
        raise ValueError("invalid guide change metadata")
    entries = card["families"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 2:
        raise ValueError("change card must have one or two families")
    values = dict.fromkeys(_CHANGE_FEATURES, 0.0)
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"family", "params"}:
            raise ValueError("invalid change family entry")
        family, params = entry["family"], entry["params"]
        if not isinstance(family, str) or family not in _PARAMETERS or family in seen:
            raise ValueError("change families must be known and distinct")
        names = {"action_scale": ("s",), "action_bias": ("b",), "camera_shift": ("shift", "angle"),
                 "occlusion": ("fraction",)} if guide else {}
        if not isinstance(params, dict) or set(params) != set(names.get(family, _PARAMETERS[family])):
            raise ValueError(f"invalid parameters for {family}")
        params = {name: _number(value, name) for name, value in params.items()}
        if guide:
            for name, value in params.items():
                if ((name in {"k", "shift", "n_action_steps"} and type(entry["params"][name]) is not int)
                        or (name in {"sigma", "gain", "contrast", "b"} and value < 0)
                        or (name in {"k", "s", "shift", "n_action_steps", "fraction"} and value <= 0)
                        or (name in {"p", "fraction"} and not 0 <= value <= 1)):
                    raise ValueError(f"invalid parameters for {family}")
            if family == "action_scale":
                params = {"scale": params["s"]}
            elif family == "action_bias":
                params = {"drift": params["b"]}
            elif family == "camera_shift":
                params = {"dx": params["shift"], "dy": params["shift"], "angle": params["angle"]}
            elif family == "occlusion":
                # Guide squares encode normalized side fractions in the existing width/height slots.
                params = {"x": 0., "y": 0., "width": params["fraction"], "height": params["fraction"]}
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


def _scenario_features(env: str, params: dict[str, Any], guide: bool = False) -> list[float]:
    if env not in {"gym_pusht/PushT-v0", "gym_aloha/AlohaTransferCube-v0"}:
        raise ValueError(f"unsupported scenario environment: {env}")
    aloha = env == "gym_aloha/AlohaTransferCube-v0"
    if not isinstance(params, dict):
        raise TypeError("invalid scenario parameters")
    expected = {"pose_bin", "lighting", "camera_jitter"}
    if aloha or "physics_jitter" in params:
        expected.add("physics_jitter")
    if (
        set(params) != expected
        or type(params["pose_bin"]) is not int
        or params["pose_bin"] not in (range(1, 5) if guide else range(4))
        or params["lighting"] not in {"normal", "low"}
        or type(params["camera_jitter"]) is not bool
        or ("physics_jitter" in params and type(params["physics_jitter"]) is not bool)
    ):
        raise ValueError("invalid scenario parameters")
    return [
        float(aloha), float(params["pose_bin"] - int(guide)), float(params["lighting"] == "low"),
        float(params["camera_jitter"]), float(params.get("physics_jitter", False)),
    ]


def _examples(compiled: Path, labels: dict[tuple[str, str], sqlite3.Row]) -> list[dict[str, Any]]:
    result = []
    for row in _compiled_rows(compiled):
        change, scenario, question = row["change"], row["scenario"], row["question"]
        label = labels.get((change, scenario))
        if label is None or label["regressed"] not in (0, 1):
            raise ValueError(f"missing or invalid ledger label for {change}/{scenario}")
        answer = ("yes" if label["regressed"] else "no") if question == "regressed" else str(label[question])
        aliases = dict(zip(_GUIDE_VALUES[question], _VALUES[question], strict=True))
        actual = row["options"][row["label"]]
        if (aliases.get(actual) if row["guide"] else actual) != answer:
            raise ValueError(f"compiled options/label disagree with ledger: {row['id']}")
        if question == "regressed" and row["evidence"] == "cheap":
            result.append({**row, "target": int(label["regressed"])})
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
    base, targets, cards = [], [], []
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
        guide = "desc" in card
        if example["guide"] and episodes != 3:
            raise ValueError("guide cheap runs require exactly three episodes")
        if example["guide"] and ((guide and (card["change_id"] != change
                                  or card["env"] != example["env"]))
                                  or c["env"] != {"pusht": "gym_pusht/PushT-v0", "aloha": "gym_aloha/AlohaTransferCube-v0"}[example["env"]]):
            raise ValueError("compiled metadata disagrees with ledger change")
        ood = any(entry["family"] in {"occlusion", "chunk_len"} for entry in card["families"])
        if (ood and example["split"] != "ood") or (example["guide"] and (ood != (example["split"] == "ood"))):
            raise ValueError("occlusion and chunk_len changes must be held out as ood")
        cards.append(card)
        base.append(change_features +
                    _scenario_features(s["env"], json.loads(s["params_json"]), guide) +
                    [successes / episodes, rewards / episodes, baseline])
        targets.append(example["target"])
    features, y = np.asarray(base, dtype=np.float64), np.asarray(targets, dtype=np.int64)
    profiles = [{f"{entry['family']}.{name}": abs(value - (1. if name in {"s", "scale", "gain", "contrast"} else 0.))
                 for entry in card["families"] for name, value in entry["params"].items()} for card in cards]
    families = [{entry["family"] for entry in card["families"]} for card in cards]
    history = []
    for index, example in enumerate(examples):
        eligible = [i for i, other in enumerate(examples)
                    if other["split"] == "train" and other["scenario"] == example["scenario"]
                    and other["change"] != example["change"]]
        if len(eligible) < (1 if example["guide"] else 5):
            minimum = "one other training change" if example["guide"] else "five other training changes"
            raise ValueError(f"{minimum} required for {example['id']}")
        candidates = features[eligible, :len(_CHANGE_FEATURES)]
        scale = np.maximum(np.max(np.abs(candidates), axis=0), 1.0)
        distances = np.sum(((candidates - features[index, :len(_CHANGE_FEATURES)]) / scale) ** 2, axis=1)
        similarities = [0.] * len(eligible)
        if example["guide"]:
            names = set().union(profiles[index], *(profiles[i] for i in eligible))
            scales = {name: max(1., *(profiles[i].get(name, 0.) for i in eligible)) for name in names}
            distances = [sum(((profiles[index].get(name, 0.) - profiles[i].get(name, 0.)) / scales[name]) ** 2
                             for name in names) for i in eligible]
            similarities = [len(families[index] & families[i]) / len(families[index] | families[i]) for i in eligible]
        neighbors = sorted(range(len(eligible)),
                           key=lambda i: (-similarities[i], distances[i], examples[eligible[i]]["change"]))[:5]
        history.append([float(y[eligible].mean()),
                        float(y[[eligible[i] for i in neighbors]].mean())])
    return {"ids": [row["id"] for row in examples],
            "changes": [row["change"] for row in examples],
            "splits": [row["split"] for row in examples],
            "feature_names": FEATURE_NAMES.copy(),
            "features": np.column_stack([features, history]), "labels": y}
