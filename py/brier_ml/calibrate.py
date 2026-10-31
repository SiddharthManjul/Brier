from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from brier_ml.evaluate import binary_metrics

_SEMANTICS = {
    "regressed": {"yes", "no"},
    "failure_mode": {"none", "stall/timeout", "overshoot/misalignment",
                     "grasp or drop failure", "collision", "oscillation"},
    "success_bucket": {"1", "2", "3", "4", "5"},
}
_SPLITS = {"train", "calibration", "test", "ood"}
_ID = re.compile(
    r"(c[0-9]{3,})/(?:pusht|aloha_transfer_cube)-s([0-9]{2,})/"
    r"(regressed|failure_mode|success_bucket)/(cheap|no_cheap)"
)
_Sample = tuple[tuple[float, ...], int, int]


def _distribution(probabilities: dict[str, float]) -> tuple[list[str], tuple[float, ...]]:
    if (
        not isinstance(probabilities, dict) or len(probabilities) < 2
        or any(not isinstance(key, str) or not key for key in probabilities)
    ):
        raise ValueError("probabilities must map at least two option letters to numbers")
    values = tuple(probabilities.values())
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not 0 < value <= 1 or not math.isfinite(value) for value in values
    ):
        raise ValueError("option probabilities must be finite and strictly positive")
    if not math.isclose(math.fsum(values), 1.0, rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError("option probabilities must sum to one")
    return list(probabilities), tuple(float(value) for value in values)


def _scale(values: tuple[float, ...], temperature: float) -> tuple[float, ...]:
    logs = tuple(math.log(value) for value in values)
    peak = max(logs)
    weights = tuple(math.exp((value - peak) / temperature) for value in logs)
    total = math.fsum(weights)
    return tuple(value / total for value in weights)


def apply_temperature(probabilities: dict[str, float], temperature: float) -> dict[str, float]:
    """Scale positive, normalized option probabilities without changing their keys."""
    if (
        isinstance(temperature, bool) or not isinstance(temperature, (int, float))
        or not 0 < temperature < math.inf
    ):
        raise ValueError("temperature must be a finite positive number")
    keys, values = _distribution(probabilities)
    return dict(zip(keys, _scale(values, temperature), strict=True))


def _loss(samples: list[_Sample], temperature: float) -> float:
    losses = []
    for values, target, _ in samples:
        logs = tuple(math.log(value) for value in values)
        peak = max(logs)
        normalizer = math.fsum(math.exp((value - peak) / temperature) for value in logs)
        losses.append((peak - logs[target]) / temperature + math.log(normalizer))
    return math.fsum(losses) / len(samples)


def _metrics(question: str, samples: list[_Sample], temperature: float) -> dict[str, float]:
    scaled = [_scale(values, temperature) for values, _, _ in samples]
    if question == "regressed":
        metrics = binary_metrics(
            [int(target == yes) for _, target, yes in samples],
            [values[yes] for values, (_, _, yes) in zip(scaled, samples, strict=True)],
        )
        return {key: float(metrics[key]) for key in ("ece", "brier", "nll")}
    confidence = np.asarray([max(values) for values in scaled])
    correct = np.asarray([
        int(np.argmax(values) == sample[1])
        for values, sample in zip(scaled, samples, strict=True)
    ])
    edges = np.unique(np.quantile(confidence, np.linspace(0, 1, 16)))
    assignments = np.searchsorted(edges[1:-1], confidence, side="left")
    ece = 0.0
    for index in np.unique(assignments):
        selected = assignments == index
        gap = abs(float(confidence[selected].mean() - correct[selected].mean()))
        ece += float(selected.mean()) * gap
    brier = math.fsum(
        math.fsum((value - int(index == sample[1])) ** 2 for index, value in enumerate(p))
        for p, sample in zip(scaled, samples, strict=True)
    ) / len(samples)
    return {"ece": ece, "brier": brier, "nll": _loss(samples, temperature)}


def fit_temperatures(records: list[dict[str, Any]] | Path | str) -> dict[str, Any]:
    """Fit temperatures from M6 calibration JSONL or return the legacy list report.

    Scored compiled rows require id, split, question, options, label, probabilities.
    Returned metrics describe calibration rows only; multiclass Brier sums options.
    """
    if isinstance(records, (str, Path)):
        return _fit_scores(_score_rows(records))
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list of scored compiled rows")
    required = {"id", "split", "question", "options", "label", "probabilities"}
    groups: dict[str, list[_Sample]] = {question: [] for question in _SEMANTICS}
    seen: set[str] = set()
    change_splits: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError("each scored row must contain all required fields")
        identifier = record["id"]
        match = _ID.fullmatch(identifier) if isinstance(identifier, str) else None
        if (
            match is None or match.group(3) != record["question"]
            or int(match.group(1)[1:]) == 0 or int(match.group(2)) == 0
        ):
            raise ValueError("row id must match its question and compiled id format")
        if identifier in seen:
            raise ValueError(f"duplicate scored row: {identifier}")
        seen.add(identifier)
        split = record["split"]
        if not isinstance(split, str) or split not in _SPLITS:
            raise ValueError("split must be train, calibration, test, or ood")
        change = match.group(1)
        if change in change_splits and change_splits[change] != split:
            raise ValueError(f"change appears in multiple splits: {change}")
        change_splits[change] = split
        options = record["options"]
        if (
            not isinstance(options, dict) or not 2 <= len(options) <= 26
            or any(not isinstance(key, str) for key in options)
            or sorted(options) != [chr(65 + index) for index in range(len(options))]
            or any(not isinstance(value, str) or not value for value in options.values())
            or len(set(options.values())) != len(options)
        ):
            raise ValueError("options must map consecutive letters to distinct meanings")
        question = record["question"]
        if set(options.values()) != _SEMANTICS[question]:
            raise ValueError(f"{question} options do not match Rust compiled semantics")
        label = record["label"]
        if not isinstance(label, str) or label not in options:
            raise ValueError("label must be one of the supplied option letters")
        keys, _ = _distribution(record["probabilities"])
        if set(keys) != set(options):
            raise ValueError("probability keys must exactly match option letters")
        keys = sorted(options, key=options.__getitem__)
        values = tuple(float(record["probabilities"][key]) for key in keys)
        yes = next((i for i, key in enumerate(keys) if options[key] == "yes"), -1)
        if split == "calibration":
            groups[question].append((values, keys.index(label), yes))
    if any(not samples for samples in groups.values()):
        raise ValueError("calibration rows are required for all three question types")
    temperatures: dict[str, float] = {}
    reports: dict[str, Any] = {}
    ratio = (math.sqrt(5) - 1) / 2
    for question, samples in groups.items():
        lower, upper = math.log(0.05), math.log(20.0)
        left, right = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
        left_loss, right_loss = _loss(samples, math.exp(left)), _loss(samples, math.exp(right))
        for _ in range(80):
            if left_loss <= right_loss:
                upper, right, right_loss = right, left, left_loss
                left = upper - ratio * (upper - lower)
                left_loss = _loss(samples, math.exp(left))
            else:
                lower, left, left_loss = left, right, right_loss
                right = lower + ratio * (upper - lower)
                right_loss = _loss(samples, math.exp(right))
        candidates = (0.05, 1.0, 20.0, math.exp((lower + upper) / 2))
        temperature = min(candidates, key=lambda value: _loss(samples, value))
        temperatures[question] = temperature
        reports[question] = {
            "count": len(samples), "before": _metrics(question, samples, 1.0),
            "after": _metrics(question, samples, temperature),
        }
    return {"temperatures": temperatures, "calibration": reports}


def _score_rows(path: Path | str) -> list[dict]:
    from brier_ml.glimmer_data import _OPTIONS

    rows, seen, changes, sources = [], set(), {}, set()
    fields = {"id", "pair_id", "split", "question", "variant", "options", "label",
              "logits", "probs", "model", "revision"}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) not in (fields, fields | {"temperature"}):
            raise ValueError("scores must contain exactly the M6 fields and optional temperature")
        match = re.fullmatch(r"(c[0-9]{3,}):((?:pusht|aloha)-s(?:0[1-9]|1[0-6])):"
                             r"(regressed|failure_mode|success_bucket):(ev|noev)", str(row["id"]))
        if (match is None or int(match[1][1:]) == 0 or row["id"] in seen
                or row["pair_id"] != f"{match[1]}:{match[2]}"
                or row["question"] != match[3] or row["variant"] != match[4]
                or not isinstance(row["split"], str) or row["split"] not in _SPLITS):
            raise ValueError("invalid, inconsistent or duplicate score identity/split")
        question, choices = row["question"], row["options"]
        if (not isinstance(choices, list) or any(not isinstance(v, str) for v in choices)
                or len(choices) != len(_OPTIONS[question]) or set(choices) != set(_OPTIONS[question])
                or not isinstance(row["label"], str) or len(row["label"]) != 1
                or row["label"] not in "ABCDEF"[:len(choices)]):
            raise ValueError("invalid score option semantics or label")
        values, probs = row["logits"], row["probs"]
        if (not isinstance(values, list) or len(values) != 6
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in values)
                or not isinstance(probs, dict) or set(probs) != set("ABCDEF"[:len(choices)])
                or any(type(v) not in (float, int) or not 0 <= v <= 1 for v in probs.values())
                or not math.isclose(math.fsum(probs.values()), 1, abs_tol=1e-6)):
            raise ValueError("scores require six finite logits and normalized valid-option probs")
        if (not isinstance(row["model"], str) or not row["model"]
                or not isinstance(row["revision"], str)
                or re.fullmatch(r"[0-9a-f]{40}", row["revision"]) is None
                or (match[1] in changes and changes[match[1]] != row["split"])):
            raise ValueError("scores require model/revision provenance and change-disjoint splits")
        if "temperature" in row and (type(row["temperature"]) not in (int, float)
                                      or not 0.05 <= row["temperature"] <= 20):
            raise ValueError("score temperature must be in [0.05, 20]")
        seen.add(row["id"])
        changes[match[1]] = row["split"]
        sources.add((row["model"], row["revision"]))
        rows.append(row)
    if not rows or len(sources) != 1:
        raise ValueError("score file must be nonempty and have one model/revision")
    return rows


def _fit_scores(rows: list[dict]) -> dict[str, float]:
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    from brier_ml.glimmer_data import _OPTIONS

    if any(row["split"] != "calibration" or "temperature" in row for row in rows):
        raise ValueError("temperature fitting requires only unscaled calibration scores")
    result = {}
    for question, options in _OPTIONS.items():
        selected = [row for row in rows if row["question"] == question]
        if not selected:
            raise ValueError("calibration rows are required for all three question types")
        logits = np.asarray([row["logits"][:len(options)] for row in selected], dtype=float)
        labels = np.asarray(["ABCDEF".index(row["label"]) for row in selected])

        def loss(log_t: float, logits: np.ndarray = logits, labels: np.ndarray = labels) -> float:
            z = logits / math.exp(log_t)
            return float((logsumexp(z, axis=1) - z[np.arange(len(labels)), labels]).mean())

        fit = minimize_scalar(loss, bounds=(math.log(.05), math.log(20)), method="bounded")
        if not fit.success or not math.isfinite(fit.fun):
            raise ValueError(f"temperature optimization failed for {question}")
        result[question] = math.exp(min((0.0, math.log(.05), math.log(20), fit.x), key=loss))
    return result


def apply(probs_jsonl: Path | str, temps: dict[str, float]) -> Path:
    """Recompute valid-option probabilities from raw logits into sibling probs_temp."""
    from scipy.special import softmax

    from brier_ml.glimmer_data import _OPTIONS

    if (set(temps) != set(_OPTIONS) or any(type(t) not in (int, float)
                                          or not .05 <= t <= 20 for t in temps.values())):
        raise ValueError("temperatures must name all three questions with values in [0.05, 20]")
    rows = _score_rows(probs_jsonl)
    for row in rows:
        t, k = temps[row["question"]], len(row["options"])
        p = softmax(np.asarray(row["logits"][:k], dtype=float) / t)
        row.update(probs=dict(zip("ABCDEF"[:k], p.tolist(), strict=True)), temperature=t)
    destination = Path(probs_jsonl).parent.parent / "probs_temp" / Path(probs_jsonl).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    return destination


def metrics(probs_jsonl: Path | str) -> dict:
    """Report pooled and ev/noev metrics, mapping shuffled options by their semantics."""
    from brier_ml import losses
    from brier_ml.glimmer_data import _OPTIONS

    rows, result = _score_rows(probs_jsonl), {}
    for question, options in _OPTIONS.items():
        variants = {}
        for variant in ("ev", "noev", "both"):
            selected = [row for row in rows if row["question"] == question
                        and (variant == "both" or row["variant"] == variant)]
            if not selected:
                variants[variant] = {"count": 0}
                continue
            p = np.asarray([[row["probs"]["ABCDEF"[row["options"].index(option)]]
                             for option in options] for row in selected])
            y = np.asarray([options.index(row["options"]["ABCDEF".index(row["label"])]) for row in selected])
            values = {"count": len(selected), "nll": losses.nll(p, y),
                      "brier": losses.brier(p, y), "accuracy": losses.accuracy(p, y)}
            if question == "regressed":
                yes = options.index("yes")
                values.update(ece=losses.ece_equal_mass(p[:, yes], y == yes),
                              auroc=losses.auroc(p[:, yes], y == yes))
            elif question == "success_bucket":
                values["rps"] = losses.rps(p, y)
            variants[variant] = values
        result[question] = {**variants["both"], "variants": variants}
    return result


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    paths = {split: args.model_dir / "probs" / f"{split}.jsonl"
             for split in ("calibration", "test", "ood")}
    datasets = {split: _score_rows(path) for split, path in paths.items()}
    all_rows = [row for rows in datasets.values() for row in rows]
    if any(any(row["split"] != split for row in rows) for split, rows in datasets.items()):
        raise ValueError("each probability filename must match every row's split")
    changes = {}
    for row in all_rows:
        changes.setdefault(row["pair_id"].split(":")[0], set()).add(row["split"])
    source = all_rows[0]
    if (source["model"] not in (args.name, "glimmer-ft-" + args.name)
            or len({(row["model"], row["revision"]) for row in all_rows}) != 1
            or any(len(splits) != 1 for splits in changes.values())):
        raise ValueError("all scoring splits must share provenance and have disjoint changes")
    temps = fit_temperatures(paths["calibration"])
    report = {"model": source["model"], "revision": source["revision"], "temperatures": temps, "splits": {}}
    for split, path in paths.items():
        scaled = apply(path, temps)
        if split != "calibration":
            report["splits"][split] = {"before": metrics(path), "after": metrics(scaled)}
    (args.model_dir / "temps.json").write_text(json.dumps(temps, allow_nan=False) + "\n", encoding="utf-8")
    (args.model_dir / "metrics.json").write_text(json.dumps(report, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    _main()
