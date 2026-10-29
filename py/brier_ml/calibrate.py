from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from brier_ml.evaluate import binary_metrics

_QUESTIONS = ("regressed", "failure_mode", "success_bucket")
_SEMANTICS = {
    "regressed": {"yes", "no"},
    "failure_mode": {"none", "stall/timeout", "overshoot/misalignment",
                     "grasp or drop failure", "collision", "oscillation"},
    "success_bucket": {"1", "2", "3", "4", "5"},
}
_SPLITS = {"train", "calibration", "test", "ood"}
_ID = re.compile(
    r"(c[0-9]{3})/[^/]+/(regressed|failure_mode|success_bucket)/(cheap|no_cheap)"
)
_Sample = tuple[tuple[float, ...], int, int]


def _distribution(probabilities: dict[str, float]) -> tuple[list[str], tuple[float, ...]]:
    if (
        not isinstance(probabilities, dict)
        or len(probabilities) < 2
        or any(not isinstance(key, str) or not key for key in probabilities)
    ):
        raise ValueError("probabilities must map at least two option letters to numbers")
    values = tuple(probabilities.values())
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value <= 1
        or not math.isfinite(value)
        for value in values
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
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
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


def fit_temperatures(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit three temperatures on calibration rows only within bounds [0.05, 20].

    Scored compiled rows supply id, split, question, options, label, probabilities.
    Reported metrics use calibration rows only; multiclass Brier sums over options.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list of scored compiled rows")
    required = {"id", "split", "question", "options", "label", "probabilities"}
    groups: dict[str, list[_Sample]] = {question: [] for question in _QUESTIONS}
    seen: set[str] = set()
    change_splits: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError("each scored row must contain all required fields")
        identifier = record["id"]
        match = _ID.fullmatch(identifier) if isinstance(identifier, str) else None
        if match is None or match.group(2) != record["question"]:
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
            not isinstance(options, dict)
            or not 2 <= len(options) <= 26
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
        keys, values = _distribution(record["probabilities"])
        if set(keys) != set(options):
            raise ValueError("probability keys must exactly match option letters")
        yes = -1
        if question == "regressed":
            yes = next(index for index, key in enumerate(keys) if options[key] == "yes")
        if split == "calibration":
            groups[question].append((values, keys.index(label), yes))
    if any(not samples for samples in groups.values()):
        raise ValueError("calibration rows are required for all three question types")
    temperatures: dict[str, float] = {}
    reports: dict[str, Any] = {}
    ratio = (math.sqrt(5) - 1) / 2
    for question, samples in groups.items():
        lower, upper = math.log(0.05), math.log(20.0)
        left = upper - ratio * (upper - lower)
        right = lower + ratio * (upper - lower)
        left_loss = _loss(samples, math.exp(left))
        right_loss = _loss(samples, math.exp(right))
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
            "count": len(samples),
            "before": _metrics(question, samples, 1.0),
            "after": _metrics(question, samples, temperature),
        }
    return {"temperatures": temperatures, "calibration": reports}
