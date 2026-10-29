from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score


class Strategy(TypedDict):
    selected: ArrayLike
    probabilities: NotRequired[ArrayLike]
    latencies_seconds: NotRequired[ArrayLike]
    costs_usd: NotRequired[ArrayLike]


def _labels(labels: ArrayLike) -> NDArray[np.int64]:
    values = np.asarray(labels)
    if values.ndim != 1 or not len(values) or not np.isin(values, [0, 1]).all():
        raise ValueError("labels must be a nonempty binary vector")
    return values.astype(np.int64)


def _vector(values: ArrayLike, count: int, name: str, lower: float, upper: float = np.inf) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count,) or not np.isfinite(array).all() or np.any((array < lower) | (array > upper)):
        raise ValueError(f"{name} must be a finite vector in [{lower}, {upper}] matching labels")
    return array


def _changes(change_ids: Sequence[str], count: int) -> list[str]:
    if len(change_ids) != count or any(not isinstance(value, str) or not value for value in change_ids):
        raise ValueError("change_ids must be nonempty strings matching labels")
    return sorted(set(change_ids))


def binary_metrics(labels: ArrayLike, probabilities: ArrayLike) -> dict[str, Any]:
    """Compute binary metrics and 15 quantile bins, keeping tied scores together.

    Empty quantile bins are omitted. NLL clips probabilities to [1e-15, 1-1e-15];
    AUROC is null when the evaluation data contains only one class.
    """
    y = _labels(labels)
    p = _vector(probabilities, len(y), "probabilities", 0, 1)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 16)))
    assignments = np.searchsorted(edges[1:-1], p, side="left")
    reliability = []
    ece = 0.0
    for index in np.unique(assignments):
        mask = assignments == index
        confidence, observed = float(p[mask].mean()), float(y[mask].mean())
        count = int(mask.sum())
        ece += count / len(y) * abs(confidence - observed)
        reliability.append({"count": count, "mean_probability": confidence,
                            "observed_regression": observed})
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "ece": float(ece), "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "reliability_bins": reliability,
    }


def evaluate_selection(change_ids: Sequence[str], labels: ArrayLike, selected: ArrayLike,
                       expensive_seconds: ArrayLike) -> dict[str, Any]:
    """Measure actual selected hours, pair misses, and detection at change level."""
    y = _labels(labels)
    changes = _changes(change_ids, len(y))
    mask = np.asarray(selected)
    if mask.shape != y.shape or mask.dtype != np.bool_:
        raise ValueError("selected must be an explicit boolean vector matching labels")
    seconds = _vector(expensive_seconds, len(y), "expensive_seconds", 0)
    if np.any(seconds <= 0):
        raise ValueError("expensive_seconds must be positive measured costs")
    positive = y == 1
    missed = positive & ~mask
    change_array = np.asarray(change_ids)
    regressed_changes = caught_changes = 0
    losses = []
    for change in changes:
        positives = positive & (change_array == change)
        count = int(positives.sum())
        if count:
            regressed_changes += 1
            caught_changes += int(np.any(positives & mask))
        losses.append(float(np.sum(missed & (change_array == change)) / count) if count else 0.0)
    return {
        "pairs": len(y), "selected_pairs": int(mask.sum()),
        "regressed_pairs": int(positive.sum()), "missed_regressions": int(missed.sum()),
        "missed_regression_fraction": float(missed.sum() / positive.sum()) if positive.any() else None,
        "mean_change_loss": float(np.mean(losses)),
        "regressed_changes": regressed_changes, "caught_changes": caught_changes,
        "change_recall": caught_changes / regressed_changes if regressed_changes else None,
        "eval_hours": float(seconds[mask].sum() / 3600),
        "run_all_hours": float(seconds.sum() / 3600),
        "eval_hours_saved": float(1 - seconds[mask].sum() / seconds.sum()),
    }


def threshold_curve(change_ids: Sequence[str], labels: ArrayLike, probabilities: ArrayLike,
                    expensive_seconds: ArrayLike) -> list[dict[str, Any]]:
    """Evaluate every distinct probability threshold, including selecting no pairs."""
    y = _labels(labels)
    p = _vector(probabilities, len(y), "probabilities", 0, 1)
    thresholds = [float(np.nextafter(p.max(), np.inf)), *sorted(set(p.tolist()), reverse=True)]
    return [{"threshold": threshold,
             **evaluate_selection(change_ids, y, p >= threshold, expensive_seconds)}
            for threshold in thresholds]


def write_report(output: Path, change_ids: Sequence[str], labels: ArrayLike,
                 expensive_seconds: ArrayLike, splits: Sequence[str],
                 strategies: Mapping[str, Strategy]) -> dict[str, Any]:
    """Write measured test/OOD strategies without fitting or claiming a guarantee."""
    y = _labels(labels)
    _changes(change_ids, len(y))
    split_array = np.asarray(splits)
    if split_array.shape != y.shape or not np.isin(split_array, ["test", "ood"]).all():
        raise ValueError("reports require exclusively held-out test/ood rows")
    membership: dict[str, str] = {}
    for change, split in zip(change_ids, splits, strict=True):
        if change in membership and membership[change] != split:
            raise ValueError(f"change {change} appears in multiple evaluation splits")
        membership[change] = split
    if not strategies:
        raise ValueError("at least one measured strategy is required")
    rows = {}
    for name, strategy in strategies.items():
        if not isinstance(name, str) or not name or "selected" not in strategy or set(strategy) - {
            "selected", "probabilities", "latencies_seconds", "costs_usd"
        }:
            raise ValueError("strategy requires a name, selected mask, and known measured fields")
        row = evaluate_selection(change_ids, y, strategy["selected"], expensive_seconds)
        if "probabilities" in strategy:
            probabilities = _vector(strategy["probabilities"], len(y), "probabilities", 0, 1)
            row.update(binary_metrics(y, probabilities))
            row["threshold_curve"] = threshold_curve(change_ids, y, probabilities, expensive_seconds)
            for split in ("test", "ood"):
                mask = split_array == split
                if mask.any():
                    row[f"{split}_metrics"] = binary_metrics(y[mask], probabilities[mask])
        if "latencies_seconds" in strategy:
            latency = _vector(strategy["latencies_seconds"], len(y), "latencies_seconds", 0)
            row["p50_latency_seconds"] = float(np.quantile(latency, .5))
            row["p95_latency_seconds"] = float(np.quantile(latency, .95))
        if "costs_usd" in strategy:
            costs = _vector(strategy["costs_usd"], len(y), "costs_usd", 0)
            row["total_cost_usd"] = float(costs.sum())
            row["cost_usd_per_change"] = float(costs.sum() / len(set(change_ids)))
        rows[name] = row
    report = {"evaluation_splits": sorted(set(splits)), "strategies": rows}
    output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    return report
