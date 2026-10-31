from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression

from brier_ml.dataset import Dataset


def _matrix(features: NDArray[np.float64], count: int) -> NDArray[np.float64]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != count or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("features must be a nonempty finite matrix matching feature_names")
    return matrix


def _raw_probabilities(booster: dict[str, Any], features: NDArray[np.float64]) -> NDArray[np.float64]:
    if booster["objective"] != "binary sigmoid:1":
        raise ValueError("T1 requires a binary LightGBM model with unit sigmoid")
    scores = np.zeros(len(features), dtype=np.float64)
    for tree in booster["tree_info"]:
        for index, row in enumerate(features):
            node = tree["tree_structure"]
            while "leaf_value" not in node:
                if node["decision_type"] != "<=":
                    raise ValueError("T1 export only supports numeric tree splits")
                node = node["left_child"] if row[node["split_feature"]] <= node["threshold"] else node["right_child"]
            scores[index] += node["leaf_value"]
    exponentials = np.exp(-np.abs(scores))
    return np.where(scores >= 0, 1 / (1 + exponentials), exponentials / (1 + exponentials))


def predict_t1(artifact: dict[str, Any], features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate exported numeric LightGBM trees and isotonic knots without pickle."""
    if artifact.get("format_version") != 1:
        raise ValueError("unsupported T1 artifact format")
    matrix = _matrix(features, len(artifact["feature_names"]))
    x, y = np.asarray(artifact["isotonic"]["x"]), np.asarray(artifact["isotonic"]["y"])
    if (
        x.ndim != 1 or y.shape != x.shape or len(x) < 1
        or not np.isfinite(x).all() or not np.isfinite(y).all()
        or np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0)
        or np.any((y < 0) | (y > 1))
    ):
        raise ValueError("invalid isotonic knots")
    return np.interp(_raw_probabilities(artifact["booster"], matrix), x, y)


def train_t1(dataset: Dataset, output: Path) -> dict[str, Any]:
    """Fit T1 on train, isotonic on calibration, and export engine-readable JSON."""
    names, labels = dataset["feature_names"], np.asarray(dataset["labels"])
    features = _matrix(dataset["features"], len(names))
    splits = np.asarray(dataset["splits"])
    if (
        len(set(names)) != len(names) or labels.shape != (len(features),)
        or splits.shape != labels.shape or not np.isin(labels, [0, 1]).all()
        or len(dataset["changes"]) != len(features) or len(dataset["ids"]) != len(features)
        or len(set(dataset["ids"])) != len(features)
        or not np.isin(splits, ["train", "calibration", "test", "ood"]).all()
    ):
        raise ValueError("invalid T1 dataset arrays, feature names, or splits")
    change_splits: dict[str, str] = {}
    for change, split in zip(dataset["changes"], splits, strict=True):
        if change in change_splits and change_splits[change] != split:
            raise ValueError(f"change {change} appears in multiple splits")
        change_splits[change] = split
    train, calibration = splits == "train", splits == "calibration"
    if len(np.unique(labels[train])) != 2 or len(np.unique(labels[calibration])) != 2:
        raise ValueError("train and calibration must each contain both regression classes")
    model = LGBMClassifier(
        objective="binary", n_estimators=100, num_leaves=15, max_depth=4,
        min_child_samples=2, learning_rate=0.05, random_state=0, n_jobs=1,
        deterministic=True, force_col_wise=True, verbosity=-1,
    ).fit(features[train], labels[train], feature_name=names)
    probabilities = model.booster_.predict(features[calibration])
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(probabilities, labels[calibration])
    artifact = {
        "format_version": 1, "feature_names": names.copy(),
        "training_change_ids": sorted({change for change, split in zip(dataset["changes"], splits, strict=True)
                                       if split == "train"}),
        "calibration_change_ids": sorted({change for change, split in zip(dataset["changes"], splits, strict=True)
                                          if split == "calibration"}),
        "booster": model.booster_.dump_model(),
        "isotonic": {"x": isotonic.X_thresholds_.tolist(), "y": isotonic.y_thresholds_.tolist()},
    }
    artifact["model_version"] = "t1@" + hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    exported = predict_t1(artifact, features)
    expected = isotonic.predict(model.booster_.predict(features))
    if not np.allclose(exported, expected, atol=1e-12, rtol=1e-12):
        raise ValueError("exported T1 predictions disagree with LightGBM/isotonic")
    output.write_text(json.dumps(artifact, allow_nan=False, separators=(",", ":")) + "\n")
    return artifact
