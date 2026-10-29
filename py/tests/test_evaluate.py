from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from brier_ml.evaluate import (
    binary_metrics,
    evaluate_selection,
    threshold_curve,
    write_report,
)


def test_binary_metrics_known_values() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [.1, .2, .8, .9])
    assert metrics["brier"] == pytest.approx(.025)
    assert metrics["ece"] == pytest.approx(.15)
    assert metrics["nll"] == pytest.approx(-(math.log(.9) + math.log(.8)) / 2)
    assert metrics["auroc"] == 1
    assert sum(row["count"] for row in metrics["reliability_bins"]) == 4


def test_equal_mass_bins_and_ties_are_order_invariant() -> None:
    metrics = binary_metrics(np.tile([0, 1], 30), np.full(60, .4))
    assert metrics["ece"] == pytest.approx(.1)
    assert metrics["auroc"] == .5
    assert metrics["reliability_bins"] == [{"count": 60, "mean_probability": pytest.approx(.4), "observed_regression": .5}]
    equal_mass = binary_metrics(np.zeros(60), np.linspace(0, 1, 60))
    assert len(equal_mass["reliability_bins"]) == 15
    assert {row["count"] for row in equal_mass["reliability_bins"]} == {4}
    assert equal_mass["auroc"] is None


def test_probability_boundaries_have_finite_documented_nll() -> None:
    metrics = binary_metrics([1, 0], [0, 1])
    assert metrics["brier"] == 1
    assert metrics["auroc"] == 0
    assert math.isfinite(metrics["nll"])
    assert metrics["nll"] > 34


def test_selection_uses_measured_seconds_and_change_detection() -> None:
    selected = evaluate_selection(["a", "a", "b", "b"], [1, 1, 1, 0],
                                  [True, False, False, True], [3600, 7200, 3600, 3600])
    assert selected["eval_hours"] == 2
    assert selected["run_all_hours"] == 5
    assert selected["eval_hours_saved"] == .6
    assert selected["missed_regression_fraction"] == pytest.approx(2 / 3)
    assert selected["mean_change_loss"] == .75
    assert selected["change_recall"] == .5
    assert selected["regressed_changes"] == 2
    assert selected["caught_changes"] == 1


def test_selection_without_regressions_reports_undefined_recall() -> None:
    metrics = evaluate_selection(["a", "b"], [0, 0], [False, False], [1, 2])
    assert metrics["change_recall"] is None
    assert metrics["missed_regression_fraction"] is None
    assert metrics["mean_change_loss"] == 0
    assert metrics["eval_hours_saved"] == 1


def test_threshold_curve_keeps_ties_and_includes_run_none_run_all() -> None:
    curve = threshold_curve(["a", "b", "c"], [1, 0, 1], [.9, .9, .1], [10, 20, 30])
    assert [row["selected_pairs"] for row in curve] == [0, 2, 3]
    assert curve[0]["missed_regression_fraction"] == 1
    assert curve[-1]["missed_regression_fraction"] == 0
    assert curve[-1]["eval_hours_saved"] == 0


def test_report_contains_only_available_measured_strategies(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    report = write_report(output, ["a", "a", "b", "b"], [0, 1, 0, 1], [10, 20, 30, 40],
                          ["test", "test", "ood", "ood"], {
                              "run_all": {"selected": [True] * 4},
                              "T1": {"selected": [False, True, False, True],
                                     "probabilities": [.1, .9, .2, .8],
                                     "latencies_seconds": [.1, .2, .3, .4],
                                     "costs_usd": [.01] * 4},
                          })
    assert json.loads(output.read_text()) == report
    assert set(report["strategies"]) == {"run_all", "T1"}
    assert "ece" not in report["strategies"]["run_all"]
    assert "guarantee" not in output.read_text()
    t1 = report["strategies"]["T1"]
    assert t1["ood_metrics"]["ece"] == pytest.approx(.2)
    assert t1["p95_latency_seconds"] == pytest.approx(.385)
    assert t1["cost_usd_per_change"] == .02


@pytest.mark.parametrize("labels, probabilities", [([], []), ([2], [.5]), ([1], [np.nan]),
                                                       ([1], [1.01]), ([1, 0], [.5])])
def test_binary_metrics_reject_invalid_arrays(labels: list[int], probabilities: list[float]) -> None:
    with pytest.raises(ValueError):
        binary_metrics(labels, probabilities)


@pytest.mark.parametrize("selected, seconds", [([1, 0], [1, 1]), ([True], [1, 1]),
                                                ([True, False], [0, 1]),
                                                ([True, False], [np.inf, 1])])
def test_selection_requires_boolean_mask_and_real_costs(selected: list[bool], seconds: list[float]) -> None:
    with pytest.raises(ValueError):
        evaluate_selection(["a", "b"], [0, 1], selected, seconds)


@pytest.mark.parametrize("splits, changes", [(["train", "test"], ["a", "b"]),
                                             (["calibration", "test"], ["a", "b"]),
                                             (["test", "ood"], ["a", "a"])])
def test_report_rejects_training_data_and_change_split_overlap(tmp_path: Path, splits: list[str], changes: list[str]) -> None:
    output = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        write_report(output, changes, [0, 1], [1, 1], splits, {"T1": {"selected": [True, True]}})
    assert not output.exists()
