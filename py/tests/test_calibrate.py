from __future__ import annotations

import copy
import math

import pytest
from brier_ml.calibrate import apply_temperature, fit_temperatures


def _records() -> list[dict]:
    options = {
        "regressed": {"A": "yes", "B": "no"},
        "failure_mode": {
            "A": "none", "B": "stall/timeout", "C": "overshoot/misalignment",
            "D": "grasp or drop failure", "E": "collision", "F": "oscillation",
        },
        "success_bucket": {chr(65 + index): str(index + 1) for index in range(5)},
    }
    rows = []
    splits = [(index, "calibration") for index in range(1, 9)]
    splits += [(101, "train"), (102, "test"), (103, "ood")]
    for change, split in splits:
        for question, choices in options.items():
            probabilities = {key: 0.01 for key in choices}
            probabilities["A"] = 1 - 0.01 * (len(choices) - 1)
            rows.append({
                "id": f"c{change:03}/pusht-s01/{question}/cheap",
                "split": split, "question": question, "options": dict(choices),
                "label": "A" if change <= 6 or split != "calibration" else "B",
                "probabilities": probabilities,
            })
    return rows


def test_apply_temperature_scales_options_and_preserves_keys() -> None:
    probabilities = {"B": 0.2, "A": 0.8}
    scaled = apply_temperature(probabilities, 2.0)
    assert list(scaled) == ["B", "A"]
    assert scaled == pytest.approx({"B": 1 / 3, "A": 2 / 3})
    assert apply_temperature(probabilities, 1.0) == pytest.approx(probabilities)
    assert probabilities == {"B": 0.2, "A": 0.8}


def test_fit_reports_calibration_metrics_without_mutating_labels() -> None:
    records = _records()
    original = copy.deepcopy(records)
    report = fit_temperatures(records)
    assert set(report["temperatures"]) == {"regressed", "failure_mode", "success_bucket"}
    expected = math.log(99) / math.log(3)
    assert report["temperatures"]["regressed"] == pytest.approx(expected, rel=1e-5)
    for question, temperature in report["temperatures"].items():
        assert 1 < temperature <= 20
        metrics = report["calibration"][question]
        assert metrics["count"] == 8
        assert set(metrics["before"]) == {"ece", "brier", "nll"}
        for key in ("ece", "brier", "nll"):
            assert metrics["after"][key] < metrics["before"][key]
    assert report["calibration"]["regressed"]["before"]["ece"] == pytest.approx(0.24)
    assert records == original


def test_noncalibration_labels_and_scores_cannot_change_the_fit() -> None:
    records = _records()
    expected = fit_temperatures(records)
    for record in records:
        if record["split"] != "calibration":
            record["label"] = "B"
            probabilities = record["probabilities"]
            probabilities["A"], probabilities["B"] = probabilities["B"], probabilities["A"]
    assert fit_temperatures(records) == expected


def test_calibration_is_invariant_to_shuffled_option_letters() -> None:
    records = _records()
    expected = fit_temperatures(records)
    for record in records:
        letters = sorted(record["options"])
        renamed = dict(zip(letters, reversed(letters), strict=True))
        record["options"] = {renamed[key]: value for key, value in record["options"].items()}
        scores = {renamed[key]: value for key, value in record["probabilities"].items()}
        record["probabilities"] = {key: scores[key] for key in letters}
        record["label"] = renamed[record["label"]]
    actual = fit_temperatures(records)
    assert actual["temperatures"] == pytest.approx(expected["temperatures"], rel=1e-6)
    for question in expected["calibration"]:
        for stage in ("before", "after"):
            assert actual["calibration"][question][stage] == pytest.approx(
                expected["calibration"][question][stage], abs=1e-9
            )


@pytest.mark.parametrize("temperature", [0, -1, math.nan, math.inf, True])
def test_apply_temperature_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature({"A": 0.8, "B": 0.2}, temperature)


@pytest.mark.parametrize("probabilities", [
    {"A": 0.0, "B": 1.0}, {"A": math.nan, "B": 0.2},
    {"A": math.inf, "B": 0.2}, {"A": 0.8, "B": 0.3}, {"A": True, "B": 0.2},
])
def test_apply_temperature_rejects_invalid_scores(probabilities: dict) -> None:
    with pytest.raises(ValueError, match="probabilit"):
        apply_temperature(probabilities, 1.0)


@pytest.mark.parametrize("question", ["regressed", "failure_mode", "success_bucket"])
def test_fit_rejects_altered_option_semantics(question: str) -> None:
    records = _records()
    record = next(row for row in records if row["question"] == question)
    record["options"]["A"] = "not-a-compiled-option"
    with pytest.raises(ValueError, match="Rust compiled semantics"):
        fit_temperatures(records)


@pytest.mark.parametrize(("field", "value"), [
    ("label", "Z"), ("split", "validation"),
    ("probabilities", {"A": 0.5, "C": 0.5}), ("question", "failure_mode"),
])
def test_fit_validates_excluded_rows(field: str, value: object) -> None:
    records = _records()
    records[-3][field] = value
    with pytest.raises(ValueError):
        fit_temperatures(records)


@pytest.mark.parametrize("duplicate", [False, True])
def test_fit_rejects_duplicate_rows_and_split_overlap(duplicate: bool) -> None:
    records = _records()
    extra = copy.deepcopy(records[0])
    if not duplicate:
        extra["id"] = "c001/pusht-s01/regressed/no_cheap"
        extra["split"] = "test"
    records.append(extra)
    with pytest.raises(ValueError, match="duplicate|multiple splits"):
        fit_temperatures(records)


def test_fit_requires_calibration_examples_for_every_question() -> None:
    records = [row for row in _records()
               if row["question"] != "success_bucket" or row["split"] != "calibration"]
    with pytest.raises(ValueError, match="all three"):
        fit_temperatures(records)
