import copy
import json
import math
import subprocess
import sys

import numpy as np
import pytest
from brier_ml.calibrate import apply, fit_temperatures, metrics
from brier_ml.glimmer_data import _OPTIONS, calibration_partition


def _row(question="regressed", index=1, *, split="calibration", variant="ev",
         logits=None, options=None, label="A", scenario=1):
    choices = list(options or _OPTIONS[question])
    values = list(logits or ([.8, .2] + [1000.] * 4 if question == "regressed"
                             else [float(i) for i in range(6)]))
    valid = np.asarray(values[:len(choices)], dtype=float)
    p = np.exp(valid - valid.max())
    p /= p.sum()
    return {"id": f"c{index:03}:pusht-s{scenario:02}:{question}:{variant}",
            "pair_id": f"c{index:03}:pusht-s{scenario:02}", "split": split,
            "question": question, "variant": variant, "options": choices, "label": label,
            "logits": values, "probs": dict(zip("ABCDEF"[:len(choices)], p.tolist(), strict=True)),
            "model": "glimmer-base-zeroshot", "revision": "a" * 40}


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _synthetic(temperature=2.7):
    rows = []
    distributions = {"regressed": [.8, .2], "failure_mode": [.3, .25, .2, .1, .1, .05],
                     "success_bucket": [.4, .25, .2, .1, .05]}
    for question, probabilities in distributions.items():
        logits = [temperature * math.log(p) for p in probabilities]
        logits += [1000.] * (6 - len(logits))
        labels = [letter for letter, p in zip("ABCDEF", probabilities) for _ in range(round(20 * p))]
        for change in range(1, 25):
            rows.extend(_row(question, change, label=label, variant="ev" if i % 2 else "noev",
                             scenario=i // 2 + 1, logits=logits) for i, label in enumerate(labels))
    return rows


def test_known_temperature_recovered_from_raw_logits_with_valid_mask(tmp_path):
    rows = _synthetic()
    # Fitting must use raw logits, rather than even valid but stale stored probs.
    for row in rows:
        row["probs"] = dict.fromkeys(row["probs"], 1 / len(row["options"]))
    path = _write(tmp_path / "calibration.jsonl", rows)
    found = fit_temperatures(path)
    assert set(found) == set(_OPTIONS)
    assert list(found.values()) == pytest.approx([2.7] * 3, rel=.05)
    assert fit_temperatures(str(path)) == found


def test_reserved_risk_labels_cannot_change_fitted_temperatures(tmp_path):
    rows = _synthetic()
    fit, risk = calibration_partition([row["pair_id"].split(":")[0] for row in rows])
    path = _write(tmp_path / "probs/calibration.jsonl", rows)
    expected = fit_temperatures(path)
    altered = 0
    for row in rows:
        if row["pair_id"].split(":")[0] in risk:
            row["label"] = "ABCDEF"[("ABCDEF".index(row["label"]) + 1) % len(row["options"])]
            altered += 1
    assert altered == 19 * 20 * 3 and len(fit) == 5
    assert fit_temperatures(_write(path, rows)) == expected


def test_temperature_fitting_refuses_too_few_changes_for_independent_risk(tmp_path):
    rows = [_row(q, change) for q in _OPTIONS for change in range(1, 20)]
    with pytest.raises(ValueError, match="20"):
        fit_temperatures(_write(tmp_path / "probs/calibration.jsonl", rows))


@pytest.mark.parametrize("metadata_name", ["meta.json", "run_meta.json"])
def test_fitting_matches_training_partition_metadata_before_writing(tmp_path, metadata_name):
    rows = _synthetic()
    fit, risk = calibration_partition([row["pair_id"].split(":")[0] for row in rows])
    partition = {"fit_change_ids": fit, "risk_change_ids": risk}
    path = _write(tmp_path / "probs/calibration.jsonl", rows)
    meta_path = tmp_path / metadata_name
    meta_path.write_text(json.dumps({"calibration_partition": partition}))
    assert list(fit_temperatures(path).values()) == pytest.approx([2.7] * 3, rel=.05)
    partition["fit_change_ids"], partition["risk_change_ids"] = risk, fit
    meta_path.write_text(json.dumps({"calibration_partition": partition}))
    with pytest.raises(ValueError, match="calibration_partition disagrees"):
        fit_temperatures(path)
    assert not (tmp_path / "probs_temp").exists() and not (tmp_path / "temps.json").exists()


def test_fit_cannot_consume_heldout_or_previously_scaled_rows(tmp_path):
    rows = _synthetic()
    for split in ("test", "ood", "train"):
        heldout = dict(rows[0], split=split, variant="ev", id="c801:pusht-s01:regressed:ev", pair_id="c801:pusht-s01")
        with pytest.raises(ValueError, match="only unscaled calibration"):
            fit_temperatures(_write(tmp_path / "input.jsonl", rows + [heldout]))
    rows[0]["temperature"] = 1.
    with pytest.raises(ValueError, match="only unscaled calibration"):
        fit_temperatures(_write(tmp_path / "input.jsonl", rows))


@pytest.mark.parametrize(("label", "expected"), [("A", .05), ("B", 20.)])
def test_known_boundary_optima_are_preserved(tmp_path, label, expected):
    rows = [_row(question, change, logits=[5., 0., 0., 0., 0., 0.], label=label)
            for question in _OPTIONS for change in range(1, 21)]
    found = fit_temperatures(_write(tmp_path / "calibration.jsonl", rows))
    assert list(found.values()) == pytest.approx([expected] * 3)


def test_apply_writes_same_metadata_recomputes_from_logits_and_shared_variant_t(tmp_path):
    rows = [_row(variant=v, split="test") for v in ("ev", "noev")]
    original = copy.deepcopy(rows)
    path = _write(tmp_path / "model/probs/test.jsonl", rows)
    temperatures = dict.fromkeys(_OPTIONS, 2.)
    destination = apply(path, temperatures)
    assert destination == tmp_path / "model/probs_temp/test.jsonl"
    scaled = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [r["temperature"] for r in scaled] == [2., 2.]
    for found, source in zip(scaled, rows, strict=True):
        assert {k: v for k, v in found.items() if k not in {"probs", "temperature"}} == {
            k: v for k, v in source.items() if k != "probs"}
        expected_a = 1 / (1 + math.exp(-.3))
        assert found["probs"] == pytest.approx({"A": expected_a, "B": 1 - expected_a})
    assert rows == original
    assert [json.loads(line) for line in path.read_text().splitlines()] == rows


@pytest.mark.parametrize("temps", [dict.fromkeys(_OPTIONS, 0.), dict.fromkeys(_OPTIONS, True),
                                    dict.fromkeys(_OPTIONS, math.inf), {"regressed": 1.}])
def test_apply_rejects_invalid_or_partial_temperatures(tmp_path, temps):
    with pytest.raises(ValueError, match="temperatures"):
        apply(_write(tmp_path / "probs/test.jsonl", [_row()]), temps)
    assert not (tmp_path / "probs_temp").exists()


def test_metrics_hand_equal_mass_ece_semantic_yes_and_variants(tmp_path):
    rows = []
    for i, (p_yes, is_yes) in enumerate(zip([.1, .3, .8, .8], [0, 1, 1, 0], strict=True)):
        # Yes is displayed in position B; tied .8 rows remain separate bins.
        rows.append(_row(index=i + 1, options=["no", "yes"], label="B" if is_yes else "A",
                         logits=[math.log(1 - p_yes), math.log(p_yes), 99., 99., 99., 99.]))
    rows.append(_row(index=5, variant="noev"))
    report = metrics(_write(tmp_path / "scores.jsonl", rows))["regressed"]
    ev = report["variants"]["ev"]
    assert ev["count"] == 4 and report["count"] == 5
    assert ev["ece"] == pytest.approx(.45)
    assert ev["brier"] == pytest.approx(2 * (.1 ** 2 + .7 ** 2 + .2 ** 2 + .8 ** 2) / 4)
    assert ev["accuracy"] == .5
    assert ev["auroc"] == pytest.approx(.625)
    assert report["variants"]["noev"]["auroc"] is None
    assert report["variants"]["both"] == {k: v for k, v in report.items() if k != "variants"}


def test_metrics_ordinal_rps_is_invariant_to_display_order(tmp_path):
    options = list(_OPTIONS["success_bucket"])
    probabilities = [.1, .2, .4, .2, .1]
    row = _row("success_bucket", options=options, label="C",
               logits=[math.log(p) for p in probabilities] + [999.])
    path = _write(tmp_path / "scores.jsonl", [row])
    expected = metrics(path)["success_bucket"]
    assert expected["rps"] == pytest.approx((.1 ** 2 + .3 ** 2 + (-.3) ** 2 + (-.1) ** 2) / 4)
    order = [3, 0, 4, 2, 1]
    row["options"] = [options[i] for i in order]
    row["probs"] = {letter: probabilities[i] for letter, i in zip("ABCDE", order, strict=True)}
    row["label"] = "D"
    row["logits"] = [row["logits"][i] for i in order] + [999.]
    assert metrics(_write(path, [row]))["success_bucket"] == expected


@pytest.mark.parametrize(("field", "value"), [
    ("extra", "unknown"), ("question", "unknown"), ("label", "F"), ("label", 0),
    ("logits", [0.] * 5), ("logits", [math.nan] * 6), ("logits", [True] * 6),
    ("probs", {"A": .8, "B": .3}), ("probs", {"A": .8, "C": .2}),
    ("options", ["yes", "yes"]), ("options", ["no", "maybe"]),
    ("revision", "moving-main"), ("pair_id", "c003:pusht-s01"), ("variant", "cheap"),
])
def test_score_contract_rejects_invalid_rows(tmp_path, field, value):
    row = _row()
    row[field] = value
    with pytest.raises(ValueError):
        metrics(_write(tmp_path / "scores.jsonl", [row]))


def test_score_contract_rejects_duplicate_ids_and_mixed_provenance(tmp_path):
    row = _row()
    for extra in (row, dict(_row(index=2), revision="b" * 40), dict(_row(index=2), model="another")):
        with pytest.raises(ValueError, match="duplicate|model/revision"):
            metrics(_write(tmp_path / "scores.jsonl", [row, extra]))


def test_cli_fits_only_calibration_preserves_source_and_writes_expected_artifacts(tmp_path):
    model_dir = tmp_path / "base"
    rows = _synthetic()
    _write(model_dir / "probs/calibration.jsonl", rows)
    for split, index in (("test", 201), ("ood", 301)):
        _write(model_dir / f"probs/{split}.jsonl", [_row(q, index, split=split, variant=v)
                                                  for q in _OPTIONS for v in ("ev", "noev")])
    result = subprocess.run([sys.executable, "-m", "brier_ml.calibrate", "--model-dir", str(model_dir),
                             "--name", "glimmer-base-zeroshot"], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    temperatures = json.loads((model_dir / "temps.json").read_text())
    assert list(temperatures.values()) == pytest.approx([2.7] * 3, rel=.05)
    report = json.loads((model_dir / "metrics.json").read_text())
    assert set(report) == {"model", "revision", "temperatures", "splits", "calibration_partition"}
    assert report["model"] == "glimmer-base-zeroshot" and report["revision"] == "a" * 40
    assert report["temperatures"] == temperatures and set(report["splits"]) == {"test", "ood"}
    fit, risk = calibration_partition([row["pair_id"].split(":")[0] for row in rows])
    assert report["calibration_partition"] == {"fit_change_ids": fit, "risk_change_ids": risk}
    assert len(risk) == 19 and not set(fit) & set(risk)
    for split in ("calibration", "test", "ood"):
        path = model_dir / f"probs_temp/{split}.jsonl"
        actual = [json.loads(line) for line in path.read_text().splitlines()]
        assert all(row["temperature"] == temperatures[row["question"]] for row in actual)
        if split == "calibration":
            assert {row["id"] for row in actual} == {row["id"] for row in rows}
            assert {row["pair_id"].split(":")[0] for row in actual} >= set(risk)
        if split != "calibration":
            assert report["splits"][split]["before"] == metrics(model_dir / f"probs/{split}.jsonl")
            assert report["splits"][split]["after"] == metrics(path)


@pytest.mark.parametrize("problem", ["name", "provenance", "overlap", "filename"])
def test_cli_rejects_invalid_split_or_provenance_before_publishing(tmp_path, problem):
    from brier_ml.calibrate import _main

    for split, index in (("calibration", 1), ("test", 201), ("ood", 301)):
        rows = [_row(q, 1 if problem == "overlap" else index, split=split) for q in _OPTIONS]
        if split == "test" and problem == "provenance":
            rows[0]["revision"] = "b" * 40
        if split == "ood" and problem == "filename":
            for row in rows:
                row["split"] = "test"
        _write(tmp_path / f"probs/{split}.jsonl", rows)
    from unittest.mock import patch

    with (patch.object(sys, "argv", ["calibrate", "--model-dir", str(tmp_path), "--name",
                                    "wrong" if problem == "name" else "glimmer-base-zeroshot"]),
          pytest.raises(ValueError, match="provenance|split|model/revision")):
        _main()
    assert not (tmp_path / "temps.json").exists() and not (tmp_path / "probs_temp").exists()
