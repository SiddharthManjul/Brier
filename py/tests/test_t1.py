from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from brier_ml.dataset import FEATURE_NAMES, _change_features, load_dataset
from brier_ml.t1 import predict_t1, train_t1


@pytest.fixture
def dataset_files(tmp_path: Path) -> tuple[Path, Path]:
    ledger, compiled = tmp_path / "ledger.sqlite", tmp_path / "compiled.jsonl"
    with sqlite3.connect(ledger) as connection:
        connection.executescript("""
            CREATE TABLE changes (change_id TEXT, env TEXT, card_json TEXT);
            CREATE TABLE scenarios (scenario_id TEXT, env TEXT, params_json TEXT, baseline_success REAL);
            CREATE TABLE labels (change_id TEXT, scenario_id TEXT, regressed INT, failure_mode TEXT, success_bucket INT);
            CREATE TABLE runs (change_id TEXT, scenario_id TEXT, tier TEXT, episodes INT, successes INT, mean_reward REAL);
        """)
        connection.execute("INSERT INTO scenarios VALUES (?, ?, ?, ?)", (
            "pusht-s01", "gym_pusht/PushT-v0",
            json.dumps({"pose_bin": 0, "lighting": "normal", "camera_jitter": False}), .9,
        ))
        examples = []
        for i in range(15):
            change = f"c{i:03}"
            split = "train" if i < 8 else "calibration" if i < 12 else "test" if i < 14 else "ood"
            target = i % 2
            family, params = ("obs_noise", {"sigma": .1 if target else .001})
            if split == "ood":
                family, params = "occlusion", {"x": 0, "y": 0, "width": 10, "height": 10}
            card = {"description": "sensor perturbation", "families": [{"family": family, "params": params}]}
            connection.execute("INSERT INTO changes VALUES (?, ?, ?)", (change, "gym_pusht/PushT-v0", json.dumps(card)))
            connection.execute("INSERT INTO labels VALUES (?, ?, ?, ?, ?)", (change, "pusht-s01", target, "stall/timeout" if target else "none", 2 if target else 5))
            connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)", (change, "pusht-s01", "cheap", 3, 1 if target else 3, .2 if target else .8))
            examples.append({"id": f"{change}/pusht-s01/regressed/cheap", "split": split,
                             "question": "regressed", "options": {"A": "no", "B": "yes"},
                             "prompt": "Exact compiled Rust prompt", "label": "B" if target else "A"})
    compiled.write_text("".join(json.dumps(row) + "\n" for row in examples))
    return ledger, compiled


def test_load_dataset_numeric_features_and_training_only_history(dataset_files: tuple[Path, Path]) -> None:
    dataset = load_dataset(*dataset_files)
    assert dataset["features"].shape == (15, len(FEATURE_NAMES))
    assert dataset["labels"].tolist() == [i % 2 for i in range(15)]
    features = dict(zip(FEATURE_NAMES, dataset["features"][0], strict=True))
    assert features["family_obs_noise"] == 1
    assert features["obs_noise_sigma"] == .001
    assert features["cheap_success"] == 1
    assert features["cheap_mean_reward"] == pytest.approx(.8)
    assert features["baseline_success"] == .9
    assert features["scenario_fragility"] == pytest.approx(4 / 7)
    assert features["similar_regressed_rate"] == pytest.approx(2 / 5)


def test_loads_all_questions_but_only_regression_cheap_becomes_features(dataset_files: tuple[Path, Path]) -> None:
    ledger, compiled = dataset_files
    regression = [json.loads(line) for line in compiled.read_text().splitlines()]
    rows = []
    for example in regression:
        target = example["label"] == "B"
        prefix = "/".join(example["id"].split("/")[:2])
        for evidence in ("cheap", "no_cheap"):
            rows.append({**example, "id": f"{prefix}/regressed/{evidence}"})
            options = {chr(65 + i): value for i, value in enumerate([
                "none", "stall/timeout", "overshoot/misalignment", "grasp or drop failure", "collision", "oscillation"])}
            rows.append({**example, "id": f"{prefix}/failure_mode/{evidence}",
                         "question": "failure_mode", "options": options, "label": "B" if target else "A"})
            rows.append({**example, "id": f"{prefix}/success_bucket/{evidence}",
                         "question": "success_bucket", "options": dict(zip("ABCDE", "12345", strict=True)),
                         "label": "B" if target else "E"})
    compiled.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert load_dataset(ledger, compiled)["features"].shape[0] == len(regression)
    rows[-1]["label"] = "A"
    compiled.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="disagree"):
        load_dataset(ledger, compiled)


def test_export_predictions_match_lightgbm_isotonic_and_json_roundtrip(dataset_files: tuple[Path, Path], tmp_path: Path) -> None:
    dataset = load_dataset(*dataset_files)
    output = tmp_path / "t1.json"
    artifact = train_t1(dataset, output)
    from lightgbm import LGBMClassifier
    from sklearn.isotonic import IsotonicRegression

    train = np.asarray(dataset["splits"]) == "train"
    calibration = np.asarray(dataset["splits"]) == "calibration"
    classifier = LGBMClassifier(
        objective="binary", n_estimators=100, num_leaves=15, max_depth=4,
        min_child_samples=2, learning_rate=.05, random_state=0, n_jobs=1,
        deterministic=True, force_col_wise=True, verbosity=-1,
    ).fit(dataset["features"][train], dataset["labels"][train], feature_name=FEATURE_NAMES)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(
        classifier.booster_.predict(dataset["features"][calibration]), dataset["labels"][calibration])
    expected = isotonic.predict(classifier.booster_.predict(dataset["features"]))
    assert np.allclose(predict_t1(artifact, dataset["features"]), expected, atol=1e-12)
    assert artifact == json.loads(output.read_text())
    assert artifact["feature_names"] == FEATURE_NAMES
    assert artifact["training_change_ids"] == [f"c{i:03}" for i in range(8)]
    assert artifact["calibration_change_ids"] == [f"c{i:03}" for i in range(8, 12)]
    assert artifact["model_version"].startswith("t1@")
    assert len(artifact["model_version"]) == 67


def test_heldout_labels_cannot_change_features_or_fitted_artifact(dataset_files: tuple[Path, Path], tmp_path: Path) -> None:
    before = load_dataset(*dataset_files)
    original = train_t1(before, tmp_path / "before.json")
    ledger, compiled = dataset_files
    examples = [json.loads(line) for line in compiled.read_text().splitlines()]
    with sqlite3.connect(ledger) as connection:
        for row in examples:
            if row["split"] in {"test", "ood"}:
                change = row["id"].split("/")[0]
                connection.execute("UPDATE labels SET regressed = 1 - regressed WHERE change_id = ?", (change,))
                row["label"] = "B" if row["label"] == "A" else "A"
    compiled.write_text("".join(json.dumps(row) + "\n" for row in examples))
    after = load_dataset(*dataset_files)
    np.testing.assert_array_equal(before["features"], after["features"])
    assert train_t1(after, tmp_path / "after.json") == original


def test_calibration_labels_change_only_isotonic(dataset_files: tuple[Path, Path], tmp_path: Path) -> None:
    dataset = load_dataset(*dataset_files)
    original = train_t1(dataset, tmp_path / "a.json")
    calibration = np.asarray(dataset["splits"]) == "calibration"
    dataset["labels"][calibration] = 1 - dataset["labels"][calibration]
    revised = train_t1(dataset, tmp_path / "b.json")
    assert original["booster"] == revised["booster"]
    assert original["isotonic"] != revised["isotonic"]


def test_constant_predictions_have_valid_single_knot_isotonic(dataset_files: tuple[Path, Path], tmp_path: Path) -> None:
    dataset = load_dataset(*dataset_files)
    dataset["features"][:] = 1
    artifact = train_t1(dataset, tmp_path / "constant.json")
    assert len(artifact["isotonic"]["x"]) == 1
    np.testing.assert_allclose(predict_t1(artifact, dataset["features"]), .5)


@pytest.mark.parametrize("mutation, match", [
    ("overlap", "multiple splits"), ("label", "disagree"), ("missing_run", "missing ledger"),
    ("small_history", "five other"), ("invalid_scenario", "scenario parameters"),
    ("unknown_family", "families"), ("missing_baseline", "finite number"),
    ("unknown_question", "invalid"), ("ood_leakage", "held out"),
])
def test_load_dataset_rejects_inconsistent_inputs(dataset_files: tuple[Path, Path], mutation: str, match: str) -> None:
    ledger, compiled = dataset_files
    examples = [json.loads(line) for line in compiled.read_text().splitlines()]
    if mutation == "overlap":
        examples.append({**examples[0], "id": "c000/pusht-s01/regressed/no_cheap", "split": "test"})
    elif mutation == "label":
        examples[0]["label"] = "B"
    elif mutation == "small_history":
        examples = [row for row in examples if row["id"].split("/")[0] not in {"c000", "c001", "c002"}]
    elif mutation == "unknown_question":
        examples[0]["question"] = "made_up_question"
    elif mutation == "ood_leakage":
        examples[-1]["split"] = "train"
    else:
        with sqlite3.connect(ledger) as connection:
            if mutation == "missing_run":
                connection.execute("DELETE FROM runs WHERE change_id = 'c000'")
            elif mutation == "missing_baseline":
                connection.execute("UPDATE scenarios SET baseline_success = NULL")
            elif mutation == "invalid_scenario":
                connection.execute("UPDATE scenarios SET params_json = '{}' ")
            elif mutation == "unknown_family":
                card = {"description": "bad", "families": [{"family": "unknown", "params": {}}]}
                connection.execute("UPDATE changes SET card_json = ? WHERE change_id = 'c000'", (json.dumps(card),))
    compiled.write_text("".join(json.dumps(row) + "\n" for row in examples))
    with pytest.raises(ValueError, match=match):
        load_dataset(ledger, compiled)


def test_train_rejects_missing_classes_and_overlap(dataset_files: tuple[Path, Path], tmp_path: Path) -> None:
    dataset = load_dataset(*dataset_files)
    dataset["labels"][:] = 0
    with pytest.raises(ValueError, match="both regression classes"):
        train_t1(dataset, tmp_path / "bad.json")
    dataset["changes"][-1] = dataset["changes"][0]
    with pytest.raises(ValueError, match="multiple splits"):
        train_t1(dataset, tmp_path / "bad.json")


def test_predict_rejects_nonfinite_features_and_bad_knots(dataset_files: tuple[Path, Path], tmp_path: Path) -> None:
    dataset = load_dataset(*dataset_files)
    artifact = train_t1(dataset, tmp_path / "t1.json")
    with pytest.raises(ValueError, match="finite matrix"):
        predict_t1(artifact, np.full_like(dataset["features"], np.nan))
    artifact["isotonic"]["x"] = [0.5, 0.5]
    artifact["isotonic"]["y"] = [0, 1]
    with pytest.raises(ValueError, match="isotonic knots"):
        predict_t1(artifact, dataset["features"])


def _guide_files(files: tuple[Path, Path]) -> tuple[Path, Path]:
    ledger, compiled = files
    rows = [json.loads(line) for line in compiled.read_text().splitlines()]
    with sqlite3.connect(ledger) as connection:
        connection.execute("UPDATE scenarios SET params_json = ?", (json.dumps({
            "pose_bin": 1, "lighting": "normal", "camera_jitter": False}),))
        for index, row in enumerate(rows):
            old, change = row["id"].split("/")[0], f"c{index + 101:03d}"
            for table in ("changes", "labels", "runs"):
                connection.execute(f"UPDATE {table} SET change_id = ? WHERE change_id = ?", (change, old))
            card = json.loads(connection.execute("SELECT card_json FROM changes WHERE change_id = ?", (change,)).fetchone()[0])
            card = {"change_id": change, "env": "pusht", "families": card["families"], "desc": card["description"], "seed": 1}
            if index == len(rows) - 1:
                card["families"] = [{"family": "occlusion", "params": {"fraction": .2}}]
            connection.execute("UPDATE changes SET card_json = ? WHERE change_id = ?", (json.dumps(card), change))
            bucket = int.from_bytes(hashlib.sha256(change.encode()).digest()[:4], "big") % 10
            split = "train" if bucket < 6 else "calibration" if bucket < 8 else "test"
            rows[index] = {"id": f"{change}:pusht-s01:regressed:ev", "pair_id": f"{change}:pusht-s01",
                           "change_id": change, "scenario_id": "pusht-s01", "env": "pusht", "variant": "ev",
                           "split": "ood" if index == len(rows) - 1 else split, "question": "regressed",
                           "options": ["yes", "no"], "label": "A" if row["label"] == "B" else "B",
                           "prompt": "<system> exact\n<state>\nx\n</state>\n<question> exact\n<options> A) yes  B) no\n<answer>"}
    compiled.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return ledger, compiled


def test_guide_dataset_keeps_ids_pose_and_training_history(dataset_files: tuple[Path, Path]) -> None:
    dataset = load_dataset(*_guide_files(dataset_files))
    assert dataset["ids"][0] == "c101:pusht-s01:regressed:ev"
    assert dataset["features"].shape == (15, len(FEATURE_NAMES))
    assert dataset["features"][:, FEATURE_NAMES.index("pose_bin")].tolist() == [0.] * 15
    assert dataset["features"][-1, FEATURE_NAMES.index("occlusion_width")] == .2
    train = [i for i, split in enumerate(dataset["splits"]) if split == "train"][:2]
    ledger, compiled = dataset_files
    rows = [json.loads(line) for line in compiled.read_text().splitlines()]
    rows = [row for i, row in enumerate(rows) if row["split"] != "train" or i in train]
    compiled.write_text("".join(json.dumps(row) + "\n" for row in rows))
    smaller = load_dataset(ledger, compiled)
    assert len(smaller["ids"]) == len(rows)
    for index, row in enumerate(rows):
        targets = [i % 2 for i in train if f"c{i + 101:03d}" != row["change_id"]]
        assert smaller["features"][index, -1] == pytest.approx(np.mean(targets))


@pytest.mark.parametrize("family,params,expected", [
    ("action_scale", {"s": .8}, {"action_scale_scale": .2}),
    ("action_bias", {"b": 2}, {"action_bias_drift": 2}),
    ("camera_shift", {"shift": 4, "angle": 3}, {"camera_shift_dx": 4, "camera_shift_dy": 4, "camera_shift_angle": 3}),
    ("occlusion", {"fraction": .2}, {"occlusion_x": 0, "occlusion_y": 0, "occlusion_width": .2, "occlusion_height": .2}),
])
def test_guide_feature_aliases_leave_original_card_unchanged(family, params, expected) -> None:
    card = {"change_id": "c001", "env": "pusht", "families": [{"family": family, "params": params}], "desc": "change", "seed": 1}
    before = json.dumps(card)
    features = dict(zip(FEATURE_NAMES, _change_features(card)))
    for key, value in expected.items():
        assert features[key] == pytest.approx(value)
    assert json.dumps(card) == before


@pytest.mark.parametrize("field,value", [("pair_id", "c999:pusht-s01"), ("env", "aloha"),
    ("scenario_id", "pusht-s02"), ("variant", "noev"), ("split", "ood"),
    ("options", ["yes", "yes"]), ("label", "F"), ("prompt", "invented<answer>")])
def test_guide_dataset_rejects_invalid_provenance_and_contract(dataset_files, field, value) -> None:
    ledger, compiled = _guide_files(dataset_files)
    rows = [json.loads(line) for line in compiled.read_text().splitlines()]
    rows[0][field] = value
    compiled.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        load_dataset(ledger, compiled)


def test_guide_history_ranks_family_jaccard_before_magnitude(dataset_files) -> None:
    ledger, compiled = _guide_files(dataset_files)
    rows = [json.loads(line) for line in compiled.read_text().splitlines()]
    training = [row for row in rows if row["split"] == "train"]
    with sqlite3.connect(ledger) as connection:
        for row in training[:3]:
            card = {"change_id": row["change_id"], "env": "pusht", "desc": "change", "seed": 1,
                    "families": [{"family": "action_latency", "params": {"k": 1}}]}
            connection.execute("UPDATE changes SET card_json = ? WHERE change_id = ?", (json.dumps(card), row["change_id"]))
    dataset = load_dataset(ledger, compiled)
    target = next(i for i, row in enumerate(rows) if row["split"] == "calibration")
    neighbors = [i for i, row in enumerate(rows) if row in training[3:]]
    assert len(neighbors) == 5
    assert dataset["features"][target, -1] == pytest.approx(dataset["labels"][neighbors].mean())
    heldout = [row for row in rows if row["split"] != "train"]
    compiled.write_text("".join(json.dumps(row) + "\n" for row in heldout))
    with pytest.raises(ValueError, match="one other training change"):
        load_dataset(ledger, compiled)


def test_new_container_preserves_legacy_aloha_ledger_ids_cards_and_pose(dataset_files) -> None:
    ledger, compiled = _guide_files(dataset_files)
    rows = [json.loads(line) for line in compiled.read_text().splitlines()]
    with sqlite3.connect(ledger) as connection:
        connection.execute("UPDATE scenarios SET scenario_id = 'aloha_transfer_cube-s01', env = 'gym_aloha/AlohaTransferCube-v0', params_json = ?",
                           (json.dumps({"pose_bin": 0, "lighting": "normal", "camera_jitter": False, "physics_jitter": False}),))
        connection.execute("UPDATE labels SET scenario_id = 'aloha_transfer_cube-s01'")
        connection.execute("UPDATE runs SET scenario_id = 'aloha_transfer_cube-s01'")
        for row in rows:
            card = json.loads(connection.execute("SELECT card_json FROM changes WHERE change_id = ?", (row["change_id"],)).fetchone()[0])
            if row["split"] == "ood":
                card["families"] = [{"family": "occlusion", "params": {"x": 0, "y": 0, "width": 10, "height": 10}}]
            old = {"description": card["desc"], "families": card["families"]}
            connection.execute("UPDATE changes SET env = 'gym_aloha/AlohaTransferCube-v0', card_json = ? WHERE change_id = ?",
                               (json.dumps(old), row["change_id"]))
            row.update(id=row["id"].replace("pusht-s01", "aloha_transfer_cube-s01"),
                       pair_id=row["pair_id"].replace("pusht-s01", "aloha_transfer_cube-s01"),
                       scenario_id="aloha_transfer_cube-s01", env="aloha")
    compiled.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = load_dataset(ledger, compiled)
    assert dataset["ids"][0] == "c101:aloha_transfer_cube-s01:regressed:ev"
    assert dataset["features"][:, FEATURE_NAMES.index("env_is_aloha")].tolist() == [1.] * 15
    assert dataset["features"][:, FEATURE_NAMES.index("pose_bin")].tolist() == [0.] * 15
