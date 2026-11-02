from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from brier_ml.glimmer_data import calibration_partition
from brier_ml.prepare import _scores, _training_receipt, prepare_bundle
from brier_ml.t1 import train_t1


def test_prepare_rejects_mutated_released_input_before_creating_bundle(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    names = ("ledger.sqlite", "compiled.jsonl", "episodes.jsonl", "changes.jsonl",
             "scenarios.jsonl", "label-summary.json", "quality-gates.json", "SHA256SUMS")
    hashes = {}
    for name in names:
        payload = f"frozen-{name}".encode()
        (dataset / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    receipt = {"schema": "brier-training-readiness-v1", "measurements": {
        "release_hashes": hashes, "checks_skipped": [],
        "gates": dict.fromkeys(("G1", "G2", "G3", "G4", "G5"), "passed"),
        "parquet_keys": 5152, "episodes": 60480, "labeled_pairs": 2560, "compiled_records": 15360,
        "rust_label_compile_replayed": True, "golden_test_passed": True,
    }}
    payload = json.dumps(receipt).encode()
    readiness = tmp_path / f"training-readiness-{hashlib.sha256(payload).hexdigest()}.json"
    readiness.write_bytes(payload)
    (dataset / "compiled.jsonl").write_text("mutated")
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="released input hash changed: compiled.jsonl"):
        prepare_bundle(dataset, tmp_path / "runs", readiness, output, tmp_path / "brier",
                       training_receipt=tmp_path / "completion.json")
    assert not output.exists()


def test_t1_isotonic_excludes_all_nineteen_protected_risk_changes(tmp_path: Path) -> None:
    fit, risk = calibration_partition(f"c{index:03}" for index in range(1, 27))
    training = [f"c{index:03}" for index in range(101, 121)]
    changes = training + fit + risk
    labels = np.array([index % 2 for index in range(len(changes))])
    data = {"ids": [f"{change}:pusht-s01:regressed:ev" for change in changes],
            "changes": changes, "splits": ["train"] * len(training) + ["calibration"] * 26,
            "feature_names": ["fixed_feature"], "features": labels[:, None].astype(float),
            "labels": labels.copy()}
    first = train_t1(data, tmp_path / "first.json", calibration_change_ids=fit)
    data["labels"][np.isin(changes, risk)] = 1 - data["labels"][np.isin(changes, risk)]
    second = train_t1(data, tmp_path / "second.json", calibration_change_ids=fit)
    assert first == second
    assert first["calibration_change_ids"] == fit
    assert set(first["calibration_change_ids"]).isdisjoint(risk)
    for invalid in (risk + ["c999"], fit + fit, []):
        with pytest.raises(ValueError, match="distinct calibration changes"):
            train_t1(data, tmp_path / "bad.json", calibration_change_ids=invalid)


@pytest.mark.parametrize("mutation", ["label", "probability", "coverage", "revision"])
def test_scores_must_match_compiled_identity_and_calibrated_logits(tmp_path: Path, mutation: str) -> None:
    directory = tmp_path / "base"
    (directory / "probs_temp").mkdir(parents=True)
    revision = "a" * 40
    temperatures = {"regressed": 1., "failure_mode": 1., "success_bucket": 1.}
    partition = {"fit_change_ids": ["c001"], "risk_change_ids": ["c002"]}
    (directory / "temps.json").write_text(json.dumps(temperatures))
    (directory / "metrics.json").write_text(json.dumps({
        "model": "glimmer-base-zeroshot", "revision": revision,
        "temperatures": temperatures, "calibration_partition": partition,
    }))
    compiled = []
    for index, split in enumerate(("calibration", "test", "ood"), 1):
        identifier = f"c{index:03}:pusht-s01:regressed:ev"
        row = {"id": identifier, "pair_id": f"c{index:03}:pusht-s01", "split": split,
               "question": "regressed", "variant": "ev", "options": ["yes", "no"], "label": "A",
               "logits": [0.] * 6, "probs": {"A": .5, "B": .5},
               "model": "glimmer-base-zeroshot", "revision": revision, "temperature": 1.}
        compiled.append({"id": identifier, "split": split, "options": {"A": "yes", "B": "no"}, "label": "A"})
        if split == "calibration":
            if mutation == "label":
                row["label"] = "B"
            elif mutation == "probability":
                row["probs"] = {"A": .6, "B": .4}
            elif mutation == "revision":
                row["revision"] = "b" * 40
            else:
                row["id"], row["pair_id"] = "c004:pusht-s01:regressed:ev", "c004:pusht-s01"
        (directory / "probs_temp" / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="score metadata|scaled scores|score coverage"):
        _scores(directory, compiled, revision, partition, {}, tmp_path, "d" * 64)


@pytest.mark.parametrize("changed", ["adapter/adapter_model.safetensors", "temps.json",
                                     "probs_temp/test.jsonl", "run_meta.json"])
def test_changed_adapter_cannot_inherit_completed_model_predictions(tmp_path: Path, changed: str) -> None:
    runs, readiness = tmp_path / "runs", tmp_path / "readiness.json"
    readiness.write_text("immutable readiness")
    directory = runs / "glimmer-ft-v1"
    names = {"adapter/adapter_config.json", "adapter/adapter_model.safetensors", "adapter/brier-model.json",
             "run_meta.json", "temps.json", "metrics.json", "train_log.jsonl"}
    names.update(f"{folder}/{split}.jsonl" for folder in ("probs", "probs_temp")
                 for split in ("calibration", "test", "ood"))
    artifacts = {}
    for name in names:
        source = directory / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(name.encode())
        artifacts[name] = {"sha256": hashlib.sha256(name.encode()).hexdigest(), "bytes": len(name)}
    partition = {"fit_change_ids": ["c001"], "risk_change_ids": ["c002"]}
    receipt = {"schema": "brier-one-epoch-training-completion-v1",
               "dataset_readiness_receipt": {"sha256": hashlib.sha256(readiness.read_bytes()).hexdigest()},
               "data_sha256": "d" * 64, "calibration_partition": partition,
               "base_revision": "cec94fcfd6512f2cfe708138c32533b346ea4f69", "epochs": 1, "training_steps": 249,
               "max_train_usd": 12, "max_gpu_usd": 22, "active_training_apps": [],
               "pending_paid_operation": None, "training_process_exited": True, "training_lock_free": True,
               "second_epoch_executed": False, "publication_authorized": False, "deployment_authorized": False,
               "step_gpu_spend_usd": {step: .1 for step in ("check", "base", "smoke", "train", "score")},
               "total_gpu_spend_usd": .5, "artifacts": artifacts}
    payload = json.dumps(receipt).encode()
    completion = tmp_path / f"training-completion-{hashlib.sha256(payload).hexdigest()}.json"
    completion.write_bytes(payload)
    _training_receipt(completion, runs, readiness, "d" * 64, partition)
    (directory / changed).write_text("changed after scoring completed")
    with pytest.raises(ValueError, match="completed training artifact changed"):
        _training_receipt(completion, runs, readiness, "d" * 64, partition)
