from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from brier_ml.calibrate import _score_rows
from brier_ml.dataset import _compiled_rows, load_dataset
from brier_ml.glimmer_data import calibration_partition
from brier_ml.t1 import predict_t1, train_t1

_SPLITS = ("calibration", "test", "ood")
_RELEASE = {"ledger.sqlite", "compiled.jsonl", "episodes.jsonl", "changes.jsonl",
            "scenarios.jsonl", "label-summary.json", "quality-gates.json", "SHA256SUMS"}


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path.name}")
    return value


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _release(dataset: Path, readiness: Path) -> tuple[dict, dict[str, str]]:
    receipt = _json(readiness)
    digest = _hash(readiness)
    if (readiness.name != f"training-readiness-{digest}.json"
            or receipt.get("schema") != "brier-training-readiness-v1"):
        raise ValueError("readiness must be an immutable checksum-bound training receipt")
    measured = receipt["measurements"]
    hashes = measured["release_hashes"]
    if (set(hashes) != _RELEASE or measured["checks_skipped"]
            or measured["gates"] != dict.fromkeys(("G1", "G2", "G3", "G4", "G5"), "passed")
            or [measured[key] for key in ("parquet_keys", "episodes", "labeled_pairs", "compiled_records")]
            != [5152, 60480, 2560, 15360]
            or not measured["rust_label_compile_replayed"] or not measured["golden_test_passed"]):
        raise ValueError("readiness does not prove every frozen rollout-v1 gate/count")
    for name, expected in hashes.items():
        if _hash(dataset / name) != expected:
            raise ValueError(f"released input hash changed: {name}")
    return measured, hashes


def _model(runs: Path, compiled_sha: str, partition: dict) -> tuple[dict, str, dict[str, str]]:
    directory = runs / "glimmer-ft-v1"
    meta = _json(directory / "run_meta.json")
    manifest = _json(directory / "adapter" / "brier-model.json")
    if (meta.get("data_sha256") != compiled_sha or meta.get("calibration_partition") != partition
            or meta.get("epochs") != 1 or meta.get("steps") != 249
            or meta["config"].get("epochs") != 1 or meta["config"].get("max_usd") != 12
            or meta["config"].get("max_steps") is not None
            or meta.get("revision") != "cec94fcfd6512f2cfe708138c32533b346ea4f69"
            or meta.get("base_repo") != "meta-models/Muse-Glimmer-30B"
            or manifest.get("step") != 249
            or manifest.get("config", {}).get("model_id") != meta["base_repo"]
            or manifest["config"].get("revision") != meta["revision"]
            or manifest["config"].get("dataset_digest") != compiled_sha
            or manifest["config"].get("recipe") != meta["config"]):
        raise ValueError("trained adapter must match the complete frozen one-epoch recipe")
    paths = [directory / "adapter" / name for name in
             ("adapter_config.json", "adapter_model.safetensors")]
    hashes = {str(path.relative_to(runs)): _hash(path) for path in paths}
    config = _json(paths[0])
    if config.get("peft_type") != "LORA" or config.get("r") != 16 or not paths[1].stat().st_size:
        raise ValueError("trained adapter must contain the expected nonempty rank-16 LoRA")
    adapter_digest = hashlib.sha256(b"".join(bytes.fromhex(hashes[str(path.relative_to(runs))])
                                           for path in paths)).hexdigest()
    return meta, f"{meta['base_repo']}@{meta['revision']}+lora@{adapter_digest}", hashes


def _training_receipt(path: Path, runs: Path, readiness: Path,
                      compiled_sha: str, partition: dict) -> str:
    receipt, digest = _json(path), _hash(path)
    if (path.name != f"training-completion-{digest}.json"
            or receipt.get("schema") != "brier-one-epoch-training-completion-v1"
            or receipt.get("dataset_readiness_receipt", {}).get("sha256") != _hash(readiness)
            or receipt.get("data_sha256") != compiled_sha or receipt.get("calibration_partition") != partition
            or receipt.get("base_revision") != "cec94fcfd6512f2cfe708138c32533b346ea4f69"
            or receipt.get("epochs") != 1 or receipt.get("training_steps") != 249
            or receipt.get("max_train_usd") != 12 or receipt.get("max_gpu_usd") != 22
            or receipt.get("active_training_apps") != [] or receipt.get("pending_paid_operation") is not None
            or receipt.get("training_process_exited") is not True or receipt.get("training_lock_free") is not True
            or any(receipt.get(key) is not False for key in
                   ("second_epoch_executed", "publication_authorized", "deployment_authorized"))):
        raise ValueError("training completion receipt does not bind the verified one-epoch run")
    caps = {"check": 1, "base": 3, "smoke": 3, "train": 12, "score": 2}
    spend = receipt.get("step_gpu_spend_usd", {})
    if (set(spend) != set(caps)
            or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= caps[step]
                   for step, value in spend.items())
            or not math.isclose(sum(spend.values()), receipt.get("total_gpu_spend_usd", -1), abs_tol=1e-12)
            or sum(spend.values()) > 22):
        raise ValueError("training completion receipt has invalid reconciled GPU spend")
    required = {"adapter/adapter_config.json", "adapter/adapter_model.safetensors", "adapter/brier-model.json",
                "run_meta.json", "temps.json", "metrics.json", "train_log.jsonl"}
    required.update(f"{folder}/{split}.jsonl" for folder in ("probs", "probs_temp") for split in _SPLITS)
    artifacts = receipt.get("artifacts", {})
    if not required <= artifacts.keys():
        raise ValueError("training completion receipt omits scored adapter artifacts")
    directory = runs / "glimmer-ft-v1"
    for name in sorted(required):
        source, expected = directory / name, artifacts[name]
        if _hash(source) != expected["sha256"] or source.stat().st_size != expected["bytes"]:
            raise ValueError(f"completed training artifact changed: {name}")
    return digest


def _scores(directory: Path, compiled: list[dict], revision: str,
            partition: dict, sources: dict[str, str], runs: Path, compiled_sha: str) -> dict[str, float]:
    report = _json(directory / "metrics.json")
    temperatures = _json(directory / "temps.json")
    if (report.get("revision") != revision or report.get("calibration_partition") != partition
            or report.get("temperatures") != temperatures):
        raise ValueError("score calibration provenance differs from the frozen model/partition")
    metadata = directory / ("run_meta.json" if directory.name == "glimmer-ft-v1" else "meta.json")
    if directory.name == "frozen-head":
        frozen = _json(metadata)
        if (frozen.get("compiled_sha256") != compiled_sha
                or frozen.get("calibration_partition") != partition or frozen.get("revision") != revision):
            raise ValueError("frozen head metadata differs from the verified dataset/partition")
    if metadata.exists():
        sources[str(metadata.relative_to(runs))] = _hash(metadata)
    result = {}
    for split in _SPLITS:
        path = directory / "probs_temp" / f"{split}.jsonl"
        rows = _score_rows(path)
        expected = {row["id"]: row for row in compiled if row["split"] == split}
        if {row["id"] for row in rows} != set(expected):
            raise ValueError(f"score coverage differs from compiled {split} IDs")
        for row in rows:
            example = expected[row["id"]]
            letters = dict(zip("ABCDEF", row["options"]))
            temperature = temperatures.get(row["question"])
            if (row["split"] != split or row["revision"] != revision
                    or row["model"] != report.get("model") or row.get("temperature") != temperature
                    or letters != example["options"] or row["label"] != example["label"]):
                raise ValueError("score metadata/label/options differ from Rust-compiled examples")
            logits = np.asarray(row["logits"][:len(letters)], dtype=float) / temperature
            weights = np.exp(logits - logits.max())
            probabilities = weights / weights.sum()
            if not np.allclose(probabilities, [row["probs"][key] for key in letters], atol=1e-12, rtol=1e-12):
                raise ValueError("scaled scores disagree with fitted temperatures")
            if row["question"] == "regressed" and row["variant"] == "ev":
                result[row["id"]] = row["probs"]["ABCDEF"[row["options"].index("yes")]]
        sources[str(path.relative_to(runs))] = _hash(path)
    for name in ("metrics.json", "temps.json"):
        path = directory / name
        sources[str(path.relative_to(runs))] = _hash(path)
    return result


def prepare_bundle(dataset: Path, runs: Path, readiness: Path, output: Path,
                   brier: Path, *, training_receipt: Path) -> dict[str, Any]:
    """Prepare a local checksummed decision/replay bundle from verified frozen inputs.

    Only the seven probability-fitting changes fit T1 isotonic calibration. The
    nineteen risk changes only calibrate Rust's policy. This CPU-only operation
    never calls model providers or changes the released dataset/trained adapter.
    """
    dataset, runs, readiness, output, brier, training_receipt = (path.resolve() for path in
        (dataset, runs, readiness, output, brier, training_receipt))
    if output.exists():
        raise FileExistsError("decision bundle output already exists")
    measured, release_hashes = _release(dataset, readiness)
    examples = _compiled_rows(dataset / "compiled.jsonl")
    data = load_dataset(dataset / "ledger.sqlite", dataset / "compiled.jsonl")
    fit, risk = calibration_partition(change for change, split in
                                      zip(data["changes"], data["splits"], strict=True)
                                      if split == "calibration")
    partition = {"fit_change_ids": fit, "risk_change_ids": risk}
    gates = _json(dataset / "quality-gates.json")
    if (fit != measured["calibration_fit_changes"] or risk != measured["calibration_risk_changes"]
            or gates["calibration_fit_changes"] != fit or gates["calibration_risk_changes"] != risk
            or len(fit) != 7 or len(risk) != 19 or len(data["ids"]) != 2560
            or len(examples) != 15360):
        raise ValueError("compiled calibration partition/counts differ from independent readiness")
    completion_sha = _training_receipt(training_receipt, runs, readiness,
                                       release_hashes["compiled.jsonl"], partition)
    meta, version, sources = _model(runs, release_hashes["compiled.jsonl"], partition)
    scores = {name: _scores(runs / directory, examples, meta["revision"], partition, sources, runs,
                           release_hashes["compiled.jsonl"])
              for name, directory in (("Glimmer zero-shot", "base"),
                                      ("Glimmer frozen head", "frozen-head"),
                                      ("Glimmer fine-tuned", "glimmer-ft-v1"))}
    output.mkdir(parents=True)
    t1 = train_t1(data, output / "t1.json", calibration_change_ids=fit)
    numeric = dict(zip(data["ids"], predict_t1(t1, data["features"]).tolist(), strict=True))
    ids = set(next(iter(scores.values())))
    scores["T1 only"] = {key: numeric[key] for key in sorted(ids)}
    scores["Full cascade"] = {key: scores["Glimmer fine-tuned"][key]
                              if .1 <= numeric[key] <= .9 else numeric[key] for key in sorted(ids)}
    training = sorted({change for change, split in zip(data["changes"], data["splits"], strict=True)
                       if split == "train"})
    replay = {"strategies": {name: {
        "model_version": t1["model_version"] if name == "T1 only" else version
        if name in {"Glimmer fine-tuned", "Full cascade"} else f"{name}@{meta['revision']}",
        "training_change_ids": [] if name == "Glimmer zero-shot" else training,
        "probability_calibration_change_ids": fit, "probabilities": values,
    } for name, values in scores.items()}}
    _write(output / "replay-scores.json", replay)
    cascade = scores["Full cascade"]
    groups: dict[str, list[dict]] = {}
    for key, change, target in zip(data["ids"], data["changes"], data["labels"], strict=True):
        if change in risk:
            groups.setdefault(change, []).append({"scenario_id": key.split(":")[1],
                                                 "probability": cascade[key], "regressed": bool(target)})
    _write(output / "risk-calibration.json", {
        "format_version": 1, "model_version": version, "t1": t1,
        "temperatures": _json(runs / "glimmer-ft-v1" / "temps.json"),
        "probability_calibration_change_ids": fit,
        "changes": [{"change_id": change, "pairs": sorted(groups[change], key=lambda p: p["scenario_id"])}
                    for change in risk],
    })
    for name in ("ledger.sqlite", "compiled.jsonl"):
        shutil.copyfile(dataset / name, output / name)
        if _hash(output / name) != release_hashes[name]:
            raise ValueError("bundle copy differs from released input")
    for environment, env_id, filename in (
        ("pusht", "gym_pusht/PushT-v0", "baseline-policy.yaml"),
        ("aloha", "gym_aloha/AlohaTransferCube-v0", "aloha-baseline-policy.yaml"),
    ):
        pin = measured["pins"]["policies"][environment]
        _write(output / filename, {
            "env": env_id, "policy": {"id": pin["repo_id"], "revision": pin["revision"]},
            "description": f"Frozen production {environment} baseline",
            "families": [{"family": "benign", "params": {}}],
        })
    try:
        subprocess.run([str(brier), "replay", "--ledger", str(output / "ledger.sqlite"),
                        "--compiled", str(output / "compiled.jsonl"), "--scores", str(output / "replay-scores.json"),
                        "--out", str(output / "report.json")], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Rust replay failed: {error.stderr.strip()}") from error
    report = _json(output / "report.json")
    cascade_result, heuristic = (report["strategies"][name]
                                 for name in ("Full cascade", "Cheap-tier heuristic"))
    result = {
        "schema": "brier-decision-bundle-v1", "readiness_sha256": _hash(readiness),
        "training_completion_sha256": completion_sha,
        "release_hashes": release_hashes, "model_version": version,
        "data_sha256": release_hashes["compiled.jsonl"],
        "calibration_partition": partition, "training_run_meta_sha256": _hash(runs / "glimmer-ft-v1" / "run_meta.json"),
        "model_artifact_hashes": sources,
        "preparation_code_sha256": {name: _hash(Path(__file__).with_name(name))
                                    for name in ("prepare.py", "t1.py", "dataset.py", "calibrate.py")},
        "replay_binary_sha256": _hash(brier),
        "files": {path.name: _hash(path) for path in sorted(output.iterdir())},
        "baseline_environments": ["pusht", "aloha"], "provider_calls": 0,
        "test_ece_target": .05,
        "test_ece_target_passed": cascade_result["ece"] <= .05,
        "quality_acceptance": {
            "test_ece": {"target": .05, "measured": cascade_result["ece"],
                         "passed": cascade_result["ece"] <= .05},
            "savings_vs_cheap_heuristic": {
                "measured": cascade_result["eval_hours_saved"],
                "cheap_heuristic": heuristic["eval_hours_saved"],
                "passed": cascade_result["eval_hours_saved"] > heuristic["eval_hours_saved"],
            },
            "replay_splits": {"minimum": 18, "total": 20,
                              "measured": cascade_result["guarantee_splits"],
                              "passed": cascade_result["guarantee_splits"] >= 18},
        },
        "deployment_authorized": False,
    }
    _write(output / "bundle-manifest.json", result)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a verified local Brier decision bundle without cloud compute")
    for name in ("dataset", "runs", "readiness", "training-receipt", "output", "brier"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare_bundle(args.dataset, args.runs, args.readiness, args.output, args.brier,
                   training_receipt=args.training_receipt)


if __name__ == "__main__":
    _main()
