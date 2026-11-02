from __future__ import annotations

import hashlib
import json
import runpy
import sqlite3
from pathlib import Path

import pytest
from brier_ml.glimmer_data import calibration_partition


def _verify_script() -> str:
    workflow = (Path(__file__).parents[2] / ".github/workflows/robot-ci.yml").read_text()
    step = workflow.split("      - name: Verify the prepared decision bundle before evaluation\n", 1)[1]
    script = step.split("          python - <<'PY'\n", 1)[1].split("          PY\n", 1)[0]
    return "\n".join(line.removeprefix("          ") for line in script.splitlines())


def _bundle(directory: Path) -> dict:
    calibration = [f"c{index:03}" for index in range(1, 160)
                   if int.from_bytes(hashlib.sha256(f"c{index:03}".encode()).digest()[:4], "big") % 10 in (6, 7)][:26]
    fit, held = calibration_partition(calibration)
    training = ["c002"]
    assert int.from_bytes(hashlib.sha256(b"c002").digest()[:4], "big") % 10 < 6
    compiled = [{"id": f"{change}:pusht-s01:regressed:ev", "pair_id": f"{change}:pusht-s01",
                 "change_id": change, "scenario_id": "pusht-s01", "env": "pusht", "variant": "ev",
                 "split": "train" if change in training else "calibration", "question": "regressed",
                 "options": ["yes", "no"], "label": "A",
                 "prompt": "<system> test\n<state>\ntest\n<question> regressed\n<options> A) yes  B) no\n<answer>"}
                for change in training + calibration]
    version = "configured/model@" + "a" * 40 + "+lora@" + "b" * 64
    t1 = {"model_version": "t1@" + "c" * 64, "calibration_change_ids": fit, "training_change_ids": training}
    risk = {"model_version": version, "t1": t1, "probability_calibration_change_ids": fit,
            "changes": [{"change_id": identity} for identity in held]}
    definitions = [{"scenario_id": f"{env}-s{index:02}", "env": env, "seeds": list(range(index * 50, (index + 1) * 50)),
                    "baseline_success": .8, "params": {"pose_bin": (index - 1) // 4 + 1, "lighting": "normal",
                                                        "camera_jitter": False, **({"physics_jitter": False} if env == "aloha" else {})}}
                   for env in ("pusht", "aloha") for index in range(1, 17)]
    ledger = directory / "ledger.sqlite"
    with sqlite3.connect(ledger) as connection:
        connection.execute("CREATE TABLE scenarios(scenario_id TEXT,env TEXT,params_json TEXT,baseline_success REAL)")
        connection.executemany("INSERT INTO scenarios VALUES(?,?,?,?)", [
            (row["scenario_id"], "gym_pusht/PushT-v0" if row["env"] == "pusht" else "gym_aloha/AlohaTransferCube-v0",
             json.dumps(row["params"]), row["baseline_success"]) for row in definitions])
    contents = {"ledger.sqlite": ledger.read_bytes(),
                "compiled.jsonl": "".join(json.dumps(row) + "\n" for row in compiled).encode(),
                "scenarios.jsonl": "".join(json.dumps(row) + "\n" for row in definitions).encode(),
                "risk-calibration.json": json.dumps(risk).encode(), "t1.json": json.dumps(t1).encode(),
                "replay-scores.json": b'{}\n', "report.json": b'{}\n', "baseline-policy.yaml": b"env: pusht\n",
                "aloha-baseline-policy.yaml": b"env: aloha\n"}
    for name, data in contents.items():
        (directory / name).write_bytes(data)
    files = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    return {"schema": "brier-decision-bundle-v1", "files": files, "model_version": version,
            "data_sha256": files["compiled.jsonl"],
            "release_hashes": {"scenarios.jsonl": files["scenarios.jsonl"]},
            "training_completion_sha256": "d" * 64,
            "calibration_partition": {"fit_change_ids": fit, "risk_change_ids": held}}


@pytest.mark.parametrize("mutation", [None, "tampered", "missing", "traversal", "compiled", "model", "holdout", "t1", "swapped_fit", "wrong_train", "missing_receipt", "invalid_receipt"])
def test_workflow_rejects_unverified_artifacts_before_rollouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str | None,
) -> None:
    manifest = _bundle(tmp_path)
    if mutation == "tampered":
        (tmp_path / "compiled.jsonl").write_text("tampered\n")
    elif mutation == "missing":
        manifest["files"].pop("risk-calibration.json")
    elif mutation == "traversal":
        manifest["files"]["../compiled.jsonl"] = manifest["files"].pop("compiled.jsonl")
    elif mutation == "compiled":
        manifest["data_sha256"] = "f" * 64
    elif mutation == "model":
        manifest["model_version"] = "configured/model@main"
    elif mutation == "holdout":
        manifest["calibration_partition"]["risk_change_ids"][0] = "c020"
    elif mutation == "t1":
        (tmp_path / "t1.json").write_text("{}")
        manifest["files"]["t1.json"] = hashlib.sha256(b"{}").hexdigest()
    elif mutation in {"swapped_fit", "wrong_train"}:
        risk = json.loads((tmp_path / "risk-calibration.json").read_text())
        if mutation == "swapped_fit":
            partition = manifest["calibration_partition"]
            partition["fit_change_ids"][0], partition["risk_change_ids"][0] = (
                partition["risk_change_ids"][0], partition["fit_change_ids"][0])
            risk["probability_calibration_change_ids"] = partition["fit_change_ids"]
            risk["changes"] = [{"change_id": change} for change in partition["risk_change_ids"]]
            risk["t1"]["calibration_change_ids"] = partition["fit_change_ids"]
        else:
            risk["t1"]["training_change_ids"] = ["c999"]
        for name, value in (("risk-calibration.json", risk), ("t1.json", risk["t1"])):
            (tmp_path / name).write_text(json.dumps(value))
            manifest["files"][name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    elif mutation == "missing_receipt":
        manifest.pop("training_completion_sha256")
    elif mutation == "invalid_receipt":
        manifest["training_completion_sha256"] = "not-a-sha256"
    (tmp_path / "bundle-manifest.json").write_text(json.dumps(manifest))
    script = tmp_path / "verify-bundle.py"
    script.write_text(_verify_script())
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    if mutation:
        with pytest.raises(SystemExit):
            runpy.run_path(str(script))
    else:
        runpy.run_path(str(script))
