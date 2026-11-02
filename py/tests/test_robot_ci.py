from __future__ import annotations

import hashlib
import json
import os
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
    risk = {"format_version": 2, "executed_changes": [{"change_id":identity} for identity in held], "model_version": version, "t1": t1, "probability_calibration_change_ids": fit,
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
            "release_hashes": {"scenarios.jsonl": files["scenarios.jsonl"], "compiled.jsonl": files["compiled.jsonl"]},
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


def test_four_real_matrix_waves_follow_completed_label_checks() -> None:
    import ast

    import yaml

    workflow = yaml.safe_load((Path(__file__).parents[2] / ".github/workflows/robot-ci.yml").read_text())
    jobs = workflow["jobs"]
    for wave in range(1, 5):
        job = jobs[f"wave{wave}"]
        assert job["strategy"]["max-parallel"] == 4
        assert "inputs.full" in job["if"] and "outputs.block" in job["if"]
        assert job["needs"] == ("evaluate" if wave == 1 else f"check{wave - 1}")
        assert jobs[f"check{wave}"]["needs"] == [("evaluate" if wave == 1 else f"check{wave - 1}"), f"wave{wave}"]
        calls = next(step for step in job["steps"] if step.get("name") == "Run one selected expensive scenario")["run"]
        script = calls.split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_episodes"
                   for node in ast.walk(ast.parse(script))) == 1
    assert jobs["finish"]["needs"] == "check4"
    assert jobs["finish"]["if"] == "needs.check4.result == 'success'"


@pytest.mark.parametrize("verdict", ["block", "pending"])
def test_each_wave_labels_only_its_new_artifacts_and_preserves_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: str,
) -> None:
    from types import SimpleNamespace

    import yaml

    jobs = yaml.safe_load((Path(__file__).parents[2] / ".github/workflows/robot-ci.yml").read_text())["jobs"]
    step = next(step for step in jobs["check1"]["steps"] if step.get("id") == "check")
    script = step["run"].split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    inputs = tmp_path / "brier-ci"
    inputs.mkdir()
    (inputs / "brier").write_bytes(b"fixture")
    (inputs / "card.json").write_text('{"change_id":"c999"}')
    (tmp_path / "decisions.json").write_text('{"decisions":[{"probabilities":{"policy":{"guarantee":"change"}}}]}')
    wave = tmp_path / "expensive/wave-2"
    wave.mkdir(parents=True)
    parquet = wave / "pusht-s05.parquet"
    parquet.write_bytes(b"actual rollout path placeholder")
    output = tmp_path / "outputs"
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0 if len(calls) == 1 else 1,
                               stdout=json.dumps({"verdict": verdict}), stderr="")

    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    monkeypatch.setenv("WAVE", "2")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    entry = tmp_path / "check.py"
    entry.write_text(script)
    runpy.run_path(str(entry))
    assert calls[0][0][1] == "label"
    assert calls[0][0][calls[0][0].index("--episodes") + 1] == str(wave)
    assert calls[1][0][1] == "report"
    assert output.read_text() == f"snapshot=state-2\nblock={str(verdict == 'block').lower()}\n"
    assert parquet.exists(), "append-only runs retain their original Parquet URI for explanation"


def test_new_heldout_changes_do_not_refit_the_original_probability_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _bundle(tmp_path)
    compiled = tmp_path / "compiled.jsonl"
    source = compiled.read_text()
    new_id = next(f"c{n:03}" for n in range(161, 999) if int.from_bytes(hashlib.sha256(f"c{n:03}".encode()).digest()[:4], "big") % 10 in (6,7))
    row = json.loads(source.splitlines()[1])
    for key in ("id", "pair_id"):
        row[key] = row[key].replace(row["change_id"], new_id)
    row["change_id"] = new_id
    compiled.write_text(source + json.dumps(row) + "\n")
    risk = json.loads((tmp_path / "risk-calibration.json").read_text())
    for key in ("changes", "executed_changes"):
        risk[key].append({"change_id": new_id})
    (tmp_path / "risk-calibration.json").write_text(json.dumps(risk))
    for name in ("compiled.jsonl", "risk-calibration.json"):
        manifest["files"][name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    manifest["data_sha256"] = manifest["files"]["compiled.jsonl"]
    (tmp_path / "bundle-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    entry = tmp_path / "verify.py"
    entry.write_text(_verify_script())
    runpy.run_path(str(entry))


def _rollout_step(*, name: str | None = None, identity: str | None = None) -> dict:
    import yaml

    jobs = yaml.safe_load((Path(__file__).parents[2] / ".github/workflows/rollouts.yml").read_text())["jobs"]
    return next(step for job in jobs.values() for step in job["steps"]
                if (name is not None and step.get("name") == name)
                or (identity is not None and step.get("id") == identity))


@pytest.mark.parametrize("override,phase,accepted", [
    ("", "pairs", True), ("", "baselines", True), ("", "pilot", True),
    ("gs://private_bucket/extensions/guide3-v1-35dab70aad0d", "pairs", True),
    ("gs://private_bucket/extensions/guide3-v1-35dab70aad0d/", "pairs", True),
    ("gs://private_bucket/extensions/guide3-v1-35dab70aad0d", "baselines", False),
    ("gs://private_bucket/extensions/guide3-v1-35dab70aad0d", "pilot", False),
    ("gs://private_bucket/extensions/guide3-v1-35dab70aad0d", "timing", False),
    ("gs://other_bucket/extensions/guide3-v1-35dab70aad0d", "pairs", False),
    ("gs://private_bucket", "pairs", False),
    ("gs://private_bucket/release/v1", "pairs", False),
    ("gs://private_bucket/extensions/guide3-x/episodes", "pairs", False),
    ("gs://private_bucket/extensions/../guide3-x", "pairs", False),
    ("gs://private_bucket/extensions/guide3-%2e%2e", "pairs", False),
    ("gs://private_bucket/extensions/guide3-x?query=1", "pairs", False),
    ("gs://private_bucket/extensions/guide3-x#fragment", "pairs", False),
    ("gs://private_bucket/extensions/guide3-x\\escape", "pairs", False),
    ("gs://private_bucket/extensions/guide3-x'$(touch nope)", "pairs", False),
    ("gs://private_bucket/extensions/guide3-x\ninjected=value", "pairs", False),
    ("file:///tmp/extension", "pairs", False),
    ("https://storage.googleapis.com/private_bucket/extensions/guide3-x", "pairs", False),
])
def test_rollout_root_override_is_isolated_and_validated_before_matrix_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: str, phase: str, accepted: bool,
) -> None:
    script = _rollout_step(identity="plan")["run"].split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    entry, output = tmp_path / "plan.py", tmp_path / "output"
    entry.write_text(script)
    for key, value in {"N": "20", "OFFSET": "0", "COUNT": "20", "PHASE": phase,
                       "DEFAULT_DATA": "gs://private_bucket", "DATA_OVERRIDE": override,
                       "GITHUB_OUTPUT": str(output)}.items():
        monkeypatch.setenv(key, value)
    if not accepted:
        with pytest.raises(SystemExit):
            runpy.run_path(str(entry))
        assert not output.exists()
    else:
        runpy.run_path(str(entry))
        lines = output.read_text().splitlines()
        assert json.loads(lines[0].removeprefix("shards=")) == list(range(20))
        assert lines[1] == "data_root=" + (override.rstrip("/") or "gs://private_bucket")
        assert os.environ["DEFAULT_DATA"] == "gs://private_bucket"


@pytest.mark.parametrize("outcome", ["valid", "missing", "wrong_source", "empty", "mismatch"])
def test_extension_manifest_and_farm_plan_are_required_before_worker_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    from brier_sim import farm

    step = _rollout_step(name="Verify isolated extension provenance before workers")
    assert step["if"] == "inputs.data_root != ''"
    script = step["run"].split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    entry = tmp_path / "extension-check.py"
    entry.write_text(script)
    root = "gs://private_bucket/extensions/guide3-v1-35dab70aad0d"
    manifest = {"source_root": "gs://other_bucket" if outcome == "wrong_source" else "gs://private_bucket",
                "appended_change_ids": {"aloha": ["c181"]}}
    calls = []

    def read(uri):
        calls.append(("read", uri))
        if outcome == "missing":
            raise FileNotFoundError(uri)
        return json.dumps(manifest).encode()

    def plan(env, phase):
        calls.append(("plan", env, phase))
        if outcome == "empty":
            return []
        return [[({"change_id": "c182" if outcome == "mismatch" else "c181"}, {}, "expensive")]]

    def worker(*args, **kwargs):
        pytest.fail("Provenance validation must never start a worker or write cloud data")

    monkeypatch.setattr(farm, "_read", read)
    monkeypatch.setattr(farm, "_guide_plan", plan)
    monkeypatch.setattr(farm, "ProcessPoolExecutor", worker)
    monkeypatch.setenv("BRIER_DATA", root)
    monkeypatch.setenv("SOURCE_DATA", "gs://private_bucket")
    monkeypatch.setenv("ROLLOUT_ENV", "aloha")
    if outcome == "valid":
        runpy.run_path(str(entry))
        assert calls == [("read", root + "/manifests/extension.json"), ("plan", "aloha", "pairs")]
    else:
        with pytest.raises(FileNotFoundError if outcome == "missing" else SystemExit):
            runpy.run_path(str(entry))
        if outcome in ("missing", "wrong_source"):
            assert len(calls) == 1


def test_rollout_dispatch_title_binds_every_assignment_and_isolated_root() -> None:
    import yaml

    workflow = yaml.safe_load((Path(__file__).parents[2] / ".github/workflows/rollouts.yml").read_text())
    title = workflow["run-name"]
    for key in ("env", "phase", "n_shards", "shard_offset", "shard_count", "data_root"):
        assert f"inputs.{key}" in title
    assert "'default'" in title
    assert workflow["env"]["BRIER_DATA"] == "${{ vars.BRIER_DATA }}"
    assert workflow["jobs"]["rollouts"]["env"]["BRIER_DATA"] == "${{ needs.setup.outputs.data_root }}"
