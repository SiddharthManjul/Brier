from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np
import pytest
from brier_explain import client, sam, spark
from PIL import Image


def episode(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    frames = []
    for index in range(8):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (6, 5), (80, 80, 80)).save(path)
        frames.append({"index": index, "path": str(path)})
    return {"id": str(uuid4()), "seed": 123, "env": "gym_pusht/PushT-v0",
            "scenario": {"pose_bin": 0, "lighting": "normal", "camera_jitter": False},
            "telemetry": {"steps": 300, "reward_sum": 1.0}, "frames": frames,
            "sam_prompts": {"object": "Rust object phrase", "gripper": "Rust gripper phrase", "target": "Rust target phrase"}}


def masks(monkeypatch, calls: list) -> None:
    def segment(path, concept):
        calls.append((path, concept))
        mask = np.zeros((5, 6), dtype=np.bool_)
        point = {"Rust object phrase": (3, 4), "Rust gripper phrase": (0, 0), "Rust target phrase": (1, 1)}[concept]
        mask[point] = True
        return [mask]

    monkeypatch.setattr(sam, "segment_image", segment, raising=False)


def test_segment_measures_actual_masks_and_explain_preserves_prompt_order(tmp_path, monkeypatch) -> None:
    source, calls = episode(tmp_path, monkeypatch), []
    masks(monkeypatch, calls)
    segmented = client.process({"operation": "segment", "episodes": [source]})["episodes"][0]
    assert len(calls) == 24 and {item[1] for item in calls} == set(source["sam_prompts"].values())
    assert segmented["id"] == source["id"] and segmented["model"] == "sam-3.1"
    assert segmented["geometry"] == {"final_object_offset": (3.0, 2.0), "gripper_object_distance": 5.0}
    for original, frame in zip(source["frames"], segmented["frames"], strict=True):
        assert frame["path"] == original["path"] and frame["index"] == original["index"]
        overlay = Path(frame["overlay_path"])
        assert overlay.is_relative_to(tmp_path / "public/overlays") and overlay.is_file()
        assert not np.array_equal(np.asarray(Image.open(overlay)), np.asarray(Image.open(original["path"])))
    sent = []

    def explain(prompt, paths):
        sent.append((prompt, paths))
        return {"model": "muse-spark-1.3-contributor", "explanation": {"evidence_frames": [7]}, "cost_usd": None}

    monkeypatch.setattr(spark, "explain", explain)
    prompt = "Rust\n EXACT geometry text"
    result = client.process({"operation": "explain", "episodes": [{"id": source["id"], "prompt": prompt,
                                                                     "frames": segmented["frames"]}]})
    assert result["episodes"][0]["id"] == source["id"]
    assert sent == [(prompt, [Path(frame[key]) for frame in segmented["frames"] for key in ("path", "overlay_path")])]


def test_bridge_escalation_sends_only_rust_prompt_and_actual_frames(tmp_path, monkeypatch) -> None:
    source, sent = episode(tmp_path, monkeypatch), []

    def escalate(prompt, paths):
        sent.append((prompt, paths))
        return {"model": "muse-spark-1.3-contributor", "recommend": "no_change", "rationale": "Observed state", "cost_usd": None}

    monkeypatch.setattr(spark, "escalate", escalate)
    result = client.process({"operation": "escalate", "prompt": "Rust advice prompt", "frames": source["frames"][:2]})
    assert result["recommend"] == "no_change" and result["cost_usd"] is None
    assert sent == [("Rust advice prompt", [Path(frame["path"]) for frame in source["frames"][:2]])]


@pytest.mark.parametrize("mutation", ["hidden", "cause", "frame_count", "frame_index", "duplicate_path",
                                     "bad_id", "duplicate_id", "sam_prompt", "nan", "outside", "seed"])
def test_segment_validation_happens_before_any_api_call(tmp_path, monkeypatch, mutation) -> None:
    source, calls = episode(tmp_path, monkeypatch), []
    masks(monkeypatch, calls)
    payload = {"operation": "segment", "episodes": [source]}
    if mutation == "hidden":
        source["change_card"] = {"family": "action_latency"}
    elif mutation == "cause":
        source["telemetry"]["cause_family"] = "hidden"
    elif mutation == "frame_count":
        source["frames"].pop()
    elif mutation == "frame_index":
        source["frames"][0]["index"] = True
    elif mutation == "duplicate_path":
        source["frames"][1]["path"] = source["frames"][0]["path"]
    elif mutation == "bad_id":
        source["id"] = "c001"
    elif mutation == "duplicate_id":
        payload["episodes"].append(copy.deepcopy(source))
    elif mutation == "sam_prompt":
        del source["sam_prompts"]["target"]
    elif mutation == "nan":
        source["telemetry"]["steps"] = float("nan")
    elif mutation == "seed":
        source["seed"] = -1
    else:
        monkeypatch.setenv("BRIER_DATA", str(tmp_path / "restricted"))
        (tmp_path / "restricted").mkdir()
    with pytest.raises(ValueError):
        client.process(payload)
    assert not calls


def test_segment_missing_masks_and_wrong_coordinates_fail_without_invented_geometry(tmp_path, monkeypatch) -> None:
    source = episode(tmp_path, monkeypatch)
    for invalid in ([], [np.ones((2, 2), dtype=np.bool_)]):
        monkeypatch.setattr(sam, "segment_image", lambda path, concept, invalid=invalid: invalid, raising=False)
        with pytest.raises(ValueError, match="no object mask|original-image coordinates"):
            client.process({"operation": "segment", "episodes": [source]})


def test_main_emits_one_json_result_and_clear_nonzero_failure(tmp_path, monkeypatch) -> None:
    episode(tmp_path, monkeypatch)
    monkeypatch.setattr(spark, "escalate", lambda prompt, frames: {"recommend": "run_expensive", "cost_usd": None})
    output, failure = io.StringIO(), io.StringIO()
    monkeypatch.setattr(client.sys, "stdout", output)
    monkeypatch.setattr(client.sys, "stderr", failure)
    monkeypatch.setattr(client.sys, "stdin", io.StringIO(json.dumps({"operation": "escalate", "prompt": "Rust", "frames": []})))
    client.main()
    assert json.loads(output.getvalue()) == {"recommend": "run_expensive", "cost_usd": None}
    monkeypatch.setattr(client.sys, "stdin", io.StringIO('{"operation":"unsupported"}'))
    with pytest.raises(SystemExit) as caught:
        client.main()
    assert caught.value.code == 1 and "operation must be" in failure.getvalue()


@pytest.mark.parametrize("value", ['{"operation":"segment","operation":"escalate"}',
                                  '{"operation":"escalate","prompt":NaN,"frames":[]}'])
def test_main_rejects_ambiguous_or_nonfinite_input(monkeypatch, value) -> None:
    failure = io.StringIO()
    monkeypatch.setattr(client.sys, "stdin", io.StringIO(value))
    monkeypatch.setattr(client.sys, "stderr", failure)
    with pytest.raises(SystemExit) as caught:
        client.main()
    assert caught.value.code == 1 and ("duplicate" in failure.getvalue() or "nonfinite" in failure.getvalue())


def _guide_run(tmp_path: Path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from brier_sim import storage

    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    source = tmp_path / "source"
    catalog = source / "manifests/scenarios_pusht.json"
    catalog.parent.mkdir(parents=True)
    params = {"pose_bin": 1, "lighting": "normal", "camera_jitter": False}
    catalog.write_text(json.dumps([{"scenario_id": "pusht-s01", "params": params, "seeds": list(range(50))}]))
    pin = json.loads(Path(storage.__file__).with_name("pins.json").read_text())
    rows = []
    for index in range(20):
        folder = source / f"public/keyframes/pusht/c001/pusht-s01/{index}"
        row = dict.fromkeys(storage._SCHEMA.names)
        row.update(env="pusht", policy_repo=pin["policies"]["pusht"]["repo_id"],
                   policy_revision=pin["policies"]["pusht"]["revision"], change_id="c001", scenario_id="pusht-s01",
                   tier="expensive", episode_idx=index, seed=index, success=index < 19, steps=300, max_steps=300,
                   reward_sum=1., max_reward=.9, final_reward=.2, coverage_final=.2, coverage_max=.8,
                   block_disp_px=20., action_flip_rate=.1, terminal_state_json='{"cause_family":"HIDDEN"}',
                   keyframes_uri=str(folder) if index == 19 else None, wall_seconds=1.,
                   lerobot_version=pin["lerobot_version"], code_commit="a" * 40, host="fixture")
        rows.append(row)
        if index == 19:
            folder.mkdir(parents=True)
            for frame in range(1, 9):
                Image.new("RGB", (8, 7), (frame, 10, 20)).save(folder / f"f{frame}.jpg", format="JPEG")
    path = source / "episodes/env=pusht/tier=expensive/change=c001/pusht-s01.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=storage._SCHEMA), path)
    run = {"run_id": str(uuid5(NAMESPACE_URL, "c001:pusht-s01:expensive")), "change_id": "c001",
           "scenario_id": "pusht-s01", "tier": "expensive", "episodes": 20, "successes": 19,
           "mean_reward": 1., "uri": str(path), "wall_seconds": 20., "env": "gym_pusht/PushT-v0", "scenario": params}
    return run, rows, path


def test_guide_evidence_validates_real_rows_and_preserves_jpeg_bytes_without_hidden_causes(tmp_path, monkeypatch):
    run, _, _ = _guide_run(tmp_path, monkeypatch)
    result = client.process({"operation": "guide_evidence", "run": run})
    assert result["env"] == run["env"] and result["scenario"] == run["scenario"]
    assert len(result["episodes"]) == 1 and result["episodes"][0]["seed"] == 19
    frames = result["episodes"][0]["frames"]
    assert len(frames) == 8 and [frame["index"] for frame in frames] == list(range(8))
    for index, frame in enumerate(frames, 1):
        source = tmp_path / f"source/public/keyframes/pusht/c001/pusht-s01/19/f{index}.jpg"
        assert Path(frame["path"]).read_bytes() == source.read_bytes()
        assert Image.open(frame["path"]).format == "JPEG"
    assert "HIDDEN" not in json.dumps(result) and "terminal_state" not in json.dumps(result)
    assert "cause_family" not in json.dumps(result) and "change_id" not in json.dumps(result)
    assert len(result["episodes"][0]["telemetry"]) == 13


@pytest.mark.parametrize("mutation", ["seed", "identity", "aggregate", "uuid", "frame_prefix", "schema"])
def test_invalid_guide_group_fails_before_hydrating_frames(tmp_path, monkeypatch, mutation):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from brier_sim import storage

    run, rows, path = _guide_run(tmp_path, monkeypatch)
    if mutation == "seed":
        rows[0]["seed"] = 999
    elif mutation == "identity":
        rows[0]["policy_revision"] = "b" * 40
    elif mutation == "aggregate":
        run["mean_reward"] = 2.
    elif mutation == "uuid":
        run["run_id"] = str(uuid4())
    elif mutation == "frame_prefix":
        rows[-1]["keyframes_uri"] += "/unexpected"
    if mutation == "schema":
        pq.write_table(pa.Table.from_pylist([{"unexpected": 1}]), path)
    else:
        pq.write_table(pa.Table.from_pylist(rows, schema=storage._SCHEMA), path)
    with pytest.raises(ValueError, match="Guide"):
        client.process({"operation": "guide_evidence", "run": run})
    assert not (tmp_path / "public/keyframes/guide").exists()


def test_guide_cloud_reads_pin_generation_verify_hash_and_never_write(tmp_path, monkeypatch):
    import base64
    import hashlib

    from brier_sim import storage

    payload, calls = b"immutable artifact", []

    class Blob:
        generation = "123"
        size = len(payload)
        md5_hash = base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode()

        def reload(self, **kwargs):
            calls.append(("reload", kwargs))

        def download_as_bytes(self, **kwargs):
            calls.append(("download", kwargs))
            return payload

    blob = Blob()
    monkeypatch.setattr(storage, "_blob", lambda location: blob)
    assert client._guide_bytes("gs://fixture/allowed/object", tmp_path, 100) == payload
    assert calls == [("reload", {"retry": None}), ("download", {
        "if_generation_match": 123, "checksum": "auto", "retry": None})]
    blob.md5_hash = "incorrect"
    with pytest.raises(ValueError, match="generation checksum"):
        client._guide_bytes("gs://fixture/allowed/object", tmp_path, 100)
    calls.clear()
    with pytest.raises(ValueError, match="bounded generation"):
        client._guide_bytes("gs://fixture/allowed/object", tmp_path, 1)
    assert calls == [("reload", {"retry": None})]


def test_new_live_guide_cheap_evidence_uses_three_actual_final_pngs(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from brier_sim import storage

    run, rows, _ = _guide_run(tmp_path, monkeypatch)
    catalog = tmp_path / "source/manifests/scenarios_pusht.json"
    (tmp_path / "scenarios.jsonl").write_text(json.dumps(json.loads(catalog.read_text())[0]) + "\n")
    rows = rows[:3]
    for row in rows:
        relative = Path(f"public/keyframes/pusht/c001/pusht-s01/{row['seed']}/{uuid4()}")
        folder = tmp_path / relative
        folder.mkdir(parents=True)
        Image.new("RGB", (8, 7)).save(folder / "frame-final.png")
        row.update(tier="cheap", keyframes_uri=relative.as_posix())
    path = tmp_path / "cheap/pusht-s01.parquet"
    path.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=storage._SCHEMA), path)
    run.update(run_id=str(uuid4()), tier="cheap", episodes=3, successes=3, uri=str(path), wall_seconds=3.)
    observed = client.process({"operation": "guide_evidence", "run": run})
    assert len(observed["episodes"]) == 3
    assert [item["seed"] for item in observed["episodes"]] == [0, 1, 2]
    assert all(len(item["frames"]) == 1 for item in observed["episodes"])
    assert all(Path(item["frames"][0]["path"]).name == "frame-final.png" for item in observed["episodes"])
    rows[0]["keyframes_uri"] = None
    pq.write_table(pa.Table.from_pylist(rows, schema=storage._SCHEMA), path)
    with pytest.raises(ValueError, match="actual final frame"):
        client.process({"operation": "guide_evidence", "run": run})
