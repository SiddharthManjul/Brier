from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from uuid import uuid4

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
