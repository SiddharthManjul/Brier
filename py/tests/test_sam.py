from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from urllib import error

import numpy as np
import pytest
from brier_explain import sam
from brier_explain.sam import failure_geometry, mask_geometry
from PIL import Image

# Documented one_bit fixture from meta-sam-parser 0.1.0, offset into an 8×7 source image.
_TEXT = "<0f>7<|box;x1=2;y1=1;x2=6;y2=5;w=8;h=7|><|mask;x=0;y=0;data=5,5,!!!!!(QO(0lu8?|>"


def _body(text: str = _TEXT) -> dict:
    return {"model": "sam-3.1", "status": "completed", "output": [{
        "type": "message", "status": "completed", "content": [{"type": "output_text", "text": text}],
    }]}


def _image(path: Path) -> Path:
    Image.new("RGB", (8, 7)).save(path, format="PNG")
    return path


def test_segment_image_sends_exact_prompt_and_decodes_official_mask_in_source_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("META_API_KEY", "unit-test-key")
    path = _image(tmp_path / "frame.png")
    calls = []

    def completed(opener, message, *, timeout):
        calls.append((opener, message, timeout))
        return io.BytesIO(json.dumps(_body()).encode())

    monkeypatch.setattr(sam.request.OpenerDirector, "open", completed)
    masks = sam.segment_image(path, "  verbatim Rust noun  ")
    assert len(masks) == 1 and masks[0].dtype == np.bool_ and masks[0].shape == (7, 8)
    expected = np.zeros((7, 8), dtype=np.bool_)
    expected[1:6, 2:7] = [[1, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0],
                           [1, 0, 1, 1, 1], [1, 0, 0, 0, 1]]
    np.testing.assert_array_equal(masks[0], expected)
    assert len(calls) == 1
    opener, message, timeout = calls[0]
    assert timeout == 60 and message.full_url == "https://api.meta.ai/v1/responses"
    assert message.method == "POST" and message.get_header("Authorization") == "Bearer unit-test-key"
    assert not any(isinstance(handler, sam.request.HTTPRedirectHandler) for handler in opener.handlers)
    assert json.loads(message.data) == {"model": "sam-3.1", "input": [{
        "type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "  verbatim Rust noun  "},
            {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()},
        ],
    }], "metadata": {"mask_encoding": "one_bit"}}


@pytest.mark.parametrize("text, count", [("", 0), (_TEXT + "\n" + _TEXT.replace("<0f>7", "<0f>9"), 2)])
def test_segment_image_preserves_zero_and_multiple_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, count: int,
) -> None:
    monkeypatch.setenv("META_API_KEY", "unit-test-key")
    monkeypatch.setattr(sam.request.OpenerDirector, "open", lambda *a, **kw: io.BytesIO(json.dumps(_body(text)).encode()))
    assert len(sam.segment_image(_image(tmp_path / "frame.png"), "object")) == count


@pytest.mark.parametrize("mutation", ["model", "incomplete", "error", "output", "message", "refusal",
                                    "text", "lanes", "malformed", "crop", "bounds", "duplicate", "prose"])
def test_segment_image_rejects_incomplete_malformed_or_misregistered_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    body = _body()
    content = body["output"][0]["content"]
    if mutation == "model":
        body["model"] = "unrequested-model"
    elif mutation == "incomplete":
        body["status"] = "incomplete"
    elif mutation == "error":
        body["error"] = {"message": "failed"}
    elif mutation == "output":
        body["output"] = None
    elif mutation == "message":
        body["output"][0]["status"] = "incomplete"
    elif mutation == "refusal":
        content[0]["type"] = "refusal"
    elif mutation == "text":
        content[0]["text"] = None
    elif mutation == "lanes":
        content.append(content[0].copy())
    else:
        content[0]["text"] = {
            "malformed": _TEXT.replace("!!!!!(QO(0lu8?", "!bad"),
            "crop": _TEXT.replace("x2=6", "x2=5"),
            "bounds": _TEXT.replace("x2=6;", "x2=8;").replace("w=8", "w=9"),
            "duplicate": _TEXT + "\n" + _TEXT,
            "prose": "No segmentation data available.",
        }[mutation]
    monkeypatch.setenv("META_API_KEY", "unit-test-key")
    monkeypatch.setattr(sam.request.OpenerDirector, "open", lambda *a, **kw: io.BytesIO(json.dumps(body).encode()))
    with pytest.raises(ValueError, match="SAM"):
        sam.segment_image(_image(tmp_path / "frame.png"), "object")


@pytest.mark.parametrize("failure", [error.HTTPError("https://api.meta.ai/v1/responses", 302,
                                                   "private response", {}, io.BytesIO(b"private body")),
                                    error.URLError("private transport detail"), TimeoutError()])
def test_segment_image_fails_once_without_retry_or_exposing_http_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception,
) -> None:
    monkeypatch.setenv("META_API_KEY", "unit-test-key")
    calls = []

    def failed(*args, **kwargs):
        calls.append(1)
        raise failure

    monkeypatch.setattr(sam.request.OpenerDirector, "open", failed)
    with pytest.raises(RuntimeError, match="SAM request failed") as caught:
        sam.segment_image(_image(tmp_path / "frame.png"), "object")
    assert len(calls) == 1 and "private" not in str(caught.value)


def test_segment_image_checks_key_prompt_and_actual_png_before_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    path = _image(tmp_path / "frame.png")
    with pytest.raises(ValueError, match="META_API_KEY"):
        sam.segment_image(path, "object")
    monkeypatch.setenv("META_API_KEY", "unit-test-key")
    with pytest.raises(ValueError, match="Rust-compiled"):
        sam.segment_image(path, " ")
    Image.new("RGB", (8, 7)).save(path, format="JPEG")
    with pytest.raises(ValueError, match="actual PNG"):
        sam.segment_image(path, "object")


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int64])
def test_mask_geometry_uses_pixel_centers_and_half_open_bounds(dtype) -> None:
    mask = np.zeros((4, 6), dtype=dtype)
    mask[1:3, 2:5] = 1
    assert mask_geometry(mask) == {"area": 6, "centroid": (3.5, 2.0), "bbox": (2, 1, 5, 3)}


def test_failure_geometry_measures_final_object_target_and_gripper_offsets() -> None:
    object_mask, gripper_mask, target_mask = (np.zeros((5, 6), dtype=np.bool_) for _ in range(3))
    object_mask[3, 4], gripper_mask[0, 0], target_mask[1, 1] = True, True, True
    assert failure_geometry(object_mask, gripper_mask, target_mask) == {
        "final_object_offset": (3.0, 2.0), "gripper_object_distance": 5.0,
    }
    with pytest.raises(ValueError, match="matching shapes"):
        failure_geometry(object_mask, gripper_mask, target_mask[:2])


@pytest.mark.parametrize("mask", [np.zeros((2, 2), dtype=np.bool_), np.zeros((0, 2), dtype=np.bool_),
                                 np.ones((2, 2, 1), dtype=np.bool_), np.ones((2, 2), dtype=np.float32),
                                 np.array([[0, 255]], dtype=np.uint8), np.array([[0, -1]], dtype=np.int64)])
def test_mask_geometry_rejects_empty_or_nonbinary_masks(mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        mask_geometry(mask)
