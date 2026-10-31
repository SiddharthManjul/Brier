from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from urllib import error, request

import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    decode_mask_to_raster,
    image_segmentation_format,
)
from PIL import Image


def segment_image(path: Path, prompt: str) -> list[np.ndarray]:
    """Segment an actual PNG with Rust's exact noun prompt into source-coordinate binary masks."""
    key = os.environ.get("META_API_KEY", "").strip()
    if not key:
        raise ValueError("META_API_KEY is required")
    if any(not 0x21 <= ord(character) <= 0x7e for character in key):
        raise ValueError("META_API_KEY must contain printable ASCII without whitespace")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("a nonempty Rust-compiled SAM prompt is required")
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG":
            raise ValueError("SAM input must be an actual PNG image")
        width, height = image.size
        image.verify()
    image_url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    payload = {"model": "sam-3.1", "input": [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": image_url},
    ]}], "metadata": {"mask_encoding": "one_bit"}}
    message = request.Request("https://api.meta.ai/v1/responses", method="POST",
                              data=json.dumps(payload, allow_nan=False).encode(),
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    opener = request.OpenerDirector()
    for handler in (request.HTTPSHandler(), request.HTTPDefaultErrorHandler(), request.HTTPErrorProcessor()):
        opener.add_handler(handler)
    try:
        with opener.open(message, timeout=60) as response:
            body = json.load(response)
    except error.HTTPError as failure:
        failure.close()
        raise RuntimeError(f"SAM request failed with HTTP {failure.code}") from None
    except (error.URLError, TimeoutError) as failure:
        raise RuntimeError("SAM request failed without a completed response") from failure
    if (
        not isinstance(body, dict) or body.get("model") != "sam-3.1"
        or body.get("status") != "completed" or body.get("error") is not None
    ):
        raise ValueError("SAM response must identify the requested model and complete successfully")
    output = body.get("output")
    if not isinstance(output, list) or not output:
        raise ValueError("SAM response requires an output message")
    texts = []
    for item in output:
        if (
            not isinstance(item, dict) or item.get("type") != "message"
            or item.get("status") != "completed" or not isinstance(item.get("content"), list)
        ):
            raise ValueError("SAM output message did not complete successfully")
        for part in item["content"]:
            if not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                raise ValueError("SAM output content must contain segmentation text")
            texts.append(part["text"])
    if len(texts) != 1:
        raise ValueError("SAM response requires exactly one segmentation text lane")
    parser = image_segmentation_format().create_parser()
    parser.push(texts[0])
    result = parser.finish(CompletedOutcome()).result
    if any(diagnostic.severity == "error" for diagnostic in result.diagnostics) or any(
        record.kind == "text" for record in result.records
    ):
        raise ValueError("SAM output contains invalid segmentation records")
    masks, identities = [], set()
    for record in result.records:
        if record.kind != "mask":
            continue
        if record.object_id in identities:
            raise ValueError("SAM response contains a duplicate object id")
        identities.add(record.object_id)
        left, top, right, bottom = (int(getattr(record.bounds, name)) for name in ("left", "top", "right", "bottom"))
        if (
            not 0 <= left < right <= width or not 0 <= top < bottom <= height
            or record.mask.width != right - left or record.mask.height != bottom - top
        ):
            raise ValueError("SAM mask bounds must match their crop within the source image")
        crop = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8).reshape(
            record.mask.height, record.mask.width
        )
        mask = np.zeros((height, width), dtype=np.bool_)
        mask[top:bottom, left:right] = crop
        if not mask.any():
            raise ValueError("SAM object mask contains no foreground pixels")
        masks.append(mask)
    return masks


def mask_geometry(mask: np.ndarray) -> dict[str, object]:
    """Return pixel area, pixel-center centroid, and half-open xyxy bounds."""
    if (
        not isinstance(mask, np.ndarray) or mask.ndim != 2
        or any(size == 0 for size in mask.shape)
    ):
        raise ValueError("SAM mask must be a nonempty two-dimensional array")
    if mask.dtype != np.bool_ and not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("SAM mask must contain binary boolean or integer values")
    if not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError("SAM mask values must be zero or one")
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("SAM mask contains no foreground pixels")
    return {
        "area": int(xs.size),
        "centroid": (float(xs.mean() + 0.5), float(ys.mean() + 0.5)),
        "bbox": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
    }


def failure_geometry(
    object_mask: np.ndarray, gripper_mask: np.ndarray, target_mask: np.ndarray
) -> dict[str, object]:
    """Measure final object-to-target xy offset and gripper-object distance in pixels."""
    masks = (object_mask, gripper_mask, target_mask)
    if not all(isinstance(mask, np.ndarray) for mask in masks) or any(
        mask.shape != object_mask.shape for mask in masks
    ):
        raise ValueError("final-frame object, gripper and target masks must have matching shapes")
    object_center, gripper_center, target_center = (
        np.asarray(mask_geometry(mask)["centroid"], dtype=np.float64) for mask in masks
    )
    offset = object_center - target_center
    return {
        "final_object_offset": (float(offset[0]), float(offset[1])),
        "gripper_object_distance": float(np.linalg.norm(gripper_center - object_center)),
    }
