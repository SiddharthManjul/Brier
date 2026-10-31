from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from PIL import Image

from brier_explain import sam, spark

_TELEMETRY = {"collision_steps", "action_direction_changes", "action_transitions",
              "object_max_height", "object_final_height", "goal_distance_min",
              "goal_distance_final", "steps", "reward_sum"}
_SCENARIO = {"pose_bin", "lighting", "camera_jitter", "physics_jitter"}


def _root() -> Path:
    value = os.environ.get("BRIER_DATA", "")
    if not value or "://" in value or not Path(value).expanduser().is_dir():
        raise ValueError("BRIER_DATA must name an existing local directory")
    return Path(value).expanduser().resolve()


def _identity(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("episode id must be a canonical UUIDv4")
    try:
        identifier = UUID(value)
    except ValueError:
        raise ValueError("episode id must be a canonical UUIDv4") from None
    if identifier.version != 4 or str(identifier) != value:
        raise ValueError("episode id must be a canonical UUIDv4")
    return value


def _path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("frame paths must be absolute PNG paths inside BRIER_DATA")
    path = Path(value).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("frame paths must be files inside BRIER_DATA")
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError("frame paths must contain actual PNG images")
        image.verify()
    return path


def _frames(value: Any, root: Path, *, overlays: bool = False, exact: bool = True) -> list[dict[str, Any]]:
    if not isinstance(value, list) or (len(value) != 8 if exact else not 0 <= len(value) <= 8):
        raise ValueError("episodes require eight frames; escalation accepts zero to eight")
    fields = {"index", "path", "overlay_path"} if overlays else {"index", "path"}
    result, paths = [], set()
    for index, frame in enumerate(value):
        if (
            not isinstance(frame, dict) or set(frame) != fields
            or type(frame["index"]) is not int or frame["index"] != index
        ):
            raise ValueError("frames must have consecutive indices beginning at zero and exact fields")
        row = {"index": index}
        for key in sorted(fields - {"index"}):
            path = _path(frame[key], root)
            if path in paths:
                raise ValueError("raw and overlay frames must reference distinct files")
            paths.add(path)
            row[key] = str(path)
        result.append(row)
    return result


def _observed(episode: dict[str, Any]) -> None:
    if "seed" in episode and (type(episode["seed"]) is not int or episode["seed"] < 0):
        raise ValueError("episode seed must be a nonnegative integer")
    if "env" in episode and episode["env"] not in ("gym_pusht/PushT-v0", "gym_aloha/AlohaTransferCube-v0"):
        raise ValueError("unsupported observed environment")
    if "scenario" in episode:
        value = episode["scenario"]
        if not isinstance(value, dict) or set(value) - _SCENARIO:
            raise ValueError("scenario metadata must contain observed scenario parameters only")
        for key, item in value.items():
            if ((key == "pose_bin" and (type(item) is not int or not 0 <= item <= 3))
                or (key == "lighting" and item not in ("normal", "low"))
                or (key in {"camera_jitter", "physics_jitter"} and type(item) is not bool)):
                raise ValueError("invalid observed scenario parameter")
    if "telemetry" in episode:
        value = episode["telemetry"]
        if not isinstance(value, dict) or set(value) - _TELEMETRY or any(
            type(item) not in (int, float) or not math.isfinite(item) for item in value.values()
        ):
            raise ValueError("telemetry must contain finite observed numeric measurements only")


def _segment(episode: dict[str, Any], root: Path) -> dict[str, Any]:
    directory = (root / "public" / "overlays" / episode["id"]).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("overlay output must reside inside BRIER_DATA")
    directory.mkdir(parents=True, exist_ok=True)
    output, final_masks = [], []
    for frame in episode["frames"]:
        path = Path(frame["path"])
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB")).copy()
        masks = []
        for concept in ("object", "gripper", "target"):
            instances = sam.segment_image(path, episode["sam_prompts"][concept])
            if not isinstance(instances, list) or not instances:
                raise ValueError(f"SAM found no {concept} mask for episode {episode['id']} frame {frame['index']}")
            union = np.zeros(pixels.shape[:2], dtype=np.bool_)
            for mask in instances:
                sam.mask_geometry(mask)
                if mask.shape != union.shape:
                    raise ValueError("SAM masks must use original-image coordinates")
                union |= mask.astype(np.bool_)
            masks.append(union)
        for mask, color in zip(masks, ((255, 64, 64), (64, 128, 255), (64, 255, 128)), strict=True):
            pixels[mask] = (pixels[mask] * 0.55 + np.asarray(color) * 0.45).astype(np.uint8)
        target = directory / f"overlay-{frame['index']}.png"
        if target.resolve() in {Path(item["path"]) for item in episode["frames"]}:
            raise ValueError("overlay output cannot replace an input frame")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".png", delete=False) as stream:
                temporary = Path(stream.name)
                Image.fromarray(pixels).save(stream, format="PNG")
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        output.append({**frame, "overlay_path": str(target)})
        final_masks = masks
    return {"id": episode["id"], "frames": output, "geometry": sam.failure_geometry(*final_masks),
            "model": "sam-3.1"}


def process(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one validated bridge operation without constructing any model prompt text."""
    if not isinstance(payload, dict) or payload.get("operation") not in ("segment", "explain", "escalate"):
        raise ValueError("operation must be segment, explain or escalate")
    operation, root = payload["operation"], _root()
    if operation == "escalate":
        if set(payload) != {"operation", "prompt", "frames"}:
            raise ValueError("escalate requires exactly operation, prompt and frames")
        frames = _frames(payload["frames"], root, exact=False)
        return spark.escalate(payload["prompt"], [Path(frame["path"]) for frame in frames])
    if set(payload) != {"operation", "episodes"} or not isinstance(payload["episodes"], list) or not payload["episodes"]:
        raise ValueError("segment/explain require a nonempty episodes array")
    episodes, identifiers = [], set()
    for source in payload["episodes"]:
        required = {"id", "frames", "sam_prompts"} if operation == "segment" else {"id", "prompt", "frames"}
        allowed = required | {"seed", "env", "scenario", "telemetry"} if operation == "segment" else required
        if not isinstance(source, dict) or not required <= set(source) or set(source) - allowed:
            raise ValueError("episode contains missing or unsupported fields")
        identifier = _identity(source["id"])
        if identifier in identifiers:
            raise ValueError("duplicate episode id")
        identifiers.add(identifier)
        frames = _frames(source["frames"], root, overlays=operation == "explain")
        if operation == "segment":
            prompts = source["sam_prompts"]
            if not isinstance(prompts, dict) or set(prompts) != {"object", "gripper", "target"} or any(
                not isinstance(value, str) or not value.strip() for value in prompts.values()
            ):
                raise ValueError("sam_prompts requires three nonempty Rust-owned concepts")
            _observed(source)
        elif not isinstance(source["prompt"], str) or not source["prompt"].strip():
            raise ValueError("explain requires a nonempty Rust-compiled prompt")
        episodes.append({**source, "frames": frames})
    output = []
    for episode in episodes:
        if operation == "segment":
            output.append(_segment(episode, root))
        else:
            paths = [Path(frame[key]) for frame in episode["frames"] for key in ("path", "overlay_path")]
            output.append({"id": episode["id"], **spark.explain(episode["prompt"], paths)})
    return {"episodes": output}


def main() -> None:
    """Read one JSON operation from stdin and write one JSON result to stdout."""
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = dict(items)
        if len(result) != len(items):
            raise ValueError("bridge JSON contains duplicate fields")
        return result

    def invalid(value: str) -> None:
        raise ValueError("bridge JSON contains a nonfinite number")

    try:
        payload = json.load(sys.stdin, object_pairs_hook=unique, parse_constant=invalid)
        result = process(payload)
        sys.stdout.write(json.dumps(result, allow_nan=False) + "\n")
    except (OSError, ValueError, RuntimeError) as failure:
        sys.stderr.write(f"Brier explanation failed: {failure}\n")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
