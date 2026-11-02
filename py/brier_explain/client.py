from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

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
        raise ValueError("frame paths must be absolute PNG or JPEG paths inside BRIER_DATA")
    path = Path(value).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("frame paths must be files inside BRIER_DATA")
    with Image.open(path) as image:
        if image.format not in ("PNG", "JPEG"):
            raise ValueError("frame paths must contain actual PNG or JPEG images")
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


def _guide_bytes(uri: str, root: Path, maximum: int) -> bytes:
    from brier_sim import storage

    location = storage._location(uri)
    if isinstance(location, Path):
        if not location.is_relative_to(root) or not location.is_file() or location.stat().st_size > maximum:
            raise ValueError("Guide evidence must be a bounded file inside BRIER_DATA")
        return location.read_bytes()
    blob = storage._blob(location)
    blob.reload(retry=None)
    if (not blob.generation or type(blob.size) is not int or not 0 < blob.size <= maximum
            or not isinstance(blob.md5_hash, str)):
        raise ValueError("Guide GCS evidence requires bounded generation/checksum metadata")
    payload = blob.download_as_bytes(if_generation_match=int(blob.generation), checksum="auto", retry=None)
    if (len(payload) != blob.size
            or base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode() != blob.md5_hash):
        raise ValueError("Guide GCS evidence bytes differ from their generation checksum")
    return payload


def _guide_evidence(run: dict[str, Any], root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from brier_sim import storage

    fields = {"run_id", "change_id", "scenario_id", "tier", "episodes", "successes",
              "mean_reward", "uri", "wall_seconds", "env", "scenario"}
    if not isinstance(run, dict) or set(run) != fields:
        raise ValueError("Guide evidence requires the complete authoritative ledger run")
    key = f"{run['change_id']}:{run['scenario_id']}:{run['tier']}"
    environments = {"gym_pusht/PushT-v0": "pusht", "gym_aloha/AlohaTransferCube-v0": "aloha"}
    env = environments.get(run["env"])
    identifier = UUID(str(run["run_id"]))
    count = {"cheap": 3, "expensive": 20}.get(run["tier"])
    if (env is None or re.fullmatch(r"c[0-9]{3,}", str(run["change_id"])) is None
            or re.fullmatch(rf"{env}-s(?:0[1-9]|1[0-6])", str(run["scenario_id"])) is None
            or identifier.version not in (4, 5) or str(identifier) != run["run_id"]
            or (identifier.version == 5 and str(uuid5(NAMESPACE_URL, key)) != run["run_id"])
            or count is None or run["episodes"] != count
            or type(run["successes"]) is not int or not 0 <= run["successes"] <= count
            or (run["tier"] == "expensive" and run["successes"] == count)
            or not isinstance(run["uri"], str)):
        raise ValueError("Guide evidence must identify its exact completed authoritative run")
    suffix = f"/episodes/env={env}/tier={run['tier']}/change={run['change_id']}/{run['scenario_id']}.parquet"
    frozen = identifier.version == 5
    if frozen and not run["uri"].endswith(suffix):
        raise ValueError("Guide Parquet URI differs from its authoritative pair key")
    if not frozen and run["uri"].startswith("gs://"):
        raise ValueError("new live Guide evidence must reference its recorded local artifact")
    source_root = run["uri"][:-len(suffix)] if frozen else str(root)
    pins = json.loads(Path(storage.__file__).with_name("pins.json").read_text())
    local_catalog = root / "scenarios.jsonl"
    catalog = ([json.loads(line) for line in local_catalog.read_text().splitlines()]
               if local_catalog.is_file() else
               json.loads(_guide_bytes(f"{source_root}/manifests/scenarios_{env}.json", root, 1024 * 1024)))
    selected = [item for item in catalog if item.get("scenario_id") == run["scenario_id"]]
    if (len(selected) != 1 or selected[0].get("params") != run["scenario"]
            or not isinstance(selected[0].get("seeds"), list) or len(selected[0]["seeds"]) != 50
            or any(type(seed) is not int or seed < 0 for seed in selected[0]["seeds"])
            or len(set(selected[0]["seeds"])) != 50):
        raise ValueError("Guide scenario differs from its frozen seed manifest")
    table = pq.read_table(io.BytesIO(_guide_bytes(run["uri"], root, 8 * 1024 * 1024)))
    if not table.schema.equals(storage._SCHEMA, check_metadata=False) or table.num_rows != count:
        raise ValueError("Guide evidence requires its complete authoritative 28-column rows")
    rows = sorted(table.to_pylist(), key=lambda row: row["episode_idx"])
    measured = ("steps", "max_steps", "reward_sum", "max_reward", "final_reward", "coverage_final",
                "coverage_max", "block_disp_px", "touched_right", "touched_left", "lifted",
                "table_contact_steps", "action_flip_rate")
    for index, row in enumerate(rows):
        if (row["episode_idx"] != index or row["env"] != env or row["tier"] != run["tier"]
                or row["change_id"] != run["change_id"] or row["scenario_id"] != run["scenario_id"]
                or row["seed"] != selected[0]["seeds"][index]
                or row["policy_repo"] != pins["policies"][env]["repo_id"]
                or row["policy_revision"] != pins["policies"][env]["revision"]
                or row["lerobot_version"] != pins["lerobot_version"]
                or not 0 < row["steps"] <= row["max_steps"] or not math.isfinite(row["wall_seconds"])
                or row["wall_seconds"] < 0
                or any(value is not None and type(value) is not bool and not math.isfinite(value)
                       for value in (row[name] for name in measured))):
            raise ValueError("Guide row identity, measurements or frozen seeds disagree")
        required_frames = run["tier"] == "cheap" or not row["success"]
        if frozen:
            expected_frames = (f"{source_root}/public/keyframes/{env}/{run['change_id']}/{run['scenario_id']}/{index}"
                               if required_frames else None)
            if row["keyframes_uri"] != expected_frames:
                raise ValueError("Guide keyframes must match their exact recorded episode prefix")
        elif required_frames:
            if not isinstance(row["keyframes_uri"], str):
                raise ValueError("Guide cheap evidence requires an actual final frame")
            folder = (root / row["keyframes_uri"]).resolve()
            parent = root / f"public/keyframes/{env}/{run['change_id']}/{run['scenario_id']}/{row['seed']}"
            if folder.parent != parent or _identity(folder.name) != folder.name:
                raise ValueError("Guide live frame prefix differs from its episode identity")
        elif row["keyframes_uri"] is not None:
            raise ValueError("successful expensive Guide episodes must not contain keyframes")
    if (sum(row["success"] for row in rows) != run["successes"]
            or not math.isclose(sum(row["reward_sum"] for row in rows) / count, run["mean_reward"], rel_tol=1e-8, abs_tol=1e-8)
            or not math.isclose(sum(row["wall_seconds"] for row in rows), run["wall_seconds"], rel_tol=1e-8, abs_tol=1e-8)):
        raise ValueError("Guide episode aggregates differ from the authoritative ledger")
    episodes = []
    for row in rows:
        if row["success"] and run["tier"] == "expensive":
            continue
        frames = []
        if not frozen:
            folder = (root / row["keyframes_uri"]).resolve()
            paths = sorted(folder.iterdir())
            if (len(paths) != (1 if run["tier"] == "cheap" else 8)
                    or any(re.fullmatch(r"frame-(?:final|[0-9]{4,})\.png", p.name) is None for p in paths)):
                raise ValueError("Guide live evidence requires the exact recorded PNG frame count")
            frames = [{"index": index, "path": str(_path(str(path), root))} for index, path in enumerate(paths)]
        for index in range((1 if run["tier"] == "cheap" else 8) if frozen else 0):
            uri = f"{row['keyframes_uri']}/f{index + 1}.jpg"
            payload = _guide_bytes(uri, root, 8 * 1024 * 1024)
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "JPEG":
                    raise ValueError("Guide keyframe must contain actual JPEG bytes")
                image.verify()
            directory = (root / "public/keyframes/guide" / hashlib.sha256(uri.encode() + payload).hexdigest()).resolve()
            if not directory.is_relative_to(root):
                raise ValueError("Guide cache must remain inside BRIER_DATA")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"f{index + 1}.jpg"
            try:
                with path.open("xb") as stream:
                    stream.write(payload)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ValueError("cached Guide keyframe differs from its immutable source") from None
            frames.append({"index": index, "path": str(path)})
        episodes.append({"seed": row["seed"], "telemetry": {name: row[name] for name in measured}, "frames": frames})
    return {"env": run["env"], "scenario": run["scenario"], "episodes": episodes}


def process(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one validated bridge operation without constructing any model prompt text."""
    if not isinstance(payload, dict) or payload.get("operation") not in ("segment", "explain", "escalate", "guide_evidence"):
        raise ValueError("operation must be segment, explain, escalate or guide_evidence")
    operation, root = payload["operation"], _root()
    if operation == "guide_evidence":
        if set(payload) != {"operation", "run"}:
            raise ValueError("Guide evidence requires exactly operation and authoritative run")
        return _guide_evidence(payload["run"], root)
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
