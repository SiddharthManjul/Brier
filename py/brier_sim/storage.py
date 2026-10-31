from __future__ import annotations

import math
import os
import re
import tempfile
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

_NULLABLE = {"coverage_final", "coverage_max", "block_disp_px", "touched_right", "touched_left", "lifted", "table_contact_steps", "keyframes_uri"}
_TYPES = {"s": pa.string(), "i": pa.int32(), "l": pa.int64(), "b": pa.bool_(), "f": pa.float64()}
_FIELDS = "env:s policy_repo:s policy_revision:s change_id:s scenario_id:s tier:s episode_idx:i seed:l success:b steps:i max_steps:i reward_sum:f max_reward:f final_reward:f coverage_final:f coverage_max:f block_disp_px:f touched_right:b touched_left:b lifted:b table_contact_steps:i action_flip_rate:f terminal_state_json:s keyframes_uri:s wall_seconds:f lerobot_version:s code_commit:s host:s"
_SCHEMA = pa.schema([pa.field(name, _TYPES[kind], nullable=name in _NULLABLE)
                     for name, kind in (field.split(":") for field in _FIELDS.split())])


def _location(uri: str) -> tuple[str, str] | Path:
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("storage URI must be a nonempty string")
    parsed = urlsplit(uri)
    if parsed.scheme == "gs" and parsed.netloc and not parsed.username and not parsed.password:
        if parsed.query or parsed.fragment or not parsed.path.strip("/"):
            raise ValueError("GCS URI must identify an object without a query or fragment")
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme == "file" and not parsed.netloc and not parsed.query and not parsed.fragment:
        return Path(unquote(parsed.path)).expanduser().resolve()
    if not parsed.scheme:
        return Path(uri).expanduser().resolve()
    raise ValueError("storage URI must be gs://, file://, or a local path")


@cache
def _client() -> Any:
    from google.cloud import storage

    return storage.Client()


def _blob(location: tuple[str, str]) -> Any:
    return _client().bucket(location[0]).blob(location[1])


def _publish(source: Path, uri: str, *, reuse: bool = False) -> None:
    destination = _location(uri)
    if isinstance(destination, Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except FileExistsError:
            if not reuse or source.read_bytes() != destination.read_bytes():
                raise
        return
    from google.api_core import exceptions as gexc
    from requests.exceptions import ChunkedEncodingError, Timeout
    from requests.exceptions import ConnectionError as RequestsConnectionError

    blob = _blob(destination)
    for attempt in range(2):
        try:
            blob.upload_from_filename(str(source), if_generation_match=0, retry=None)
            return
        except gexc.PreconditionFailed:
            if not reuse or blob.download_as_bytes(retry=None) != source.read_bytes():
                raise
            return
        except (gexc.DeadlineExceeded, gexc.ServerError, gexc.TooManyRequests, ConnectionError, RequestsConnectionError, Timeout, ChunkedEncodingError):
            if attempt:
                raise


def key_uri(env: str, change_id: str, scenario_id: str, tier: str) -> str:
    """Resolve one immutable episode key under the configured data root."""
    if (any(not isinstance(value, str) for value in (env, change_id, scenario_id, tier))
            or env not in ("pusht", "aloha") or tier not in ("cheap", "expensive", "baseline", "pilot")
            or not re.fullmatch(rf"{env}-s(?:0[1-9]|[1-9]\d)", scenario_id)
            or not (change_id == "base" or re.fullmatch(r"c\d{3,}", change_id) and int(change_id[1:]) > 0)
            or tier == "baseline" and change_id != "base"):
        raise ValueError("invalid environment, tier, change id or scenario id")
    root = os.environ.get("BRIER_DATA", "").strip()
    if not root:
        raise ValueError("BRIER_DATA is required")
    suffix = f"episodes/env={env}/tier={tier}/change={change_id}/{scenario_id}.parquet"
    if root.startswith("gs://"):
        uri = f"{root.rstrip('/')}/{suffix}"
        _location(uri)
        return uri
    location = _location(root)
    if not isinstance(location, Path):
        raise TypeError("BRIER_DATA must be a GCS prefix or local directory")
    return (location / suffix).as_uri()


def exists(uri: str) -> bool:
    """Check key existence; authorization and other storage errors propagate."""
    location = _location(uri)
    return location.is_file() if isinstance(location, Path) else _blob(location).exists(retry=None)


def write_parquet(rows: list[dict[str, Any]], uri: str) -> None:
    """Validate complete typed rows and atomically publish a create-only Parquet key."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a nonempty list")
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(_SCHEMA.names):
            raise ValueError("episode rows must contain exactly the guide schema fields")
        for field in _SCHEMA:
            value = row[field.name]
            valid = value is None and field.nullable
            if value is not None:
                try:
                    valid = (type(value) is str if pa.types.is_string(field.type) else
                             type(value) is bool if pa.types.is_boolean(field.type) else
                             type(value) is int and -(2 ** (field.type.bit_width - 1)) <= value < 2 ** (field.type.bit_width - 1)
                             if pa.types.is_integer(field.type) else type(value) in (int, float) and math.isfinite(value))
                except OverflowError:
                    valid = False
            if not valid:
                raise ValueError(f"invalid type, range or nullability for {field.name}")
    destination = _location(uri)
    parent = destination.parent if isinstance(destination, Path) else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        source = Path(temporary) / "episodes.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA), source)
        _publish(source, uri)


def upload_keyframes(folder_uri: str, frames: list[np.ndarray]) -> str:
    """Publish eight JPEG frames, reusing existing files only when bytes match exactly."""
    if not isinstance(frames, list) or len(frames) != 8 or any(
        not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
        or frame.shape[-1] != 3 or 0 in frame.shape for frame in frames
    ):
        raise ValueError("keyframes require eight nonempty uint8 RGB arrays")
    destination = _location(folder_uri)
    parent = destination.parent if isinstance(destination, Path) else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        for index, frame in enumerate(frames, 1):
            source = Path(temporary) / f"f{index}.jpg"
            Image.fromarray(frame).save(source, format="JPEG", quality=90)
            _publish(source, f"{folder_uri.rstrip('/')}/{source.name}", reuse=True)
    return folder_uri
