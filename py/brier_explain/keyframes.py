from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image


def select_keyframes(
    frames: Sequence[np.ndarray], *, success: bool, tier: str
) -> list[tuple[int, np.ndarray]]:
    """Select eight distinct, evenly spaced RGB observations from a failed expensive episode."""
    if not isinstance(success, (bool, np.bool_)) or success or tier != "expensive":
        raise ValueError("keyframes require a failed expensive episode")
    if len(frames) < 8:
        raise ValueError("episode must contain at least eight distinct frames")
    if any(
        not isinstance(frame, np.ndarray) or frame.ndim != 3
        or frame.shape[-1] != 3 or frame.shape[0] == 0 or frame.shape[1] == 0
        or frame.dtype != np.uint8 or frame.shape != frames[0].shape
        for frame in frames
    ):
        raise ValueError("episode frames must be matching nonempty uint8 H×W×3 arrays")
    indices = np.linspace(0, len(frames) - 1, 8, dtype=np.int64)
    return [(int(index), frames[index].copy()) for index in indices]


def save_keyframes(
    frames: Sequence[np.ndarray], out: Path, *, success: bool, tier: str
) -> Path:
    """Save eight lossless PNGs inside BRIER_DATA without replacing existing artifacts."""
    selected = select_keyframes(frames, success=success, tier=tier)
    root = os.environ.get("BRIER_DATA", "")
    if not root or "://" in root or not Path(root).expanduser().is_dir():
        raise ValueError("BRIER_DATA must name an existing local data directory")
    out = out.expanduser().resolve()
    if not out.is_relative_to(Path(root).expanduser().resolve()):
        raise ValueError("keyframe output must be inside BRIER_DATA")
    out.mkdir(parents=True, exist_ok=False)
    try:
        for index, frame in selected:
            with (out / f"frame-{index:04d}.png").open("xb") as stream:
                Image.fromarray(frame).save(stream, format="PNG")
    except BaseException:
        shutil.rmtree(out)
        raise
    return out
