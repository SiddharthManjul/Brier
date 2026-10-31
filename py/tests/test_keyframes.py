from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from brier_explain.keyframes import save_keyframes, select_keyframes
from PIL import Image


def _frames() -> list[np.ndarray]:
    return [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(15)]


def test_select_keyframes_spans_failed_episode_without_mutating_frames() -> None:
    frames = _frames()
    selected = select_keyframes(frames, success=False, tier="expensive")
    assert [index for index, _ in selected] == [0, 2, 4, 6, 8, 10, 12, 14]
    assert all(np.all(frame == index) for index, frame in selected)
    selected[0][1][:] = 255
    assert np.all(frames[0] == 0)


def test_save_keyframes_preserves_pixels_and_existing_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    directory = tmp_path / "public" / "keyframes" / "episode"
    assert save_keyframes(_frames(), directory, success=False, tier="expensive") == directory
    files = sorted(directory.glob("*.png"))
    assert [path.name for path in files] == [f"frame-{index:04d}.png" for index in range(0, 15, 2)]
    for path, index in zip(files, range(0, 15, 2), strict=True):
        with Image.open(path) as image:
            assert image.mode == "RGB" and image.size == (3, 2)
            assert np.all(np.asarray(image) == index)
    with pytest.raises(FileExistsError):
        save_keyframes(_frames(), directory, success=False, tier="expensive")
    assert len(list(directory.glob("*.png"))) == 8


@pytest.mark.parametrize("success,tier,length", [(True, "expensive", 8), (False, "cheap", 8), (False, "expensive", 7)])
def test_select_keyframes_rejects_ineligible_episodes(success: bool, tier: str, length: int) -> None:
    with pytest.raises(ValueError):
        select_keyframes(_frames()[:length], success=success, tier=tier)


@pytest.mark.parametrize("frame", [np.zeros((2, 2, 3), dtype=np.float32), np.zeros((2, 2), dtype=np.uint8)])
def test_select_keyframes_requires_matching_rgb_pixels(frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="uint8"):
        select_keyframes([frame] * 8, success=False, tier="expensive")


def test_save_keyframes_cleans_partial_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    directory = tmp_path / "frames"

    def fail(*args, **kwargs):
        raise OSError("PNG write failed")

    monkeypatch.setattr(Image.Image, "save", fail)
    with pytest.raises(OSError, match="PNG write failed"):
        save_keyframes(_frames(), directory, success=False, tier="expensive")
    assert not directory.exists()
