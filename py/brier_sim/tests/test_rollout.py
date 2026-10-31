from __future__ import annotations

import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from brier_sim import runner, storage
from PIL import Image

_NAMES = ["env", "policy_repo", "policy_revision", "change_id", "scenario_id", "tier", "episode_idx", "seed", "success", "steps", "max_steps", "reward_sum", "max_reward", "final_reward", "coverage_final", "coverage_max", "block_disp_px", "touched_right", "touched_left", "lifted", "table_contact_steps", "action_flip_rate", "terminal_state_json", "keyframes_uri", "wall_seconds", "lerobot_version", "code_commit", "host"]


def _path(uri: str) -> Path:
    return Path(unquote(urlsplit(uri).path))


def _scenario(env: str = "pusht") -> dict:
    return {"scenario_id": f"{env}-s01", "seeds": list(range(50)),
            "params": {"pose_bin": 1, "lighting": "normal", "camera_jitter": False,
                       **({"physics_jitter": False} if env == "aloha" else {})}}


def _row() -> dict:
    return dict(zip(_NAMES, ["pusht", "lerobot/diffusion_pusht", "a" * 40, "c001", "pusht-s01",
                            "cheap", 0, 0, False, 30, 30, 1., 1., 1., .1, .2, 3., None, None,
                            None, None, .2, "{}", None, .5, "0.3.3", "b" * 40, "test-host"], strict=True))


def test_native_cheap_key_uses_exact_schema_cache_and_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brier_sim.scenarios import build_scenarios

    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    original, loads = runner.envs._load, []

    def load(env: str) -> tuple:
        loaded = original(env)
        loaded[0].envs[0]._max_episode_steps = 60
        loads.append(env)
        return loaded

    monkeypatch.setattr(runner.envs, "_load", load)
    change = {"change_id": "c001", "env": "pusht", "families": [{"family": "benign", "params": {}}],
              "desc": "Keep base policy", "seed": 2026}
    scenario, cache = build_scenarios("pusht")[0], {}
    try:
        rows, frames = runner.run_key("pusht", change, scenario, "cheap", cache)
        assert len(rows) == 3 and frames == [] and loads == ["pusht"]
        assert [row["seed"] for row in rows] == scenario["seeds"][:3]
        assert all(row["max_steps"] == 30 and 1 <= row["steps"] <= 30 for row in rows)
        assert all(set(row) == set(_NAMES) and row["keyframes_uri"] is None for row in rows)
        assert all(len(row["policy_revision"]) == len(row["code_commit"]) == 40 for row in rows)
        assert all(row["lerobot_version"] == "0.3.3" and row["host"] for row in rows)
        uri = storage.key_uri("pusht", "c001", scenario["scenario_id"], "cheap")
        table = pq.ParquetFile(_path(uri)).read()
        assert table.schema == storage._SCHEMA and table.schema.names == _NAMES
        assert table.schema.field("episode_idx").type == pa.int32()
        assert table.schema.field("seed").type == pa.int64()
        assert table.to_pylist() == rows
        monkeypatch.setattr(runner.envs, "_load", lambda _: pytest.fail("skip loaded a policy"))
        assert runner.run_key("pusht", change, scenario, "cheap", {}) == ([], [])
        assert loads == ["pusht"]
    finally:
        for vector, _, _ in cache.values():
            vector.close()


class _Child:
    _max_episode_steps = 8

    def __init__(self, *, finish: int = 8, success: bool = False) -> None:
        self.unwrapped = SimpleNamespace(agent=SimpleNamespace(position=(1., 2.)),
                                         block=SimpleNamespace(position=(10., 20.), angle=.25))
        self.finish, self.success = finish, success
        self.actions: list[np.ndarray] = []

    def observation(self) -> dict:
        return {"pixels": np.full((96, 96, 3), self.step_count, np.uint8),
                "agent_pos": np.array([1., 2.])}

    def reset(self, *, seed: int) -> tuple:
        self.seed, self.step_count = seed, 0
        self.unwrapped.block.position = (10., 20.)
        return self.observation(), {"is_success": False}

    def step(self, action: np.ndarray) -> tuple:
        self.actions.append(action.copy())
        self.step_count += 1
        self.unwrapped.block.position = (10. + self.step_count, 20.)
        return self.observation(), float(self.step_count), self.step_count == self.finish, False, {
            "is_success": self.success and self.step_count == 1, "coverage": self.step_count / 10}


class _Policy:
    def reset(self) -> None:
        self.index = 0

    def select_action(self, batch: dict) -> torch.Tensor:
        values = [0., 2., 1., 3., 3., 3., 3., 3.]
        value = values[self.index]
        self.index += 1
        return torch.tensor([[value, 0.]])


def _loaded(monkeypatch: pytest.MonkeyPatch, child: _Child) -> tuple:
    monkeypatch.setattr(runner, "scenario_reset_hook", lambda *_: {})
    monkeypatch.setattr(runner, "scenario_obs_hook", lambda _, observation: observation)
    monkeypatch.setattr(runner, "tier_obs_noise_hook", lambda observation, *_: observation)
    return SimpleNamespace(envs=[child]), _Policy(), None


def test_raw_frames_stats_any_success_and_direct_child_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    child = _Child(success=True)
    row, frames = runner._guide_episode(_loaded(monkeypatch, child), "pusht", _scenario(), "expensive", 0)
    assert row["success"] is True and row["steps"] == row["max_steps"] == 8
    assert row["reward_sum"] == 36 and row["max_reward"] == row["final_reward"] == 8
    assert row["coverage_final"] == row["coverage_max"] == .8
    assert row["block_disp_px"] == 8
    assert row["action_flip_rate"] == pytest.approx(2 / 6)
    assert json.loads(row["terminal_state_json"])["block_pose"] == [18., 20., .25]
    assert [int(frame[0, 0, 0]) for frame in frames] == list(range(8))
    assert all(frame.shape == (384, 384, 3) for frame in frames)


def test_early_terminal_fills_unreached_keyframes_and_denominator(monkeypatch: pytest.MonkeyPatch) -> None:
    child = _Child(finish=3)
    row, frames = runner._guide_episode(_loaded(monkeypatch, child), "pusht", _scenario(), "expensive", 0)
    assert row["steps"] == 3 and row["action_flip_rate"] == 1
    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 1, 2, 3, 3, 3, 3, 3]
    assert json.loads(row["terminal_state_json"])["block_pose"][0] == 13


def test_hook_order_and_raw_keyframes_precede_perturbations(monkeypatch: pytest.MonkeyPatch) -> None:
    child, events = _Child(finish=1), []
    loaded = _loaded(monkeypatch, child)

    def hook(name: str, observation: dict) -> dict:
        events.append(name)
        observation["observation.image"][:] = 1
        return observation

    monkeypatch.setattr(runner, "scenario_obs_hook", lambda _, obs: hook("scenario", obs))
    monkeypatch.setattr(runner, "tier_obs_noise_hook", lambda obs, *_: hook("tier", obs))
    monkeypatch.setattr(runner, "obs_hook", lambda _, obs, rng: hook("change_obs", obs))
    monkeypatch.setattr(runner, "action_hook", lambda _, action, raw, rng: (events.append("change_action") or action))
    state = SimpleNamespace(reset=loaded[1].reset)
    row, frames = runner._guide_episode((*loaded[:2], state), "pusht", _scenario(), "expensive", 0)
    assert events == ["scenario", "tier", "change_obs", "change_action"]
    assert frames[0].max() == 0 and row["steps"] == 1


def test_missing_success_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    child = _Child(finish=1)
    original = child.step

    def step(action: np.ndarray) -> tuple:
        observation, reward, done, truncated, info = original(action)
        del info["is_success"]
        return observation, reward, done, truncated, info

    child.step = step
    with pytest.raises(KeyError, match="is_success"):
        runner._guide_episode(_loaded(monkeypatch, child), "pusht", _scenario(), "cheap", 0)


def test_aloha_contact_filter_and_terminal_positions() -> None:
    names = ["table", "vx300s_left/10_left_gripper_finger", "vx300s_left/arm", "red_box"]
    physics = SimpleNamespace(data=SimpleNamespace(ncon=1, contact=[SimpleNamespace(geom1=0, geom2=2)]),
                              model=SimpleNamespace(id2name=lambda index, kind: names[index]),
                              named=SimpleNamespace(data=SimpleNamespace(geom_xpos={"red_box": np.array([.1, .5, .05])})))
    raw = SimpleNamespace(_env=SimpleNamespace(physics=physics, task=SimpleNamespace(get_qpos=lambda _: np.arange(14))))
    assert runner._table_contact(raw) is False
    physics.data.contact[0].geom2 = 1
    assert runner._table_contact(raw) is True
    physics.data.contact[0] = SimpleNamespace(geom1=1, geom2=0)
    assert runner._table_contact(raw) is True
    physics.data.contact[0] = SimpleNamespace(geom1=1, geom2=3)
    assert runner._table_contact(raw) is False
    assert runner._guide_terminal(raw, "aloha") == {"box_xyz": [.1, .5, .05], "gripper_qpos": [6., 13.]}


def test_aloha_episode_records_reward_stages_contact_steps_and_rerender(monkeypatch: pytest.MonkeyPatch) -> None:
    child = _Child(finish=4, success=True)
    loaded = _loaded(monkeypatch, child)
    child.observation = lambda: {"pixels": {"top": np.zeros((16, 16, 3), np.uint8)},
                                 "agent_pos": np.zeros(14)}
    physics = SimpleNamespace(data=SimpleNamespace(ncon=1, contact=[SimpleNamespace(geom1=0, geom2=1)]),
                              model=SimpleNamespace(id2name=lambda index, kind: ["table", "gripper_finger"][index]),
                              named=SimpleNamespace(data=SimpleNamespace(geom_xpos={"red_box": np.array([.1, .5, .05])})))
    rendered = []
    child.unwrapped._env = SimpleNamespace(physics=physics, task=SimpleNamespace(
        get_observation=lambda _: (rendered.append(True) or child.observation()), get_qpos=lambda _: np.arange(14)))
    child.unwrapped._format_raw_obs = lambda observation: observation
    row, frames = runner._guide_episode(loaded, "aloha", _scenario("aloha"), "baseline", 0)
    assert rendered == [True] and row["success"] is True
    assert row["touched_right"] is row["touched_left"] is row["lifted"] is True
    assert row["table_contact_steps"] == row["steps"] == 4
    assert row["coverage_final"] is row["coverage_max"] is row["block_disp_px"] is None
    assert row["max_reward"] == 4 and row["reward_sum"] == 10 and frames == []


def test_run_key_counts_cached_policy_changes_and_publication_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path / "episodes" / "nested"))
    loads, changes, events, cache = [], [], [], {}
    monkeypatch.setattr(runner.envs, "_load", lambda env: (loads.append(env) or SimpleNamespace(close=lambda: None), _Policy()))
    monkeypatch.setattr(runner, "apply_change", lambda policy, change: changes.append(change) or object())
    previous_frames = []

    def episode(loaded: tuple, env: str, scenario: dict, tier: str, seed: int) -> tuple:
        assert all(reference() is None for reference in previous_frames)
        frames = [np.zeros((8, 8, 3), np.uint8)] * 8
        previous_frames[:] = [weakref.ref(frame) for frame in frames]
        row = _row()
        row.update(seed=seed, success=False)
        events.append(("episode", tier, seed))
        return row, frames

    def upload(folder: str, values: list) -> str:
        assert all(reference() is frame for reference, frame in zip(previous_frames, values, strict=True))
        assert "/episodes/nested/public/keyframes/" in folder
        events.append(("frames", folder))
        return folder

    def write(rows: list, uri: str) -> None:
        assert len([item for item in events if item[0] == "frames"]) == (20 if rows[0]["tier"] == "expensive" else 0)
        events.append(("parquet", uri))

    monkeypatch.setattr(runner, "_guide_episode", episode)
    monkeypatch.setattr(storage, "upload_keyframes", upload)
    monkeypatch.setattr(storage, "write_parquet", write)
    for tier, count in [("baseline", 50), ("cheap", 3), ("pilot", 10)]:
        events.clear()
        rows, folders = runner.run_key("pusht", None, _scenario(), tier, cache)
        assert len(rows) == count and folders == []
        assert all(row["change_id"] == "base" for row in rows)
        assert events[-1][0] == "parquet"
    assert loads == ["pusht"] and changes == []
    change = {"change_id": "c001", "env": "pusht"}
    for scenario in (_scenario(), {**_scenario(), "scenario_id": "pusht-s02"}):
        events.clear()
        rows, folders = runner.run_key("pusht", change, scenario, "expensive", cache)
        assert len(rows) == len(folders) == 20
        assert all(row["keyframes_uri"] in folders for row in rows)
        assert events[-1][0] == "parquet"
    assert loads == ["pusht", "pusht"] and changes == [change]
    assert all(reference() is None for reference in previous_frames)


def test_invalid_scenario_fails_before_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    monkeypatch.setattr(runner.envs, "_load", lambda _: pytest.fail("invalid scenario loaded policy"))
    for seeds in ([0, 1], [0, 0, 1], [0, 1, -1], [0, 1, True]):
        with pytest.raises(ValueError, match="distinct nonnegative"):
            runner.run_key("pusht", None, {**_scenario(), "seeds": seeds}, "cheap", {})


def test_key_uri_exists_and_atomic_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRIER_DATA", tmp_path.as_uri())
    uri = storage.key_uri("pusht", "c001", "pusht-s01", "cheap")
    assert uri == (tmp_path / "episodes/env=pusht/tier=cheap/change=c001/pusht-s01.parquet").as_uri()
    assert storage.exists(uri) is False
    storage.write_parquet([_row()], uri)
    original = _path(uri).read_bytes()
    assert storage.exists(uri) is True
    assert pq.ParquetFile(_path(uri)).read().schema == storage._SCHEMA
    with pytest.raises(FileExistsError):
        storage.write_parquet([{**_row(), "steps": 4}], uri)
    assert _path(uri).read_bytes() == original
    assert list(_path(uri).parent.iterdir()) == [_path(uri)]
    monkeypatch.setenv("BRIER_DATA", "gs://bucket/prefix")
    assert storage.key_uri("aloha", "base", "aloha-s01", "baseline") == "gs://bucket/prefix/episodes/env=aloha/tier=baseline/change=base/aloha-s01.parquet"
    assert "change=base" in storage.key_uri("pusht", "base", "pusht-s01", "cheap")


@pytest.mark.parametrize("patch", [{"steps": True}, {"episode_idx": 2**31}, {"success": 1},
                                 {"max_reward": float("nan")}, {"host": None}, {"extra": "bad"}])
def test_parquet_rejects_invalid_rows_before_publish(tmp_path: Path, patch: dict) -> None:
    uri = (tmp_path / "bad.parquet").as_uri()
    with pytest.raises(ValueError):
        storage.write_parquet([{**_row(), **patch}], uri)
    assert not (tmp_path / "bad.parquet").exists()


def test_eight_jpegs_are_create_only_and_identical_reuse_is_safe(tmp_path: Path) -> None:
    frames = [np.full((8, 8, 3), value, np.uint8) for value in range(8)]
    folder = (tmp_path / "frames").as_uri()
    assert storage.upload_keyframes(folder, frames) == folder
    assert sorted(path.name for path in _path(folder).iterdir()) == [f"f{index}.jpg" for index in range(1, 9)]
    with Image.open(_path(folder) / "f1.jpg") as image:
        assert image.format == "JPEG" and image.size == (8, 8)
    storage.upload_keyframes(folder, frames)
    with pytest.raises(FileExistsError):
        storage.upload_keyframes(folder, [255 - frame for frame in frames])
    with pytest.raises(ValueError, match="eight"):
        storage.upload_keyframes(folder, frames[:7])


def test_gcs_generation_guard_single_retry_and_access_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from google.api_core.exceptions import (
        Forbidden,
        PreconditionFailed,
        ServiceUnavailable,
    )

    calls, behavior = [], [ServiceUnavailable("transient"), None]

    def upload(source: str, **kwargs: object) -> None:
        calls.append(kwargs)
        assert Path(source).is_file()
        error = behavior.pop(0)
        if error is not None:
            raise error

    blob = SimpleNamespace(upload_from_filename=upload, exists=lambda **kwargs: True)
    monkeypatch.setattr(storage, "_blob", lambda _: blob)
    assert storage.exists("gs://bucket/key") is True
    storage.write_parquet([_row()], "gs://bucket/key")
    assert calls == [{"if_generation_match": 0, "retry": None}] * 2
    for error in (Forbidden("denied"), PreconditionFailed("already exists")):
        behavior[:] = [error]
        calls.clear()
        with pytest.raises(type(error)):
            storage.write_parquet([_row()], "gs://bucket/key")
        assert len(calls) == 1
    behavior[:] = [ServiceUnavailable("first"), ServiceUnavailable("second")]
    calls.clear()
    with pytest.raises(ServiceUnavailable):
        storage.write_parquet([_row()], "gs://bucket/key")
    assert len(calls) == 2


@pytest.mark.parametrize("exception", ["builtin", "ConnectionError", "Timeout", "ChunkedEncodingError"])
def test_gcs_retries_transport_failure_only_once(exception: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from requests import exceptions

    error_type = ConnectionError if exception == "builtin" else getattr(exceptions, exception)
    behavior, calls = [error_type("broken connection"), None], []

    def upload(source: str, **kwargs: object) -> None:
        calls.append(kwargs)
        error = behavior.pop(0)
        if error is not None:
            raise error

    monkeypatch.setattr(storage, "_blob", lambda _: SimpleNamespace(upload_from_filename=upload))
    storage.write_parquet([_row()], "gs://bucket/key")
    assert calls == [{"if_generation_match": 0, "retry": None}] * 2
    behavior[:] = [error_type("first"), error_type("second")]
    calls.clear()
    with pytest.raises(error_type):
        storage.write_parquet([_row()], "gs://bucket/key")
    assert len(calls) == 2


def test_gcs_client_is_reused_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.cloud import storage as google_storage

    calls = []

    def client() -> SimpleNamespace:
        calls.append("client")
        return SimpleNamespace(bucket=lambda bucket: SimpleNamespace(blob=lambda key: (bucket, key)))

    storage._client.cache_clear()
    monkeypatch.setattr(google_storage, "Client", client)
    try:
        assert storage._blob(("bucket", "one")) == ("bucket", "one")
        assert storage._blob(("bucket", "two")) == ("bucket", "two")
        assert calls == ["client"]
    finally:
        storage._client.cache_clear()
