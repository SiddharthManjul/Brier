from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from brier_sim import farm


def _pins() -> str:
    return hashlib.sha256(Path(farm.__file__).with_name("pins.json").read_bytes()).hexdigest()


def _record(env: str, cheap: float = 2, expensive: float = 3, **overrides: object) -> dict:
    return {
        "env": env,
        "host": "github-runner",
        "nproc": 4,
        "pins_sha256": _pins(),
        "github_actions": True,
        "samples_seconds": {"cheap": [cheap] * 3, "expensive": [expensive] * 3},
        "median_seconds": {"cheap": cheap, "expensive": expensive},
        **overrides,
    }


def _write(tmp_path: Path, record: object, name: str) -> None:
    directory = tmp_path / "timing"
    directory.mkdir(exist_ok=True)
    (directory / f"timing_{name}.json").write_text(json.dumps(record))


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> tuple[list, list]:
    calls, closed = [], []
    vector, policy = SimpleNamespace(close=lambda: closed.append(True)), object()
    scenario = {"scenario_id": "pusht-s01", "seeds": [11, 37, 51, 99]}
    monkeypatch.setattr(farm, "build_scenarios", lambda env: [scenario])
    monkeypatch.setattr(farm.envs, "_load", lambda env: (vector, policy))
    monkeypatch.setattr(farm.torch, "set_num_threads", lambda count: calls.append(("threads", count)))
    monkeypatch.setattr(farm.socket, "gethostname", lambda: "local-host")
    ticks = iter([10, 11, 20, 23, 30, 32, 40, 50, 60, 80, 90, 120])
    monkeypatch.setattr(farm.time, "perf_counter", lambda: next(ticks))

    def episode(loaded: tuple, env: str, definition: dict, tier: str, seed: int) -> tuple:
        assert loaded == (vector, policy, None)
        assert definition is scenario
        calls.append((env, tier, seed))
        return {"seed": seed}, [object()]

    monkeypatch.setattr(farm, "_guide_episode", episode)
    return calls, closed


def test_timing_loads_once_measures_six_native_calls_and_only_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: tuple[list, list]
) -> None:
    calls, closed = native
    loaded = farm.envs._load
    loads = []
    monkeypatch.setattr(farm.envs, "_load", lambda env: (loads.append(env), loaded(env))[1])
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    result = farm._timing("pusht")
    assert loads == ["pusht"]
    assert calls == [("threads", 1)] + [
        ("pusht", tier, seed) for tier in ("cheap", "expensive") for seed in (11, 37, 51)
    ]
    assert closed == [True]
    assert result == {
        "env": "pusht",
        "host": "local-host",
        "nproc": result["nproc"],
        "pins_sha256": _pins(),
        "github_actions": False,
        "samples_seconds": {"cheap": [1, 3, 2], "expensive": [10, 20, 30]},
        "median_seconds": {"cheap": 2, "expensive": 20},
    }
    assert result["nproc"] > 0
    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert files == [tmp_path / "timing/timing_pusht_local-host.json"]
    assert json.loads(files[0].read_text()) == result


def test_timing_out_is_local_and_records_runtime_github_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: tuple[list, list]
) -> None:
    monkeypatch.delenv("BRIER_DATA", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    destination = tmp_path / "measurement.json"
    result = farm._timing("aloha", str(destination))
    assert result["github_actions"] is True
    assert json.loads(destination.read_text()) == result
    assert native[1] == [True]


def test_timing_closes_on_episode_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: tuple[list, list]
) -> None:
    def fail(*args: object) -> None:
        raise RuntimeError("native failure")

    monkeypatch.setattr(farm, "_guide_episode", fail)
    destination = tmp_path / "absent.json"
    with pytest.raises(RuntimeError, match="native failure"):
        farm._timing("pusht", str(destination))
    assert native[1] == [True]
    assert not destination.exists()


@pytest.mark.parametrize("env,out,message", [("bad", None, "env"), ("pusht", "gs://bucket/out.json", "local")])
def test_timing_rejects_invalid_environment_and_remote_out(env: str, out: str | None, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        farm._timing(env, out)


def test_json_publication_is_create_only(tmp_path: Path) -> None:
    destination = tmp_path / "new/file.json"
    farm._timing_json({"sample": 1}, str(destination))
    with pytest.raises(FileExistsError):
        farm._timing_json({"sample": 2}, str(destination))
    assert json.loads(destination.read_text()) == {"sample": 1}


def test_sizing_uses_runner_medians_ignores_local_data_and_keeps_full_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    _write(tmp_path, _record("pusht", 1, 2), "pusht_a")
    _write(tmp_path, _record("pusht", 5, 8), "pusht_b")
    _write(tmp_path, _record("pusht", 3, 4), "pusht_c")
    _write(tmp_path, _record("aloha", 2, 3), "aloha_a")
    _write(tmp_path, {"github_actions": False, "pins_sha256": "old"}, "local")
    result = farm._sizing()
    assert result["capacity"] == {"github_actions_processes": 80, "spot_processes": 0, "total_processes": 80}
    assert result["pins_sha256"] == _pins()
    assert result["notes"] == []
    expected = {"pusht": (3, 4), "aloha": (2, 3)}
    total = 0
    for env, (cheap, expensive) in expected.items():
        projection = result["environments"][env]
        assert (projection["changes"], projection["scenarios"], projection["pairs"]) == (80, 16, 1280)
        assert projection["median_seconds"] == {"cheap": cheap, "expensive": expensive}
        hours = (1280 * (3 * cheap + 20 * expensive) + 16 * 50 * expensive) / 3600
        assert projection["cpu_hours"] == pytest.approx(hours)
        total += hours
    assert result["wall_hours"] == pytest.approx(total / 80)
    assert result["target_wall_hours"] == result["wall_hours"]
    assert json.loads((tmp_path / "manifests/sizing.json").read_text()) == result


@pytest.mark.parametrize("target,changes,scenarios,note", [
    (10, 80, 16, None), (10.01, 76, 16, None), (12, 64, 16, None),
    (20, 48, 16, "48-change floor"), (20.01, 40, 12, "eight retained families"),
])
def test_sizing_decision_boundaries(
    target: float, changes: int, scenarios: int, note: str | None,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    cheap, pusht_expensive = 1, 10
    pusht_hours = (1280 * (3 * cheap + 20 * pusht_expensive) + 800 * pusht_expensive) / 3600
    aloha_expensive = ((target * 80 - pusht_hours) * 3600 - 1280 * 3 * cheap) / (1280 * 20 + 800)
    _write(tmp_path, _record("pusht", cheap, pusht_expensive), "pusht")
    _write(tmp_path, _record("aloha", cheap, aloha_expensive), "aloha")
    result = farm._sizing()
    assert result["target_wall_hours"] == pytest.approx(target)
    assert all((item["changes"], item["scenarios"]) == (changes, scenarios) for item in result["environments"].values())
    if note is not None:
        assert any(note in entry for entry in result["notes"])
    else:
        assert result["notes"] == []
    if target <= 10:
        assert result["wall_hours"] <= 10
    elif target < 20:
        assert result["wall_hours"] <= 10
        larger_pairs = (changes + 4) * 16
        larger = sum((larger_pairs * (3 * item["median_seconds"]["cheap"] + 20 * item["median_seconds"]["expensive"])
                      + 800 * item["median_seconds"]["expensive"]) / 3600 for item in result["environments"].values()) / 80
        assert larger > 10
    elif target == 20:
        assert result["wall_hours"] > 10


@pytest.mark.parametrize("bad", [
    None, {}, {"github_actions": "true"},
    _record("bad"), _record([]), _record("pusht", pins_sha256="stale"),
    _record("pusht", host=""), _record("pusht", nproc=True), _record("pusht", nproc=0),
    _record("pusht", samples_seconds=None), _record("pusht", median_seconds=[]),
    _record("pusht", samples_seconds={"cheap": [1, 2], "expensive": [3, 3, 3]}),
    _record("pusht", samples_seconds={"cheap": [0, 2, 3], "expensive": [3, 3, 3]}),
    _record("pusht", samples_seconds={"cheap": [True, 2, 3], "expensive": [3, 3, 3]}),
    _record("pusht", samples_seconds={"cheap": [1, float("nan"), 3], "expensive": [3, 3, 3]}),
    _record("pusht", samples_seconds={"cheap": [1, 10**1000, 3], "expensive": [3, 3, 3]}),
    _record("pusht", median_seconds={"cheap": 10**1000, "expensive": 3}),
    _record("pusht", median_seconds={"cheap": float("inf"), "expensive": 3}),
    _record("pusht", median_seconds={"cheap": 20, "expensive": 3}),
])
def test_sizing_rejects_malformed_runner_records(
    bad: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    _write(tmp_path, bad, "bad")
    _write(tmp_path, _record("pusht"), "pusht")
    _write(tmp_path, _record("aloha"), "aloha")
    with pytest.raises(ValueError, match="timing"):
        farm._sizing()
    assert not (tmp_path / "manifests/sizing.json").exists()


def test_sizing_does_not_substitute_local_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    _write(tmp_path, _record("pusht"), "pusht")
    _write(tmp_path, _record("aloha", github_actions=False), "aloha_local")
    with pytest.raises(ValueError, match="both pusht and aloha"):
        farm._sizing()


@pytest.mark.parametrize("steps,accepted", [(None, False), (100, False), (10, True), (2, True)])
def test_sizing_requires_retimed_reduced_diffusion_before_cutting_data(
    steps: int | None, accepted: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    pins = json.loads(Path(farm.__file__).with_name("pins.json").read_text())
    pins["policies"]["pusht"].pop("num_inference_steps", None)
    if steps is not None:
        pins["policies"]["pusht"]["num_inference_steps"] = steps
    code = tmp_path / "code"
    code.mkdir()
    (code / "pins.json").write_text(json.dumps(pins))
    monkeypatch.setattr(farm, "__file__", str(code / "farm.py"))
    _write(tmp_path, _record("pusht", expensive=30.01), "pusht")
    _write(tmp_path, _record("aloha"), "aloha")
    if accepted:
        result = farm._sizing()
        assert result["environments"]["pusht"]["median_seconds"]["expensive"] == 30.01
        assert result["environments"]["pusht"]["changes"] == 80
        return
    with pytest.raises(ValueError, match="adjust and record policy pins"):
        farm._sizing()
    assert not (tmp_path / "manifests/sizing.json").exists()


def test_sizing_reads_gcs_timing_and_publishes_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRIER_DATA", "gs://test-bucket/data")
    requested, published = [], []
    blobs = [SimpleNamespace(name=f"data/timing/timing_{env}_host.json", download_as_bytes=lambda retry, env=env: json.dumps(_record(env)).encode())
             for env in ("aloha", "pusht")]
    blobs.append(SimpleNamespace(name="data/timing/unrelated.json", download_as_bytes=lambda **kwargs: pytest.fail("unrelated object read")))

    def list_blobs(bucket: str, *, prefix: str, retry: object) -> list:
        requested.append((bucket, prefix, retry))
        return blobs

    monkeypatch.setattr(farm.storage, "_client", lambda: SimpleNamespace(list_blobs=list_blobs))
    monkeypatch.setattr(farm.storage, "_publish", lambda source, uri: published.append((json.loads(source.read_text()), uri)))
    result = farm._sizing()
    assert requested == [("test-bucket", "data/timing/", None)]
    assert published == [(result, "gs://test-bucket/data/manifests/sizing.json")]


def test_data_root_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRIER_DATA", raising=False)
    with pytest.raises(ValueError, match="BRIER_DATA"):
        farm._sizing()


def test_cli_prints_medians_and_preserves_old_flat_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []
    monkeypatch.setattr(farm, "_timing", lambda env, out: (calls.append((env, out)), {"median_seconds": {"cheap": 1, "expensive": 2}})[1])
    assert farm._timing_cli(["timing", "--env", "pusht", "--out", "/tmp/t.json"])
    assert calls == [("pusht", "/tmp/t.json")]
    assert json.loads(capsys.readouterr().out) == {"cheap": 1, "expensive": 2}
    monkeypatch.setattr(farm, "_sizing", lambda: {"wall_hours": 4})
    assert farm._timing_cli(["sizing"])
    assert json.loads(capsys.readouterr().out) == {"wall_hours": 4}
    assert farm._timing_cli(["--env", "pusht", "--shard", "0", "--shards", "1"]) is False
    assert farm._timing_cli([]) is False
    with pytest.raises(SystemExit):
        farm._timing_cli(["sizing", "--env", "pusht"])
