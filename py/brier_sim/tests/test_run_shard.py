from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from brier_sim import farm, storage


@pytest.fixture
def manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    directory = tmp_path / "manifests"
    directory.mkdir()
    definitions = {
        env: [{"scenario_id": f"{env}-s{index + 1:02}",
               "params": {"pose_bin": index // 4 + 1, "lighting": "normal" if index % 4 < 2 else "low",
                          "camera_jitter": bool(index % 2) if env == "pusht" else bool(index // 4 % 2 == 0),
                          **({"physics_jitter": bool(index % 2)} if env == "aloha" else {})},
               "seeds": list(range(index * 100 + 11, index * 100 + 61))} for index in range(16)]
        for env in ("pusht", "aloha")
    }
    monkeypatch.setattr(farm, "build_scenarios", lambda env: deepcopy(definitions[env]))
    sizing = {"pins_sha256": hashlib.sha256(Path(farm.__file__).with_name("pins.json").read_bytes()).hexdigest(),
              "environments": {env: {"changes": 40, "scenarios": 12} for env in definitions}}
    (directory / "sizing.json").write_text(json.dumps(sizing))
    for env, scenarios in definitions.items():
        (directory / f"scenarios_{env}.json").write_text(json.dumps(list(reversed(scenarios))))
        start = 1 if env == "pusht" else 81
        cards = [{"change_id": f"c{index:03}", "env": env, "families": [{"family": "benign", "params": {}}],
                  "desc": "Update checkpoint metadata", "seed": index} for index in range(start, start + 40)]
        (directory / f"changes_{env}.jsonl").write_text("\n".join(json.dumps(card) for card in reversed(cards)) + "\n")
    return {"root": tmp_path, "definitions": definitions, "sizing": sizing}


def _uri(env: str, key: tuple) -> str:
    change, scenario, tier = key
    return storage.key_uri(env, "base" if change is None else change["change_id"], scenario["scenario_id"], tier)


def _touch(uri: str) -> None:
    path = storage._location(uri)
    assert isinstance(path, Path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _baselines(env: str, manifests: dict) -> None:
    for scenario in manifests["definitions"][env]:
        _touch(storage.key_uri(env, "base", scenario["scenario_id"], "baseline"))


class _Pool:
    def __init__(self, *, max_workers: int, mp_context: object) -> None:
        assert max_workers == 3
        assert mp_context.get_start_method() == "spawn"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def map(self, function: object, tasks: list) -> list:
        return [function(task) for task in tasks]


@pytest.mark.parametrize("env", ["pusht", "aloha"])
def test_baselines_are_all_sixteen_frozen_scenarios_without_pair_manifests(env: str, manifests: dict) -> None:
    for name in ("sizing.json", f"changes_{env}.jsonl"):
        (manifests["root"] / "manifests" / name).unlink()
    groups = farm._guide_plan(env, "baselines")
    assert len(groups) == 16
    assert groups == [[(None, scenario, "baseline")] for scenario in manifests["definitions"][env]]


@pytest.mark.parametrize("env,first", [("pusht", "c001"), ("aloha", "c081")])
def test_pair_plan_is_deterministic_and_shards_cover_every_key_once(env: str, first: str, manifests: dict) -> None:
    _baselines(env, manifests)
    groups = farm._guide_plan(env, "pairs")
    assert groups == farm._guide_plan(env, "pairs")
    assert len(groups) == 40 and all(len(group) == 24 for group in groups)
    assert groups[0][0][0]["change_id"] == first
    assert [key[2] for key in groups[0][:4]] == ["cheap", "expensive", "cheap", "expensive"]
    assert {key[1]["scenario_id"] for group in groups for key in group} == {f"{env}-s{i:02}" for i in range(1, 13)}
    partitions = [{_uri(env, key) for group in groups[shard::7] for key in group} for shard in range(7)]
    assert sum(map(len, partitions)) == len(set.union(*partitions)) == 40 * 12 * 2
    assert len({key[0]["change_id"] for key in groups[0]}) == 1
    manifests["sizing"]["environments"][env]["scenarios"] = 16
    (manifests["root"] / "manifests/sizing.json").write_text(json.dumps(manifests["sizing"]))
    assert len(farm._guide_plan(env, "pairs")[0]) == 32


def test_pairs_refuse_missing_baseline_even_when_dropped_from_pair_scenarios(manifests: dict) -> None:
    _baselines("pusht", manifests)
    storage._location(storage.key_uri("pusht", "base", "pusht-s16", "baseline")).unlink()
    with pytest.raises(ValueError, match="all 16 completed baseline"):
        farm._guide_plan("pusht", "pairs")


@pytest.mark.parametrize("fault", ["count", "duplicate", "wrong-env", "unhashable-id", "stale-pins", "bad-scenarios", "changed-seed", "seed-type"])
def test_plan_refuses_inconsistent_manifests(fault: str, manifests: dict) -> None:
    _baselines("pusht", manifests)
    directory = manifests["root"] / "manifests"
    cards = [json.loads(line) for line in (directory / "changes_pusht.jsonl").read_text().splitlines()]
    if fault == "count":
        cards.pop()
    elif fault == "duplicate":
        cards[0] = cards[1]
    elif fault == "wrong-env":
        cards[0]["env"] = "aloha"
    elif fault == "unhashable-id":
        cards[0]["change_id"] = []
    elif fault in ("stale-pins", "bad-scenarios"):
        if fault == "stale-pins":
            manifests["sizing"]["pins_sha256"] = "stale"
        else:
            manifests["sizing"]["environments"]["pusht"]["scenarios"] = 13
        (directory / "sizing.json").write_text(json.dumps(manifests["sizing"]))
    else:
        definitions = deepcopy(manifests["definitions"]["pusht"])
        definitions[0]["seeds"][0] = 999 if fault == "changed-seed" else float(definitions[0]["seeds"][0])
        (directory / "scenarios_pusht.json").write_text(json.dumps(definitions))
    (directory / "changes_pusht.jsonl").write_text("\n".join(json.dumps(card) for card in cards))
    with pytest.raises(ValueError, match="manifest|sizing"):
        farm._guide_plan("pusht", "pairs")


@pytest.mark.parametrize("env,first", [("pusht", "c1001"), ("aloha", "c2001")])
def test_pilot_has_every_grid_magnitude_and_four_matched_unjittered_scenarios(
    env: str, first: str, manifests: dict, caplog: pytest.LogCaptureFixture,
) -> None:
    for name in ("sizing.json", f"changes_{env}.jsonl"):
        (manifests["root"] / "manifests" / name).unlink()
    before = (manifests["root"] / f"manifests/scenarios_{env}.json").read_bytes()
    with caplog.at_level(logging.INFO):
        groups = farm._guide_plan(env, "pilot")
    assert "176 keys, 1760 episodes" in caplog.text
    assert len(groups) == 44 and all(len(keys) == 4 for keys in groups)
    assert groups == farm._guide_plan(env, "pilot")
    assert groups[0][0][0] is None and groups[1][0][0]["change_id"] == first
    expected = [(family, params) for family, values in sorted(farm._grid(env).items()) for params in values]
    actual = [(group[0][0]["families"][0]["family"], group[0][0]["families"][0]["params"]) for group in groups[1:]]
    assert actual == expected
    for keys in groups:
        for index, (_, scenario, tier) in enumerate(keys):
            original = manifests["definitions"][env][index * 4]
            assert tier == "pilot" and scenario["scenario_id"] == original["scenario_id"]
            assert scenario["seeds"] == original["seeds"][:10]
            assert scenario["params"]["lighting"] == "normal"
            assert scenario["params"]["camera_jitter"] is False
            assert scenario["params"].get("physics_jitter", False) is False
    assert (manifests["root"] / f"manifests/scenarios_{env}.json").read_bytes() == before


def test_group_skips_existing_rechecks_each_key_logs_outcomes_and_closes_cache(
    manifests: dict, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    keys = farm._guide_plan("pusht", "baselines")[:3]
    keys = [group[0] for group in keys]
    _touch(_uri("pusht", keys[0]))
    calls, closed, caches = [], [], []

    def run(env: str, change: dict | None, scenario: dict, tier: str, cache: dict) -> tuple:
        calls.append(scenario["scenario_id"])
        caches.append(id(cache))
        cache.setdefault("base", (SimpleNamespace(close=lambda: closed.append(True)), None, None))
        _touch(storage.key_uri(env, "base", scenario["scenario_id"], tier))
        _touch(_uri(env, keys[2]))  # A second worker completes the next key meanwhile.
        return [{"success": True}, {"success": False}], []

    monkeypatch.setattr(farm, "run_key", run)
    with caplog.at_level(logging.INFO):
        assert farm._group(("pusht", keys, float("inf"))) is False
    assert calls == ["pusht-s02"] and len(set(caches)) == 1 and closed == [True]
    assert "successes=1 n=2 wall_seconds=" in caplog.text


def test_group_closes_cache_when_rollout_fails(manifests: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    closed = []

    def fail(env: str, change: object, scenario: object, tier: str, cache: dict) -> None:
        cache["base"] = (SimpleNamespace(close=lambda: closed.append(True)), None, None)
        raise RuntimeError("native rollout failed")

    monkeypatch.setattr(farm, "run_key", fail)
    with pytest.raises(RuntimeError, match="native rollout failed"):
        farm._group(("pusht", farm._guide_plan("pusht", "baselines")[0], float("inf")))
    assert closed == [True]


def test_group_checks_absolute_deadline_before_each_new_key(manifests: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [group[0] for group in farm._guide_plan("pusht", "baselines")[:2]]
    clock, ran = iter([8, 10]), []
    monkeypatch.setattr(farm.time, "monotonic", lambda: next(clock))

    def run(env: str, change: object, scenario: dict, tier: str, cache: dict) -> tuple:
        ran.append(scenario["scenario_id"])
        _touch(storage.key_uri(env, "base", scenario["scenario_id"], tier))
        return [], []

    monkeypatch.setattr(farm, "run_key", run)
    assert farm._group(("pusht", keys, 10)) is True
    assert ran == ["pusht-s01"]


def test_run_uses_spawn_fresh_group_caches_and_skips_complete_shard(
    manifests: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(farm, "ProcessPoolExecutor", _Pool)
    caches, closed = [], []

    def run(env: str, change: object, scenario: dict, tier: str, cache: dict) -> tuple:
        caches.append(cache)
        cache["base"] = (SimpleNamespace(close=lambda: closed.append(scenario["scenario_id"])), None, None)
        _touch(storage.key_uri(env, "base", scenario["scenario_id"], tier))
        return [], []

    monkeypatch.setattr(farm, "run_key", run)
    assert farm._run("pusht", "baselines", 0, 8, 3) == 0
    assert len(caches) == 2 and caches[0] is not caches[1]
    assert closed == ["pusht-s01", "pusht-s09"]
    monkeypatch.setattr(farm, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("complete shard must skip the pool"))
    assert farm._run("pusht", "baselines", 0, 8, 3) == 0
    assert farm._run("pusht", "baselines", 19, 20, 3) == 0


def test_real_spawned_worker_imports_and_skips_a_completed_key(manifests: dict) -> None:
    keys = farm._guide_plan("pusht", "baselines")[0]
    _touch(_uri("pusht", keys[0]))
    with farm.ProcessPoolExecutor(max_workers=1, mp_context=farm.get_context("spawn")) as pool:
        assert list(pool.map(farm._group, [("pusht", keys, float("inf"))])) == [False]


def test_deadline_includes_planning_io_and_reports_three_only_with_missing_keys(
    manifests: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = farm._guide_plan
    clock = [100.]
    monkeypatch.setattr(farm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(farm, "ProcessPoolExecutor", _Pool)

    def plan(env: str, phase: str) -> list:
        result = original(env, phase)
        clock[0] += 3600
        return result

    monkeypatch.setattr(farm, "_guide_plan", plan)
    monkeypatch.setattr(farm, "run_key", lambda *args: pytest.fail("deadline expired during planning"))
    assert farm._run("pusht", "baselines", 0, 16, 3, 1) == 3
    _touch(storage.key_uri("pusht", "base", "pusht-s01", "baseline"))
    assert farm._run("pusht", "baselines", 0, 16, 3, 1) == 0


def test_missing_work_without_deadline_stop_is_an_error(manifests: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(farm, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(farm, "run_key", lambda *args: ([], []))
    with pytest.raises(RuntimeError, match="missing assigned keys"):
        farm._run("pusht", "baselines", 0, 16, 3)


@pytest.mark.parametrize("args", [(0, 0, 3, None), (1, 1, 3, None), (True, 2, 3, None),
                                   (0, 1, 0, None), (0, 1, 3, 0), (0, 1, 3, -1),
                                   (0, 1, 3, float("nan")), (0, 1, 3, float("inf")), (0, 1, 3, True), (0, 1, 3, 1e308)])
def test_invalid_execution_bounds_fail_before_planning(args: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(farm, "_guide_plan", lambda *args: pytest.fail("invalid limits must not plan"))
    with pytest.raises(ValueError, match="valid positive finite"):
        farm._run("pusht", "baselines", *args)


def test_pilot_shard_zero_bootstraps_only_four_base_keys_and_other_shards_refuse_first(
    manifests: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = []
    monkeypatch.setattr(farm, "ProcessPoolExecutor", lambda **kwargs: pytest.fail("bootstrap must not start changed groups"))

    def run(env: str, change: object, scenario: dict, tier: str, cache: dict) -> tuple:
        assert change is None and tier == "pilot"
        ran.append(scenario["scenario_id"])
        _touch(storage.key_uri(env, "base", scenario["scenario_id"], tier))
        return [], []

    monkeypatch.setattr(farm, "run_key", run)
    with pytest.raises(ValueError, match="four matched base keys"):
        farm._run("pusht", "pilot", 1, 44, 3)
    assert farm._run("pusht", "pilot", 0, 44, 3) == 0
    assert ran == ["pusht-s01", "pusht-s05", "pusht-s09", "pusht-s13"]
    assert not (manifests["root"] / "manifests/magnitudes_pusht.json").exists()


def _pilot_rows(env: str, key: tuple, successes: int) -> list[dict]:
    change, scenario, tier = key
    return [{"env": env, "change_id": "base" if change is None else change["change_id"],
             "scenario_id": scenario["scenario_id"], "tier": tier, "seed": seed, "success": index < successes}
            for index, seed in enumerate(scenario["seeds"])]


def _write_rows(uri: str, rows: list[dict]) -> None:
    path = storage._location(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_pilot_aggregates_actual_paired_rows_only_after_global_completion_and_reuses_identical_json(
    manifests: dict, caplog: pytest.LogCaptureFixture,
) -> None:
    env, groups = "aloha", farm._guide_plan("aloha", "pilot")
    for group_index, keys in enumerate(groups):
        for scenario_index, key in enumerate(keys):
            # Drops 2/40 and 24/40 are retained; 1/40, 25/40 and zero are omitted.
            successes = (8 if group_index == 0 or group_index > 4 else
                         (8 if scenario_index < 3 else 6) if group_index == 1 else
                         2 if group_index == 2 else
                         (8 if scenario_index < 3 else 7) if group_index == 3 else
                         2 if scenario_index < 3 else 1)
            _write_rows(_uri(env, key), _pilot_rows(env, key, successes))
    last_key = _uri(env, groups[-1][-1])
    last_path = storage._location(last_key)
    saved = last_path.read_bytes()
    last_path.unlink()
    destination = manifests["root"] / "manifests/magnitudes_aloha.json"
    farm._pilot_magnitudes(env, groups)
    assert not destination.exists()
    last_path.write_bytes(saved)
    with caplog.at_level(logging.INFO):
        farm._pilot_magnitudes(env, groups)
    result = json.loads(destination.read_text())
    first, second = [groups[index][0][0]["families"][0] for index in (1, 2)]
    assert result == {first["family"]: [{"params": first["params"], "mean_drop": .05},
                                       {"params": second["params"], "mean_drop": .6}]}
    assert "dropped families" in caplog.text and "weight_noise" in caplog.text
    before = destination.read_bytes()
    farm._pilot_magnitudes(env, groups)
    assert destination.read_bytes() == before
    # Different measured results must not overwrite a frozen summary.
    _write_rows(_uri(env, groups[1][0]), _pilot_rows(env, groups[1][0], 7))
    with pytest.raises(FileExistsError):
        farm._pilot_magnitudes(env, groups)


@pytest.mark.parametrize("field,value", [("seed", 99999), ("success", 1), ("change_id", "c099"),
                                         ("scenario_id", "aloha-s01"), ("env", "aloha"), ("tier", "expensive")])
def test_pilot_rejects_unpaired_or_wrong_key_rows(field: str, value: object, manifests: dict) -> None:
    env, groups = "pusht", farm._guide_plan("pusht", "pilot")
    for keys in groups:
        for key in keys:
            _write_rows(_uri(env, key), _pilot_rows(env, key, 8))
    rows = _pilot_rows(env, groups[0][0], 8)
    if field == "success":
        for row in rows:
            row[field] = value
    else:
        rows[0][field] = value
    _write_rows(_uri(env, groups[0][0]), rows)
    with pytest.raises(ValueError, match="matched seeds and correct key"):
        farm._pilot_magnitudes(env, groups)


def test_cloud_reads_disable_sdk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(storage, "_blob", lambda location: SimpleNamespace(download_as_bytes=lambda **kwargs: (calls.append((location, kwargs)), b"payload")[1]))
    assert farm._read("gs://bucket/manifests/scenarios_pusht.json") == b"payload"
    assert calls == [(("bucket", "manifests/scenarios_pusht.json"), {"retry": None})]


def test_run_cli_forwards_only_specified_flags_and_preserves_exit_status(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(farm, "_run", lambda *args: (calls.append(args), 3)[1])
    assert farm._run_cli(["--env", "pusht"]) is False
    assert farm._run_cli([]) is False
    with pytest.raises(SystemExit) as result:
        farm._run_cli(["run", "--env", "aloha", "--phase", "pilot", "--shard", "2", "--n-shards", "44", "--procs", "4", "--max-hours", "5.5"])
    assert result.value.code == 3
    assert calls == [("aloha", "pilot", 2, 44, 4, 5.5)]
