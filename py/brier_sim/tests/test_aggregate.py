from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from brier_sim import aggregate, storage


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _path(uri: str) -> Path:
    return Path(unquote(urlsplit(uri).path))


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "source"
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    monkeypatch.setenv("BRIER_DATA", str(root))
    pin_bytes = Path(aggregate.__file__).with_name("pins.json").read_bytes()
    pins = json.loads(pin_bytes)
    sizing = {"pins_sha256": hashlib.sha256(pin_bytes).hexdigest(),
              "environments": {env: {"changes": 2, "scenarios": 2} for env in ("pusht", "aloha")}}
    (manifests / "sizing.json").write_text(json.dumps(sizing))
    for env, start in (("pusht", 1), ("aloha", 81)):
        scenarios = [{"scenario_id": f"{env}-s{index:02}", "params": {"pose_bin": 1, "lighting": "normal"},
                      "seeds": list(range(index * 50, (index + 1) * 50))} for index in range(1, 17)]
        (manifests / f"scenarios_{env}.json").write_text(json.dumps(list(reversed(scenarios))))
        cards = [{"change_id": f"c{index:03}", "env": env, "families": [{"family": "benign", "params": {}}],
                  "desc": "Refactor logging", "seed": 2026 + index} for index in range(start, start + 2)]
        (manifests / f"changes_{env}.jsonl").write_text("".join(json.dumps(card) + "\n" for card in reversed(cards)))
        keys = [("base", item, "baseline") for item in scenarios]
        keys += [(card["change_id"], item, tier) for card in cards for item in scenarios[:2] for tier in ("cheap", "expensive")]
        for change_id, scenario, tier in keys:
            count = {"baseline": 50, "cheap": 3, "expensive": 20}[tier]
            rows = []
            for index, seed in enumerate(scenario["seeds"][:count]):
                row = {field.name: (None if field.nullable else "value" if pa.types.is_string(field.type)
                                   else False if pa.types.is_boolean(field.type) else 0) for field in storage._SCHEMA}
                row.update(env=env, change_id=change_id, scenario_id=scenario["scenario_id"], tier=tier,
                           episode_idx=index, seed=seed, success=index < 40, max_steps=400, steps=400,
                           policy_repo=pins["policies"][env]["repo_id"], policy_revision=pins["policies"][env]["revision"],
                           lerobot_version=pins["lerobot_version"], terminal_state_json="{}")
                row.update({"coverage_final": .2, "coverage_max": .4, "block_disp_px": 10.} if env == "pusht" else
                           {"touched_right": False, "touched_left": False, "lifted": False, "table_contact_steps": 0})
                rows.append(row)
            storage.write_parquet(list(reversed(rows)), storage.key_uri(env, change_id, scenario["scenario_id"], tier))
    return root


def test_tiny_two_changes_by_two_scenarios_and_all_baselines(data: Path, tmp_path: Path) -> None:
    out = tmp_path / "output"
    aggregate._main(["--out", str(out)])
    episodes, changes, scenarios = [_jsonl(out / f"{name}.jsonl") for name in ("episodes", "changes", "scenarios")]
    assert len(episodes) == 2 * (16 * 50 + 2 * 2 * (3 + 20))
    assert len(changes) == 4 and len(scenarios) == 32
    assert [card["change_id"] for card in changes] == ["c001", "c002", "c081", "c082"]
    assert changes[0] == _jsonl(data / "manifests/changes_pusht.jsonl")[1]
    assert all(set(row) == set(storage._SCHEMA.names) for row in episodes)
    assert all(item["baseline_success"] == .8 and len(item["seeds"]) == 50 for item in scenarios)
    assert {item["env"] for item in scenarios} == {"pusht", "aloha"}
    assert [row["episode_idx"] for row in episodes[:50]] == list(range(50))
    assert all(row["change_id"] == "base" for row in episodes if row["tier"] == "baseline")
    assert all(int(row["scenario_id"][-2:]) <= 2 for row in episodes if row["tier"] != "baseline")


def test_missing_keys_lists_every_absence_before_creating_outputs(data: Path, tmp_path: Path) -> None:
    missing = [storage.key_uri("pusht", "base", "pusht-s16", "baseline"),
               storage.key_uri("pusht", "c002", "pusht-s02", "expensive"),
               storage.key_uri("aloha", "base", "aloha-s01", "baseline")]
    for uri in missing:
        _path(uri).unlink()
    _path(storage.key_uri("pusht", "c001", "pusht-s01", "cheap")).write_bytes(b"corrupt but not read before missing audit")
    out = tmp_path / "output"
    with pytest.raises(FileNotFoundError) as error:
        aggregate._aggregate(out)
    assert all(uri in str(error.value) for uri in missing)
    assert str(error.value).count(".parquet") == 3
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["row_count", "seed", "episode_idx", "env", "change_id", "scenario_id", "tier",
                                      "policy_repo", "policy_revision", "lerobot_version", "nan"])
def test_corrupt_key_preserves_existing_outputs(data: Path, tmp_path: Path, mutation: str) -> None:
    path = _path(storage.key_uri("pusht", "c002", "pusht-s02", "expensive"))
    rows = pq.ParquetFile(path).read().to_pylist()
    if mutation == "row_count":
        rows.pop()
    elif mutation == "nan":
        rows[0]["reward_sum"] = float("nan")
    else:
        rows[0][mutation] = -1 if mutation in ("seed", "episode_idx") else "wrong"
    pq.write_table(pa.Table.from_pylist(rows, schema=storage._SCHEMA), path)
    out = tmp_path / "output"
    out.mkdir()
    for name in ("episodes", "changes", "scenarios"):
        (out / f"{name}.jsonl").write_text("existing\n")
    with pytest.raises(ValueError):
        aggregate._aggregate(out)
    assert all((out / f"{name}.jsonl").read_text() == "existing\n" for name in ("episodes", "changes", "scenarios"))


@pytest.mark.parametrize("kind", ["extra", "missing", "wrong_type", "null_success"])
def test_rejects_parquet_schema_drift(data: Path, tmp_path: Path, kind: str) -> None:
    path = _path(storage.key_uri("pusht", "c001", "pusht-s01", "cheap"))
    table = pq.ParquetFile(path).read()
    if kind == "extra":
        table = table.append_column("extra", pa.array([0] * 3))
    elif kind == "missing":
        table = table.drop(["host"])
    elif kind == "wrong_type":
        index = table.schema.get_field_index("episode_idx")
        table = table.set_column(index, "episode_idx", pa.array([0, 1, 2], pa.int64()))
    else:
        index = table.schema.get_field_index("success")
        table = table.set_column(index, pa.field("success", pa.bool_(), nullable=True), pa.array([None, False, False], pa.bool_()))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="28-column contract"):
        aggregate._aggregate(tmp_path / "output")


@pytest.mark.parametrize("kind", ["stale_pins", "changes_count", "scenario_count", "duplicate_change", "wrong_env", "unhashable_change",
                                  "duplicate_scenario", "unhashable_scenario", "missing_scenario", "missing_seeds", "duplicate_seeds", "negative_seed"])
def test_rejects_inconsistent_manifests(data: Path, tmp_path: Path, kind: str) -> None:
    manifests = data / "manifests"
    if kind in ("stale_pins", "changes_count", "scenario_count"):
        path = manifests / "sizing.json"
        item = json.loads(path.read_text())
        if kind == "stale_pins":
            item["pins_sha256"] = "wrong"
        else:
            item["environments"]["pusht"]["changes" if kind == "changes_count" else "scenarios"] = 0
        path.write_text(json.dumps(item))
    elif kind in ("duplicate_change", "wrong_env", "unhashable_change"):
        path = manifests / "changes_pusht.jsonl"
        items = _jsonl(path)
        if kind == "unhashable_change":
            items[1]["change_id"] = []
        else:
            items[1]["change_id" if kind == "duplicate_change" else "env"] = items[0]["change_id"] if kind == "duplicate_change" else "aloha"
        path.write_text("".join(json.dumps(item) + "\n" for item in items))
    else:
        path = manifests / "scenarios_pusht.json"
        items = json.loads(path.read_text())
        if kind == "duplicate_scenario":
            items[0] = items[1]
        elif kind == "unhashable_scenario":
            items[0]["scenario_id"] = {}
        elif kind == "missing_scenario":
            items.pop()
        elif kind == "missing_seeds":
            items[0]["seeds"] = None
        elif kind == "duplicate_seeds":
            items[0]["seeds"] = items[1]["seeds"]
        else:
            items[0]["seeds"][0] = -1
        path.write_text(json.dumps(items))
    with pytest.raises(ValueError):
        aggregate._aggregate(tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_gcs_reads_have_no_sdk_retries_and_ignore_unplanned_pilot_keys(data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Blob:
        def __init__(self, name: str) -> None:
            self.path = data / name.removeprefix("root/")

        def exists(self, *, retry: None) -> bool:
            assert retry is None
            calls.append(("exists", self.path))
            return self.path.is_file()

        def download_as_bytes(self, *, retry: None) -> bytes:
            assert retry is None
            calls.append(("download", self.path))
            return self.path.read_bytes()

    monkeypatch.setenv("BRIER_DATA", "gs://test-bucket/root")
    monkeypatch.setattr(storage, "_blob", lambda location: Blob(location[1]))
    pilot = data / "episodes/env=pusht/tier=pilot/change=c1001/pusht-s01.parquet"
    pilot.parent.mkdir(parents=True)
    pilot.write_bytes(b"not a parquet file")
    out = tmp_path / "output"
    aggregate._aggregate(out)
    assert len([item for item in calls if item[0] == "exists"]) == 48
    assert all(path != pilot for _, path in calls)
    assert len(_jsonl(out / "episodes.jsonl")) == 1784


def test_requires_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRIER_DATA", raising=False)
    with pytest.raises(ValueError, match="BRIER_DATA is required"):
        aggregate._aggregate(tmp_path)


def test_local_file_uri_root(data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRIER_DATA", data.as_uri() + "/")
    out = tmp_path / "file-uri-output"
    aggregate._aggregate(out)
    assert len(_jsonl(out / "episodes.jsonl")) == 1784


@pytest.mark.parametrize("env,field,value", [("pusht", "steps", 0), ("pusht", "max_steps", 1),
                                            ("pusht", "wall_seconds", -1.), ("pusht", "action_flip_rate", 1.1),
                                            ("pusht", "reward_sum", -1.), ("pusht", "final_reward", 2.),
                                            ("pusht", "coverage_max", 1.1), ("pusht", "coverage_final", None),
                                            ("pusht", "block_disp_px", -1.), ("pusht", "touched_left", False),
                                            ("aloha", "table_contact_steps", -1), ("aloha", "lifted", None),
                                            ("aloha", "coverage_final", .1), ("aloha", "max_reward", float("inf"))])
def test_rejects_toxic_scalar_ranges_and_environment_nulls(
    data: Path, tmp_path: Path, env: str, field: str, value: object,
) -> None:
    path = _path(storage.key_uri(env, "base", f"{env}-s01", "baseline"))
    rows = pq.ParquetFile(path).read().to_pylist()
    rows[0][field] = value
    pq.write_table(pa.Table.from_pylist(rows, schema=storage._SCHEMA), path)
    with pytest.raises(ValueError, match="invalid scalar"):
        aggregate._aggregate(tmp_path / "output")
    assert not (tmp_path / "output").exists()
