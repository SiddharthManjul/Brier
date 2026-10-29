import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from brier_sim import farm


def test_catalog_has_global_ids_fixed_scenarios_and_benign_fraction() -> None:
    first = farm.catalog()
    assert first == farm.catalog()
    assert len(first["changes"]) == 160 and len(first["scenarios"]) == 32
    assert len({c["change_id"] for c in first["changes"]}) == 160
    for env, _ in farm._ENVS.values():
        cards = [c["card"] for c in first["changes"] if c["env"] == env]
        assert sum(c["families"][0]["family"] == "benign" for c in cards) == 28
        assert all(1 <= len(c["families"]) <= 2 for c in cards)
    assert {c["card"]["families"][0]["family"] for c in first["changes"]} >= {
        "occlusion",
        "chunk_len",
        "weight_noise",
        "benign",
    }


def test_shard_keys_skip_valid_results_and_reject_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    scenario = farm.scenarios(farm._ENVS["pusht"][0])[0]
    definitions = farm.catalog()
    definitions["changes"] = definitions["changes"][:2]
    definitions["scenarios"] = [scenario]
    monkeypatch.setattr(farm, "catalog", lambda: definitions)
    monkeypatch.setattr(farm, "scenarios", lambda _: [scenario])
    calls = []

    def run(
        env: str,
        policy: str,
        episodes: int,
        seed: int,
        change_id: str,
        scenario_id: str,
        tier: str,
        **kwargs: Any,
    ) -> Path:
        calls.append((change_id, tier))
        assert kwargs["scenario"] == scenario
        output = Path(os.environ["BRIER_DATA"]) / f"raw-{len(calls)}.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "env": env,
                        "policy": policy,
                        "change_id": change_id,
                        "scenario_id": scenario_id,
                        "tier": tier,
                        "seed": seed + i,
                        "success": False,
                        "steps": 2,
                        "terminal_state": "{}",
                        "reward_sum": 0.0,
                        "keyframes_uri": None,
                        "wall_seconds": 0.1,
                    }
                    for i in range(episodes)
                ],
                schema=farm._SCHEMA,
            ),
            output,
        )
        return output

    monkeypatch.setattr(farm, "run_episodes", run)
    keys = farm.run_shard("pusht", 1, 2)
    assert len(keys) == 2 and calls == [("c002", "cheap"), ("c002", "expensive")]
    assert farm.run_shard("pusht", 1, 2) == keys
    assert len(calls) == 2
    assert json.loads((tmp_path / "catalog.json").read_text()) == definitions
    assert os.environ["BRIER_DATA"] == str(tmp_path)
    Path(keys[0]).write_bytes(b"incomplete")
    with pytest.raises(Exception, match="Parquet"):
        farm.run_shard("pusht", 1, 2)


@pytest.mark.parametrize(
    "env,shard,shards", [("other", 0, 1), ("pusht", -1, 1), ("pusht", 0, 0)]
)
def test_shard_rejects_invalid_partition(env: str, shard: int, shards: int) -> None:
    with pytest.raises(ValueError):
        farm.run_shard(env, shard, shards)


def test_cloud_permission_error_is_not_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        farm.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stderr="403 permission denied"
        ),
    )
    with pytest.raises(RuntimeError, match="403"):
        farm._existing("gs://example/episode.parquet", Path("unused"))


def test_baselines_shard_scenarios_and_run_fifty_unmodified_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    calls = []

    def run(
        env: str,
        policy: str,
        count: int,
        seed: int,
        change_id: str,
        scenario_id: str,
        tier: str,
        **kwargs: Any,
    ) -> Path:
        calls.append((change_id, scenario_id, tier, count))
        assert kwargs["change"]["families"] == [{"family": "benign", "params": {}}]
        output = tmp_path / "raw.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "env": env,
                        "policy": policy,
                        "change_id": change_id,
                        "scenario_id": scenario_id,
                        "tier": tier,
                        "seed": seed + i,
                        "success": True,
                        "steps": 400,
                        "terminal_state": "{}",
                        "reward_sum": 4.0,
                        "keyframes_uri": None,
                        "wall_seconds": 1.0,
                    }
                    for i in range(count)
                ],
                schema=farm._SCHEMA,
            ),
            output,
        )
        return output

    monkeypatch.setattr(farm, "run_episodes", run)
    keys = farm.run_shard("aloha_transfer_cube", 1, 4, baseline=True)
    assert len(keys) == 4
    assert calls == [
        ("c081", f"aloha_transfer_cube-s{i:02}", "baseline", 50) for i in (2, 6, 10, 14)
    ]
    assert farm.run_shard("aloha_transfer_cube", 1, 4, baseline=True) == keys
    assert len(calls) == 4
