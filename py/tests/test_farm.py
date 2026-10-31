import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from brier_sim import farm

_TRAJECTORY = {
    "collision_steps": 0,
    "action_direction_changes": 0,
    "action_transitions": 1,
    "object_max_height": 0.0,
    "object_final_height": 0.0,
    "goal_distance_min": 1.0,
    "goal_distance_final": 1.0,
}


def _raw_file(
    path: Path,
    change: dict[str, Any],
    scenario: dict[str, Any],
    tier: str,
    **overrides: Any,
) -> Path:
    count = {"cheap": 3, "expensive": 20, "baseline": 50}[tier]
    rows = [
        {
            "env": change["env"],
            "policy": "lerobot/diffusion_pusht",
            "change_id": change["change_id"],
            "scenario_id": scenario["scenario_id"],
            "tier": tier,
            "seed": seed,
            "success": False,
            "steps": 2,
            "terminal_state": json.dumps({"trajectory": _TRAJECTORY}),
            "reward_sum": 0.0,
            "keyframes_uri": None,
            "wall_seconds": 0.1,
            **overrides,
        }
        for seed in scenario["seeds"][:count]
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=farm._SCHEMA), path)
    return path


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


@pytest.mark.parametrize("tier,count", [("cheap", 1), ("expensive", 8)])
def test_upload_keyframes_preserves_relative_public_paths_before_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: str, count: int,
) -> None:
    relative = "public/keyframes/pusht/c001/pusht-s01/0/attempt"
    directory = tmp_path / relative
    directory.mkdir(parents=True)
    for i in range(count):
        (directory / f"frame-{i:04}.png").write_bytes(b"test-image")
    scenario = farm.scenarios(farm._ENVS["pusht"][0])[0]
    change = farm.catalog()["changes"][0]
    parquet = _raw_file(tmp_path / "episode.parquet", change, scenario, tier, keyframes_uri=relative)
    copies = []
    monkeypatch.setattr(farm, "_copy", lambda source, target: copies.append((source, target)))
    farm._upload_keyframes(parquet, tmp_path, "gs://test-bucket")
    assert len(copies) == count
    assert copies[0] == (str(directory / "frame-0000.png"),
                         f"gs://test-bucket/{relative}/frame-0000.png")
    (directory / "frame-0000.png").unlink()
    with pytest.raises(ValueError, match="eight"):
        farm._upload_keyframes(parquet, tmp_path, "gs://test-bucket")
    parquet = _raw_file(parquet, change, scenario, "expensive", keyframes_uri="../outside")
    with pytest.raises(ValueError, match="relative directory"):
        farm._upload_keyframes(parquet, tmp_path, "gs://test-bucket")


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
                        "seed": kwargs["scenario"]["seeds"][i],
                        "success": False,
                        "steps": 2,
                        "terminal_state": json.dumps({"trajectory": _TRAJECTORY}),
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
                        "seed": kwargs["scenario"]["seeds"][i],
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


@pytest.mark.parametrize(
    "prefix", ["gs://", "gs:///", "gs:bucket", "gs://bucket?x=1", "https://bucket", " "]
)
def test_rejects_invalid_storage_prefix_before_creating_files(
    prefix: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRIER_DATA", prefix)
    monkeypatch.setattr(
        farm, "catalog", lambda: pytest.fail("invalid prefix reached catalog")
    )
    with pytest.raises(ValueError, match="BRIER_DATA"):
        farm.run_shard("pusht", 0, 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"steps": 0},
        {"terminal_state": "not JSON"},
        {"terminal_state": "[]"},
        {"terminal_state": "{}"},
        {
            "terminal_state": json.dumps(
                {"trajectory": {**_TRAJECTORY, "collision_steps": -1}}
            )
        },
        {
            "terminal_state": json.dumps(
                {"trajectory": {**_TRAJECTORY, "collision_steps": True}}
            )
        },
        {
            "terminal_state": json.dumps(
                {"trajectory": {**_TRAJECTORY, "collision_steps": float("nan")}}
            )
        },
        {
            "terminal_state": json.dumps(
                {"trajectory": {**_TRAJECTORY, "action_direction_changes": 2}}
            )
        },
        {
            "terminal_state": json.dumps(
                {"trajectory": {**_TRAJECTORY, "goal_distance_min": 2}}
            )
        },
    ],
)
def test_rejects_completed_records_that_cannot_be_labeled(
    tmp_path: Path,
    overrides: dict[str, Any],
) -> None:
    change = farm.catalog()["changes"][0]
    scenario = farm.scenarios(change["env"])[0]
    path = _raw_file(
        tmp_path / "bad.parquet", change, scenario, "expensive", **overrides
    )
    with pytest.raises(ValueError, match="completed rollout"):
        farm._validate(path, change, scenario, "expensive")


@pytest.mark.parametrize("winner", ["catalog", "episode"])
def test_remote_upload_validates_concurrent_winner_and_restores_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winner: str,
) -> None:
    prefix = "gs://brier-test/rollouts"
    monkeypatch.setenv("BRIER_DATA", prefix)
    definitions = farm.catalog()
    definitions["changes"] = definitions["changes"][:1]
    scenario = definitions["scenarios"][0]
    definitions["scenarios"] = [scenario]
    monkeypatch.setattr(farm, "catalog", lambda: definitions)
    monkeypatch.setattr(farm, "scenarios", lambda _: [scenario])
    cloud: dict[str, bytes] = {}
    calls: list[str] = []

    def gcloud(arguments: list[str], **kwargs: Any) -> SimpleNamespace:
        if arguments[2] == "objects":
            return SimpleNamespace(
                returncode=0 if arguments[4] in cloud else 1, stderr="404 Not Found"
            )
        source, destination = arguments[-2:]
        if source.startswith("gs://"):
            assert "--no-clobber" not in arguments
            Path(destination).write_bytes(cloud[source])
        else:
            assert "--no-clobber" in arguments
            if winner == "catalog" and destination.endswith("catalog.json"):
                cloud[destination] = b'{"different": "catalog"}'
            elif winner == "episode" and destination.endswith("cheap.parquet"):
                bad = _raw_file(
                    tmp_path / "winner.parquet",
                    definitions["changes"][0],
                    scenario,
                    "cheap",
                    steps=0,
                )
                cloud[destination] = bad.read_bytes()
            else:
                cloud.setdefault(destination, Path(source).read_bytes())
        return SimpleNamespace(returncode=0, stderr="")

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
        calls.append(tier)
        return _raw_file(
            Path(os.environ["BRIER_DATA"]) / "raw.parquet",
            definitions["changes"][0],
            scenario,
            tier,
        )

    monkeypatch.setattr(farm.subprocess, "run", gcloud)
    monkeypatch.setattr(farm, "run_episodes", run)
    with pytest.raises(
        ValueError,
        match="catalog differs" if winner == "catalog" else "completed rollout key",
    ):
        farm.run_shard("pusht", 0, 1)
    assert calls == ([] if winner == "catalog" else ["cheap"])
    assert os.environ["BRIER_DATA"] == prefix


def test_failed_download_preserves_cache_and_removes_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cached.parquet"
    cache.write_bytes(b"previous valid cache")

    def gcloud(arguments: list[str], **kwargs: Any) -> SimpleNamespace:
        if arguments[2] == "objects":
            return SimpleNamespace(returncode=0, stderr="")
        Path(arguments[-1]).write_bytes(b"partial download")
        raise farm.subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(farm.subprocess, "run", gcloud)
    with pytest.raises(farm.subprocess.CalledProcessError):
        farm._existing("gs://brier-test/episode.parquet", cache)
    assert cache.read_bytes() == b"previous valid cache"
    assert list(tmp_path.iterdir()) == [cache]


def test_spot_invalid_shard_range_stops_before_cloud_calls(tmp_path: Path) -> None:
    marker = tmp_path / "cloud-called"
    gcloud = tmp_path / "gcloud"
    gcloud.write_text(
        f"#!/bin/sh\nprintf called > {shlex.quote(str(marker))}\nexit 99\n"
    )
    gcloud.chmod(0o755)
    repository = Path(farm.__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(repository / "py/brier_sim/gcp_spot.sh"),
         "up", "pusht", "pilot", "1", "2", "2"],
        check=False,
        cwd=repository,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "GCP_PROJECT": "example-project",
            "BRIER_DATA": "gs://brier-test/rollouts",
        },
    )
    assert result.returncode == 2
    assert "first_shard + n_vms <= n_shards" in result.stderr
    assert not marker.exists()
