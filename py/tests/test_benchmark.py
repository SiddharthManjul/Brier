import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from brier_sim import benchmark as module
from brier_sim.scenarios import scenarios

ENV = "gym_pusht/PushT-v0"
ALOHA = "gym_aloha/AlohaTransferCube-v0"


@pytest.fixture
def rollouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

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
        rows = [
            {
                "env": env,
                "policy": policy,
                "change_id": change_id,
                "scenario_id": scenario_id,
                "tier": tier,
                "seed": kwargs["scenario"]["seeds"][i],
                "wall_seconds": float(i + 1),
            }
            for i in range(episodes)
        ]
        path = tmp_path / f"{tier}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        calls.append({"rows": rows, "path": path, "kwargs": kwargs})
        return path

    monkeypatch.setattr(module, "run_episodes", run)
    monkeypatch.setattr(module.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(module.socket, "gethostname", lambda: "test-host")
    return calls


@pytest.mark.parametrize("env,change_id", [(ENV, "c001"), (ALOHA, "c081")])
def test_benchmark_reports_actual_episode_durations(
    tmp_path: Path,
    rollouts: list[dict[str, Any]],
    env: str,
    change_id: str,
) -> None:
    output = tmp_path / "timing.json"
    result = module.benchmark(env, output)
    assert json.loads(output.read_text()) == result
    assert result["env"] == env and result["total_episodes"] == 23
    assert result["scenario_id"] == scenarios(env)[0]["scenario_id"]
    assert result["hostname"] == "test-host" and result["platform"] == "test-platform"
    assert set(result["tiers"]) == {"cheap", "expensive"}
    for index, (tier, count) in enumerate((("cheap", 3), ("expensive", 20))):
        summary = result["tiers"][tier]
        assert summary["episodes"] == count
        assert summary["total_seconds"] == count * (count + 1) / 2
        assert summary["seconds_per_episode"] == (count + 1) / 2
        assert summary["parquet_path"] == str(rollouts[index]["path"])
        assert rollouts[index]["rows"][0]["change_id"] == change_id
        assert rollouts[index]["kwargs"]["scenario"] == scenarios(env)[0]
        assert rollouts[index]["kwargs"]["change"]["families"] == [
            {"family": "benign", "params": {}}
        ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("env", "bad"),
        ("policy", "bad"),
        ("change_id", "bad"),
        ("scenario_id", "bad"),
        ("tier", "baseline"),
        ("seed", -1),
        ("wall_seconds", -1.0),
        ("wall_seconds", float("nan")),
        ("wall_seconds", None),
        ("wall_seconds", True),
        ("count", 2),
    ],
)
def test_benchmark_rejects_invalid_raw_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rollouts: list[dict[str, Any]],
    field: str,
    value: Any,
) -> None:
    original = module.run_episodes

    def run(*args: Any, **kwargs: Any) -> Path:
        path = original(*args, **kwargs)
        rows = rollouts[-1]["rows"]
        if field == "count":
            rows = rows[:value]
        else:
            rows[0][field] = value
            if field == "wall_seconds" and value is True:
                for row in rows:
                    row[field] = value
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path

    monkeypatch.setattr(module, "run_episodes", run)
    output = tmp_path / "timing.json"
    with pytest.raises(ValueError, match="benchmark.*(record|requires)"):
        module.benchmark(ENV, output)
    assert not output.exists()


def test_benchmark_rejects_bad_inputs_before_running(
    tmp_path: Path,
    rollouts: list[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError, match="Unsupported env"):
        module.benchmark("unknown", tmp_path / "timing.json")
    with pytest.raises(TypeError, match="pathlib.Path"):
        module.benchmark(ENV, "timing.json")
    with pytest.raises(ValueError, match="parent directory"):
        module.benchmark(ENV, tmp_path / "missing" / "timing.json")
    existing = tmp_path / "timing.json"
    existing.write_text("preserve")
    with pytest.raises(FileExistsError, match="already exists"):
        module.benchmark(ENV, existing)
    assert existing.read_text() == "preserve" and not rollouts
