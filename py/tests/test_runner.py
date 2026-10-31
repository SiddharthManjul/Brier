import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, NoReturn

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from brier_sim import runner
from PIL import Image

ENV = "gym_pusht/PushT-v0"
POLICY = "lerobot/diffusion_pusht"
ARGS = (ENV, POLICY, 2, 17, "c001", "pusht-s01", "cheap")
SCHEMA = pa.schema(
    [
        pa.field("env", pa.string(), nullable=False),
        pa.field("policy", pa.string(), nullable=False),
        pa.field("change_id", pa.string(), nullable=False),
        pa.field("scenario_id", pa.string(), nullable=False),
        pa.field("tier", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("success", pa.bool_(), nullable=False),
        pa.field("steps", pa.int64(), nullable=False),
        pa.field("terminal_state", pa.string(), nullable=False),
        pa.field("reward_sum", pa.float64(), nullable=False),
        pa.field("keyframes_uri", pa.string()),
        pa.field("wall_seconds", pa.float64(), nullable=False),
    ]
)
_Step = tuple[dict[str, Any], float, bool, bool, dict[str, Any]]


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resets = 0
        self.observations: list[dict[str, torch.Tensor]] = []

    def reset(self) -> None:
        self.resets += 1

    def select_action(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        self.observations.append(observation)
        return torch.rand(1, 2) * 512


class _RecordedEnv(gym.Wrapper):
    def __init__(self) -> None:
        super().__init__(
            gym.make(ENV, obs_type="pixels_agent_pos", max_episode_steps=2)
        )
        self.seeds: list[int] = []
        self.episodes: list[list[_Step]] = []
        self.closed = False

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.seeds.append(kwargs["seed"])
        self.episodes.append([])
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> _Step:
        result = self.env.step(action)
        self.episodes[-1].append(result)
        return result

    def close(self) -> None:
        self.closed = True
        self.env.close()


_Rollout = tuple[Path, _RecordedEnv, _Policy]


@pytest.fixture
def rollout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Rollout:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    environment, policy = _RecordedEnv(), _Policy()
    monkeypatch.setattr(runner, "_load", lambda *_: (environment, policy))
    return tmp_path, environment, policy


def test_runner_persists_raw_telemetry_and_typed_schema(rollout: _Rollout) -> None:
    directory, environment, policy = rollout
    output = runner.run_episodes(*ARGS)
    assert isinstance(output, Path)
    assert output.is_relative_to(directory) and output.suffix == ".parquet"
    table = pq.read_table(output)
    assert table.schema == SCHEMA
    rows = table.to_pylist()
    assert len(rows) == 2
    assert environment.seeds == [17, 18]
    assert policy.resets == 2 and environment.closed
    assert len(policy.observations) == 4
    observation = policy.observations[0]
    assert observation["observation.state"].shape == (1, 2)
    image = observation["observation.image"]
    assert image.shape == (1, 3, 96, 96) and image.dtype == torch.float32
    assert 0 <= image.min() <= image.max() <= 1
    for index, row in enumerate(rows):
        assert [row[key] for key in SCHEMA.names[:5]] == [
            ENV,
            POLICY,
            "c001",
            "pusht-s01",
            "cheap",
        ]
        assert row["seed"] == 17 + index
        episode = environment.episodes[index]
        assert row["steps"] == len(episode) == 2
        assert row["success"] == any(step[4]["is_success"] for step in episode)
        assert row["reward_sum"] == pytest.approx(sum(step[1] for step in episode))
        terminal = json.loads(row["terminal_state"])
        assert terminal["terminated"] is False
        assert terminal["truncated"] is True
        trajectory = terminal["trajectory"]
        assert trajectory["action_transitions"] == 0
        assert trajectory["collision_steps"] == 0
        assert trajectory["object_final_height"] == 0
        assert trajectory["goal_distance_final"] >= trajectory["goal_distance_min"]
        for key, value in episode[-1][4].items():
            np.testing.assert_equal(terminal[key], value)
        files = list((directory / row["keyframes_uri"]).glob("*.png"))
        assert len(files) == 1 and files[0].name == "frame-0002.png"
        with Image.open(files[0]) as image:
            np.testing.assert_equal(np.asarray(image), episode[-1][0]["pixels"])
        assert row["wall_seconds"] >= 0


def test_runner_preserves_pinned_revision_in_loader_and_raw_provenance(
    rollout: _Rollout, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, environment, policy = rollout
    pinned = "a" * 40
    received = []

    def load(env: str, checkpoint: str, tier: str, *, revision: str):
        received.append((env, checkpoint, tier, revision))
        return environment, policy

    monkeypatch.setattr(runner, "_load", load)
    rows = pq.read_table(runner.run_episodes(*ARGS, revision=pinned)).to_pylist()
    assert received == [(ENV, POLICY, "cheap", pinned)]
    assert all(row["policy"] == POLICY for row in rows)
    assert all(json.loads(row["terminal_state"])["checkpoint_revision"] == pinned for row in rows)


@pytest.mark.parametrize("revision", ["main", "a" * 39, "z" * 40, True, 1])
def test_runner_rejects_unpinned_revisions_before_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, revision,
) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    load = runner._load

    def unexpected(*args, **kwargs):
        pytest.fail("invalid policy revisions must fail before loading")

    monkeypatch.setattr(runner, "_load", unexpected)
    with pytest.raises(ValueError, match="immutable 40-hex"):
        runner.run_episodes(*ARGS, revision=revision)
    with pytest.raises(ValueError, match="immutable 40-hex"):
        load(ENV, POLICY, "cheap", revision=revision)


@pytest.mark.parametrize("aloha", [False, True])
@pytest.mark.parametrize("tier,success,writer_failure", [("expensive", False, False), ("expensive", True, False),
                                                       ("cheap", False, False), ("baseline", False, False),
                                                       ("expensive", False, True)])
def test_runner_records_failure_frames_and_cheap_final_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aloha: bool, tier: str, success: bool, writer_failure: bool,
) -> None:
    clock = [0.0]

    class FrameEnv:
        def __init__(self) -> None:
            self.index, self.closed = 0, False
            self.pixels = np.zeros((4, 5, 3), dtype=np.uint8)
            self.unwrapped = SimpleNamespace(_env=SimpleNamespace(physics=SimpleNamespace(
                data=SimpleNamespace(qpos=np.zeros(3), qvel=np.zeros(3)))))

        def observation(self) -> dict[str, Any]:
            self.pixels.fill(self.index)
            return {"pixels": {"top": self.pixels} if aloha else self.pixels,
                    "agent_pos": np.zeros(2)}

        def reset(self, **kwargs: Any):
            return self.observation(), {}

        def step(self, action: np.ndarray):
            self.index += 1
            clock[0] += 1
            return self.observation(), 0.0, False, self.index == 8, {"is_success": success}

        def close(self) -> None:
            self.closed = True

    environment = FrameEnv()
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    monkeypatch.setattr(runner, "_load", lambda *_: (environment, _Policy()))
    monkeypatch.setattr(runner, "snapshot", lambda _: {})
    monkeypatch.setattr(runner, "summarize", lambda *_: {})
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    save = runner.save_keyframes

    def timed_save(*args, **kwargs):
        directory = save(*args, **kwargs)
        clock[0] += 3
        return directory

    monkeypatch.setattr(runner, "save_keyframes", timed_save)
    args = ("gym_aloha/AlohaTransferCube-v0", "lerobot/act_aloha_sim_transfer_cube_human",
            1, 17, "c081", "aloha_transfer_cube-s01", tier) if aloha else (
                ENV, POLICY, 1, 17, "c001", "pusht-s01", tier)
    if writer_failure:
        def fail(*args, **kwargs):
            raise OSError("Parquet write failed")

        monkeypatch.setattr(runner.pq, "write_table", fail)
        with pytest.raises(OSError, match="Parquet write failed"):
            runner.run_episodes(*args)
        assert environment.closed and not list(tmp_path.rglob("*.png"))
        assert not list(tmp_path.rglob("*.parquet"))
        return
    row = pq.read_table(runner.run_episodes(*args)).to_pylist()[0]
    assert environment.closed and row["success"] is success
    if tier == "expensive" and not success:
        assert row["keyframes_uri"].startswith("public/keyframes/")
        directory = tmp_path / row["keyframes_uri"]
        files = sorted(directory.glob("*.png"))
        assert len(files) == 8
        assert [int(path.stem.split("-")[1]) for path in files] == [0, 1, 2, 3, 4, 5, 6, 8]
        for path in files:
            with Image.open(path) as image:
                assert np.all(np.asarray(image) == int(path.stem.split("-")[1]))
        assert row["wall_seconds"] == 11.0
    elif tier == "cheap":
        files = list((tmp_path / row["keyframes_uri"]).glob("*.png"))
        assert len(files) == 1 and files[0].name == "frame-0008.png"
        with Image.open(files[0]) as image:
            assert np.all(np.asarray(image) == 8)
        assert row["wall_seconds"] == 8.0
    else:
        assert row["keyframes_uri"] is None
        assert not list(tmp_path.rglob("*.png"))
        assert row["wall_seconds"] == 8.0


def test_runner_repeats_environment_and_torch_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    environments: list[_RecordedEnv] = []

    def load(*_: str) -> tuple[_RecordedEnv, _Policy]:
        environment = _RecordedEnv()
        environments.append(environment)
        return environment, _Policy()

    monkeypatch.setattr(runner, "_load", load)
    first = pq.read_table(runner.run_episodes(*ARGS)).to_pylist()
    second = pq.read_table(runner.run_episodes(*ARGS)).to_pylist()
    for left, right in zip(first, second, strict=True):
        left_frame = next((tmp_path / left["keyframes_uri"]).glob("*.png"))
        right_frame = next((tmp_path / right["keyframes_uri"]).glob("*.png"))
        assert left_frame.read_bytes() == right_frame.read_bytes()
    for rows in (first, second):
        for row in rows:
            row.pop("wall_seconds")
            row.pop("keyframes_uri")
    assert first == second
    assert len(environments) == 2 and all(env.closed for env in environments)


def test_runner_uses_fixed_scenario_seed_prefix(rollout: _Rollout) -> None:
    _, environment, _ = rollout
    scenario = runner.scenarios(ENV)[3]
    args = (
        ENV,
        POLICY,
        2,
        scenario["seeds"][0],
        "c001",
        scenario["scenario_id"],
        "cheap",
    )
    rows = pq.read_table(runner.run_episodes(*args, scenario=scenario)).to_pylist()
    assert environment.seeds == scenario["seeds"][:2]
    assert [row["seed"] for row in rows] == scenario["seeds"][:2]
    assert environment.closed


def test_runner_rejects_inconsistent_scenario_before_loading(rollout: _Rollout) -> None:
    scenario = runner.scenarios(ENV)[0]
    with pytest.raises(ValueError, match="scenario"):
        runner.run_episodes(*ARGS, scenario=scenario)


def test_runner_applies_ordered_change_card(
    rollout: _Rollout, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, environment, policy = rollout
    raw_actions: list[torch.Tensor] = []
    sent_actions: list[np.ndarray] = []
    change_seeds: list[int] = []
    select_action, step, apply = (
        policy.select_action,
        environment.step,
        runner.apply_change,
    )

    def select(observation: dict[str, torch.Tensor]) -> torch.Tensor:
        action = select_action(observation)
        raw_actions.append(action.clone())
        return action

    def send(action: np.ndarray) -> _Step:
        sent_actions.append(action.copy())
        return step(action)

    def wrap(base: Any, card: dict[str, Any], *, seed: int) -> Any:
        change_seeds.append(seed)
        return apply(base, card, seed=seed)

    monkeypatch.setattr(policy, "select_action", select)
    monkeypatch.setattr(environment, "step", send)
    monkeypatch.setattr(runner, "apply_change", wrap)
    card = {
        "description": "Retune gain and introduce controller drift",
        "families": [
            {"family": "action_scale", "params": {"scale": 0.8}},
            {"family": "action_bias", "params": {"drift": 0.1}},
        ],
    }
    runner.run_episodes(*ARGS, change=card)
    assert change_seeds == [1]
    assert policy.resets == 3 and environment.closed
    assert len(sent_actions) == len(raw_actions) == 4
    for index, (raw, sent) in enumerate(zip(raw_actions, sent_actions, strict=True)):
        expected = raw.squeeze(0).numpy() * 0.8 + 0.1 * (index % 2 + 1)
        np.testing.assert_allclose(sent, expected)


def test_runner_closes_environment_when_change_card_is_invalid(
    rollout: _Rollout,
) -> None:
    directory, environment, _ = rollout
    card = {
        "description": "Invalid gain",
        "families": [{"family": "action_scale", "params": {"scale": -1}}],
    }
    with pytest.raises(ValueError):
        runner.run_episodes(*ARGS, change=card)
    assert environment.closed
    assert not list(directory.rglob("*.parquet"))


@pytest.mark.parametrize(
    "index,value,message",
    [
        (0, "pusht", "env"),
        (1, "lerobot/act_aloha_sim_transfer_cube_human", "policy"),
        (2, 0, "episodes"),
        (2, True, "episodes"),
        (3, -1, "seed"),
        (3, True, "seed"),
        (4, "change1", "change_id"),
        (5, "aloha_transfer_cube-s01", "scenario_id"),
        (6, "unknown", "tier"),
    ],
)
def test_runner_rejects_invalid_inputs_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index: int,
    value: str | int | bool,
    message: str,
) -> None:
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))

    def unexpected_load(*_: str) -> NoReturn:
        pytest.fail("Invalid inputs must be rejected before model loading")

    monkeypatch.setattr(runner, "_load", unexpected_load)
    args = list(ARGS)
    args[index] = value
    with pytest.raises(ValueError, match=message):
        runner.run_episodes(*args)
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("location", [None, "", "gs://brier-data"])
def test_runner_requires_local_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, location: str | None
) -> None:
    if location is None:
        monkeypatch.delenv("BRIER_DATA", raising=False)
    else:
        monkeypatch.setenv("BRIER_DATA", location)

    def unexpected_load(*_: str) -> NoReturn:
        pytest.fail("BRIER_DATA must be validated before model loading")

    monkeypatch.setattr(runner, "_load", unexpected_load)
    with pytest.raises(ValueError, match="BRIER_DATA"):
        runner.run_episodes(*ARGS)


def test_runner_closes_environment_without_partial_parquet(
    rollout: _Rollout, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, environment, policy = rollout
    select_action = policy.select_action

    def fail(observation: dict[str, torch.Tensor]) -> torch.Tensor:
        if policy.resets == 2:
            raise RuntimeError("policy inference failed")
        return select_action(observation)

    monkeypatch.setattr(policy, "select_action", fail)
    with pytest.raises(RuntimeError, match="policy inference failed"):
        runner.run_episodes(*ARGS)
    assert environment.closed
    assert len(environment.episodes[0]) == 2
    assert not list(directory.rglob("*.parquet"))


def test_runner_removes_partial_parquet_after_writer_failure(
    rollout: _Rollout, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, environment, _ = rollout

    def fail(_table: pa.Table, output_file: BinaryIO) -> NoReturn:
        output_file.write(b"PAR1partial")
        raise OSError("disk failed")

    monkeypatch.setattr(runner.pq, "write_table", fail)
    with pytest.raises(OSError, match="disk failed"):
        runner.run_episodes(*ARGS)
    assert environment.closed
    assert not list(directory.rglob("*.parquet"))


def test_runner_preserves_existing_parquet_on_output_collision(
    rollout: _Rollout, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, environment, _ = rollout
    output_dir = directory / "episodes"
    output_dir.mkdir()
    existing = output_dir / "existing.parquet"
    existing.write_bytes(b"existing episode data")
    monkeypatch.setattr(runner, "uuid4", lambda: "existing")
    with pytest.raises(FileExistsError):
        runner.run_episodes(*ARGS)
    assert environment.closed
    assert existing.read_bytes() == b"existing episode data"
    assert list(directory.rglob("*.parquet")) == [existing]


@pytest.mark.parametrize(
    "environment_id,checkpoint,policy_name,horizon",
    [
        (ENV, POLICY, "DiffusionPolicy", 300),
        (
            "gym_aloha/AlohaTransferCube-v0",
            "lerobot/act_aloha_sim_transfer_cube_human",
            "ACTPolicy",
            400,
        ),
    ],
)
@pytest.mark.parametrize("tier", ["cheap", "expensive", "baseline"])
@pytest.mark.parametrize("revision", [None, "a" * 40])
def test_loader_uses_strict_checkpoint_and_tier_horizon(
    monkeypatch: pytest.MonkeyPatch,
    environment_id: str,
    checkpoint: str,
    policy_name: str,
    horizon: int,
    tier: str,
    revision: str | None,
) -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    environment, policy = _RecordedEnv(), _Policy()
    config = SimpleNamespace(pretrained_backbone_weights="IMAGENET1K_V1")
    loaded: list[str] = []
    phases = []
    expected_revision = revision

    def make(identifier: str, **kwargs: Any) -> _RecordedEnv:
        assert identifier == environment_id
        assert kwargs == {
            "obs_type": "pixels_agent_pos",
            "max_episode_steps": horizon // 2 if tier == "cheap" else horizon,
        }
        return environment

    def load_config(identifier: str, **kwargs: Any) -> SimpleNamespace:
        assert identifier == checkpoint
        assert kwargs == {"cli_overrides": ["--device=cpu"], **({"revision": revision} if revision else {})}
        phases.append("config")
        return config

    monkeypatch.setattr(runner.gym, "make", make)
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", load_config)
    for model in (ACTPolicy, DiffusionPolicy):

        def load_policy(
            identifier: str,
            *,
            config: SimpleNamespace,
            strict: bool,
            revision: str | None = None,
            name: str = model.__name__,
        ) -> _Policy:
            assert identifier == checkpoint and strict is True
            assert config.pretrained_backbone_weights is None
            assert phases == ["config"]
            assert revision == expected_revision
            phases.append("policy")
            loaded.append(name)
            return policy

        monkeypatch.setattr(model, "from_pretrained", load_policy)
    try:
        assert runner._load(environment_id, checkpoint, tier, revision=revision) == (environment, policy)
        assert loaded == [policy_name]
        assert phases == ["config", "policy"]
        assert policy.training is False
    finally:
        environment.close()
