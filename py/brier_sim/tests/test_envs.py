import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch
from brier_sim import envs


class _RecordedEnv(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            "pixels": gym.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8),
            "agent_pos": gym.spaces.Box(-1, 1, (2,), dtype=np.float32),
        }
    )
    action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    _max_episode_steps = 3

    def __init__(self, trace: list[str], successes: tuple[bool, ...]) -> None:
        self.trace, self.successes = trace, successes
        self.seeds: list[int] = []
        self.actions: list[np.ndarray] = []
        self.closed = False

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "pixels": np.full((8, 8, 3), 128, dtype=np.uint8),
            "agent_pos": np.zeros(2, dtype=np.float32),
        }

    def reset(self, *, seed: int | None = None, options: Any = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self.trace.append("reset")
        self.seeds.append(seed)
        self.steps = 0
        return self._observation(), {"is_success": False}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self.trace.append("step")
        self.actions.append(action.copy())
        success = self.successes[self.steps]
        self.steps += 1
        return self._observation(), 0.0, False, self.steps == 3, {"is_success": success}

    def close(self) -> None:
        self.closed = True


class _Policy(torch.nn.Module):
    def __init__(self, trace: list[str]) -> None:
        super().__init__()
        self.trace = trace
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def reset(self) -> None:
        self.trace.append("policy_reset")

    def select_action(self, observation: dict) -> torch.Tensor:
        assert torch.is_inference_mode_enabled()
        assert torch.get_num_threads() == 1
        assert observation["task"] == [""]
        self.trace.append("select_action")
        return torch.zeros((1, 2))


def _loaded(trace: list[str], successes: tuple[bool, ...]) -> tuple:
    raw = _RecordedEnv(trace, successes)
    vec = gym.vector.SyncVectorEnv([lambda: raw])
    return vec, _Policy(trace)


@pytest.mark.parametrize("successes", [(True, False, False), (False, False, True)])
def test_plain_episodes_preserve_hooks_success_and_seeds(
    successes: tuple[bool, ...], monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    loaded = _loaded(trace, successes)
    raw = loaded[0].envs[0].unwrapped
    monkeypatch.setattr(envs, "_load", lambda name: loaded)

    def obs_hook(observation: dict) -> dict:
        trace.append("obs_hook")
        image = observation["observation.image"]
        assert image.shape == (1, 3, 8, 8) and image.dtype == torch.float32
        assert 0 <= image.min() <= image.max() <= 1
        assert observation["observation.state"].shape == (1, 2)
        assert "task" not in observation
        return observation

    def action_hook(action: np.ndarray, environment: gym.Env) -> np.ndarray:
        trace.append("action_hook")
        assert action.shape == (1, 2) and environment is raw
        return action + 0.25

    rows = envs.run_plain_episodes(
        "pusht", 2, [11, 23], obs_hook=obs_hook, action_hook=action_hook
    )
    assert rows == [
        {"seed": 11, "success": True, "steps": 3},
        {"seed": 23, "success": True, "steps": 3},
    ]
    assert raw.seeds[0] == 11 and 23 in raw.seeds and raw.closed
    assert trace[:2] == ["policy_reset", "reset"]
    expected = ["obs_hook", "select_action", "action_hook", "step"]
    assert trace[2:6] == expected
    assert trace.count("policy_reset") == 2
    np.testing.assert_equal(raw.actions, np.full((6, 2), 0.25))


@pytest.mark.parametrize(
    "args, message",
    [
        (("other", 1, [0]), "env_name"),
        (("pusht", 0, []), "n must be positive"),
        (("pusht", 2, [0]), "number of seeds"),
        (("aloha", 1, [-1]), "nonnegative integers"),
        (("aloha", 1, [True]), "nonnegative integers"),
    ],
)
def test_plain_episodes_reject_invalid_input_before_loading(args: tuple, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        envs.run_plain_episodes(*args)


def test_plain_episodes_close_environment_after_hook_error(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = _loaded([], (False, False, False))
    monkeypatch.setattr(envs, "_load", lambda name: loaded)

    def fail(observation: dict) -> dict:
        raise ValueError("invalid observation hook")

    with pytest.raises(ValueError, match="invalid observation hook"):
        envs.run_plain_episodes("pusht", 1, [0], obs_hook=fail)
    assert loaded[0].envs[0].unwrapped.closed


@pytest.mark.parametrize("env_name", ["pusht", "aloha"])
def test_loader_pins_config_and_weights_to_same_cpu_checkpoint(
    env_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import factory as env_factory
    from lerobot.policies import factory as policy_factory

    pin = {"repo_id": envs._POLICIES[env_name], "revision": "a" * 40}
    if env_name == "pusht":
        pin["num_inference_steps"] = 10
    (tmp_path / "pins.json").write_text(json.dumps({
        "lerobot_version": "0.3.3", "policies": {env_name: pin},
    }))
    monkeypatch.setattr(envs, "__file__", str(tmp_path / "envs.py"))
    monkeypatch.setattr(envs, "version", lambda name: "0.3.3")
    loaded = _loaded([], (False, False, False))
    config = SimpleNamespace(pretrained_path=None)
    calls: dict[str, Any] = {}

    def download(repo: str, *, revision: str, allow_patterns: list[str]) -> str:
        calls["download"] = (repo, revision, allow_patterns)
        return str(tmp_path / "snapshot")

    def config_from_pretrained(path: str, *, cli_overrides: list[str]) -> Any:
        calls["config"] = (path, cli_overrides)
        return config

    def make_env(env_cfg: Any, *, n_envs: int, use_async_envs: bool) -> Any:
        calls["env"] = (env_cfg, n_envs, use_async_envs)
        return loaded[0]

    def make_policy(*, cfg: Any, env_cfg: Any) -> Any:
        assert cfg is config and cfg.pretrained_path == tmp_path / "snapshot"
        assert env_cfg is calls["env"][0]
        if env_name == "pusht":
            assert cfg.num_inference_steps == 10
        else:
            assert not hasattr(cfg, "num_inference_steps")
        return loaded[1]

    def load_model(policy: Any, filename: str, *, strict: bool, device: str) -> None:
        assert policy is loaded[1]
        calls["weights"] = (filename, strict, device)

    monkeypatch.setattr(envs, "snapshot_download", download)
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", config_from_pretrained)
    monkeypatch.setattr(env_factory, "make_env", make_env)
    monkeypatch.setattr(policy_factory, "make_policy", make_policy)
    monkeypatch.setattr(envs, "load_model", load_model)
    result = envs._load(env_name)
    assert result == loaded and not result[1].training
    assert calls["download"] == (pin["repo_id"], pin["revision"], ["*.json", "*.safetensors"])
    assert calls["config"] == (str(tmp_path / "snapshot"), ["--device=cpu"])
    assert calls["weights"] == (str(tmp_path / "snapshot/model.safetensors"), True, "cpu")
    env_cfg, n_envs, use_async_envs = calls["env"]
    assert n_envs == 1 and use_async_envs is False
    assert env_cfg.task == ("PushT-v0" if env_name == "pusht" else "AlohaTransferCube-v0")
    assert env_cfg.episode_length == (300 if env_name == "pusht" else 400)
    result[0].close()


@pytest.mark.parametrize("env_name, steps", [
    ("pusht", value) for value in (True, False, 0, -1, 1.5, "10", None, [], {})
] + [("aloha", 10), ("aloha", None)])
def test_loader_rejects_invalid_inference_override_before_download(
    env_name: str, steps: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pins.json").write_text(json.dumps({
        "lerobot_version": "0.3.3", "policies": {env_name: {
            "repo_id": envs._POLICIES[env_name], "revision": "a" * 40,
            "num_inference_steps": steps,
        }},
    }))
    monkeypatch.setattr(envs, "__file__", str(tmp_path / "envs.py"))
    monkeypatch.setattr(envs, "version", lambda name: "0.3.3")

    def download(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid inference override must be rejected before downloading weights")

    monkeypatch.setattr(envs, "snapshot_download", download)
    with pytest.raises(ValueError, match="num_inference_steps must be a positive integer for pusht only"):
        envs._load(env_name)


@pytest.mark.parametrize("env_name", ["pusht", "aloha"])
def test_real_policy_smoke(env_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    original_load = envs._load
    counts = {"observations": 0, "actions": 0}

    def capped_load(name: str) -> tuple:
        loaded = original_load(name)
        assert isinstance(loaded[0].envs[0], gym.wrappers.TimeLimit)
        loaded[0].envs[0]._max_episode_steps = 20
        return loaded

    def obs_hook(observation: dict) -> dict:
        counts["observations"] += 1
        image_key = "observation.image" if env_name == "pusht" else "observation.images.top"
        image = observation[image_key]
        assert image.dtype == torch.float32 and 0 <= image.min() <= image.max() <= 1
        assert "observation.state" in observation
        return observation

    def action_hook(action: np.ndarray, raw_env: gym.Env) -> np.ndarray:
        counts["actions"] += 1
        if env_name == "pusht":
            assert raw_env.success_threshold == 0.95
            assert hasattr(raw_env.block, "position") and hasattr(raw_env.block, "angle")
        else:
            assert raw_env._env.physics.model.nq > 0
        return action

    monkeypatch.setattr(envs, "_load", capped_load)
    rows = envs.run_plain_episodes(env_name, 1, [0], obs_hook=obs_hook, action_hook=action_hook)
    assert len(rows) == 1 and type(rows[0]["success"]) is bool
    assert 0 < rows[0]["steps"] <= 20
    assert counts["observations"] == counts["actions"] == rows[0]["steps"]
