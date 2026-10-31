import json
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from brier_sim import changes, envs
from brier_sim.changes import action_hook, apply_change, generate_changes, obs_hook


class _Policy(torch.nn.Module):
    def __init__(self, **config: Any) -> None:
        super().__init__()
        self.config = SimpleNamespace(**{"type": "act", "chunk_size": 50,
                                        "n_action_steps": 2, "temporal_ensemble_coeff": None, **config})
        self.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32))
        self.frozen = torch.nn.Parameter(torch.arange(3, dtype=torch.float32), requires_grad=False)
        self.scalar = torch.nn.Parameter(torch.tensor(4.))
        self.register_buffer("running_mean", torch.ones(3))
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def _card(*families: tuple[str, dict[str, Any]], env: str = "pusht", change_id: str = "c005",
          seed: int = 2026) -> dict[str, Any]:
    return {"change_id": change_id, "env": env, "families": [{"family": family, "params": params}
            for family, params in families], "desc": "Update controller calibration", "seed": seed}


def _obs(value: float = .3) -> dict[str, Any]:
    return {"observation.state": torch.tensor([[11., 22.]]),
            "observation.environment_state": torch.tensor([[-3., 4.]]),
            "observation.image": torch.full((1, 3, 10, 10), value, dtype=torch.float32), "task": ["transfer"]}


def _raw(position: list[float], env: str = "pusht") -> SimpleNamespace:
    return (SimpleNamespace(agent=SimpleNamespace(position=position)) if env == "pusht" else
            SimpleNamespace(_env=SimpleNamespace(task=SimpleNamespace(get_qpos=lambda physics: position), physics=object())))


def test_observation_noise_changes_images_only_and_does_not_mutate_input() -> None:
    original = _obs(.5)
    state = apply_change(_Policy(), _card(("obs_noise", {"sigma": .1})))
    result = obs_hook(state, original, np.random.default_rng(12))
    noise = result["observation.image"] - original["observation.image"]
    assert noise.std().item() == pytest.approx(.1, abs=.02)
    assert not torch.equal(result["observation.image"], original["observation.image"])
    for key in ("observation.state", "observation.environment_state"):
        assert torch.equal(result[key], original[key])
    assert result["task"] == ["transfer"] and original["observation.image"].min() == .5
    assert 0 <= result["observation.image"].min() <= result["observation.image"].max() <= 1
    repeated = obs_hook(state, original, np.random.default_rng(12))
    assert torch.equal(result["observation.image"], repeated["observation.image"])


def test_latency_holds_current_measured_position_then_delays_k_steps_and_resets() -> None:
    state = apply_change(_Policy(), _card(("action_latency", {"k": 2})))
    rng = np.random.default_rng(1)
    outputs = [action_hook(state, np.array([[i, i + 1.]]), _raw([10 + i, 20 + i]), rng)
               for i in range(5)]
    np.testing.assert_array_equal(outputs, [[[10, 20]], [[11, 21]], [[0, 1]], [[1, 2]], [[2, 3]]])
    state.reset()
    np.testing.assert_array_equal(action_hook(state, np.array([[9., 9.]]), _raw([7., 8.]), rng), [[7., 8.]])


@pytest.mark.parametrize("env,position", [("pusht", [11., 22.]), ("aloha", list(range(14)))])
def test_scale_acts_on_displacement_from_fresh_native_position(env: str, position: list[float]) -> None:
    state = apply_change(_Policy(), _card(("action_scale", {"s": .8}), env=env))
    action = np.array([position]) + 10
    original = action.copy()
    actual = action_hook(state, action, _raw(position, env), np.random.default_rng(1))
    np.testing.assert_allclose(actual, np.array([position]) + 8)
    np.testing.assert_array_equal(action, original)


@pytest.mark.parametrize("env,dimensions", [("pusht", 2), ("aloha", 14)])
def test_bias_fixed_unit_direction_per_change_t_zero_and_grippers_excluded(env: str, dimensions: int) -> None:
    card = _card(("action_bias", {"b": 4}), env=env)
    state, rng = apply_change(_Policy(), card), np.random.default_rng(12)
    action = np.ones((1, dimensions), dtype=np.float32)
    np.testing.assert_array_equal(action_hook(state, action, None, rng), action)
    second = action_hook(state, action, None, rng)
    assert np.linalg.norm(second - action) == pytest.approx(.04)
    third = action_hook(state, action, None, rng)
    np.testing.assert_allclose(third - action, 2 * (second - action), atol=1e-12)
    if env == "aloha":
        np.testing.assert_array_equal(second[:, [6, 13]], action[:, [6, 13]])
    state.reset()
    action_hook(state, action, None, rng)
    np.testing.assert_array_equal(action_hook(state, action, None, rng), second)
    repeated = apply_change(_Policy(), card)
    action_hook(repeated, action, None, rng)
    np.testing.assert_array_equal(action_hook(repeated, action, None, rng), second)


def test_camera_shift_is_fixed_affine_and_preserves_nonimage_data() -> None:
    state = apply_change(_Policy(), _card(("camera_shift", {"shift": 2, "angle": 0})))
    original = _obs(0)
    original["observation.image"][..., 5, 5] = 1
    transformed = obs_hook(state, original, np.random.default_rng(1))
    dx, dy = state.translation * 2
    expected = torch.zeros_like(original["observation.image"])
    expected[..., 5 + dy, 5 + dx] = 1
    torch.testing.assert_close(transformed["observation.image"], expected, atol=1e-6, rtol=0)
    assert torch.equal(transformed["observation.state"], original["observation.state"])
    assert original["observation.image"].sum() == 3
    state.reset()
    assert torch.equal(obs_hook(state, original, np.random.default_rng(42))["observation.image"], transformed["observation.image"])


def test_occlusion_is_midgray_relative_square_at_fixed_location() -> None:
    state = apply_change(_Policy(), _card(("occlusion", {"fraction": .2})))
    original = _obs(0)
    transformed = obs_hook(state, original, np.random.default_rng(1))
    image = transformed["observation.image"]
    assert torch.count_nonzero(image).item() == 3 * 2 * 2
    assert set(image.unique().tolist()) == {0., .5}
    assert torch.equal(transformed["observation.state"], original["observation.state"])
    assert original["observation.image"].sum() == 0
    state.reset()
    assert torch.equal(obs_hook(state, original, np.random.default_rng(42))["observation.image"], image)


def test_brightness_clips_contrast_before_gain() -> None:
    state = apply_change(_Policy(), _card(("brightness", {"gain": .6, "contrast": 1.2})))
    original = _obs(1.)
    transformed = obs_hook(state, original, np.random.default_rng(1))
    torch.testing.assert_close(transformed["observation.image"], torch.full((1, 3, 10, 10), .6))
    assert torch.equal(transformed["observation.state"], original["observation.state"])
    assert original["observation.image"].min() == 1


def test_frame_drop_holds_complete_previous_observation_and_resets() -> None:
    state = apply_change(_Policy(), _card(("frame_drop", {"p": 1})))
    rng = np.random.default_rng(1)
    first = obs_hook(state, _obs(.2), rng)
    current = _obs(.9)
    current["observation.state"] += 100
    current["task"] = ["new task"]
    second = obs_hook(state, current, rng)
    assert second["task"] == first["task"]
    for key in ("observation.state", "observation.image", "observation.environment_state"):
        assert torch.equal(first[key], second[key])
        assert current["observation.image"].min() == .9
    state.reset()
    assert obs_hook(state, current, rng)["task"] == ["new task"]


def test_scale_and_latency_use_current_native_state_despite_frame_drop() -> None:
    state = apply_change(_Policy(), _card(("frame_drop", {"p": 1}), ("action_scale", {"s": .8})))
    rng = np.random.default_rng(1)
    obs_hook(state, _obs(), rng)
    obs_hook(state, _obs(), rng)
    result = action_hook(state, np.array([[30., 40.]]), _raw([20., 30.]), rng)
    np.testing.assert_allclose(result, [[28., 38.]])


def test_chunk_length_checks_native_policy_limits_before_mutation() -> None:
    policy = _Policy(type="diffusion", horizon=16, n_obs_steps=2)
    state = apply_change(policy, _card(("chunk_len", {"n_action_steps": 12})))
    assert policy.config.n_action_steps == 12 and policy.resets == 1
    state.reset()
    assert policy.config.n_action_steps == 12 and policy.resets == 2
    with pytest.raises(ValueError, match="at most 15"):
        apply_change(policy, _card(("chunk_len", {"n_action_steps": 16})))
    assert policy.config.n_action_steps == 12 and policy.resets == 2


def test_weight_noise_uses_change_id_relative_std_and_all_float_parameters_once() -> None:
    policy = _Policy()
    originals = deepcopy(policy.state_dict())
    state = apply_change(policy, _card(("weight_noise", {"sigma": .03})))
    generator = torch.Generator().manual_seed(5)
    for name, parameter in policy.named_parameters():
        old = originals[name]
        noise = torch.randn(old.shape, dtype=old.dtype, generator=generator)
        std = old.std() if old.numel() > 1 else 0
        assert torch.equal(parameter, old + noise * .03 * std)
    assert not torch.equal(policy.frozen, originals["frozen"])
    assert torch.equal(policy.running_mean, originals["running_mean"])
    perturbed = deepcopy(policy.state_dict())
    state.reset()
    assert all(torch.equal(value, perturbed[name]) for name, value in policy.state_dict().items())
    repeat, different = _Policy(), _Policy()
    apply_change(repeat, _card(("weight_noise", {"sigma": .03}), seed=999))
    apply_change(different, _card(("weight_noise", {"sigma": .03}), change_id="c006"))
    assert torch.equal(repeat.weight, policy.weight)
    assert not torch.equal(different.weight, policy.weight)


@pytest.mark.parametrize("env,first,last", [("pusht", "c001", "c080"), ("aloha", "c081", "c160")])
def test_generator_counts_coverage_titles_and_frozen_seed(env: str, first: str, last: str) -> None:
    cards = generate_changes(env)
    assert cards == generate_changes(env) and cards != generate_changes(env, seed=2027)
    assert cards[0]["change_id"] == first and cards[-1]["change_id"] == last
    assert len(cards) == 80 and len({c["change_id"] for c in cards}) == 80
    counts = Counter(entry["family"] for card in cards for entry in card["families"])
    assert counts.pop("benign") == 28 and len(counts) == 10 and min(counts.values()) >= 4
    assert sum(len(card["families"]) == 2 for card in cards) == 16
    assert sum(len(card["families"]) == 1 and card["families"][0]["family"] != "benign" for card in cards) == 36
    json.dumps(cards)
    for card in cards:
        assert set(card) == {"change_id", "env", "families", "desc", "seed"}
        assert card["env"] == env and type(card["seed"]) is int
        family = card["families"][0]["family"]
        assert len(changes._TITLES[family].split("|")) >= 4
        assert card["desc"] in changes._TITLES[family].split("|") and "\n" not in card["desc"]
        assert len({entry["family"] for entry in card["families"]}) == len(card["families"])
        if env == "pusht":
            assert all(entry["params"].get("n_action_steps", 1) <= 15 for entry in card["families"])


def _pilot(env: str = "pusht") -> dict[str, list[dict[str, Any]]]:
    return {family: [{"params": params[0], "mean_drop": (index + 1) / 20}]
            for index, (family, params) in enumerate(changes._grid(env).items())}


def test_measured_pilot_restricts_params_ranks_composites_and_drops_families() -> None:
    pilot = _pilot()
    del pilot["frame_drop"], pilot["brightness"]
    original = deepcopy(pilot)
    cards = generate_changes("pusht", n=40, n_benign=14, magnitudes=pilot)
    assert pilot == original
    for card in cards:
        families = card["families"]
        if families[0]["family"] == "benign":
            continue
        effects = [pilot[entry["family"]][0]["mean_drop"] for entry in families]
        assert effects == sorted(effects, reverse=True)
        assert all(entry["params"] == pilot[entry["family"]][0]["params"] for entry in families)


@pytest.mark.parametrize("pilot", [
    {}, {"unknown": [{"params": {}, "mean_drop": .2}]}, {"obs_noise": []},
    {"action_latency": [{"params": {"k": True}, "mean_drop": .2}]},
    {"action_latency": [{"params": {"k": 1.0}, "mean_drop": .2}]},
    {"camera_shift": [{"params": {"shift": 2.0, "angle": 0}, "mean_drop": .2}]},
    {"chunk_len": [{"params": {"n_action_steps": 4.0}, "mean_drop": .2}]},
    {"action_latency": [{"params": {"k": 1}, "mean_drop": True}]},
    {"action_latency": [{"params": {"k": 1}, "mean_drop": float("nan")}]},
    {"action_latency": [{"params": {"k": 1}, "mean_drop": 10**1000}]},
    {"chunk_len": [{"params": {"n_action_steps": 16}, "mean_drop": .2}]},
    {"obs_noise": [{"params": {"sigma": .02}, "mean_drop": .2}]},
    {"obs_noise": [{"params": {"sigma": .01}, "mean_drop": .61}]},
])
def test_generator_rejects_invalid_pilot_tables(pilot: Any) -> None:
    with pytest.raises(ValueError):
        generate_changes("pusht", magnitudes=pilot)


@pytest.mark.parametrize("kwargs", [{"n": 40, "n_benign": 14}, {"n": True}, {"n": 81},
                                     {"n_benign": -1}, {"seed": -1}, {"seed": True}])
def test_generator_rejects_infeasible_or_invalid_counts(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        generate_changes("pusht", **kwargs)


@pytest.mark.parametrize("family,params", [("obs_noise", {"sigma": 10**1000}),
    ("action_scale", {"s": 0}), ("action_bias", {"b": -1}), ("camera_shift", {"shift": True, "angle": 0}),
    ("occlusion", {"fraction": 1.1}), ("brightness", {"gain": -1, "contrast": 1}),
    ("frame_drop", {"p": float("nan")}), ("chunk_len", {"n_action_steps": 51}),
    ("weight_noise", {"sigma": -1}), ("benign", {"unused": 0})])
def test_invalid_guide_parameters_reject_before_mutation(family: str, params: dict[str, Any]) -> None:
    policy = _Policy()
    original = policy.weight.clone()
    with pytest.raises(ValueError):
        apply_change(policy, _card(("weight_noise", {"sigma": .01}), (family, params)))
    assert torch.equal(policy.weight, original) and policy.resets == 0


def test_image_hook_rejects_missing_images_and_new_cards_require_explicit_hooks() -> None:
    state = apply_change(_Policy(), _card(("obs_noise", {"sigma": .03})))
    with pytest.raises(ValueError, match="observation.image"):
        obs_hook(state, {"observation.state": torch.zeros(1, 2)}, np.random.default_rng(1))
    with pytest.raises(ValueError, match="require obs_hook"):
        state.select_action(_obs())


def test_cli_writes_only_measured_pilot_ranked_manifests_and_refuses_overwrite(tmp_path: Path) -> None:
    pilot = tmp_path / "magnitudes_pusht.json"
    pilot.write_text(json.dumps(_pilot()))
    command = [sys.executable, "-m", "brier_sim.changes", "--env", "pusht", "--magnitudes", str(pilot), "--write"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == {"changes": 80, "benign": 28, "two_family": 16}
    output = tmp_path / "changes_pusht.jsonl"
    written = output.read_text()
    assert len(written.splitlines()) == 80
    repeated = subprocess.run(command, text=True, capture_output=True, check=False)
    assert repeated.returncode != 0 and "refusing to overwrite frozen manifest" in repeated.stderr
    assert output.read_text() == written
    missing = subprocess.run([sys.executable, "-m", "brier_sim.changes", "--env", "pusht", "--write"], text=True, capture_output=True, check=False)
    assert missing.returncode != 0 and "requires measured --magnitudes" in missing.stderr


def test_benign_is_bit_identical_for_real_pinned_policy_twenty_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub import snapshot_download
    from lerobot.utils.random_utils import set_seed

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setattr(envs, "snapshot_download", lambda *args, **kwargs: snapshot_download(*args, local_files_only=True, **kwargs))
    torch.set_num_threads(1)
    loaded = envs._load("pusht")
    loaded[0].envs[0]._max_episode_steps = 20
    original_actions, changed_actions = [], []
    try:
        set_seed(19)
        envs._run_episode(loaded, 19, None, lambda action, raw: original_actions.append(action.copy()) or action)
        state = apply_change(loaded[1], _card(("benign", {})))
        rng = np.random.default_rng(19)

        def capture(action: np.ndarray, raw: Any) -> np.ndarray:
            result = action_hook(state, action, raw, rng)
            changed_actions.append(result.copy())
            return result

        set_seed(19)
        envs._run_episode(loaded, 19, lambda obs: obs_hook(state, obs, rng), capture)
        assert len(original_actions) == len(changed_actions) == 20
        np.testing.assert_array_equal(original_actions, changed_actions)
    finally:
        loaded[0].close()
