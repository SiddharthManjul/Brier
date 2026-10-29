from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from brier_sim.changes import apply_change


class _Policy(torch.nn.Module):
    def __init__(self, **config: Any) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            **{
                "type": "act",
                "chunk_size": 4,
                "n_action_steps": 2,
                "temporal_ensemble_coeff": None,
                **config,
            }
        )
        if config.get("type") == "diffusion":
            del self.config.chunk_size
        self.weight = torch.nn.Parameter(torch.ones(12))
        self.frozen = torch.nn.Parameter(torch.ones(3), requires_grad=False)
        self.register_buffer("running_mean", torch.ones(3))
        self.observations: list[dict[str, torch.Tensor]] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def select_action(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        self.observations.append(observation)
        return observation["observation.state"]


def _card(*families: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "description": "Update controller calibration",
        "families": [
            {"family": family, "params": params} for family, params in families
        ],
    }


def _observation(value: float = 0.5) -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.tensor([[value, value + 1]]),
        "observation.image": torch.full((1, 3, 5, 5), value),
        "observation.environment_state": torch.tensor([[-2.0, 3.0]]),
        "task_index": torch.tensor([1]),
    }


def test_benign_preserves_all_observations_actions_and_random_stream() -> None:
    policy, observation = _Policy(), _observation()
    originals = deepcopy(observation)
    torch.manual_seed(31)
    expected_random = torch.rand(4)
    torch.manual_seed(31)
    changed = apply_change(policy, _card(("benign", {})), seed=9)
    output = changed.select_action(observation)
    torch.testing.assert_close(output, originals["observation.state"], rtol=0, atol=0)
    for key, value in originals.items():
        assert torch.equal(observation[key], value)
        assert torch.equal(policy.observations[-1][key], value)
    assert torch.equal(torch.rand(4), expected_random)


def test_observation_noise_is_gaussian_unclipped_for_state_and_clipped_for_images() -> (
    None
):
    policy = _Policy()
    observation = {
        "observation.state": torch.zeros(1, 10000),
        "observation.environment_state": torch.full((1, 10000), -2.0),
        "observation.image": torch.full((1, 3, 64, 64), 0.5),
        "task_index": torch.tensor([1]),
    }
    originals = deepcopy(observation)
    changed = apply_change(policy, _card(("obs_noise", {"sigma": 0.3})), seed=1)
    changed.select_action(observation)
    seen = policy.observations[-1]
    for key in ("observation.state", "observation.environment_state"):
        noise = seen[key] - observation[key]
        assert abs(noise.mean().item()) < 0.02
        assert noise.std().item() == pytest.approx(0.3, abs=0.02)
    assert seen["observation.state"].min() < 0
    assert 0 <= seen["observation.image"].min() <= seen["observation.image"].max() <= 1
    assert not torch.equal(seen["observation.image"], observation["observation.image"])
    for key, value in observation.items():
        assert torch.equal(value, originals[key])
    assert torch.equal(seen["task_index"], observation["task_index"])


@pytest.mark.parametrize(
    "family,params,expected",
    [
        ("action_scale", {"scale": 1.5}, [[0.75, 2.25]]),
        ("action_bias", {"drift": 0.25}, [[0.75, 1.75]]),
    ],
)
def test_action_changes_preserve_base_actions(
    family: str, params: dict[str, float], expected: list[list[float]]
) -> None:
    observation = _observation()
    changed = apply_change(_Policy(), _card((family, params)), seed=0)
    assert torch.equal(changed.select_action(observation), torch.tensor(expected))
    assert torch.equal(observation["observation.state"], torch.tensor([[0.5, 1.5]]))
    if family == "action_bias":
        assert torch.equal(
            changed.select_action(observation), torch.tensor([[1.0, 2.0]])
        )
        changed.reset()
        assert torch.equal(changed.select_action(observation), torch.tensor(expected))


def test_latency_delays_by_k_steps_with_first_action_warmup_and_reset() -> None:
    changed = apply_change(_Policy(), _card(("action_latency", {"k": 2})), seed=0)
    results = [
        changed.select_action(_observation(float(i)))[0, 0].item() for i in range(1, 6)
    ]
    assert results == [1, 1, 1, 2, 3]
    changed.reset()
    assert changed.select_action(_observation(9))[0, 0].item() == 9


@pytest.mark.parametrize(
    "params,target",
    [
        ({"dx": 1, "dy": 1, "angle": 0}, (2, 4)),
        ({"dx": 0, "dy": 0, "angle": 180}, (3, 1)),
    ],
)
def test_camera_shift_moves_landmark_and_fills_black(
    params: dict[str, int], target: tuple[int, int]
) -> None:
    policy, observation = _Policy(), _observation()
    observation["observation.image"].zero_()
    observation["observation.image"][:, :, 1, 3] = 1
    changed = apply_change(policy, _card(("camera_shift", params)), seed=0)
    changed.select_action(observation)
    expected = torch.zeros_like(observation["observation.image"])
    expected[:, :, target[0], target[1]] = 1
    torch.testing.assert_close(
        policy.observations[-1]["observation.image"], expected, atol=1e-6, rtol=0
    )
    assert observation["observation.image"].sum() == 3
    assert torch.equal(
        policy.observations[-1]["observation.state"], observation["observation.state"]
    )


def test_occlusion_blacks_only_requested_rectangle() -> None:
    policy, observation = _Policy(), _observation()
    changed = apply_change(
        policy, _card(("occlusion", {"x": 1, "y": 2, "width": 2, "height": 1})), seed=0
    )
    changed.select_action(observation)
    expected = observation["observation.image"].clone()
    expected[:, :, 2:3, 1:3] = 0
    assert torch.equal(policy.observations[-1]["observation.image"], expected)
    assert observation["observation.image"].min() == 0.5
    invalid = apply_change(
        _Policy(),
        _card(("occlusion", {"x": 4, "y": 0, "width": 2, "height": 1})),
        seed=0,
    )
    with pytest.raises(ValueError, match="occlusion"):
        invalid.select_action(observation)


def test_brightness_uses_midgray_contrast_then_gain_and_clamps() -> None:
    policy, observation = _Policy(), _observation()
    observation["observation.image"][:] = torch.linspace(0, 1, 5)
    changed = apply_change(
        policy, _card(("brightness", {"gain": 2, "contrast": 0.5})), seed=0
    )
    changed.select_action(observation)
    expected = torch.tensor([0.5, 0.75, 1, 1, 1]).expand(1, 3, 5, 5)
    assert torch.equal(policy.observations[-1]["observation.image"], expected)
    assert torch.equal(
        observation["observation.image"][0, 0, 0], torch.linspace(0, 1, 5)
    )


def test_frame_drop_holds_only_images_keeps_first_frame_and_resets() -> None:
    policy = _Policy()
    changed = apply_change(policy, _card(("frame_drop", {"p": 1})), seed=0)
    changed.select_action(_observation(0.25))
    changed.select_action(_observation(0.75))
    seen = policy.observations[-1]
    assert torch.all(seen["observation.image"] == 0.25)
    assert torch.equal(
        seen["observation.state"], _observation(0.75)["observation.state"]
    )
    changed.reset()
    changed.select_action(_observation(0.9))
    assert torch.all(policy.observations[-1]["observation.image"] == 0.9)


@pytest.mark.parametrize(
    "config,maximum",
    [({}, 4), ({"type": "diffusion", "horizon": 6, "n_obs_steps": 2}, 5)],
)
def test_chunk_length_updates_config_and_rejects_policy_bounds(
    config: dict[str, Any], maximum: int
) -> None:
    policy = _Policy(**config)
    changed = apply_change(
        policy, _card(("chunk_len", {"n_action_steps": maximum})), seed=0
    )
    assert policy.config.n_action_steps == maximum
    changed.reset()
    assert policy.config.n_action_steps == maximum
    with pytest.raises(ValueError, match="n_action_steps"):
        apply_change(
            _Policy(**config),
            _card(("chunk_len", {"n_action_steps": maximum + 1})),
            seed=0,
        )
    if not config:
        with pytest.raises(ValueError, match="n_action_steps"):
            apply_change(
                _Policy(temporal_ensemble_coeff=0.1),
                _card(("chunk_len", {"n_action_steps": 2})),
                seed=0,
            )


def test_weight_noise_is_seeded_once_and_protects_frozen_weights_and_stats() -> None:
    policy = _Policy()
    original = deepcopy(policy.state_dict())
    card = _card(("weight_noise", {"sigma": 0.2}))
    changed = apply_change(policy, card, seed=123)
    assert not torch.equal(policy.weight, original["weight"])
    assert torch.equal(policy.frozen, original["frozen"])
    assert torch.equal(policy.running_mean, original["running_mean"])
    perturbed = policy.weight.detach().clone()
    changed.reset()
    assert torch.equal(policy.weight, perturbed)
    repeated, different = _Policy(), _Policy()
    apply_change(repeated, card, seed=123)
    apply_change(different, card, seed=124)
    assert torch.equal(repeated.weight, perturbed)
    assert not torch.equal(different.weight, perturbed)


def test_composition_uses_authored_order() -> None:
    bias, scale = ("action_bias", {"drift": 0.5}), ("action_scale", {"scale": 2})
    first = apply_change(_Policy(), _card(bias, scale), seed=0)
    second = apply_change(_Policy(), _card(scale, bias), seed=0)
    assert first.select_action(_observation(0))[0, 0].item() == 1
    assert second.select_action(_observation(0))[0, 0].item() == 0.5


def test_episode_rng_is_reproducible_and_does_not_consume_global_rng() -> None:
    policy = _Policy()
    torch.manual_seed(77)
    expected = torch.rand(5)
    torch.manual_seed(77)
    changed = apply_change(
        policy, _card(("obs_noise", {"sigma": 0.1}), ("frame_drop", {"p": 0.5})), seed=7
    )
    first = [changed.select_action(_observation()).clone() for _ in range(4)]
    assert torch.equal(torch.rand(5), expected)
    torch.manual_seed(77)
    changed.reset()
    second = [changed.select_action(_observation()).clone() for _ in range(4)]
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    torch.manual_seed(78)
    changed.reset()
    assert not torch.equal(changed.select_action(_observation()), first[0])


@pytest.mark.parametrize(
    "family,params",
    [
        ("obs_noise", {"sigma": float("nan")}),
        ("action_latency", {"k": 0}),
        ("action_scale", {"scale": 0}),
        ("action_bias", {"drift": float("inf")}),
        ("camera_shift", {"dx": 0.1, "dy": 0, "angle": 0}),
        ("occlusion", {"x": 0, "y": 0, "width": 0, "height": 1}),
        ("brightness", {"gain": -1, "contrast": 1}),
        ("frame_drop", {"p": 1.1}),
        ("chunk_len", {"n_action_steps": 0}),
        ("weight_noise", {"sigma": -1}),
        ("benign", {"sigma": 0}),
        ("obs_noise", {"sigma": 0.1, "unexpected": 2}),
        ("unknown", {}),
    ],
)
def test_invalid_parameters_reject_entire_card_before_mutating_policy(
    family: str, params: dict[str, Any]
) -> None:
    policy = _Policy()
    original = policy.weight.detach().clone()
    with pytest.raises(ValueError):
        apply_change(
            policy, _card(("weight_noise", {"sigma": 0.1}), (family, params)), seed=0
        )
    assert torch.equal(policy.weight, original)
    assert policy.resets == 0


@pytest.mark.parametrize(
    "change",
    [
        {"description": "", "families": [{"family": "benign", "params": {}}]},
        {"description": "two\nlines", "families": [{"family": "benign", "params": {}}]},
        {**_card(("benign", {})), "extra": 1},
        _card(),
        _card(("action_bias", {"drift": 0}), ("action_bias", {"drift": 1})),
        _card(("benign", {}), ("action_scale", {"scale": 1})),
        _card(
            ("action_bias", {"drift": 0}),
            ("action_scale", {"scale": 1}),
            ("obs_noise", {"sigma": 0}),
        ),
        {
            "description": "one line",
            "families": [{"family": "benign", "params": {}, "extra": 1}],
        },
    ],
)
def test_malformed_cards_are_rejected(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        apply_change(_Policy(), change, seed=0)
