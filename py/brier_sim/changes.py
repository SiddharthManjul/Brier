from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Any

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy


def _validate_number(value: Any, name: str, *, integer: bool = False) -> None:
    if type(value) not in (int, float) or (integer and type(value) is not int):
        raise ValueError(f"{name} must be {'an integer' if integer else 'a number'}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite")


def _validate_change(
    policy: PreTrainedPolicy, change: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(change, dict) or set(change) != {"description", "families"}:
        raise ValueError("change must contain exactly description and families")
    description, families = change["description"], change["families"]
    if (
        not isinstance(description, str)
        or not description.strip()
        or description.splitlines() != [description]
    ):
        raise ValueError("description must be a nonempty single line")
    if not isinstance(families, list) or not 1 <= len(families) <= 2:
        raise ValueError("families must contain one or two entries")
    validated: list[tuple[str, dict[str, Any]]] = []
    for entry in families:
        if not isinstance(entry, dict) or set(entry) != {"family", "params"}:
            raise ValueError("each family entry must contain exactly family and params")
        family, params = entry["family"], entry["params"]
        if not isinstance(family, str) or not isinstance(params, dict):
            raise ValueError("invalid family name or parameter object")  # noqa: TRY004
        if family in [name for name, _ in validated]:
            raise ValueError("families must be distinct")
        match family:
            case "obs_noise" | "weight_noise":
                expected = {"sigma"}
            case "action_latency":
                expected = {"k"}
            case "action_scale":
                expected = {"scale"}
            case "action_bias":
                expected = {"drift"}
            case "camera_shift":
                expected = {"dx", "dy", "angle"}
            case "occlusion":
                expected = {"x", "y", "width", "height"}
            case "brightness":
                expected = {"gain", "contrast"}
            case "frame_drop":
                expected = {"p"}
            case "chunk_len":
                expected = {"n_action_steps"}
            case "benign":
                expected = set()
                if len(families) != 1:
                    raise ValueError("benign must be the only family")
            case _:
                raise ValueError(f"Unknown change family: {family}")
        if set(params) != expected:
            raise ValueError(f"{family} params must contain exactly {sorted(expected)}")
        for name, value in params.items():
            _validate_number(
                value,
                name,
                integer=name
                in {"k", "dx", "dy", "x", "y", "width", "height", "n_action_steps"},
            )
            if name in {"sigma", "gain", "contrast", "x", "y"} and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            if (
                name in {"k", "scale", "width", "height", "n_action_steps"}
                and value <= 0
            ):
                raise ValueError(f"{name} must be positive")
            if name == "p" and not 0 <= value <= 1:
                raise ValueError("p must be between 0 and 1")
        if family == "chunk_len":
            config, n = policy.config, params["n_action_steps"]
            if config.type == "act":
                limit = config.chunk_size
                if (
                    getattr(config, "temporal_ensemble_coeff", None) is not None
                    and n != 1
                ):
                    raise ValueError(
                        "chunk_len: ACT temporal ensembling requires n_action_steps=1"
                    )
            elif config.type == "diffusion":
                limit = config.horizon - config.n_obs_steps + 1
            else:
                raise ValueError("chunk_len requires an ACT or Diffusion policy config")
            if n > limit:
                raise ValueError(f"chunk_len: n_action_steps must be at most {limit}")
        validated.append((family, params.copy()))
    return validated


def _noise(value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(value.shape, dtype=value.dtype, generator=generator).to(
        value.device
    )


class _ChangedPolicy:
    def __init__(
        self,
        policy: PreTrainedPolicy,
        families: list[tuple[str, dict[str, Any]]],
        seed: int,
    ) -> None:
        self.policy = policy
        self._families = families
        self._rng = torch.Generator(device="cpu")
        self._frame_rng = torch.Generator(device="cpu")
        self._previous: dict[str, torch.Tensor] = {}
        self._delay: deque[torch.Tensor] = deque()
        self._step = 0
        weights_rng = torch.Generator(device="cpu").manual_seed(seed)
        for family, params in families:
            if family == "chunk_len":
                policy.config.n_action_steps = params["n_action_steps"]
            elif family == "weight_noise" and params["sigma"]:
                with torch.no_grad():
                    for parameter in policy.parameters():
                        if parameter.requires_grad and parameter.is_floating_point():
                            parameter.add_(
                                _noise(parameter, weights_rng) * params["sigma"]
                            )
        self.reset()

    def reset(self) -> None:
        self._rng.manual_seed(torch.initial_seed())
        self._frame_rng.manual_seed(torch.initial_seed() ^ 0x5DEECE66D)
        self._previous.clear()
        self._delay.clear()
        self._step = 0
        self.policy.reset()

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        observation = {key: value.clone() for key, value in batch.items()}
        for family, params in self._families:
            if family == "frame_drop":
                images = {
                    key: value
                    for key, value in observation.items()
                    if key.startswith("observation.image")
                }
                drop = bool(
                    self._previous
                    and params["p"] > 0
                    and torch.rand((), generator=self._frame_rng).item() < params["p"]
                )
                if drop:
                    observation.update(
                        {key: value.clone() for key, value in self._previous.items()}
                    )
                else:
                    self._previous = {
                        key: value.clone() for key, value in images.items()
                    }
            for key, value in observation.items():
                image = key.startswith("observation.image")
                if (
                    family == "obs_noise"
                    and params["sigma"]
                    and value.is_floating_point()
                ):
                    if image or key in {
                        "observation.state",
                        "observation.environment_state",
                    }:
                        value = value + _noise(value, self._rng) * params["sigma"]
                        observation[key] = value.clamp(0, 1) if image else value
                elif image and family == "camera_shift":
                    observation[key] = TF.affine(
                        value,
                        angle=params["angle"],
                        translate=[params["dx"], params["dy"]],
                        scale=1.0,
                        shear=[0.0, 0.0],
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    )
                elif image and family == "occlusion":
                    x, y = params["x"], params["y"]
                    width, height = params["width"], params["height"]
                    if x + width > value.shape[-1] or y + height > value.shape[-2]:
                        raise ValueError(
                            "occlusion rectangle is outside the camera image"
                        )
                    value[..., y : y + height, x : x + width] = 0
                elif image and family == "brightness":
                    observation[key] = (
                        ((value - 0.5) * params["contrast"] + 0.5) * params["gain"]
                    ).clamp(0, 1)
        action = self.policy.select_action(observation)
        for family, params in self._families:
            if family == "action_latency":
                if not self._delay:
                    self._delay.extend(action.clone() for _ in range(params["k"]))
                self._delay.append(action.clone())
                action = self._delay.popleft()
            elif family == "action_scale":
                action = action * params["scale"]
            elif family == "action_bias":
                action = action + params["drift"] * (self._step + 1)
        self._step += 1
        return action


def apply_change(
    policy: PreTrainedPolicy, change: dict[str, Any], *, seed: int
) -> _ChangedPolicy:
    """Own a freshly loaded policy and apply its validated, ordered change card.

    Chunk configuration and trainable weights are mutated in place, so callers must
    not reuse the base policy for another change. Weight noise uses ``seed`` once;
    episode noise and histories reset from ``torch.initial_seed()`` on ``reset()``.
    """
    if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
        raise ValueError("seed must be an integer between 0 and 2**64 - 1")
    return _ChangedPolicy(policy, _validate_change(policy, change), seed)
