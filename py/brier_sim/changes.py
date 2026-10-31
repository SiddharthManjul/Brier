from __future__ import annotations

import argparse
import json
import math
import re
from collections import deque
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

_PARAMS = {
    "obs_noise": "sigma", "action_latency": "k", "action_scale": "s",
    "action_bias": "b", "camera_shift": "shift angle", "occlusion": "fraction",
    "brightness": "gain contrast", "frame_drop": "p", "chunk_len": "n_action_steps",
    "weight_noise": "sigma", "benign": "",
}
_TITLES = {
    "obs_noise": "Switch to new camera driver|Use cheaper USB camera|Update image sensor firmware|Enable camera denoising",
    "action_latency": "Move inference to edge box|Run policy server over the network|Batch controller requests|Change inference transport",
    "action_scale": "Retune gripper gains|Adjust controller response|Update motion gain calibration|Revise position gain settings",
    "action_bias": "Apply calibration offset from new fixture|Update joint zero calibration|Replace calibration fixture|Correct controller reference frame",
    "camera_shift": "Re-mount wrist camera|Adjust camera mounting bracket|Replace camera housing|Update camera alignment",
    "occlusion": "Route cable across camera view|Fit protective camera cover|Relocate camera cable guide|Install camera mounting guard",
    "brightness": "Swap lighting rig|Update camera exposure|Replace overhead lights|Change camera contrast settings",
    "frame_drop": "Raise camera FPS under load|Update image transport queue|Reduce camera buffer allocation|Change video capture scheduling",
    "chunk_len": "Shorten action chunks for responsiveness|Update action execution window|Retune controller chunk length|Revise policy execution cadence",
    "weight_noise": "Checkpoint from overnight run|Update pretrained policy checkpoint|Refresh controller model weights|Promote latest policy checkpoint",
    "benign": "Bump logging, no policy change|Refactor obs pipeline|Rename config keys|Update checkpoint metadata",
}


def _grid(env: str) -> dict[str, list[dict[str, Any]]]:
    values = [[.01, .03, .05, .1], [1, 2, 4], [.8, .9, 1.1, 1.2], [1, 2, 4],
              [2, 4, 6] if env == "pusht" else [8, 16, 32], [.1, .2, .3],
              [.6, .8, 1.25, 1.5], [.1, .2, .4], [4, 12] if env == "pusht" else [25, 50], [.01, .03, .05]]
    # Pinned PushT diffusion: horizon 16 - n_obs_steps 2 + 1 = 15; 16 is unsupported.
    grid = {family: [{_PARAMS[family].split()[0]: value} for value in choices]
            for family, choices in zip(list(_PARAMS)[:-1], values, strict=True)}
    grid["camera_shift"] = [{"shift": shift, "angle": angle} for shift, angle in product(values[4], [0, 3])]
    grid["brightness"] = [{"gain": gain, "contrast": contrast} for gain, contrast in product(values[6], [.8, 1., 1.2])]
    return grid


def generate_changes(env: str, n: int = 80, n_benign: int = 28, seed: int = 2026,
                     magnitudes: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    """Generate deterministic candidate or pilot-ranked cards.

    Pilot entries are {"params": {...}, "mean_drop": measured float}. Omitted
    families are dropped. Default full-grid cards are provisional candidates:
    their composite order is seeded, not a claim about measured expected effect.
    """
    if env not in {"pusht", "aloha"} or type(n) is not int or not 1 <= n <= 80:
        raise ValueError("env must be pusht or aloha and n must be between 1 and 80")
    if type(n_benign) is not int or not 0 <= n_benign <= n or type(seed) is not int or seed < 0:
        raise ValueError("n_benign must be between 0 and n; seed must be a nonnegative integer")
    grid, rng = _grid(env), np.random.default_rng(seed)
    allowed = {family: [{"params": params, "mean_drop": None} for params in choices] for family, choices in grid.items()}
    if magnitudes is not None:
        if not isinstance(magnitudes, dict) or not magnitudes or set(magnitudes) - set(grid):
            raise ValueError("pilot magnitudes must name known non-benign families")
        allowed = deepcopy(magnitudes)
        for family, entries in allowed.items():
            if not isinstance(entries, list) or not entries:
                raise ValueError("each retained pilot family needs a nonempty list")
            for entry in entries:
                if (not isinstance(entry, dict) or set(entry) != {"params", "mean_drop"}
                        or not isinstance(entry["params"], dict)
                        or any(type(value) not in (int, float) for value in entry["params"].values())
                        or any(type(value) is not int for name, value in entry["params"].items() if name in {"k", "shift", "n_action_steps"})
                        or entry["params"] not in grid[family]
                        or type(entry["mean_drop"]) not in (int, float)
                        or not .05 <= entry["mean_drop"] <= .6):
                    raise ValueError(f"invalid pilot params or measured mean_drop for {family}")
    count, doubles = n - n_benign, round(.3 * (n - n_benign))
    if count + doubles < 4 * len(allowed) or (doubles and len(allowed) < 2):
        raise ValueError("change counts cannot provide four appearances per retained family")
    slots = list(np.resize(rng.permutation(list(allowed)), count + doubles))
    cards = []
    for index in range(count):
        chosen = [slots.pop(0) for _ in range(2 if index < doubles else 1)]
        entries = [(family, allowed[family][int(rng.integers(len(allowed[family])))]) for family in chosen]
        if magnitudes is not None:
            entries.sort(key=lambda item: item[1]["mean_drop"], reverse=True)
        cards.append([{"family": family, "params": deepcopy(entry["params"])} for family, entry in entries])
    cards.extend([[{"family": "benign", "params": {}}] for _ in range(n_benign)])
    rng.shuffle(cards)
    return [{"change_id": f"c{index + (1 if env == 'pusht' else 81):03d}", "env": env,
             "families": families, "desc": str(rng.choice(_TITLES[families[0]['family']].split("|"))),
             "seed": int(rng.integers(0, 2**32))} for index, families in enumerate(cards)]


def _validate(policy: Any, change: dict[str, Any], legacy: bool) -> list[tuple[str, dict[str, Any]]]:
    keys = {"description", "families"} if legacy else {"change_id", "env", "families", "desc", "seed"}
    if not isinstance(change, dict) or set(change) != keys:
        raise ValueError(f"change must contain exactly {sorted(keys)}")
    desc = change["description" if legacy else "desc"]
    if not isinstance(desc, str) or not desc.strip() or desc.splitlines() != [desc]:
        raise ValueError("description must be a nonempty single line")
    if not legacy and (change["env"] not in {"pusht", "aloha"}
                       or not isinstance(change["change_id"], str) or not re.fullmatch(r"c[0-9]{3,}", change["change_id"])
                       or int(change["change_id"][1:]) == 0 or type(change["seed"]) is not int or change["seed"] < 0):
        raise ValueError("invalid change id, env or seed")
    families = change["families"]
    if not isinstance(families, list) or not 1 <= len(families) <= 2:
        raise ValueError("families must contain one or two entries")
    validated = []
    names = {**_PARAMS, **({"action_scale": "scale", "action_bias": "drift", "camera_shift": "dx dy angle", "occlusion": "x y width height"} if legacy else {})}
    for entry in families:
        if not isinstance(entry, dict) or set(entry) != {"family", "params"}:
            raise ValueError("each family entry must contain exactly family and params")
        family, params = entry["family"], entry["params"]
        if not isinstance(family, str) or family not in names or not isinstance(params, dict):
            raise ValueError("unknown family or invalid parameter object")
        if family in [name for name, _ in validated] or (family == "benign" and len(families) != 1):
            raise ValueError("families must be distinct and benign must be alone")
        if set(params) != set(names[family].split()):
            raise ValueError(f"invalid params for {family}")
        for name, value in params.items():
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name} must be a finite number")
            if name in {"k", "dx", "dy", "x", "y", "width", "height", "shift", "n_action_steps"} and type(value) is not int:
                raise ValueError(f"{name} must be an integer")
            if ((name in {"sigma", "gain", "contrast", "x", "y", "b"} and value < 0)
                    or (name in {"k", "scale", "s", "width", "height", "shift", "n_action_steps", "fraction"} and value <= 0)
                    or (name in {"p", "fraction"} and not 0 <= value <= 1)):
                raise ValueError(f"{name} is outside its allowed bounds")
        if family == "chunk_len":
            cfg, steps = policy.config, params["n_action_steps"]
            if cfg.type not in {"act", "diffusion"}:
                raise ValueError("chunk_len requires an ACT or Diffusion policy config")
            limit = cfg.chunk_size if cfg.type == "act" else cfg.horizon - cfg.n_obs_steps + 1
            if steps > limit or (cfg.type == "act" and getattr(cfg, "temporal_ensemble_coeff", None) is not None and steps != 1):
                raise ValueError(f"chunk_len: n_action_steps must be at most {limit}; temporal ensembling requires 1")
        validated.append((family, params.copy()))
    return validated


def _noise(value: torch.Tensor, rng: Any) -> torch.Tensor:
    sample = (torch.randn(value.shape, dtype=value.dtype, generator=rng) if isinstance(rng, torch.Generator)
              else torch.as_tensor(rng.normal(size=value.shape), dtype=value.dtype))
    return sample.to(value.device)


class ChangeState:
    """Own policy effects and episode buffers; reset before each episode."""

    def __init__(self, policy: Any, families: list, seed: int, legacy: bool, env: str, fixed_seed: int) -> None:
        self.policy, self.families, self.legacy, self.env = policy, families, legacy, env
        self.fixed = np.random.default_rng(fixed_seed)
        self.location, self.translation = self.fixed.random(2), self.fixed.choice([-1, 1], size=2)
        self.direction: np.ndarray | None = None
        weights = torch.Generator(device="cpu").manual_seed(seed)
        for family, params in families:
            if family == "chunk_len":
                policy.config.n_action_steps = params["n_action_steps"]
            if family == "weight_noise" and params["sigma"]:
                with torch.no_grad():
                    for parameter in policy.parameters():
                        if parameter.is_floating_point() and (not legacy or parameter.requires_grad):
                            scale = 1 if legacy else (parameter.std() if parameter.numel() > 1 else 0)
                            parameter.add_(_noise(parameter, weights) * params["sigma"] * scale)
        self.reset()

    def reset(self) -> None:
        """Clear episode history without reapplying policy-level changes."""
        self.rng = torch.Generator(device="cpu").manual_seed(torch.initial_seed())
        self.frame_rng = torch.Generator(device="cpu").manual_seed(torch.initial_seed() ^ 0x5DEECE66D)
        self.previous: dict = {}
        self.delay: deque = deque()
        self.step = 0
        self.policy.reset()

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute the explicitly seeded legacy inner-card contract."""
        if not self.legacy:
            raise ValueError("flat guide cards require obs_hook and action_hook around policy inference")
        action = self.policy.select_action(obs_hook(self, batch, self.rng))
        return action_hook(self, action, None, self.rng)


def obs_hook(state: ChangeState, obs: dict[str, Any], rng: Any) -> dict[str, Any]:
    """Apply ordered image changes and whole-observation frame drops."""
    observation = deepcopy(obs)
    needs_images = any(family in {"obs_noise", "camera_shift", "occlusion", "brightness"} for family, _ in state.families)
    if needs_images and not any(key.startswith("observation.image") for key in observation):
        raise ValueError("image perturbations require observation.image keys")
    for family, params in state.families:
        if family == "frame_drop":
            draw = (lambda: torch.rand((), generator=state.frame_rng).item()) if state.legacy else rng.random
            if state.previous and params["p"] and draw() < params["p"]:
                observation.update(deepcopy(state.previous))
            else:
                state.previous = deepcopy({key: value for key, value in observation.items() if not state.legacy or key.startswith("observation.image")})
        for key, value in observation.items():
            image = key.startswith("observation.image")
            if family == "obs_noise" and params["sigma"] and isinstance(value, torch.Tensor) and value.is_floating_point():
                if image or (state.legacy and key in {"observation.state", "observation.environment_state"}):
                    value = value + _noise(value, rng) * params["sigma"]
                    observation[key] = value.clamp(0, 1) if image else value
            elif image and family == "camera_shift":
                translation = [params["dx"], params["dy"]] if state.legacy else (state.translation * params["shift"]).tolist()
                observation[key] = TF.affine(value, params["angle"], translation, 1., [0., 0.], InterpolationMode.BILINEAR, fill=0)
            elif image and family == "occlusion":
                if state.legacy:
                    x, y, width, height = (params[name] for name in ("x", "y", "width", "height"))
                else:
                    width = height = max(1, round(params["fraction"] * min(value.shape[-2:])))
                    x, y = (state.location * [value.shape[-1] - width, value.shape[-2] - height]).astype(int)
                if x + width > value.shape[-1] or y + height > value.shape[-2]:
                    raise ValueError("occlusion rectangle is outside the camera image")
                value[..., y:y + height, x:x + width] = 0 if state.legacy else .5
            elif image and family == "brightness":
                value = (value - .5) * params["contrast"] + .5
                observation[key] = ((value if state.legacy else value.clamp(0, 1)) * params["gain"]).clamp(0, 1)
    return observation


def action_hook(state: ChangeState, action: Any, raw_env: Any, rng: Any) -> Any:
    """Apply ordered action changes using the current state vector for absolute actions."""
    result = action.clone() if isinstance(action, torch.Tensor) else np.array(action, copy=True)
    for family, params in state.families:
        if family in {"action_latency", "action_scale"} and not state.legacy:
            measured = (raw_env.agent.position if state.env == "pusht"
                        else raw_env._env.task.get_qpos(raw_env._env.physics))
            position = np.asarray(measured, dtype=result.dtype).reshape(1, -1)
            if position.shape != result.shape:
                raise ValueError("measured state shape must match absolute position actions")
        if family == "action_latency":
            if state.legacy and not state.delay:
                state.delay.extend(result.clone() for _ in range(params["k"]))
            state.delay.append(deepcopy(result))
            result = state.delay.popleft() if len(state.delay) > params["k"] else position.copy()
        elif family == "action_scale":
            result = result * params["scale"] if state.legacy else position + params["s"] * (result - position)
        elif family == "action_bias":
            if state.legacy:
                result = result + params["drift"] * (state.step + 1)
            else:
                if state.direction is None:
                    state.direction = state.fixed.normal(size=result.shape[-1])
                    if state.env == "aloha":
                        state.direction[[6, 13]] = 0
                    state.direction /= np.linalg.norm(state.direction)
                result = result + params["b"] / 100 * state.step * state.direction
    state.step += 1
    return result


def apply_change(policy: Any, change: dict[str, Any], *, seed: int | None = None) -> ChangeState:
    """Apply a flat guide card or explicit legacy inner-card plus seed contract.

    Guide weights use the numeric change id; fixed affine, patch and bias draws
    use the card seed. Episode noise draws come only from the supplied hook RNG.
    """
    legacy = seed is not None
    families = _validate(policy, change, legacy)
    seed = seed if legacy else int(change["change_id"][1:])
    if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
        raise ValueError("seed must be an integer between 0 and 2**64 - 1")
    return ChangeState(policy, families, seed, legacy, change.get("env", ""), change.get("seed", seed))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate pilot-ranked change manifests")
    parser.add_argument("--env", choices=["pusht", "aloha"], required=True)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--magnitudes", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write and args.magnitudes is None:
        parser.error("--write requires measured --magnitudes from the pilot")
    magnitudes = json.loads(args.magnitudes.read_text()) if args.magnitudes else None
    changes = generate_changes(args.env, args.n, round(.35 * args.n), magnitudes=magnitudes)
    if args.write:
        output = args.magnitudes.parent / f"changes_{args.env}.jsonl"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite frozen manifest: {output}")
        with output.open("x") as stream:
            stream.write("".join(json.dumps(change) + "\n" for change in changes))
    print(json.dumps({"changes": len(changes), "benign": sum(c["families"][0]["family"] == "benign" for c in changes),
                      "two_family": sum(len(c["families"]) == 2 for c in changes)}))


if __name__ == "__main__":
    _main()
