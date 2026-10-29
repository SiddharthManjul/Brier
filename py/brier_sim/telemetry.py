from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.size == 0
        or array.dtype.kind not in "iuf"
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a nonempty finite numeric vector")
    return array.astype(np.float64)


def snapshot(environment: gym.Env) -> dict[str, Any]:
    """Capture object position, target distance and robot/table contact telemetry."""
    env_id = environment.spec.id if environment.spec is not None else None
    raw = environment.unwrapped
    collision = False
    if env_id == "gym_pusht/PushT-v0":
        position = np.array([*raw.block.position, 0.0], dtype=np.float64)
        goal = np.array([*raw.goal_pose[:2], 0.0], dtype=np.float64)
    elif env_id == "gym_aloha/AlohaTransferCube-v0":
        physics = raw._env.physics
        position = _vector(physics.named.data.geom_xpos["red_box"], "object position")
        goal = _vector(
            physics.named.data.geom_xpos["vx300s_left/10_left_gripper_finger"],
            "goal position",
        )
        for index in range(physics.data.ncon):
            contact = physics.data.contact[index]
            first = physics.model.id2name(contact.geom1, "geom")
            second = physics.model.id2name(contact.geom2, "geom")
            if (
                first == "table" and isinstance(second, str) and second.startswith("vx300s_")
            ) or (
                second == "table" and isinstance(first, str) and first.startswith("vx300s_")
            ):
                collision = True
    else:
        raise ValueError(f"Unsupported telemetry env: {env_id}")
    if position.shape != (3,) or goal.shape != (3,):
        raise ValueError("object and goal positions must contain three coordinates")
    with np.errstate(over="ignore", invalid="ignore"):
        distance = float(np.linalg.norm(position - goal))
    if not np.all(np.isfinite(position)) or not math.isfinite(distance):
        raise ValueError("object position and goal distance must be finite")
    return {
        "object_position": position.tolist(),
        "goal_distance": distance,
        "collision": collision,
    }


def summarize(
    actions: list[np.ndarray], snapshots: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize reset and step snapshots with raw action-direction statistics."""
    if not snapshots or len(snapshots) != len(actions) + 1:
        raise ValueError("snapshots must contain one reset plus one entry per action")
    positions, distances = [], []
    for item in snapshots:
        position = _vector(item.get("object_position"), "object position")
        distance = item.get("goal_distance")
        if (
            position.shape != (3,)
            or type(distance) not in (int, float)
            or not math.isfinite(distance)
            or distance < 0
            or type(item.get("collision")) is not bool
        ):
            raise ValueError("snapshots require finite positions, distances and boolean collisions")
        positions.append(position)
        distances.append(float(distance))
    vectors = [_vector(action, "action") for action in actions]
    if vectors and any(vector.shape != vectors[0].shape for vector in vectors):
        raise ValueError("actions must have matching shapes")
    direction_changes = 0
    if len(vectors) > 2:
        with np.errstate(over="ignore", invalid="ignore"):
            deltas = np.diff(np.stack(vectors), axis=0)
            products = np.sum(deltas[:-1] * deltas[1:], axis=1)
        if not np.all(np.isfinite(products)):
            raise ValueError("action-direction products must be finite")
        direction_changes = int(np.count_nonzero(products < 0))
    return {
        "collision_steps": sum(item["collision"] for item in snapshots[1:]),
        "action_direction_changes": direction_changes,
        "action_transitions": max(len(actions) - 2, 0),
        "object_max_height": float(max(position[2] for position in positions)),
        "object_final_height": float(positions[-1][2]),
        "goal_distance_min": min(distances),
        "goal_distance_final": distances[-1],
    }
