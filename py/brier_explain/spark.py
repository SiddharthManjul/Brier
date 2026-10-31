from __future__ import annotations

import base64
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib import error, request

from PIL import Image

_MODEL = "muse-spark-1.3-contributor"
_URL = "https://api.meta.ai/v1/chat/completions"
_FAILURES = ("none", "stall/timeout", "overshoot/misalignment",
             "grasp or drop failure", "collision", "oscillation")
_EXPLANATION = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "failure_mode": {"type": "string", "enum": list(_FAILURES)},
        "cause_hypothesis": {"type": "string", "minLength": 1},
        "evidence_frames": {"type": "array", "minItems": 1, "maxItems": 8,
                            "items": {"type": "integer", "minimum": 0, "maximum": 7}},
        "suggested_check": {"type": "string", "minLength": 1},
    },
    "required": ["failure_mode", "cause_hypothesis", "evidence_frames", "suggested_check"],
}
_ESCALATION = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "recommend": {"type": "string", "enum": ["run_expensive", "run_more_cheap", "no_change"]},
        "rationale": {"type": "string", "minLength": 1},
    },
    "required": ["recommend", "rationale"],
}


def _json(value: str | bytes) -> Any:
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("Muse JSON contains duplicate fields")
            result[key] = item
        return result

    def invalid(value: str) -> None:
        raise ValueError("Muse JSON contains a nonfinite number")

    return json.loads(value, object_pairs_hook=unique, parse_constant=invalid)


def _image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError("Muse images must be actual PNG files")
        image.verify()
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def _call(prompt: str, frames: Sequence[Path], schema: dict[str, Any], name: str) -> tuple[str, dict[str, Any]]:
    key = os.environ.get("META_API_KEY", "").strip()
    if not key:
        raise ValueError("META_API_KEY is required")
    if any(not 0x21 <= ord(character) <= 0x7e for character in key):
        raise ValueError("META_API_KEY must contain printable ASCII without whitespace")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("a nonempty Rust-compiled prompt is required")
    content = [{"type": "text", "text": prompt}, *[_image(Path(path)) for path in frames]]
    payload = {"model": _MODEL, "messages": [{"role": "user", "content": content}],
               "response_format": {"type": "json_schema", "json_schema": {
                   "name": name, "strict": True, "schema": schema}}}
    message = request.Request(_URL, data=json.dumps(payload, allow_nan=False).encode(), method="POST",
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    opener = request.OpenerDirector()
    for handler in (request.HTTPSHandler(), request.HTTPDefaultErrorHandler(), request.HTTPErrorProcessor()):
        opener.add_handler(handler)
    try:
        with opener.open(message, timeout=60) as response:
            body = _json(response.read())
    except error.HTTPError as failure:
        failure.close()
        raise RuntimeError(f"Muse request failed with HTTP {failure.code}") from None
    except (error.URLError, TimeoutError) as failure:
        raise RuntimeError("Muse request failed without a completed response") from failure
    if not isinstance(body, dict) or body.get("model") != _MODEL:
        raise ValueError("Muse response does not identify the requested Contributor model")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Muse response requires exactly one choice")
    choice = choices[0]
    if choice.get("finish_reason") != "stop" or not isinstance(choice.get("message"), dict):
        raise ValueError("Muse response did not finish normally")
    value = choice["message"].get("content")
    if not isinstance(value, str) or not value:
        raise ValueError("Muse response content must contain a JSON string")
    result = _json(value)
    if not isinstance(result, dict) or set(result) != set(schema["required"]):
        raise ValueError("Muse structured response fields differ from the required schema")
    return body["model"], result


def explain(prompt: str, frames: Sequence[Path]) -> dict[str, Any]:
    """Send Rust's prompt and sixteen ordered raw/overlay PNGs; return a strict explanation."""
    if isinstance(frames, (str, bytes)) or len(frames) != 16:
        raise ValueError("explanation requires eight raw and eight overlay PNGs")
    model, result = _call(prompt, frames, _EXPLANATION, "robot_failure")
    indices = result["evidence_frames"]
    if (
        result["failure_mode"] not in _FAILURES
        or any(not isinstance(result[key], str) or not result[key].strip()
               for key in ("cause_hypothesis", "suggested_check"))
        or not isinstance(indices, list) or not 1 <= len(indices) <= 8
        or any(type(index) is not int or not 0 <= index <= 7 for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("invalid Muse failure explanation")
    return {"model": model, "explanation": result, "cost_usd": None}


def escalate(prompt: str, frames: Sequence[Path]) -> dict[str, Any]:
    """Return a strict advisory recommendation without changing any engine decision."""
    if isinstance(frames, (str, bytes)) or not 0 <= len(frames) <= 8:
        raise ValueError("escalation accepts at most eight actual cheap-tier PNGs")
    model, result = _call(prompt, frames, _ESCALATION, "robot_escalation")
    if (
        result["recommend"] not in _ESCALATION["properties"]["recommend"]["enum"]
        or not isinstance(result["rationale"], str) or not result["rationale"].strip()
    ):
        raise ValueError("invalid Muse escalation recommendation")
    return {"model": model, **result, "cost_usd": None}
