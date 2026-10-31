from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import modal

_Generate = Callable[[list[str], list[list[int]]], Sequence[Any]]
_ANSWERS = {
    "regressed": {"yes", "no"},
    "failure_mode": {"none", "stall/timeout", "overshoot/misalignment",
                     "grasp or drop failure", "collision", "oscillation"},
    "success_bucket": {"1", "2", "3", "4", "5"},
}
_GUIDE_ANSWERS = {**_ANSWERS,
    "failure_mode": {"none", "stall or timeout", "overshoot or misalignment",
                     "grasp or drop", "collision", "oscillation"},
    "success_bucket": {"0-19%", "20-39%", "40-59%", "60-79%", "80-100%"}}


def score_options(generate: _Generate, tokenizer: Any,
                  questions: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Score all three Rust-compiled questions using complete option logprobs."""
    if not isinstance(questions, list) or len(questions) != 3:
        raise ValueError("questions must contain all three questions for one pair")
    prompts, token_ids, identities, kinds, pairs = [], [], [], set(), set()
    for question in questions:
        if not isinstance(question, dict) or set(question) != {"id", "prompt", "options"}:
            raise ValueError("each question requires exactly id, prompt, options")
        identity, prompt, options = question["id"], question["prompt"], question["options"]
        guide = isinstance(identity, str) and ":" in identity
        parts = identity.split(":" if guide else "/") if isinstance(identity, str) else []
        if (len(parts) != 4 or parts[2] not in _ANSWERS
                or parts[3] not in ({"ev", "noev"} if guide else {"cheap", "no_cheap"})
                or not re.fullmatch(r"c[0-9]{3,}", parts[0])
                or not re.fullmatch(r"(?:pusht|aloha(?:_transfer_cube)?)-s[0-9]{2,}" if guide else r"(?:pusht|aloha|aloha_transfer_cube)-s[0-9]{2,}", parts[1])):
            raise ValueError("question id must be change/scenario/question/evidence")
        if not parts[0] or not parts[1] or parts[2] in kinds:
            raise ValueError("question ids must be distinct and contain a change/scenario")
        if guide and (int(parts[0][1:]) == 0 or int(parts[1].rsplit("s", 1)[1]) == 0):
            raise ValueError("guide change/scenario ids must be positive")
        kinds.add(parts[2])
        pairs.add((parts[0], parts[1], parts[3]))
        if guide and isinstance(options, list):
            if len(options) != len(_GUIDE_ANSWERS[parts[2]]):
                raise ValueError("invalid guide option count")
            options = dict(zip("ABCDEF", options))
        allowed = [_GUIDE_ANSWERS[parts[2]]] if guide else [_ANSWERS[parts[2]], _GUIDE_ANSWERS[parts[2]]]
        if (
            not isinstance(prompt, str) or not prompt.endswith("<answer>")
            or not isinstance(options, dict) or set(options) != set("ABCDEF"[:len(_ANSWERS[parts[2]])])
            or not all(isinstance(value, str) for value in options.values())
            or set(options.values()) not in allowed
        ):
            raise ValueError("prompt or question options do not match the compiled contract")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        if not 1 <= len(prompt_ids) <= 1536:
            raise ValueError("compiled prompt must fit 1536 tokens before the answer")
        ids = []
        for letter in sorted(options):
            encoded = tokenizer.encode(" " + letter, add_special_tokens=False)
            if (
                len(encoded) != 1 or type(encoded[0]) is not int
                or encoded[0] < 0 or encoded[0] in ids
                or tokenizer.encode(prompt + " " + letter, add_special_tokens=True) != prompt_ids + encoded
            ):
                raise ValueError("option letters must be distinct single tokens at the answer position")
            ids.append(encoded[0])
        prompts.append(prompt)
        token_ids.append(ids)
        identities.append(identity)
    if len(pairs) != 1:
        raise ValueError("all questions must refer to the same change/scenario/evidence")
    outputs = generate(prompts, token_ids)
    if len(outputs) != len(questions):
        raise RuntimeError("vLLM returned a different number of question results")
    result = {}
    for identity, ids, output in zip(identities, token_ids, outputs, strict=True):
        if len(output.outputs) != 1 or not output.outputs[0].logprobs or len(output.outputs[0].logprobs) != 1:
            raise RuntimeError("vLLM must return one answer-position logprob dictionary")
        logprobs = output.outputs[0].logprobs[0]
        if any(token not in logprobs or not math.isfinite(logprobs[token].logprob) for token in ids):
            raise RuntimeError("vLLM omitted an option token or returned a nonfinite logprob")
        values = [logprobs[token].logprob for token in ids]
        maximum = max(values)
        weights = [math.exp(value - maximum) for value in values]
        total = sum(weights)
        result[identity] = {letter: weight / total for letter, weight in zip("ABCDEF", weights)}
    return result


def _api(generate: _Generate, tokenizer: Any, version: str) -> Any:
    from fastapi import FastAPI, HTTPException

    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @web.post("/decide")
    def decide(payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload) != {"questions"}:
            raise HTTPException(status_code=422, detail="request requires exactly questions")
        try:
            predictions = score_options(generate, tokenizer, payload["questions"])
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"model_version": version, "questions": predictions}

    return web


def _checkpoint_path() -> Path | None:
    value = os.environ.get("MODAL_CHECKPOINT_DIR")
    if value is None:
        return None
    if not value or not os.environ.get("MODAL_VOLUME_NAME", "").strip():
        raise ValueError("LoRA serving requires MODAL_CHECKPOINT_DIR and MODAL_VOLUME_NAME")
    path = Path(value)
    if not path.is_absolute() or not path.resolve().is_relative_to("/checkpoints"):
        raise ValueError("MODAL_CHECKPOINT_DIR must be inside /checkpoints")
    return path.resolve()


def _adapter(directory: Path, model: str, revision: str) -> tuple[Any, str]:
    adapter = directory / "adapter"
    for name in ("adapter_config.json", "adapter_model.safetensors", "brier-model.json"):
        if not (adapter / name).is_file():
            raise ValueError(f"LoRA checkpoint is missing {name}")
    try:
        manifest = json.loads((adapter / "brier-model.json").read_text())
        adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("LoRA checkpoint has unreadable JSON metadata") from error
    config = manifest.get("config") if isinstance(manifest, dict) else None
    if not isinstance(config, dict) or config.get("model_id") != model or config.get("revision") != revision:
        raise ValueError("LoRA checkpoint does not match the pinned base model")
    if (
        not isinstance(adapter_config, dict) or adapter_config.get("peft_type") != "LORA"
        or type(adapter_config.get("r")) is not int or not 1 <= adapter_config["r"] <= 16
    ):
        raise ValueError("LoRA checkpoint requires a PEFT LoRA config with rank at most 16")
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        with (adapter / name).open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    from vllm.lora.request import LoRARequest

    return LoRARequest("brier", 1, str(adapter)), f"{model}@{revision}+lora@{digest.hexdigest()}"


def _load() -> tuple[_Generate, Any, str]:
    model, revision = os.environ.get("MODAL_MODEL_ID"), os.environ.get("MODAL_MODEL_REVISION")
    if not model or not revision:
        raise ValueError("brier-model secret requires MODAL_MODEL_ID and MODAL_MODEL_REVISION")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("MODAL_MODEL_REVISION must be an immutable 40-character Hugging Face commit")
    version, request, lora_options = f"{model}@{revision}", None, {}
    checkpoint = _checkpoint_path()
    if checkpoint is not None:
        request, version = _adapter(checkpoint, model, revision)
        lora_options = {"enable_lora": True, "max_lora_rank": 16}
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=model, revision=revision, tokenizer_revision=revision,
        dtype="bfloat16", trust_remote_code=False, language_model_only=True,
        enable_prefix_caching=True, max_model_len=1537, max_logprobs=20,
        logprobs_mode="raw_logprobs", seed=0, **lora_options,
    )
    tokenizer = engine.get_tokenizer()

    def generate(prompts: list[str], ids: list[list[int]]) -> Sequence[Any]:
        parameters = [SamplingParams(max_tokens=1, logprobs=20, temperature=0.0) for _ in ids]
        inputs = [{"prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=True)}
                  for prompt in prompts]
        options = {"lora_request": request} if request is not None else {}
        return engine.generate(inputs, parameters, use_tqdm=False, **options)

    return generate, tokenizer, version


image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .pip_install("vllm==0.28.0", "fastapi[standard]==0.162.0")
)
_checkpoint = _checkpoint_path()
_volumes = {}
if _checkpoint is not None:
    image = image.env({"MODAL_CHECKPOINT_DIR": str(_checkpoint),
                       "MODAL_VOLUME_NAME": os.environ["MODAL_VOLUME_NAME"]})
    _volumes = {"/checkpoints": modal.Volume.from_name(os.environ["MODAL_VOLUME_NAME"])}
app = modal.App("brier")


@app.function(image=image, gpu="A100-80GB", secrets=[modal.Secret.from_name("brier-model")],
              scaledown_window=120, timeout=600, volumes=_volumes)
@modal.asgi_app()
def serve() -> Any:
    """Load the pinned base and any requested LoRA once and serve POST /decide."""
    generate, tokenizer, version = _load()
    return _api(generate, tokenizer, version)
