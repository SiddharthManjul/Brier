from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Sequence
from typing import Any

import modal

_Generate = Callable[[list[str], list[list[int]]], Sequence[Any]]
_ANSWERS = {
    "regressed": {"yes", "no"},
    "failure_mode": {"none", "stall/timeout", "overshoot/misalignment",
                     "grasp or drop failure", "collision", "oscillation"},
    "success_bucket": {"1", "2", "3", "4", "5"},
}


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
        parts = identity.split("/") if isinstance(identity, str) else []
        if len(parts) != 4 or parts[2] not in _ANSWERS or parts[3] not in {"cheap", "no_cheap"}:
            raise ValueError("question id must be change/scenario/question/evidence")
        if not parts[0] or not parts[1] or parts[2] in kinds:
            raise ValueError("question ids must be distinct and contain a change/scenario")
        kinds.add(parts[2])
        pairs.add((parts[0], parts[1], parts[3]))
        if (
            not isinstance(prompt, str) or not prompt.endswith("<answer>")
            or not isinstance(options, dict) or set(options) != set("ABCDEF"[:len(_ANSWERS[parts[2]])])
            or not all(isinstance(value, str) for value in options.values())
            or set(options.values()) != _ANSWERS[parts[2]]
        ):
            raise ValueError("prompt or question options do not match the compiled contract")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not 1 <= len(prompt_ids) <= 1535:
            raise ValueError("compiled prompt must fit the 1536-token sequence with one answer")
        ids = []
        for letter in sorted(options):
            encoded = tokenizer.encode(letter, add_special_tokens=False)
            if (
                len(encoded) != 1 or type(encoded[0]) is not int
                or encoded[0] < 0 or encoded[0] in ids
                or tokenizer.encode(prompt + letter, add_special_tokens=False) != prompt_ids + encoded
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


def _load() -> tuple[_Generate, Any, str]:
    model, revision = os.environ.get("MODAL_MODEL_ID"), os.environ.get("MODAL_MODEL_REVISION")
    if not model or not revision:
        raise ValueError("brier-model secret requires MODAL_MODEL_ID and MODAL_MODEL_REVISION")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("MODAL_MODEL_REVISION must be an immutable 40-character Hugging Face commit")
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=model, revision=revision, tokenizer_revision=revision,
        dtype="bfloat16", quantization="bitsandbytes", trust_remote_code=False,
        enable_prefix_caching=True, max_model_len=1536, max_logprobs=20,
        logprobs_mode="raw_logprobs", seed=0,
    )
    tokenizer = engine.get_tokenizer()

    def generate(prompts: list[str], ids: list[list[int]]) -> Sequence[Any]:
        parameters = [SamplingParams(max_tokens=1, logprobs=20, temperature=1.0,
                                     allowed_token_ids=tokens, logprob_token_ids=tokens)
                      for tokens in ids]
        inputs = [{"prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False)}
                  for prompt in prompts]
        return engine.generate(inputs, parameters, use_tqdm=False)

    return generate, tokenizer, f"{model}@{revision}"


image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install("vllm==0.21.0", "bitsandbytes==0.49.2", "fastapi[standard]==0.162.0")
)
app = modal.App("brier")


@app.function(image=image, gpu="L4", secrets=[modal.Secret.from_name("brier-model")],
              scaledown_window=120, timeout=600)
@modal.asgi_app()
def serve() -> Any:
    """Load the pinned base model once per container and serve POST /decide."""
    generate, tokenizer, version = _load()
    return _api(generate, tokenizer, version)
