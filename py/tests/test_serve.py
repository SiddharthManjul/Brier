from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

_SPEC = importlib.util.spec_from_file_location("brier_serving", Path(__file__).parents[2] / "serve/modal_app.py")
assert _SPEC and _SPEC.loader
serving = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = serving
_SPEC.loader.exec_module(serving)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) for char in text]


def _questions() -> list[dict[str, Any]]:
    return [{"id": f"c001/pusht-s01/{kind}/cheap", "prompt": f"{kind}\n<answer>",
             "options": {chr(65 + i): answer for i, answer in enumerate(sorted(answers))}}
            for kind, answers in serving._ANSWERS.items()]


def _generate(prompts: list[str], token_ids: list[list[int]]) -> list[Any]:
    return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{
        token: SimpleNamespace(logprob=-float(index + 1)) for index, token in enumerate(ids)}])])
            for ids in token_ids]


def test_score_options_preserves_prompts_and_renormalizes_all_letters() -> None:
    questions = _questions()

    def generate(prompts: list[str], ids: list[list[int]]) -> list[Any]:
        assert prompts == [question["prompt"] for question in questions]
        assert list(map(len, ids)) == [2, 6, 5]
        return _generate(prompts, ids)

    result = serving.score_options(generate, _Tokenizer(), questions)
    assert set(result) == {question["id"] for question in questions}
    regression = result[questions[0]["id"]]
    assert regression["A"] == pytest.approx(1 / (1 + math.exp(-1)))
    assert regression["B"] == pytest.approx(1 - regression["A"])
    assert all(sum(probabilities.values()) == pytest.approx(1) for probabilities in result.values())


@pytest.mark.parametrize("mutation", ["missing_question", "wrong_pair", "duplicate", "label_leak", "options", "prompt"])
def test_score_rejects_invalid_compiled_contract(mutation: str) -> None:
    questions = _questions()
    if mutation == "missing_question":
        questions.pop()
    elif mutation == "wrong_pair":
        questions[0]["id"] = questions[0]["id"].replace("c001", "c002")
    elif mutation == "duplicate":
        questions[0] = questions[1]
    elif mutation == "label_leak":
        questions[0]["label"] = "A"
    elif mutation == "options":
        questions[0]["options"]["A"] = "unknown"
    else:
        questions[0]["prompt"] = "X" * 1600 + "<answer>"
    with pytest.raises(ValueError):
        serving.score_options(_generate, _Tokenizer(), questions)


def test_missing_logprob_fails_loudly() -> None:
    def generate(prompts: list[str], ids: list[list[int]]) -> list[Any]:
        results = _generate(prompts, ids)
        results[0].outputs[0].logprobs[0].pop(ids[0][0])
        return results

    with pytest.raises(RuntimeError, match="omitted an option"):
        serving.score_options(generate, _Tokenizer(), _questions())


def test_multiple_tokens_and_context_mismatch_fail() -> None:
    class BadTokenizer(_Tokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return [1, 2] if len(text) == 1 else super().encode(text, add_special_tokens=add_special_tokens)

    with pytest.raises(ValueError, match="distinct single tokens"):
        serving.score_options(_generate, BadTokenizer(), _questions())


def test_load_requires_model_and_immutable_revision_before_importing_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODAL_MODEL_ID", raising=False)
    monkeypatch.delenv("MODAL_MODEL_REVISION", raising=False)
    with pytest.raises(ValueError, match="MODAL_MODEL_ID"):
        serving._load()
    monkeypatch.setenv("MODAL_MODEL_ID", "configured/model")
    monkeypatch.setenv("MODAL_MODEL_REVISION", "main")
    with pytest.raises(ValueError, match="immutable"):
        serving._load()


def test_remote_factory_routes_and_model_version_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serving, "_load", lambda: (_generate, _Tokenizer(), "configured/model@revision"))
    web = serving.serve.local()
    assert [(route.path, route.methods) for route in web.routes] == [("/decide", {"POST"})]
    with TestClient(web) as client:
        response = client.post("/decide", json={"questions": _questions()})
        assert response.status_code == 200
        assert response.json()["model_version"] == "configured/model@revision"
        assert client.post("/decide", json={"questions": []}).status_code == 422
        assert client.post("/decide", json={"questions": _questions(), "label": "A"}).status_code == 422
        malformed = _questions()
        malformed[0]["options"]["A"] = ["yes"]
        assert client.post("/decide", json={"questions": malformed}).status_code == 422
        assert client.get("/docs").status_code == 404


def test_load_requests_raw_complete_option_scores_and_exact_token_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    class Engine:
        def __init__(self, **kwargs: Any) -> None:
            calls["engine"] = kwargs

        def get_tokenizer(self) -> _Tokenizer:
            return _Tokenizer()

        def generate(self, prompts: list[Any], parameters: list[Any], *, use_tqdm: bool) -> list[Any]:
            calls["inputs"], calls["parameters"] = prompts, parameters
            assert use_tqdm is False
            return _generate([], [parameter["allowed_token_ids"] for parameter in parameters])

    monkeypatch.setenv("MODAL_MODEL_ID", "configured/model")
    monkeypatch.setenv("MODAL_MODEL_REVISION", "a" * 40)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=lambda **kwargs: kwargs))
    generate, tokenizer, version = serving._load()
    serving.score_options(generate, tokenizer, _questions())
    assert calls["engine"]["logprobs_mode"] == "raw_logprobs"
    assert calls["engine"]["enable_prefix_caching"] is True
    assert calls["engine"]["quantization"] == "bitsandbytes"
    assert version == "configured/model@" + "a" * 40
    for question, tokens, parameters in zip(_questions(), calls["inputs"], calls["parameters"], strict=True):
        assert tokens == {"prompt_token_ids": tokenizer.encode(question["prompt"], add_special_tokens=False)}
        assert parameters["max_tokens"] == 1 and parameters["logprobs"] == 20
        assert parameters["allowed_token_ids"] == parameters["logprob_token_ids"]
