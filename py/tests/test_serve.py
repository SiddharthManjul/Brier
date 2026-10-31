from __future__ import annotations

import importlib.util
import json
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
        suffix = len(text) >= 2 and text[-2] == " " and text[-1] in "ABCDEF"
        return ([1] if add_special_tokens else []) + ([ord(char) for char in text[:-2]] + [1000 + ord(text[-1])] if suffix else [ord(char) for char in text])


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
            return [1, 2] if text in {" " + letter for letter in "ABCDEF"} else super().encode(text, add_special_tokens=add_special_tokens)

    with pytest.raises(ValueError, match="distinct single tokens"):
        serving.score_options(_generate, BadTokenizer(), _questions())

    class BoundaryTokenizer(_Tokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            return ids[:-1] + [999] if text.endswith("> A") else ids

    with pytest.raises(ValueError, match="distinct single tokens"):
        serving.score_options(_generate, BoundaryTokenizer(), _questions())


@pytest.mark.parametrize("array_options", [True, False])
@pytest.mark.parametrize("scenario", ["pusht-s01", "aloha-s01", "aloha_transfer_cube-s01"])
def test_guide_questions_use_default_special_tokens_and_spaced_answer_tokens(array_options, scenario) -> None:
    questions = [{"id": f"c001:{scenario}:{kind}:ev", "prompt": f"{kind}\n<answer>",
                  "options": sorted(answers)} for kind, answers in serving._GUIDE_ANSWERS.items()]
    if not array_options:
        for question in questions:
            question["options"] = dict(zip("ABCDEF", question["options"]))

    class Tokenizer(_Tokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens == (not text.startswith(" "))
            return super().encode(text, add_special_tokens=add_special_tokens)

    def generate(prompts, ids):
        assert prompts == [question["prompt"] for question in questions]
        assert ids[0] == [1000 + ord("A"), 1000 + ord("B")]
        return _generate(prompts, ids)

    assert set(serving.score_options(generate, Tokenizer(), questions)) == {question["id"] for question in questions}
    questions[1]["options"] = [*sorted(serving._GUIDE_ANSWERS["failure_mode"]), "extra"]
    with pytest.raises(ValueError, match="option count"):
        serving.score_options(generate, Tokenizer(), questions)


def test_live_aloha_transport_accepts_guide_scenario_with_legacy_option_semantics() -> None:
    questions = _questions()
    for question in questions:
        question["id"] = question["id"].replace("pusht-s01", "aloha-s01")
        kind = question["id"].split("/")[2]
        question["prompt"] = "<system> exact\n<state>\nSCENARIO: env=aloha_transfer_cube pose_bin=1\n</state>\n<question> exact\n<options> " + "  ".join(
            f"{letter}) {text}" for letter, text in zip("ABCDEF", sorted(serving._GUIDE_ANSWERS[kind]))) + "\n<answer>"
    result = serving.score_options(_generate, _Tokenizer(), questions)
    assert set(result) == {question["id"] for question in questions}
    assert [len(result[question["id"]]) for question in questions] == [2, 6, 5]


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


@pytest.mark.parametrize("with_adapter", [False, True])
def test_load_requests_raw_complete_option_scores_and_exact_token_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_adapter: bool,
) -> None:
    calls: dict[str, Any] = {}

    class Engine:
        def __init__(self, **kwargs: Any) -> None:
            calls["engine"] = kwargs

        def get_tokenizer(self) -> _Tokenizer:
            return _Tokenizer()

        def generate(self, prompts: list[Any], parameters: list[Any], *, use_tqdm: bool, **kwargs: Any) -> list[Any]:
            calls["inputs"], calls["parameters"] = prompts, parameters
            calls["request"] = kwargs
            assert use_tqdm is False
            return _generate([], [[1000 + ord(letter) for letter in row["options"]] for row in _questions()])

    monkeypatch.setenv("MODAL_MODEL_ID", "configured/model")
    monkeypatch.setenv("MODAL_MODEL_REVISION", "a" * 40)
    monkeypatch.delenv("MODAL_CHECKPOINT_DIR", raising=False)
    if with_adapter:
        _lora_checkpoint(tmp_path, monkeypatch)
        monkeypatch.setattr(serving, "_checkpoint_path", lambda: tmp_path)
    def sampling_params(**kwargs: Any) -> dict[str, Any]:
        assert kwargs == {"max_tokens": 1, "logprobs": 20, "temperature": 0.0}
        return kwargs

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=sampling_params))
    generate, tokenizer, version = serving._load()
    serving.score_options(generate, tokenizer, _questions())
    assert calls["engine"]["logprobs_mode"] == "raw_logprobs"
    assert calls["engine"]["enable_prefix_caching"] is True
    assert calls["engine"]["dtype"] == "bfloat16"
    assert calls["engine"]["language_model_only"] is True
    assert "quantization" not in calls["engine"]
    if with_adapter:
        assert calls["engine"]["enable_lora"] is True
        assert calls["engine"]["max_lora_rank"] == 16
        assert calls["request"]["lora_request"] == ("brier", 1, str(tmp_path / "adapter"))
        assert version.startswith("configured/model@" + "a" * 40 + "+lora@")
        with TestClient(serving._api(generate, tokenizer, version)) as client:
            assert client.post("/decide", json={"questions": _questions()}).json()["model_version"] == version
    else:
        assert "enable_lora" not in calls["engine"] and calls["request"] == {}
        assert version == "configured/model@" + "a" * 40
    for question, tokens, parameters in zip(_questions(), calls["inputs"], calls["parameters"], strict=True):
        assert tokens == {"prompt_token_ids": tokenizer.encode(question["prompt"], add_special_tokens=True)}
        assert parameters["max_tokens"] == 1
        assert parameters["logprobs"] == 20 and parameters["temperature"] == 0.0
        assert "allowed_token_ids" not in parameters and "logprob_token_ids" not in parameters


def _lora_checkpoint(directory: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    adapter = directory / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 16, "lora_alpha": 32}))
    (adapter / "adapter_model.safetensors").write_bytes(b"test-adapter-weights")
    (adapter / "brier-model.json").write_text(json.dumps({
        "config": {"model_id": "configured/model", "revision": "a" * 40}, "step": 50,
    }))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", SimpleNamespace(LoRARequest=lambda *args: args))
    return adapter


def test_adapter_version_changes_for_weights_and_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _lora_checkpoint(tmp_path, monkeypatch)
    _, original = serving._adapter(tmp_path, "configured/model", "a" * 40)
    config = json.loads((adapter / "adapter_config.json").read_text())
    config["lora_alpha"] = 64
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    _, changed_config = serving._adapter(tmp_path, "configured/model", "a" * 40)
    (adapter / "adapter_model.safetensors").write_bytes(b"different-test-weights")
    _, changed_weights = serving._adapter(tmp_path, "configured/model", "a" * 40)
    assert len({original, changed_config, changed_weights}) == 3


@pytest.mark.parametrize("broken", ["adapter_config.json", "adapter_model.safetensors", "brier-model.json",
                                   "model_id", "revision", "invalid_json", "rank"])
def test_requested_adapter_fails_before_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: str,
) -> None:
    adapter = _lora_checkpoint(tmp_path, monkeypatch)
    if broken in {"model_id", "revision"}:
        manifest = json.loads((adapter / "brier-model.json").read_text())
        manifest["config"][broken] = "incorrect"
        (adapter / "brier-model.json").write_text(json.dumps(manifest))
    elif broken == "invalid_json":
        (adapter / "adapter_config.json").write_text("not JSON")
    elif broken == "rank":
        (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 32}))
    else:
        (adapter / broken).unlink()
    monkeypatch.setenv("MODAL_MODEL_ID", "configured/model")
    monkeypatch.setenv("MODAL_MODEL_REVISION", "a" * 40)
    monkeypatch.setattr(serving, "_checkpoint_path", lambda: tmp_path)
    with pytest.raises(ValueError, match="LoRA checkpoint"):
        serving._load()


def test_checkpoint_requires_volume_and_mount_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODAL_CHECKPOINT_DIR", "/checkpoints/run")
    monkeypatch.delenv("MODAL_VOLUME_NAME", raising=False)
    with pytest.raises(ValueError, match="MODAL_VOLUME_NAME"):
        serving._checkpoint_path()
    monkeypatch.setenv("MODAL_VOLUME_NAME", "test-volume")
    assert serving._checkpoint_path() == Path("/checkpoints/run")
    monkeypatch.setenv("MODAL_CHECKPOINT_DIR", "/tmp/run")
    with pytest.raises(ValueError, match="/checkpoints"):
        serving._checkpoint_path()


def test_image_keeps_python311_and_pins_native_glimmer_runtime_without_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    original_registry = serving.modal.Image.from_registry
    original_pip = serving.modal.Image.pip_install

    def from_registry(tag: str, **kwargs: Any) -> Any:
        calls["registry"], calls["python"] = tag, kwargs["add_python"]
        return original_registry(tag, **kwargs)

    def pip_install(image: Any, *packages: str, **kwargs: Any) -> Any:
        calls["packages"] = packages
        return original_pip(image, *packages, **kwargs)

    monkeypatch.delenv("MODAL_CHECKPOINT_DIR", raising=False)
    monkeypatch.setattr(serving.modal.Image, "from_registry", from_registry)
    monkeypatch.setattr(serving.modal.Image, "pip_install", pip_install)
    spec = importlib.util.spec_from_file_location("brier_serving_image", _SPEC.origin)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert calls["registry"] == "nvidia/cuda:13.0.3-devel-ubuntu24.04"
    assert calls["python"] == "3.11"
    assert calls["packages"] == ("vllm==0.28.0", "fastapi[standard]==0.162.0")
    assert module.serve._spec_.gpus == "A100-80GB"
