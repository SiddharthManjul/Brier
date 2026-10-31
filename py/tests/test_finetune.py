import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from brier_ml import finetune as module


def _row(change: str = "c001", split: str = "train") -> dict[str, Any]:
    return {"id": f"{change}/pusht-s01/regressed/cheap", "split": split,
            "question": "regressed", "options": {"A": "yes", "B": "no"},
            "prompt": "<system> Brier Ω\n<state>\ncompiled\n</state>\n<question> q\n<options> A) yes  B) no\n<answer>",
            "label": "A"}


@pytest.mark.parametrize("question,values,penalty", [
    ("regressed", ["yes", "no"], 0.5),
    ("failure_mode", ["none", "stall/timeout", "overshoot/misalignment",
                      "grasp or drop failure", "collision", "oscillation"], 5 / 6),
    ("success_bucket", ["1", "2", "3", "4", "5"], 0.1),
])
def test_option_loss_matches_proper_scoring_rules(
    question: str, values: list[str], penalty: float,
) -> None:
    logits = torch.zeros(len(values), requires_grad=True)
    target = 2 if question == "success_bucket" else 0
    loss = module.option_loss(logits, target, question, values)
    assert loss.item() == pytest.approx(math.log(len(values)) + penalty)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_score_loss_uses_semantic_order_when_letters_are_shuffled() -> None:
    logits = torch.tensor([1.0, -1.0, 0.2, 0.4, -0.5])
    order = [4, 0, 2, 1, 3]
    original = module.option_loss(logits, 2, "success_bucket", ["1", "2", "3", "4", "5"])
    shuffled = module.option_loss(logits[order], order.index(2), "success_bucket", [str(i + 1) for i in order])
    assert original.item() == pytest.approx(shuffled.item())
    with pytest.raises(ValueError, match="option logits"):
        module.option_loss(torch.tensor([float("nan"), 1.0]), 0, "regressed", ["yes", "no"])


def test_load_training_data_filters_splits_and_preserves_prompt(tmp_path: Path) -> None:
    rows = [_row(), _row("c002", "calibration"), _row("c003", "test"), _row("c004", "ood")]
    compiled = tmp_path / "compiled.jsonl"
    compiled.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert module.load_training_data(compiled) == rows[:1]


@pytest.mark.parametrize("mutation", ["id", "split", "question", "options", "prompt", "label", "extra", "duplicate", "leak", "no_train", "json"])
def test_load_training_data_rejects_invalid_contract(tmp_path: Path, mutation: str) -> None:
    row, rows = _row(), []
    if mutation == "extra":
        row["extra"] = 0
    elif mutation == "duplicate":
        rows.append(row.copy())
    elif mutation == "leak":
        other = _row(split="test")
        other["id"] = "c001/pusht-s01/regressed/no_cheap"
        rows.append(other)
    elif mutation == "no_train":
        row["split"] = "test"
    elif mutation != "json":
        row[mutation] = {} if mutation == "options" else "invalid"
    rows.insert(0, row)
    compiled = tmp_path / "compiled.jsonl"
    compiled.write_text("bad json" if mutation == "json" else "\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):
        module.load_training_data(compiled)


class _Tokenizer:
    def __init__(self, mode: str = "valid") -> None:
        self.mode, self.calls = mode, []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.calls.append(text)
        if text in ("A", "B"):
            return [3, 4] if self.mode == "non_atomic" else [3 + (text == "B")]
        if text.endswith(("A", "B")):
            return [1, 2, 1, 3 + (text[-1] == "B")] if self.mode != "boundary" else [7]
        return [1] * 1537 if self.mode == "long" else [1, 2, 1]


def test_encoding_uses_compiled_prompt_without_template_or_truncation() -> None:
    tokenizer = _Tokenizer()
    encoded = module._encode([_row()], tokenizer)
    assert tokenizer.calls[0] == _row()["prompt"]
    assert encoded == [{"tokens": [1, 2, 1], "option_tokens": [3, 4],
                        "target": 0, "question": "regressed", "values": ["yes", "no"]}]
    for mode, message in [("long", "1535"), ("non_atomic", "single-token"), ("boundary", "answer boundary")]:
        with pytest.raises(ValueError, match=message):
            module._encode([_row()], _Tokenizer(mode))


@pytest.mark.parametrize("native,missing_text", [(False, False), (True, False), (True, True)])
def test_model_loader_uses_native_glimmer_and_only_text_lora_projections(
    monkeypatch: pytest.MonkeyPatch, native: bool, missing_text: bool,
) -> None:
    config = SimpleNamespace(is_encoder_decoder=False, vision_config=None, use_cache=True)
    if native:
        config.model_type = "muse_glimmer"
        config.architectures = ["MuseGlimmerForConditionalGeneration"]
        config.vision_config = SimpleNamespace(model_type="muse_glimmer_vision")
        config.text_config = SimpleNamespace(model_type="muse_glimmer_text", max_position_embeddings=131072, use_cache=True)
    model, calls = torch.nn.Module(), {}
    model.config = config
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in
        ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj")})
    layer.mlp = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in
        ("gate_proj", "up_proj", "down_proj") if name != "up_proj" or not missing_text})
    model.model.language_model.layers = torch.nn.ModuleList([layer])
    if native:
        model.model.vision_tower = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in
            ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")})
    tokenizer = object()

    def load(identifier, **kwargs):
        assert identifier == "configured/model" and kwargs["revision"] == "a" * 40
        assert kwargs["config"] is config and not kwargs["trust_remote_code"]
        assert kwargs["device_map"] == {"": 0} and kwargs["dtype"] == torch.bfloat16
        assert kwargs["quantization_config"] == {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                                  "bnb_4bit_compute_dtype": torch.bfloat16}
        return model

    def wrong_loader(*args, **kwargs):
        pytest.fail("the loader selected the wrong Transformers model class")

    def prepare(value, *, use_gradient_checkpointing):
        assert value is model and use_gradient_checkpointing
        calls["prepared"] = True
        return model.requires_grad_(False)

    def lora(**kwargs):
        calls["lora"] = kwargs
        return SimpleNamespace(**kwargs)

    def wrap(value, config):
        assert value is model and calls["prepared"]
        for name, projection in model.named_modules():
            if name in config.target_modules or (not native and name.rsplit(".", 1)[-1] in config.target_modules):
                projection.register_parameter("lora_A", torch.nn.Parameter(torch.ones(1)))
        return model

    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        LoraConfig=lora, get_peft_model=wrap, prepare_model_for_kbit_training=prepare))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: config),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
        BitsAndBytesConfig=lambda **kw: kw,
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=wrong_loader if native else load),
        MuseGlimmerForConditionalGeneration=SimpleNamespace(from_pretrained=load if native else wrong_loader),
    ))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    if missing_text:
        with pytest.raises(ValueError, match="required attention and MLP projections"):
            module._load_model("configured/model", "a" * 40)
        assert "prepared" not in calls
        return
    loaded, actual_tokenizer = module._load_model("configured/model", "a" * 40)
    assert loaded is model and actual_tokenizer is tokenizer and not config.use_cache
    assert calls["lora"]["r"] == 16 and calls["lora"]["task_type"] == "CAUSAL_LM"
    if native:
        assert not config.text_config.use_cache
        expected = [f"model.language_model.layers.0.self_attn.{name}" for name in
                    ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj")]
        expected += [f"model.language_model.layers.0.mlp.{name}" for name in ("gate_proj", "up_proj", "down_proj")]
        assert calls["lora"]["target_modules"] == expected
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        assert trainable and all(name.startswith("model.language_model.") and name.endswith("lora_A") for name in trainable)
        assert not any(parameter.requires_grad for parameter in model.model.vision_tower.parameters())
    else:
        assert calls["lora"]["target_modules"] == module._PROJECTIONS


@pytest.mark.parametrize("kind", ["unknown_vision", "wrong_architecture", "wrong_text", "wrong_vision", "encoder_decoder"])
def test_native_glimmer_exception_rejects_unverified_multimodal_configurations(kind: str) -> None:
    config = SimpleNamespace(model_type="muse_glimmer", architectures=["MuseGlimmerForConditionalGeneration"],
        vision_config=SimpleNamespace(model_type="muse_glimmer_vision"),
        text_config=SimpleNamespace(model_type="muse_glimmer_text"))
    if kind == "unknown_vision":
        config.model_type = "unverified_vision_model"
    elif kind == "wrong_architecture":
        config.architectures = ["UnverifiedForConditionalGeneration"]
    elif kind == "wrong_text":
        config.text_config.model_type = "unverified_text_model"
    elif kind == "wrong_vision":
        config.vision_config.model_type = "unverified_vision_model"
    else:
        config.text_config.is_encoder_decoder = True
    with pytest.raises(ValueError, match="native Muse Glimmer|Muse Glimmer requires"):
        module._text_config(config)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scores = torch.nn.Parameter(torch.zeros(1, 3, 8))

    def forward(self, input_ids: torch.Tensor, use_cache: bool) -> Any:
        assert input_ids.shape == (1, 3) and use_cache is False
        return SimpleNamespace(logits=self.scores)

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "adapter.pt")


def test_training_accumulates_partial_batch_at_answer_position_and_resumes(tmp_path: Path) -> None:
    model = _Model()
    encoded = module._encode([_row()] * 3, _Tokenizer())
    encoded[-1]["target"] = 1
    config = {"dataset_digest": "one", "base_config": {}, "smoke": False}
    commits: list[bool] = []
    result = module._fit(model, encoded, tmp_path, config, False, lambda: commits.append(True))
    state = torch.load(tmp_path / "training-state.pt", weights_only=False)
    assert result["optimizer_steps"] == 1 and result["tokens"] == 9
    assert json.loads((tmp_path / "training-report.json").read_text()) == result
    assert state["cursor"] == 3 and len(commits) == 2
    assert json.loads((tmp_path / "adapter" / "brier-model.json").read_text()) == {"config": config, "step": 1}
    momentum = state["optimizer"]["state"][0]["exp_avg"]
    torch.testing.assert_close(momentum[0, -1, 3:5], torch.tensor([-1 / 30, 1 / 30]))
    assert torch.count_nonzero(model.scores[:, :-1]) == 0
    assert torch.count_nonzero(model.scores[0, -1, :3]) == 0
    expected = torch.rand(3)
    fresh = _Model()
    resumed = module._fit(fresh, encoded, tmp_path, config, False, lambda: None)
    assert resumed["optimizer_steps_this_run"] == 0 and resumed["tokens"] == 0
    assert json.loads((tmp_path / "training-report.json").read_text()) == resumed
    torch.testing.assert_close(model.scores, fresh.scores)
    torch.testing.assert_close(torch.rand(3), expected)
    for key in ("dataset_digest", "base_config", "smoke"):
        with pytest.raises(ValueError, match="configuration, dataset digest or smoke"):
            module._fit(_Model(), encoded, tmp_path, {**config, key: "different"}, False, lambda: None)


def test_checkpoint_publishes_model_manifest_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {"model_id": "configured/model", "revision": "a" * 40, "smoke": False}
    manifest = tmp_path / "adapter" / "brier-model.json"
    replacements = []
    replace = Path.replace

    def publish(path: Path, target: Path) -> Path:
        if target == manifest:
            assert path == manifest.with_suffix(".json.tmp") and not manifest.exists()
            assert json.loads(path.read_text()) == {"config": config, "step": 1}
            replacements.append(target)
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", publish)
    module._fit(_Model(), module._encode([_row()], _Tokenizer()), tmp_path, config, False, lambda: None)
    assert replacements == [manifest]
    assert not manifest.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("distillation", [False, True])
def test_smoke_performs_exactly_fifty_optimizer_steps(tmp_path: Path, distillation: bool) -> None:
    encoded = module._encode([_row()], _Tokenizer())
    result = module._fit(_Model(), encoded, tmp_path, {"smoke": True}, True, lambda: None,
                         teacher_distributions=[[0.25, 0.75]] if distillation else None)
    state = torch.load(tmp_path / "training-state.pt", weights_only=False)
    assert result["optimizer_steps"] == 50 and state["cursor"] == 1600
    assert result["tokens"] == 4800 and result["tokens_per_second"] > 0


@pytest.mark.parametrize("distillation", [False, True])
def test_two_hundred_step_checkpoint_resumes_after_interruption(tmp_path: Path, distillation: bool) -> None:
    encoded = module._encode([_row()], _Tokenizer()) * 6401
    config = {"dataset_digest": "long", "smoke": False}
    targets = [[0.25, 0.75]] * len(encoded) if distillation else None

    def interrupt() -> None:
        raise RuntimeError("preempted after checkpoint commit")

    with pytest.raises(RuntimeError, match="preempted"):
        module._fit(_Model(), encoded, tmp_path, config, False, interrupt, teacher_distributions=targets)
    state = torch.load(tmp_path / "training-state.pt", weights_only=False)
    assert state["step"] == 200 and state["cursor"] == 6400
    result = module._fit(_Model(), encoded, tmp_path, config, False, lambda: None, teacher_distributions=targets)
    final = torch.load(tmp_path / "training-state.pt", weights_only=False)
    assert result["optimizer_steps_this_run"] == 1 and result["tokens"] == 3
    assert final["step"] == 201 and final["cursor"] == 6401


def test_distillation_adds_teacher_kl_and_binds_resume_targets(tmp_path: Path) -> None:
    encoded = module._encode([_row()] * 3, _Tokenizer())
    encoded[-1]["target"] = 1
    config = {"dataset_digest": "distillation", "smoke": False}
    targets = [[0.25, 0.75]] * len(encoded)
    module._fit(_Model(), encoded, tmp_path, config, False, lambda: None, teacher_distributions=targets)
    state = torch.load(tmp_path / "training-state.pt", weights_only=False)
    momentum = state["optimizer"]["state"][0]["exp_avg"]
    torch.testing.assert_close(momentum[0, -1, 3:5], torch.tensor([-1 / 120, 1 / 120]))
    assert "distillation_digest" in state["config"] and "distillation_digest" not in config
    for different in (None, [[0.75, 0.25]] * len(encoded)):
        with pytest.raises(ValueError, match="configuration"):
            module._fit(_Model(), encoded, tmp_path, config, False, lambda: None,
                        teacher_distributions=different)


@pytest.mark.parametrize("targets", [[], [[float("nan"), 0.5]], [[True, False]], [[0.1, 0.1]], [[0.1, 0.2, 0.7]]])
def test_distillation_rejects_invalid_targets_before_checkpoint_creation(tmp_path: Path, targets) -> None:
    with pytest.raises(ValueError, match="teacher distributions"):
        module._fit(_Model(), module._encode([_row()], _Tokenizer()), tmp_path, {}, False,
                    lambda: None, teacher_distributions=targets)
    assert not list(tmp_path.iterdir())


def test_train_requires_configuration_without_loading_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODAL_MODEL_ID", raising=False)
    with pytest.raises(ValueError, match="MODAL_MODEL_ID is required"):
        module.train(tmp_path / "compiled.jsonl")
