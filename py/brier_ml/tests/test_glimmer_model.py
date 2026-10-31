import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("transformers")
pytest.importorskip("peft")

from brier_ml.glimmer_model import (
    add_lora,
    last_hidden,
    last_logits,
    load_base,
    option_logits,
)


@pytest.fixture
def llama():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = tf.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=2, num_attention_heads=2,
                           num_key_value_heads=1, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    return tf.LlamaForCausalLM(config).eval()


@pytest.fixture
def glimmer():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = tf.MuseGlimmerConfig(
        text_config={"vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 2, "num_attention_heads": 2,
                     "num_key_value_heads": 1, "head_dim": 8,
                     "layer_types": ["sliding_attention", "full_attention"],
                     "layer_rope_theta": [10000.0, 0], "sliding_window": 8},
        vision_config={"hidden_size": 16, "intermediate_size": 32,
                       "num_hidden_layers": 1, "num_attention_heads": 2,
                       "layer_types": ["full_attention"], "patch_size": 2,
                       "patch_temporal": 1, "pos_emb_height": 2, "pos_emb_width": 2},
        out_hidden_size=16, projector_hidden_size=16)
    config._attn_implementation = "sdpa"
    return tf.MuseGlimmerForConditionalGeneration(config).eval()


def batch(rows):
    ids = torch.tensor(rows)
    mask = (ids != 0).long()
    return {"input_ids": ids, "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp(min=0)}


@pytest.mark.parametrize("model_fixture", ["llama", "glimmer"])
def test_left_padding_and_last_position_only(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
    seen = []
    hook = model.get_output_embeddings().register_forward_pre_hook(
        lambda _module, args: seen.append(tuple(args[0].shape)))
    with torch.inference_mode():
        together = last_logits(model, batch([[0, 0, 4, 5], [3, 4, 5, 6]]))
        alone = last_logits(model, batch([[4, 5]]))
    hook.remove()
    assert together.dtype == torch.float32
    torch.testing.assert_close(together[0], alone[0], atol=1e-4, rtol=0)
    assert seen == [(2, 1, 16), (1, 1, 16)]


def test_last_hidden_is_post_norm_and_does_not_call_lm_head(glimmer):
    observed = []
    norm_hook = glimmer.model.language_model.norm.register_forward_hook(
        lambda _module, _args, output: observed.append(output.detach()))
    head_hook = glimmer.lm_head.register_forward_pre_hook(
        lambda *_: pytest.fail("Embedding extraction must not allocate vocabulary logits"))
    with torch.inference_mode():
        hidden = last_hidden(glimmer, batch([[0, 4, 5], [3, 4, 5]]))
    norm_hook.remove()
    head_hook.remove()
    assert hidden.shape == (2, 16)
    torch.testing.assert_close(hidden, observed[0][:, -1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_manual_last_logits_preserve_glimmer_output_scaling(glimmer, monkeypatch, dtype):
    glimmer.to(dtype=dtype)
    inputs = batch([[0, 4, 5], [3, 4, 5]])
    with torch.no_grad():
        glimmer.lm_head.weight.mul_(1000)
        expected = last_logits(glimmer, inputs)
    original = glimmer.forward

    def no_logits_to_keep(input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        return original(input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, **kwargs)

    monkeypatch.setattr(glimmer, "forward", no_logits_to_keep)
    seen = []
    hook = glimmer.lm_head.register_forward_pre_hook(
        lambda _module, args: seen.append(tuple(args[0].shape)))
    actual = last_logits(glimmer, inputs)
    hook.remove()
    assert seen == [(2, 16)]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
    assert actual.abs().max() <= glimmer.config.text_config.final_logit_softcapping


@pytest.mark.parametrize("model_fixture", ["llama", "glimmer"])
def test_lora_targets_and_gradients_are_text_only(request, model_fixture):
    model = add_lora(request.getfixturevalue(model_fixture))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)
    assert not any(any(tag in name for tag in ("vision", "perception", "projector", "lm_head", "embed"))
                   for name in trainable)
    leaves = {name.split(".lora_")[0].rsplit(".", 1)[-1] for name in trainable}
    assert {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"} <= leaves
    if model_fixture == "glimmer":
        assert all("language_model" in name for name in trainable)
        assert any("self_attn.gate_proj.lora_" in name for name in trainable)
    assert not model.config.use_cache
    assert model.is_gradient_checkpointing
    model.train()
    last_logits(model, batch([[4, 5, 6]])).sum().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for name, parameter in model.named_parameters() if "lora_" in name)


def test_lora_rejects_missing_attention_projection(llama):
    llama.model.layers[0].self_attn.q_proj = torch.nn.Identity()
    llama.model.layers[1].self_attn.q_proj = torch.nn.Identity()
    with pytest.raises(ValueError, match="Missing text attention"):
        add_lora(llama)


def test_option_logits_shape_order_and_gradient():
    logits = torch.randn(2, 64, requires_grad=True)
    ids = [7, 2, 9, 30, 4, 12]
    selected = option_logits(logits, ids)
    assert selected.shape == (2, 6)
    torch.testing.assert_close(selected, logits[:, ids])
    selected.sum().backward()
    assert logits.grad[:, ids].eq(1).all()


@pytest.mark.parametrize("ids", [[1, 2], [1, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 64],
                                  [True, 2, 3, 4, 5, 6]])
def test_invalid_letter_ids(ids):
    with pytest.raises(ValueError, match="six distinct"):
        option_logits(torch.zeros(2, 64), ids)


def test_padding_and_position_ids_are_required(llama):
    wrong = batch([[4, 5, 0]])
    with pytest.raises(ValueError, match="left-padded"):
        last_logits(llama, wrong)
    wrong = batch([[0, 4, 5]])
    wrong["position_ids"] += 1
    with pytest.raises(ValueError, match="Position ids"):
        last_hidden(llama, wrong)


def install_loaders(monkeypatch, model, causal=False):
    calls = []
    tokenizer = object()
    prefix = "model.language_model." if hasattr(model.model, "language_model") else "model."
    perception = sum(parameter.numel() for name, parameter in model.named_parameters()
                     if not name.startswith((prefix, "lm_head.")))
    model.num_parameters = lambda: 27_800_000_000 + perception

    def loader(kind):
        def load(path, **kwargs):
            calls.append((kind, path, kwargs))
            if kind == "causal" and not causal:
                raise ValueError("Unrecognized configuration class MuseGlimmerConfig")
            return model
        return load

    monkeypatch.setattr(tf.AutoModelForCausalLM, "from_pretrained", loader("causal"))
    monkeypatch.setattr(tf.AutoModelForMultimodalLM, "from_pretrained", loader("multimodal"))
    monkeypatch.setattr(tf.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    return calls, tokenizer


def test_load_base_records_local_revision_and_actual_final_norm(glimmer, monkeypatch):
    calls, tokenizer = install_loaders(monkeypatch, glimmer)
    model, loaded_tokenizer, info = load_base("/local/weights", "a" * 40, "bf16")
    assert model is glimmer and loaded_tokenizer is tokenizer
    assert [call[0] for call in calls] == ["causal", "multimodal"]
    assert all(call[2] == {"revision": "a" * 40, "local_files_only": True,
                           "dtype": torch.bfloat16, "device_map": {"": 0},
                           "attn_implementation": "sdpa"} for call in calls)
    assert info["final_norm"] == "model.language_model.norm"
    assert info["hidden_state_after_norm"] and info["logits_to_keep"]
    assert info["text_parameters"] == 27_800_000_000
    assert info["parameters"] == 27_800_000_000 + info["perception_parameters"]
    assert info["perception_parameters"] > 0
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_load_causal_candidate_with_vision_is_rejected(glimmer, monkeypatch):
    calls, _ = install_loaders(monkeypatch, glimmer, causal=True)
    load_base("/local/weights", "a" * 40)
    assert [call[0] for call in calls] == ["causal", "multimodal"]


def test_load_text_only_causal_candidate(llama, monkeypatch):
    calls, _ = install_loaders(monkeypatch, llama, causal=True)
    _, _, info = load_base("/local/weights", "a" * 40)
    assert [call[0] for call in calls] == ["causal"]
    assert info["final_norm"] == "model.norm"
    assert info["parameters"] == info["text_parameters"] == 27_800_000_000


def test_nf4_skips_perception_and_prepares_kbit_training(glimmer, monkeypatch):
    import peft

    calls, _ = install_loaders(monkeypatch, glimmer)
    monkeypatch.setattr(glimmer.model.language_model, "num_parameters",
                        lambda: pytest.fail("Only the root records NF4 packing metadata"))
    monkeypatch.setattr(tf.AutoConfig, "from_pretrained", lambda *a, **kw: glimmer.config)
    monkeypatch.setattr(tf.AutoModelForMultimodalLM, "from_config", lambda config: glimmer)
    monkeypatch.setattr(tf, "BitsAndBytesConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    prepared = []
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training",
                        lambda model, **kwargs: prepared.append(kwargs) or model)
    _, _, info = load_base("/local/weights", "a" * 40, "nf4")
    config = calls[-1][2]["quantization_config"]
    assert config.load_in_4bit and config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant
    assert config.bnb_4bit_compute_dtype == torch.bfloat16
    assert "lm_head" in config.llm_int8_skip_modules
    assert "model.vision_tower" in config.llm_int8_skip_modules
    assert prepared == [{"use_gradient_checkpointing": True}]
    assert info["quant"] == "nf4"


def test_weight_or_config_errors_do_not_trigger_architecture_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Checkpoint files are corrupt")

    monkeypatch.setattr(tf.AutoModelForCausalLM, "from_pretrained", fail)
    monkeypatch.setattr(tf.AutoModelForMultimodalLM, "from_pretrained",
                        lambda *a, **kw: pytest.fail("Unexpected fallback"))
    with pytest.raises(ValueError, match="corrupt"):
        load_base("/local/weights", "a" * 40)


def test_unknown_quant_or_unpinned_revision_fails():
    for revision, quant in [("main", "bf16"), ("a" * 40, "int8")]:
        with pytest.raises(ValueError, match="pinned"):
            load_base("/local/weights", revision, quant)


def test_native_glimmer_forward_contract_is_available(glimmer):
    assert "logits_to_keep" in inspect.signature(glimmer.forward).parameters
