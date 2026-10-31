import inspect
import logging
import re
from typing import Any

_SKIP = ("vision", "visual", "perception", "projector", "multi_modal", "lm_head", "embed")


def _base(model: Any) -> Any:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _decoder(model: Any) -> tuple[Any, str]:
    base = _base(model)
    if hasattr(base.model, "language_model"):
        decoder, path = base.model.language_model, "model.language_model"
    else:
        decoder, path = base.get_decoder(), "model"
    if not hasattr(decoder, "norm"):
        raise ValueError("Cannot identify the decoder's final norm")
    return decoder, path


def _inputs(batch: dict) -> dict:
    import torch

    inputs = {key: batch[key] for key in ("input_ids", "attention_mask", "position_ids")}
    mask = inputs["attention_mask"]
    if (mask.ndim != 2 or not mask.shape[1] or not mask[:, -1].all()
            or not ((mask == 0) | (mask == 1)).all()
            or (mask[:, 1:] < mask[:, :-1]).any()):
        raise ValueError("Inputs require nonempty, left-padded attention masks")
    if any(value.shape != mask.shape for value in inputs.values()):
        raise ValueError("Input ids, masks and position ids must have matching shapes")
    if not torch.equal(inputs["position_ids"], (mask.cumsum(-1) - 1).clamp(min=0)):
        raise ValueError("Position ids must count only unmasked tokens")
    return {**inputs, "use_cache": False, "return_dict": True}


def load_base(path: str, revision: str, quant: str = "bf16") -> tuple[Any, Any, dict]:
    """Load pinned local Glimmer weights with BF16 or explicit NF4 quantization."""
    import torch
    import transformers as tf

    if quant not in {"bf16", "nf4"} or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected bf16/nf4 and a pinned 40-character revision")
    args = {"revision": revision, "local_files_only": True, "dtype": torch.bfloat16,
            "device_map": {"": 0}, "attn_implementation": "sdpa"}
    if quant == "nf4":
        config = tf.AutoConfig.from_pretrained(path, revision=revision, local_files_only=True)
        with torch.device("meta"):
            shape = tf.AutoModelForMultimodalLM.from_config(config)
        skip = [name for name, _ in shape.named_modules()
                if name and any(tag in name for tag in _SKIP[:5])]
        args["quantization_config"] = tf.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["lm_head", *skip])
        del shape
    try:
        model = tf.AutoModelForCausalLM.from_pretrained(path, **args)
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        model = None
    if model is not None and (any(any(tag in name for tag in _SKIP[:3])
                                 for name, _ in model.named_parameters())
                              or not 27e9 <= model.num_parameters() <= 28.6e9):
        del model
        model = None
    if model is None:
        model = tf.AutoModelForMultimodalLM.from_pretrained(path, **args)
    _, prefix = _decoder(model)
    total = model.num_parameters()
    text = total - sum(parameter.numel() for name, parameter in model.named_parameters()
                       if not name.startswith((f"{prefix}.", "lm_head.")))
    if not 27e9 <= text <= 28.6e9:
        raise ValueError(f"Unexpected text decoder parameter count: {text}")
    if quant == "nf4":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.requires_grad_(False)
    tokenizer = tf.AutoTokenizer.from_pretrained(path, revision=revision, local_files_only=True)
    info = {"class": type(model).__name__, "parameters": total, "text_parameters": text,
            "perception_parameters": total - text,
            "dtype": str(model.dtype), "quant": quant, "transformers": tf.__version__,
            "logits_to_keep": "logits_to_keep" in inspect.signature(_base(model).forward).parameters,
            "final_norm": f"{prefix}.norm", "hidden_state_after_norm": True,
            "revision": revision}
    return model, tokenizer, info


def add_lora(model: Any) -> Any:
    """Attach rank-16 LoRA to text attention and MLP projections only."""
    import torch
    from peft import LoraConfig, get_peft_model

    _, prefix = _decoder(model)
    names = sorted({name.rsplit(".", 1)[-1] for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear)
                    and not any(tag in name for tag in _SKIP)})
    required = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if not required.issubset(names):
        raise ValueError(f"Missing text attention or MLP LoRA projections: {required - set(names)}")
    target = rf"{re.escape(prefix)}\..*\.({'|'.join(map(re.escape, names))})$"
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=target))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    trainable, total = model.get_nb_trainable_parameters()
    logging.getLogger(__name__).info("LoRA targets=%s trainable=%d total=%d", names, trainable, total)
    return model


def last_logits(model: Any, batch: dict) -> Any:
    """Compute float32 vocabulary logits for the final answer position only."""
    inputs = _inputs(batch)
    if "logits_to_keep" in inspect.signature(_base(model).forward).parameters:
        return model(**inputs, logits_to_keep=1).logits[:, -1, :].float()
    import torch

    base = _base(model)
    hidden = last_hidden(model, batch)
    logits = base.get_output_embeddings()(hidden)
    config = getattr(base.config, "text_config", base.config)
    logits = logits * getattr(config, "output_multiplier", 1.0)
    cap = getattr(config, "final_logit_softcapping", None)
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits.float()


def option_logits(logits: Any, letter_ids: list[int]) -> Any:
    """Select the six spaced-letter answer logits in alphabetical order."""
    import torch

    if (logits.ndim != 2 or len(letter_ids) != 6 or len(set(letter_ids)) != 6
            or any(type(index) is not int or not 0 <= index < logits.shape[1]
                   for index in letter_ids)):
        raise ValueError("Expected [batch, vocabulary] logits and six distinct valid letter ids")
    return logits.index_select(1, torch.tensor(letter_ids, device=logits.device)).float()


def last_hidden(model: Any, batch: dict) -> Any:
    """Return the last token after the decoder's final norm without vocabulary logits."""
    decoder, _ = _decoder(model)
    return decoder(**_inputs(batch)).last_hidden_state[:, -1, :]
