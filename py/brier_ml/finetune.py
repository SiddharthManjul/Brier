from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import modal
import numpy as np
import torch
from torch.nn import functional as F

_VALUES = {
    "regressed": {"yes", "no"},
    "failure_mode": {"none", "stall/timeout", "overshoot/misalignment",
                     "grasp or drop failure", "collision", "oscillation"},
    "success_bucket": {"1", "2", "3", "4", "5"},
}
_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def option_loss(
    logits: torch.Tensor, target: int, question: str, option_values: list[str]
) -> torch.Tensor:
    """Compute answer-position cross entropy plus Brier or ordinal ranked probability loss."""
    if (
        question not in _VALUES or logits.ndim != 1
        or len(option_values) != len(_VALUES[question])
        or set(option_values) != _VALUES[question]
        or logits.numel() != len(option_values)
        or type(target) is not int or not 0 <= target < logits.numel()
        or not bool(torch.isfinite(logits).all())
    ):
        raise ValueError("invalid finite option logits, target or question semantics")
    scores = logits.float()
    probability = scores.softmax(dim=0)
    truth = torch.zeros_like(probability)
    truth[target] = 1.0
    ce = F.cross_entropy(scores.unsqueeze(0), torch.tensor([target], device=scores.device))
    if question == "success_bucket":
        order = sorted(range(len(option_values)), key=lambda i: int(option_values[i]))
        proper = ((probability[order].cumsum(0) - truth[order].cumsum(0))[:-1] ** 2).mean()
    else:
        proper = ((probability - truth) ** 2).sum()
    return ce + proper


def load_training_data(compiled: Path) -> list[dict[str, Any]]:
    """Validate Rust-compiled records and retain only change-disjoint training rows."""
    rows, ids, splits = [], set(), {}
    for number, line in enumerate(compiled.read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid compiled JSON at line {number}") from error
        if not isinstance(row, dict) or set(row) != {
            "id", "split", "question", "options", "prompt", "label"
        }:
            raise ValueError(f"invalid compiled fields at line {number}")
        identifier, question = row["id"], row["question"]
        if not isinstance(identifier, str) or not re.fullmatch(
            r"c[0-9]{3,}/(?:pusht|aloha_transfer_cube)-s[0-9]{2,}/"
            r"(?:regressed|failure_mode|success_bucket)/(?:cheap|no_cheap)", identifier
        ):
            raise ValueError(f"invalid compiled id at line {number}")
        change, scenario, id_question, _ = identifier.split("/")
        options, prompt = row["options"], row["prompt"]
        if (
            identifier in ids or int(change[1:]) == 0 or int(scenario.rsplit("s", 1)[1]) == 0
            or question != id_question or not isinstance(row["split"], str)
            or row["split"] not in {"train", "calibration", "test", "ood"}
            or not isinstance(options, dict) or not all(isinstance(v, str) for v in options.values())
            or set(options) != set("ABCDEF"[:len(_VALUES[question])])
            or set(options.values()) != _VALUES[question]
            or not isinstance(row["label"], str) or row["label"] not in options
            or not isinstance(prompt, str) or not prompt.startswith("<system> ")
            or not prompt.endswith("<answer>")
            or any(marker not in prompt for marker in ("\n<state>\n", "\n<question> ", "\n<options> "))
        ):
            raise ValueError(f"invalid compiled contract at line {number}")
        if change in splits and splits[change] != row["split"]:
            raise ValueError(f"change {change} appears in multiple splits")
        ids.add(identifier)
        splits[change] = row["split"]
        if row["split"] == "train":
            rows.append(row)
    if not rows:
        raise ValueError("compiled data contains no training examples")
    return rows


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _encode(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    encoded = []
    for row in rows:
        tokens = tokenizer.encode(row["prompt"], add_special_tokens=False)
        letters = sorted(row["options"])
        option_tokens = [tokenizer.encode(letter, add_special_tokens=False) for letter in letters]
        if not tokens or len(tokens) > 1535:
            raise ValueError(f"compiled prompt must contain 1..1535 tokens, reserving one answer token: {row['id']}")
        if any(len(value) != 1 for value in option_tokens) or len({tuple(v) for v in option_tokens}) != len(letters):
            raise ValueError("option letters must have distinct atomic single-token encodings")
        if any(tokenizer.encode(row["prompt"] + letter, add_special_tokens=False) != tokens + value
               for letter, value in zip(letters, option_tokens, strict=True)):
            raise ValueError("option tokenization changes at the compiled answer boundary")
        encoded.append({"tokens": tokens, "option_tokens": [v[0] for v in option_tokens],
                        "target": letters.index(row["label"]), "question": row["question"],
                        "values": [row["options"][letter] for letter in letters]})
    return encoded


def _load_model(identifier: str, revision: str) -> tuple[Any, Any]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    config = AutoConfig.from_pretrained(identifier, revision=revision, trust_remote_code=False)
    if getattr(config, "vision_config", None) is not None:
        raise ValueError("MODAL_MODEL_ID must name a supported text-decoder-only checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(identifier, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        identifier, revision=revision, config=config, trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()},
    )
    suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if not set(_PROJECTIONS).issubset(suffixes):
        raise ValueError("text decoder does not expose all required attention and MLP projections")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=_PROJECTIONS,
        bias="none", task_type="CAUSAL_LM",
    ))
    return model, tokenizer


def _save(
    model: Any, optimizer: Any, scheduler: Any, directory: Path,
    config: dict[str, Any], step: int, cursor: int, commit: Callable[[], None],
) -> None:
    state = {
        "config": config, "step": step, "cursor": cursor,
        "parameters": {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
    }
    model.save_pretrained(directory / "adapter")
    temporary = directory / "training-state.tmp"
    torch.save(state, temporary)
    temporary.replace(directory / "training-state.pt")
    commit()


def _fit(
    model: Any, encoded: list[dict[str, Any]], directory: Path,
    config: dict[str, Any], smoke: bool, commit: Callable[[], None],
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    total = 50 if smoke else math.ceil(len(encoded) / 32)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    warmup = math.ceil(total * 0.03)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min((step - warmup) / max(1, total - warmup), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    step = cursor = 0
    path = directory / "training-state.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError("checkpoint base configuration, dataset digest or smoke mode differs")
        expected = {name for name, value in model.named_parameters() if value.requires_grad}
        if set(state["parameters"]) != expected:
            raise ValueError("checkpoint trainable parameter names differ")
        step, cursor = state["step"], state["cursor"]
        if type(step) is not int or not 0 <= step <= total or cursor != (step * 32 if smoke else min(step * 32, len(encoded))):
            raise ValueError("checkpoint optimizer step and dataset offset are inconsistent")
        model.load_state_dict(state["parameters"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    first_step, tokens, started = step, 0, time.perf_counter()
    device = next(model.parameters()).device
    model.train()
    while step < total:
        count = 32 if smoke else min(32, len(encoded) - cursor)
        optimizer.zero_grad(set_to_none=True)
        for offset in range(count):
            row = encoded[(cursor + offset) % len(encoded)]
            inputs = torch.tensor([row["tokens"]], dtype=torch.long, device=device)
            logits = model(input_ids=inputs, use_cache=False).logits[0, -1, row["option_tokens"]]
            loss = option_loss(logits, row["target"], row["question"], row["values"]) / count
            loss.backward()
            tokens += len(row["tokens"])
        optimizer.step()
        scheduler.step()
        step += 1
        cursor += count
        if step % 200 == 0:
            _save(model, optimizer, scheduler, directory, config, step, cursor, commit)
    _save(model, optimizer, scheduler, directory, config, step, cursor, commit)
    seconds = time.perf_counter() - started
    result = {"optimizer_steps": step, "optimizer_steps_this_run": step - first_step,
              "training_examples": len(encoded), "tokens": tokens, "seconds": seconds,
              "tokens_per_second": tokens / seconds, "checkpoint": str(directory)}
    report = directory / "training-report.tmp"
    report.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    report.replace(directory / "training-report.json")
    commit()
    return result


def train(compiled: Path, smoke: bool = False) -> dict[str, Any]:
    """Train or resume a text-decoder QLoRA adapter using verbatim compiled prompts."""
    identifier, revision = _required("MODAL_MODEL_ID"), _required("MODAL_MODEL_REVISION")
    directory = Path(_required("MODAL_CHECKPOINT_DIR")).expanduser().resolve()
    data = _required("BRIER_DATA")
    volume_name = _required("MODAL_VOLUME_NAME")
    if type(smoke) is not bool or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("smoke must be boolean and MODAL_MODEL_REVISION must be an immutable 40-hex commit")
    if "://" in data or not Path(data).is_dir():
        raise ValueError("BRIER_DATA must be an existing local directory on the Modal Volume; transfer gs:// data first")
    if not directory.is_relative_to("/checkpoints") or not Path(data).resolve().is_relative_to("/checkpoints"):
        raise ValueError("BRIER_DATA and MODAL_CHECKPOINT_DIR must reside under the /checkpoints Volume mount")
    if not compiled.resolve().is_relative_to(Path(data).resolve()):
        raise ValueError("compiled JSONL must reside inside BRIER_DATA")
    rows = load_training_data(compiled)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("QLoRA requires a CUDA GPU supporting bfloat16")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model, tokenizer = _load_model(identifier, revision)
    encoded = _encode(rows, tokenizer)
    config = {"model_id": identifier, "revision": revision, "smoke": smoke,
              "dataset_digest": hashlib.sha256(compiled.read_bytes()).hexdigest(),
              "base_config": model.config.to_dict(),
              "recipe": {"batch": 32, "sequence": 1536, "epochs": 1, "lr": 1e-4,
                         "optimizer": "AdamW", "schedule": "cosine", "warmup": 0.03,
                         "seed": 0, "proper_loss_weight": 1.0,
                         "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                         "projections": _PROJECTIONS, "quantization": "nf4", "compute": "bf16"}}
    volume = modal.Volume.from_name(volume_name)
    return _fit(model, encoded, directory, config, smoke, volume.commit)


app = modal.App("brier-finetune")
_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "transformers==4.57.1", "peft==0.17.1",
                 "accelerate==1.10.1", "bitsandbytes==0.49.2", "numpy==2.2.6", "modal==1.6.3")
    .add_local_python_source("brier_ml")
)


@app.function(image=_image, gpu="A100-80GB", timeout=86400)
def _remote_train(compiled: str, smoke: bool = False) -> dict[str, Any]:
    return train(Path(compiled), smoke)


@app.local_entrypoint()
def _launch(compiled: str, smoke: bool = False) -> None:
    names = ("MODAL_MODEL_ID", "MODAL_MODEL_REVISION", "MODAL_CHECKPOINT_DIR", "BRIER_DATA", "MODAL_VOLUME_NAME")
    environment = {name: _required(name) for name in names}
    volume = modal.Volume.from_name(environment["MODAL_VOLUME_NAME"], create_if_missing=True)
    _remote_train.with_options(env=environment, volumes={"/checkpoints": volume}).remote(compiled, smoke)
