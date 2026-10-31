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


def _text_config(config: Any) -> Any:
    if getattr(config, "model_type", None) == "muse_glimmer":
        text = getattr(config, "text_config", None)
        if (
            getattr(config, "architectures", None) != ["MuseGlimmerForConditionalGeneration"]
            or getattr(text, "model_type", None) != "muse_glimmer_text"
            or getattr(getattr(config, "vision_config", None), "model_type", None) != "muse_glimmer_vision"
            or getattr(config, "is_encoder_decoder", False)
            or getattr(text, "is_encoder_decoder", False)
        ):
            raise ValueError("Muse Glimmer requires its native conditional-generation and text-decoder configuration")
        return text
    if getattr(config, "is_encoder_decoder", False) or getattr(config, "vision_config", None) is not None:
        raise ValueError("model must expose a supported text decoder or native Muse Glimmer text path")
    return config


def _load_model(identifier: str, revision: str) -> tuple[Any, Any]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    config = AutoConfig.from_pretrained(identifier, revision=revision, trust_remote_code=False)
    text_config = _text_config(config)
    native_glimmer = getattr(config, "model_type", None) == "muse_glimmer"
    model_class = AutoModelForCausalLM
    if native_glimmer:
        from transformers import MuseGlimmerForConditionalGeneration

        model_class = MuseGlimmerForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(identifier, revision=revision, trust_remote_code=False)
    model = model_class.from_pretrained(
        identifier, revision=revision, config=config, trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()},
    )
    targets = _PROJECTIONS
    if native_glimmer:
        targets = [name for name, _ in model.named_modules() if re.fullmatch(
            r"model\.language_model\.layers\.[0-9]+\.(?:self_attn|mlp)\." + "(?:" + "|".join(_PROJECTIONS) + ")", name
        )]
    suffixes = {name.rsplit(".", 1)[-1] for name in targets} if native_glimmer else {
        name.rsplit(".", 1)[-1] for name, _ in model.named_modules()
    }
    if not set(_PROJECTIONS).issubset(suffixes):
        raise ValueError("text decoder does not expose all required attention and MLP projections")
    config.use_cache = text_config.use_cache = model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets,
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
    manifest = directory / "adapter" / "brier-model.json"
    manifest_temp = manifest.with_suffix(".json.tmp")
    manifest_temp.write_text(json.dumps({"config": config, "step": step}, sort_keys=True) + "\n")
    manifest_temp.replace(manifest)
    temporary = directory / "training-state.tmp"
    torch.save(state, temporary)
    temporary.replace(directory / "training-state.pt")
    commit()


def _fit(
    model: Any, encoded: list[dict[str, Any]], directory: Path,
    config: dict[str, Any], smoke: bool, commit: Callable[[], None],
    *, teacher_distributions: list[list[float]] | None = None,
) -> dict[str, Any]:
    if teacher_distributions is not None:
        if not isinstance(teacher_distributions, list) or len(teacher_distributions) != len(encoded) or any(
            not isinstance(values, list) or len(values) != len(row["option_tokens"])
            or any(type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value)
                   for value in values)
            or not math.isclose(sum(values), 1.0, abs_tol=1e-6, rel_tol=0)
            for row, values in zip(encoded, teacher_distributions, strict=True)
        ):
            raise ValueError("teacher distributions must match encoded options and sum to one")
        digest = hashlib.sha256(json.dumps(teacher_distributions, allow_nan=False).encode()).hexdigest()
        config = {**config, "distillation_digest": digest}
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
            index = (cursor + offset) % len(encoded)
            row = encoded[index]
            inputs = torch.tensor([row["tokens"]], dtype=torch.long, device=device)
            logits = model(input_ids=inputs, use_cache=False).logits[0, -1, row["option_tokens"]]
            loss = option_loss(logits, row["target"], row["question"], row["values"])
            if teacher_distributions is not None:
                teacher = torch.tensor(teacher_distributions[index], dtype=torch.float32, device=device)
                loss = loss + F.kl_div(F.log_softmax(logits.float(), dim=0), teacher, reduction="sum")
            (loss / count).backward()
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


def _training_paths(compiled: Path, smoke: bool, revision: str) -> tuple[Path, str]:
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
    return directory, volume_name


def _seed_training() -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("QLoRA requires a CUDA GPU supporting bfloat16")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def _training_config(identifier: str, revision: str, compiled: Path, model: Any, smoke: bool) -> dict[str, Any]:
    return {"model_id": identifier, "revision": revision, "smoke": smoke,
              "dataset_digest": hashlib.sha256(compiled.read_bytes()).hexdigest(),
              "base_config": model.config.to_dict(),
              "recipe": {"batch": 32, "sequence": 1536, "epochs": 1, "lr": 1e-4,
                         "optimizer": "AdamW", "schedule": "cosine", "warmup": 0.03,
                         "seed": 0, "proper_loss_weight": 1.0,
                         "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                         "projections": _PROJECTIONS, "quantization": "nf4", "compute": "bf16"}}


def train(compiled: Path | dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    """Train or resume a text-decoder QLoRA adapter using verbatim compiled prompts."""
    if isinstance(compiled, dict):
        return _guide_train(compiled)
    identifier, revision = _required("MODAL_MODEL_ID"), _required("MODAL_MODEL_REVISION")
    directory, volume_name = _training_paths(compiled, smoke, revision)
    rows = load_training_data(compiled)
    _seed_training()
    model, tokenizer = _load_model(identifier, revision)
    encoded = _encode(rows, tokenizer)
    config = _training_config(identifier, revision, compiled, model, smoke)
    volume = modal.Volume.from_name(volume_name)
    return _fit(model, encoded, directory, config, smoke, volume.commit)


def _guide_train(cfg: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    import importlib.metadata
    import logging
    import shutil

    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    from brier_ml import glimmer_data as gd
    from brier_ml import glimmer_model as gm
    from brier_ml.losses import option_probs, proper_loss
    from brier_ml.modal_train import BASE_DIR, BASE_REPO, MODELS

    logger = logging.getLogger(__name__)
    defaults = {"effective_batch": 32, "lr": 1e-4, "warmup_frac": 0.03, "epochs": 1,
                    "max_steps": None, "ckpt_every": 50, "eval_every": 50, "seed": 2026, "init_adapter": None}
    if set(cfg) - (set(defaults) | {"run_id", "quant", "micro_batch", "max_usd", "usd_per_sec"}):
        raise ValueError("unknown training configuration fields")
    cfg = defaults | cfg
    required = {"run_id", "quant", "micro_batch", "max_usd", "usd_per_sec"}
    if not required <= cfg.keys() or not isinstance(cfg["run_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", cfg["run_id"]):
        raise ValueError("training requires run_id, quant, micro_batch, max_usd and usd_per_sec")
    if (cfg["quant"] not in {"bf16", "nf4"}
            or any(type(cfg[k]) is not int or cfg[k] < 1 for k in
                   ("micro_batch", "effective_batch", "epochs", "ckpt_every", "eval_every"))
            or cfg["effective_batch"] % cfg["micro_batch"] or type(cfg["seed"]) is not int or cfg["seed"] < 0
            or (cfg["max_steps"] is not None and (type(cfg["max_steps"]) is not int or cfg["max_steps"] < 1))
            or any(type(cfg[k]) not in (int, float) or not math.isfinite(cfg[k]) or cfg[k] <= 0
                   for k in ("lr", "max_usd", "usd_per_sec"))
            or not 0 <= cfg["warmup_frac"] <= 1
            or (cfg["init_adapter"] is not None and not isinstance(cfg["init_adapter"], str))):
        raise ValueError("invalid finite training configuration or accumulation count")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    path = Path(_required("BRIER_DATA")) / "compiled.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    revision = (Path(BASE_DIR) / "revision.txt").read_text().strip()
    model, tokenizer, info = gm.load_base(BASE_DIR, revision, cfg["quant"])
    model = gm.add_lora(model)
    rows = gd.load_examples(path, ["train", "calibration"])
    train_rows = gd.tokenize([r for r in rows if r["split"] == "train"], tokenizer)
    evaluation = sorted((r for r in rows if r["split"] == "calibration"
                         and r["question"] == "regressed" and r["variant"] == "ev"), key=lambda r: r["id"])[:256]
    eval_rows = gd.tokenize(evaluation, tokenizer)
    letters, pad = gd.letter_token_ids(tokenizer), train_rows[0]["pad_id"]
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.999),
                                 weight_decay=0.0, fused=device.type == "cuda")
    per_epoch = math.ceil(len(train_rows) / cfg["effective_batch"])
    total = cfg["max_steps"] or per_epoch * cfg["epochs"]
    warmup = math.ceil(cfg["warmup_frac"] * total)

    def factor(s: int) -> float:
        return (s + 1) / warmup if s < warmup else (
            0.5 * (1 + math.cos(math.pi * min(1, (s - warmup + 1) / max(1, total - warmup)))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    run = Path(BASE_DIR).parent.parent / "runs" / cfg["run_id"]
    run.mkdir(parents=True, exist_ok=True)
    log = run / "train_log.jsonl"
    step = epoch = cursor = tokens = examples = 0
    prior = 0.0
    best, best_step, last_eval = None, None, None
    def restore_adapter(location: str) -> None:
        weights = load_peft_weights(location, device="cpu")
        expected = get_peft_model_state_dict(model)
        if weights.keys() != expected.keys() or any(weights[k].shape != expected[k].shape for k in expected):
            raise ValueError("checkpoint adapter parameter names or shapes differ")
        set_peft_model_state_dict(model, weights)

    checkpoints = sorted(p for p in run.glob("ckpt-*") if re.fullmatch(r"ckpt-\d{6}", p.name))
    if checkpoints:
        checkpoint = checkpoints[-1]
        state = json.loads((checkpoint / "state.json").read_text())
        if state["config"] != cfg or state["data_hash"] != digest or state["revision"] != revision:
            raise ValueError("checkpoint configuration, dataset hash or base revision differs")
        step, epoch, cursor = (state[k] for k in ("step", "epoch", "cursor"))
        accum = cfg["effective_batch"] // cfg["micro_batch"]
        if (any(type(n) is not int or n < 0 for n in (step, epoch, cursor)) or step > total
                or not 0 <= step - epoch * per_epoch <= per_epoch
                or cursor != min((step - epoch * per_epoch) * accum, math.ceil(len(train_rows) / cfg["micro_batch"]))):
            raise ValueError("checkpoint step, epoch or micro-batch cursor is inconsistent")
        restore_adapter(str(checkpoint))
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=device, weights_only=True))
        scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", weights_only=True))
        rng = state["rng"]
        random.setstate((rng["python"][0], tuple(rng["python"][1]), rng["python"][2]))
        ns = rng["numpy"]
        np.random.set_state((ns[0], np.array(ns[1], dtype="uint32"), *ns[2:]))
        torch.set_rng_state(torch.tensor(rng["torch"], dtype=torch.uint8))
        if rng["cuda"]:
            torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in rng["cuda"]])
        prior, tokens, examples = (state[k] for k in ("seconds", "tokens", "examples"))
        best, best_step, last_eval = (state[k] for k in ("best_eval_nll", "best_step", "eval"))
        if log.exists():
            lines = log.read_text().splitlines(keepends=True)
            retained = []
            for i, line in enumerate(lines):
                try:
                    if json.loads(line)["step"] <= step:
                        retained.append(line.rstrip("\n") + "\n")
                except json.JSONDecodeError:
                    if i != len(lines) - 1 or line.endswith("\n"):
                        raise
            log.write_text("".join(retained))
    elif cfg["init_adapter"]:
        restore_adapter(cfg["init_adapter"])
    model.train()

    def seconds() -> float:
        return prior + time.perf_counter() - started

    def batch(selected: list[dict]) -> dict:
        return {k: v.to(device) for k, v in gd.collate(selected, pad).items()}

    def logits(b: dict) -> Any:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return gm.option_logits(gm.last_logits(model, b), letters)

    def evaluate() -> dict:
        model.eval()
        nll = brier = 0.0
        with torch.no_grad():
            for start in range(0, len(eval_rows), cfg["micro_batch"]):
                b = batch(eval_rows[start:start + cfg["micro_batch"]])
                p = option_probs(logits(b), b["valid"])
                truth = F.one_hot(b["label_idx"], 6)
                nll += (-p.gather(1, b["label_idx"][:, None]).clamp_min(1e-12).log()).sum().item()
                brier += ((p - truth) ** 2).sum().item()
        model.train()
        result = {"nll": nll / len(eval_rows), "brier": brier / len(eval_rows)}
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError(f"nonfinite evaluation at step {step}")
        return result

    def save() -> None:
        dest = run / f"ckpt-{step:06d}"
        temporary = run / f".ckpt-{step:06d}.tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        model.save_pretrained(temporary)
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        ns = np.random.get_state()
        rng = {"python": random.getstate(), "numpy": [ns[0], ns[1].tolist(), *ns[2:]],
               "torch": torch.get_rng_state().tolist(),
               "cuda": [s.tolist() for s in torch.cuda.get_rng_state_all()] if device.type == "cuda" else []}
        state = {"step": step, "epoch": epoch, "cursor": cursor, "rng": rng, "config": cfg, "data_hash": digest,
                     "revision": revision, "seconds": seconds(), "tokens": tokens, "examples": examples,
                     "best_eval_nll": best, "best_step": best_step, "eval": last_eval}
        (temporary / "state.json").write_text(json.dumps(state, allow_nan=False) + "\n")
        if dest.exists():
            shutil.rmtree(dest)
        temporary.rename(dest)
        existing = sorted(p for p in run.glob("ckpt-*") if re.fullmatch(r"ckpt-\d{6}", p.name))
        keep = set(existing[-2:]) | {run / f"ckpt-{best_step:06d}" if best_step is not None else dest}
        for old in set(existing) - keep:
            shutil.rmtree(old)
        MODELS.commit()

    accum = cfg["effective_batch"] // cfg["micro_batch"]
    lengths = [len(r["input_ids"]) for r in train_rows]
    while step < total:
        if seconds() * cfg["usd_per_sec"] >= cfg["max_usd"]:
            save()
            logger.warning("Budget stop at step %d: elapsed %.2fs, spend $%.4f", step, seconds(), seconds() * cfg["usd_per_sec"])
            return {"status": "budget_stop", "step": step, "seconds": seconds(), "spend": seconds() * cfg["usd_per_sec"]}
        order = gd.length_grouped_order(lengths, cfg["micro_batch"], cfg["seed"] + epoch)
        batches = [order[i:i + cfg["micro_batch"]] for i in range(0, len(order), cfg["micro_batch"])]
        if cursor == len(batches):
            epoch, cursor = epoch + 1, 0
            continue
        selected = batches[cursor:cursor + accum]
        optimizer.zero_grad(set_to_none=True)
        loss_sum = ce_sum = extra_sum = 0.0
        tick, used_tokens, used_examples = time.perf_counter(), 0, 0
        for indices in selected:
            source = [train_rows[i] for i in indices]
            b = batch(source)
            scores = logits(b)
            if not torch.isfinite(scores).all():
                raise FloatingPointError(f"nonfinite logits at step {step + 1}: {[r['id'] for r in source]}")
            loss, parts = proper_loss(scores, b["valid"], b["label_idx"], b["k"].eq(5), b["ordinal_map"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step + 1}: {[r['id'] for r in source]}")
            (loss / len(selected)).backward()
            loss_sum += loss.item() / len(selected)
            ce_sum += parts["ce"].item() / len(selected)
            extra_sum += parts["extra"].item() / len(selected)
            used_tokens += int(b["attention_mask"].sum())
            used_examples += len(source)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"nonfinite gradient at step {step + 1}: {[train_rows[i]['id'] for ids in selected for i in ids]}")
        optimizer.step()
        scheduler.step()
        step, cursor = step + 1, cursor + len(selected)
        tokens, examples = tokens + used_tokens, examples + used_examples
        duration = time.perf_counter() - tick
        record = {"step": step, "loss": loss_sum, "ce": ce_sum, "extra": extra_sum, "lr": optimizer.param_groups[0]["lr"],
                      "grad_norm": float(norm), "examples_per_second": used_examples / duration,
                      "tokens_per_second": used_tokens / duration, "step_seconds": duration,
                      "peak_memory": torch.cuda.max_memory_reserved() if device.type == "cuda" else 0,
                      "elapsed_seconds": seconds(), "spend": seconds() * cfg["usd_per_sec"]}
        if step % cfg["eval_every"] == 0 or step == total:
            last_eval = evaluate()
            record["eval"] = last_eval
            if best is None or last_eval["nll"] < best:
                best, best_step = last_eval["nll"], step
        with log.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        logger.info("Training %s", record)
        if step % cfg["ckpt_every"] == 0 or step == total or best_step == step:
            save()
    model.save_pretrained(run / "adapter")
    manifest = run / "adapter" / "brier-model.json"
    manifest_temp = manifest.with_suffix(".json.tmp")
    manifest_temp.write_text(json.dumps({"config": {"model_id": BASE_REPO, "revision": revision,
                             "dataset_digest": digest, "recipe": cfg}, "step": step}, allow_nan=False) + "\n")
    manifest_temp.replace(manifest)
    meta = dict(info, base_repo=BASE_REPO, revision=revision, config=cfg, steps=step, epochs=epoch + 1,
                data_sha256=digest, letter_token_ids=letters, pad_id=pad, pad_source=train_rows[0]["pad_source"],
                lora_config={k: v.to_dict() for k, v in model.peft_config.items()},
                target_names=sorted(n for n, module in model.named_modules() if hasattr(module, "lora_A")), final_eval_nll=last_eval["nll"],
                tokens_per_second=tokens / seconds(), gpu_type=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                seconds=seconds(), spend=seconds() * cfg["usd_per_sec"],
                versions={name: importlib.metadata.version(name) for name in
                          ("torch", "transformers", "peft", "accelerate", "safetensors", "huggingface_hub", "numpy", "modal")})
    (run / "run_meta.json").write_text(json.dumps(meta, default=lambda v: sorted(v) if isinstance(v, set) else str(v), allow_nan=False) + "\n")
    if cfg["max_steps"] is not None:
        records = [json.loads(line) for line in log.read_text().splitlines()]
        measured = records[5:]
        median = float(np.median([r["step_seconds"] for r in measured])) if measured else None
        report = {"median_seconds_per_step": median, "tokens_per_second": meta["tokens_per_second"],
                      "peak_memory": max(r["peak_memory"] for r in records),
                      "first_10_mean_loss": float(np.mean([r["loss"] for r in records[:10]])),
                      "last_10_mean_loss": float(np.mean([r["loss"] for r in records[-10:]])),
                      "projected_epoch_cost": None if median is None else per_epoch * median * cfg["usd_per_sec"]}
        (run / "smoke_report.json").write_text(json.dumps(report, allow_nan=False) + "\n")
    MODELS.commit()
    return meta


app = modal.App("brier-finetune")
_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "transformers==5.15.0", "peft==0.19.1",
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
