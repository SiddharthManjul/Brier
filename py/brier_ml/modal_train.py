# Pinned Modal training environment and immutable Glimmer weight download.

import hashlib
import json
import logging
import math
import re
import statistics
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import modal

LOGGER = logging.getLogger(__name__)
APP = modal.App("brier-train")
MODELS = modal.Volume.from_name("brier-models", create_if_missing=True)
DATA = modal.Volume.from_name("brier-data", create_if_missing=True)
HF = modal.Secret.from_name("huggingface")
BASE_REPO = "meta-models/Muse-Glimmer-30B"
BASE_DIR = "/models/base/muse-glimmer-30b"

# Resolved for Linux x86_64 / Python 3.11 on 2026-10-31, separately from LeRobot.
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "modal==1.6.3", "torch==2.15.0", "transformers==5.22.1", "peft==0.21.2",
        "accelerate==1.15.0", "safetensors==0.8.0", "huggingface-hub==2.4.0",
        "bitsandbytes==0.50.2", "numpy==2.4.6", "scipy==1.17.1",
        "scikit-learn==1.9.1", "pytest==9.2.0",
    )
    .env({"HF_HOME": "/models/hf-home", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("brier_ml")
)


def _verify_weights(directory: Path) -> None:
    required = ("config.json", "tokenizer.json", "tokenizer_config.json",
                "model.safetensors.index.json")
    for name in required:
        if not (directory / name).is_file() or (directory / name).stat().st_size == 0:
            raise ValueError(f"Incomplete Glimmer snapshot: missing or empty {name}")
    config = json.loads((directory / "config.json").read_text())
    expected = {
        "model_type": "muse_glimmer_text", "num_hidden_layers": 52,
        "hidden_size": 6656, "num_attention_heads": 32, "num_key_value_heads": 2,
        "head_dim": 128, "intermediate_size": 19968, "hidden_activation": "silu",
        "sliding_window": 2048, "vocab_size": 202048,
        "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
    }
    expected["layer_types"] *= 13
    if (config.get("model_type") != "muse_glimmer"
            or config.get("architectures") != ["MuseGlimmerForConditionalGeneration"]
            or config.get("dtype") != "bfloat16"):
        raise ValueError("Glimmer architecture or weight dtype differs from Guide 2")
    for name, value in expected.items():
        if config.get("text_config", {}).get(name) != value:
            raise ValueError(f"Glimmer text config differs from Guide 2: {name}")
    vision = config.get("vision_config", {})
    if vision.get("patch_size") != 14 or vision.get("hidden_size") != 1536:
        raise ValueError("Glimmer perception encoder differs from Guide 2")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    if not shards:
        raise ValueError("Glimmer shard index has no weights")
    for shard in shards:
        path = directory / shard
        if Path(shard).name != shard or not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Incomplete Glimmer snapshot: missing or empty shard {shard}")


@APP.function(image=IMAGE, cpu=4, memory=16384, timeout=3600,
              secrets=[HF], volumes={"/models": MODELS})
def download_weights() -> None:
    """Download one immutable revision, or verify an existing complete snapshot."""
    from huggingface_hub import model_info, snapshot_download

    directory = Path(BASE_DIR)
    revision = directory / "revision.txt"
    if revision.exists():
        if not re.fullmatch(r"[0-9a-f]{40}", revision.read_text().strip()):
            raise ValueError("Glimmer revision.txt must contain a 40-character commit SHA")
        _verify_weights(directory)
        LOGGER.info("Verified existing Glimmer revision %s", revision.read_text().strip())
        return
    directory.mkdir(parents=True, exist_ok=True)
    sha = model_info(BASE_REPO).sha
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Hugging Face did not return an immutable Glimmer commit SHA")
    revision.write_text(sha + "\n")
    snapshot_download(BASE_REPO, revision=sha, local_dir=BASE_DIR,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                      "*.jinja", "tokenizer*"])
    _verify_weights(directory)
    MODELS.commit()
    LOGGER.info("Downloaded and committed Glimmer revision %s", sha)

RATES = {"bf16": 0.001097, "nf4": 0.000694}  # Guide 2 USD/s; verify current official rates before launch.
_GPU = {"image": IMAGE, "volumes": {"/models": MODELS, "/data": DATA}, "secrets": [HF],
            "env": {"BRIER_DATA": "/data"}, "retries": 0, "max_containers": 1}

@contextmanager
def _attempt(step: str, quant: str, max_usd: float, cap: float):
    logging.basicConfig(level=logging.INFO)
    started = time.perf_counter()
    MODELS.reload()
    if quant not in RATES or not math.isfinite(max_usd) or not 0 < max_usd <= cap:
        raise ValueError("Unknown quantization or per-step budget exceeds its hard cap")
    ledger = Path("/models/runs/spend.json")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(ledger.read_text()) if ledger.exists() else []
    max_usd -= sum(row["spend"] for row in records if row.get("step") == step) if ":" in step else 0
    if max_usd <= 0 or sum(row["spend"] for row in records) + max_usd > 22:
        raise RuntimeError("Training pool $22 cap would be exceeded, including unresolved reservations")
    record = {"id": str(uuid.uuid4()), "step": step, "quant": quant, "status": "reserved", "spend": max_usd}
    records.append(record)
    ledger.write_text(json.dumps(records, allow_nan=False) + "\n")
    MODELS.commit()
    def remaining() -> float:
        value = max_usd - (time.perf_counter() - started) * RATES[quant]
        if value <= 0:
            raise RuntimeError(f"{step} GPU budget cap ${max_usd:g} reached")
        return value
    try:
        yield record, remaining
        record["status"] = record.get("result", {}).get("status", "complete")
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        record.update(seconds=time.perf_counter() - started, max_usd=max_usd)
        record["spend"] = record["seconds"] * RATES[quant]
        ledger.write_text(json.dumps(records, default=lambda value: sorted(value) if isinstance(value, set) else str(value), allow_nan=False) + "\n")
        LOGGER.info("GPU attempt %s", record)
        MODELS.commit()

@APP.function(**_GPU, gpu="H100", timeout=1800)
def check(quant: str = "bf16", max_usd: float = 1.0) -> dict:
    """Check every real-tokenizer boundary and one eight-example Glimmer forward."""
    with _attempt("check", quant, max_usd, 1) as (record, remaining):
        import torch

        from brier_ml import glimmer_data as gd
        from brier_ml import glimmer_model as gm

        revision = (Path(BASE_DIR) / "revision.txt").read_text().strip()
        model, tokenizer, info = gm.load_base(BASE_DIR, revision, quant)
        remaining()
        rows = gd.load_examples("/data/compiled.jsonl", ("train", "calibration", "test", "ood"))
        encoded = gd.tokenize(rows, tokenizer)
        if len(encoded) < 8:
            raise ValueError("Glimmer check requires at least eight compiled examples")
        lengths = [len(row["input_ids"]) for row in encoded]
        remaining()
        batch = {k: v.to(next(model.parameters()).device) for k, v in gd.collate(encoded[:8], encoded[0]["pad_id"]).items()}
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            logits = gm.option_logits(gm.last_logits(model.eval(), batch), gd.letter_token_ids(tokenizer))
        if logits.shape != (8, 6) or not torch.isfinite(logits).all():
            raise ValueError("Glimmer check requires finite [8,6] option logits")
        model = gm.add_lora(model)
        targets = sorted(n for n, module in model.named_modules() if hasattr(module, "lora_A"))
        result = {"info": info, "quant": quant, "revision": revision, "max_length": max(lengths),
                      "median_length": statistics.median(lengths), "examples": len(rows),
                      "peak_memory": torch.cuda.max_memory_reserved(), "lora_targets": targets}
        LOGGER.info("Glimmer check %s", result)
        (Path("/models/runs") / "check.json").write_text(json.dumps(result, allow_nan=False) + "\n")
        remaining()
        record["result"] = result
        return record

@APP.function(**_GPU, gpu="H100", timeout=7200)
def base_pass_fn(max_usd: float = 3.0) -> dict:
    """Run zero-shot scoring with the quantization accepted by the check."""
    MODELS.reload()
    quant = json.loads(Path("/models/runs/check.json").read_text())["quant"]
    with _attempt("base", quant, max_usd, 3) as (record, remaining):
        from brier_ml.score import base_pass
        record["result"] = base_pass("/models/runs/base", max_usd=remaining(), usd_per_sec=RATES[quant])
        return record

@APP.function(**_GPU, gpu="H100", timeout=14400)
def train_fn(run_id: str, quant: str, micro_batch: int, max_steps: int | None,
             epochs: int, max_usd: float, init_adapter: str = "", lr: float = 1e-4) -> dict:
    """Train or resume LoRA after enforcing the smoke gate and shared spend cap."""
    smoke = max_steps == 50
    with _attempt(("smoke:" if smoke else "train:") + run_id, quant, max_usd, 3 if smoke else 12) as (record, remaining):
        from brier_ml.finetune import train
        summary = Path("/models/runs/smoke.json")
        if not smoke:
            proof = json.loads(summary.read_text())
            if (proof["quant"] != quant or proof["micro_batch"] != micro_batch
                    or proof["revision"] != (Path(BASE_DIR) / "revision.txt").read_text().strip()
                    or proof["data_sha256"] != hashlib.sha256(Path("/data/compiled.jsonl").read_bytes()).hexdigest()
                    or not 0 < proof["projected_epoch_cost"] < max_usd
                    or proof["peak_memory"] >= 78e9 or proof["last_10_mean_loss"] >= proof["first_10_mean_loss"]):
                raise ValueError("Full training requires a passing smoke with matching quant and micro-batch")
        remaining()
        record["result"] = train({"run_id": run_id, "quant": quant, "micro_batch": micro_batch,
            "max_steps": max_steps, "epochs": epochs, "max_usd": max_usd, "usd_per_sec": RATES[quant],
            "init_adapter": init_adapter or None, "lr": lr})
        if smoke and record["result"].get("status") != "budget_stop":
            report = json.loads((Path("/models/runs") / run_id / "smoke_report.json").read_text())
            meta = json.loads((Path("/models/runs") / run_id / "run_meta.json").read_text())
            summary.write_text(json.dumps(dict(report, quant=quant, micro_batch=micro_batch, revision=meta["revision"], data_sha256=meta["data_sha256"]), allow_nan=False) + "\n")
        return record

@APP.function(**_GPU, gpu="H100", timeout=3600)
def score_fn(run_id: str, max_usd: float = 2.0) -> dict:
    """Score a trained adapter using its recorded quantization and shared budget."""
    MODELS.reload()
    run = Path("/models/runs") / run_id
    quant = json.loads((run / "run_meta.json").read_text())["quant"]
    with _attempt("score", quant, max_usd, 2) as (record, remaining):
        from brier_ml.score import score_adapter
        record["result"] = score_adapter(run, max_usd=remaining(), usd_per_sec=RATES[quant])
        return record

@APP.local_entrypoint()
def main(step: str, run_id: str = "glimmer-ft-v1", quant: str = "bf16",
         micro_batch: int = 8, max_steps: int = 0, epochs: int = 1,
         max_usd: float = 12.0, init_adapter: str = "", lr: float = 1e-4) -> None:
    """Dispatch one blocking, budget-limited Modal job with the selected GPU."""
    logging.basicConfig(level=logging.INFO)
    if step not in {"download", "check", "base", "smoke", "train", "score"} or quant not in RATES:
        raise ValueError("Unknown training step or quantization")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id) or not math.isfinite(max_usd) or max_usd <= 0:
        raise ValueError("Expected a safe run id and positive finite max_usd")
    if step == "download":
        return download_weights.remote()
    if step in {"base", "score", "smoke", "train"}:
        source = f"runs/{run_id}/run_meta.json" if step == "score" else "runs/check.json"
        selected = json.loads(b"".join(MODELS.read_file(source)))["quant"]
        quant = selected if step in {"base", "score"} or selected == "nf4" else quant
    cap = min(max_usd, 3 if step == "train" and max_steps == 50 else {"check": 1, "base": 3, "smoke": 3, "train": 12, "score": 2}[step])
    function = {"check": check, "base": base_pass_fn, "smoke": train_fn, "train": train_fn, "score": score_fn}[step]
    try:
        records = json.loads(b"".join(MODELS.read_file("runs/spend.json")))
    except FileNotFoundError:
        records = []
    spent = sum(row["spend"] for row in records)
    budget = cap - sum(row["spend"] for row in records if row.get("step") == f'{"smoke" if step == "train" and max_steps == 50 else step}:{run_id}') if step in {"smoke", "train"} else cap
    if budget <= 0 or spent >= 22 or (step not in {"smoke", "train"} and spent + cap > 22):
        raise RuntimeError("Training pool $22 cap would be exceeded before GPU launch")
    timeout = min({"check": 1800, "base": 7200, "smoke": 14400, "train": 14400, "score": 3600}[step], math.floor(min(budget, 22 - spent) / RATES[quant]))
    if timeout < 1: raise ValueError("Budget is too small for one billed GPU second")
    function = function.with_options(gpu="A100-80GB" if quant == "nf4" else "H100", timeout=timeout)
    args = (quant, cap) if step == "check" else ((cap,) if step == "base" else ((run_id, cap) if step == "score" else
        (run_id, quant, micro_batch, 50 if step == "smoke" else max_steps or None, epochs, cap, init_adapter, lr)))
    LOGGER.info("Modal result %s", function.remote(*args))
