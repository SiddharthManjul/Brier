from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from brier_ml.finetune import _text_config, load_training_data, option_loss

_VALUES = {
    "regressed": ["no", "yes"],
    "failure_mode": [
        "none", "stall/timeout", "overshoot/misalignment",
        "grasp or drop failure", "collision", "oscillation",
    ],
    "success_bucket": ["1", "2", "3", "4", "5"],
}
_SPLITS = {"train", "calibration", "test", "ood"}
_ID = re.compile(
    r"(c[0-9]{3,})/(?:pusht|aloha_transfer_cube)-s([0-9]{2,})/"
    r"(regressed|failure_mode|success_bucket)/(?:cheap|no_cheap)"
)
_EMBEDDINGS = "brier-frozen-embeddings-v1"
_HEAD = "brier-frozen-head-v1"


def _path(path: Path) -> Path:
    root = os.environ.get("BRIER_DATA", "")
    if not root or "://" in root or not Path(root).is_dir():
        raise ValueError("BRIER_DATA must name an existing local data directory")
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError("frozen-head paths must be inside BRIER_DATA")
    return resolved


def _output(path: Path) -> Path:
    resolved = _path(path)
    if resolved.exists():
        raise FileExistsError(f"output already exists: {resolved}")
    return resolved


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)


def _read_embeddings(path: Path) -> dict[str, Any]:
    payload = torch.load(_path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != _EMBEDDINGS:
        raise ValueError("invalid frozen-embedding artifact")
    records, vectors = payload.get("records"), payload.get("embeddings")
    if (
        not isinstance(records, list) or not records
        or not isinstance(vectors, torch.Tensor) or vectors.ndim != 2
        or vectors.shape[0] != len(records) or vectors.shape[1] == 0
        or not vectors.is_floating_point() or not torch.isfinite(vectors).all()
    ):
        raise ValueError("embeddings must be a finite matrix matching the records")
    if not isinstance(payload.get("model_version"), str) or not re.fullmatch(
        r".+@[0-9a-f]{40}", payload["model_version"]
    ):
        raise ValueError("embedding artifact is missing its pinned model version")
    if re.fullmatch(r"[0-9a-f]{64}", str(payload.get("source_sha256"))) is None:
        raise ValueError("embedding artifact is missing its source digest")
    seen: set[str] = set()
    splits: dict[str, str] = {}
    for row in records:
        if not isinstance(row, dict) or set(row) != {
            "id", "split", "question", "options", "label"
        }:
            raise ValueError("invalid embedded-record fields")
        if not all(isinstance(row[key], str)
                   for key in ("id", "split", "question", "label")):
            raise ValueError("invalid embedded-record metadata")
        match = _ID.fullmatch(row["id"])
        if (
            match is None or match.group(3) != row["question"]
            or int(match.group(1)[1:]) == 0 or int(match.group(2)) == 0
            or row["id"] in seen or row["split"] not in _SPLITS
        ):
            raise ValueError("invalid compiled record ID or split")
        options, expected = row["options"], _VALUES[row["question"]]
        if (
            not isinstance(options, dict)
            or set(options) != set("ABCDEF"[:len(expected)])
            or not all(isinstance(value, str) for value in options.values())
            or set(options.values()) != set(expected) or row["label"] not in options
        ):
            raise ValueError("invalid semantic options or label")
        change = match.group(1)
        if change in splits and splits[change] != row["split"]:
            raise ValueError(f"change belongs to multiple splits: {change}")
        splits[change] = row["split"]
        seen.add(row["id"])
    payload["embeddings"] = vectors.float()
    return payload


def _heads(dimension: int, values: dict[str, list[str]]) -> nn.ModuleDict:
    return nn.ModuleDict({
        question: nn.Sequential(
            nn.Linear(dimension, 128), nn.ReLU(), nn.Linear(128, len(values[question]))
        ) for question in _VALUES
    })


def extract_embeddings(compiled: Path, out: Path) -> Path:
    """Save frozen last-token embeddings for every validated Rust-compiled record."""
    compiled, out = _path(compiled), _output(out)
    load_training_data(compiled)
    raw = compiled.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    model_id = os.environ.get("MODAL_MODEL_ID", "").strip()
    revision = os.environ.get("MODAL_MODEL_REVISION", "")
    if not model_id or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("provide MODAL_MODEL_ID and a 40-hex MODAL_MODEL_REVISION")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("frozen extraction requires the Transformers model runtime") from error

    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    text_config = _text_config(config)
    model_class = AutoModelForCausalLM
    if getattr(config, "model_type", None) == "muse_glimmer":
        from transformers import MuseGlimmerForConditionalGeneration

        model_class = MuseGlimmerForConditionalGeneration
    context = getattr(text_config, "max_position_embeddings", 1536)
    if not isinstance(context, int) or context < 2:
        raise ValueError("model must declare a usable context length")
    limit = min(1535, context - 1)
    config.use_cache = text_config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model = model_class.from_pretrained(
        model_id, revision=revision, config=config, trust_remote_code=False,
        dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    model.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    if device.type == "meta":
        raise ValueError("model input embeddings were not materialized")
    vectors, metadata = [], []
    with torch.no_grad():
        for row in records:
            inputs = tokenizer(row["prompt"], add_special_tokens=False, return_tensors="pt")
            if not 1 <= inputs["input_ids"].shape[-1] <= limit:
                raise ValueError(f"prompt exceeds the model input limit: {row['id']}")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            result = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
            vector = result.hidden_states[-1][0, -1].detach().float().cpu()
            if vector.ndim != 1 or not torch.isfinite(vector).all():
                raise ValueError(f"invalid model embedding: {row['id']}")
            vectors.append(vector)
            metadata.append({key: row[key] for key in ("id", "split", "question", "options", "label")})
    _save(out, {
        "format": _EMBEDDINGS, "model_version": f"{model_id}@{revision}",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "records": metadata, "embeddings": torch.stack(vectors),
    })
    return out


def train_head(embeddings: Path, out: Path) -> Path:
    """Train three CPU MLP heads using only the embedding artifact's train split."""
    out = _output(out)
    payload = _read_embeddings(embeddings)
    records, vectors = payload["records"], payload["embeddings"]
    indices = [index for index, row in enumerate(records) if row["split"] == "train"]
    if {records[index]["question"] for index in indices} != set(_VALUES):
        raise ValueError("training split must contain all three questions")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        heads = _heads(vectors.shape[1], _VALUES)
        heads.train()
        optimizer = torch.optim.Adam(heads.parameters(), lr=1e-3)
        for _ in range(20):
            order = [indices[index] for index in torch.randperm(len(indices)).tolist()]
            for start in range(0, len(order), 32):
                losses = []
                for index in order[start:start + 32]:
                    row = records[index]
                    question = row["question"]
                    semantic = row["options"][row["label"]]
                    losses.append(option_loss(
                        heads[question](vectors[index]), _VALUES[question].index(semantic),
                        question, _VALUES[question],
                    ))
                optimizer.zero_grad(set_to_none=True)
                torch.stack(losses).mean().backward()
                optimizer.step()
        state = heads.state_dict()
    _save(out, {
        "format": _HEAD, "dimension": vectors.shape[1], "values": _VALUES,
        "state_dict": state, "model_version": payload["model_version"],
        "source_sha256": payload["source_sha256"],
    })
    return out


def predict_head(embeddings: Path, checkpoint: Path) -> dict[str, dict[str, float]]:
    """Return option-letter probabilities using the head's stored semantic order."""
    payload = _read_embeddings(embeddings)
    saved = torch.load(_path(checkpoint), map_location="cpu", weights_only=True)
    if (
        not isinstance(saved, dict) or saved.get("format") != _HEAD
        or saved.get("dimension") != payload["embeddings"].shape[1]
        or saved.get("model_version") != payload["model_version"]
        or not isinstance(saved.get("values"), dict) or set(saved["values"]) != set(_VALUES)
    ):
        raise ValueError("head checkpoint does not match the embedding model")
    values = saved["values"]
    if any(
        not isinstance(values[question], list)
        or len(values[question]) != len(_VALUES[question])
        or not all(isinstance(value, str) for value in values[question])
        or set(values[question]) != set(_VALUES[question])
        for question in _VALUES
    ):
        raise ValueError("head checkpoint has invalid semantic options")
    heads = _heads(saved["dimension"], values)
    heads.load_state_dict(saved["state_dict"], strict=True)
    heads.eval()
    predictions = {}
    with torch.no_grad():
        for row, vector in zip(payload["records"], payload["embeddings"], strict=True):
            question = row["question"]
            probabilities = torch.softmax(heads[question](vector), dim=-1)
            if not torch.isfinite(probabilities).all():
                raise ValueError(f"head produced non-finite probabilities: {row['id']}")
            predictions[row["id"]] = {
                letter: float(probabilities[values[question].index(semantic)])
                for letter, semantic in row["options"].items()
            }
    return predictions
