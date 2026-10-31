from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from brier_ml import finetune

_VERSION = re.compile(r"([^@]+)@([0-9a-f]{40})(?:\+lora@[0-9a-f]{64})?")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate teacher JSON key: {key}")
        result[key] = value
    return result


def load_teacher(compiled: Path, teacher: Path) -> tuple[list[dict[str, Any]], list[list[float]], str]:
    """Validate source-bound teacher probabilities and retain only training targets."""
    train_rows = finetune.load_training_data(compiled)
    rows = [json.loads(line) for line in compiled.read_text().splitlines()]
    try:
        artifact = json.loads(teacher.read_text(), object_pairs_hook=_unique)
    except (OSError, ValueError) as error:
        raise ValueError("teacher artifact must contain valid unique-key JSON") from error
    required = {"model_version", "source_sha256", "questions"}
    if (
        not isinstance(artifact, dict) or not required <= set(artifact)
        or set(artifact) - required - {"latencies_seconds"}
        or not isinstance(artifact["model_version"], str)
        or _VERSION.fullmatch(artifact["model_version"]) is None
        or artifact["source_sha256"] != hashlib.sha256(compiled.read_bytes()).hexdigest()
    ):
        raise ValueError("teacher requires a pinned model version and the exact compiled source_sha256")
    probabilities = artifact["questions"]
    if not isinstance(probabilities, dict) or set(probabilities) != {row["id"] for row in rows}:
        raise ValueError("teacher questions must match every compiled ID exactly")
    for row in rows:
        values = probabilities[row["id"]]
        if (
            not isinstance(values, dict) or set(values) != set(row["options"])
            or any(type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value)
                   for value in values.values())
            or not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6, rel_tol=0)
        ):
            raise ValueError(f"invalid teacher option distribution: {row['id']}")
    if "latencies_seconds" in artifact:
        latency = artifact["latencies_seconds"]
        if not isinstance(latency, dict) or set(latency) != set(probabilities) or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in latency.values()
        ):
            raise ValueError("teacher latency metadata must match compiled IDs and be finite")
    targets = [[float(probabilities[row["id"]][letter]) for letter in sorted(row["options"])]
               for row in train_rows]
    return train_rows, targets, artifact["model_version"]


def validate_tokenizers(teacher: Any, student: Any, rows: list[dict[str, Any]]) -> None:
    """Require equal full vocabularies, special-token roles, and compiled answer encodings."""
    vocabulary = teacher.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary or vocabulary != student.get_vocab():
        raise ValueError("teacher and student full token vocabularies differ")
    if (
        teacher.special_tokens_map != student.special_tokens_map
        or sorted(teacher.all_special_ids) != sorted(student.all_special_ids)
    ):
        raise ValueError("teacher and student special-token roles or IDs differ")
    if finetune._encode(rows, teacher) != finetune._encode(rows, student):
        raise ValueError("teacher and student compiled prompt or option encodings differ")


def train_student(compiled: Path, teacher: Path, smoke: bool = False) -> dict[str, Any]:
    """Distill a pinned compatible student from training-only teacher targets and Rust labels."""
    student_id = finetune._required("MODAL_STUDENT_MODEL_ID")
    student_revision = finetune._required("MODAL_STUDENT_MODEL_REVISION")
    teacher_id = finetune._required("MODAL_MODEL_ID")
    teacher_revision = finetune._required("MODAL_MODEL_REVISION")
    if re.fullmatch(r"[0-9a-f]{40}", teacher_revision) is None:
        raise ValueError("teacher MODAL_MODEL_REVISION must be an immutable 40-hex commit")
    directory, volume_name = finetune._training_paths(compiled, smoke, student_revision)
    if not teacher.resolve().is_relative_to(Path(os.environ["BRIER_DATA"]).resolve()):
        raise ValueError("teacher artifact must reside inside BRIER_DATA")
    rows, targets, version = load_teacher(compiled, teacher)
    if version.split("+lora@", 1)[0] != f"{teacher_id}@{teacher_revision}":
        raise ValueError("teacher artifact does not match the configured pinned teacher model")
    finetune._seed_training()
    model, tokenizer = finetune._load_model(student_id, student_revision)
    from transformers import AutoTokenizer

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_id, revision=teacher_revision, trust_remote_code=False
    )
    all_rows = [json.loads(line) for line in compiled.read_text().splitlines()]
    validate_tokenizers(teacher_tokenizer, tokenizer, all_rows)
    encoded = finetune._encode(rows, tokenizer)
    config = finetune._training_config(student_id, student_revision, compiled, model, smoke)
    config["teacher"] = {"model_version": version, "artifact_digest": hashlib.sha256(teacher.read_bytes()).hexdigest()}
    config["recipe"]["distillation_kl_weight"] = 1.0
    volume = finetune.modal.Volume.from_name(volume_name)
    return finetune._fit(model, encoded, directory, config, smoke, volume.commit,
                         teacher_distributions=targets)


app = finetune.modal.App("brier-distill")


@app.function(image=finetune._image, gpu="A100-80GB", timeout=86400)
def _remote_student(compiled: str, teacher: str, smoke: bool = False) -> dict[str, Any]:
    return train_student(Path(compiled), Path(teacher), smoke)


@app.local_entrypoint()
def _launch(compiled: str, teacher: str, smoke: bool = False) -> None:
    names = ("MODAL_STUDENT_MODEL_ID", "MODAL_STUDENT_MODEL_REVISION", "MODAL_MODEL_ID",
             "MODAL_MODEL_REVISION", "MODAL_CHECKPOINT_DIR", "BRIER_DATA", "MODAL_VOLUME_NAME")
    environment = {name: finetune._required(name) for name in names}
    volume = finetune.modal.Volume.from_name(environment["MODAL_VOLUME_NAME"], create_if_missing=True)
    _remote_student.with_options(env=environment, volumes={"/checkpoints": volume}).remote(compiled, teacher, smoke)
