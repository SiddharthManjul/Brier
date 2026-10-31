from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from brier_ml import distill, finetune

_PROMPT = "<system> Brier Ω\n<state>\ncompiled\n</state>\n<question> q\n<options> A) no B) yes\n<answer>"
_TEACHER = "configured/teacher@" + "a" * 40 + "+lora@" + "c" * 64


def _data(directory: Path):
    rows = [{"id": f"c{index:03d}/pusht-s01/regressed/cheap", "split": split,
             "question": "regressed", "options": {"A": "no", "B": "yes"},
             "prompt": _PROMPT, "label": "A"}
            for index, split in enumerate(("train", "calibration", "test", "ood"), 1)]
    compiled, teacher = directory / "compiled.jsonl", directory / "teacher.json"
    compiled.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    artifact = {"model_version": _TEACHER, "source_sha256": hashlib.sha256(compiled.read_bytes()).hexdigest(),
                "questions": {row["id"]: {"B": 0.8, "A": 0.2} for row in rows},
                "latencies_seconds": {row["id"]: 0.1 for row in rows}}
    teacher.write_text(json.dumps(artifact))
    return compiled, teacher, rows, artifact


def test_load_teacher_binds_letter_order_to_source_and_filters_splits(tmp_path: Path) -> None:
    compiled, teacher, rows, artifact = _data(tmp_path)
    train, targets, version = distill.load_teacher(compiled, teacher)
    assert train == rows[:1] and targets == [[0.2, 0.8]] and version == _TEACHER
    rows[0]["options"] = {"A": "yes", "B": "no"}
    compiled.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="source_sha256"):
        distill.load_teacher(compiled, teacher)
    artifact["source_sha256"] = hashlib.sha256(compiled.read_bytes()).hexdigest()
    teacher.write_text(json.dumps(artifact))
    train, targets, _ = distill.load_teacher(compiled, teacher)
    assert train[0]["options"] == {"A": "yes", "B": "no"} and targets == [[0.2, 0.8]]


@pytest.mark.parametrize("mutation", ["missing", "extra", "nan", "boolean", "mass", "letters", "source", "version", "duplicate"])
def test_load_teacher_rejects_incomplete_or_malformed_artifacts(tmp_path: Path, mutation: str) -> None:
    compiled, teacher, rows, artifact = _data(tmp_path)
    probabilities = artifact["questions"][rows[-1]["id"]]
    if mutation == "missing":
        artifact["questions"].pop(rows[-1]["id"])
    elif mutation == "extra":
        artifact["questions"]["unknown"] = {"A": 0.2, "B": 0.8}
    elif mutation in {"nan", "boolean", "mass"}:
        probabilities["A"] = {"nan": float("nan"), "boolean": True, "mass": 0.9}[mutation]
    elif mutation == "letters":
        probabilities["C"] = probabilities.pop("A")
    elif mutation == "source":
        artifact["source_sha256"] = "0" * 64
    elif mutation == "version":
        artifact["model_version"] = "configured/teacher@main"
    teacher.write_text(json.dumps(artifact))
    if mutation == "duplicate":
        teacher.write_text(teacher.read_text()[:-1] + ', "questions": {}}')
    with pytest.raises(ValueError):
        distill.load_teacher(compiled, teacher)


class _Tokenizer:
    def __init__(self, mode: str = "valid") -> None:
        self.mode, self.calls = mode, []
        self.special_tokens_map = {"pad_token": "<pad>"} if mode != "roles" else {"bos_token": "<pad>"}
        self.all_special_ids = [1] if mode == "special_ids" else [0]

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "first": 1, "second": 2, "A": 3, "B": 4,
                "unused": 6 if self.mode == "vocabulary" else 5}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        self.calls.append(text)
        context = [2, 2, 1] if self.mode == "prompt" else [1, 2, 1]
        if text in ("A", "B"):
            return [3 + (text == "B")]
        if text.endswith(("A", "B")):
            return [7] if self.mode == "boundary" else context + [3 + (text[-1] == "B")]
        return context


def test_validate_tokenizers_preserves_exact_rust_answer_boundaries(tmp_path: Path) -> None:
    _, _, rows, _ = _data(tmp_path)
    teacher, student = _Tokenizer(), _Tokenizer()
    distill.validate_tokenizers(teacher, student, rows)
    assert teacher.calls == student.calls
    assert teacher.calls[0] == _PROMPT and _PROMPT + "A" in teacher.calls


@pytest.mark.parametrize("mode", ["vocabulary", "roles", "special_ids", "prompt", "boundary"])
def test_validate_tokenizers_rejects_incompatible_companions(tmp_path: Path, mode: str) -> None:
    _, _, rows, _ = _data(tmp_path)
    with pytest.raises(ValueError):
        distill.validate_tokenizers(_Tokenizer(), _Tokenizer(mode), rows)


def test_train_student_uses_only_train_targets_with_pinned_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    compiled, teacher, rows, artifact = _data(tmp_path)
    for name, value in {"BRIER_DATA": str(tmp_path), "MODAL_MODEL_ID": "configured/teacher",
                        "MODAL_MODEL_REVISION": "a" * 40, "MODAL_STUDENT_MODEL_ID": "configured/student",
                        "MODAL_STUDENT_MODEL_REVISION": "b" * 40}.items():
        monkeypatch.setenv(name, value)
    calls = []

    class Student(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(1, 3, 8))
            self.config = SimpleNamespace(to_dict=dict)

        def forward(self, input_ids: torch.Tensor, use_cache: bool):
            assert input_ids.tolist() == [[1, 2, 1]] and use_cache is False
            return SimpleNamespace(logits=self.scores)

        def save_pretrained(self, directory: Path) -> None:
            directory.mkdir()
            torch.save(self.state_dict(), directory / "adapter.pt")

    def load_model(identifier: str, revision: str):
        calls.append((identifier, revision))
        return Student(), _Tokenizer()

    def teacher_tokenizer(identifier: str, **kwargs: Any):
        assert identifier == "configured/teacher"
        assert kwargs == {"revision": "a" * 40, "trust_remote_code": False}
        return _Tokenizer()

    directories = iter((tmp_path / "first", tmp_path / "second"))
    monkeypatch.setattr(finetune, "_training_paths", lambda *args: (next(directories), "test-volume"))
    monkeypatch.setattr(finetune, "_seed_training", lambda: None)
    monkeypatch.setattr(finetune, "_load_model", load_model)
    monkeypatch.setattr(finetune.modal.Volume, "from_name", lambda name: SimpleNamespace(commit=lambda: None))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=teacher_tokenizer)))
    first = distill.train_student(compiled, teacher)
    for row in rows[1:]:
        artifact["questions"][row["id"]] = {"A": 0.8, "B": 0.2}
    teacher.write_text(json.dumps(artifact))
    second = distill.train_student(compiled, teacher)
    states = [torch.load(tmp_path / name / "training-state.pt", weights_only=False) for name in ("first", "second")]
    assert calls == [("configured/student", "b" * 40)] * 2
    assert first["training_examples"] == second["training_examples"] == 1
    assert first["tokens"] == second["tokens"] == 3
    torch.testing.assert_close(states[0]["parameters"]["scores"], states[1]["parameters"]["scores"])
    assert states[0]["config"]["teacher"]["model_version"] == _TEACHER
    assert states[0]["config"]["revision"] == "b" * 40


def test_train_student_requires_explicit_student_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODAL_STUDENT_MODEL_ID", raising=False)
    with pytest.raises(ValueError, match="MODAL_STUDENT_MODEL_ID"):
        distill.train_student(tmp_path / "compiled.jsonl", tmp_path / "teacher.json")
