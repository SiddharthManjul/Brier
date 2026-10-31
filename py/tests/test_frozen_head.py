from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from brier_ml import frozen_head as frozen
from torch import nn

_VALUES = {
    "regressed": ["no", "yes"],
    "failure_mode": ["none", "stall/timeout", "overshoot/misalignment",
                     "grasp or drop failure", "collision", "oscillation"],
    "success_bucket": ["1", "2", "3", "4", "5"],
}
_VERSION = "test/frozen-decoder@" + "a" * 40


def _records() -> list[dict]:
    records = []
    for index, split in enumerate(("train", "calibration", "test", "ood"), 1):
        for question, values in _VALUES.items():
            for evidence in ("cheap", "no_cheap"):
                records.append({
                    "id": f"c{index:03d}/pusht-s01/{question}/{evidence}",
                    "split": split, "question": question,
                    "options": dict(zip("ABCDEF", values, strict=False)),
                    "label": "A" if split == "train" else "B",
                })
    return records


def _artifact(path: Path) -> dict:
    records = _records()
    payload = {
        "format": frozen._EMBEDDINGS, "model_version": _VERSION,
        "source_sha256": "b" * 64, "records": records,
        "embeddings": torch.tensor([[1.0, 0.5]] * len(records)),
    }
    torch.save(payload, path)
    return payload


@pytest.mark.parametrize("native, context", [(False, 1536), (True, 131072), (True, 2)])
def test_extract_embeddings_uses_verbatim_prompts_and_last_token(tmp_path, monkeypatch, native, context):
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    monkeypatch.setenv("MODAL_MODEL_ID", "test/frozen-decoder")
    monkeypatch.setenv("MODAL_MODEL_REVISION", "a" * 40)
    prompt = "<system> Brier Ω\n<state>\ncompiled\n</state>\n<question> q\n<options> options\n<answer>"
    rows = [{**row, "prompt": prompt} for row in _records()]
    compiled = tmp_path / "compiled.jsonl"
    compiled.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    prompts = []

    class Tokenizer:
        def __call__(self, prompt, *, add_special_tokens, return_tensors):
            assert not add_special_tokens and return_tensors == "pt"
            prompts.append(prompt)
            return {"input_ids": torch.tensor([[1, len(prompts) + 1]])}

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 2)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, *, output_hidden_states, return_dict, use_cache):
            assert not self.training and not torch.is_grad_enabled()
            assert output_hidden_states and return_dict and not use_cache
            hidden = torch.stack((input_ids, input_ids + 10), dim=-1).float()
            return SimpleNamespace(hidden_states=(hidden,))

    config = SimpleNamespace(is_encoder_decoder=False, vision_config=None, max_position_embeddings=context)
    if native:
        config = SimpleNamespace(model_type="muse_glimmer", architectures=["MuseGlimmerForConditionalGeneration"],
            vision_config=SimpleNamespace(model_type="muse_glimmer_vision"),
            text_config=SimpleNamespace(model_type="muse_glimmer_text", max_position_embeddings=context))

    def load_model(identifier, **kwargs):
        assert identifier == "test/frozen-decoder" and kwargs["revision"] == "a" * 40
        assert kwargs["config"] is config and not kwargs["trust_remote_code"]
        assert kwargs["dtype"] == torch.bfloat16
        return Model()

    def wrong_loader(*args, **kwargs):
        pytest.fail("frozen extraction selected the wrong Transformers model class")

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: config),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=wrong_loader if native else load_model),
        MuseGlimmerForConditionalGeneration=SimpleNamespace(from_pretrained=load_model if native else wrong_loader),
    ))
    out = tmp_path / "embeddings.pt"
    if context == 2:
        with pytest.raises(ValueError, match="model input limit"):
            frozen.extract_embeddings(compiled, out)
        assert not out.exists()
        return
    assert frozen.extract_embeddings(compiled, out) == out
    assert not config.use_cache and (not native or not config.text_config.use_cache)
    payload = torch.load(out, weights_only=True)
    assert prompts == [row["prompt"] for row in rows]
    assert payload["embeddings"][0].tolist() == [2.0, 12.0]
    assert payload["embeddings"].shape == (24, 2)
    assert all("prompt" not in row for row in payload["records"])
    with pytest.raises(FileExistsError):
        frozen.extract_embeddings(compiled, out)
    rows[-1]["split"] = "train"
    compiled.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="multiple splits"):
        frozen.extract_embeddings(compiled, tmp_path / "invalid.pt")


def test_train_head_ignores_heldout_labels_reproducibly(tmp_path, monkeypatch):
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    source = tmp_path / "embeddings.pt"
    payload = _artifact(source)
    calls, original = [], frozen.option_loss

    def loss(logits, target, question, option_values):
        calls.append(question)
        return original(logits, target, question, option_values)

    monkeypatch.setattr(frozen, "option_loss", loss)
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    assert frozen.train_head(source, first) == first
    for row in payload["records"]:
        if row["split"] != "train":
            row["label"] = "A"
    torch.save(payload, source)
    frozen.train_head(source, second)
    left, right = (torch.load(path, weights_only=True) for path in (first, second))
    assert set(calls) == set(_VALUES)
    assert left["values"] == _VALUES
    assert all(torch.equal(value, right["state_dict"][key]) for key, value in left["state_dict"].items())
    predicted = frozen.predict_head(source, first)
    assert all(predicted[row["id"]]["A"] > 1 / len(row["options"])
               for row in payload["records"] if row["split"] == "train")


def test_predict_head_maps_shuffled_letters_to_semantics(tmp_path, monkeypatch):
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    source, checkpoint = tmp_path / "embeddings.pt", tmp_path / "head.pt"
    payload = _artifact(source)
    heads = frozen._heads(2, _VALUES)
    with torch.no_grad():
        for head in heads.values():
            for parameter in head.parameters():
                parameter.zero_()
            head[-1].bias.copy_(torch.arange(head[-1].out_features).float())
    torch.save({"format": frozen._HEAD, "dimension": 2, "values": _VALUES,
                "state_dict": heads.state_dict(), "model_version": _VERSION,
                "source_sha256": "b" * 64}, checkpoint)
    before = frozen.predict_head(source, checkpoint)
    row = payload["records"][0]
    row["options"], row["label"] = {"A": "yes", "B": "no"}, "B"
    torch.save(payload, source)
    after = frozen.predict_head(source, checkpoint)
    assert before[row["id"]]["A"] == pytest.approx(after[row["id"]]["B"])
    assert before[row["id"]]["B"] == pytest.approx(after[row["id"]]["A"])
    assert all(sum(value.values()) == pytest.approx(1.0) for value in after.values())


@pytest.mark.parametrize("corruption", ["split", "id", "question", "semantics", "finite"])
def test_rejects_corrupted_embedding_artifacts(tmp_path, monkeypatch, corruption):
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    source = tmp_path / "embeddings.pt"
    payload = _artifact(source)
    if corruption == "split":
        payload["records"][-1]["id"] = "c001/pusht-s02/success_bucket/no_cheap"
    elif corruption == "id":
        payload["records"][0]["id"] = "invalid"
    elif corruption == "question":
        payload["records"][0]["question"] = "failure_mode"
    elif corruption == "semantics":
        payload["records"][2]["options"] = {"A": "contact", "B": "timeout"}
    else:
        payload["embeddings"][0, 0] = float("nan")
    torch.save(payload, source)
    with pytest.raises(ValueError):
        frozen.train_head(source, tmp_path / "head.pt")
