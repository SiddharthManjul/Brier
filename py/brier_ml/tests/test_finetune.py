import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from brier_ml import finetune, modal_train
from brier_ml import glimmer_model as gm
from tokenizers import Tokenizer
from transformers import LlamaConfig, LlamaForCausalLM

_OPTIONS = {
    "regressed": ["yes", "no"],
    "failure_mode": ["none", "stall or timeout", "overshoot or misalignment",
                     "grasp or drop", "collision", "oscillation"],
    "success_bucket": ["0-19%", "20-39%", "40-59%", "60-79%", "80-100%"],
}
_QUESTIONS = {
    "regressed": "Will the expensive evaluation regress by more than 15 points?",
    "failure_mode": "If the expensive evaluation regresses, what is the main failure mode?",
    "success_bucket": "What will the expensive success rate be?",
}


class _Tokenizer:
    def __init__(self):
        self.raw = Tokenizer.from_file(str(Path(__file__).parent / "fixtures/glimmer_bpe.json"))
        self.pad_token_id = self.raw.token_to_id("[PAD]")
        self.eos_token_id = self.raw.token_to_id("[EOS]")

    def encode(self, text, *, add_special_tokens):
        return self.raw.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(self, text, *, add_special_tokens):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}


def _example(i, split):
    change = next(f"c{n:03}" for n in range(1, 1000) if
                  (int.from_bytes(hashlib.sha256(f"c{n:03}".encode()).digest()[:4], "big") % 10 < 6)
                  == (split == "train") and (split == "train" or
                  int.from_bytes(hashlib.sha256(f"c{n:03}".encode()).digest()[:4], "big") % 10 < 8))
    scenario, question = f"pusht-s{i // 3 + 1:02}", list(_QUESTIONS)[i % 3] if split == "train" else "regressed"
    if split != "train":
        scenario = f"pusht-s{i + 1:02}"
    options = _OPTIONS[question].copy()
    if split == "train":
        options.reverse()
    prompt = ("<system> Brier\n<state>\nstate " + ("short " * (i % 3))
              + "\n</state>\n<question> " + _QUESTIONS[question] + "\n<options> "
              + "  ".join(f"{letter}) {option}" for letter, option in zip("ABCDEF", options)) + "\n<answer>")
    return {"id": f"{change}:{scenario}:{question}:ev", "pair_id": f"{change}:{scenario}",
            "change_id": change, "scenario_id": scenario, "env": "pusht", "variant": "ev",
            "split": split, "question": question, "options": options, "prompt": prompt,
            "label": "ABCDEF"[i % len(options)]}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    base = tmp_path / "models/base/glimmer"
    base.mkdir(parents=True)
    (base / "revision.txt").write_text("1" * 40)
    compiled = tmp_path / "compiled.jsonl"
    rows = [_example(i, "train") for i in range(22)] + [_example(i, "calibration") for i in range(2)]
    compiled.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    monkeypatch.setattr(modal_train, "BASE_DIR", str(base))
    commits, models = [], []
    monkeypatch.setattr(modal_train, "MODELS", SimpleNamespace(commit=lambda: commits.append(True)))

    def load(path, revision, quant):
        tokenizer = _Tokenizer()
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=tokenizer.raw.get_vocab_size(), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=1536, attention_dropout=0.1))
        models.append(model)
        return model, tokenizer, {"class": type(model).__name__, "revision": revision, "quant": quant}

    monkeypatch.setattr(gm, "load_base", load)
    cfg = {"run_id": "full", "quant": "bf16", "micro_batch": 2, "effective_batch": 4,
           "max_usd": 100, "usd_per_sec": 0.001097, "max_steps": 6,
           "ckpt_every": 3, "eval_every": 3}
    return cfg, tmp_path / "models/runs", commits, models, compiled


def _logs(run):
    return [json.loads(line) for line in (run / "train_log.jsonl").read_text().splitlines()]


def test_train_six_steps_real_tiny_lora_resume_matches_uninterrupted(setup, monkeypatch):
    cfg, runs, commits, models, _ = setup
    full = finetune.train(cfg)
    reference = {n: p.detach().clone() for n, p in models[-1].named_parameters() if "lora_" in n}
    cfg = cfg | {"run_id": "resumed"}
    original = modal_train.MODELS.commit

    def interrupt():
        original()
        if (runs / "resumed/ckpt-000003/state.json").exists():
            raise InterruptedError("preemption after optimizer step 3")

    monkeypatch.setattr(modal_train.MODELS, "commit", interrupt)
    with pytest.raises(InterruptedError, match="step 3"):
        finetune.train(cfg)
    assert not (runs / "resumed/adapter").exists()
    state = json.loads((runs / "resumed/ckpt-000003/state.json").read_text())
    assert (state["step"], state["epoch"], state["cursor"]) == (3, 0, 6)
    assert set(state["rng"]) == {"python", "numpy", "torch", "cuda"}
    monkeypatch.setattr(modal_train.MODELS, "commit", original)
    resumed = finetune.train(cfg)
    assert full["steps"] == resumed["steps"] == 6
    assert [r["loss"] for r in _logs(runs / "full")] == [r["loss"] for r in _logs(runs / "resumed")]
    assert [r["lr"] for r in _logs(runs / "full")] == [r["lr"] for r in _logs(runs / "resumed")]
    for name, value in models[-1].named_parameters():
        if name in reference:
            assert torch.equal(value, reference[name])
    assert commits
    assert (runs / "resumed/adapter/adapter_model.safetensors").exists()
    manifest = json.loads((runs / "resumed/adapter/brier-model.json").read_text())
    assert manifest["config"]["model_id"] == modal_train.BASE_REPO
    assert manifest["config"]["revision"] == "1" * 40
    assert manifest["config"]["dataset_digest"] == resumed["data_sha256"]
    assert manifest["config"]["recipe"] == resumed["config"] and manifest["step"] == 6
    assert not (runs / "resumed/adapter/brier-model.json.tmp").exists()
    assert np.isfinite(resumed["final_eval_nll"])
    assert resumed["letter_token_ids"] == [_Tokenizer().encode(" " + c, add_special_tokens=False)[0] for c in "ABCDEF"]
    assert resumed["data_sha256"] == state["data_hash"]
    assert resumed["versions"]["torch"] == torch.__version__.split("+")[0]
    for record in _logs(runs / "resumed"):
        assert {"step", "loss", "ce", "extra", "lr", "grad_norm", "tokens_per_second",
                "examples_per_second", "peak_memory", "elapsed_seconds", "spend"} <= record.keys()
        assert all(np.isfinite(record[key]) for key in ("loss", "ce", "extra", "grad_norm"))
    smoke = json.loads((runs / "resumed/smoke_report.json").read_text())
    assert smoke["median_seconds_per_step"] == _logs(runs / "resumed")[-1]["step_seconds"]
    assert smoke["projected_epoch_cost"] > 0


def test_train_full_epoch_uses_partial_final_accumulation_and_eval_subset(setup):
    cfg, runs, _, _, _ = setup
    result = finetune.train(cfg | {"max_steps": None, "epochs": 2})
    logs = _logs(runs / "full")
    assert result["steps"] == 12 and result["epochs"] == 2
    state = json.loads((runs / "full/ckpt-000012/state.json").read_text())
    assert state["examples"] == 44 and state["epoch"] == 1 and state["cursor"] == 11
    assert [r["step"] for r in logs if "eval" in r] == [3, 6, 9, 12]
    assert not (runs / "full/smoke_report.json").exists()


def test_train_budget_stop_saves_checkpoint_without_final_adapter(setup, caplog):
    cfg, runs, commits, _, _ = setup
    result = finetune.train(cfg | {"max_usd": 1e-12})
    assert result["status"] == "budget_stop" and result["step"] == 0
    assert (runs / "full/ckpt-000000/state.json").exists() and commits
    assert not (runs / "full/adapter").exists()
    assert "Budget stop at step 0" in caplog.text


def test_train_nan_saves_nothing_and_identifies_batch(setup, monkeypatch):
    cfg, runs, commits, _, _ = setup
    from brier_ml import losses

    def bad(*args):
        loss = torch.tensor(float("nan"), requires_grad=True)
        return loss, {"ce": loss, "extra": loss}

    monkeypatch.setattr(losses, "proper_loss", bad)
    with pytest.raises(FloatingPointError, match=r"nonfinite loss at step 1:.*c[0-9]+:pusht-s"):
        finetune.train(cfg)
    assert not commits and not list((runs / "full").rglob("*"))


@pytest.mark.parametrize("mutation", ["config", "data", "revision", "cursor"])
def test_train_resume_rejects_changed_identity_or_cursor(setup, mutation):
    cfg, runs, _, _, compiled = setup
    finetune.train(cfg)
    checkpoint = runs / "full/ckpt-000006/state.json"
    state = json.loads(checkpoint.read_text())
    if mutation == "config":
        cfg = cfg | {"lr": 3e-5}
    elif mutation == "data":
        compiled.write_text(" " + compiled.read_text())
    elif mutation == "revision":
        state["revision"] = "2" * 40
        checkpoint.write_text(json.dumps(state))
    else:
        state["cursor"] += 1
        checkpoint.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="checkpoint"):
        finetune.train(cfg)


def test_train_checkpoint_retention_keeps_two_newest_and_best(setup, monkeypatch):
    cfg, runs, _, _, _ = setup
    from brier_ml import losses

    original = losses.option_probs
    calls = 0

    def probabilities(logits, valid):
        nonlocal calls
        calls += 1
        p = original(logits, valid)
        # Calibration has labels A/B; make its first evaluation best, then degrade.
        good = calls <= 1
        return p * 0 + torch.tensor([0.5, 0.5, 0, 0, 0, 0] if good else [0.01, 0.99, 0, 0, 0, 0])

    monkeypatch.setattr(losses, "option_probs", probabilities)
    finetune.train(cfg | {"ckpt_every": 1, "eval_every": 1})
    names = sorted(p.name for p in (runs / "full").glob("ckpt-*"))
    assert names == ["ckpt-000001", "ckpt-000005", "ckpt-000006"]


@pytest.mark.parametrize("change", [{"extra": True}, {"micro_batch": 3}, {"max_usd": float("nan")},
                                    {"run_id": "../escape"}, {"quant": "fp32"}, {"max_steps": True}])
def test_train_rejects_invalid_configuration_before_model_load(setup, monkeypatch, change):
    cfg, runs, _, models, _ = setup
    with pytest.raises(ValueError):
        finetune.train(cfg | change)
    assert not models and not runs.exists()


def test_train_resume_discards_uncheckpointed_and_partial_log_tail(setup, monkeypatch):
    cfg, runs, _, _, _ = setup
    original = modal_train.MODELS.commit

    def interrupt():
        original()
        if (runs / "full/ckpt-000003/state.json").exists():
            raise InterruptedError

    monkeypatch.setattr(modal_train.MODELS, "commit", interrupt)
    with pytest.raises(InterruptedError):
        finetune.train(cfg)
    log = runs / "full/train_log.jsonl"
    with log.open("a") as stream:
        stream.write('{"step":4,"loss":999}\n{"step":5')
    monkeypatch.setattr(modal_train.MODELS, "commit", original)
    finetune.train(cfg)
    assert [r["step"] for r in _logs(runs / "full")] == list(range(1, 7))
    assert all(r["loss"] != 999 for r in _logs(runs / "full"))


def test_train_actual_nonfinite_logits_stop_before_checkpoint(setup, monkeypatch):
    cfg, runs, commits, _, _ = setup
    original = gm.option_logits
    monkeypatch.setattr(gm, "option_logits", lambda *args: original(*args) * float("inf"))
    with pytest.raises(FloatingPointError, match="nonfinite logits at step 1"):
        finetune.train(cfg)
    assert not commits and not list((runs / "full").rglob("*"))


def test_train_init_adapter_loads_previous_weights_without_optimizer_or_cursor(setup, monkeypatch):
    cfg, runs, _, _, _ = setup
    from peft import get_peft_model_state_dict

    finetune.train(cfg)
    from peft.utils.save_and_load import load_peft_weights

    weights = load_peft_weights(str(runs / "full/adapter"), device="cpu")
    observed = []
    original = gm.last_logits

    def observe(model, batch):
        if not observed:
            observed.append({k: value.clone() for k, value in get_peft_model_state_dict(model).items()})
        return original(model, batch)

    monkeypatch.setattr(gm, "last_logits", observe)
    result = finetune.train(cfg | {"run_id": "epoch2", "max_steps": 1,
                                   "init_adapter": str(runs / "full/adapter"), "lr": 3e-5})
    assert result["steps"] == 1
    assert all(torch.equal(weights[key], value) for key, value in observed[0].items())
    assert _logs(runs / "epoch2")[0]["step"] == 1
