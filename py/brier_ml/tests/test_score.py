import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("modal")

from brier_ml import glimmer_model as gm
from brier_ml import modal_train, score
from test_glimmer_data import _BPE, _row


@pytest.fixture
def setup(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = tf.LlamaForCausalLM(tf.LlamaConfig(
        vocab_size=394, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, attention_dropout=0.0)).eval()
    tokenizer = _BPE()
    base = tmp_path / "models/base/glimmer"
    base.mkdir(parents=True)
    revision = "a" * 40
    (base / "revision.txt").write_text(revision)
    (tmp_path / "models/runs").mkdir()
    check = tmp_path / "models/runs/check.json"
    check.write_text(json.dumps({"revision": revision, "quant": "bf16"}))
    rows = [_row(question, split, env, variant,
                 "occlusion" if split == "ood" else "action_latency")
            for question in ["success_bucket", "regressed", "failure_mode"]
            for variant in ["noev", "ev"] for env in ["aloha", "pusht"]
            for split in ["ood", "test", "train", "calibration"]]
    compiled = tmp_path / "data"
    compiled.mkdir()
    (compiled / "compiled.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setenv("BRIER_DATA", str(compiled))
    monkeypatch.setattr(modal_train, "BASE_DIR", str(base))
    loads = []

    def load(path, sha, quant):
        loads.append((path, sha, quant))
        return model, tokenizer, {"final_norm": "model.norm"}

    monkeypatch.setattr(gm, "load_base", load)
    return tmp_path, model, tokenizer, rows, loads, check


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_base_single_forward_formats_order_and_embeddings(setup):
    root, model, _tokenizer, rows, loads, _ = setup
    forwards, heads, norms = [], [], []
    model.register_forward_pre_hook(lambda _m, _a, kw: forwards.append(
        (len(kw["input_ids"]), torch.is_grad_enabled(), kw["logits_to_keep"])), with_kwargs=True)
    model.get_output_embeddings().register_forward_pre_hook(
        lambda _m, args: heads.append(tuple(args[0].shape)))
    model.model.norm.register_forward_hook(lambda _m, _a, out: norms.append(
        out[:, -1].detach().float().numpy()))
    out = root / "models/runs/base"
    meta = score.base_pass(out, usd_per_sec=1e-12)
    assert forwards == [(32, False, 1), (16, False, 1)]
    assert heads == [(32, 1, 16), (16, 1, 16)]
    assert len(norms) == 2 and len(loads) == 1
    all_hidden = np.concatenate(norms)
    for split in score._SPLITS:
        expected = [row for row in rows if row["split"] == split]
        actual = _read(out / "probs" / f"{split}.jsonl")
        indices = [i for i, row in enumerate(rows) if row["split"] == split]
        assert json.loads((out / "emb" / f"{split}.ids.json").read_text()) == [r["id"] for r in actual]
        hidden = np.load(out / "emb" / f"{split}.npy")
        assert hidden.dtype == np.float16 and hidden.shape == (12, 16)
        np.testing.assert_array_equal(hidden, all_hidden[indices].astype(np.float16))
        assert [r["id"] for r in actual] == [r["id"] for r in expected]
        for found, source in zip(actual, expected, strict=True):
            assert set(found) == {"id", "pair_id", "split", "question", "variant", "options",
                                  "label", "logits", "probs", "model", "revision"}
            assert found["options"] == source["options"] and found["label"] == source["label"]
            assert len(found["logits"]) == 6 and all(type(v) is float for v in found["logits"])
            assert list(found["probs"]) == list("ABCDEF"[:len(source["options"])])
            assert sum(found["probs"].values()) == pytest.approx(1)
            assert list(found["probs"].values()) == pytest.approx(
                torch.tensor(found["logits"][:len(source["options"])]).softmax(-1).tolist())
            assert found["model"] == "glimmer-base-zeroshot" and found["revision"] == "a" * 40
    assert meta["rows"] == 48 and meta["quant"] == "bf16" and meta["spend"] > 0
    assert not (out / "score_checkpoint.pt").exists()


def test_adapter_loads_actual_lora_and_recorded_nf4_quantization(setup):
    root, model, tokenizer, rows, loads, _ = setup
    run = root / "models/runs/glimmer-ft-test"
    run.mkdir()
    (run / "run_meta.json").write_text(json.dumps({"revision": "a" * 40, "quant": "nf4"}))
    adapter = gm.add_lora(model)
    adapter.save_pretrained(run / "adapter")
    # Reload onto a fresh base rather than attaching another adapter to the mutated model.
    fresh = tf.LlamaForCausalLM(model.config).eval()
    def load(path, sha, quant):
        loads.append((path, sha, quant))
        return fresh, tokenizer, {"final_norm": "model.norm"}
    monkey = pytest.MonkeyPatch()
    monkey.setattr(gm, "load_base", load)
    try:
        meta = score.score_adapter(run, usd_per_sec=1e-12)
    finally:
        monkey.undo()
    assert meta["quant"] == "nf4" and loads[-1][2] == "nf4"
    assert not (run / "probs/train.jsonl").exists() and not (run / "emb").exists()
    for split in ("calibration", "test", "ood"):
        actual = _read(run / "probs" / f"{split}.jsonl")
        assert [r["id"] for r in actual] == [r["id"] for r in rows if r["split"] == split]
        assert all(r["model"] == "glimmer-ft-glimmer-ft-test" for r in actual)


def test_oom_halves_same_batch_without_duplicates_and_removes_hooks(setup, monkeypatch, caplog):
    root, model, _, rows, _, _ = setup
    original, sizes = gm.last_logits, []
    def forward(model, batch):
        sizes.append(len(batch["input_ids"]))
        if sizes[-1] > 8:
            raise torch.OutOfMemoryError("test allocation")
        return original(model, batch)
    monkeypatch.setattr(gm, "last_logits", forward)
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    out = root / "base"
    meta = score.base_pass(out, usd_per_sec=1e-12)
    assert sizes == [32, 16, 8, 8, 8, 8, 8, 8] and len(emptied) == 2
    assert meta["micro_batch"] == 8 and "halving micro-batch to 16" in caplog.text
    assert not model.model.norm._forward_hooks
    assert sum(len(_read(out / "probs" / f"{s}.jsonl")) for s in score._SPLITS) == len(rows)


def test_single_example_oom_fails_loudly_without_final_outputs(setup, monkeypatch):
    root, model, *_ = setup
    def oom(*_):
        raise torch.OutOfMemoryError("does not fit")
    monkeypatch.setattr(gm, "last_logits", oom)
    with pytest.raises(torch.OutOfMemoryError, match="does not fit"):
        score.base_pass(root / "base", usd_per_sec=1e-12)
    assert not model.model.norm._forward_hooks and not (root / "base/probs").exists()


def test_budget_after_batch_saves_cpu_progress_and_stops_before_next_forward(setup, monkeypatch):
    root, model, *_ = setup
    clock = [0.0]
    monkeypatch.setattr(score.time, "perf_counter", lambda: clock[0])
    calls = []
    def advance(*_):
        calls.append(True)
        clock[0] += 4
    model.register_forward_hook(advance)
    out = root / "base"
    with pytest.raises(RuntimeError, match="budget cap.*row 32"):
        score.base_pass(out, max_usd=3, usd_per_sec=1)
    state = torch.load(out / "score_checkpoint.pt", weights_only=False)
    assert len(calls) == 1 and state["cursor"] == 32 and state["spend"] == 4
    assert len(state["rows"]) == len(state["embeddings"]) == 32
    assert all(v.dtype == np.float16 for v in state["embeddings"])
    assert not (out / "probs").exists()


def test_budget_accounts_for_load_before_first_forward(setup, monkeypatch):
    root, model, _, _, _, _ = setup
    original, clock = gm.load_base, [0.0]
    monkeypatch.setattr(score.time, "perf_counter", lambda: clock[0])
    def slow_load(*args):
        clock[0] = 4
        return original(*args)
    monkeypatch.setattr(gm, "load_base", slow_load)
    model.register_forward_pre_hook(lambda *_: pytest.fail("Budget expired during loading"))
    with pytest.raises(RuntimeError, match="row 0"):
        score.base_pass(root / "base", max_usd=3, usd_per_sec=1)


def test_budget_after_serialization_leaves_no_partial_final_outputs(setup, monkeypatch):
    root, *_ = setup
    original, clock = np.save, [0.0]
    monkeypatch.setattr(score.time, "perf_counter", lambda: clock[0])
    def save(*args):
        original(*args)
        clock[0] = 4
    monkeypatch.setattr(score.np, "save", save)
    out = root / "base"
    with pytest.raises(RuntimeError, match="row 48"):
        score.base_pass(out, max_usd=3, usd_per_sec=1)
    assert list(out.iterdir()) == [out / "score_checkpoint.pt"]


def test_single_decoder_forward_also_captures_embeddings_without_logits_to_keep(setup, monkeypatch):
    root, model, *_ = setup
    original = model.forward
    def forward(input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        return original(input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, **kwargs)
    monkeypatch.setattr(model, "forward", forward)
    calls = []
    model.model.register_forward_pre_hook(lambda *_: calls.append(True))
    score.base_pass(root / "base", usd_per_sec=1e-12)
    assert len(calls) == 2 and not model.model.norm._forward_hooks


@pytest.mark.parametrize("mutation", [{"revision": "b" * 40, "quant": "bf16"},
                                      {"revision": "a" * 40, "quant": "auto"}, {}])
def test_missing_or_incompatible_check_fails_before_model_loading(setup, mutation):
    root, _, _, _, loads, check = setup
    check.write_text(json.dumps(mutation))
    with pytest.raises((ValueError, KeyError)):
        score.base_pass(root / "base")
    assert not loads


def test_base_uses_checked_nf4_rate_and_no_bf16_fallback(setup, monkeypatch):
    root, model, _, _, loads, check = setup
    check.write_text(json.dumps({"revision": "a" * 40, "quant": "nf4"}))
    clock = [0.0]
    monkeypatch.setattr(score.time, "perf_counter", lambda: clock[0])
    def advance(*_):
        clock[0] += .5
    model.register_forward_hook(advance)
    meta = score.base_pass(root / "base")
    assert loads[-1][2] == "nf4" and meta["seconds"] == 1
    assert meta["spend"] == pytest.approx(.000694)


@pytest.mark.parametrize("arguments", [{"max_usd": 0}, {"max_usd": float("inf")},
                                       {"usd_per_sec": -1}, {"usd_per_sec": float("nan")},
                                       {"usd_per_sec": 0}])
def test_invalid_budget_rejected_before_model_loading(setup, arguments):
    root, _, _, _, loads, _ = setup
    with pytest.raises(ValueError):
        score.base_pass(root / "base", **arguments)
    assert not loads
