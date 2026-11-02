import ast
import hashlib
import json
import logging
import math
import re
import tempfile
import types
import uuid
from contextlib import contextmanager
from pathlib import Path

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


def test_heldout_ids_score_complete_extension_without_overwriting_existing_outputs(setup, monkeypatch):
    root, model, tokenizer, _, loads, _ = setup
    run = root / "models/runs/glimmer-ft-test"
    run.mkdir()
    (run / "run_meta.json").write_text(json.dumps({"revision": "a" * 40, "quant": "bf16"}))
    gm.add_lora(model).save_pretrained(run / "adapter")
    fresh = tf.LlamaForCausalLM(model.config).eval()
    monkeypatch.setattr(gm, "load_base", lambda *args: (loads.append(args) or (fresh, tokenizer, {"final_norm": "model.norm"})))
    rows = []
    for scenario in range(1, 17):
        for question in ("regressed", "failure_mode", "success_bucket"):
            for variant in ("ev", "noev"):
                row = _row(question, "calibration", "pusht", variant)
                old, old_scenario = row["change_id"], row["scenario_id"]
                for key in ("id", "pair_id", "prompt"):
                    row[key] = row[key].replace(old, "c161").replace(old_scenario, f"pusht-s{scenario:02d}")
                row.update(change_id="c161", scenario_id=f"pusht-s{scenario:02d}")
                rows.append(row)
    compiled = root / "data/compiled.jsonl"
    compiled.write_text("".join(json.dumps(row) + "\n" for row in rows))
    ids = root / "data/new-ids.json"
    ids.write_text(json.dumps(["c161"]))
    (run / "probs").mkdir()
    (run / "probs/calibration.jsonl").write_text("preserved-original\n")
    result = score.score_adapter(run, ids_from=ids, usd_per_sec=1e-12)
    output = root / result["output"]
    assert result["rows"] == 96 and output != run
    assert (run / "probs/calibration.jsonl").read_text() == "preserved-original\n"
    assert {row["id"] for row in _read(output / "probs/calibration.jsonl")} == {row["id"] for row in rows}
    before = len(loads)
    with pytest.raises(FileExistsError, match="reconcile"):
        score.score_adapter(run, ids_from=ids, usd_per_sec=1e-12)
    assert len(loads) == before


def test_heldout_ids_reject_partial_changes_before_loading_model(setup):
    root, _model, _, _, loads, _ = setup
    run = root / "models/runs/glimmer-ft-test"
    run.mkdir()
    (run / "run_meta.json").write_text(json.dumps({"revision": "a" * 40, "quant": "bf16"}))
    ids = root / "data/new-ids.json"
    ids.write_text(json.dumps(["c161"]))
    row = _row("regressed", "calibration", "pusht", "ev")
    old = row["change_id"]
    for key in ("id", "pair_id", "prompt"):
        row[key] = row[key].replace(old, "c161")
    row["change_id"] = "c161"
    (root / "data/compiled.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="partial changes"):
        score.score_adapter(run, ids_from=ids)
    assert not loads


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


def _modal_ast(names, namespace):
    nodes = []
    for node in ast.parse(Path(modal_train.__file__).read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = [d for d in node.decorator_list if isinstance(d, ast.Name) and d.id == "contextmanager"]
            nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), modal_train.__file__, "exec"), namespace)  # noqa: S102 - isolates real entrypoints from the Modal SDK
    return namespace


def _extension_fixture():
    rows = []
    for scenario in range(1, 17):
        for question in ("regressed", "failure_mode", "success_bucket"):
            for variant in ("ev", "noev"):
                row = _row(question, "calibration", "pusht", variant)
                change, old = row["change_id"], row["scenario_id"]
                for key in ("id", "pair_id", "prompt"):
                    row[key] = row[key].replace(change, "c161").replace(old, f"pusht-s{scenario:02d}")
                row.update(change_id="c161", scenario_id=f"pusht-s{scenario:02d}")
                rows.append(row)
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def _modal_offline(monkeypatch, tokenizer, records, ids=b'["c161"]', compiled=None):
    events = []
    class Volume:
        def __init__(self, files, name):
            self.files, self.name = files, name
        def read_file(self, name):
            events.append(("read", self.name, name))
            if name not in self.files:
                raise FileNotFoundError(name)
            return [self.files[name]]
    data_files = {"guide3/compiled.jsonl": _extension_fixture() if compiled is None else compiled}
    if ids is not None:
        data_files["guide3/ids.json"] = ids
    model_files = {"runs/glimmer-ft-v1/run_meta.json": json.dumps({"quant": "bf16", "revision": "a" * 40}).encode(),
                   "runs/check.json": b'{"quant":"bf16"}', "runs/spend.json": json.dumps(records).encode(),
                   "base/muse-glimmer-30b/revision.txt": ("a" * 40).encode()}
    model_files.update({f"base/muse-glimmer-30b/{name}": b"{}" for name in modal_train._TOKENIZER_FILES[:3]})
    namespace = {"Path": Path, "json": json, "math": math, "re": re, "hashlib": hashlib, "logging": logging,
                 "tempfile": tempfile, "LOGGER": logging.getLogger("offline-modal"), "RATES": modal_train.RATES,
                 "BASE_DIR": modal_train.BASE_DIR, "_TOKENIZER_FILES": modal_train._TOKENIZER_FILES,
                 "MODELS": Volume(model_files, "models"), "DATA": Volume(data_files, "data")}
    def tokenizer_load(*_args, **kwargs):
        assert kwargs == {"local_files_only": True}
        events.append(("cpu_tokenizer",))
        return tokenizer
    monkeypatch.setattr(tf.AutoTokenizer, "from_pretrained", tokenizer_load)
    for name in ("download_weights", "check", "base_pass_fn", "train_fn", "score_fn"):
        function = types.SimpleNamespace()
        def options(_name=name, _function=function, **kwargs):
            events.append(("gpu_options", _name, kwargs))
            return _function
        def remote(*args, _name=name, **kwargs):
            events.append(("gpu_remote", _name, args, kwargs))
            return {"offline": True}
        function.with_options, function.remote = options, remote
        namespace[name] = function
    return _modal_ast({"main", "_validated_spend", "_extension_preflight"}, namespace), events, data_files


def test_modal_extension_preflight_uses_real_validation_and_tokenizer_before_forwarding(monkeypatch):
    namespace, events, files = _modal_offline(monkeypatch, _BPE(), [{"step": "score", "status": "complete", "spend": .3}])
    namespace["main"]("score", max_usd=1, ids_from="/data/guide3/ids.json")
    call = next(event for event in events if event[0] == "gpu_remote")
    assert call[1:3] == ("score_fn", ("glimmer-ft-v1", 1))
    assert call[3]["ids_from"] == "/data/guide3/ids.json"
    proof = call[3]["input_sha256"]
    assert proof["compiled"] == hashlib.sha256(files["guide3/compiled.jsonl"]).hexdigest()
    assert proof["ids"] == hashlib.sha256(files["guide3/ids.json"]).hexdigest()
    assert proof["rows"] == 96 and proof["revision"] == "a" * 40
    assert set(proof["tokenizer"]) == set(modal_train._TOKENIZER_FILES[:3])
    assert next(i for i, e in enumerate(events) if e[0] == "cpu_tokenizer") < next(i for i, e in enumerate(events) if e[0] == "gpu_options")
    assert ("read", "data", "compiled.jsonl") not in events


@pytest.mark.parametrize("fault", ["missing", "malformed", "duplicate", "partial", "tokenizer", "traversal"])
def test_modal_invalid_extension_fails_before_any_gpu_request(monkeypatch, fault):
    ids = None if fault == "missing" else b"not-json" if fault == "malformed" else b'["c161","c161"]' if fault == "duplicate" else b'["c161"]'
    compiled = _extension_fixture()
    if fault == "partial":
        compiled = compiled.splitlines()[0] + b"\n"
    tokenizer = _BPE()
    if fault == "tokenizer":
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = None
    namespace, events, _ = _modal_offline(monkeypatch, tokenizer, [], ids=ids, compiled=compiled)
    with pytest.raises((ValueError, FileNotFoundError)):
        namespace["main"]("score", max_usd=1, ids_from="/data/../bad.json" if fault == "traversal" else "/data/guide3/ids.json")
    assert not any(event[0].startswith("gpu_") for event in events)


@pytest.mark.parametrize("record", [{"step": "score", "status": "reserved", "spend": .2},
    {"step": "train:other", "status": "running", "spend": .2}, {"step": "score", "spend": .2},
    {"step": "score", "status": "complete", "spend": -1}, {"step": "score", "status": "complete", "spend": float("nan")},
    {"step": "score", "status": "complete", "spend": float("inf")}, {"step": "score", "status": "complete", "spend": 1.5}])
def test_modal_unresolved_invalid_or_exhausted_spend_blocks_before_launch(monkeypatch, record):
    namespace, events, _ = _modal_offline(monkeypatch, _BPE(), [record])
    with pytest.raises((ValueError, RuntimeError)):
        namespace["main"]("score", max_usd=1, ids_from="/data/guide3/ids.json")
    assert not any(event[0].startswith("gpu_") for event in events)
    assert not any(event[:2] == ("read", "data") for event in events)


@pytest.mark.parametrize("records", [[{"step": "score", "status": "reserved", "spend": .2}],
    [{"step": "score", "status": "complete", "spend": -1}], [{"step": "score", "status": "complete", "spend": float("nan")}],
    [{"step": "score", "status": "complete", "spend": 1.5}]])
def test_modal_attempt_rejects_invalid_reservations_without_writing(tmp_path, records):
    ledger = tmp_path / "spend.json"
    ledger.write_text(json.dumps(records))
    before = ledger.read_bytes()
    commits = []
    namespace = {"Path": lambda *_: ledger, "json": json, "math": math, "logging": logging,
                 "LOGGER": logging.getLogger("offline-modal"), "RATES": modal_train.RATES,
                 "time": types.SimpleNamespace(perf_counter=lambda: 0.0), "uuid": uuid,
                 "contextmanager": contextmanager, "MODELS": types.SimpleNamespace(reload=lambda: None, commit=lambda: commits.append(True))}
    _modal_ast({"_attempt", "_validated_spend"}, namespace)
    with pytest.raises((ValueError, RuntimeError)), namespace["_attempt"]("score", "bf16", 1, 2):
        pytest.fail("Invalid reservation reached paid work")
    assert ledger.read_bytes() == before and not commits


def test_modal_legacy_score_has_no_extension_option_or_tokenizer_preflight(monkeypatch):
    namespace, events, _ = _modal_offline(monkeypatch, _BPE(), [{"step": "score", "status": "complete", "spend": .3}])
    namespace["main"]("score", max_usd=1)
    call = next(event for event in events if event[0] == "gpu_remote")
    assert call[1:4] == ("score_fn", ("glimmer-ft-v1", 1), {})
    assert not any(event[0] == "cpu_tokenizer" or event[:2] == ("read", "data") for event in events)


@pytest.mark.parametrize("fault", [None, "compiled", "ids", "tokenizer", "revision", "missing-proof", "asset-path"])
def test_modal_extension_remote_rechecks_prelaunch_proof_before_reserving(tmp_path, monkeypatch, fault):
    namespace, events, files = _modal_offline(monkeypatch, _BPE(), [])
    namespace["main"]("score", max_usd=1, ids_from="/data/guide3/ids.json")
    proof = next(event for event in events if event[0] == "gpu_remote")[3]["input_sha256"]
    data = tmp_path / "data/guide3"
    data.mkdir(parents=True)
    for name, contents in files.items():
        (data / Path(name).name).write_bytes(contents)
    base = tmp_path / "models/base/muse-glimmer-30b"
    base.mkdir(parents=True)
    for name in modal_train._TOKENIZER_FILES[:3]:
        (base / name).write_bytes(b"{}")
    (base / "revision.txt").write_text("a" * 40)
    run = tmp_path / "models/runs/glimmer-ft-v1"
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"quant": "bf16", "revision": "a" * 40}))
    if fault == "compiled":
        path = data / "compiled.jsonl"
        path.write_bytes(path.read_bytes().replace(b"<system> ", b"<system> changed ", 1))
    elif fault == "ids":
        path = data / "ids.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif fault == "tokenizer":
        (base / "tokenizer.json").write_bytes(b"changed")
    elif fault == "revision":
        (base / "revision.txt").write_text("b" * 40)
    elif fault == "missing-proof":
        proof = None
    elif fault == "asset-path":
        proof["tokenizer"]["../unexpected"] = "0" * 64
    calls = []
    @contextmanager
    def attempt(*_args):
        calls.append("reservation")
        yield {}, lambda: .9
    def adapter(*args, **kwargs):
        calls.append((args, kwargs))
        return {"rows": 96}
    monkeypatch.setattr(score, "score_adapter", adapter)
    remote = {"Path": lambda value: tmp_path / str(value).lstrip("/"), "json": json, "hashlib": hashlib,
              "BASE_DIR": modal_train.BASE_DIR, "_TOKENIZER_FILES": modal_train._TOKENIZER_FILES,
              "RATES": modal_train.RATES, "MODELS": types.SimpleNamespace(reload=lambda: None), "_attempt": attempt}
    _modal_ast({"score_fn"}, remote)
    if fault:
        with pytest.raises(ValueError):
            remote["score_fn"]("glimmer-ft-v1", 1, "/data/guide3/ids.json", proof)
        assert not calls
    else:
        result = remote["score_fn"]("glimmer-ft-v1", 1, "/data/guide3/ids.json", proof)
        assert calls[0] == "reservation" and result["input_sha256"] == proof
        assert calls[1][1]["ids_from"] == data / "ids.json"
