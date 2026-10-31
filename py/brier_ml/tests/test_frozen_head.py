import hashlib
import json
import sys

import numpy as np
import pytest
import torch
from brier_ml import frozen_head as frozen
from brier_ml import glimmer_data as data
from brier_ml import losses

_SPLITS = ("train", "calibration", "test", "ood")
_REVISION = "a" * 40


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _change(split):
    for index in range(1, 1000):
        change = f"c{index:03}"
        bucket = int.from_bytes(hashlib.sha256(change.encode()).digest()[:4], "big") % 10
        expected = "train" if bucket < 6 else "calibration" if bucket < 8 else "test"
        if expected == split or (split == "ood" and index == 900):
            return change
    raise AssertionError("no fixture change")


def _fixture(tmp_path):
    base, rows, arrays = tmp_path / "base", [], {}
    (base / "emb").mkdir(parents=True)
    (base / "probs").mkdir()
    for split in _SPLITS:
        change, selected, vectors = _change(split), [], []
        for question, canonical in data._OPTIONS.items():
            for scenario in range(1, 17):
                for variant in ("ev", "noev"):
                    options = list(reversed(canonical)) if split == "train" else list(canonical)
                    target = (scenario - 1) % len(options)
                    identity = f"{change}:pusht-s{scenario:02}:{question}:{variant}"
                    family = "occlusion" if split == "ood" else "action_latency"
                    prompt = (f"<system> Brier\n<state>\nCHANGE: id={change} family={family} k=2 desc=\"Move inference\""
                              f"\n</state>\n<question> {data._QUESTIONS[question]}\n<options> "
                              + "  ".join(f"{letter}) {option}" for letter, option in zip("ABCDEF", options))
                              + "\n<answer>")
                    selected.append({"id": identity, "pair_id": f"{change}:pusht-s{scenario:02}",
                                     "change_id": change, "scenario_id": f"pusht-s{scenario:02}", "env": "pusht",
                                     "split": split, "question": question, "variant": variant,
                                     "options": options, "label": "ABCDEF"[target], "prompt": prompt})
                    vectors.append([10 * int(i == target) + 2 for i in range(6)] + [3])
        rows.extend(selected)
        # Deliberately use embedding order rather than the compiled file's order.
        selected.reverse()
        array = np.asarray(vectors[::-1], dtype=np.float16)
        arrays[split] = array
        np.save(base / "emb" / f"{split}.npy", array)
        (base / "emb" / f"{split}.ids.json").write_text(json.dumps([r["id"] for r in selected]))
        (base / "probs" / f"{split}.jsonl").write_text("".join(
            json.dumps({key: row[key] for key in ("id", "pair_id", "split", "question", "variant", "options", "label")}
                       | {"revision": _REVISION, "model": "glimmer-base-zeroshot"}) + "\n" for row in selected))
    compiled = tmp_path / "compiled.jsonl"
    compiled.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return base, compiled, arrays


def _cli(monkeypatch, base, compiled, out):
    monkeypatch.setattr(sys, "argv", ["brier_ml.frozen_head", "--base", str(base),
                                     "--compiled", str(compiled), "--out", str(out)])
    frozen._guide_main()


def test_cli_learns_separable_embeddings_and_preserves_m6_join(tmp_path, monkeypatch):
    base, compiled, arrays = _fixture(tmp_path)
    out = tmp_path / "head"
    _cli(monkeypatch, base, compiled, out)
    saved = torch.load(out / "head.pt", weights_only=True)
    assert saved["format"] == "brier-glimmer-frozen-head-v2"
    assert saved["revision"] == _REVISION
    assert saved["dimension"] == 7
    assert all(value.device.type == "cpu" for value in saved["state_dict"].values())
    assert saved["mean"].dtype == saved["std"].dtype == torch.float32
    assert torch.equal(saved["mean"], torch.from_numpy(arrays["train"]).float().mean(0))
    assert saved["std"][-1].item() == pytest.approx(1e-6)
    for split in _SPLITS[1:]:
        results = [json.loads(line) for line in (out / "probs" / f"{split}.jsonl").read_text().splitlines()]
        assert [row["id"] for row in results] == json.loads((base / "emb" / f"{split}.ids.json").read_text())
        assert sum(max(row["probs"], key=row["probs"].get) == row["label"] for row in results) / len(results) > 0.95
        for row in results:
            assert set(row) == {"id", "pair_id", "split", "question", "variant", "options", "label",
                                "logits", "probs", "model", "revision"}
            assert row["model"] == "glimmer-frozen-head" and row["revision"] == _REVISION
            assert len(row["logits"]) == 6 and np.isfinite(row["logits"]).all()
            k = len(row["options"])
            assert list(row["probs"]) == list("ABCDEF"[:k])
            assert list(row["probs"].values()) == pytest.approx(torch.tensor(row["logits"][:k]).softmax(-1).tolist())
    assert not (out / "probs" / "train.jsonl").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["device"] == "cpu" and meta["seed"] == 2026
    assert meta["compiled_sha256"] == hashlib.sha256(compiled.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        _cli(monkeypatch, base, compiled, out)


def test_early_stopping_uses_only_calibration_regressed_and_restores_best(tmp_path):
    base, compiled, _ = _fixture(tmp_path)
    inputs, _ = frozen._guide_inputs(base, compiled)
    cal = inputs["calibration"]
    for index, row in enumerate(cal[0]):
        if row["question"] == "regressed":
            cal[3][index] = 1 - cal[3][index]
    model, meta = frozen._guide_fit(inputs)
    history = meta["history"]
    assert 6 <= len(history) < 40
    assert len(history) == meta["best_epoch"] + 5
    rows = [i for i, row in enumerate(cal[0]) if row["question"] == "regressed"]
    with torch.no_grad():
        logits = frozen._guide_logits(model, (cal[1][rows] - meta["mean"]) / meta["std"], [cal[0][i] for i in rows])
    assert torch.nn.functional.cross_entropy(logits[:, :2], cal[3][rows]).item() == pytest.approx(
        meta["calibration_regressed_nll"])


def test_training_uses_proper_loss_with_shuffled_ordinal_masks_and_fixed_recipe(tmp_path, monkeypatch):
    base, compiled, _ = _fixture(tmp_path)
    inputs, _ = frozen._guide_inputs(base, compiled)
    train = inputs["train"]
    inputs["train"] = (train[0] * 4, *(tensor.repeat((4,) + (1,) * (tensor.ndim - 1)) for tensor in train[1:]))
    original, calls, optimizers = losses.proper_loss, [], []
    adamw = torch.optim.AdamW

    def proper(logits, valid, labels, score, ordinal):
        calls.append((valid.clone(), score.clone(), ordinal.clone()))
        return original(logits, valid, labels, score, ordinal)

    def optimizer(parameters, **kwargs):
        optimizers.append(kwargs)
        return adamw(parameters, **kwargs)

    monkeypatch.setattr(losses, "proper_loss", proper)
    monkeypatch.setattr(torch.optim, "AdamW", optimizer)
    model, meta = frozen._guide_fit(inputs)
    assert optimizers == [{"lr": 1e-3, "weight_decay": 1e-4}]
    assert model["trunk"][0].out_features == 256
    assert isinstance(model["trunk"][1], torch.nn.GELU)
    assert model["trunk"][2].p == 0.1
    assert calls and all(len(mask) <= 256 for mask, _, _ in calls)
    assert max(len(mask) for mask, _, _ in calls) == 256
    for mask, score, ordinal in calls:
        assert set(mask.sum(-1).tolist()) <= {2, 5, 6}
        assert torch.all(ordinal[~score] == -1)
        assert torch.all(ordinal[score, :5] == torch.tensor([4, 3, 2, 1, 0]))
    assert meta["history"][-1]["epoch"] <= 40


def test_standardizer_and_fit_never_use_test_ood_rows_or_heldout_statistics(tmp_path):
    base, compiled, arrays = _fixture(tmp_path)
    inputs, _ = frozen._guide_inputs(base, compiled)
    first, meta = frozen._guide_fit(inputs)
    for split in ("test", "ood"):
        inputs[split][1].fill_(50000)
        inputs[split][3].zero_()
    second, other = frozen._guide_fit(inputs)
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())
    assert torch.equal(meta["mean"], other["mean"])
    assert torch.equal(meta["std"], torch.from_numpy(arrays["train"]).float().std(0, unbiased=False).clamp_min(1e-6))
    inputs["calibration"][1].fill_(100)
    _, shifted_calibration = frozen._guide_fit(inputs)
    assert torch.equal(meta["mean"], shifted_calibration["mean"])
    assert torch.equal(meta["std"], shifted_calibration["std"])


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "wrong_split", "dtype", "rank", "dimension",
                                       "nan", "empty", "revision", "mismatched_revision", "metadata"])
def test_rejects_bad_embedding_joins_and_scoring_provenance(tmp_path, corruption):
    base, compiled, arrays = _fixture(tmp_path)
    split = "ood"
    ids_path = base / "emb" / f"{split}.ids.json"
    ids = json.loads(ids_path.read_text())
    if corruption == "missing":
        ids.pop()
    elif corruption == "duplicate":
        ids[-1] = ids[0]
    elif corruption == "wrong_split":
        ids[-1] = json.loads((base / "emb" / "train.ids.json").read_text())[0]
    ids_path.write_text(json.dumps(ids))
    array = arrays[split]
    if corruption == "dtype":
        array = array.astype(np.float32)
    elif corruption == "rank":
        array = array.ravel()
    elif corruption == "dimension":
        array = array[:, :-1]
    elif corruption == "nan":
        array[0, 0] = np.nan
    elif corruption == "empty":
        array = array[:, :0]
    np.save(base / "emb" / f"{split}.npy", array)
    if corruption in {"revision", "mismatched_revision", "metadata"}:
        path = base / "probs" / f"{split}.jsonl"
        scored = [json.loads(line) for line in path.read_text().splitlines()]
        if corruption == "metadata":
            scored[0]["label"] = "A" if scored[0]["label"] != "A" else "B"
        else:
            for row in scored:
                row["revision"] = "invalid" if corruption == "revision" else "b" * 40
        path.write_text("".join(json.dumps(row) + "\n" for row in scored))
    with pytest.raises(ValueError):
        frozen._guide_inputs(base, compiled)


def test_rejects_missing_training_question_or_calibration_regressed(tmp_path):
    base, compiled, _ = _fixture(tmp_path)
    inputs, _ = frozen._guide_inputs(base, compiled)
    original = inputs["train"]
    inputs["train"] = ([row for row in original[0] if row["question"] != "failure_mode"], *original[1:])
    with pytest.raises(ValueError, match="all three"):
        frozen._guide_fit(inputs)
    inputs["train"] = original
    inputs["calibration"] = ([], *inputs["calibration"][1:])
    with pytest.raises(ValueError, match="calibration requires"):
        frozen._guide_fit(inputs)
