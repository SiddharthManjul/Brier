import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from brier_ml import glimmer_data as data
from tokenizers import Tokenizer

_FIXTURE = Path(__file__).parent / "fixtures" / "glimmer_bpe.json"
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


class _BPE:
    def __init__(self, raw: dict[str, Any] | None = None) -> None:
        self.backend = Tokenizer.from_file(str(_FIXTURE)) if raw is None else Tokenizer.from_str(json.dumps(raw))
        self.pad_token_id = self.backend.token_to_id("[PAD]")
        self.eos_token_id = self.backend.token_to_id("[EOS]")
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        self.calls.append((text, add_special_tokens))
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}


def _row(question: str = "regressed", split: str = "train", env: str = "pusht",
         variant: str = "ev", family: str | None = "action_latency") -> dict[str, Any]:
    changes = (f"c{i:03}" for i in range(900 if split == "ood" else 1, 1000))
    change = next(c for c in changes if split == "ood" or
                  ("train" if (b := int.from_bytes(hashlib.sha256(c.encode()).digest()[:4], "big") % 10) < 6
                   else "calibration" if b < 8 else "test") == split)
    scenario = f"{env}-s01"
    options = _OPTIONS[question].copy()
    if split == "train":
        options.reverse()
    prefix = "compiled\n" if family is None else f'CHANGE: id={change} family={family} k=2 desc="Move inference"\n'
    prompt = ("<system> Brier Ω\n<state>\n" + prefix + "</state>\n<question> "
              + _QUESTIONS[question] + "\n<options> "
              + "  ".join(f"{letter}) {value}" for letter, value in zip("ABCDEF", options)) + "\n<answer>")
    return {"id": f"{change}:{scenario}:{question}:{variant}", "pair_id": f"{change}:{scenario}",
            "change_id": change, "scenario_id": scenario, "env": env, "variant": variant,
            "split": split, "question": question, "options": options, "prompt": prompt, "label": "B"}


def _write(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "compiled.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    return path


def test_load_examples_all_splits_questions_variants_preserves_raw_prompt(tmp_path: Path) -> None:
    rows = [_row(question, split, env, variant, "occlusion" if split == "ood" else "action_latency")
            for split in ["train", "calibration", "test", "ood"]
            for question in _QUESTIONS for env in ["pusht", "aloha"] for variant in ["ev", "noev"]]
    path = _write(tmp_path, rows)
    assert data.load_examples(path, ["test", "ood"]) == [r for r in rows if r["split"] in {"test", "ood"}]
    assert data.load_examples(path, ["train", "calibration", "test", "ood"]) == rows


@pytest.mark.parametrize("field", sorted(_row()))
@pytest.mark.parametrize("mutation", ["missing", "wrong_type"])
def test_load_examples_requires_each_typed_field(tmp_path: Path, field: str, mutation: str) -> None:
    row = _row()
    if mutation == "missing":
        del row[field]
    else:
        row[field] = {} if field == "options" else ["invalid"]
    with pytest.raises(ValueError, match="invalid compiled line 1"):
        data.load_examples(_write(tmp_path, [row]), ["train"])


@pytest.mark.parametrize("field,value", [
    ("id", "c000:pusht-s01:regressed:ev"), ("id", "c001:pusht-s17:regressed:ev"),
    ("id", "c001/pusht-s01/regressed/cheap"), ("id", "c001:aloha_transfer_cube-s01:regressed:ev"),
    ("pair_id", "c999:pusht-s01"), ("change_id", "c999"), ("scenario_id", "pusht-s02"),
    ("env", "aloha"), ("question", "other"), ("variant", "cheap"), ("split", "validation"),
    ("split", "test"), ("options", ["yes", "yes"]), ("options", ["perhaps", "no"]),
    ("options", ["yes", "no", "extra"]), ("label", "C"), ("label", "yes"), ("label", ""),
    ("prompt", "<answer>"), ("extra", 0),
])
def test_load_examples_rejects_unknown_or_inconsistent_values(tmp_path: Path, field: str, value: Any) -> None:
    row = _row()
    row[field] = value
    with pytest.raises(ValueError):
        data.load_examples(_write(tmp_path, [row]), ["train", "test"])


@pytest.mark.parametrize("kind", ["duplicate", "leak", "unselected_invalid", "empty", "json", "blank"])
def test_load_examples_validates_whole_file_before_filtering(tmp_path: Path, kind: str) -> None:
    row = _row()
    rows = [row, row.copy()] if kind == "duplicate" else [row]
    if kind == "leak":
        other = _row(split="ood", variant="noev", family=None)
        other.update({key: row[key] for key in ["change_id", "pair_id", "scenario_id"]})
        other["id"] = row["id"].removesuffix(":ev") + ":noev"
        rows.append(other)
    if kind == "unselected_invalid":
        rows.append({**_row(split="test"), "label": "F"})
    path = _write(tmp_path, [] if kind == "empty" else rows)
    if kind in {"json", "blank"}:
        path.write_text("not json" if kind == "json" else "\n")
    with pytest.raises(ValueError):
        data.load_examples(path, ["train"])


@pytest.mark.parametrize("splits", [[], ["unknown"], "train"])
def test_load_examples_requires_known_requested_splits(tmp_path: Path, splits: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        data.load_examples(_write(tmp_path, [_row()]), splits)


def test_load_examples_requires_selected_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no examples"):
        data.load_examples(_write(tmp_path, [_row()]), ["calibration"])


@pytest.mark.parametrize("family", ["occlusion", "chunk_len", "action_latency + chunk_len", "action_latency + family=occlusion"])
def test_ood_family_split_is_checked_when_change_families_are_present(tmp_path: Path, family: str) -> None:
    row = _row(split="ood", family=family)
    assert data.load_examples(_write(tmp_path, [row]), ["ood"]) == [row]
    row["split"] = "train"
    with pytest.raises(ValueError, match="split or families"):
        data.load_examples(_write(tmp_path, [row]), ["train"])


def test_ood_metadata_without_family_line_and_description_text_are_not_reinterpreted(tmp_path: Path) -> None:
    row = _row(split="ood", family=None)
    assert data.load_examples(_write(tmp_path, [row]), ["ood"]) == [row]
    row = _row()
    row["prompt"] = row["prompt"].replace("Move inference", "do not use family=occlusion + chunk_len")
    assert data.load_examples(_write(tmp_path, [row]), ["train"]) == [row]
    row = _row(split="ood", family="action_latency")
    with pytest.raises(ValueError, match="split or families"):
        data.load_examples(_write(tmp_path, [row]), ["ood"])


@pytest.mark.parametrize("mutation", ["question", "options", "newline", "canonical", "unknown_family"])
def test_prompt_text_metadata_and_nontrain_option_order_agree(tmp_path: Path, mutation: str) -> None:
    row = _row(split="test")
    if mutation == "question":
        row["prompt"] = row["prompt"].replace(_QUESTIONS["regressed"], "Wrong question?")
    elif mutation == "options":
        row["prompt"] = row["prompt"].replace("A) yes  B) no", "A) no  B) yes")
    elif mutation == "newline":
        row["prompt"] += "\n"
    elif mutation == "canonical":
        row["options"].reverse()
        row["prompt"] = row["prompt"].replace("A) yes  B) no", "A) no  B) yes")
    else:
        row["prompt"] = row["prompt"].replace("family=action_latency", "family=unknown")
    with pytest.raises(ValueError):
        data.load_examples(_write(tmp_path, [row]), ["test"])


def test_letter_token_ids_uses_actual_local_bpe_and_requires_all_six_atomic_letters() -> None:
    tokenizer = _BPE()
    ids = data.letter_token_ids(tokenizer)
    assert len(ids) == len(set(ids)) == 6
    assert all(tokenizer.backend.decode([token]) == " " + letter for token, letter in zip(ids, "ABCDEF"))
    raw = json.loads(_FIXTURE.read_text())
    raw["model"]["merges"].remove(["Ġ", "F"])
    with pytest.raises(ValueError, match="single tokens"):
        data.letter_token_ids(_BPE(raw))


def test_tokenize_checks_every_letter_preserves_raw_text_and_records_padding_source() -> None:
    rows = [_row(question) for question in _QUESTIONS]
    tokenizer = _BPE()
    encoded = data.tokenize(rows, tokenizer)
    assert len(tokenizer.calls) == sum(1 + len(row["options"]) for row in rows)
    assert all(special for _, special in tokenizer.calls)
    for row, item in zip(rows, encoded, strict=True):
        assert item["input_ids"] == tokenizer.encode(row["prompt"], add_special_tokens=True)
        assert item["input_ids"][0] == tokenizer.backend.token_to_id("[BOS]")
        assert item["prompt"] == row["prompt"]
        assert item["k"] == len(row["options"]) and item["label_idx"] == 1
        assert item["pad_id"] == tokenizer.pad_token_id and item["pad_source"] == "pad"
    assert encoded[-1]["pos_to_ordinal"] == [4, 3, 2, 1, 0]
    assert all(row["pos_to_ordinal"] == [] for row in encoded[:-1])
    tokenizer.pad_token_id = None
    fallback = data.tokenize([rows[0]], tokenizer)[0]
    assert (fallback["pad_id"], fallback["pad_source"]) == (tokenizer.eos_token_id, "eos")
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="pad token or an EOS"):
        data.tokenize([rows[0]], tokenizer)


def test_tokenize_boundary_failure_uses_real_bpe_with_default_eos_processing() -> None:
    raw = json.loads(_FIXTURE.read_text())
    raw["post_processor"]["single"].append({"SpecialToken": {"id": "[EOS]", "type_id": 0}})
    raw["post_processor"]["special_tokens"]["[EOS]"] = {
        "id": "[EOS]", "ids": [2], "tokens": ["[EOS]"]}
    with pytest.raises(ValueError, match="boundary fails.*letter A.*brier compile"):
        data.tokenize([_row()], _BPE(raw))


def test_tokenize_hard_limit_inclusive_and_never_truncates() -> None:
    row, tokenizer = _row(), _BPE()
    length = len(tokenizer.encode(row["prompt"], add_special_tokens=True))
    assert len(data.tokenize([row], tokenizer, length)[0]["input_ids"]) == length
    with pytest.raises(ValueError, match="exceeds max_len"):
        data.tokenize([row], tokenizer, length - 1)
    row["prompt"] = row["prompt"].replace("Brier Ω", "Ω" * 2000)
    with pytest.raises(ValueError, match="1536"):
        data.tokenize([row], tokenizer)
    for maximum in [0, -1, 1537, True]:
        with pytest.raises(ValueError, match="max_len"):
            data.tokenize([_row()], tokenizer, maximum)
    with pytest.raises(ValueError):
        data.tokenize([], tokenizer)


def test_collate_left_padding_masks_positions_and_ordinal_padding() -> None:
    rows = data.tokenize([_row(question) for question in _QUESTIONS], _BPE())
    batch = data.collate(rows, rows[0]["pad_id"])
    assert set(batch) == {"input_ids", "attention_mask", "position_ids", "label_idx", "k", "ordinal_map", "valid"}
    width = max(len(row["input_ids"]) for row in rows)
    assert batch["input_ids"].shape == (3, width)
    for i, row in enumerate(rows):
        length = len(row["input_ids"])
        assert batch["input_ids"][i, -length:].tolist() == row["input_ids"]
        assert batch["attention_mask"][i].tolist() == [0] * (width - length) + [1] * length
        assert batch["position_ids"][i].tolist() == [0] * (width - length) + list(range(length))
        assert batch["valid"][i].tolist() == [j < row["k"] for j in range(6)]
    assert batch["ordinal_map"].tolist() == [[-1] * 6, [-1] * 6, [4, 3, 2, 1, 0, -1]]
    assert batch["label_idx"].tolist() == [1, 1, 1]
    assert batch["k"].tolist() == [2, 6, 5]
    assert batch["valid"].dtype == torch.bool
    assert all(value.dtype == torch.long for key, value in batch.items() if key != "valid")


@pytest.mark.parametrize("field,value", [("input_ids", []), ("input_ids", [-1]), ("input_ids", [True]),
                                          ("k", 3), ("k", True), ("label_idx", 2),
                                          ("pos_to_ordinal", [0, 0]), ("pos_to_ordinal", [True, False])])
def test_collate_rejects_invalid_encoded_rows(field: str, value: Any) -> None:
    row = data.tokenize([_row()], _BPE())[0]
    row[field] = value
    with pytest.raises(ValueError):
        data.collate([row], 0)


def test_collate_rejects_empty_batch_missing_score_map_or_invalid_pad() -> None:
    row = data.tokenize([_row("success_bucket")], _BPE())[0]
    row["pos_to_ordinal"] = []
    for batch, pad in [([], 0), ([row], 0), ([row], -1), ([row], True)]:
        with pytest.raises(ValueError):
            data.collate(batch, pad)


def test_length_grouped_order_matches_seeded_chunk_algorithm() -> None:
    lengths, micro, seed = list(range(400, 0, -1)), 3, 2026
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(lengths)).tolist()
    batches = []
    for offset in range(0, len(lengths), 64 * micro):
        chunk = sorted(shuffled[offset:offset + 64 * micro], key=lengths.__getitem__)
        batches.extend(chunk[start:start + micro] for start in range(0, len(chunk), micro))
    expected = [index for b in rng.permutation(len(batches)) for index in batches[b]]
    result = data.length_grouped_order(lengths, micro, seed)
    assert result == expected == data.length_grouped_order(lengths, micro, seed)
    assert sorted(result) == list(range(len(lengths)))
    assert result != data.length_grouped_order(lengths, micro, seed + 1)
    assert data.length_grouped_order([], micro, seed) == []


@pytest.mark.parametrize("lengths,micro,seed", [([0], 1, 0), ([True], 1, 0), ([1], 0, 0),
                                                ([1], True, 0), ([1], 1, -1), ([1], 1, True)])
def test_length_grouped_order_rejects_invalid_arguments(lengths: list[int], micro: int, seed: int) -> None:
    with pytest.raises(ValueError):
        data.length_grouped_order(lengths, micro, seed)


def test_calibration_partition_reserves_nineteen_label_free_changes_invariant_to_order_and_duplicates() -> None:
    ids = [f"c{i:03}" for i in range(1, 41)]
    labels = {change: i % 2 for i, change in enumerate(ids)}
    fit, risk = data.calibration_partition(labels)
    assert (fit, risk) == data.calibration_partition(reversed(ids + ids))
    assert (fit, risk) == data.calibration_partition({change: 1 - label for change, label in labels.items()})
    assert len(risk) == 19 and len(fit) == 21
    assert set(fit).isdisjoint(risk) and set(fit) | set(risk) == set(ids)
    assert fit == sorted(fit) and risk == sorted(risk)
    expected = sorted(ids, key=lambda change: (hashlib.sha256(f"risk-calibration:{change}:2026".encode()).digest(), change))
    assert risk == sorted(expected[:19])


def test_calibration_partition_uses_id_as_hash_collision_tie_break(monkeypatch) -> None:
    class Digest:
        def digest(self):
            return b"same"

    monkeypatch.setattr(data.hashlib, "sha256", lambda _: Digest())
    fit, risk = data.calibration_partition(reversed([f"c{i:03}" for i in range(1, 21)]))
    assert fit == ["c020"] and risk == [f"c{i:03}" for i in range(1, 20)]


@pytest.mark.parametrize("count", [0, 1, 19])
def test_calibration_partition_requires_twenty_unique_changes(count: int) -> None:
    with pytest.raises(ValueError, match="at least 20 unique"):
        data.calibration_partition([f"c{i + 1:03}" for i in range(count)] * 2)


@pytest.mark.parametrize("invalid", ["c000", "c01", "c0001", "C001", "cxyz", 1, None])
def test_calibration_partition_requires_canonical_change_ids(invalid: Any) -> None:
    with pytest.raises(ValueError, match="canonical cNNN"):
        data.calibration_partition([f"c{i:03}" for i in range(1, 21)] + [invalid])
