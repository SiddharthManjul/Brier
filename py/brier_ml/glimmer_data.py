from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LETTERS = "ABCDEF"
_SPLITS = {"train", "calibration", "test", "ood"}
_FIELDS = {"id", "pair_id", "change_id", "scenario_id", "env", "variant", "split",
           "question", "options", "prompt", "label"}
_OPTIONS = {
    "regressed": ("yes", "no"),
    "failure_mode": ("none", "stall or timeout", "overshoot or misalignment",
                     "grasp or drop", "collision", "oscillation"),
    "success_bucket": ("0-19%", "20-39%", "40-59%", "60-79%", "80-100%"),
}
_QUESTIONS = {
    "regressed": "Will the expensive evaluation regress by more than 15 points?",
    "failure_mode": "If the expensive evaluation regresses, what is the main failure mode?",
    "success_bucket": "What will the expensive success rate be?",
}
_FAMILIES = {"obs_noise", "action_latency", "action_scale", "action_bias", "camera_shift",
             "occlusion", "brightness", "frame_drop", "chunk_len", "weight_noise", "benign"}


def _validate(row: Any, identities: set[str], splits: dict[str, str]) -> None:
    if (not isinstance(row, dict) or set(row) != _FIELDS
            or any(not isinstance(row[key], str) for key in _FIELDS - {"options"})):
        raise ValueError("compiled example must contain exactly the eleven typed fields")
    match = re.fullmatch(
        r"(c[0-9]{3,}):((pusht|aloha)-s(0[1-9]|1[0-6])):"
        r"(regressed|failure_mode|success_bucket):(ev|noev)", row["id"])
    if match is None or int(match[1][1:]) == 0 or row["id"] in identities:
        raise ValueError(f"invalid or duplicate compiled id: {row['id']}")
    change, scenario, env, _, question, variant = match.groups()
    if (row["pair_id"] != f"{change}:{scenario}" or row["change_id"] != change
            or row["scenario_id"] != scenario or row["env"] != env
            or row["question"] != question or row["variant"] != variant
            or row["split"] not in _SPLITS):
        raise ValueError(f"compiled identity metadata disagrees: {row['id']}")
    canonical, options, split = _OPTIONS[question], row["options"], row["split"]
    if (not isinstance(options, list) or any(not isinstance(v, str) for v in options)
            or len(options) != len(canonical) or set(options) != set(canonical)
            or row["label"] not in tuple(_LETTERS[:len(canonical)])
            or (split != "train" and options != list(canonical))):
        raise ValueError(f"invalid compiled options or label: {row['id']}")
    bucket = int.from_bytes(hashlib.sha256(change.encode()).digest()[:4], "big") % 10
    expected = "train" if bucket < 6 else "calibration" if bucket < 8 else "test"
    family_lines = re.findall(r"(?m)^CHANGE: .*$", row["prompt"])
    families = set()
    for line in family_lines:
        content = line.split(" desc=", 1)[0]
        first = re.match(rf"CHANGE: id={re.escape(change)} family=([a-z_]+)(?: |$)", content)
        if first is None or len(family_lines) != 1:
            raise ValueError("compiled CHANGE line has invalid identity or families")
        families.update([first[1], *re.findall(r" \+ (?:family=)?([a-z_]+)(?: |$)", content)])
    if (families - _FAMILIES or (split != "ood" and split != expected)
            or (families and (split == "ood") != bool(families & {"occlusion", "chunk_len"}))
            or (change in splits and splits[change] != split)):
        raise ValueError(f"invalid compiled change split or families: {row['id']}")
    ending = (f"\n<question> {_QUESTIONS[question]}\n<options> "
              + "  ".join(f"{letter}) {value}" for letter, value in zip(_LETTERS, options))
              + "\n<answer>")
    if (not row["prompt"].startswith("<system> ") or not row["prompt"].endswith(ending)
            or "\n<state>\n" not in row["prompt"] or "\n</state>" not in row["prompt"]):
        raise ValueError(f"compiled prompt disagrees with question/options or answer boundary: {row['id']}")
    identities.add(row["id"])
    splits[change] = split


def load_examples(path: Path | str, splits: Iterable[str]) -> list[dict[str, Any]]:
    """Validate every compiled row, retaining the requested change-disjoint splits."""
    if isinstance(splits, str):
        raise TypeError("splits must be a collection of split names")
    selected = set(splits)
    if not selected or not selected <= _SPLITS:
        raise ValueError("unknown or empty requested splits")
    rows, identities, change_splits = [], set(), {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            _validate(row, identities, change_splits)
        except (ValueError, TypeError) as error:
            raise ValueError(f"invalid compiled line {number}: {error}") from error
        if row["split"] in selected:
            rows.append(row)
    if not identities or not rows:
        raise ValueError("compiled dataset has no examples in the requested splits")
    return rows


def letter_token_ids(tokenizer: Any) -> list[int]:
    """Require distinct single-token encodings for the six space-prefixed letters."""
    encoded = [tokenizer.encode(" " + letter, add_special_tokens=False) for letter in _LETTERS]
    if (any(len(ids) != 1 or type(ids[0]) is not int or ids[0] < 0 for ids in encoded)
            or len({tuple(ids) for ids in encoded}) != 6):
        raise ValueError("space-prefixed option letters must be six distinct single tokens")
    return [ids[0] for ids in encoded]


def tokenize(examples: Sequence[dict[str, Any]], tokenizer: Any,
             max_len: int = 1536) -> list[dict[str, Any]]:
    """Tokenize raw prompts and prove every valid letter's exact next-token boundary."""
    if type(max_len) is not int or not 1 <= max_len <= 1536 or not examples:
        raise ValueError("max_len must be in 1..1536 and examples must be nonempty")
    pad, source = tokenizer.pad_token_id, "pad"
    if pad is None:
        pad, source = tokenizer.eos_token_id, "eos"
    if type(pad) is not int or pad < 0:
        raise ValueError("tokenizer requires a pad token or an EOS token for masked padding")
    letters, result, identities, splits = letter_token_ids(tokenizer), [], set(), {}
    for row in examples:
        _validate(row, identities, splits)
        ids = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
        if (not ids or any(type(i) is not int or i < 0 for i in ids)
                or len(ids) > max_len):
            raise ValueError(f"prompt exceeds max_len={max_len} or has invalid ids: {row['id']}")
        k = len(row["options"])
        for letter, token in zip(_LETTERS[:k], letters[:k], strict=True):
            actual = tokenizer(row["prompt"] + " " + letter, add_special_tokens=True)["input_ids"]
            if actual != ids + [token]:
                raise ValueError(f"answer-token boundary fails for {row['id']} letter {letter}; fix brier compile")
        ordinal = ([_OPTIONS["success_bucket"].index(option) for option in row["options"]]
                   if row["question"] == "success_bucket" else [])
        result.append({**row, "input_ids": ids, "label_idx": _LETTERS.index(row["label"]),
                       "k": k, "pos_to_ordinal": ordinal, "pad_id": pad, "pad_source": source})
    return result


def collate(batch: Sequence[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    """Left-pad encoded examples, keeping the final position as the answer position."""
    if not batch or type(pad_id) is not int or pad_id < 0:
        raise ValueError("collate requires a nonempty batch and a nonnegative pad id")
    for row in batch:
        ids, k, label, ordinal = (row[key] for key in ("input_ids", "k", "label_idx", "pos_to_ordinal"))
        if (not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids)
                or type(k) is not int or k not in {2, 5, 6}
                or type(label) is not int or not 0 <= label < k
                or not isinstance(ordinal, list) or any(type(i) is not int for i in ordinal)
                or (ordinal != [] and sorted(ordinal) != list(range(k)))
                or (k == 5) != bool(ordinal)):
            raise ValueError("invalid encoded ids, labels, K or ordinal map")
    width = max(len(row["input_ids"]) for row in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask, ordinal = torch.zeros_like(ids), torch.full((len(batch), 6), -1, dtype=torch.long)
    for i, row in enumerate(batch):
        ids[i, -len(row["input_ids"]):] = torch.tensor(row["input_ids"])
        mask[i, -len(row["input_ids"]):] = 1
        ordinal[i, :len(row["pos_to_ordinal"])] = torch.tensor(row["pos_to_ordinal"], dtype=torch.long)
    k = torch.tensor([row["k"] for row in batch], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
            "label_idx": torch.tensor([row["label_idx"] for row in batch], dtype=torch.long),
            "k": k, "ordinal_map": ordinal, "valid": torch.arange(6)[None, :] < k[:, None]}


def length_grouped_order(lengths: Sequence[int], micro_batch: int, seed: int) -> list[int]:
    """Seed, group by length within shuffled chunks, then shuffle micro-batch order."""
    if (type(micro_batch) is not int or micro_batch < 1 or type(seed) is not int or seed < 0
            or any(type(length) is not int or length < 1 for length in lengths)):
        raise ValueError("lengths, micro_batch and seed must be valid integers")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(lengths)).tolist()
    batches, chunk_size = [], micro_batch * 64
    for start in range(0, len(indices), chunk_size):
        chunk = sorted(indices[start:start + chunk_size], key=lambda index: lengths[index])
        batches.extend(chunk[offset:offset + micro_batch] for offset in range(0, len(chunk), micro_batch))
    return [index for b in rng.permutation(len(batches)) for index in batches[b]]
