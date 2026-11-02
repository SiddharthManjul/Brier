import hashlib
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from brier_ml import glimmer_data as data
from brier_ml import glimmer_model as gm

_SPLITS = ("train", "calibration", "test", "ood")
_LOGGER = logging.getLogger(__name__)


def _forward(model, batch: dict, letters: list[int], norm) -> tuple:
    hidden = []
    hook = (norm.register_forward_hook(lambda _m, _a, output: hidden.append(
        output[:, -1, :].detach().to(device="cpu", dtype=torch.float16))) if norm else None)
    try:
        logits = gm.option_logits(gm.last_logits(model, batch), letters).cpu()
        if not torch.isfinite(logits).all() or (norm and len(hidden) != 1):
            raise ValueError("Scoring requires finite logits and one decoder final-norm call")
        return logits, hidden[0].numpy() if norm else None
    finally:
        if hook:
            hook.remove()


def _extension_rows(compiled_source: bytes, ids_source: bytes,
                    splits: tuple = ("calibration", "test", "ood")) -> tuple[list, list]:
    ids = json.loads(ids_source)
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(i, str) or not i.startswith("c") or not i[1:].isdigit()
                   or int(i[1:]) <= 160 for i in ids) or len(set(ids)) != len(ids)):
        raise ValueError("ids-from requires distinct held-out extension IDs after c160")
    rows, tokenization_rows, identities, split_map = [], [], set(), {}
    for line in compiled_source.decode().splitlines():
        row = json.loads(line)
        checked = row.copy()
        change = row.get("change_id", "")
        if change.startswith("c") and change[1:].isdigit() and int(change[1:]) > 160 and row.get("split") == "calibration":
            bucket = int.from_bytes(hashlib.sha256(change.encode()).digest()[:4], "big") % 10
            checked["split"] = "train" if bucket < 6 else "calibration" if bucket < 8 else "test"
        data._validate(checked, identities, split_map)
        if row["change_id"] in ids and row["split"] in splits:
            rows.append(row)
            tokenization_rows.append(checked)
    if {row["id"].split(":")[0] for row in rows} != set(ids) or any(row["split"] == "train" for row in rows):
        raise ValueError("ids-from must cover complete non-training changes")
    for change in ids:
        selected = [row for row in rows if row["id"].split(":")[0] == change]
        if len(selected) != 96 or len({row["scenario_id"] for row in selected}) != 16:
            raise ValueError("ids-from excludes partial changes: all 16 scenarios and six questions/variants required")
    return rows, tokenization_rows


def _run(out_dir: Path, splits: tuple, adapter: Path | None,
         max_usd: float, usd_per_sec: float | None, ids_from: Path | None = None) -> dict:
    from brier_ml.modal_train import BASE_DIR

    if (not splits or len(set(splits)) != len(splits) or set(splits) - set(_SPLITS)
            or not math.isfinite(max_usd) or max_usd <= 0):
        raise ValueError("Scoring requires distinct known splits and a positive finite budget")
    started = time.perf_counter()
    revision = (Path(BASE_DIR) / "revision.txt").read_text().strip()
    source = adapter / "run_meta.json" if adapter else Path(BASE_DIR).parent.parent / "runs/check.json"
    meta = json.loads(source.read_text())
    quant = meta["quant"]
    if quant not in {"bf16", "nf4"} or meta["revision"] != revision:
        raise ValueError("Scoring quantization or base revision disagrees with the checked model")
    rate = usd_per_sec if usd_per_sec is not None else {"bf16": 0.001097, "nf4": 0.000694}[quant]
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Scoring GPU rate must be finite and positive")
    compiled = (ids_from.parent if ids_from else Path(os.environ["BRIER_DATA"])) / "compiled.jsonl"
    if ids_from is not None:
        rows, tokenization_rows = _extension_rows(compiled.read_bytes(), ids_from.read_bytes(), splits)
    else:
        rows = data.load_examples(compiled, splits)
        tokenization_rows = rows
    if ids_from is None and set(splits) - {row["split"] for row in rows}:
        raise ValueError("Every requested scoring split must have examples")
    result, embeddings, cursor = [], [], 0
    out_dir.mkdir(parents=True, exist_ok=True)

    def guard() -> None:
        elapsed = time.perf_counter() - started
        if elapsed * rate >= max_usd:
            torch.save({"cursor": cursor, "rows": result, "embeddings": embeddings,
                        "revision": revision, "quant": quant, "seconds": elapsed,
                        "spend": elapsed * rate}, out_dir / "score_checkpoint.pt")
            raise RuntimeError(f"Scoring budget cap ${max_usd:g} reached; checkpoint saved at row {cursor}")

    guard()
    model, tokenizer, info = gm.load_base(BASE_DIR, revision, quant)
    guard()
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter / "adapter", is_trainable=False)
    model.eval()
    encoded, letters = data.tokenize(tokenization_rows, tokenizer), data.letter_token_ids(tokenizer)
    for row, original in zip(encoded, rows, strict=True):
        row["split"] = original["split"]
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    norm = base.get_submodule(info["final_norm"]) if adapter is None else None
    device, size = next(model.parameters()).device, 32
    with torch.inference_mode():
        while cursor < len(encoded):
            guard()
            selected = encoded[cursor:cursor + size]
            batch = {}
            try:
                batch = {key: value.to(device) for key, value in data.collate(selected, selected[0]["pad_id"]).items()}
                logits, hidden = _forward(model, batch, letters, norm)
            except torch.OutOfMemoryError:
                if size == 1:
                    raise
                size //= 2
                del batch
                torch.cuda.empty_cache()
                _LOGGER.warning("Scoring OOM: halving micro-batch to %d", size)
                continue
            _LOGGER.info("Scoring micro-batch size=%d rows=%d", size, len(selected))
            for index, row in enumerate(selected):
                probs = logits[index, :row["k"]].softmax(-1).tolist()
                result.append({key: row[key] for key in
                               ("id", "pair_id", "split", "question", "variant", "options", "label")}
                              | {"logits": logits[index].tolist(),
                                 "probs": dict(zip("ABCDEF"[:row["k"]], probs, strict=True)),
                                 "model": f"glimmer-ft-{adapter.name}" if adapter else "glimmer-base-zeroshot",
                                 "revision": revision})
            if hidden is not None:
                embeddings.extend(hidden)
            cursor += len(selected)
            guard()
    with tempfile.TemporaryDirectory(dir=out_dir) as scratch:
        destination = Path(scratch)
        (destination / "probs").mkdir()
        if adapter is None:
            (destination / "emb").mkdir()
        for split in splits:
            indices = [index for index, row in enumerate(result) if row["split"] == split]
            (destination / "probs" / f"{split}.jsonl").write_text(
                "".join(json.dumps(result[i], allow_nan=False) + "\n" for i in indices))
            if adapter is None:
                np.save(destination / "emb" / f"{split}.npy", np.stack([embeddings[i] for i in indices]))
                (destination / "emb" / f"{split}.ids.json").write_text(json.dumps([result[i]["id"] for i in indices]) + "\n")
        guard()
        for directory in destination.iterdir():
            (out_dir / directory.name).mkdir(exist_ok=True)
            for artifact in directory.iterdir():
                artifact.replace(out_dir / directory.name / artifact.name)
    (out_dir / "score_checkpoint.pt").unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    return {"quant": quant, "revision": revision, "seconds": elapsed,
            "spend": elapsed * rate, "rows": cursor, "micro_batch": size,
            **({"compiled_sha256": hashlib.sha256(compiled.read_bytes()).hexdigest(),
                "ids_sha256": hashlib.sha256(ids_from.read_bytes()).hexdigest(),
                "output": str(out_dir)} if ids_from else {})}


def base_pass(out_dir: Path | str, *, max_usd: float = 3.0,
              usd_per_sec: float | None = None) -> dict:
    """Score all compiled splits and save paired float16 final-token embeddings."""
    return _run(Path(out_dir), _SPLITS, None, max_usd, usd_per_sec)


def score_adapter(run_dir: Path | str, splits: tuple = ("calibration", "test", "ood"), *,
                  max_usd: float = 2.0, usd_per_sec: float | None = None,
                  ids_from: Path | str | None = None) -> dict:
    """Score a trained adapter using its recorded base revision and quantization."""
    ids = Path(ids_from) if ids_from is not None else None
    output = Path(run_dir) / ("heldout-" + hashlib.sha256(ids.read_bytes()).hexdigest()[:16]) if ids else Path(run_dir)
    if ids is not None and output.exists():
        raise FileExistsError("held-out scoring output already exists; reconcile its outcome before retrying")
    return _run(output, splits, Path(run_dir), max_usd, usd_per_sec, ids)
