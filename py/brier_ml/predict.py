from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from brier_ml.dataset import _compiled_rows


def score_compiled(compiled: Path, output: Path) -> dict[str, object]:
    """Score complete Rust-compiled states and bind option probabilities to their source."""
    root = os.environ.get("BRIER_DATA", "")
    endpoint = os.environ.get("MODAL_DECIDE_URL", "")
    if not root or "://" in root or not Path(root).is_dir():
        raise ValueError("BRIER_DATA must name an existing local directory")
    compiled, output = compiled.resolve(), output.resolve()
    if not all(path.is_relative_to(Path(root).resolve()) for path in (compiled, output)):
        raise ValueError("compiled inputs and predictions must reside inside BRIER_DATA")
    if output.exists():
        raise FileExistsError(f"prediction output already exists: {output}")
    address = urlsplit(endpoint)
    if (
        address.scheme not in {"http", "https"} or not address.netloc
        or address.username or address.password or address.query or address.fragment
        or address.path != "/decide"
    ):
        raise ValueError("MODAL_DECIDE_URL must be the full HTTP(S) /decide endpoint")
    rows = _compiled_rows(compiled)
    source = compiled.read_bytes()
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        if (not row["prompt"].startswith("<system> ") or not row["prompt"].endswith("<answer>")
                or any(marker not in row["prompt"] for marker in ("\n<state>\n", "\n<question> ", "\n<options> "))):
            raise ValueError("invalid compiled prompt")
        groups.setdefault((row["change"], row["scenario"], row["evidence"]), []).append(
            {key: row[key] for key in ("id", "prompt", "options")}
        )
    if any(len(rows) != 3 for rows in groups.values()):
        raise ValueError("each pair/evidence group must contain all three compiled questions")
    predictions: dict[str, dict[str, float]] = {}
    latencies: dict[str, float] = {}
    version = None
    for questions in groups.values():
        request = Request(endpoint, data=json.dumps({"questions": questions}).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=600) as response:
                result = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"decision endpoint returned HTTP {error.code}") from error
        except URLError as error:
            raise RuntimeError("decision endpoint request failed") from error
        elapsed = time.perf_counter() - started
        if not isinstance(result, dict) or set(result) != {"model_version", "questions"}:
            raise ValueError("decision endpoint returned an invalid response")
        if not isinstance(result["model_version"], str) or not result["model_version"].strip():
            raise ValueError("decision endpoint did not identify its model")
        if version is not None and version != result["model_version"]:
            raise ValueError("decision model version changed during batch scoring")
        version = result["model_version"]
        scores = result["questions"]
        if not isinstance(scores, dict) or set(scores) != {row["id"] for row in questions}:
            raise ValueError("decision endpoint returned mismatched question IDs")
        for row in questions:
            values = scores[row["id"]]
            if (
                not isinstance(values, dict) or set(values) != set(row["options"])
                or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                       for p in values.values())
                or not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6)
            ):
                raise ValueError(f"invalid option probabilities for {row['id']}")
            predictions[row["id"]] = values
            latencies[row["id"]] = elapsed
    artifact = {"model_version": version, "source_sha256": hashlib.sha256(source).hexdigest(),
                "questions": predictions, "latencies_seconds": latencies}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(artifact, allow_nan=False, sort_keys=True) + "\n")
    return artifact
