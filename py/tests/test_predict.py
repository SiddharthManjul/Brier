import hashlib
import io
import json
from pathlib import Path

import pytest
from brier_ml import predict


def _compiled(path: Path) -> Path:
    rows = []
    for evidence in ("cheap", "no_cheap"):
        for kind, values in {
            "regressed": ["no", "yes"],
            "failure_mode": ["none", "stall/timeout", "overshoot/misalignment",
                             "grasp or drop failure", "collision", "oscillation"],
            "success_bucket": ["1", "2", "3", "4", "5"],
        }.items():
            rows.append({"id": f"c001/pusht-s01/{kind}/{evidence}", "split": "train",
                         "question": kind, "options": dict(zip("ABCDEF", values)),
                         "label": "A", "prompt": "<system> exact\n<state>\nx\n</state>\n"
                         "<question> exact\n<options> exact\n<answer>"})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_batch_scoring_preserves_rust_prompts_and_binds_source(tmp_path, monkeypatch):
    source = _compiled(tmp_path / "compiled.jsonl")
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    monkeypatch.setenv("MODAL_DECIDE_URL", "https://model.example/decide")
    calls = []

    def respond(request, timeout):
        assert timeout == 600
        rows = json.loads(request.data)["questions"]
        calls.append(rows)
        assert all(set(row) == {"id", "options", "prompt"} for row in rows)
        result = {row["id"]: dict.fromkeys(row["options"], 1 / len(row["options"])) for row in rows}
        return io.StringIO(json.dumps({"model_version": "fixed-model", "questions": result}))

    monkeypatch.setattr(predict, "urlopen", respond)
    output = tmp_path / "scores.json"
    artifact = predict.score_compiled(source, output)
    expected = [json.loads(line) for line in source.read_text().splitlines()]
    assert [row["prompt"] for group in calls for row in group] == [row["prompt"] for row in expected]
    assert len(calls) == 2 and len(artifact["questions"]) == 6
    assert artifact["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads(output.read_text()) == artifact
    with pytest.raises(FileExistsError):
        predict.score_compiled(source, output)


def test_legacy_transport_with_guide_aloha_scenario_id_is_valid(tmp_path):
    source = _compiled(tmp_path / "compiled.jsonl")
    source.write_text(source.read_text().replace("pusht-s01", "aloha-s01"))
    rows = predict._compiled_rows(source)
    assert len(rows) == 6 and {row["scenario"] for row in rows} == {"aloha-s01"}


@pytest.mark.parametrize("broken", ["version", "missing", "probability", "field"])
def test_batch_scoring_rejects_model_changes_or_bad_scores(tmp_path, monkeypatch, broken):
    source = _compiled(tmp_path / "compiled.jsonl")
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    monkeypatch.setenv("MODAL_DECIDE_URL", "https://model.example/decide")
    calls = 0

    def respond(request, timeout):
        nonlocal calls
        calls += 1
        rows = json.loads(request.data)["questions"]
        questions = {row["id"]: dict.fromkeys(row["options"], 1 / len(row["options"])) for row in rows}
        result = {"model_version": "fixed-model", "questions": questions}
        if broken == "version" and calls == 2:
            result["model_version"] = "different-model"
        elif broken == "missing":
            questions.pop(rows[0]["id"])
        elif broken == "probability":
            questions[rows[0]["id"]]["A"] = -1
        elif broken == "field":
            result["extra"] = True
        return io.StringIO(json.dumps(result))

    monkeypatch.setattr(predict, "urlopen", respond)
    output = tmp_path / "scores.json"
    with pytest.raises(ValueError):
        predict.score_compiled(source, output)
    assert not output.exists()


@pytest.mark.parametrize("scenario", ["pusht-s01", "aloha-s01", "aloha_transfer_cube-s01"])
def test_batch_scores_guide_array_options_without_rewriting_prompts(tmp_path, monkeypatch, scenario):
    source = _compiled(tmp_path / "compiled.jsonl")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    values = {"regressed": ["yes", "no"], "failure_mode": ["none", "stall or timeout", "overshoot or misalignment",
              "grasp or drop", "collision", "oscillation"], "success_bucket": ["0-19%", "20-39%", "40-59%", "60-79%", "80-100%"]}
    for row in rows:
        question = row["question"]
        variant = "ev" if row["id"].endswith("/cheap") else "noev"
        row.update(id=f"c101:pusht-s01:{question}:{variant}", pair_id="c101:pusht-s01", change_id="c101",
                   scenario_id="pusht-s01", env="pusht", variant=variant, options=values[question])
        row["prompt"] = "<system> exact\n<state>\nx\n</state>\n<question> exact\n<options> " + "  ".join(
            f"{letter}) {text}" for letter, text in zip("ABCDEF", row["options"])) + "\n<answer>"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source.write_text(source.read_text().replace("pusht-s01", scenario).replace('"env": "pusht"', f'"env": "{"pusht" if scenario.startswith("pusht") else "aloha"}"'))
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    monkeypatch.setenv("BRIER_DATA", str(tmp_path))
    monkeypatch.setenv("MODAL_DECIDE_URL", "https://model.example/decide")
    calls = []

    def respond(request, timeout):
        questions = json.loads(request.data)["questions"]
        calls.extend(questions)
        return io.StringIO(json.dumps({"model_version": "fixed", "questions": {
            row["id"]: dict.fromkeys(row["options"], 1 / len(row["options"])) for row in questions}}))

    monkeypatch.setattr(predict, "urlopen", respond)
    artifact = predict.score_compiled(source, tmp_path / "scores.json")
    assert set(artifact["questions"]) == {row["id"] for row in rows}
    assert [row["prompt"] for row in calls] == [row["prompt"] for row in rows]
    assert list(calls[1]["options"].values()) == values["failure_mode"]
