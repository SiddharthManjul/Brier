from __future__ import annotations

import io
import json
from pathlib import Path
from urllib import error, request

import pytest
from brier_explain import spark
from PIL import Image


def fixture(tmp_path: Path, monkeypatch) -> tuple[list[Path], list[dict]]:
    monkeypatch.setenv("META_API_KEY", "unit-test-secret")
    paths = []
    for index in range(16):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (3, 4), (index, 0, 0)).save(path)
        paths.append(path)
    return paths, []


def response(monkeypatch, content: dict, calls: list[dict], **overrides) -> None:
    body = {"model": "muse-spark-1.3-contributor", "choices": [{
        "finish_reason": "stop", "message": {"content": json.dumps(content)}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4}, **overrides}

    def open_request(self, value, timeout):
        calls.append({"url": value.full_url, "headers": dict(value.header_items()),
                      "payload": json.loads(value.data), "timeout": timeout,
                      "redirect_handler": any(isinstance(item, request.HTTPRedirectHandler) for item in self.handlers)})
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(request.OpenerDirector, "open", open_request)


def test_explain_sends_exact_prompt_pngs_schema_and_requested_model(tmp_path, monkeypatch) -> None:
    paths, calls = fixture(tmp_path, monkeypatch)
    explanation = {"failure_mode": "stall/timeout", "cause_hypothesis": "Object remains short of target",
                   "evidence_frames": [2, 7], "suggested_check": "Inspect final object motion"}
    response(monkeypatch, explanation, calls)
    prompt = "Rust prompt\n  exact spacing Ω\n<answer>"
    result = spark.explain(prompt, paths)
    assert result == {"model": "muse-spark-1.3-contributor", "explanation": explanation, "cost_usd": None}
    assert len(calls) == 1
    sent = calls[0]
    assert sent["url"] == "https://api.meta.ai/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer unit-test-secret"
    assert sent["timeout"] == 60 and not sent["redirect_handler"]
    payload = sent["payload"]
    assert payload["model"] == result["model"]
    assert payload["messages"][0]["content"][0] == {"type": "text", "text": prompt}
    images = payload["messages"][0]["content"][1:]
    assert len(images) == 16
    assert all(item["image_url"]["url"].startswith("data:image/png;base64,") for item in images)
    schema = payload["response_format"]["json_schema"]
    assert schema["strict"] and schema["schema"]["additionalProperties"] is False
    assert set(schema["schema"]["required"]) == set(explanation)


@pytest.mark.parametrize("recommend", ["run_expensive", "run_more_cheap", "no_change"])
def test_escalation_returns_only_advice_and_unknown_cost(tmp_path, monkeypatch, recommend) -> None:
    paths, calls = fixture(tmp_path, monkeypatch)
    response(monkeypatch, {"recommend": recommend, "rationale": "Observed evidence is uncertain"}, calls)
    result = spark.escalate("Rust escalation state", paths[:3])
    assert result["recommend"] == recommend and result["cost_usd"] is None
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["wrong_model", "truncated", "missing", "extra", "bad_index", "duplicate_index",
                                     "boolean_index", "bad_failure", "empty_cause", "not_json"])
def test_explanation_rejects_malformed_response_without_retry(tmp_path, monkeypatch, mutation) -> None:
    paths, calls = fixture(tmp_path, monkeypatch)
    value = {"failure_mode": "collision", "cause_hypothesis": "Observed contact", "evidence_frames": [0],
             "suggested_check": "Check contact telemetry"}
    overrides = {}
    if mutation == "wrong_model":
        overrides["model"] = "muse-spark-1.3"
    elif mutation in {"truncated", "not_json"}:
        overrides["choices"] = [{"finish_reason": "length" if mutation == "truncated" else "stop",
                                 "message": {"content": "not JSON"}}]
    elif mutation == "missing":
        del value["suggested_check"]
    elif mutation == "extra":
        value["cause_family"] = "hidden"
    elif mutation in {"bad_index", "duplicate_index", "boolean_index"}:
        value["evidence_frames"] = {"bad_index": [8], "duplicate_index": [0, 0], "boolean_index": [True]}[mutation]
    elif mutation == "bad_failure":
        value["failure_mode"] = "arbitrary"
    else:
        value["cause_hypothesis"] = " "
    response(monkeypatch, value, calls, **overrides)
    with pytest.raises(ValueError):
        spark.explain("exact Rust text", paths)
    assert len(calls) == 1


@pytest.mark.parametrize("status", [302, 429, 500])
def test_http_errors_and_redirects_are_not_retried_or_leaked(tmp_path, monkeypatch, status) -> None:
    paths, calls = fixture(tmp_path, monkeypatch)

    def fail(self, value, timeout):
        calls.append(value.full_url)
        raise error.HTTPError(value.full_url, status, "secret provider body", {}, io.BytesIO(b"unit-test-secret"))

    monkeypatch.setattr(request.OpenerDirector, "open", fail)
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as caught:
        spark.explain("Rust prompt", paths)
    assert "secret" not in str(caught.value) and len(calls) == 1


def test_missing_key_invalid_png_and_invalid_advice_fail(tmp_path, monkeypatch) -> None:
    paths, calls = fixture(tmp_path, monkeypatch)
    response(monkeypatch, {"recommend": "skip_everything", "rationale": "unsafe"}, calls)
    monkeypatch.delenv("META_API_KEY")
    with pytest.raises(ValueError, match="META_API_KEY"):
        spark.escalate("Rust prompt", [])
    assert not calls
    monkeypatch.setenv("META_API_KEY", "unit-test-secret")
    bad = tmp_path / "not-png.png"
    Image.new("RGB", (2, 2)).save(bad, format="JPEG")
    with pytest.raises(ValueError, match="actual PNG"):
        spark.escalate("Rust prompt", [bad])
    assert not calls
    with pytest.raises(ValueError, match="recommendation"):
        spark.escalate("Rust prompt", paths[:1])
    assert len(calls) == 1


@pytest.mark.parametrize("content", ['{"recommend":"no_change","recommend":"run_expensive","rationale":"x"}',
                                  '{"recommend":"no_change","rationale":NaN}'])
def test_json_duplicate_fields_and_nonfinite_values_are_rejected(tmp_path, monkeypatch, content) -> None:
    _, calls = fixture(tmp_path, monkeypatch)
    response(monkeypatch, {}, calls, choices=[{"finish_reason": "stop", "message": {"content": content}}])
    with pytest.raises(ValueError, match="duplicate|nonfinite"):
        spark.escalate("Rust prompt", [])
    assert len(calls) == 1


@pytest.mark.parametrize("suffix", ["\r\ninjected", "\x01", "é"])
def test_header_whitespace_cannot_expose_api_key(tmp_path, monkeypatch, suffix) -> None:
    _, calls = fixture(tmp_path, monkeypatch)
    response(monkeypatch, {}, calls)
    monkeypatch.setenv("META_API_KEY", "unit-test-secret" + suffix)
    with pytest.raises(ValueError, match="whitespace") as caught:
        spark.escalate("Rust prompt", [])
    assert "unit-test-secret" not in str(caught.value) and not calls
