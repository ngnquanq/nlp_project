"""Gateway -> sidecar forwarding, driven entirely by httpx.MockTransport."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import pytest

from mt_pipeline.serving.errors import ServiceError
from mt_pipeline.serving.factory import create_app
from mt_pipeline.serving.registry import E2_MODEL_ID, E2_SPEC
from mt_pipeline.serving.remote import RemoteTranslator
from mt_pipeline.serving.settings import UISettings
from mt_pipeline.serving.translator import MODEL_ID, TranslationInputError, TranslationResult

BASE_URL = "http://127.0.0.1:8001"

TRANSLATION_BODY = {
    "translation": "帝遣使如宋",
    "normalized_input": "vua sai sứ sang nước tống",
    "source_token_count": 12,
    "target_token_count": 5,
    "unknown_tokens": [],
    "latency_ms": 4_200,
    "model_id": E2_MODEL_ID,
}

HEALTH_BODY = {
    "status": "ready",
    "default_model": "e2",
    "models": [
        {
            "key": "e2",
            "model_id": E2_MODEL_ID,
            "label": "E2 (sidecar copy)",
            "sublabel": "sidecar copy",
            "status": "not_loaded",
            "device": "cuda",
            "parameter_count": None,
            "message": None,
            "limits": {
                "max_characters": 4_000,
                "max_units": 512,
                "unit": "token",
                "unit_label": "token",
                "client_estimate": True,
                "chars_per_unit": 3.39,
            },
            "reports_unknown_tokens": False,
            "slow_first_request": True,
        }
    ],
}


def error_body(code: str) -> dict:
    return {"error": {"code": code, "message": f"upstream {code}", "request_id": "abc123"}}


def remote(handler, **kwargs) -> RemoteTranslator:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    return RemoteTranslator(BASE_URL, E2_SPEC, client=client, **kwargs)


def test_successful_translation_is_rebuilt() -> None:
    seen = {}

    def capture(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=TRANSLATION_BODY)

    result = remote(capture).translate("vua sai sứ sang nước tống")

    assert isinstance(result, TranslationResult)
    assert result.translation == "帝遣使如宋"
    assert result.latency_ms == 4_200
    assert result.unknown_tokens == []
    assert seen["url"] == f"{BASE_URL}/api/translate"
    assert seen["payload"] == {"text": "vua sai sứ sang nước tống", "model": "e2"}


def test_upstream_422_keeps_its_code() -> None:
    def handler(_request):
        return httpx.Response(422, json=error_body("INPUT_TOO_LONG"))

    with pytest.raises(TranslationInputError) as excinfo:
        remote(handler).translate("vua")
    assert excinfo.value.code == "INPUT_TOO_LONG"
    assert "upstream" in excinfo.value.message


def test_upstream_503_carries_the_real_reason() -> None:
    def handler(_request):
        return httpx.Response(503, json=error_body("MODEL_UNAVAILABLE"))

    with pytest.raises(ServiceError) as excinfo:
        remote(handler).translate("vua")
    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "MODEL_UNAVAILABLE"


def test_upstream_500_becomes_a_bad_gateway() -> None:
    def handler(_request):
        return httpx.Response(500, json=error_body("TRANSLATION_FAILED"))

    with pytest.raises(ServiceError) as excinfo:
        remote(handler).translate("vua")
    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "E2_UPSTREAM_ERROR"


def test_connect_error_tells_the_user_to_start_the_sidecar() -> None:
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ServiceError) as excinfo:
        remote(handler).translate("vua")
    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "E2_SIDECAR_UNREACHABLE"
    assert "ui-api-e2" in excinfo.value.message


def test_read_timeout_explains_the_cold_start() -> None:
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(ServiceError) as excinfo:
        remote(handler).translate("vua")
    assert excinfo.value.status_code == 504
    assert excinfo.value.code == "E2_TIMEOUT"


def test_health_probe_is_ttl_cached() -> None:
    calls = []
    now = [1_000.0]

    def handler(_request):
        calls.append(1)
        return httpx.Response(200, json=HEALTH_BODY)

    translator = remote(handler, health_ttl_seconds=5.0, clock=lambda: now[0])

    translator.describe()
    translator.describe()
    assert len(calls) == 1

    now[0] += 6.0
    translator.describe()
    assert len(calls) == 2


def test_health_reports_sidecar_status_under_gateway_identity() -> None:
    def handler(_request):
        return httpx.Response(200, json=HEALTH_BODY)

    health = remote(handler).describe()
    # Status and limits come from the sidecar; the label stays the gateway's.
    assert health.status == "not_loaded"
    assert health.limits.max_units == 512
    assert health.label == "E2"
    assert health.sublabel == "Qwen3-8B QLoRA 4-bit"


def test_unreachable_sidecar_still_reports_usable_limits() -> None:
    """The selector renders in this state, so max_units must be a real number."""

    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    health = remote(handler).describe()
    assert health.status == "unavailable"
    assert health.limits.max_units == 512
    assert "ui-api-e2" in (health.message or "")


class FakeE1:
    model_id = MODEL_ID
    device = "cpu"
    parameter_count = 67_652_608

    def translate(self, text: str) -> TranslationResult:
        return TranslationResult("王遣使如北", text, 3, 5, [], 12)


def test_gateway_forwards_e2_end_to_end() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/health":
            return httpx.Response(200, json=HEALTH_BODY)
        return httpx.Response(200, json=TRANSLATION_BODY)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    settings = UISettings(
        checkpoint_path=Path("/private/checkpoint_best.pt"),
        data_bin_path=Path("/private/data-bin"),
        device="cpu",
        serve_frontend=False,
        e2_base_url=BASE_URL,
    )
    app = create_app(
        settings,
        lambda _s: FakeE1(),
        e2_translator_factory=lambda _s: RemoteTranslator(BASE_URL, E2_SPEC, client=client),
    )

    with TestClient(app) as test_client:
        health = test_client.get("/api/health").json()
        e1 = test_client.post("/api/translate", json={"text": "vua sai sứ"})
        e2 = test_client.post("/api/translate", json={"text": "vua sai sứ", "model": "e2"})

    assert [model["key"] for model in health["models"]] == ["e1", "e2"]
    assert health["models"][1]["status"] == "not_loaded"
    assert e1.json()["model_id"] == MODEL_ID
    assert e2.status_code == 200
    assert e2.json()["translation"] == "帝遣使如宋"
    assert e2.json()["model_id"] == E2_MODEL_ID
