"""Sidecar shape checks. Runs under .conda/e2-ui via `make ui-check-e2`."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mt_pipeline.serving.e2_app import create_e2_app
from mt_pipeline.serving.registry import E2_MODEL_ID
from mt_pipeline.serving.settings import UISettings
from mt_pipeline.serving.translator import TranslationResult


class FakeE2:
    model_id = E2_MODEL_ID
    device = "cuda"
    parameter_count = None
    status = "not_loaded"

    def translate(self, text: str) -> TranslationResult:
        return TranslationResult("帝遣使如宋", text, 3, 5, [], 4_200)


def sidecar_app():
    settings = UISettings(None, None, serve_frontend=False, port=8001)
    return create_e2_app(settings, translator_factory=lambda _s: FakeE2())


def test_sidecar_serves_only_e2() -> None:
    with TestClient(sidecar_app()) as client:
        body = client.get("/api/health").json()

    assert [model["key"] for model in body["models"]] == ["e2"]
    assert body["default_model"] == "e2"
    assert body["models"][0]["status"] == "not_loaded"


def test_sidecar_rejects_e1() -> None:
    with TestClient(sidecar_app()) as client:
        response = client.post("/api/translate", json={"text": "vua sai sứ", "model": "e1"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNKNOWN_MODEL"


def test_sidecar_translates_e2_and_mounts_no_frontend() -> None:
    with TestClient(sidecar_app()) as client:
        response = client.post("/api/translate", json={"text": "vua sai sứ", "model": "e2"})
        root = client.get("/")

    assert response.status_code == 200
    assert response.json()["model_id"] == E2_MODEL_ID
    assert root.status_code == 404
