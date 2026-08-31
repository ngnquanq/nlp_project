from __future__ import annotations

from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient

from mt_pipeline.serving.app import create_app
from mt_pipeline.serving.settings import UISettings
from mt_pipeline.serving.translator import (
    EXPECTED_PARAMETER_COUNT,
    MODEL_ID,
    TranslationResult,
    prepare_input,
)


class FakeTranslator:
    model_id = MODEL_ID
    device = "cpu"
    parameter_count = EXPECTED_PARAMETER_COUNT

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._counter_lock = threading.Lock()

    def translate(self, text: str) -> TranslationResult:
        with self._counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            return TranslationResult(
                translation="王遣使如北",
                normalized_input=text,
                source_token_count=len(text.split()),
                target_token_count=6,
                unknown_tokens=["phương"],
                latency_ms=12,
            )
        finally:
            with self._counter_lock:
                self.active -= 1


def settings(*, serve_frontend: bool = False) -> UISettings:
    return UISettings(
        checkpoint_path=Path("/private/checkpoint_best.pt"),
        data_bin_path=Path("/private/data-bin"),
        device="cpu",
        serve_frontend=serve_frontend,
    )


def test_prepare_input_normalizes_nfc_and_whitespace() -> None:
    prepared = prepare_input("  nha\u0300   vua\n")
    assert prepared.text == "nhà vua"
    assert prepared.source_token_count == 2
    assert prepared.encoded_position_count == 3


def test_health_and_translation_contract() -> None:
    translator = FakeTranslator()
    app = create_app(settings(), lambda _settings: translator)

    with TestClient(app) as client:
        health = client.get("/api/health")
        response = client.post("/api/translate", json={"text": "  Vua  sai sứ  "})

    assert health.status_code == 200
    assert health.json() == {
        "status": "ready",
        "model_id": MODEL_ID,
        "device": "cpu",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "message": None,
    }
    assert response.status_code == 200
    assert response.json() == {
        "translation": "王遣使如北",
        "normalized_input": "Vua sai sứ",
        "source_token_count": 3,
        "target_token_count": 6,
        "unknown_tokens": ["phương"],
        "latency_ms": 12,
        "model_id": MODEL_ID,
    }


def test_blank_and_long_inputs_return_stable_error_codes() -> None:
    app = create_app(settings(), lambda _settings: FakeTranslator())
    with TestClient(app) as client:
        blank = client.post("/api/translate", json={"text": "  \n "})
        too_many_tokens = client.post(
            "/api/translate", json={"text": " ".join(["từ"] * 256)}
        )
        too_many_characters = client.post(
            "/api/translate", json={"text": "a" * 4_001}
        )

    assert blank.status_code == 422
    assert blank.json()["error"]["code"] == "EMPTY_INPUT"
    assert too_many_tokens.status_code == 422
    assert too_many_tokens.json()["error"]["code"] == "INPUT_TOO_LONG"
    assert too_many_characters.status_code == 422
    assert too_many_characters.json()["error"]["code"] == "INPUT_TOO_LONG"


def test_missing_artifacts_keep_health_available() -> None:
    missing = UISettings(None, None, serve_frontend=False)
    app = create_app(missing)
    with TestClient(app) as client:
        health = client.get("/api/health")
        translation = client.post("/api/translate", json={"text": "Vua sai sứ"})

    assert health.status_code == 200
    assert health.json()["status"] == "model_unavailable"
    assert translation.status_code == 503
    assert translation.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_translation_requests_are_serialized() -> None:
    translator = FakeTranslator(delay=0.05)
    app = create_app(settings(), lambda _settings: translator)
    barrier = threading.Barrier(3)

    with TestClient(app) as client:
        def request_translation() -> None:
            barrier.wait()
            response = client.post("/api/translate", json={"text": "Vua sai sứ"})
            assert response.status_code == 200

        threads = [threading.Thread(target=request_translation) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

    assert translator.max_active == 1
