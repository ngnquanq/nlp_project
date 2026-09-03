from __future__ import annotations

from pathlib import Path
import threading

from fastapi.testclient import TestClient

from mt_pipeline.serving.factory import create_app
from mt_pipeline.serving.registry import E2_MODEL_ID, e2_limits
from mt_pipeline.serving.settings import UISettings
from mt_pipeline.serving.translator import MODEL_ID, TranslationResult


class FakeE1:
    model_id = MODEL_ID
    device = "cpu"
    parameter_count = 67_652_608

    def __init__(self, barrier: threading.Barrier | None = None) -> None:
        self.barrier = barrier
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def translate(self, text: str) -> TranslationResult:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait()
            return TranslationResult("王遣使如北", text, len(text.split()), 6, ["phương"], 12)
        finally:
            with self._lock:
                self.active -= 1


class FakeE2:
    model_id = E2_MODEL_ID
    device = "cuda"
    parameter_count = None

    def __init__(
        self,
        *,
        status: str = "ready",
        max_prompt_tokens: int = 512,
        barrier: threading.Barrier | None = None,
    ) -> None:
        self.status = status
        self.limits = e2_limits(max_prompt_tokens)
        self.count_units = lambda text: len(text.split())
        self.barrier = barrier
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def translate(self, text: str) -> TranslationResult:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait()
            return TranslationResult("帝遣使如宋", text, len(text.split()), 5, [], 4_200)
        finally:
            with self._lock:
                self.active -= 1


def settings() -> UISettings:
    return UISettings(
        checkpoint_path=Path("/private/checkpoint_best.pt"),
        data_bin_path=Path("/private/data-bin"),
        device="cpu",
        serve_frontend=False,
    )


def two_model_app(e1=None, e2=None):
    e1 = e1 or FakeE1()
    e2 = e2 or FakeE2()
    app = create_app(
        settings(),
        lambda _s: e1,
        e2_translator_factory=lambda _s: e2,
    )
    return app, e1, e2


def test_health_lists_both_models() -> None:
    app, _, _ = two_model_app()
    with TestClient(app) as client:
        body = client.get("/api/health").json()

    assert body["status"] == "ready"
    assert body["default_model"] == "e1"
    assert [model["key"] for model in body["models"]] == ["e1", "e2"]

    e2 = body["models"][1]
    assert e2["model_id"] == E2_MODEL_ID
    assert e2["label"] == "E2"
    assert e2["device"] == "cuda"
    assert e2["parameter_count"] is None
    assert e2["reports_unknown_tokens"] is False
    assert e2["slow_first_request"] is True
    assert e2["limits"]["unit"] == "token"
    assert e2["limits"]["max_units"] == 512
    assert e2["limits"]["client_estimate"] is True
    # max_units is never null: the browser renders it directly.
    assert isinstance(e2["limits"]["max_units"], int)


def test_translate_dispatches_to_the_requested_model() -> None:
    app, _, _ = two_model_app()
    with TestClient(app) as client:
        default = client.post("/api/translate", json={"text": "vua sai sứ"})
        explicit = client.post("/api/translate", json={"text": "vua sai sứ", "model": "e2"})
        by_model_id = client.post(
            "/api/translate", json={"text": "vua sai sứ", "model": E2_MODEL_ID}
        )

    assert default.json()["model_id"] == MODEL_ID
    assert explicit.json()["model_id"] == E2_MODEL_ID
    assert explicit.json()["translation"] == "帝遣使如宋"
    assert by_model_id.json()["model_id"] == E2_MODEL_ID


def test_unknown_model_is_rejected() -> None:
    app, _, _ = two_model_app()
    with TestClient(app) as client:
        response = client.post("/api/translate", json={"text": "vua sai sứ", "model": "e3"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNKNOWN_MODEL"


def test_model_absent_from_this_app_is_rejected() -> None:
    """E1-only gateway must not pretend to serve E2."""
    app = create_app(settings(), lambda _s: FakeE1())
    with TestClient(app) as client:
        response = client.post("/api/translate", json={"text": "vua sai sứ", "model": "e2"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNKNOWN_MODEL"


def test_failed_e2_factory_degrades_without_taking_e1_down() -> None:
    def broken(_settings):
        raise RuntimeError("no gpu")

    app = create_app(settings(), lambda _s: FakeE1(), e2_translator_factory=broken)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        e1 = client.post("/api/translate", json={"text": "vua sai sứ"})
        e2 = client.post("/api/translate", json={"text": "vua sai sứ", "model": "e2"})

    assert health["status"] == "degraded"
    assert health["models"][0]["status"] == "ready"
    assert health["models"][1]["status"] == "unavailable"
    assert health["models"][1]["message"]
    assert e1.status_code == 200
    assert e2.status_code == 503
    assert e2.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_not_loaded_model_is_still_submittable() -> None:
    """Lazy loading is unreachable if the UI refuses to submit to not_loaded."""
    app, _, _ = two_model_app(e2=FakeE2(status="not_loaded"))
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        response = client.post("/api/translate", json={"text": "vua sai sứ", "model": "e2"})

    assert health["models"][1]["status"] == "not_loaded"
    assert health["status"] == "ready"
    assert response.status_code == 200


def test_over_limit_message_is_model_specific() -> None:
    app, _, _ = two_model_app(e2=FakeE2(max_prompt_tokens=8))
    long_text = " ".join(["từ"] * 20)
    with TestClient(app) as client:
        e1 = client.post("/api/translate", json={"text": " ".join(["từ"] * 300)})
        e2 = client.post("/api/translate", json={"text": long_text, "model": "e2"})

    assert e1.status_code == 422
    assert "E1" in e1.json()["error"]["message"]
    assert e2.status_code == 422
    assert e2.json()["error"]["code"] == "INPUT_TOO_LONG"
    assert "E2" in e2.json()["error"]["message"]
    assert "8 token" in e2.json()["error"]["message"]


def test_models_do_not_queue_behind_each_other() -> None:
    """A slow E2 request must not block E1.

    Both fakes wait on one barrier inside translate(); a shared lock would
    stall the second request until the barrier times out and breaks.
    """
    barrier = threading.Barrier(2, timeout=10)
    app, e1, e2 = two_model_app(e1=FakeE1(barrier=barrier), e2=FakeE2(barrier=barrier))
    responses: list[int] = []
    lock = threading.Lock()

    with TestClient(app) as client:
        def call(model: str) -> None:
            response = client.post("/api/translate", json={"text": "vua sai sứ", "model": model})
            with lock:
                responses.append(response.status_code)

        threads = [
            threading.Thread(target=call, args=("e1",)),
            threading.Thread(target=call, args=("e2",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert responses == [200, 200]
    assert e1.max_active == 1
    assert e2.max_active == 1
