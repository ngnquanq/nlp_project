from __future__ import annotations

import threading

import pytest

from mt_pipeline.serving.lazy import LazyTranslator
from mt_pipeline.serving.registry import E2_MODEL_ID, e2_limits
from mt_pipeline.serving.translator import ModelArtifactError, TranslationResult


class Inner:
    device = "cuda"
    parameter_count = 123

    def translate(self, text: str) -> TranslationResult:
        return TranslationResult("帝", text, 1, 1, [], 7)


def test_reports_not_loaded_before_first_use() -> None:
    lazy = LazyTranslator(
        model_id=E2_MODEL_ID, device="cuda", build=Inner, limits=e2_limits()
    )
    assert lazy.status == "not_loaded"
    assert lazy.parameter_count is None
    assert lazy.message is None


def test_concurrent_first_requests_build_once() -> None:
    builds = []
    barrier = threading.Barrier(8, timeout=10)

    def build():
        builds.append(1)
        return Inner()

    lazy = LazyTranslator(model_id=E2_MODEL_ID, device="cuda", build=build)

    def call() -> None:
        barrier.wait()
        lazy.translate("vua sai sứ")

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(builds) == 1
    assert lazy.status == "ready"
    assert lazy.parameter_count == 123


def test_load_failure_is_sticky_and_not_retried() -> None:
    attempts = []

    def build():
        attempts.append(1)
        raise ModelArtifactError("CUDA không khả dụng")

    lazy = LazyTranslator(model_id=E2_MODEL_ID, device="cuda", build=build)

    with pytest.raises(ModelArtifactError):
        lazy.translate("vua sai sứ")
    with pytest.raises(ModelArtifactError):
        lazy.translate("vua sai sứ")

    assert len(attempts) == 1
    assert lazy.status == "unavailable"
    assert "CUDA không khả dụng" in (lazy.message or "")
