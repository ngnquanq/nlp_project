from __future__ import annotations

import os
from pathlib import Path

import pytest

from mt_pipeline.serving.translator import E1Translator


@pytest.mark.private_model
def test_private_e1_model_is_deterministic() -> None:
    checkpoint = os.environ.get("MT_CHECKPOINT_PATH")
    data_bin = os.environ.get("MT_DATA_BIN_PATH")
    if not checkpoint or not data_bin:
        pytest.skip("MT_CHECKPOINT_PATH and MT_DATA_BIN_PATH are required")

    translator = E1Translator(Path(checkpoint), Path(data_bin), os.getenv("MT_DEVICE", "cpu"))
    source = "Vua sai sứ sang phương Bắc"
    first = translator.translate(source)
    second = translator.translate(source)
    assert first.translation
    assert first.translation == second.translation
