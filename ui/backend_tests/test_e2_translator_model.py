"""Loads the real E2 model. Requires a CUDA GPU and the private adapter.

Run with `make ui-e2-model` under .conda/e2-ui.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mt_pipeline.serving.e2_translator import E2Preload, E2Translator
from mt_pipeline.serving.settings import UISettings


PREDICTIONS = Path("predictions/e2_qwen3_8b_qlora_vi_zh_v1.val.jsonl")


@pytest.fixture(scope="module")
def translator() -> E2Translator:
    settings = UISettings(None, None, serve_frontend=False, e2_device="cuda")
    error = settings.e2_artifact_error()
    if error:
        pytest.skip(error)
    return E2Translator(E2Preload(settings))


@pytest.mark.private_e2_model
def test_repeated_decoding_is_deterministic(translator: E2Translator) -> None:
    source = "vua sai sứ sang phương bắc"
    first = translator.translate(source)
    second = translator.translate(source)

    assert first.translation
    assert "<think>" not in first.translation
    assert first.translation == second.translation
    assert first.unknown_tokens == []


@pytest.mark.private_e2_model
def test_ui_path_reproduces_the_offline_predictions(translator: E2Translator) -> None:
    """The serving path must not drift from `make e2-val`.

    Same prompt, same decoding parameters, same post-processing — so a row from
    the frozen validation predictions must come back character for character.
    """
    if not PREDICTIONS.is_file():
        pytest.skip(f"Missing {PREDICTIONS}")

    rows = [json.loads(line) for line in PREDICTIONS.read_text(encoding="utf-8").splitlines()]
    sample = [
        row for row in rows if 8 <= len(row["source"].split()) <= 16 and "  " not in row["source"]
    ][:3]
    assert sample, "no suitable validation rows found"

    for row in sample:
        result = translator.translate(row["source"])
        assert result.translation == row["prediction"], row["sample_id"]
