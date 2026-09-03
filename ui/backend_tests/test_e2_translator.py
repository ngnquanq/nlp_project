"""Pure E2 logic — no GPU, no transformers, so this runs in `make ui-check`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mt_pipeline.serving.e2_translator import (
    compute_max_new_tokens,
    finalize_generation,
    read_resolved_revision,
)
from mt_pipeline.serving.translator import ModelArtifactError, TranslationFailure


DECODING = {
    "max_new_tokens_cap": 512,
    "output_length_multiplier": 2,
    "output_length_offset": 32,
}


def test_max_new_tokens_matches_predict_qlora_formula() -> None:
    assert compute_max_new_tokens(DECODING, 0) == 32
    assert compute_max_new_tokens(DECODING, 10) == 52
    assert compute_max_new_tokens(DECODING, 1_000) == 512  # capped


def test_clean_generation_passes_through() -> None:
    assert finalize_generation("  帝遣使如宋  ") == "帝遣使如宋"
    assert finalize_generation("译文：帝遣使如宋") == "帝遣使如宋"


def test_thinking_content_is_rejected() -> None:
    with pytest.raises(TranslationFailure) as excinfo:
        finalize_generation("<think>hmm</think>帝遣使如宋")
    assert excinfo.value.code == "THINKING_EMITTED"


def test_empty_generation_is_rejected() -> None:
    with pytest.raises(TranslationFailure) as excinfo:
        finalize_generation("   ")
    assert excinfo.value.code == "EMPTY_GENERATION"


def test_revision_comes_from_the_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"resolved_model_revision": "abc123"}), encoding="utf-8")
    assert read_resolved_revision(manifest) == "abc123"


def test_missing_manifest_names_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "run_manifest.json"
    with pytest.raises(ModelArtifactError) as excinfo:
        read_resolved_revision(missing)
    assert str(missing) in str(excinfo.value)


def test_manifest_without_revision_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"backend": "qlora"}), encoding="utf-8")
    with pytest.raises(ModelArtifactError) as excinfo:
        read_resolved_revision(manifest)
    assert "resolved_model_revision" in str(excinfo.value)


def test_real_manifest_carries_the_expected_revision() -> None:
    """Guards against the adapter being paired with a different base model."""
    manifest = Path("work/e2_qwen3_8b_qlora_vi_zh_v1/run_manifest.json")
    if not manifest.is_file():
        pytest.skip("E2 run manifest is not present")
    assert read_resolved_revision(manifest) == (
        "b968826d9c46dd6066d109eabc6255188de91218"
    )
