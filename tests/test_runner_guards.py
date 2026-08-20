import pytest

from mt_pipeline.fairseq_runner import ensure_clean_checkpoint_dir
from mt_pipeline.llm_runner import ensure_clean_adapter_dir
from mt_pipeline.runtime import tee_output


def test_fairseq_guard_allows_empty_checkpoint_dir(tmp_path):
    ensure_clean_checkpoint_dir(tmp_path, {})


def test_fairseq_guard_blocks_silent_resume(tmp_path):
    (tmp_path / "checkpoint_last.pt").write_bytes(b"")
    with pytest.raises(RuntimeError, match="would be resumed from"):
        ensure_clean_checkpoint_dir(tmp_path, {})


def test_fairseq_guard_honours_explicit_resume(tmp_path):
    (tmp_path / "checkpoint_last.pt").write_bytes(b"")
    ensure_clean_checkpoint_dir(tmp_path, {"resume": True})


def test_qlora_guard_allows_empty_checkpoint_dir(tmp_path):
    ensure_clean_adapter_dir(tmp_path, {})


def test_qlora_guard_blocks_adapter_overwrite(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already holds a trained adapter"):
        ensure_clean_adapter_dir(tmp_path, {})


def test_qlora_resume_requires_trainer_checkpoint(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="requires an existing"):
        ensure_clean_adapter_dir(tmp_path, {"resume": True})


def test_qlora_guard_resolves_latest_resume_checkpoint(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "trainer" / "checkpoint-200").mkdir(parents=True)
    expected = tmp_path / "trainer" / "checkpoint-546"
    expected.mkdir()

    assert ensure_clean_adapter_dir(tmp_path, {"resume": True}) == expected


def test_qlora_guard_blocks_unfinished_trainer_state(tmp_path):
    (tmp_path / "trainer" / "checkpoint-200").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="unfinished Trainer state"):
        ensure_clean_adapter_dir(tmp_path, {})


def test_tee_output_preserves_existing_transcript(tmp_path, capsys):
    canonical = tmp_path / "train.log"
    canonical.write_text("historical evidence\n", encoding="utf-8")

    with tee_output(canonical) as transcript:
        print("new run")

    assert canonical.read_text(encoding="utf-8") == "historical evidence\n"
    assert transcript != canonical
    assert transcript.read_text(encoding="utf-8") == "new run\n"
    assert "new run" in capsys.readouterr().out
